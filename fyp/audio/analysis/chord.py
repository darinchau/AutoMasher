from __future__ import annotations
import os
import numpy as np
import torch
from .base import DiscreteLatentFeatures, ContinuousLatentFeatures
from .. import Audio
from typing import Callable
from ...model import chord_inference as inference
from ...model.chord import ChordModelOutput
from ...util.note import (
    get_idx2voca_chord,
    get_inv_voca_map,
    transpose_chord,
    get_chord_notes,
    get_chord_note_inv,
    small_voca_to_chord,
    large_voca_to_small_voca_map,
    small_voca_to_large_voca,
    simplify_chord
)
from ...util import YouTubeURL
from functools import lru_cache
import json
import numba
from numpy.typing import NDArray
from dataclasses import dataclass
from math import log
import typing
import warnings
from functools import cache

from enum import Enum

if typing.TYPE_CHECKING:
    from ..dataset import SongDataset

# The penalty score per bar for not having a chord
NO_CHORD_PENALTY = 3

# The penalty score per bar for having an unknown chord
UNKNOWN_CHORD_PENALTY = 3

# The penalty score for having a chord that is not in the same quality for the circle-of-fifths distnace
DIFFERENT_QUALITY_PENALTY = 3

LARGE_VOCA_CHORD_DATASET_KEY = "chord_btc_large_voca"
SIMPLE_CHORD_DATASET_KEY = "chord_btc"


class ChordMetric(Enum):
    """The chord metric to use for the analysis"""
    DEFAULT = "symmetric_diff"
    MONTE_CARLO = "monte_carlo"
    KL_DIVERGENCE = "kl_divergence"
    JENSEN_SHANNON = "jensen_shannon"

    def get_dist_array(self) -> NDArray:
        """Get the distance matrix for the chord metric"""
        if self == ChordMetric.DEFAULT:
            return ChordAnalysisResult.get_dist_array()
        elif self == ChordMetric.MONTE_CARLO:
            return np.load("resources/deep_dist.npz")['distances_sum'][:, :, 0] / np.load("resources/deep_dist.npz")['count'].clip(min=1)
        elif self == ChordMetric.KL_DIVERGENCE:
            return np.load("resources/kl_divergence_matrix.npz")['kl_matrix']
        elif self == ChordMetric.JENSEN_SHANNON:
            return np.load("resources/jensen_shannon_matrix.npz")['js_matrix']
        else:
            raise ValueError(f"Invalid chord metric: {self}")


@dataclass(frozen=True)
class ChordAnalysisResult(DiscreteLatentFeatures[str]):
    """A class with the following attributes:

    labels: list[int] - The chord labels for each frame
    chords: list[str] - The chord names. `chords[labels[i]]` is the chord name for the ith frame
    times: list[float] - The times of each frame

    This uses the large voca chord system. Override ChordAnalysisResult to use a different chord metric"""
    @classmethod
    def latent_size(cls):
        return len(get_idx2voca_chord())

    @classmethod
    def distance(cls, x: int, y: int) -> float:
        x_idx = cls.map_feature_index(x)
        y_idx = cls.map_feature_index(y)
        chord_notes_map = get_chord_notes()

        assert x_idx in chord_notes_map, f"{x_idx} not a recognised chord"
        assert y_idx in chord_notes_map, f"{y_idx} not a recognised chord"

        match (x_idx, y_idx):
            case ("No chord", "No chord"):
                return 0

            case (_, "No chord"):
                return NO_CHORD_PENALTY

            case ("No chord", _):
                return NO_CHORD_PENALTY

            case (_, "Unknown"):
                return UNKNOWN_CHORD_PENALTY

            case ("Unknown", _):
                return UNKNOWN_CHORD_PENALTY

            case (_, _):
                chord_notes_inv = get_chord_note_inv()

                # Rule 0. If the chords are the same, distance is 0
                if x_idx == y_idx:
                    return 0

                notes1 = chord_notes_map[x_idx]
                notes2 = chord_notes_map[y_idx]

                # Rule 1. If one chord is a subset of the other, distance is 0
                if notes1 <= notes2 or notes2 <= notes1:
                    return 1

                # Rule 2. If the union of two chords is the same as the notes of one chord, distance is 1
                notes_union = notes1 | notes2
                if notes_union in chord_notes_inv:
                    return 1

                # Rule 3. If the union of two chords is the same as the notes of one chord, distance is the number of notes in the symmetric difference
                diff = set(notes1) ^ set(notes2)
                return len(diff)

    @classmethod
    def map_feature_index(cls, x: int) -> str:
        return get_idx2voca_chord()[x]

    @classmethod
    def map_feature_name(cls, x: str) -> int:
        return get_inv_voca_map()[x]

    @classmethod
    def from_data(cls, duration: float, labels: list[int], times: list[float]):
        """Construct a ChordAnalysisResult from native python types. Should function identically as the old constructor."""
        return cls(duration, np.array(labels, dtype=np.uint32), np.array(times, dtype=np.float64))

    @property
    def chords(self) -> list[str]:
        """Returns a list of chord labels"""
        chords = get_idx2voca_chord()
        chord_labels = [chords[label] for label in self.features]
        return chord_labels

    @property
    def info(self) -> list[tuple[str, float]]:
        """A utility for getting the chord info for real-time display"""
        return list(zip(self.chords, self.times.tolist()))

    def get_duration(self):
        return self.duration

    def __repr__(self):
        s = "ChordAnalysisResult("
        cr = self.group()
        for chord, time in zip(cr.chords, cr.times):
            s += "\n\t" + chord + " " + str(time)
        s += "\n)"
        return s

    def transpose(self, semitones: int) -> ChordAnalysisResult:
        """Transpose the chords by semitones"""
        voca = get_idx2voca_chord()
        inv_map = get_inv_voca_map()
        labels = [inv_map[transpose_chord(voca[label], semitones)] for label in self.features]
        np_labels = np.array(labels, dtype=np.uint32)
        return ChordAnalysisResult(self.duration, np_labels, self.times)

    def save(self, path: str):
        json_dict = {
            "duration": self.duration,
            "labels": self.features.tolist(),
            "times": self.times.tolist()
        }

        with open(path, "w") as f:
            json.dump(json_dict, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_data(data["duration"], data["labels"], data["times"])


class ChordFeatures(ContinuousLatentFeatures):
    @staticmethod
    def latent_dims():
        return 128

    @staticmethod
    def distance(a: NDArray, b: NDArray) -> float:
        return np.linalg.norm(a - b, 2).item()


def analyse_chord_transformer(
    audio: Audio | None = None, *,
    results: ChordModelOutput | None = None,
    dataset: SongDataset | None = None,
    url: YouTubeURL | None = None,
    regularizer: float = 0.5,
    model_path: str = "./resources/ckpts/btc_model_large_voca.pt",
    use_large_voca: bool = True,
    use_cache: bool = True
) -> ChordAnalysisResult:
    """Analyse the chords of an audio file using the transformer model

    Parameters:
    audio (Audio): The audio file to analyse
    results (ChordModelOutput): The results of the model - if the results is somehow precomputed, this will be used instead of the model
    regularizer (float): The regularizer to use. This value penalizes the model for changing chords too often. Defaults to 0.5
    model_path (str): The path to the model checkpoint. Defaults to "./resources/ckpts/btc_model_large_voca.pt"
    use_large_voca (bool): Whether to use the large voca chord system. Defaults to True
    use_cache (bool): Whether to use the cache. Defaults to True
    """
    if results is None:
        assert audio is not None, f"Audio must be provided if results is None"
        duration = audio.duration
        results = get_chord_result(audio, dataset, url, model_path, use_large_voca, use_cache)
    else:
        if audio is not None:
            warnings.warn("Audio is not None, but results is provided. Audio will be ignored.")
        duration = results.duration

    time_idx = int(duration / results.time_resolution)

    logit_tensor = results.logits[:time_idx].numpy()
    T, K = logit_tensor.shape
    dp = np.zeros_like(logit_tensor)
    path = np.zeros_like(logit_tensor, dtype=int)

    dp[0] = logit_tensor[0]

    for t in range(1, T):
        for k in range(K):
            scores = dp[t-1] - (regularizer * (np.arange(K) != k))
            best_prev_class = np.argmax(scores)
            dp[t, k] = logit_tensor[t, k] + scores[best_prev_class]
            path[t, k] = best_prev_class

    max_indices = np.zeros(T, dtype=np.uint32)
    max_indices[-1] = np.argmax(dp[-1])

    for t in range(T-2, -1, -1):
        max_indices[t] = path[t+1, max_indices[t+1]]

    labels: list[int] = max_indices.tolist()
    if use_large_voca:
        chords = get_idx2voca_chord()
        inv_voca = get_inv_voca_map()
        labels = [inv_voca[chords[r]] for r in labels]  # Deduplicate the chords
    else:
        labels = [small_voca_to_large_voca(r) for r in labels]

    times = np.arange(T) * results.time_resolution

    cr = ChordAnalysisResult(
        duration=duration,
        features=max_indices,
        times=times.astype(np.float64)
    ).group()
    return cr


def analyse_chord_features(
    audio: Audio | None = None, *,
    results: ChordModelOutput | None = None,
    dataset: SongDataset | None = None,
    url: YouTubeURL | None = None,
    model_path: str = "./resources/ckpts/btc_model_large_voca.pt",
    use_large_voca: bool = True,
    use_cache: bool = True
) -> ChordFeatures:
    """Analyse the chords of an audio file using the transformer model

    Parameters:
    audio (Audio): The audio file to analyse
    results (ChordModelOutput): The results of the model - if the results is somehow precomputed, this will be used instead of the model
    regularizer (float): The regularizer to use. This value penalizes the model for changing chords too often. Defaults to 0.5
    model_path (str): The path to the model checkpoint. Defaults to "./resources/ckpts/btc_model_large_voca.pt"
    use_large_voca (bool): Whether to use the large voca chord system. Defaults to True
    use_cache (bool): Whether to use the cache. Defaults to True
    """
    if results is None:
        assert audio is not None, "Audio must be provided if results is None"
        results = get_chord_result(audio, dataset, url, model_path, use_large_voca, use_cache)
        duration = audio.duration
    else:
        if audio is not None:
            warnings.warn("Audio is not None, but results is provided. Audio will be ignored.")
        duration = results.duration

    time_idx = int(duration / results.time_resolution)
    latent = results.features[:time_idx].float().numpy()

    return ChordFeatures(
        duration=duration,
        features=latent,
        times=np.arange(time_idx) * results.time_resolution
    )


def get_chord_result(audio: Audio,
                     dataset: SongDataset | None = None,
                     url: YouTubeURL | None = None,
                     model_path: str = "./resources/ckpts/btc_model_large_voca.pt",
                     use_large_voca: bool = True,
                     use_cache: bool = True) -> ChordModelOutput:
    results = None
    key = LARGE_VOCA_CHORD_DATASET_KEY if use_large_voca else SIMPLE_CHORD_DATASET_KEY
    if use_cache and dataset is not None:
        dataset.register(SIMPLE_CHORD_DATASET_KEY, "{video_id}.chord")
        dataset.register(LARGE_VOCA_CHORD_DATASET_KEY, "{video_id}.chord")
        if url is not None and not url.is_placeholder and dataset.has_path(key, url):
            results = ChordModelOutput.load(dataset.get_path(key, url))

    if results is None:
        results = inference(audio, model_path=model_path, use_voca=use_large_voca)
        if use_cache and dataset is not None and url is not None and not url.is_placeholder:
            results.save(dataset.get_path(key, url))
    return results
