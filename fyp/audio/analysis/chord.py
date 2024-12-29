from __future__ import annotations
import os
import numpy as np
import torch
from .base import DiscreteLatentFeatures, ContinuousLatentFeatures
from .. import Audio
from typing import Callable
from ...model import chord_inference as inference
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
from functools import lru_cache
import json
import numba
from numpy.typing import NDArray
from dataclasses import dataclass

# The penalty score per bar for not having a chord
NO_CHORD_PENALTY = 3

# The penalty score per bar for having an unknown chord
UNKNOWN_CHORD_PENALTY = 3

@dataclass(frozen=True)
class ChordAnalysisResult(DiscreteLatentFeatures[str]):
    """A class with the following attributes:

    labels: list[int] - The chord labels for each frame
    chords: list[str] - The chord names. `chords[labels[i]]` is the chord name for the ith frame
    times: list[float] - The times of each frame

    This uses the large voca chord system."""
    @classmethod
    def latent_size(cls):
        return len(get_idx2voca_chord())

    @classmethod
    def distance(cls, x: int, y: int):
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

    def grouped_end_times_labels(self):
        """Returns a tuple of grouped end times and grouped labels. This is mostly useful for dataset search"""
        ct = self.group()
        end_times = ct.times

        ct.features.flags.writeable = True
        ct.times.flags.writeable = True

        end_times[:-1] = end_times[1:]
        end_times[-1] = self.duration

        ct.features.flags.writeable = False
        end_times.flags.writeable = False
        return end_times, ct.features

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

    def simplify(self) -> SimpleChordAnalysisResult:
        """Convert the ChordAnalysisResult to a SimpleChordAnalysisResult"""
        simplified_labels = np.array([simplify_chord(label) for label in self.features], dtype=np.uint32)
        return SimpleChordAnalysisResult(self.duration, simplified_labels, self.times)

class SimpleChordAnalysisResult(ChordAnalysisResult):
    """A class with the following attributes:

    labels: list[int] - The chord labels for each frame
    chords: list[str] - The chord names. `chords[labels[i]]` is the chord name for the ith frame
    times: list[float] - The times of each frame

    This uses the regular chord system."""
    @classmethod
    def latent_size(cls):
        return 25

    @classmethod
    def distance(cls, x: int, y: int):
        # Use circle of fifths to calculate distance
        no_chord = 24
        key_to_sharp = [0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5,
                        -3, 4, -1, 6, 1, -4, 3, -2, 5, 0, -5, 2, -3]
        if x == no_chord and y == no_chord:
            return 0
        if x == no_chord or y == no_chord:
            return NO_CHORD_PENALTY
        dist = key_to_sharp[y] - key_to_sharp[x]
        if dist > 6:
            dist = 12 - dist
        if dist < -6:
            dist = 12 + dist
        return abs(dist)

    @classmethod
    def map_feature_index(cls, x: int) -> str:
        return small_voca_to_chord(x)

    @classmethod
    def map_feature_name(cls, x: str) -> int:
        # Need to avoid the ChordAnalysisResult.map_feature_name because it uses the large voca inversion
        try:
            return large_voca_to_small_voca_map()[get_inv_voca_map()[x]]
        except KeyError:
            if x in get_inv_voca_map():
                raise ValueError(f"Chord {x} is not a simple chord ({get_inv_voca_map()[x]})")
            raise ValueError(f"Unknown chord {x}")

    @property
    def chords(self) -> list[str]:
        """Returns a list of chord labels"""
        chord_labels = [small_voca_to_chord(label) for label in self.features]
        return chord_labels

    def transpose(self, semitones: int) -> SimpleChordAnalysisResult:
        """Transpose the chords by semitones"""
        labels = [
            large_voca_to_small_voca_map()[get_inv_voca_map()[transpose_chord(small_voca_to_chord(label), semitones)]]
            for label in self.features]
        np_labels = np.array(labels, dtype=np.uint32)
        return SimpleChordAnalysisResult(self.duration, np_labels, self.times)

    def save(self, path: str):
        json_dict = {
            "duration": self.duration,
            "labels": [small_voca_to_large_voca(x) for x in self.features.tolist()],
            "times": self.times.tolist()
        }
        with open(path, "w") as f:
            json.dump(json_dict, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        labels = [large_voca_to_small_voca_map()[x] for x in data["labels"]]
        return cls.from_data(data["duration"], labels, data["times"])

class ChordFeatures(ContinuousLatentFeatures):
    @staticmethod
    def latent_dims():
        return 128

    @staticmethod
    def distance(a: NDArray, b: NDArray) -> float:
        return np.linalg.norm(a - b, 2).item()

def analyse_chord_transformer(audio: Audio, *, model_path: str = "./resources/ckpts/btc_model_large_voca.pt", use_large_voca: bool = True, use_loaded_model: bool = True) -> ChordAnalysisResult:
    results, _ = inference(audio, model_path=model_path, use_loaded_model=use_loaded_model, use_voca=use_large_voca)
    chords = get_idx2voca_chord()
    times = [r[0] for r in results]
    inv_voca = get_inv_voca_map()
    labels = [inv_voca[chords[r[1]]] for r in results]

    cr = ChordAnalysisResult.from_data(
        audio.duration,
        labels = labels,
        times = times,
    )
    return cr

def analyse_chord_features(audio: Audio, *, model_path: str = "./resources/ckpts/btc_model_large_voca.pt", use_large_voca: bool = True, use_loaded_model: bool = True) -> ContinuousLatentFeatures:
    _, features = inference(audio, model_path=model_path, use_loaded_model=use_loaded_model, use_voca=use_large_voca)
    latent = torch.cat(features, dim = 1)[0].cpu().numpy()
    latent = np.array(latent, dtype=np.float32)

    time_idx = int(audio.duration * 10.8)
    latent = latent[:time_idx]
    return ChordFeatures(
        duration=audio.duration,
        features=latent,
        times=np.arange(time_idx) / 10.8
    )
