from __future__ import annotations
import os
import numpy as np
import torch
from .base import DiscreteLatentFeatures, ContinuousLatentFeatures
from .. import Audio
from typing import Callable
from ...model import chord_inference as inference
from ...util.note import get_idx2voca_chord, get_inv_voca_map, transpose_chord, get_chord_notes, get_chord_note_inv
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
    times: list[float] - The times of each frame"""
    @classmethod
    def latent_size(cls):
        return len(get_idx2voca_chord())

    @classmethod
    def distance(cls, x: int, y: int):
        x_idx = cls.map_feature_index(x)
        y_idx = cls.map_feature_index(y)
        dist, _ = _calculate_distance_of_two_chords(x_idx, y_idx)
        return dist

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

class ChordFeatures(ContinuousLatentFeatures):
    @staticmethod
    def latent_dims():
        return 128

    @staticmethod
    def distance(a: NDArray, b: NDArray) -> float:
        return np.linalg.norm(a - b, 2).item()

# Gets the distance of two chords and the closest approximating chord
def _calculate_distance_of_two_chords(chord1: str, chord2: str) -> tuple[int, str]:
    """Gives an empirical score of the similarity of two chords. The lower the score, the more similar the chords are. The second value is the closest approximating chord."""
    chord_notes_map = get_chord_notes()

    assert chord1 in chord_notes_map, f"{chord1} not a recognised chord"
    assert chord2 in chord_notes_map, f"{chord2} not a recognised chord"

    match (chord1, chord2):
        case ("No chord", "No chord"):
            score, result = 0, "No chord"

        case ("No chord", "Unknown"):
            score, result = NO_CHORD_PENALTY, "Unknown"

        case ("Unknown", "No chord"):
            score, result = NO_CHORD_PENALTY, "Unknown"

        case ("Unknown", "Unknown"):
            score, result = UNKNOWN_CHORD_PENALTY, "Unknown"

        case (_, "No chord"):
            score, result = NO_CHORD_PENALTY, chord1

        case (_, "Unknown"):
            score, result = UNKNOWN_CHORD_PENALTY, "Unknown"

        case ("No chord", _):
            score, result = NO_CHORD_PENALTY, chord2

        case ("Unknown", _):
            score, result = UNKNOWN_CHORD_PENALTY, "Unknown"

        case (_, _):
            score, result = _distance_of_two_nonempty_chord(chord1, chord2)

    assert result in chord_notes_map, f"{result} not a recognised chord"
    return score, result

# Gets the distance of two chords and the closest approximating chord
def _distance_of_two_nonempty_chord(chord1: str, chord2: str) -> tuple[int, str]:
    """Gives the distance between two non-empty chords and the closest approximating chord."""
    chord_notes_map = get_chord_notes()
    chord_notes_inv = get_chord_note_inv()

    # Rule 0. If the chords are the same, distance is 0
    if chord1 == chord2:
        return 0, chord1

    notes1 = chord_notes_map[chord1]
    notes2 = chord_notes_map[chord2]

    # Rule 1. If one chord is a subset of the other, distance is 0
    if notes1 <= notes2:
        return 1, chord2

    if notes2 <= notes1:
        return 1, chord1

    # Rule 2. If the union of two chords is the same as the notes of one chord, distance is 1
    notes_union = notes1 | notes2
    if notes_union in chord_notes_inv:
        return 1, chord_notes_inv[notes_union]

    # Rule 3. If the union of two chords is the same as the notes of one chord, distance is the number of notes in the symmetric difference
    diff = set(notes1) ^ set(notes2)
    return len(diff), "Unknown"

def analyse_chord_transformer(audio: Audio, *, model_path: str = "./resources/ckpts/btc_model_large_voca.pt", use_loaded_model: bool = True) -> ChordAnalysisResult:
    results, _ = inference(audio, model_path=model_path, use_loaded_model=use_loaded_model)
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

def analyse_chord_features(audio: Audio, *, model_path: str = "./resources/ckpts/btc_model_large_voca.pt", use_loaded_model: bool = True) -> ContinuousLatentFeatures:
    _, features = inference(audio, model_path=model_path, use_loaded_model=use_loaded_model)
    latent = torch.cat(features, dim = 1)[0].cpu().numpy()
    latent = np.array(latent, dtype=np.float32)

    time_idx = int(audio.duration * 10.8)
    latent = latent[:time_idx]
    return ChordFeatures(
        duration=audio.duration,
        features=latent,
        times=np.arange(time_idx) / 10.8
    )
