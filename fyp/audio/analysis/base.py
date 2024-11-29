from __future__ import annotations
from torch import Tensor
from typing import Any, Callable, TypeVar, Generic
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import librosa
from ...util.note import get_keys, get_idx2voca_chord, transpose_chord, get_inv_voca_map, get_chord_notes, get_chord_note_inv
from ...audio import Audio
import torch
from numpy.typing import NDArray
import numpy as np
import numba
import json
from ..dataset import DatasetEntry
from abc import ABC, abstractmethod

# The penalty score per bar for not having a chord
NO_CHORD_PENALTY = 3

# The penalty score per bar for having an unknown chord
UNKNOWN_CHORD_PENALTY = 3

@dataclass(frozen=True)
class OnsetFeatures:
    """A class of features that represents segmentations over the song"""
    duration: float
    onsets: NDArray[np.float32]

    def __post_init__(self):
        assert len(self.onsets) == 0 or self.duration >= self.onsets[-1]
        assert isinstance(self.onsets, np.ndarray)
        assert self.onsets.dtype == np.float32

        # Assert onsets are sorted by time
        assert np.all(self.onsets[1:] >= self.onsets[:-1]), "Onsets must be sorted"

        self.onsets.flags.writeable = False

    @property
    def tempo(self):
        """Returns the average onset per minute of the song"""
        return float(np.average(1 / (self.onsets[1:] - self.onsets[:-1]) * 60))

    @property
    def nsegments(self):
        """Returns the number of segments in the song"""
        return self.onsets.shape[0]

    def slice_seconds(self, start: float, end: float) -> OnsetFeatures:
        """Slice the beat analysis result by seconds. includes start and excludes end"""
        assert start >= 0.
        if abs(end - self.duration) < 1e-6 or end == -1:
            end = self.duration
        assert end > start and end <= self.duration

        beat_mask = (self.onsets >= start) & (self.onsets < end)
        beats = self.onsets[beat_mask] - start

        return OnsetFeatures(end - start, beats)

    def change_speed(self, speed: float) -> OnsetFeatures:
        """Change the speed of the beat analysis result"""
        beats = self.onsets / speed
        return OnsetFeatures(self.duration / speed, beats)

    def make_click_track(self, audio: Audio, frequency: int = 1000) -> Audio:
        click_track = librosa.clicks(times=self.onsets, sr=audio.sample_rate, length=audio.nframes, click_freq=frequency)

        click_track = audio.numpy() * 0.5 + click_track
        click_track = torch.tensor([click_track])
        return Audio(click_track, audio.sample_rate)

@dataclass(frozen=True)
class BeatAnalysisResult:
    """A class that represents the result of a beat analysis."""
    _beats: OnsetFeatures
    _downbeats: OnsetFeatures

    @property
    def duration(self):
        return self._beats.duration

    @property
    def beats(self):
        return self._beats.onsets

    @property
    def downbeats(self):
        return self._downbeats.onsets

    def __post_init__(self):
        assert self._beats.onsets == self._downbeats.onsets

    @property
    def tempo(self):
        return self._beats.tempo

    @property
    def nbars(self):
        """Returns the number of bars in the song"""
        return self._downbeats.nsegments

    @classmethod
    def from_data_entry(cls, data_entry: DatasetEntry):
        return cls.from_data(data_entry.length, data_entry.beats, data_entry.downbeats)

    @classmethod
    def from_data(cls, duration: float, beats: list[float], downbeats: list[float]):
        _beats = OnsetFeatures(duration, np.array(beats, dtype=np.float32))
        _downbeats = OnsetFeatures(duration, np.array(downbeats, dtype=np.float32))
        return cls(_beats, _downbeats)

    def slice_seconds(self, start: float, end: float) -> BeatAnalysisResult:
        """Slice the beat analysis result by seconds. includes start and excludes end"""
        return BeatAnalysisResult(
            self._beats.slice_seconds(start, end),
            self._downbeats.slice_seconds(start, end)
        )

    def change_speed(self, speed: float) -> BeatAnalysisResult:
        """Change the speed of the beat analysis result"""
        return BeatAnalysisResult(
            self._beats.change_speed(speed),
            self._downbeats.change_speed(speed)
        )

    def save(self, path: str):
        json_dict = {
            "duration": self.duration,
            "beats": self.beats.tolist(),
            "downbeats": self.downbeats.tolist()
        }

        with open(path, "w") as f:
            json.dump(json_dict, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_data(data["duration"], data["beats"], data["downbeats"])


T = TypeVar("T")
@dataclass(frozen=True)
class DiscreteLatentFeatures(ABC, Generic[T]):
    """A class that represents the latent features of a song"""
    duration: float
    features: NDArray[np.uint32]
    times: NDArray[np.float64]

    def __post_init__(self):
        assert self.duration > 0
        assert len(self.features.shape) == 2
        assert self.features.shape[0] > 0
        assert self.features.shape[1] == self.latent_dims()
        assert np.all(self.features < self.latent_size())
        assert self.features.dtype == np.uint32
        assert isinstance(self.features, np.ndarray)

        assert len(self.times.shape) == 1
        assert self.times.shape[0] > 0
        assert self.duration >= self.times[-1]
        assert self.times[0] == 0
        assert np.all(self.times[1:] >= self.times[:-1]), "Times must be sorted"

        assert self.features.shape[0] == self.times.shape[0]

        self.features.flags.writeable = False
        self.times.flags.writeable = False

    @staticmethod
    @abstractmethod
    def latent_dims():
        """The number of latent dimensions"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def latent_size():
        """The number of features of the latent space. Since it is discrete it should be finite"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def distance(a: int, b: int) -> float:
        """The distance between two latent features"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def map_feature_index(x: int) -> T:
        """Map the feature index to the latent space"""
        raise NotImplementedError

    def group(self):
        """Group the latent features, and deletes duplicates."""
        features = np.zeros_like(self.features)
        times = np.zeros_like(self.times)
        idx = 0
        for i in range(self.features.shape[0]):
            if idx == 0 or not np.all(self.features[i] == self.features[i-1]):
                features[idx] = self.features[i]
                times[idx] = self.times[i]
                idx += 1
        return self.__class__(self.duration, features[:idx], times[:idx])

    def slice_seconds(self, start: float, end: float):
        """Slice the chord analysis result by seconds and shifts the times to start from 0 includes start and excludes end"""
        assert start >= 0
        if abs(end - self.duration) < 1e-6 or end == -1:
            end = self.duration
        assert end > start and end <= self.duration
        new_times, new_labels = _slice_features(self.times, self.features, start, end)
        return self.__class__(end - start, new_labels, new_times)

@dataclass(frozen=True)
class ContinuousLatentFeatures(ABC):
    """A class that represents the continuous latent features of a song"""
    duration: float
    features: NDArray[np.float32]
    times: NDArray[np.float64]

    def __post_init__(self):
        assert self.duration > 0
        assert len(self.features.shape) == 2
        assert self.features.shape[0] > 0
        assert self.features.shape[1] == self.latent_dims()
        assert self.features.dtype == np.float32
        assert isinstance(self.features, np.ndarray)

        assert len(self.times.shape) == 1
        assert self.times.shape[0] > 0
        assert self.duration >= self.times[-1]
        assert self.times[0] == 0
        assert np.all(self.times[1:] >= self.times[:-1]), "Times must be sorted"

        assert self.features.shape[0] == self.times.shape[0]

        self.features.flags.writeable = False
        self.times.flags.writeable = False

    @staticmethod
    @abstractmethod
    def latent_dims():
        """The number of latent dimensions"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def distance(a: NDArray, b: NDArray) -> float:
        """The distance between two latent features. Inputs should be two 1D latent feature vectors"""
        raise NotImplementedError

    def slice_seconds(self, start: float, end: float):
        """Slice the chord analysis result by seconds and shifts the times to start from 0 includes start and excludes end"""
        assert start >= 0
        if abs(end - self.duration) < 1e-6 or end == -1:
            end = self.duration
        assert end > start and end <= self.duration
        new_times, new_features = _slice_features(self.times, self.features, start, end)

        return self.__class__(end - start, new_features, new_times)

@dataclass(frozen=True)
class ChordAnalysisResult(DiscreteLatentFeatures[str]):
    """A class with the following attributes:

    labels: list[int] - The chord labels for each frame
    chords: list[str] - The chord names. `chords[labels[i]]` is the chord name for the ith frame
    times: list[float] - The times of each frame"""
    @property
    def labels(self):
        return np.array(self.features[:, 0], dtype=np.uint8)

    @staticmethod
    def latent_dims():
        return 1

    @staticmethod
    def latent_size():
        return len(get_idx2voca_chord())

    @staticmethod
    def distance(x: int, y: int):
        x_idx = ChordAnalysisResult.map_feature_index(x)
        y_idx = ChordAnalysisResult.map_feature_index(y)
        dist, _ = _calculate_distance_of_two_chords(x_idx, y_idx)
        return dist

    @staticmethod
    def map_feature_index(x: int) -> str:
        return get_idx2voca_chord()[x]

    @classmethod
    def from_data(cls, duration: float, labels: list[int], times: list[float]):
        """Construct a ChordAnalysisResult from native python types. Should function identically as the old constructor."""
        return cls(duration, np.array(labels, dtype=np.uint32)[:, None], np.array(times, dtype=np.float64))

    def grouped_end_times_labels(self):
        """Returns a tuple of grouped end times and grouped labels. This is mostly useful for dataset search"""
        ct = self.group()
        end_times = ct.times

        # Unlock the arrays is safe now because ct is a copy of self and is not shared with other objects
        ct.labels.flags.writeable = True
        ct.times.flags.writeable = True

        end_times[:-1] = end_times[1:]
        end_times[-1] = self.duration
        return end_times, ct.labels

    @property
    def chords(self) -> list[str]:
        """Returns a list of chord labels"""
        chords = get_idx2voca_chord()
        chord_labels = [chords[label] for label in self.labels]
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
        labels = [inv_map[transpose_chord(voca[label], semitones)] for label in self.labels]
        np_labels = np.array(labels, dtype=np.uint32)[:, None]
        return ChordAnalysisResult(self.duration, np_labels, self.times)

    @classmethod
    def from_data_entry(cls, entry: DatasetEntry):
        return cls.from_data(entry.length, entry.chords, entry.chord_times)

    def save(self, path: str):
        json_dict = {
            "duration": self.duration,
            "labels": self.labels.tolist(),
            "times": self.times.tolist()
        }

        with open(path, "w") as f:
            json.dump(json_dict, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_data(data["duration"], data["labels"], data["times"])

@numba.jit(nopython=True)
def _slice_features(times: NDArray[np.float64], features: NDArray, start: float, end: float):
    start_idx = np.searchsorted(times, start, side='right') - 1
    end_idx = np.searchsorted(times, end, side='right')
    new_times = times[start_idx:end_idx] - start
    # Set the start to 0 if the first chord is before the start
    new_times[0] = 0.
    new_features = features[start_idx:end_idx]
    return new_times, new_features

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

@lru_cache(maxsize=1)
def _get_distance_array():
    """Calculates the distance array for all chords. The distance array is a 2D array where the (i, j)th element is the distance between the ith and jth chords.
    This will be cached and used for all future calculations."""
    chord_mapping = get_idx2voca_chord()
    n = len(chord_mapping)
    distance_array = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            distance_array[i][j], _ = _calculate_distance_of_two_chords(chord_mapping[i], chord_mapping[j])
    return np.array(distance_array, dtype = np.int32)
