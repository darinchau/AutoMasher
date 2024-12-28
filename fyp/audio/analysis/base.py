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
from abc import ABC, abstractmethod
import typing

@dataclass(frozen=True)
class OnsetFeatures:
    """A class of features that represents segmentations over the song. The onsets are in seconds and marks the start of each segment"""
    duration: float
    onsets: NDArray[np.float64]

    def __post_init__(self):
        assert len(self.onsets) == 0 or self.duration >= self.onsets[-1]
        assert isinstance(self.onsets, np.ndarray)
        assert self.onsets.dtype == np.float64

        # Assert onsets are sorted by time
        assert np.all(self.onsets[1:] >= self.onsets[:-1]), "Onsets must be sorted"

        self.onsets.flags.writeable = False

    def __len__(self):
        return self.onsets.shape[0]

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

T = TypeVar("T")
@dataclass(frozen=True)
class DiscreteLatentFeatures(ABC, Generic[T]):
    """A class that represents the latent features of a song"""
    duration: float
    features: NDArray[np.uint32]
    times: NDArray[np.float64]

    def __post_init__(self):
        assert self.duration > 0
        assert len(self.features.shape) == 1
        assert self.features.shape[0] > 0
        assert np.all(self.features < self.latent_size())
        assert self.features.dtype == np.uint32
        assert isinstance(self.features, np.ndarray)
        assert self.latent_size() > 0

        assert len(self.times.shape) == 1
        assert self.times.shape[0] > 0
        assert self.duration >= self.times[-1]
        assert self.times[0] == 0
        assert np.all(self.times[1:] >= self.times[:-1]), "Times must be sorted"

        assert self.features.shape[0] == self.times.shape[0]

        self.features.flags.writeable = False
        self.times.flags.writeable = False

    @classmethod
    @abstractmethod
    def latent_size(cls) -> int:
        """The number of features of the latent space. Since it is discrete it should be finite"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def distance(cls, a: int, b: int) -> float:
        """The distance between two latent features. The inputs should be two latent feature indices"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def map_feature_index(cls, x: int) -> T:
        """Map the feature index to the latent space"""
        raise NotImplementedError

    @classmethod
    def map_feature_name(cls, x: T) -> int:
        """Map the feature name to the latent space"""
        for i in range(cls.latent_size()):
            if cls.map_feature_index(i) == x:
                return i
        raise ValueError(f"Feature {x} not found in latent space")

    @classmethod
    def fdist(cls, a: T, b: T) -> float:
        """The distance between two latent features. The inputs should be two latent feature indices"""
        return cls.distance(cls.map_feature_name(a), cls.map_feature_name(b))

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

    @classmethod
    @lru_cache(maxsize=1)
    def get_dist_array(cls):
        """Get the distance array between all latent features"""
        dist_array = np.zeros((cls.latent_size(), cls.latent_size()))
        for i in range(cls.latent_size()):
            for j in range(cls.latent_size()):
                dist = cls.distance(i, j)
                assert dist >= 0, f"Distance between {i} and {j} is negative"
                dist_array[i, j] = dist
        return dist_array

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

@numba.jit(nopython=True)
def _slice_features(times: NDArray[np.float64], features: NDArray, start: float, end: float):
    start_idx = np.searchsorted(times, start, side='right') - 1
    end_idx = np.searchsorted(times, end, side='right')
    new_times = times[start_idx:end_idx] - start
    # Set the start to 0 if the first chord is before the start
    new_times[0] = 0.
    new_features = features[start_idx:end_idx]
    return new_times, new_features

def _dist_discrete_latent(times1, times2, chords1, chords2, distances, duration) -> float:
    """A jitted version of the latent result distance calculation, which is defined to be the sum of distances times time
    between the two latent feature. Refer to our report for more detalis.

    This assumes times1 and times2 comes from a DiscreteLatentFeatures object with the same duration"""
    score = 0.
    cumulative_duration = 0.

    idx1 = 0
    idx2 = 0
    len_t1 = len(times1) - 1
    len_t2 = len(times2) - 1

    while cumulative_duration < duration and (idx1 < len_t1 or idx2 < len_t2):
        # Find the duration of the next segment to calculate
        next_x = duration
        if idx1 < len_t1 and times1[idx1 + 1] < next_x:
            next_x = times1[idx1 + 1]
        if idx2 < len_t2 and times2[idx2 + 1] < next_x:
            next_x = times2[idx2 + 1]

        score += distances[chords1[idx1]][chords2[idx2]] * (next_x - cumulative_duration)
        cumulative_duration = next_x

        if idx1 < len_t1 and next_x == times1[idx1 + 1]:
            idx1 += 1
        if idx2 < len_t2 and next_x == times2[idx2 + 1]:
            idx2 += 1
    score += distances[chords1[idx1]][chords2[idx2]] * (duration - cumulative_duration)
    return score

F = typing.TypeVar('F', bound=DiscreteLatentFeatures)
def dist_discrete_latent_features(a: F, b: F, dist_array: NDArray[np.float64] | None = None) -> float:
    """Calculate the distance between two discrete latent features"""
    raise NotImplementedError("This function is not implemented yet")

G = typing.TypeVar('G', bound=ContinuousLatentFeatures)
def dist_continuous_latent_features(a: G, b: G) -> float:
    """Calculate the distance between two discrete latent features"""
    score = 0.
    cumulative_duration = 0.

    times1 = a.times
    times2 = b.times

    idx1 = 0
    idx2 = 0
    len_t1 = len(times1)
    len_t2 = len(times2)

    while idx1 < len_t1 and idx2 < len_t2:
        # Find the duration of the next segment to calculate
        min_time: float = min(times1[idx1], times2[idx2])

        # Score = sum of (distance * duration)
        # Assuming symmetric zero distance - we can aggregate the calculations first
        label1 = a.features[idx1]
        label2 = b.features[idx2]
        score += a.distance(label1, label2) * (min_time - cumulative_duration)
        cumulative_duration = min_time

        if times1[idx1] <= min_time:
            idx1 += 1
        if times2[idx2] <= min_time:
            idx2 += 1
    return score
