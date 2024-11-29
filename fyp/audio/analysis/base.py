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

@numba.jit(nopython=True)
def _slice_features(times: NDArray[np.float64], features: NDArray, start: float, end: float):
    start_idx = np.searchsorted(times, start, side='right') - 1
    end_idx = np.searchsorted(times, end, side='right')
    new_times = times[start_idx:end_idx] - start
    # Set the start to 0 if the first chord is before the start
    new_times[0] = 0.
    new_features = features[start_idx:end_idx]
    return new_times, new_features
