from __future__ import annotations
from torch import Tensor
from typing import Any, Callable
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import librosa
from ...util.note import get_keys, get_idx2voca_chord, transpose_chord, get_inv_voca_map
from ...audio import Audio
import torch
from numpy.typing import NDArray
import numpy as np
import numba
import json
from ..dataset import DatasetEntry
import re

@dataclass(frozen=True)
class BeatAnalysisResult:
    """A class that represents the result of a beat analysis."""
    duration: float
    beats: NDArray[np.float32]
    downbeats: NDArray[np.float32]

    def __post_init__(self):
        assert len(self.beats) == 0 or self.duration >= self.beats[-1]
        assert len(self.downbeats) == 0 or self.duration >= self.downbeats[-1]
        assert isinstance(self.beats, np.ndarray)
        assert isinstance(self.downbeats, np.ndarray)
        assert self.beats.dtype == np.float32
        assert self.downbeats.dtype == np.float32

        self.beats.flags.writeable = False
        self.downbeats.flags.writeable = False

    @property
    def tempo(self):
        bpm = float(np.average(1 / (self.beats[1:] - self.beats[:-1]) * 60))
        return bpm

    @property
    def nbars(self):
        """Returns the number of bars in the song"""
        return self.downbeats.shape[0]

    @classmethod
    def from_data_entry(cls, data_entry: DatasetEntry):
        return cls.from_data(data_entry.length, data_entry.beats, data_entry.downbeats)

    @classmethod
    def from_data(cls, duration: float, beats: list[float], downbeats: list[float]):
        return cls(
            duration,
            np.array(beats, dtype=np.float32),
            np.array(downbeats, dtype=np.float32)
        )

    def slice_seconds(self, start: float, end: float) -> BeatAnalysisResult:
        """Slice the beat analysis result by seconds. includes start and excludes end"""
        assert start >= 0.
        if abs(end - self.duration) < 1e-6 or end == -1:
            end = self.duration
        assert end > start and end <= self.duration

        beat_mask = (self.beats >= start) & (self.beats < end)
        beats = self.beats[beat_mask] - start

        downbeat_mask = (self.downbeats >= start) & (self.downbeats < end)
        downbeats = self.downbeats[downbeat_mask] - start

        return BeatAnalysisResult(end - start, beats, downbeats)

    def change_speed(self, speed: float) -> BeatAnalysisResult:
        """Change the speed of the beat analysis result"""
        beats = self.beats / speed
        downbeats = self.downbeats / speed
        return BeatAnalysisResult(self.duration / speed, beats, downbeats)

    def join(self, other: BeatAnalysisResult) -> BeatAnalysisResult:
        """Join two beat analysis results. shift_amount is the amount to shift the times of the second beat analysis result by."""
        shift_amount = self.duration
        beats = np.concatenate([self.beats, other.beats + shift_amount])
        downbeats = np.concatenate([self.downbeats, other.downbeats + shift_amount])
        return BeatAnalysisResult(self.duration + other.duration, beats, downbeats)

    def get_duration(self):
        return self.duration

    def make_click_track(self, audio: Audio):
        click_track = librosa.clicks(times=self.beats, sr=audio.sample_rate, length=audio.nframes)
        down_click_track = librosa.clicks(times=self.downbeats, sr=audio.sample_rate, length=audio.nframes, click_freq=1500)

        click_track = audio.numpy() * 0.5 + click_track + down_click_track
        click_track = torch.tensor([click_track])
        return Audio(click_track, audio.sample_rate)

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

@dataclass(frozen=True)
class KeyAnalysisResult:
    """Has the following properties:

    key_correlation: list[float] - The correlation of each key."""
    key_correlation: tuple[float, ...]
    chromagram: NDArray[np.float32]

    def __post_init__(self):
        assert len(self.key_correlation) == len(get_keys())
        assert self.chromagram.shape[0] == 12
        self.chromagram.flags.writeable = False

    @property
    def key(self):
        return np.argmax(np.array(self.key_correlation)).item()

    @property
    def key_name(self):
        return get_keys()[self.key]

    def get_correlation(self, key: str):
        assert key in get_keys()
        return self.key_correlation[get_keys().index(key)]

    def show_correlations(self):
        for key, corr in zip(get_keys(), self.key_correlation):
            print(f"{key}: {corr}")

    def plot_chromagram(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.chromagram, aspect='auto', origin='lower')
        plt.show()

@dataclass(frozen=True)
class ChordAnalysisResult:
    """A class with the following attributes:

    labels: list[int] - The chord labels for each frame
    chords: list[str] - The chord names. `chords[labels[i]]` is the chord name for the ith frame
    times: list[float] - The times of each frame"""
    duration: float
    labels: NDArray[np.uint8]
    times: NDArray[np.float64]

    def __post_init__(self):
        assert self.times[0] == 0, "First chord must appear at 0"

        chords = get_idx2voca_chord()
        assert self.labels.shape[0] == self.times.shape[0]
        assert np.all(self.labels < len(chords))
        assert self.times.shape[0] > 0
        assert self.duration >= self.times[-1]

        assert isinstance(self.labels, np.ndarray)
        assert isinstance(self.times, np.ndarray)
        assert self.labels.dtype == np.uint8
        assert self.times.dtype == np.float64

        # Check that the times are sorted
        assert np.all(self.times[1:] >= self.times[:-1]), "Times must be sorted"

        self.labels.flags.writeable = False
        self.times.flags.writeable = False

    @classmethod
    def from_data(cls, duration: float, labels: list[int], times: list[float]):
        """Construct a ChordAnalysisResult from native python types. Should function identically as the old constructor."""
        return cls(duration, np.array(labels, dtype=np.uint8), np.array(times, dtype=np.float64))

    def group(self) -> ChordAnalysisResult:
        """Group the chord analysis result by chord, and deletes the duplicate chords."""
        labels = np.zeros_like(self.labels)
        times = np.zeros_like(self.times)
        idx = 0
        for i in range(self.labels.shape[0]):
            if idx == 0 or self.labels[i] != labels[idx - 1]:
                labels[idx] = self.labels[i]
                times[idx] = self.times[i]
                idx += 1
        return ChordAnalysisResult(self.duration, labels[:idx], times[:idx])

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

    def slice_seconds(self, start: float, end: float) -> ChordAnalysisResult:
        """Slice the chord analysis result by seconds and shifts the times to start from 0
        includes start and excludes end"""
        assert start >= 0
        if abs(end - self.duration) < 1e-6 or end == -1:
            end = self.duration
        assert end > start and end <= self.duration
        new_times, new_labels = _slice_chord_result(self.times, self.labels, start, end)
        return ChordAnalysisResult(end - start, new_labels, new_times)

    def join(self, other: ChordAnalysisResult) -> ChordAnalysisResult:
        """Join two chord analysis results. shift_amount is the amount to shift the times of the second chord analysis result by."""
        shift_amount = self.duration
        labels = np.concatenate([self.labels, other.labels])
        times = np.concatenate([self.times, other.times + shift_amount])
        return ChordAnalysisResult(self.duration + other.duration, labels, times)

    def change_speed(self, speed: float) -> ChordAnalysisResult:
        """Change the speed of the chord analysis result"""
        times = self.times / speed
        return ChordAnalysisResult(self.duration / speed, self.labels, times)

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
        np_labels = np.array(labels, dtype=np.uint8)
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
def _slice_chord_result(times: NDArray[np.float64], labels: NDArray[np.uint8], start: float, end: float):
    """This function is used as an optimization to calling slice_seconds, then group_labels/group_times on a ChordAnalysis Result"""
    start_idx = np.searchsorted(times, start, side='right') - 1
    end_idx = np.searchsorted(times, end, side='right')
    new_times = times[start_idx:end_idx] - start
    # Set the start to 0 if the first chord is before the start
    new_times[0] = 0.
    new_labels = labels[start_idx:end_idx]
    return new_times, new_labels
