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
from ..base import TimeSeries
from numpy.typing import NDArray
import numpy as np
import numba
import json
from enum import Enum
from ..dataset import DatasetEntry

@dataclass(frozen=True)
class BeatAnalysisResult(TimeSeries):
    """A class that represents the result of a beat analysis."""
    duration: float
    beats: NDArray[np.float32]
    downbeats: NDArray[np.float32]

    def __post_init__(self):
        assert len(self.beats) == 0 or self.duration >= self.beats[-1]
        assert len(self.downbeats) == 0 or self.duration >= self.downbeats[-1]
        assert isinstance(self.beats, np.ndarray)
        assert isinstance(self.downbeats, np.ndarray)

    @property
    def tempo(self):
        bpm = float(np.average(1 / (self.beats[1:] - self.beats[:-1]) * 60))
        return bpm

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
    key_correlation: list[float]

    @property
    def key(self):
        return int(np.argmax(np.array(self.key_correlation)))

    @property
    def key_name(self):
        return get_keys()[self.key]

@dataclass(frozen=True)
class ChordAnalysisResult(TimeSeries):
    """A class with the following attributes:

    labels: list[int] - The chord labels for each frame
    chords: list[str] - The chord names. `chords[labels[i]]` is the chord name for the ith frame
    times: list[float] - The times of each frame"""
    duration: float
    labels: list[int]
    times: list[float]

    def __post_init__(self):
        assert self.times[0] == 0, "First chord must appear at 0"

        chords = get_idx2voca_chord()
        assert len(self.labels) == len(self.times)
        assert all(-1 <= label < len(chords) for label in self.labels) # -1 can be a shorthand for no chord
        # assert all(t1 < t2 for t1, t2 in zip(self.times, self.times[1:])) # Check monotonicity

        assert len(self.times) > 0

        assert self.duration >= self.times[-1]

    def group(self) -> ChordAnalysisResult:
        """Group the chord analysis result by chord, and deletes the duplicate chords."""
        labels: list[int] = []
        times: list[float] = []
        for chord, time in zip(self.labels, self.times):
            if len(labels) == 0 or chord != labels[-1]:
                labels.append(chord)
                times.append(time)
        return ChordAnalysisResult(self.duration, labels, times)

    @property
    def grouped_end_time_np(self):
        """Returns a numpy array of grouped end times. This is mostly useful for dataset search"""
        return np.array(self.group().times[1:] + [self.duration], dtype=np.float64)

    @property
    def grouped_labels_np(self):
        """Returns a numpy array of grouped labels. This is mostly useful for dataset search"""
        return np.array(self.group().labels, dtype=np.int32)

    @property
    def chords(self):
        """Returns a list of chord labels"""
        chords = get_idx2voca_chord()
        chord_labels = [chords[label] for label in self.labels]
        return chord_labels

    @property
    def info(self) -> list[tuple[str, float]]:
        """A utility for getting the chord info for real-time display"""
        return list(zip(self.chords, self.times))

    def slice_seconds(self, start: float, end: float) -> ChordAnalysisResult:
        """Slice the chord analysis result by seconds and shifts the times to start from 0
        includes start and excludes end"""
        assert start >= 0
        if abs(end - self.duration) < 1e-6 or end == -1:
            end = self.duration
        assert end > start and end <= self.duration

        # Find the first chord on or before start
        start_idx = 0
        if start > 0:
            while start_idx < len(self.times) and self.times[start_idx] < start:
                start_idx += 1
            if start_idx >= len(self.times) or self.times[start_idx] > start:
                start_idx -= 1

        times, chords = [0.], [self.labels[start_idx]]
        for i in range(start_idx + 1, len(self.times)):
            if self.times[i] >= end:
                break
            if self.times[i] >= start:
                times.append(self.times[i] - start)
                chords.append(self.labels[i])

        return ChordAnalysisResult(end - start, chords, times)

    def join(self, other: ChordAnalysisResult) -> ChordAnalysisResult:
        """Join two chord analysis results. shift_amount is the amount to shift the times of the second chord analysis result by."""
        shift_amount = self.duration
        self_infos = [(label, time) for label, time in zip(self.labels, self.times)]
        other_infos = [(label, time) for label, time in zip(other.labels, other.times)]

        infos = self_infos + [(label, time + shift_amount) for label, time in other_infos]
        infos.sort(key=lambda x: x[1])
        labels = [info[0] for info in infos]
        times = [info[1] for info in infos]
        return ChordAnalysisResult(self.duration + other.duration, labels, times)

    def change_speed(self, speed: float) -> ChordAnalysisResult:
        """Change the speed of the chord analysis result"""
        labels = self.labels
        times = [time / speed for time in self.times]
        return ChordAnalysisResult(self.duration / speed, labels, times)

    def get_duration(self):
        return self.duration

    def copy(self):
        # Its fine to do a shallow copy because the underlying data is immutable
        labels = self.labels.copy()
        times = self.times.copy()
        return ChordAnalysisResult(self.duration, labels, times)

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
        return ChordAnalysisResult(self.duration, labels, self.times)

    @classmethod
    def from_data_entry(cls, entry: DatasetEntry):
        return cls(entry.length, entry.chords, entry.chord_times)

    def save(self, path: str):
        json_dict = {
            "duration": self.duration,
            "labels": self.labels,
            "times": self.times
        }

        with open(path, "w") as f:
            json.dump(json_dict, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data["duration"], data["labels"], data["times"])

@dataclass(frozen=True)
class TimeSegmentResult(TimeSeries):
    """A class that represents the result of a time segment analysis. Internally this owns a beat analysis result
    with only the beat times that are in the time segment. This is useful for aligning time segments with beats."""

    _ba : BeatAnalysisResult

    @classmethod
    def from_data(cls, times: list[float], duration: float):
        return cls(BeatAnalysisResult(duration, np.array(times, dtype=np.float32), np.array([], dtype=np.float32)))

    def slice_seconds(self, start: float, end: float) -> TimeSegmentResult:
        ba = self._ba.slice_seconds(start, end)
        return TimeSegmentResult(ba)

    def change_speed(self, speed: float) -> TimeSegmentResult:
        ba = self._ba.change_speed(speed)
        return TimeSegmentResult(ba)

    def join(self, other: TimeSegmentResult) -> TimeSegmentResult:
        ba = self._ba.join(other._ba)
        return TimeSegmentResult(ba)

    def get_duration(self):
        return self._ba.get_duration()

    def align_with_closest_downbeats(self, beat_result: BeatAnalysisResult) -> list[int]:
        """Align the time segments with the closest downbeats"""
        diffs = self._ba.beats[None, :] - beat_result.downbeats[:, None]
        beat_indices = np.argmin(np.abs(diffs), axis = 0)
        return beat_indices.tolist()
