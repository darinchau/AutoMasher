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

class BeatAnalysisResult(TimeSeries):
    """A class that represents the result of a beat analysis."""
    def __init__(self, duration: float, beat_frames: list[float] | NDArray[np.float32], downbeat_frames: list[float] | NDArray[np.float32]):
        # TODO: Check sortedness for the beat frames
        assert len(beat_frames) == 0 or duration >= beat_frames[-1]
        self.beats: np.ndarray = np.array(beat_frames, dtype=np.float32)
        self.downbeats: np.ndarray = np.array(downbeat_frames, dtype=np.float32)
        self.duration = duration

    @property
    def tempo(self):
        bpm = float(np.average(1 / (self.beats[1:] - self.beats[:-1]) * 60))
        return bpm

    @classmethod
    def from_data_entry(cls, data_entry: DatasetEntry):
        return cls(data_entry.length, data_entry.beats, data_entry.downbeats)

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
        return cls(data["duration"], data["beats"], data["downbeats"])

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

@dataclass(init=False)
class TuningAnalysisResult:
    """A data class with the following

    "tuning" (float -0.5 <= x < 0.5): The tuning correlation in terms of semitones"""
    tuning: float

    def __init__(self, tuning: float):
        assert -0.5 <= tuning < 0.5
        self.tuning = tuning

class ChordAnalysisResult(TimeSeries):
    """A class with the following attributes:

    labels: list[int] - The chord labels for each frame
    chords: list[str] - The chord names. `chords[labels[i]]` is the chord name for the ith frame
    times: list[float] - The times of each frame"""

    def __init__(self, duration: float, labels: list[int], times: list[float], sanity_check: bool = False):
        if sanity_check:
            chords = get_idx2voca_chord()
            assert len(labels) == len(times)
            assert all(-1 <= label < len(chords) for label in labels) # -1 can be a shorthand for no chord
            # Check that the times array is monotonically increasing
            last = -1
            for time in times:
                assert time > last
                last = time

            assert len(times) > 0

            assert duration >= times[-1]

        # No chord is also a chord thus we enforce this rule
        assert times[0] == 0, "First chord must appear at 0"

        self.labels = labels
        self.times = times
        self.duration = duration

    def _group(self):
        labels: list[int] = []
        times: list[float] = []
        for chord, time in zip(self.labels, self.times):
            if len(labels) == 0 or chord != labels[-1]:
                labels.append(chord)
                times.append(time)
        self._group_labels = labels
        self._group_times = times

    @property
    def grouped_labels(self) -> list[int]:
        if not hasattr(self, "_group_labels"):
            self._group()
        return self._group_labels

    @property
    def grouped_times(self) -> list[float]:
        if not hasattr(self, "_group_times"):
            self._group()
        return self._group_times

    @property
    def grouped_chords(self) -> list[str]:
        if not hasattr(self, "_groups"):
            self._group()
        chords = get_idx2voca_chord()
        return [chords[label] for label in self.grouped_labels]

    @property
    def grouped_end_time_np(self):
        if not hasattr(self, "_grouped_end_time_np"):
            self._grouped_end_time_np = np.array(self.grouped_times[1:] + [self.duration])
        return self._grouped_end_time_np

    @property
    def grouped_labels_np(self):
        if not hasattr(self, "_grouped_labels_np"):
            self._grouped_labels_np = np.array(self.grouped_labels)
        return self._grouped_labels_np

    @property
    def info(self) -> list[tuple[str, float]]:
        """A utility for getting the chord info for real-time display"""
        return list(zip(self.grouped_chords, self.grouped_times))

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
        for chord, time in zip(self.grouped_chords, self.grouped_times):
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

@numba.jit(nopython=True)
def _slice_chord_result(times, labels, start, end):
    """This function is used as an optimization to calling slice_seconds, then group_labels/group_times on a ChordAnalysis Result"""
    start_idx = np.searchsorted(times, start, side='right') - 1
    end_idx = np.searchsorted(times, end, side='right')

    new_times = times[start_idx:end_idx] - start

    # shift index by 1 and add the duration at the end for the ending times
    new_times[:-1] = new_times[1:]
    new_times[-1] = end - start
    new_labels = labels[start_idx:end_idx]
    return new_times, new_labels

# Time Segment result is almost the same implementation as Beat Analysis Result
class TimeSegmentResult(TimeSeries):
    def __init__(self, valid_start_points: list[float] | NDArray[np.floating], duration: float):
        """Takes the valid starting points for the time segments"""
        self._ba = BeatAnalysisResult(duration, valid_start_points, [])

    def slice_seconds(self, start: float, end: float) -> TimeSegmentResult:
        ba = self._ba.slice_seconds(start, end)
        return TimeSegmentResult(ba.beats.tolist(), ba.get_duration())

    def change_speed(self, speed: float) -> TimeSegmentResult:
        ba = self._ba.change_speed(speed)
        return TimeSegmentResult(ba.beats, ba.get_duration())

    def join(self, other: TimeSegmentResult) -> TimeSegmentResult:
        ba = self._ba.join(other._ba)
        return TimeSegmentResult(ba.beats, ba.get_duration())

    def get_duration(self):
        return self._ba.get_duration()

    def align_with_closest_downbeats(self, beat_result: BeatAnalysisResult) -> list[int]:
        """Align the time segments with the closest downbeats"""
        diffs = self._ba.beats[None, :] - beat_result.downbeats[:, None]
        beat_indices = np.argmin(np.abs(diffs), axis = 0)
        return beat_indices.tolist()
