from __future__ import annotations
import unittest
from dataclasses import dataclass
from fyp.audio.analysis import ChordAnalysisResult as NewChordAnalysisResult
from fyp.audio.base import TimeSeries
from fyp.util import get_idx2voca_chord, get_inv_voca_map, transpose_chord
import numpy as np
from numpy.typing import NDArray
import json
from fyp.audio.dataset import DatasetEntry
from tqdm.auto import trange

@dataclass(frozen=True)
class ChordAnalysisResult(TimeSeries):
    """The old implementation of Chord Analysis Result to test against"""
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

import random

def equal_timestamps(a: NDArray[np.float64], b: list[float], epsilon=1e-6):
    assert len(a.shape) == 1
    assert a.shape[0] == len(b)
    assert all(abs(xa - xb) < epsilon for xa, xb in zip(a, b))

def equal_labels(a: NDArray[np.uint8], b: list[int]):
    assert len(a.shape) == 1
    assert a.shape[0] == len(b)
    assert all(xa == xb for xa, xb in zip(a, b))

def new_equals_old(old: ChordAnalysisResult, new: NewChordAnalysisResult, *, epsilon=1e-6):
    assert abs(old.duration - new.duration) < epsilon
    equal_labels(new.labels, old.labels)
    equal_timestamps(new.times, old.times)

def get_random_pair(n: int):
    labels = [random.randint(0, 169) for _ in range(n)]
    times = [random.random() for _ in range(n)]
    for i in range(1, n):
        times[i] += times[i - 1]
    times[0] = 0.

    duration = times[-1] + random.random() * 3
    cr = ChordAnalysisResult(duration, labels, times)
    new_cr = NewChordAnalysisResult.from_data(duration, labels, times)
    new_equals_old(cr, new_cr)
    return cr, new_cr

def test(n: int):
    cr, new_cr = get_random_pair(n)
    new_equals_old(cr.group(), new_cr.group())
    t1, l1 = new_cr.grouped_end_time_np, new_cr.grouped_labels_np
    t2, l2 = cr.grouped_end_time_np, cr.grouped_labels_np
    assert np.all(l1 == l2)
    equal_labels(l1, l2.tolist())
    assert np.all(t1 == t2)
    equal_timestamps(t1, t2.tolist())
    assert cr.chords == new_cr.chords
    assert cr.info == new_cr.info

    # Test slice_seconds
    start = random.random() * cr.duration
    end = random.random() * cr.duration
    if start > end:
        start, end = end, start
    cr2 = cr.slice_seconds(start, end)
    new_cr2 = new_cr.slice_seconds(start, end)
    new_equals_old(cr2, new_cr2)

    # Test join
    cr3, new_cr3 = get_random_pair(n)
    cr4 = cr.join(cr3)
    new_cr4 = new_cr.join(new_cr3)
    new_equals_old(cr4, new_cr4)

    # Test change_speed
    speed = random.random() * 2
    cr5 = cr.change_speed(speed)
    new_cr5 = new_cr.change_speed(speed)
    new_equals_old(cr5, new_cr5)

    assert cr.get_duration() == new_cr.get_duration()
    assert cr.__repr__() == new_cr.__repr__()

    # Test transpose
    semitones = random.randint(-12, 12)
    cr6 = cr.transpose(semitones)
    new_cr6 = new_cr.transpose(semitones)
    new_equals_old(cr6, new_cr6)


class TestChordAnalysisResult(unittest.TestCase):
    def test_chord_analysis_impl(self):
        for _ in trange(100000):
            n = random.randint(1, 1000)
            test(n)

if __name__ == "__main__":
    unittest.main()
