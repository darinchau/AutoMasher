# The module that implements the mashability score
from ...util.note import get_chord_notes
from dataclasses import dataclass
from typing import Any
from ..base.audio import Audio
from ...util.note import get_keys, get_idx2voca_chord, transpose_chord, get_inv_voca_map, get_chord_note_inv
from ...util import YouTubeURL
from ..analysis import ChordAnalysisResult, BeatAnalysisResult
from ..analysis.chord import _get_distance_array
from tqdm.auto import tqdm
from threading import Thread
from queue import Queue, Empty
from typing import Callable
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray
import numba
import heapq
import typing
from ..dataset import SongDataset, DatasetEntry, SongGenre
from ..dataset.create import get_normalized_chord_result
from math import exp

@dataclass(frozen=True)
class MashabilityResult:
    """A class to store the result of the mashability of the songs.
    id: The id of the song in the dataset (aka the 11 character url id)
    start_bar: The starting bar of the song
    transpose: Number of semitones to transpose the sample song to match the submitted song
    title: The title of the song
    timestamp: The timestamp of the song in the Audio
    genre: The genre of the song"""
    url: YouTubeURL
    start_bar: int
    transpose: int
    timestamp: float
    genre: SongGenre

    @property
    def title(self):
        return self.url.title

    def __repr__(self):
        return f"MashabilityResult({self.url}/{self.start_bar}/{self.transpose})"

MashabilityResultType = tuple[int, int, Any, DatasetEntry] # (start_bar, transpose, starting_downbeat, entry)

class MashabilityList:
    def __init__(self):
        self.ls: list[tuple[float, MashabilityResultType]] = []

    def insert(self, distance: float, result: MashabilityResultType):
        self.ls.append((distance, result))

    def get(self, keep_first_k: int, filter_top_scores: bool) -> list[tuple[float, MashabilityResult]]:
        results = [(distance, MashabilityResult(
            url=entry.url,
            start_bar=i,
            transpose=k,
            timestamp=start,
            genre=entry.genre,
        )) for distance, (i, k, start, entry) in self.ls]

        # Separate into few cases to make this efficient
        if keep_first_k > 0 and not filter_top_scores:
            results = heapq.nsmallest(keep_first_k, results, key=lambda x: x[0]) # This is sorted

        elif filter_top_scores:
            results_ = sorted(results, key=lambda x: x[0])
            seen = set()
            results: list[tuple[float, MashabilityResult]] = []
            for distance, result in results_:
                if result.url not in seen:
                    results.append((distance, result))
                    seen.add(result.url)
                if len(results) == keep_first_k:
                    break
        return results

def curve_score(score: float) -> float:
    """Returns the curve score"""
    return round(100 * exp(-.05 * score), 2)

# Calculates the distance of chord results except we only put everything as np arrays for numba jit
@numba.jit(numba.float64(numba.float64[:], numba.float64[:], numba.uint8[:], numba.uint8[:], numba.int32[:, :]), locals={
    "score": numba.float64,
    "cumulative_duration": numba.float64,
    "idx1": numba.int32,
    "idx2": numba.int32,
    "len_t1": numba.int32,
    "len_t2": numba.int32,
    "min_time": numba.float64,
    "new_score": numba.float64
},nopython=True)
def _dist_chord_results(times1, times2, chords1, chords2, distances):
    """A jitted version of the chord result distance calculation, which is defined to be the sum of distances times time
    between the two chord results. The distance between two chords is defined to be the distance between the two chords.
    Refer to our report for more detalis."""
    score = 0
    cumulative_duration = 0.

    idx1 = 0
    idx2 = 0
    len_t1 = len(times1)
    len_t2 = len(times2)

    while idx1 < len_t1 and idx2 < len_t2:
        # Find the duration of the next segment to calculate
        min_time = min(times1[idx1], times2[idx2])

        # Score = sum of (distance * duration)
        # new_score = distances[label1[idx1]][label2[idx2]]
        new_score = distances[chords1[idx1]][chords2[idx2]]
        score += new_score * (min_time - cumulative_duration)
        cumulative_duration = min_time

        if times1[idx1] <= min_time:
            idx1 += 1
        if times2[idx2] <= min_time:
            idx2 += 1
    return score

def calculate_boundaries(beat_result: BeatAnalysisResult, sample_beat_result: BeatAnalysisResult) -> tuple[list[float], list[float]]:
    """Calculates the boundaries and factors to align the beats of sample song to submitted song
    Requires that t=0 are downbeats for both beat results
    Assume submitted song has the same number of bars as sample song
    the speed of the trailing segment of sample song will follow submitted song"""
    assert beat_result.downbeats.shape[0] > 1, "There are not enough downbeat information about submitted song"
    assert sample_beat_result.downbeats.shape[0] > 1, "There are not enough downbeat information about sample song"
    assert beat_result.downbeats.shape[0] == sample_beat_result.downbeats.shape[0], "The number of downbeats in submitted song and sample song are different"
    assert beat_result.downbeats[0] < 1e-5, "The first downbeat of submitted song is not at t=0"
    assert sample_beat_result.downbeats[0] < 1e-5, "The first downbeat of sample song is not at t=0"

    submitted_beat_times = np.append(beat_result.downbeats, beat_result.duration)
    sample_beat_times = np.append(sample_beat_result.downbeats, sample_beat_result.duration)

    orig_lengths = submitted_beat_times[1:] - submitted_beat_times[:-1]
    new_lengths = sample_beat_times[1:] - sample_beat_times[:-1]
    factors = new_lengths / orig_lengths
    boundaries = sample_beat_times[1:]

    return factors.tolist(), boundaries.tolist()

def get_valid_starting_points(music_duration: list[float],
                              sample_downbeats: NDArray[np.float64],
                              sample_beats: NDArray[np.float64],
                              nbars: int,
                              min_music_percentage: float) -> list[int]:
    cumulative_music_durations = np.cumsum(music_duration)
    cumulative_music_durations[nbars:] = cumulative_music_durations[nbars:] - cumulative_music_durations[:-nbars]
    cumulative_music_durations = cumulative_music_durations[nbars-1:-1]
    good_music_duration = cumulative_music_durations >= min_music_percentage * nbars

    # Make sure all the nbars of indices have 4 beats and a good alignment
    beat_alignment_arr = np.abs(sample_beats[:, None] - sample_downbeats[None, :])
    beat_align_idx = beat_alignment_arr.argmin(axis = 0)
    is_4_beats = (beat_align_idx[1:] - beat_align_idx[:-1]) == 4
    beat_alignment = beat_alignment_arr.min(axis = 0)
    is_alignment_within_tolerance = (beat_alignment < 0.1)[:-1]
    good_alignment = np.convolve(is_4_beats & is_alignment_within_tolerance, np.ones(nbars, dtype=int), mode='valid') == nbars

    # Find the indices where the sum of the music duration is greater than the minimum music duration
    # and satisfy the beat alignment
    valid_indices = np.where(good_music_duration & good_alignment)[0]
    return valid_indices.tolist()

def calculate_mashability(submitted_chord_result: ChordAnalysisResult, submitted_beat_result: BeatAnalysisResult,
                            dataset: SongDataset,
                            max_transpose: typing.Union[int, tuple[int, int]] = 3,
                            min_music_percentage: float = 0.5,
                            max_delta_bpm: float = 1.1,
                            min_delta_bpm: float = 0.9,
                            max_distance: float = float("inf"),
                            keep_first_k: int = 10,
                            filter_top_scores: bool = True,
                            should_curve_score: bool = True,
                            verbose: bool = True,
        ) -> list[tuple[float, MashabilityResult]]:
    """Calculate the mashability of the submitted song with the dataset.
    Assuming the chord result is always short enough
    Assume t=0 is always a downbeat
    Returns the best score and the best entry in the dataset."""
    assert submitted_beat_result.downbeats is not None
    assert submitted_beat_result.downbeats[0] == 0.
    assert submitted_chord_result.duration == submitted_beat_result.duration

    submitted_normalized_cr = get_normalized_chord_result(submitted_chord_result, submitted_beat_result)

    # Transpose the submitted chord result in the opposite direction to speed up calculation
    transposed_crs: list[tuple[int, NDArray[np.float64], NDArray[np.uint8]]] = []
    if isinstance(max_transpose, int):
        max_transpose = (-max_transpose, max_transpose)
    else:
        max_transpose = max_transpose
    for transpose_semitone in range(max_transpose[0], max_transpose[1] + 1):
        new_chord_result = submitted_normalized_cr.transpose(-transpose_semitone)
        times1, chords1 = new_chord_result.grouped_end_times_labels()
        transposed_crs.append((transpose_semitone, times1, chords1))

    # Precalculate chord distances as a numpy array to take advantage of jit
    distances = _get_distance_array()
    scores = MashabilityList()
    nbars = len(submitted_beat_result.downbeats)

    # Initiate progress bar
    for entry in tqdm(dataset, desc="Searching database", disable=not verbose):
        sample_downbeats = np.array(entry.downbeats, dtype=np.float64)
        sample_normalized_chords = np.array(entry.chords, dtype = np.uint8)
        sample_normalized_chord_times = np.array(entry.normalized_chord_times, dtype = np.float64)
        submitted_beat_times = np.append(submitted_beat_result.downbeats, submitted_beat_result.duration)
        submitted_beat_times_diff = submitted_beat_times[1:] - submitted_beat_times[:-1]

        for i in get_valid_starting_points(entry.music_duration, sample_downbeats, np.array(entry.beats, dtype=np.float64), nbars, min_music_percentage):
            is_bpm_within_tolerance = _calculate_tolerance(submitted_beat_times_diff, sample_downbeats, max_delta_bpm, min_delta_bpm, nbars, i)
            if not is_bpm_within_tolerance:
                continue

            times2, chords2 = _slice_chord_result(sample_normalized_chord_times, sample_normalized_chords, i, i+nbars)
            starting_downbeat: float = sample_downbeats[i].item()
            for transpose_semitone, times1, chords1 in transposed_crs:
                new_distance = _dist_chord_results(times1, times2, chords1, chords2, distances)
                if new_distance > max_distance:
                    continue
                # Use a tuple instead of a dataclass for now, will change it back to a dataclass in scores.get
                # This is because using tuple literal syntax skips the step to find the dataclass constructor name
                # in global scope. We profiled the code line by line and found this to save around 30% of runtime
                # Since this gets hit so many times, it's worth it
                new_id = (
                    i,
                    transpose_semitone,
                    starting_downbeat,
                    entry
                )
                scores.insert(new_distance, new_id)
    scores_list = scores.get(keep_first_k, filter_top_scores)

    if should_curve_score:
        scores_list = [(curve_score(x[0]), x[1]) for x in scores_list]
    return scores_list

@numba.njit
def _calculate_tolerance(orig_lengths: np.ndarray, sample_downbeats: np.ndarray, max_delta_bpm: float, min_delta_bpm: float, nbars: int, i: int):
    """An optimized version of the calculate_tolerance function that, given the downbeats, finds if the sample downbeats are within the tolerance of the submitted downbeats.
    This does the slicing and everything in numpy, which is much faster than the original implementation."""
    downbeat_mask = (sample_downbeats >= sample_downbeats[i]) & (sample_downbeats < sample_downbeats[i+nbars])
    downbeats_ = sample_downbeats[downbeat_mask] - sample_downbeats[i]
    downbeats_ = np.append(downbeats_, sample_downbeats[i+nbars] - sample_downbeats[i])
    new_lengths = downbeats_[1:] - downbeats_[:-1]
    factors = new_lengths / orig_lengths
    return factors.max() <= max_delta_bpm and factors.min() >= min_delta_bpm

@numba.jit(nopython=True)
def _slice_chord_result(times: NDArray[np.float64], labels: NDArray[np.uint8], start: float, end: float) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    """This function is used as an optimization to calling slice_seconds, then group_labels/group_times on a ChordAnalysis Result"""
    start_idx = np.searchsorted(times, start, side='right') - 1
    end_idx = np.searchsorted(times, end, side='right')

    new_times = times[start_idx:end_idx] - start

    # shift index by 1 and add the duration at the end for the ending times
    new_times[:-1] = new_times[1:]
    new_times[-1] = end - start
    new_labels = labels[start_idx:end_idx]
    return new_times, new_labels

# Also export function for distance of chord results
def distance_of_chord_results(submitted_chord_result: ChordAnalysisResult, sample_chord_result: ChordAnalysisResult) -> float:
    """Calculate the distance between two chord results."""
    assert submitted_chord_result.duration == sample_chord_result.duration
    times1, chords1 = submitted_chord_result.grouped_end_times_labels()
    times2, chords2 = sample_chord_result.grouped_end_times_labels()
    distances = _get_distance_array()
    return _dist_chord_results(times1, times2, chords1, chords2, distances)
