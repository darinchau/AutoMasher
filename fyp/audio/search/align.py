# The module that implements the mashability score
from ...util.note import get_chord_notes
from dataclasses import dataclass
from typing import Any
from ... import Audio
from ...util.note import get_keys, get_idx2voca_chord, transpose_chord, get_inv_voca_map, get_chord_note_inv
from ..analysis import ChordAnalysisResult, BeatAnalysisResult
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
from .search_config import SearchConfig
NO_CHORD_PENALTY = 3

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
            score, result = 3, "Unknown"

        case (_, "No chord"):
            score, result = NO_CHORD_PENALTY, chord1

        case (_, "Unknown"):
            score, result = 3, "Unknown"

        case ("No chord", _):
            score, result = NO_CHORD_PENALTY, chord2

        case ("Unknown", _):
            score, result = 3, "Unknown"

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

@dataclass(frozen=True)
class MashabilityResult:
    """A class to store the result of the mashability of the songs.
    id: The id of the song in the dataset (aka the 11 character url id)
    start_bar: The starting bar of the song
    transpose: Number of semitones to transpose the sample song to match the submitted song
    title: The title of the song
    timestamp: The timestamp of the song in the Audio
    genre: The genre of the song
    views: The number of views of the song as of the time of the dataset creation"""
    url_id: str
    start_bar: int
    transpose: int
    title: str
    timestamp: float
    genre: SongGenre
    views: int

    def __repr__(self):
        return f"MashabilityResult({self.url_id}/{self.start_bar}/{self.transpose})"

MashabilityResultType = tuple[int, int, Any, DatasetEntry]

# Use a more efficient data scructure to store the scores
class MashabilityHeap:
    """A class to store the scores of the mashability of the songs. It keeps the top k scores in a heap."""
    def __init__(self, keep_first_k: int):
        assert keep_first_k > 0, "Use a Mashability List instead"
        self.heap: list[tuple[float, tuple]] = [(float("-inf"), ()) for _ in range(keep_first_k)]

    def insert(self, score: float, id: MashabilityResultType):
        heapq.heappushpop(self.heap, (-score, id))

    def get(self) -> list[tuple[float, MashabilityResult]]:
        entries = [(-score, id) for score, id in self.heap if len(id) > 0]
        ls = MashabilityList()
        ls.heap = entries
        return ls.get()

class MashabilityList:
    def __init__(self):
        self.heap: list[tuple[float, MashabilityResultType]] = []

    def insert(self, score: float, id: MashabilityResultType):
        self.heap.append((score, id))

    def get(self) -> list[tuple[float, MashabilityResult]]:
        return sorted([(score, MashabilityResult(
            url_id=entry.url_id,
            start_bar=i,
            transpose=k,
            title=entry.audio_name,
            timestamp=start,
            genre=entry.genre,
            views=entry.views
        )) for score, (i, k, start, entry) in self.heap], key=lambda x: x[0])

def calculate_mashability(submitted_chord_result: ChordAnalysisResult, submitted_beat_result: BeatAnalysisResult,
                            dataset: SongDataset,
                            max_transpose: typing.Union[int, tuple[int, int]] = 3,
                            min_music_percentage: float = 0.5,
                            max_delta_bpm: float = 1.1,
                            min_delta_bpm: float = 0.9,
                            max_score: float = float("inf"),
                            keep_first_k: int = 10,
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
    scores = MashabilityHeap(keep_first_k=keep_first_k) if keep_first_k > 0 else MashabilityList()
    nbars = len(submitted_beat_result.downbeats)

    # Initiate progress bar
    for entry in tqdm(dataset, desc="Searching database", disable=not verbose):
        sample_downbeats = np.array(entry.downbeats, dtype=np.float64)
        sample_normalized_chords = np.array(entry.chords, dtype = np.uint8)
        sample_normalized_chord_times = np.array(entry.normalized_chord_times, dtype = np.float64)
        submitted_beat_times = np.append(submitted_beat_result.downbeats, submitted_beat_result.duration)
        submitted_beat_times_diff = submitted_beat_times[1:] - submitted_beat_times[:-1]

        for i in entry.get_valid_starting_points(nbars, min_music_percentage):
            is_bpm_within_tolerance = _calculate_tolerance(submitted_beat_times_diff, sample_downbeats, max_delta_bpm, min_delta_bpm, nbars, i)
            if not is_bpm_within_tolerance:
                continue

            times2, chords2 = _slice_chord_result(sample_normalized_chord_times, sample_normalized_chords, i, i+nbars)
            starting_downbeat = sample_downbeats[i]
            for transpose_semitone, times1, chords1 in transposed_crs:
                new_score = _dist_chord_results(times1, times2, chords1, chords2, distances)
                if new_score > max_score:
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
                scores.insert(new_score, new_id)
    scores_list = scores.get()
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
