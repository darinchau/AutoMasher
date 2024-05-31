# The module that implements the mashability score
from ...util.note import get_chord_notes
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
from ..dataset import SongDataset, DatasetEntry
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
            score, result = distance_of_two_nonempty_chord(chord1, chord2)
        
    assert result in chord_notes_map, f"{result} not a recognised chord"
    return score, result

# Gets the distance of two chords and the closest approximating chord
def distance_of_two_nonempty_chord(chord1: str, chord2: str) -> tuple[int, str]:
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
def calculate_distance_array():
    """Calculates the distance array for all chords. The distance array is a 2D array where the (i, j)th element is the distance between the ith and jth chords.
    This will be cached and used for all future calculations."""
    chord_mapping = get_idx2voca_chord()
    n = len(chord_mapping)
    distance_array = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            distance_array[i][j], _ = _calculate_distance_of_two_chords(chord_mapping[i], chord_mapping[j])
    return np.array(distance_array)

# Calculates the distance of chord results except we only put everything as np arrays for numba jit
@numba.njit
def _dist_chord_results(times1, times2, chords1, chords2, distances):
    """A jitted version of the chord result distance calculation, which is defined to be the sum of distances times time
    between the two chord results. The distance between two chords is defined to be the distance between the two chords.
    Refer to our report for more detalis."""
    score = 0
    cumulative_duration = 0.

    idx1 = 0
    idx2 = 0

    while idx1 < len(times1) and idx2 < len(times2):
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

# Optimize the calls to calculate_boundaries except we don't need to calculate the boundaries and we don't need to convert the factors back to a python list
# We can also jit this
@numba.njit
def _calculate_max_min_factors(db, sample_db, duration, sample_duration, max_factor, min_factor):
    # We take everything in good faith ok so we don't need to check the input
    submitted_beat_times = np.append(db, duration)
    sample_beat_times = np.append(sample_db, sample_duration)

    orig_lengths = submitted_beat_times[1:] - submitted_beat_times[:-1]
    new_lengths = sample_beat_times[1:] - sample_beat_times[:-1]
    factors = new_lengths / orig_lengths
    return factors.max() <= max_factor and factors.min() >= min_factor 

@numba.njit
def _get_sliced_downbeats(downbeats, bounds):
    downbeat_mask = (downbeats >= bounds[0]) & (downbeats < bounds[1])
    return downbeats[downbeat_mask] - bounds[0]

# This is now not needed during the search step because we have precalculated it
def get_music_duration(chord_result: ChordAnalysisResult):
    """Get the duration of actual music in the chord result. This is calculated by summing the duration of all chords that are not "No chord"."""
    music_duration = 0.
    times = chord_result.grouped_times + [chord_result.duration]
    no_chord_idx = get_inv_voca_map()["No chord"]
    for chord, start, end in zip(chord_result.labels, times[:-1], times[1:]):
        if chord != no_chord_idx:
            music_duration += end - start
    return music_duration

def get_normalized_chord_result(cr: ChordAnalysisResult, br: BeatAnalysisResult):
    """Normalize the chord result with the beat result. This is done by retime the chord result as the number of downbeats."""
    # For every time stamp in the chord result, retime it as the number of downbeats.
    # For example, if the time stamp is half way between downbeat[1] and downbeat[2], then it should be 1.5
    # If the time stamp is before the first downbeat, then it should be 0.
    # If the time stamp is after the last downbeat, then it should be the number of downbeats.
    assert cr.duration == br.duration
    downbeats = br.downbeats.tolist() + [br.duration]
    new_chord_times = []
    curr_downbeat, curr_downbeat_idx, next_downbeat = 0, 0, downbeats[1]
    for chord_times in cr.grouped_times:
        while chord_times > next_downbeat:
            curr_downbeat_idx += 1
            curr_downbeat = next_downbeat
            next_downbeat = downbeats[curr_downbeat_idx + 1]
        normalized_time = curr_downbeat_idx + (chord_times - curr_downbeat) / (next_downbeat - curr_downbeat)
        new_chord_times.append(normalized_time)
    return ChordAnalysisResult(len(br.downbeats), cr.grouped_labels, new_chord_times, sanity_check=True)

# Use a more efficient data scructure to store the scores
class MashabilityScore:
    """A class to store the scores of the mashability search. It is a heap that keeps the top k scores if k > 0. Otherwise this behaves like a list."""
    def __init__(self, keep_first_k: int):
        assert keep_first_k > 0, "Use a Mashability List instead"
        self.keep_first_k = keep_first_k
        self.heap = []

    def insert(self, score: float, id: str):
        if len(self.heap) < self.keep_first_k:
            heapq.heappush(self.heap, (-score, id))
        else:
            heapq.heappushpop(self.heap, (-score, id))

    def get(self):
        return sorted([(-score, id) for score, id in self.heap])
    
class MashabilityList:
    def __init__(self):
        self.heap = []

    def insert(self, score: float, id: str):
        self.heap.append((score, id))

    def get(self):
        return sorted(self.heap)

# An optimized version of the legacy code
def search_database(submitted_chord_result: ChordAnalysisResult, submitted_beat_result: BeatAnalysisResult, 
                         dataset: SongDataset, first_k: int | None = None,
                         search_config: SearchConfig | None = None) -> list[tuple[float, str]]:
    """Find the best song in the data list that matches the given audio.
    Assuming the chord result is always short enough
    Assume t=0 is always a downbeat
    Returns the best score and the best entry in the dataset."""
    assert submitted_beat_result.downbeats is not None
    assert submitted_beat_result.downbeats[0] == 0.
    assert submitted_chord_result.duration == submitted_beat_result.duration
    search_config = search_config or SearchConfig()
    
    # Normalize the submitted chord result with the submitted beat result
    submitted_normalized_cr = get_normalized_chord_result(submitted_chord_result, submitted_beat_result)
    times1 = submitted_normalized_cr.grouped_end_time_np
    chords1 = submitted_normalized_cr.grouped_labels_np

    # Transpose the submitted chord result in the opposite direction to speed up calculation
    transposed_crs = []
    if isinstance(search_config.max_transpose, int):
        max_transpose = (-search_config.max_transpose, search_config.max_transpose)
    for k in range(max_transpose[0], max_transpose[1] + 1):
        new_chord_result = submitted_normalized_cr.transpose(-k)
        times1 = new_chord_result.grouped_end_time_np
        chords1 = new_chord_result.grouped_labels_np
        transposed_crs.append((k, times1, chords1))

    # Calculate chord distances
    distances = calculate_distance_array()

    scores = MashabilityScore(keep_first_k=search_config.keep_first_k)
    num_calculated = 0

    # At this point both the submitted and sample are normalized
    nbars = len(submitted_beat_result.downbeats)

    # Initiate progress bar
    total: int = first_k if first_k is not None else len(dataset)
    for entry in tqdm(dataset, total=total, desc="Searching database", disable=not search_config.verbose):
        # Early exit if needed
        if first_k is not None and num_calculated >= first_k:
            break

        # Create and normalize the sample chord result and beat result
        id = entry.url[-11:]
        downbeats = entry.downbeats

        sample_beat_result = BeatAnalysisResult.from_data_entry(entry)
        sample_normalized_cr = ChordAnalysisResult(len(downbeats), entry.chords, entry.normalized_chord_times)
        sample_music_duration = entry.music_duration

        # Calculate the scores for each bar
        for i in range(len(downbeats) - nbars):
            if sum(sample_music_duration[i:i + nbars]) < search_config.min_music_percentage * nbars:
                continue

            # Calculate the boundaries
            start_downbeat = downbeats[i]
            end_downbeat = downbeats[i + nbars]
            start_end = np.array([start_downbeat, end_downbeat], dtype = np.float32)

            trimmed_downbeats = _get_sliced_downbeats(sample_beat_result.downbeats, start_end)
            is_bpm_within_tolerance =  _calculate_max_min_factors(submitted_beat_result.downbeats, trimmed_downbeats, 
                                              submitted_beat_result.duration, end_downbeat - start_downbeat, 
                                              search_config.max_delta_bpm, search_config.min_delta_bpm)
            if not is_bpm_within_tolerance:
                continue

            times2, chords2 = sample_normalized_cr.get_sliced_np(i, i+nbars)

            for k, times1, chords1 in transposed_crs:
                new_score = _dist_chord_results(times1, times2, chords1, chords2, distances)
                
                if search_config.extra_info:
                    timestamp = f"{downbeats[i]//60}:{downbeats[i]%60:.2f}"
                    views = f"{entry.views//1e9}B" if entry.views > 1e9 else f"{entry.views//1e6}M" if entry.views > 1e6 else f"{entry.views//1e3}K"
                    title = entry.audio_name if len(entry.audio_name) < 40 else entry.audio_name[:37] + "..."
                    info_ = search_config.extra_info.format(
                        title = title,
                        timestamp = timestamp,
                        genre = entry.genre,
                        views = views
                    )
                    new_id = f"{id}/{i}/{k}/{info_}"
                else:
                    new_id = f"{id}/{i}/{k}"
                scores.insert(new_score, new_id)
        num_calculated += 1
    scores_list = scores.get()
    return scores_list

# Also export function for distance of chord results
def distance_of_chord_results(submitted_chord_result: ChordAnalysisResult, sample_chord_result: ChordAnalysisResult) -> float:
    """Calculate the distance between two chord results."""
    assert submitted_chord_result.duration == sample_chord_result.duration
    times1 = submitted_chord_result.grouped_end_time_np
    chords1 = submitted_chord_result.grouped_labels_np
    times2 = sample_chord_result.grouped_end_time_np
    chords2 = sample_chord_result.grouped_labels_np

    distances = calculate_distance_array()
    return _dist_chord_results(times1, times2, chords1, chords2, distances)

# Postprocessing filter and mapping functions
def filter_dataset(example):
    beats = example['beats']
    downbeats = example['downbeats']
    chord_times = example['chord_times']

    if not beats or beats[-1] > example['length']:
        return False
    
    if not downbeats or downbeats[-1] > example['length']:
        return False
    
    if not chord_times or chord_times[-1] > example['length']:
        return False
    
    return True

def mapping_dataset(length: float, beats: list[float], downbeats: list[float], chords: list[int], chord_times: list[float],
                    *, genre: str, audio_name: str, url: str, playlist: str, views: int):
    chord_result = ChordAnalysisResult(length, chords, chord_times)
    beat_result = BeatAnalysisResult(length, beats, downbeats)
    normalized_cr = get_normalized_chord_result(chord_result, beat_result)

    # For each bar, calculate its music duration
    music_duration: list[float] = []
    for i in range(len(downbeats)):
        bar_cr = normalized_cr.slice_seconds(i, i + 1)
        music_duration.append(get_music_duration(bar_cr))
    
    return DatasetEntry(
        chords=chords,
        chord_times=chord_times,
        downbeats=downbeats,
        beats=beats,
        genre=genre,
        audio_name=audio_name,
        url=url,
        playlist=playlist,
        views=views,
        length=length,
        normalized_chord_times=normalized_cr.times,
        music_duration=music_duration
    )
