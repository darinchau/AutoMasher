# The module that implements the mashability score
import os
from dataclasses import dataclass
from typing import Any
from ...util import YouTubeURL
from ..analysis import ChordAnalysisResult, BeatAnalysisResult, SimpleChordAnalysisResult
from tqdm.auto import tqdm
import numpy as np
from numpy.typing import NDArray
import numba
import heapq
import typing
from ..dataset import SongDataset, DatasetEntry, SongGenre
from math import exp
from ..analysis.base import (
    OnsetFeatures,
    DiscreteLatentFeatures,
    ContinuousLatentFeatures,
    dist_discrete_latent_features,
    _dist_discrete_latent,
    dist_continuous_latent_features,
)
from ..analysis.chord import ChordAnalysisResult
from ...util import simplify_chord

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

MashabilityResultType = tuple[int, int, float, DatasetEntry] # (start_bar, transpose, starting_downbeat, entry)

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

def calculate_onset_boundaries(submitted_onset_result: OnsetFeatures, sample_onset_result: OnsetFeatures) -> tuple[list[float], list[float]]:
    """Aligns sample_onset_result to submitted_onset_result, and returns the boundaries and factors

    The returned results satisfy and are intended to be used in the following manner:
    - Slice the sample song at 0 -> boundaries[0], boundaries[0] -> boundaries[1], ...
    - Stretch each slice by factors[0], factors[1], ... respectively
    - The result should be aligned with the submitted onset

    This requires:
    - The first downbeat of the submitted song to be at t=0
    - The first downbeat of the sample song to be at t=0
    - The number of downbeats in the submitted song and the sample song to be the same
    - The number of downbeats in the submitted song and the sample song to be more than 1
    """
    assert submitted_onset_result.onsets.shape[0] > 1, "There are not enough downbeat information about submitted song"
    assert sample_onset_result.onsets.shape[0] > 1, "There are not enough downbeat information about sample song"
    assert submitted_onset_result.onsets.shape[0] == sample_onset_result.onsets.shape[0], "The number of downbeats in submitted song and sample song are different"
    assert submitted_onset_result.onsets[0] < 1e-5, "The first downbeat of submitted song is not at t=0"
    assert sample_onset_result.onsets[0] < 1e-5, "The first downbeat of sample song is not at t=0"

    submitted_beat_times = np.append(submitted_onset_result.onsets, submitted_onset_result.duration)
    sample_beat_times = np.append(sample_onset_result.onsets, sample_onset_result.duration)

    orig_lengths = submitted_beat_times[1:] - submitted_beat_times[:-1]
    new_lengths = sample_beat_times[1:] - sample_beat_times[:-1]
    factors = new_lengths / orig_lengths
    boundaries = sample_beat_times[1:]

    return factors.tolist(), boundaries.tolist()

def get_valid_starting_points(music_duration: NDArray[np.float64],
                              sample_downbeats: NDArray[np.float64],
                              sample_beats: NDArray[np.float64],
                              nbars: int,
                              min_music_percentage: float) -> NDArray[np.int64]:
    """Get the valid starting points for the sample song to align with the submitted song"""
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
    return valid_indices

def calculate_mashability(
        submitted_entry: DatasetEntry,
        dataset: SongDataset,
        submitted_features: list[DiscreteLatentFeatures | ContinuousLatentFeatures] | None = None,
        features_fn: list[typing.Callable[[SongDataset, YouTubeURL, int], DiscreteLatentFeatures | ContinuousLatentFeatures]] | None = None,
        weights: list[float] | None = None,
        max_transpose: typing.Union[int, tuple[int, int]] = 3,
        min_music_percentage: float = 0.5,
        delta_bpm: tuple[float, float] = (0.9, 1.1),
        max_distance: float = float("inf"),
        use_simplified_chord_distance: bool = False,
        keep_first_k: int = 10,
        filter_top_scores: bool = True,
        verbose: bool = True,
    ) -> list[tuple[float, MashabilityResult]]:
    """Calculate the mashability of the submitted song with the dataset.

    Args:
        submitted_entry: The downbeats of the submitted song
        dataset: The dataset to compare the submitted song to
        submitted_features: The latent features of the submitted song. If None, then we will calculate the latent features using the features_fn.
            If the submitted_entry has the placeholder URL and features_fn is not None, this has to be provided
        features_fn: A list of functions that takes in a SongDataset, a YouTubeURL, and a transposition and returns the latent features of the song
            We will try to calculate everything in parallel - so please make sure that the functions are as efficient as possible and support
            parallel computation
        weights: The weights of the latent features. This weight should be relative to the chord result weight. If None, then all the weights are equal
        max_transpose: The maximum number of semitones to transpose the sample song to match the submitted song
        min_music_percentage: The minimum percentage of music that the sample song must have to be considered
        delta_bpm: The minimum and maximum delta bpm between the sample song and the submitted song
        max_distance: The maximum distance between the sample song and the submitted song. This allows early exit if the distance is too large
            and may significantly speed up the calculation
        use_simplified_chord_distance: Whether to use the simplified chord distance. This is potentially more accurate
        keep_first_k: The number of top scores to keep in the returned results. This will in turn allow us to use a heap to keep track of the top scores
        filter_top_scores: Whether to filter the top scores to only keep the top scores that are unique
        verbose: Whether to print the progress bar

    Returns:
        A list of tuples containing the score and the MashabilityResult"""
    # 0. Sanity check
    nbars = len(submitted_entry.downbeats)

    assert submitted_entry.downbeats.onsets[0] == 0.
    assert submitted_entry.beats.onsets[0] == 0.
    assert submitted_entry.downbeats.duration == submitted_entry.beats.duration

    if features_fn is not None:
        if weights is None:
            weights = [1.] * len(features_fn)
        assert len(weights) == len(features_fn), f"Length of weights and submitted_features must be the same. Got {len(weights)} and {len(features_fn)}"

        if submitted_features is None:
            if submitted_entry.url.is_placeholder:
                raise ValueError("If the submitted entry has a placeholder URL, then the submitted_features must be provided")
            submitted_features = [fn(dataset, submitted_entry.url, 0) for fn in features_fn]
    else:
        features_fn = []
        weights = []
        submitted_features = []

    # 1. Pretranspose the chord results
    # Transpose in the opposite direction for the submitted song for the best performance
    transposed_normalized_crs: list[tuple[int, NDArray[np.float64], NDArray[np.uint32]]] = []
    if isinstance(max_transpose, int):
        max_transpose = (-max_transpose, max_transpose)
    for transpose_semitone in range(max_transpose[0], max_transpose[1] + 1):
        transpose_cr = submitted_entry.chords.transpose(-transpose_semitone)
        if use_simplified_chord_distance:
            transpose_cr = transpose_cr.simplify()
        #TODO group the times and labels
        transposed_normalized_crs.append((transpose_semitone, submitted_entry.normalized_times, transpose_cr.features))

    scores = MashabilityList()

    # 2. Precalculate distance arrays
    dist_arrays = [x.get_dist_array() if isinstance(x, DiscreteLatentFeatures) and x.latent_size() < 3000 else None for x in submitted_features]

    # 3. Calculate the distance between the submitted song and the sample song for each song
    #TODO - Implement the parallel version of this
    if use_simplified_chord_distance:
        chord_distances_array = SimpleChordAnalysisResult.get_dist_array()
    else:
        chord_distances_array = ChordAnalysisResult.get_dist_array()
    for entry in tqdm(dataset, desc="Searching database", disable=not verbose):
        valid_start_points = get_valid_starting_points(
            music_duration=entry.music_duration,
            sample_downbeats=entry.downbeats.onsets,
            sample_beats=entry.beats.onsets,
            nbars=nbars,
            min_music_percentage=min_music_percentage
        )

        submitted_beat_times = np.append(submitted_entry.downbeats.onsets, submitted_entry.duration)
        submitted_beat_times_diff = submitted_beat_times[1:] - submitted_beat_times[:-1]

        for i in valid_start_points:
            is_bpm_within_tolerance = _calculate_tolerance(
                submitted_beat_times_diff,
                entry.downbeats.onsets,
                delta_bpm[1],
                delta_bpm[0],
                nbars,
                i
            )
            if not is_bpm_within_tolerance:
                continue

            times2, chords2 = _slice_and_group_end(entry.normalized_times, entry.chords.features, i, i+nbars)
            starting_downbeat_time: float = entry.downbeats.onsets[i].item()
            best_distance_for_current_song = max_distance
            if use_simplified_chord_distance:
                chords2 = np.array([simplify_chord(label) for label in chords2], dtype=np.uint32)
            for transpose_semitone, times1, chords1 in transposed_normalized_crs:
                new_distance = _dist_discrete_latent(times1, times2, chords1, chords2, chord_distances_array, nbars)
                if new_distance > best_distance_for_current_song:
                    continue

                # Calculate the distance between the latent features
                should_break = False
                if submitted_features is not None:
                    for i, (fn, feat, da, w) in enumerate(zip(features_fn, submitted_features, dist_arrays, weights)):
                        feature = fn(dataset, entry.url, transpose_semitone)
                        assert isinstance(feat, type(feature)) and isinstance(feature, type(feat)), f"Expected {type(feature)} for the {i}-th submitted feature, got {type(feat)}"
                        if isinstance(feat, DiscreteLatentFeatures) and isinstance(feature, DiscreteLatentFeatures):
                            new_distance += w * dist_discrete_latent_features(feat, feature, da)
                        elif isinstance(feat, ContinuousLatentFeatures) and isinstance(feature, ContinuousLatentFeatures):
                            new_distance += w * dist_continuous_latent_features(feat, feature)

                        if new_distance > best_distance_for_current_song:
                            should_break = True
                            break
                    if should_break:
                        continue

                # Use a tuple instead of a dataclass for now, will change it back to a dataclass in scores.get
                # This is because using tuple literal syntax skips the step to find the dataclass constructor name
                # in global scope. We profiled the code line by line and found this to save around 30% of runtime
                # Since this gets hit so many times, it's worth it
                new_id = (
                    i,
                    transpose_semitone,
                    starting_downbeat_time,
                    entry
                )
                scores.insert(new_distance, new_id)
    scores_list = scores.get(keep_first_k, filter_top_scores)
    return scores_list

@numba.njit
def _calculate_tolerance(orig_lengths: np.ndarray, sample_downbeats: np.ndarray, max_delta_bpm: float, min_delta_bpm: float, nbars: int, i: int) -> bool:
    """An optimized version of the calculate_tolerance function that, given the submitted downbeats, finds if the
    sample downbeats from bar i to bar i+nbar are within the BPM tolerance of the submitted downbeats.
    This does the slicing and everything in numpy, which is much faster than the original implementation."""
    downbeat_mask = (sample_downbeats >= sample_downbeats[i]) & (sample_downbeats < sample_downbeats[i+nbars])
    downbeats_ = sample_downbeats[downbeat_mask] - sample_downbeats[i]
    downbeats_ = np.append(downbeats_, sample_downbeats[i+nbars] - sample_downbeats[i])
    new_lengths = downbeats_[1:] - downbeats_[:-1]
    factors = new_lengths / orig_lengths
    return factors.max() <= max_delta_bpm and factors.min() >= min_delta_bpm

@numba.jit(nopython=True)
def _slice_and_group_end(times: NDArray[np.float64], labels: NDArray[np.uint32], start: float, end: float) -> tuple[NDArray[np.float64], NDArray[np.uint32]]:
    """This function is used as an optimization to calling slice_seconds, then group_labels/group_times on a ChordAnalysis Result"""
    start_idx = np.searchsorted(times, start, side='right') - 1
    end_idx = np.searchsorted(times, end, side='right')

    new_times = times[start_idx:end_idx] - start

    # shift index by 1 and add the duration at the end for the ending times
    new_times[:-1] = new_times[1:]
    new_times[-1] = end - start
    new_labels = labels[start_idx:end_idx]
    return new_times, new_labels
