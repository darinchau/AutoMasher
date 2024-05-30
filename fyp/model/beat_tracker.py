# A snippet of madmom's beat tracking implementation

import numpy as np
import itertools

def threshold_activations(activations, threshold):
    """
    Threshold activations to include only the main segment exceeding the given
    threshold (i.e. first to last time/index exceeding the threshold).
    """
    first = last = 0
    idx = np.nonzero(activations >= threshold)[0]
    if idx.any():
        first = max(first, np.min(idx))
        last = min(len(activations), np.max(idx) + 1)
    return activations[first:last], first

# Isolate this point of madmom requirement to test
def get_path(activations, transitions, min_interval, max_interval, num_tempi, transition_lambda, observation_lambda, threshold):
    from madmom.features.beats_hmm import (BeatStateSpace, BeatTransitionModel, RNNBeatTrackingObservationModel)
    from madmom.ml.hmm import HiddenMarkovModel
    st = BeatStateSpace(min_interval, max_interval, num_tempi)
    tm = BeatTransitionModel(st, transition_lambda)
    om = RNNBeatTrackingObservationModel(st, observation_lambda)
    hmm = HiddenMarkovModel(tm, om, None)
    path, score = hmm.viterbi(activations)
    if not path.any():
        return path, None, score
    beat_range = om.pointers[path]
    return path, beat_range, score

def get_bar_path(activations, nbeats, transitions, min_interval, max_interval, num_tempi, transition_lambda, observation_lambda, threshold):
    from madmom.features.beats_hmm import (BarStateSpace, BarTransitionModel, RNNDownBeatTrackingObservationModel)
    from madmom.ml.hmm import HiddenMarkovModel
    st = BarStateSpace(nbeats, min_interval, max_interval, num_tempi)
    tm = BarTransitionModel(st, transition_lambda)
    om = RNNDownBeatTrackingObservationModel(st, observation_lambda)
    hmm = HiddenMarkovModel(tm, om, None)
    path, score = hmm.viterbi(activations)
    beat_positions = st.state_positions[path].astype(int) + 1
    if not path.any():
        return path, None, beat_positions, score
    beat_range = om.pointers[path]
    return path, beat_range, beat_positions, score


def unpack_beats(activations: np.ndarray, *, 
                 min_bpm: float, max_bpm: float, fps: float, threshold: float,
                 num_tempi: int | None = None,
                 transition_lambda: float = 100, 
                 observation_lambda: float = 6
                 ) -> np.ndarray:
    beats = np.empty(0, dtype=int)
    first = 0
    if threshold:
        activations, first = threshold_activations(activations, threshold)

    if not activations.any():
        return beats
    
    # Initialize the HMM and observation model
    min_interval = 60. * fps / max_bpm
    max_interval = 60. * fps / min_bpm

    path, beat_range, _ = get_path(activations, None, min_interval, max_interval, 
                                num_tempi, transition_lambda, observation_lambda, threshold)
    
    if beat_range is None:
        return beats

    idx = np.nonzero(np.diff(beat_range))[0] + 1
    if beat_range[0]:
        idx = np.r_[0, idx]
    if beat_range[-1]:
        idx = np.r_[idx, beat_range.size]
    if idx.any():
        for left, right in idx.reshape((-1, 2)):
            peak = np.argmax(activations[left:right]) + left
            beats = np.hstack((beats, peak))
    return (beats + first) / float(fps)

def unpack_downbeats(activations: np.ndarray, beats_per_bar: list[int], *,
                     min_bpm: float, max_bpm: float, fps: float, threshold: float,
                     num_tempi: int | None = None,
                     transition_lambda: float = 100,
                     observation_lambda: float = 6
                     ) -> np.ndarray:
    def _process_dbn(x):
        return x[0].viterbi(x[1])

    first = 0
    if threshold:
        activations, first = threshold_activations(activations, threshold)

    if not activations.any():
        return np.empty((0, 2))
    
    # Initialize the HMM and observation model
    min_interval = 60. * fps / max_bpm
    max_interval = 60. * fps / min_bpm

    best_path = None
    best_beat_range = None
    best_score = -1
    best_idx = -1
    best_beat_numbers = None

    for b, beats in enumerate(beats_per_bar):
        path_, beat_range_, beat_numbers_, score_ = get_bar_path(activations, beats, None, min_interval, max_interval, num_tempi, transition_lambda, observation_lambda, threshold)
        if score_ > best_score or best_score == -1:
            best_path = path_
            best_beat_range = beat_range_
            best_beat_numbers = beat_numbers_
            best_score = score_
            best_idx = b

    beats = np.empty(0, dtype=int)
    if best_beat_range is None:
        return np.empty((0, 2))

    idx = np.nonzero(np.diff(best_beat_range.astype(int)))[0] + 1
    if best_beat_range[0]:
        idx = np.r_[0, idx]
    if best_beat_range[-1]:
        idx = np.r_[idx, best_beat_range.size]
    if idx.any():
        for left, right in idx.reshape((-1, 2)):
            peak = np.argmax(activations[left:right]) // 2 + left
            beats = np.hstack((beats, peak))
    # return the beat positions (converted to seconds) and beat numbers
    return np.vstack(((beats + first) / float(fps), best_beat_numbers[beats])).T
