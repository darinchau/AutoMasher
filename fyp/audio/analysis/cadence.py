from ... import Audio
from .base import CadenceAnalysisResult, ChordAnalysisResult, BeatAnalysisResult, KeyAnalysisResult
from .key import analyse_key_center_chroma
from .beat import analyse_beat_transformer
from .chord import analyse_chord_transformer
from ...util.note import get_chord_notes, get_idx2voca_chord, notes_to_idx, get_inv_voca_map, get_keys, transpose_chord, idx_to_notes
import numpy as np
from numpy.typing import NDArray

def transpose_chord_idx(chord: int, semitone: int) -> int:
    chord_name = get_idx2voca_chord()[chord]
    return get_inv_voca_map()[transpose_chord(chord_name, semitone)]

def get_major_minor_chords():
    inv_voca = get_inv_voca_map()

    major_chords = [
        inv_voca["C"],
        inv_voca["D:min"],
        inv_voca["E:min"],
        inv_voca["F"],
        inv_voca["G"],
        inv_voca["A:min"],
        inv_voca["B:dim"]
    ]

    minor_chords = [
        inv_voca["C:min"],
        inv_voca["D"],
        inv_voca["D#"],
        inv_voca["F:min"],
        inv_voca["G"],
        inv_voca["G#"],
        inv_voca["B:dim"]
    ]

    assert len(major_chords) == len(minor_chords) == 7
    return major_chords, minor_chords

# Durations is a (slice_start, slice_mid, slice_end, multiplier) tuple
# Each value is a time in seconds
# The slice_mid is the chord change point
# The multiplier is a number to multiply to the chord distance
# The longer the duration, the stronger the cadence ought to be.
# And the more likely the match is actually real.
def get_boundaries(bt: BeatAnalysisResult, bar_number: int):
    b_2 = bt.downbeats[bar_number - 2]
    b_1 = bt.downbeats[bar_number - 1]
    b_0 = bt.downbeats[bar_number]
    b_half = (b_1 + b_0) / 2
    b_quarter = (b_1 + b_0 * 3) / 4

    durations_to_consider: list[tuple[float, float, float, float]] = [
        (b_2, b_1, b_0, 1),
        (b_1, b_half, b_0, 0.9),
        (b_half, b_quarter, b_0, 0.8)
    ]

    b_plus_1 = bt.downbeats[bar_number + 1]
    b_plus_2 = bt.downbeats[bar_number + 2]
    b_plus_half = (b_0 + b_plus_1) / 2

    if bar_number < bt.downbeats.shape[0] - 1:
        durations_to_consider.extend([
            (b_1, b_0, b_plus_1, 1.05),
            (b_half, b_0, b_plus_half, 0.95)
        ])

    if bar_number < bt.downbeats.shape[0] - 2:
        durations_to_consider.append((b_2, b_0, b_plus_2, 1.1))

    return durations_to_consider

def chroma_chord(audio: Audio, hop: int = 512, ct: ChordAnalysisResult | None = None, **kwargs):
    if ct is None:
        ct = analyse_chord_transformer(audio)

    dims = (12, int(audio.nframes / hop + 0.5))
    chroma = np.zeros(dims, dtype=np.float32)

    boundaries = ct.times.tolist() + [ct.duration]
    for i in range(len(boundaries) - 1):
        lower = int(boundaries[i] * audio.sample_rate / hop + 0.5)
        upper = int(boundaries[i + 1] * audio.sample_rate / hop + 0.5)
        notes = np.array([notes_to_idx(note) for note in get_chord_notes()[ct.chords[i]]], dtype=int)
        chroma[notes, lower:upper] = 1

    return chroma

# Slice the audio according to the duration
# At this stage, calculate everything because we don't know what we want yet
# If we are sure about what we want, we can probably optimize this
# But for now, this is an O(1) operation because despite all the loops, the number of iterations is fixed
# And the innter chord results is run at most 24 keys * 6 durations * 7 chords * 7 chords = 7056 times
def compute_chord_pair_correlations(ct: ChordAnalysisResult, boundadries: list[tuple[float, float, float, float]], chord_options: list[list[int]]):
    """Compute the cadence array. ct is the chord progression, durations is the array from get_durations, and chord_options is the roman numeral analysis chord array for the 7 scale degrees.
    The chord_options must be a list of 7 chords, each with 7 notes.

    Returns a 4D array, where arr[i, j, k, l] is the correspondence score between the kth and the lth chord in the ith chord option in the jth duration to consider"""
    from ..search.align import distance_of_chord_results

    for c in chord_options:
        assert len(c) == 7, "Not a valid chord option"

    boundary_ = [
        (b3 - b1, b2 - b1, ct.slice_seconds(b1, b3), multiplier) for b1, b2, b3, multiplier in boundadries
    ]

    cadence_correlations = np.zeros((len(chord_options), len(boundadries), 7, 7), dtype=np.float32)

    for l, option in enumerate(chord_options):
        for k, (duration, mid_boundary, sliced_ct, multiplier) in enumerate(boundary_):
            for i, c1 in enumerate(option):
                for j, c2 in enumerate(option):
                    cadence = ChordAnalysisResult.from_data(
                        duration = duration,
                        labels = [c1, c2],
                        times = [0, mid_boundary],
                    )
                    # The smaller the distance, the better, so we divide by the multiplier
                    dist = distance_of_chord_results(sliced_ct, cadence) / duration / multiplier
                    cadence_correlations[l, k, i, j] = dist

    return cadence_correlations

def calculate_cadence_array(ct: ChordAnalysisResult, bt: BeatAnalysisResult, kt: KeyAnalysisResult, bar_number: int, probability_inflate_factor: float = 8) -> NDArray[np.float32]:
    """Calculate the cadence array. ct is the chord progression, bt is the beat analysis, kt is the key analysis, bar_number is the bar number to consider, and

    probability_inflate_factor is the factor to inflate the probability of the key center

    Return a 3D array, where arr[i, j, k] is the correspondence score of the j -> k chord in the ith key"""
    major_chords, minor_chords = get_major_minor_chords()
    ex = np.array(kt.key_correlation) * probability_inflate_factor
    key_log_probs = ex - np.log(np.sum(np.exp(ex)))

    cadences = np.zeros((len(get_keys()), 7, 7), dtype=np.float32)
    for i, key in enumerate(get_keys()):
        key_morph_idx = notes_to_idx(key.split(" ")[0].strip())
        chord = major_chords if key.strip()[-5:].lower() == "major" else minor_chords
        chord = [transpose_chord_idx(c, key_morph_idx) for c in chord]
        cadences_for_key = compute_chord_pair_correlations(ct, get_boundaries(bt, bar_number), [chord])[0]
        cadences[i] = np.min(cadences_for_key, axis=0)

    # Calculate the log probabilities to be more numerically stable when multiplying by the inflate factors
    cadences = np.log(cadences)
    cadences += key_log_probs[:, None, None]
    return cadences

def inquire_cadence_score(cadences: np.ndarray, key: str, inquire_cadence: str) -> float:
    """Example: inquire_cadence_score(cadences, "C major", "I -> IV")"""
    roman_to_idx = {
        "I": 0,
        "II": 1,
        "III": 2,
        "IV": 3,
        "V": 4,
        "VI": 5,
        "VII": 6
    }

    assert key in get_keys(), f"Invalid key: {key}"
    assert CadenceAnalysisResult.is_cadence_result(inquire_cadence), f"Invalid cadence: {inquire_cadence}"
    key_idx = get_keys().index(key)
    cadence_idx1 = roman_to_idx[inquire_cadence.split("->")[0].strip()]
    cadence_idx2 = roman_to_idx[inquire_cadence.split("->")[1].strip()]
    cadence_score = np.exp(cadences[key_idx, cadence_idx1, cadence_idx2]).item()
    return cadence_score

def create_cadence_analysis_result(cadences: np.ndarray):
    cadences_to_consider = [
        "V -> I",
        "I -> V",
        "II -> V",
        "IV -> V",
        "IV -> I",
        "I -> I",
        "V -> VI",
        "V -> IV",
        "V -> V",
        "I -> IV"
    ]

    max_key = None
    max_score = float("-inf")
    max_cadence = None
    for key in get_keys():
        for cadence in cadences_to_consider:
            score = inquire_cadence_score(cadences, key, cadence)
            if score > max_score:
                max_score = score
                max_key = key
                max_cadence = cadence

    assert max_key is not None
    assert max_cadence is not None
    return CadenceAnalysisResult(max_key, max_cadence, max_score)

def analyse_cadence(audio: Audio, bar_number: int, ct: ChordAnalysisResult, bt: BeatAnalysisResult, hop: int = 512, probability_inflate_factor: float = 6, key_deduction_window: float = 10.) -> CadenceAnalysisResult:
    """Analyse the cadence of an audio file. Bar number must be at least 2 and less than the number of bars in the song."""
    if bt is None:
        bt = analyse_beat_transformer(audio)

    if bar_number < 2 or bar_number >= bt.nbars:
        raise ValueError(f"Invalid bar number (2 <= bar_number < {bt.nbars})")

    slice_lower = max(0, bt.downbeats[bar_number] - key_deduction_window)
    slice_upper = slice_lower + key_deduction_window
    sliced = audio.slice_seconds(slice_lower, slice_upper)

    key = analyse_key_center_chroma(sliced, chroma_chord(sliced, hop, ct.slice_seconds(slice_lower, slice_upper)), hop=hop)
    cadences = calculate_cadence_array(ct, bt, key, bar_number = bar_number, probability_inflate_factor=probability_inflate_factor)
    return create_cadence_analysis_result(cadences)
