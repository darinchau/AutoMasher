from fyp import Audio
from fyp.audio.analysis.base import CadenceAnalysisResult, ChordAnalysisResult, BeatAnalysisResult, KeyAnalysisResult
from fyp.audio.analysis.key import analyse_key_center_chroma
from fyp.audio.analysis.beat import analyse_beat_transformer
from fyp.audio.analysis.chord import analyse_chord_transformer
from fyp.util.note import get_chord_notes, get_idx2voca_chord, notes_to_idx, get_inv_voca_map, get_keys, transpose_chord, idx_to_notes
import numpy as np
from numpy.typing import NDArray
from functools import lru_cache

@lru_cache(maxsize=128)
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

# Get the chromagram of the audio based on the chord analysis result
def chroma_chord(audio: Audio, ct: ChordAnalysisResult, hop: int = 512, **kwargs):
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

def calculate_chord_distance(duration: float, mid_boundary: float, sliced_ct: ChordAnalysisResult, multiplier: float, c1: int, c2: int):
    from fyp.audio.search.align import distance_of_chord_results
    cadence = ChordAnalysisResult.from_data(
        duration = duration,
        labels = [c1, c2],
        times = [0, mid_boundary],
    )
    # The smaller the distance, the better, so we divide by the multiplier
    dist = distance_of_chord_results(sliced_ct, cadence) / duration / multiplier
    return dist

def inquire_cadence_score(ct: ChordAnalysisResult, bt: BeatAnalysisResult, kt: KeyAnalysisResult, bar_number: int, key: str, inquire_cadence: str) -> float:
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
    # key_idx = get_keys().index(key)
    cadence_idx1 = roman_to_idx[inquire_cadence.split("->")[0].strip()]
    cadence_idx2 = roman_to_idx[inquire_cadence.split("->")[1].strip()]

    # Prepare the chords array
    major_chords, minor_chords = get_major_minor_chords()
    key_morph_idx = notes_to_idx(key.split(" ")[0].strip())
    chord = major_chords if key.strip()[-5:].lower() == "major" else minor_chords
    chord = [transpose_chord_idx(c, key_morph_idx) for c in chord]

    # Prepare the boundaries
    durations_to_consider = get_boundaries(bt, bar_number)
    results = np.zeros((len(durations_to_consider)), dtype=np.float32)
    for i, (start, mid, end, multiplier) in enumerate(durations_to_consider):
        sliced_ct = ct.slice_seconds(start, end)
        results[i] = calculate_chord_distance(end - start, mid - start, sliced_ct, multiplier, chord[cadence_idx1], chord[cadence_idx2])

    cadence_score = (results[i]).item()
    return cadence_score

def create_cadence_analysis_result(ct: ChordAnalysisResult, bt: BeatAnalysisResult, kt: KeyAnalysisResult, bar_number: int) -> CadenceAnalysisResult:
    """Create a list of cadence analysis results from the cadences array"""
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

    cadences_list: list[CadenceAnalysisResult] = []
    for key in get_keys():
        for cadence in cadences_to_consider:
            score = inquire_cadence_score(ct, bt, kt, bar_number, key, cadence)
            cadences_list.append(CadenceAnalysisResult(key, cadence, score))
    return min(cadences_list, key=lambda x: x.score)

def analyse_cadence(audio: Audio,
                    bar_number: int,
                    ct: ChordAnalysisResult,
                    bt: BeatAnalysisResult,
                    hop: int = 512,
                    key_deduction_window: float = 10.) -> CadenceAnalysisResult:
    """Analyse the cadence of an audio file. Bar number must be at least 2 and less than the number of bars in the song."""
    if bar_number < 2 or bar_number >= bt.nbars:
        raise ValueError(f"Invalid bar number (2 <= bar_number < {bt.nbars})")

    slice_lower = max(0, bt.downbeats[bar_number] - key_deduction_window)
    slice_upper = slice_lower + key_deduction_window
    sliced = audio.slice_seconds(slice_lower, slice_upper)

    key = analyse_key_center_chroma(sliced, chroma_chord(sliced, ct.slice_seconds(slice_lower, slice_upper), hop=hop))
    return create_cadence_analysis_result(ct, bt, key, bar_number)
