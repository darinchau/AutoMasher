from fyp import Audio
from fyp.audio.analysis.base import CadenceAnalysisResult, ChordAnalysisResult, BeatAnalysisResult, KeyAnalysisResult
from fyp.audio.analysis.key import analyse_key_center_chroma
from fyp.audio.analysis.beat import analyse_beat_transformer
from fyp.audio.analysis.chord import analyse_chord_transformer
from fyp.util.note import get_chord_notes, get_idx2voca_chord, notes_to_idx, get_inv_voca_map, get_keys, transpose_chord, idx_to_notes
import numpy as np
from numpy.typing import NDArray
from functools import lru_cache
from dataclasses import dataclass

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

@dataclass(frozen=True)
class BoundaryPoints:
    b_2: float
    b_1: float
    b_0: float
    b_plus_1: float
    b_plus_2: float

    def __post_init__(self):
        assert self.b_2 <= self.b_1 < self.b_0 < self.b_plus_1 <= self.b_plus_2, "Invalid boundary points"
        assert self.b_0 >= 0 and self.b_0 < float("inf"), "Invalid boundary points"

    @property
    def b_half(self) -> float:
        return (self.b_1 + self.b_0) / 2 if self.b_1 >= 0 else -1

    @property
    def b_quarter(self) -> float:
        return (self.b_1 + self.b_0 * 3) / 4 if self.b_1 >= 0 else -1

    @property
    def b_plus_half(self) -> float:
        return (self.b_0 + self.b_plus_1) / 2 if self.b_plus_1 < float("inf") else float("inf")

# Durations is a (slice_start, slice_mid, slice_end, multiplier) tuple
# Each value is a time in seconds
# The slice_mid is the chord change point
# The multiplier is a number to multiply to the chord distance
# The longer the duration, the stronger the cadence ought to be.
# And the more likely the match is actually real.
def get_boundaries(song_duration: float, timestamps: BoundaryPoints) -> list[tuple[float, float, float, float]]:
    durations_to_consider: list[tuple[float, float, float, float]] = []

    if 0 <= timestamps.b_2 <= song_duration:
        durations_to_consider.append((timestamps.b_2, timestamps.b_1, timestamps.b_0, 1))
        if 0 <= timestamps.b_plus_2 <= song_duration:
            durations_to_consider.append((timestamps.b_2, timestamps.b_0, timestamps.b_plus_2, 1.1))

    if 0 <= timestamps.b_1 <= song_duration:
        durations_to_consider.append((timestamps.b_1, timestamps.b_half, timestamps.b_0, 0.9))
        # durations_to_consider.append((timestamps.b_half, timestamps.b_quarter, timestamps.b_0, 0.8))
        if 0 <= timestamps.b_plus_1 <= song_duration:
            durations_to_consider.append((timestamps.b_1, timestamps.b_0, timestamps.b_plus_1, 1))
            durations_to_consider.append((timestamps.b_half, timestamps.b_0, timestamps.b_plus_half, 0.95))

    return durations_to_consider

def get_boundaries_from_time_diffs(song_duration: float, t: float, diff: float) -> list[tuple[float, float, float, float]]:
    if t <= 0 or t >= song_duration:
        return []
    boundaries = BoundaryPoints(
        b_2 = t - diff - diff,
        b_1 = t - diff,
        b_0 = t,
        b_plus_1 = t + diff,
        b_plus_2 = t + diff + diff
    )
    return get_boundaries(song_duration, boundaries)

def get_boundaries_from_bt(bt: BeatAnalysisResult, bar_number: int) -> list[tuple[float, float, float, float]]:
    boundaries = BoundaryPoints(
        b_2 = bt.downbeats[bar_number - 2] if bar_number >= 2 else -1,
        b_1 = bt.downbeats[bar_number - 1] if bar_number >= 1 else -1,
        b_0 = bt.downbeats[bar_number],
        b_plus_1 = bt.downbeats[bar_number + 1] if bar_number + 1 < bt.nbars else float("inf"),
        b_plus_2 = bt.downbeats[bar_number + 2] if bar_number + 2 < bt.nbars else float("inf")
    )
    return get_boundaries(bt.duration, boundaries)

# Get the chromagram of the audio based on the chord analysis result
def chroma_chord(ct: ChordAnalysisResult, chromagram_sample_rate: int = 44100, hop: int = 512):
    dims = (12, int(chromagram_sample_rate * ct.duration / hop + 0.5))
    chroma = np.zeros(dims, dtype=np.float32)

    boundaries = ct.times.tolist() + [ct.duration]
    for i in range(len(boundaries) - 1):
        lower = int(boundaries[i] * chromagram_sample_rate / hop + 0.5)
        upper = int(boundaries[i + 1] * chromagram_sample_rate / hop + 0.5)
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

def inquire_cadence_score(ct: ChordAnalysisResult, key: str, inquire_cadence: str, durations_to_consider: list[tuple[float, float, float, float]]) -> float:
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
    if not durations_to_consider:
        return float("inf")

    cadence_idx1 = roman_to_idx[inquire_cadence.split("->")[0].strip()]
    cadence_idx2 = roman_to_idx[inquire_cadence.split("->")[1].strip()]

    # Prepare the chords array
    major_chords, minor_chords = get_major_minor_chords()
    key_morph_idx = notes_to_idx(key.split(" ")[0].strip())
    chord = major_chords if key.strip()[-5:].lower() == "major" else minor_chords
    chord = [transpose_chord_idx(c, key_morph_idx) for c in chord]

    results: list[float] = []
    for start, mid, end, multiplier in durations_to_consider:
        sliced_ct = ct.slice_seconds(start, end)
        results.append(calculate_chord_distance(end - start, mid - start, sliced_ct, multiplier, chord[cadence_idx1], chord[cadence_idx2]))

    # The smaller the distance, the better
    cadence_score = min(results)
    return cadence_score

def create_cadence_analysis_result(ct: ChordAnalysisResult, kt: KeyAnalysisResult, bt: BeatAnalysisResult | None = None, bar_number: int | None = None, t: float | None = None, bar_length: float | None = None) -> CadenceAnalysisResult:
    """Create a list of cadence analysis results from the cadences array, and find the minimum"""
    cadences_to_consider = [
        "V -> I",
        "I -> V",
        "II -> V",
        "IV -> V",
        "IV -> I",
        "V -> VI",
        "V -> IV",
        "I -> IV"
    ]

    assert (bt is not None and bar_number is not None) or (t is not None and bar_length is not None), "Either bt and bar_number or t and bar_length must be provided"

    # Prepare the boundaries
    if bt is not None and bar_number is not None:
        durations_to_consider = get_boundaries_from_bt(bt, bar_number)
    else:
        assert t is not None and bar_length is not None, "Invalid t and bar_length"
        durations_to_consider = get_boundaries_from_time_diffs(ct.duration, t, bar_length)

    # There is a possibility that the cadence is not possible because of insufficient bar number
    # In this case, erh, we just return infinity
    # TODO: unknown cadence
    if not durations_to_consider:
        return CadenceAnalysisResult("C major", "V -> I", float("inf"))

    cadences_list: list[CadenceAnalysisResult] = []
    for key in get_keys():
        if kt.get_correlation(key) < 0:
            continue
        for cadence in cadences_to_consider:
            score = inquire_cadence_score(ct, key, cadence, durations_to_consider)
            if score == float("inf"):
                continue
            cadences_list.append(CadenceAnalysisResult(key, cadence, score))
    return min(cadences_list, key=lambda x: x.score)

def analyse_cadence_beat(ct: ChordAnalysisResult,
                    bt: BeatAnalysisResult,
                    bar_number: int,
                    hop: int = 512,
                    key_deduction_window: float = 10.) -> CadenceAnalysisResult:
    """Analyse the cadence of an audio file. Bar number must be at least 2 and less than the number of bars in the song."""
    if bar_number >= bt.nbars:
        raise ValueError(f"Invalid bar number (bar_number < {bt.nbars})")

    slice_lower = max(0, bt.downbeats[bar_number] - key_deduction_window)
    slice_upper = slice_lower + key_deduction_window
    key = analyse_key_center_chroma(chroma_chord(ct.slice_seconds(slice_lower, slice_upper), hop=hop))
    return create_cadence_analysis_result(ct, key, bt=bt, bar_number=bar_number)

def analyse_cadence_time_diff(ct: ChordAnalysisResult,
                    t: float,
                    bar_length: float,
                    hop: int = 512,
                    key_deduction_window: float = 10.) -> CadenceAnalysisResult:
    """Analyse the cadence of an audio file. Bar number must be at least 2 and less than the number of bars in the song."""
    if t <= 0 or t >= ct.duration:
        raise ValueError(f"Invalid time t (0 <= t < {ct.duration})")

    slice_lower = max(0, t - key_deduction_window)
    slice_upper = slice_lower + key_deduction_window
    key = analyse_key_center_chroma(chroma_chord(ct.slice_seconds(slice_lower, slice_upper), hop=hop))
    return create_cadence_analysis_result(ct, key, t=t, bar_length=bar_length)
