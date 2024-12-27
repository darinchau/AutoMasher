# Exports the create entry method
import numpy as np
from typing import Callable
from .base import SongGenre, DatasetEntry
from ...util.note import get_inv_voca_map
from ...util import YouTubeURL, get_url
from ..analysis import ChordAnalysisResult, BeatAnalysisResult, analyse_beat_transformer, analyse_chord_transformer
from ..separation import DemucsAudioSeparator
from ...audio.base import Audio
from ...audio.base.audio_collection import DemucsCollection

# This is now not needed during the search step because we have precalculated it
def get_music_duration(chord_result: ChordAnalysisResult):
    """Get the duration of actual music in the chord result. This is calculated by summing the duration of all chords that are not "No chord"."""
    music_duration = 0.
    times = chord_result.group().times + [chord_result.duration]
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
    cr = cr.group()
    downbeats = br.downbeats.tolist() + [br.duration]
    new_chord_times = []
    curr_downbeat, curr_downbeat_idx, next_downbeat = 0, 0, downbeats[1]
    for chord_times in cr.times:
        while chord_times > next_downbeat:
            curr_downbeat_idx += 1
            curr_downbeat = next_downbeat
            next_downbeat = downbeats[curr_downbeat_idx + 1]
        normalized_time = curr_downbeat_idx + (chord_times - curr_downbeat) / (next_downbeat - curr_downbeat)
        new_chord_times.append(normalized_time)
    return ChordAnalysisResult(len(br.downbeats), cr.features, np.array(new_chord_times, dtype=np.float64))

# Create a dataset entry from the given data
def create_entry(length: float, beats: list[float], downbeats: list[float], chords: list[int], chord_times: list[float],
                    *, genre: SongGenre, url: YouTubeURL, views: int):
    """Creates the dataset entry from the data - performs normalization and music duration postprocessing"""
    chord_result = ChordAnalysisResult.from_data(length, chords, chord_times).group()
    beat_result = BeatAnalysisResult.from_data(length, beats, downbeats)
    normalized_cr = get_normalized_chord_result(chord_result, beat_result)

    # For each bar, calculate its music duration
    music_duration: list[float] = []
    for i in range(len(downbeats)):
        bar_cr = normalized_cr.slice_seconds(i, i + 1)
        music_duration.append(get_music_duration(bar_cr))

    if chord_result.labels.shape[0] != len(chord_times):
        chords = chord_result.labels.tolist()
        chord_times = chord_result.times.tolist()

    return DatasetEntry(
        chords=chords,
        chord_times=chord_times,
        downbeats=downbeats,
        beats=beats,
        genre=genre,
        url=url,
        views=views,
        length=length,
        normalized_chord_times=normalized_cr.times.tolist(),
        music_duration=music_duration
    )

_DEMUCS = None
def get_demucs():
    global _DEMUCS
    if not _DEMUCS:
        _DEMUCS = DemucsAudioSeparator()
    return _DEMUCS

# Returns None if the chord result is valid, otherwise returns an error message
def verify_chord_result(cr: ChordAnalysisResult, length: float, video_url: YouTubeURL | None = None) -> str | None:
    labels = cr.labels
    chord_times = cr.times

    if len(labels) != len(chord_times):
        return f"Length mismatch: labels({len(labels)}) != chord_times({len(chord_times)})"

    # Check if the chord times are sorted monotonically
    if len(chord_times) == 0 or chord_times[-1] > length:
        return f"Chord times error: {video_url}"

    # Check if the chord times are sorted monotonically
    if not all([t1 < t2 for t1, t2 in zip(chord_times, chord_times[1:])]):
        return f"Chord times not sorted monotonically: {video_url}"

    # New in v3: Check if there are enough chords
    no_chord_duration = 0.
    for t1, t2, c in zip(chord_times, chord_times[1:], labels):
        if c == get_inv_voca_map()["No chord"]:
            no_chord_duration += t2 - t1
    if no_chord_duration > 0.5 * length:
        return f"Too much no chord: {video_url} ({no_chord_duration}) (Proportion: {no_chord_duration / length})"

    return None

def verify_parts_result(parts: DemucsCollection, mean_vocal_threshold: float, video_url: YouTubeURL | None = None) -> str | None:
    # New in v3: Check if there are enough vocals
    mean_vocal_volume = parts.vocals.volume
    if mean_vocal_volume < mean_vocal_threshold:
        return f"Too few vocals: {video_url} ({mean_vocal_volume})"
    return None

def verify_beats_result(br: BeatAnalysisResult, length: float, video_url: YouTubeURL | None = None, reject_weird_meter: bool = True, bad_alignment_threshold: float = 0.1) -> str | None:
    """Verify the beat result. If strict is True, then it will reject songs with weird meters."""
    # New in v3: Reject if there are too few downbeats
    if len(br.downbeats) < 12:
        return f"Too few downbeats: {video_url} ({len(br.downbeats)})"

    # New in v3: Reject songs with weird meters
    # Remove all songs with 3/4 meter as well because 96% of the songs are 4/4
    # This is rejecting way too many songs. Making this optional
    beat_align_idx = np.abs(br.beats[:, None] - br.downbeats[None, :]).argmin(axis = 0)
    nbeat_in_bar = beat_align_idx[1:] - beat_align_idx[:-1]
    if reject_weird_meter and not np.all(nbeat_in_bar == 4):
        return f"Weird meter: {video_url} ({nbeat_in_bar})"

    # New in v3: Reject songs with a bad alignment
    beat_alignment = np.abs(br.beats[:, None] - br.downbeats[None, :]).min(axis = 0)
    if np.max(beat_alignment) > bad_alignment_threshold:
        return f"Bad alignment: {video_url} ({np.max(beat_alignment)})"

    # Check if beats and downbeats make sense
    if len(br.beats) == 0 or br.beats[-1] >= length:
        return f"Beats error: {video_url}"

    if len(br.downbeats) == 0 or br.downbeats[-1] >= length:
        return f"Downbeats error: {video_url}"

    if not all([b1 < b2 for b1, b2 in zip(br.beats, br.beats[1:])]):
        return f"Beats not sorted monotonically: {video_url}"

    if not all([d1 < d2 for d1, d2 in zip(br.downbeats, br.downbeats[1:])]):
        return f"Downbeats not sorted monotonically: {video_url}"

    return None

def process_audio(audio: Audio,
                   video_url: YouTubeURL,
                   genre: SongGenre, *,
                   verbose: bool = True,
                   reject_weird_meter: bool = False,
                   mean_vocal_threshold: float = 0.1,
                   chord_model_path: str = "./resources/ckpts/btc_model_large_voca.pt",
                   beat_model_path: str = "./resources/ckpts/beat_transformer.pt",
                   additional_parts_verification: Callable[[DemucsCollection], str | None] = lambda _: None,
                   additional_beats_verification: Callable[[BeatAnalysisResult], str | None] = lambda _: None,
                   additional_chords_verification: Callable[[ChordAnalysisResult], str | None] = lambda _: None) -> tuple[DatasetEntry, DemucsCollection] | str:
    if verbose:
        print(f"Audio length: {audio.duration} ({get_url(video_url).get_length()})")
    length = audio.duration

    if verbose:
        print(f"Analysing chords...")
    chord_result = analyse_chord_transformer(audio, model_path=chord_model_path, use_loaded_model=True)

    cr = chord_result.group()
    error = verify_chord_result(cr, length, video_url)
    if error is not None:
        return error

    if additional_chords_verification is not None:
        error = additional_chords_verification(cr)
        if error is not None:
            return error

    if verbose:
        print("Separating audio...")
    parts = get_demucs().separate(audio)
    error = verify_parts_result(parts, mean_vocal_threshold, video_url)
    if error is not None:
        return error

    if additional_parts_verification is not None:
        error = additional_parts_verification(parts)
        if error is not None:
            return error

    if verbose:
        print(f"Analysing beats...")
    beat_result = analyse_beat_transformer(parts=parts, model_path=beat_model_path, use_loaded_model=True)
    error = verify_beats_result(beat_result, length, video_url, reject_weird_meter=reject_weird_meter)
    if error is not None:
        return error

    if additional_beats_verification is not None:
        error = additional_beats_verification(beat_result)
        if error is not None:
            return error

    if verbose:
        print("Postprocessing...")

    beats: list[float] = beat_result.beats.tolist()
    downbeats: list[float] = beat_result.downbeats.tolist()
    labels = cr.labels
    chord_times = cr.times

    if verbose:
        print("Creating entry...")

    try:
        views = get_url(video_url).get_views()
        if views is None:
            views = 42069
    except Exception as e:
        views = 42069 # Deprecated in v3: No views - so subsitute with a filler if not fetched

    return create_entry(
        length = audio.duration,
        beats = beats,
        downbeats = downbeats,
        chords = labels.tolist(),
        chord_times = chord_times.tolist(),
        genre = genre,
        url = video_url,
        views = views
    ), parts
