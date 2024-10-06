# Exports the create entry method
import numpy as np
from .base import SongGenre, DatasetEntry
from ...util.note import get_inv_voca_map
from ...util import YouTubeURL
from ..analysis import ChordAnalysisResult, BeatAnalysisResult, analyse_beat_transformer, analyse_chord_transformer
from ..separation import DemucsAudioSeparator
from ... import Audio

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
    return ChordAnalysisResult(len(br.downbeats), cr.labels, np.array(new_chord_times, dtype=np.float64))

# Create a dataset entry from the given data
def create_entry(length: float, beats: list[float], downbeats: list[float], chords: list[int], chord_times: list[float],
                    *, genre: SongGenre, audio_name: str, url: YouTubeURL, playlist: str | None, views: int):
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
        audio_name=audio_name,
        url=url,
        playlist=playlist,
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

def process_audio_(audio: Audio, video_url: YouTubeURL, genre: SongGenre, *, verbose: bool = True, mean_vocal_threshold: float = 0.1) -> DatasetEntry | str:
    if verbose:
        print(f"Audio length: {audio.duration} ({YouTubeURL(video_url).length})")
    length = audio.duration

    if verbose:
        print(f"Analysing chords...")
    chord_result = analyse_chord_transformer(audio, model_path="./resources/ckpts/btc_model_large_voca.pt", use_loaded_model=True)

    cr = chord_result.group()
    labels = cr.labels
    chord_times = cr.times
    # Check if the length of the labels and chord times are the same
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

    if verbose:
        print("Separating audio...")
    parts = get_demucs().separate(audio)

    # New in v3: Check if there are enough vocals
    mean_vocal_volume = parts.vocals.stft()._data.abs().mean()
    if mean_vocal_volume < mean_vocal_threshold:
        return f"Too few vocals: {video_url} ({mean_vocal_volume})"

    if verbose:
        print(f"Analysing beats...")
    beat_result = analyse_beat_transformer(parts=parts, model_path="./resources/ckpts/beat_transformer.pt", use_loaded_model=True)

    # New in v3: Reject if there are too few downbeats
    if len(beat_result.downbeats) < 12:
        return f"Too few downbeats: {video_url} ({len(beat_result.downbeats)})"

    # New in v3: Reject songs with weird meters
    beat_align_idx = np.abs(beat_result.beats[:, None] - beat_result.downbeats[None, :]).argmin(axis = 0)
    nbeat_in_bar = beat_align_idx[1:] - beat_align_idx[:-1]
    if not np.all(nbeat_in_bar == nbeat_in_bar[0]) and nbeat_in_bar[0] in (3, 4):
        return f"Weird meter: {video_url} ({nbeat_in_bar})"

    # New in v3: Reject songs with a bad alignment
    beat_alignment = np.abs(beat_result.beats[:, None] - beat_result.downbeats[None, :]).min(axis = 0)
    if np.max(beat_alignment) > 0.1:
        return f"Bad alignment: {video_url} ({np.max(beat_alignment)})"

    if verbose:
        print("Postprocessing...")
    beats: list[float] = beat_result.beats.tolist()
    downbeats: list[float] = beat_result.downbeats.tolist()

    if not beats or beats[-1] > length:
        return f"Beats error: {video_url}"

    if not downbeats or downbeats[-1] > length:
        return f"Downbeats error: {video_url}"

    if verbose:
        print("Creating entry...")

    try:
        views = YouTubeURL(video_url).get_views()
    except Exception as e:
        views = 42069                               # Deprecated in v3: No views - so subsitute with a filler if not fetched

    return create_entry(
        length = audio.duration,
        beats = beats,
        downbeats = downbeats,
        chords = labels.tolist(),
        chord_times = chord_times.tolist(),
        genre = genre,
        audio_name = YouTubeURL(video_url).title,
        url = video_url,
        playlist = None,                            # Deprecated in v3: No playlist
        views = views
    )
