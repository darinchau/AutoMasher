# Packs the entire application into a single function which can be called by the main script and packaged into a WSGI application.
# Exports the main function mashup_song which is used to mashup the given audio with the dataset.
import os
import numpy as np
from dataclasses import dataclass
from typing import Callable
from ..audio import Audio, AudioCollection
from ..audio.analysis import ChordAnalysisResult, BeatAnalysisResult, analyse_beat_transformer, analyse_chord_transformer, analyse_beat_transformer
from ..audio.base import AudioCollection
from ..audio.dataset import SongDataset, DatasetEntry, SongGenre
from ..audio.search import create_mashup, MashabilityResult, calculate_mashability, MashupMode
from ..audio.dataset.cache import CacheHandler
from ..audio.separation import DemucsAudioSeparator
from ..util import YouTubeURL
from numpy.typing import NDArray
import librosa

class InvalidMashup(Exception):
    pass

@dataclass(frozen=True)
class MashupConfig:
    """
    The configuration for the mashup.

    Attributes:
        starting_point: The starting point of the mashup. Must be provided as a float which indicates the starting time in seconds.

        dataset_path: The path to the dataset to mashup with.

        min_transpose: The minimum number of semitones to transpose up the audio.

        max_transpose: The maximum number of semitones to transpose up the audio.

        min_music_percentage: The minimum percentage of music in the audio. Default is 0.8.

        max_delta_bpm: The maximum bpm deviation allowed for the queried song. Default is 1.25.

        min_delta_bpm: The minimum bpm deviation allowed for the queried song. Default is 0.8.

        nbars: The number of bars the resulting mashup will contain. Default is 8.

        max_score: The maximum mashability score allowed for the queried song. Default is infinity.

        filter_first: Whether to include only the best result of each song from the search. This does not affect the runtime of the search since the filtering is done after the search. Default is True.

        search_radius: The radius (in seconds) to search for the starting point before the first downbeat and the last downbeat. Default is 3.

        keep_first_k_results: The number of results to keep. Set to -1 to keep all results. Default is 10.

        filter_uneven_bars: Whether to filter out songs with uneven bars. Default is True.

        filter_uneven_bars_min_threshold: The minimum threshold for the mean difference between downbeats. Default is 0.9.

        filter_uneven_bars_max_threshold: The maximum threshold for the mean difference between downbeats. Default is 1.1.

        filter_short_song_bar_threshold: The minimum number of bars for a song to be considered long enough. Default is 12.

        filter_uncached: Whether to filter out songs that are not cached. Default is True.

        """
    starting_point: float
    min_transpose: int = -3
    max_transpose: int = 3
    min_music_percentage: float = 0.8
    max_delta_bpm: float = 1.25
    min_delta_bpm: float = 0.8
    nbars: int = 8
    max_score: float = float("inf")
    filter_first: bool = True
    search_radius: float = 3
    keep_first_k_results: int = 10

    cache: bool = True

    filter_uneven_bars: bool = True
    filter_uneven_bars_min_threshold: float = 0.9
    filter_uneven_bars_max_threshold: float = 1.1
    filter_short_song_bar_threshold: int = 12
    filter_uncached: bool = True

    mashup_mode: MashupMode = MashupMode.NATURAL
    natural_drum_activity_threshold: float = 1
    natural_drum_proportion_threshold: float = 0.8
    natural_vocal_activity_threshold: float = 1
    natural_vocal_proportion_threshold: float = 0.8
    natural_window_size: int = 10

    # The path of stuff should not be exposed to the user
    dataset_path: str = "resources/dataset/audio-infos-v2.1.db"
    beat_model_path: str = "resources/ckpts/beat_transformer.pt"
    chord_model_path: str = "resources/ckpts/btc_model_large_voca.pt"

def is_regular(downbeats: NDArray[np.float32], range_threshold: float = 0.2, std_threshold: float = 0.1) -> bool:
    """Return true if the downbeats are evenly spaced."""
    downbeat_diffs = downbeats[1:] - downbeats[:-1]
    downbeat_diff = np.mean(downbeat_diffs).item()
    downbeat_diff_std = np.std(downbeat_diffs).item()
    downbeat_diff_range = (np.max(downbeat_diffs) - np.min(downbeat_diffs)) / downbeat_diff
    return (downbeat_diff_range < range_threshold).item() and downbeat_diff_std < std_threshold

def extrapolate_downbeat(downbeats: NDArray[np.float32], t: float, nbars: int):
    """Extrapolate the downbeats to the starting point t. Returns the new downbeats and the new duration.

    Starting point is guaranteed >= 0"""
    downbeat_diffs = downbeats[1:] - downbeats[:-1]
    downbeat_diff = np.mean(downbeat_diffs).item()
    start: float = downbeat_diff - round((downbeats[0] - t) / downbeat_diff)
    if start < 0:
        start += downbeat_diff
    new_downbeats = np.arange(nbars) * downbeat_diff + start
    return new_downbeats.astype(np.float32), downbeat_diff * nbars

def load_dataset(config: MashupConfig) -> SongDataset:
    """Load the dataset and apply the filters speciied by config."""
    dataset = SongDataset.load(config.dataset_path)
    filters: list[Callable[[DatasetEntry], bool]] = []

    if config.filter_short_song_bar_threshold > 0:
        filters.append(lambda x: len(x.downbeats) >= config.filter_short_song_bar_threshold)

    if config.filter_uneven_bars:
        def filter_func(x: DatasetEntry):
            db = np.array(x.downbeats)
            db_diff = np.diff(db)
            mean_diff = db_diff / np.mean(db_diff)
            return np.all((config.filter_uneven_bars_min_threshold < mean_diff) & (mean_diff < config.filter_uneven_bars_max_threshold)).item()
        filters.append(filter_func)

    if config.filter_uncached:
        filters.append(lambda x: x.cached)

    dataset = dataset.filter(lambda x: all(f(x) for f in filters))
    return dataset

def determine_slice_results(audio: Audio, bt: BeatAnalysisResult, config: MashupConfig) -> tuple[BeatAnalysisResult, float, float]:
    """
    Determine the slice point for the audio. Returns the sliced chord and beat analysis results.

    returns (Beat Analysis Result of Sliced Result, slice start in seconds, slice end in seconds)
    """
    # If the config starting point is within the range of beats, use the closest downbeat to the starting point
    if bt.downbeats[0] - config.search_radius <= config.starting_point <= bt.downbeats[-1] + config.search_radius:
        index = np.argmin(np.abs(np.array(bt.downbeats) - config.starting_point))
        if index >= bt.nbars - config.nbars:
            index = bt.nbars - config.nbars - 1
        start = bt.downbeats[index]
        end = bt.downbeats[index + config.nbars]
        return bt.slice_seconds(start, end), start, end

    # If the config starting point is not within the range of beats, try to extrapolate using a few heuristics
    if bt.downbeats[0] > config.starting_point:
        new_downbeats = new_duration = None
        if is_regular(bt.downbeats[:config.nbars]):
            new_downbeats, new_duration = extrapolate_downbeat(bt.downbeats[:config.nbars], config.starting_point, config.nbars)

        elif bt.nbars > config.nbars and is_regular(bt.downbeats[1:config.nbars + 1]):
            new_downbeats, new_duration = extrapolate_downbeat(bt.downbeats[1:config.nbars + 1], config.starting_point, config.nbars)

        if new_downbeats is not None and new_duration is not None and new_downbeats[0] + new_duration < bt.duration:
            bt = BeatAnalysisResult(
                duration=new_duration,
                beats = np.empty_like(new_downbeats),
                downbeats = new_downbeats,
            )
            return bt, new_downbeats[0], new_downbeats[0] + new_duration

        # Unable to extrapolate the result. Fall through and handle later
        pass

    # There are very few cases where theres still a sufficient amount of audio to slice after the last downbeat
    # If the starting point is after the last downbeat, just pretend it is not a valid starting point
    # In these cases, we were not able to find a valid starting point
    # Our current last resort is to pretend the starting point is the first downbeat
    # This is not ideal but it is the best we can do for now
    # TODO in the future if there are ways to detect music phrase boundaries reliably, we can use that to determine the starting point
    tempo, _ = librosa.beat.beat_track(y=audio.numpy(), sr=audio.sample_rate)
    downbeat_diff = 60 / tempo * 4
    slice_end = config.starting_point + config.nbars * downbeat_diff
    return BeatAnalysisResult(
        duration=config.nbars * downbeat_diff,
        beats = np.array([], dtype=np.float32),
        downbeats = np.arange(config.nbars) * downbeat_diff + config.starting_point,
    ), config.starting_point, slice_end

# Beat Result Stage
def get_beat_result(audio: Audio, config: MashupConfig, cache_handler: CacheHandler) -> BeatAnalysisResult:
    beat_result = cache_handler.get_beat_analysis() or analyse_beat_transformer(audio, model_path=config.beat_model_path)
    cache_handler.store_beat_analysis(beat_result)
    if beat_result.nbars < config.nbars:
        raise InvalidMashup("The audio is too short to mashup with the dataset.")
    return beat_result

# Demucs Stage
def get_parts_result(audio: Audio) -> AudioCollection:
    demucs = DemucsAudioSeparator()
    parts_result = demucs.separate(audio)
    return parts_result

def get_audio_from_url(url: YouTubeURL, cache_handler: CacheHandler) -> Audio:
    audio = cache_handler.get_audio() or Audio.load(url)
    cache_handler.store_audio(audio)
    return audio

# Search stage
def get_search_result(config: MashupConfig, chord_result: ChordAnalysisResult, submitted_beat_result: BeatAnalysisResult, slice_start: float, slice_end: float, dataset: SongDataset):
    # TODO Do some processing to get submitted_chord_result and submitted_beat_result
    # Need to figure out how to do this with a timestamp instead of a bar number
    submitted_chord_result = chord_result.slice_seconds(slice_start, slice_end)

    # Perform the search
    scores = calculate_mashability(
        submitted_chord_result=submitted_chord_result,
        submitted_beat_result=submitted_beat_result,
        dataset=dataset,
        max_transpose=(config.min_transpose, config.max_transpose),
        min_music_percentage=config.min_music_percentage,
        max_delta_bpm=config.max_delta_bpm,
        min_delta_bpm=config.min_delta_bpm,
        max_score=config.max_score,
        keep_first_k=config.keep_first_k_results,
        filter_top_scores=config.filter_first,
        should_curve_score=True,
        verbose=False,
    )
    return scores

def get_mashup_result(config: MashupConfig, a_beat: BeatAnalysisResult, transpose: int, b_beat: BeatAnalysisResult, a_parts: AudioCollection, b_parts: AudioCollection):
    return create_mbashup(
        submitted_t_a=a_beat,
        submitted_parts_a=a_parts,
        submitted_bt_b=b_beat,
        submitted_parts_b=b_parts,
        transpose=transpose,
        mode=config.mashup_mode,
        volume_hop=512,
        natural_drum_activity_threshold=config.natural_drum_activity_threshold,
        natural_drum_proportion_threshold=config.natural_drum_proportion_threshold,
        natural_vocal_activity_threshold=config.natural_vocal_activity_threshold,
        natural_vocal_proportion_threshold=config.natural_vocal_proportion_threshold,
        natural_window_size=config.natural_window_size,
    )


def mashup_song(config: MashupConfig, cache_handler_factory: Callable[[YouTubeURL], CacheHandler]):
    """
    Mashup the given audio with the dataset.
    """
    dataset = load_dataset(config)
    
    a_cache_handler = cache_handler_factory(a_url)
    a_audio = a_cache_handler.get_audio()

    beat_result = a_cache_handler.get_beat_result(fallback = lambda: analyse_beat_transformer(a_audio, model_path = config.beat_model_path))
    if beat_result.nbars < config.nbars:
        raise InvalidMashup("The audio is too short to mashup with the dataset.")

    chord_result = a_cache_handler.get_chord_result(fallback = lambda: analyse_chord_transformer(a_audio, model_path = config.chord_model_path))

    a_beat, slice_start_a, slice_end_a = determine_slice_results(a_audio, beat_result, config)

    # Create the mashup
    scores = get_search_result(config, chord_result, a_beat, slice_start_a, slice_end_a, dataset)
    if len(scores) == 0:
        raise InvalidMashup("No suitable songs found in the dataset.")
    
    # TODO iterate through all scores if entries raise invalid mashup
    best_result = scores[0][1]
    
    b_cache_handler = cache_handler_factory(best_result.url)
    b_audio = b_cache_handler.get_audio()
    b_beat = BeatAnalysisResult.from_data_entry(dataset[best_result.url])
    slice_start_b, slice_end_b = b_beat.downbeats[best_result.start_bar], b_beat.downbeats[best_result.start_bar + config.nbars]
    b_beat = b_beat.slice_seconds(slice_start_b, slice_end_b)

    parts_result = get_parts_result(audio).slice_seconds(slice_start_a, slice_end_a)
    b_parts = get_parts_result(b_audio).slice_seconds(slice_start_b, slice_end_b)

    mashup = get_mashup_result(config, a_beat, best_result.transpose, b_beat, parts_result, b_parts)
    return mashup
