# Packs the entire application into a single function which can be called by the main script and packaged into a WSGI application.
# Exports the main function mashup_song which is used to mashup the given audio with the dataset.
import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Callable
from ..audio import Audio, DemucsCollection
from ..audio.analysis import ChordAnalysisResult, BeatAnalysisResult, analyse_beat_transformer, analyse_chord_transformer, analyse_beat_transformer
from ..audio.dataset import SongDataset, DatasetEntry, SongGenre
from ..audio.mix import create_mashup, MashabilityResult, calculate_mashability, MashupMode
from ..audio.cache import CacheHandler, MemoryCache
from ..audio.separation import DemucsAudioSeparator
from ..util import YouTubeURL, get_url
from numpy.typing import NDArray
import librosa
import base64
import copy

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

        max_distance: The maximum song distance allowed for the queried song. Around 3-5 will give good results. Default is infinity.

        filter_first: Whether to include only the best result of each song from the search. This does not affect the runtime of the search since the filtering is done after the search. Default is True.

        search_radius: The radius (in seconds) to search for the starting point before the first downbeat and the last downbeat. Default is 3.

        keep_first_k_results: The number of results to keep. Set to -1 to keep all results. Default is 10.

        filter_uneven_bars: Whether to filter out songs with uneven bars. Default is True.

        filter_uneven_bars_min_threshold: The minimum threshold for the mean difference between downbeats. Default is 0.9.

        filter_uneven_bars_max_threshold: The maximum threshold for the mean difference between downbeats. Default is 1.1.

        filter_short_song_bar_threshold: The minimum number of bars for a song to be considered long enough. Default is 12.

        filter_uncached: Whether to filter out songs that are not cached. Default is False.
        """
    starting_point: float
    min_transpose: int = -3
    max_transpose: int = 3
    min_music_percentage: float = 0
    max_delta_bpm: float = 1.25
    min_delta_bpm: float = 0.8
    nbars: int = 8
    max_distance: float = float("inf")
    filter_first: bool = True
    search_radius: float = 3
    keep_first_k_results: int = 10

    filter_uneven_bars: bool = True
    filter_uneven_bars_min_threshold: float = 0.9
    filter_uneven_bars_max_threshold: float = 1.1
    filter_short_song_bar_threshold: int = 12
    filter_uncached: bool = False

    mashup_mode: MashupMode = MashupMode.NATURAL
    natural_drum_activity_threshold: float = 1
    natural_drum_proportion_threshold: float = 0.8
    natural_vocal_activity_threshold: float = 1
    natural_vocal_proportion_threshold: float = 0.8
    natural_window_size: int = 10
    left_pan: float = 0.15
    save_original: bool = False # If true, save the original audio in "./.cache"

    # The path of stuff should not be exposed to the user
    _dataset_path: str = "resources/dataset/audio-infos-v3.0.fast.db"
    _beat_model_path: str = "resources/ckpts/beat_transformer.pt"
    _chord_model_path: str = "resources/ckpts/btc_model_large_voca.pt"

    _verbose: bool = False
    _skip_mashup: bool = False
    _max_show_results: int = 5 # Maximum number of results to show in the demo message box

@dataclass(frozen=True)
class MashupID:
    song_a: YouTubeURL
    song_a_start_time: float
    song_b: YouTubeURL
    song_b_start_bar: int
    transpose: int

    def to_string(self) -> str:
        song_a_start_time_str = str(int(self.song_a_start_time * 1000)).rjust(6, "0")
        song_b_start_bar_str = str(self.song_b_start_bar).rjust(3, "0")
        transpose_str = str(self.transpose + 6).rjust(2, "0")
        id = f"{self.song_a.video_id}{song_a_start_time_str}{self.song_b.video_id}{song_b_start_bar_str}{transpose_str}"
        assert len(id) == 33
        return id

    @classmethod
    def from_string(cls, st: str):
        assert len(st) == 33
        song_a = YouTubeURL(get_url(st[:11]))
        song_a_start_time = int(st[11:17]) / 1000
        song_b = YouTubeURL(get_url(st[17:28]))
        song_b_start_bar = int(st[28:31])
        transpose = int(st[31:33]) - 6
        return cls(song_a, song_a_start_time, song_b, song_b_start_bar, transpose)

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
    # TODO write better code when I am awake and well
    start = downbeats[0]
    while start > t:
        start -= downbeat_diff
    while start < 0:
        start += downbeat_diff
    if abs(start - t) > abs(start + downbeat_diff - t):
        start += downbeat_diff
    new_downbeats = np.arange(nbars) * downbeat_diff + start
    return new_downbeats.astype(np.float32), downbeat_diff * nbars

def validate_config(config: MashupConfig):
    if config.starting_point < 0:
        raise InvalidMashup("Starting point must be >= 0.")

    if config.min_transpose > config.max_transpose:
        raise InvalidMashup("Minimum transpose must be <= maximum transpose.")

    if config.min_music_percentage < 0 or config.min_music_percentage > 1:
        raise InvalidMashup("Minimum music percentage must be between 0 and 1.")

    if config.max_delta_bpm < 0:
        raise InvalidMashup("Maximum delta bpm must be >= 0.")

    if config.min_delta_bpm < 0:
        raise InvalidMashup("Minimum delta bpm must be >= 0.")

    if config.max_delta_bpm < config.min_delta_bpm:
        raise InvalidMashup("Maximum delta bpm must be >= minimum delta bpm.")

    if config.nbars <= 0:
        raise InvalidMashup("Number of bars must be > 0.")

    if config.max_distance < 0:
        raise InvalidMashup("Maximum score must be >= 0.")

    if config.search_radius < 0:
        raise InvalidMashup("Search radius must be >= 0.")

    if config.keep_first_k_results < -1:
        raise InvalidMashup("Keep first k results must be >= -1.")

    if config.filter_uneven_bars_min_threshold < 0:
        raise InvalidMashup("Filter uneven bars min threshold must be >= 0.")

    if config.filter_uneven_bars_max_threshold < 0:
        raise InvalidMashup("Filter uneven bars max threshold must be >= 0.")

    if config.filter_uneven_bars_min_threshold > config.filter_uneven_bars_max_threshold:
        raise InvalidMashup("Filter uneven bars min threshold must be <= max threshold.")

    if config.filter_short_song_bar_threshold <= 0:
        raise InvalidMashup("Filter short song bar threshold must be > 0.")

    if config.natural_drum_activity_threshold < 0:
        raise InvalidMashup("Natural drum activity threshold must be >= 0.")

    if config.natural_drum_proportion_threshold < 0:
        raise InvalidMashup("Natural drum proportion threshold must be >= 0.")

    if config.natural_vocal_activity_threshold < 0:
        raise InvalidMashup("Natural vocal activity threshold must be >= 0.")

    if config.natural_vocal_proportion_threshold < 0:
        raise InvalidMashup("Natural vocal proportion threshold must be >= 0.")

    if config.natural_window_size <= 0:
        raise InvalidMashup("Natural window size must be > 0.")

def load_dataset(config: MashupConfig) -> SongDataset:
    """Load the dataset and apply the filters speciied by config."""
    try:
        dataset = SongDataset.load(config._dataset_path)
    except Exception as e:
        if ".fast.db" in config._dataset_path:
            new_path = config._dataset_path.replace(".fast.db", ".db")
            dataset = SongDataset.load(new_path)
        else:
            raise ValueError("Failed to load dataset") from e
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
            print(new_downbeats, new_duration)
            return bt, new_downbeats[0], new_downbeats[0] + new_duration

        # Unable to extrapolate the result. Fall through and handle later
        pass

    # There are very few cases where theres still a sufficient amount of audio to slice after the last downbeat
    # If the starting point is after the last downbeat, just pretend it is not a valid starting point
    # In these cases, we were not able to find a valid starting point
    # Our current last resort is to pretend the starting point is the first downbeat
    # This is not ideal but it is the best we can do for now
    # TODO in the future if there are ways to detect music phrase boundaries reliably, we can use that to determine the starting point
    raise InvalidMashup("Unable to find a valid starting point.")

# Search stage
def perform_search(config: MashupConfig, chord_result: ChordAnalysisResult, submitted_beat_result: BeatAnalysisResult, slice_start: float, slice_end: float, dataset: SongDataset):
    """Perform the search now that everything is properly sliced. Returns the scores."""
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
        max_distance=config.max_distance,
        keep_first_k=config.keep_first_k_results,
        filter_top_scores=config.filter_first,
        should_curve_score=True,
        verbose=False,
    )
    return scores

def create_mash(cache_handler_factory: Callable[[YouTubeURL], CacheHandler], dataset: SongDataset,
                submitted_audio_a: Audio,
                submitted_parts_a: DemucsCollection,
                submitted_beats_a: BeatAnalysisResult,
                best_result_url: YouTubeURL,
                best_result_start_bar: int,
                best_result_transpose: int,
                nbars: int = 8,
                natural_drum_activity_threshold: float = 1,
                natural_drum_proportion_threshold: float = 0.8,
                natural_vocal_activity_threshold: float = 1,
                natural_vocal_proportion_threshold: float = 0.8,
                natural_window_size: int = 10,
                mashup_mode: MashupMode = MashupMode.NATURAL,
                left_pan: float = 0.15,
                save_original: bool = False,
                mashup_id: MashupID | None = None,
                verbose: bool = False):
    """Creates the mashup from the given parameters. Returns the mashup and the original audio for track B."""
    write = print if verbose else lambda x: None

    write(f"Loading audio for song B from {best_result_url}...")
    b_cache_handler = cache_handler_factory(best_result_url)
    b_audio = b_cache_handler.get_audio()
    write(f"Got audio with duration {b_audio.duration} seconds.")

    write("Analyzing beats for song B...")
    b_beat = BeatAnalysisResult.from_data_entry(dataset[best_result_url])
    slice_start_b, slice_end_b = b_beat.downbeats[best_result_start_bar], b_beat.downbeats[best_result_start_bar + nbars]
    submitted_audio_b = b_audio.slice_seconds(slice_start_b, slice_end_b)

    write("Analyzing parts for song B...")
    submitted_parts_b = b_cache_handler.get_parts_result().slice_seconds(slice_start_b, slice_end_b)

    write("Creating mashup...")
    mashup, mode_used = create_mashup(
        submitted_audio_a=submitted_audio_a,
        submitted_audio_b=submitted_audio_b,
        submitted_bt_a=submitted_beats_a,
        submitted_parts_a=submitted_parts_a,
        submitted_bt_b=b_beat.slice_seconds(slice_start_b, slice_end_b),
        submitted_parts_b=submitted_parts_b,
        transpose=best_result_transpose,
        mode=mashup_mode,
        volume_hop=512,
        natural_drum_activity_threshold=natural_drum_activity_threshold,
        natural_drum_proportion_threshold=natural_drum_proportion_threshold,
        natural_vocal_activity_threshold=natural_vocal_activity_threshold,
        natural_vocal_proportion_threshold=natural_vocal_proportion_threshold,
        natural_window_size=natural_window_size,
        left_pan=left_pan,
        verbose=verbose,
    )
    write(f"Got mashup with mode {mode_used}.")

    if save_original:
        mashup_id_str = mashup_id.to_string() if mashup_id is not None else "mashup"
        submitted_audio_a.save(f".cache/{mashup_id_str}_a.mp3")
        submitted_audio_b.save(f".cache/{mashup_id_str}_b.mp3")
        mashup.save(f".cache/{mashup_id_str}_mashup.mp3")
    return mashup

def save_mashup(mashup_id: MashupID, config: MashupConfig, a_audio: Audio, b_audio: Audio, mashup: Audio, dataset: SongDataset, slice_start_a: float, slice_end_a: float):
    a_audio.slice_seconds(slice_start_a, slice_end_a).save(f".cache/{mashup_id.to_string()}_a.mp3")
    b_beat = BeatAnalysisResult.from_data_entry(dataset[mashup_id.song_b])
    slice_start_b, slice_end_b = b_beat.downbeats[mashup_id.song_b_start_bar], b_beat.downbeats[mashup_id.song_b_start_bar + config.nbars]
    b_audio.slice_seconds(slice_start_b, slice_end_b).save(f".cache/{mashup_id.to_string()}_b.mp3")
    mashup.save(f".cache/{mashup_id.to_string()}_mashup.mp3")
    print(mashup.nchannels)

def mashup_from_id(mashup_id: MashupID | str,
                   config: MashupConfig | None = None,
                   cache_handler_factory: Callable[[YouTubeURL], CacheHandler] | None = None) -> Audio:
    if isinstance(mashup_id, str):
        mashup_id = MashupID.from_string(mashup_id)

    if config is not None:
        # Copy all aspects of the config, except for the starting point
        # where we set it to song a's starting point
        # Use a little hack just don't tell anybody else is ok :)
        config = copy.deepcopy(config)
        object.__setattr__(config, "starting_point", mashup_id.song_a_start_time)
    else:
        config = MashupConfig(starting_point=mashup_id.song_a_start_time)

    write = lambda x: print(x, flush=True) if config._verbose else lambda x: None
    write(f"Creating mashup for ID: {mashup_id.to_string()}")

    dataset = load_dataset(config)
    write(f"Loaded dataset with {len(dataset)} songs.")

    cache_handler_factory = cache_handler_factory or (lambda x: MemoryCache(x))
    a_cache_handler = cache_handler_factory(mashup_id.song_a)

    write(f"Loading audio for song A from {mashup_id.song_a}...")
    a_audio = a_cache_handler.get_audio()
    write(f"Got audio with duration {a_audio.duration} seconds.")

    write("Analyzing beats for song A...")
    beat_result = a_cache_handler.get_beat_analysis_result(model_path = config._beat_model_path)
    if beat_result.nbars < config.nbars:
        raise InvalidMashup("The audio is too short to mashup with the dataset.")
    write(f"Got beat analysis result with {beat_result.nbars} bars.")

    write("Determining slice results...")
    submitted_beats_a, slice_start_a, slice_end_a = determine_slice_results(a_audio, beat_result, config)
    assert slice_start_a >= 0, "Starting point must be >= 0"
    write(f"Got slice results with start {slice_start_a} and end {slice_end_a}.")

    submitted_audio_a = a_audio.slice_seconds(slice_start_a, slice_end_a)

    # Create the mashup
    mashup = create_mash(
        cache_handler_factory, dataset,
        submitted_audio_a=submitted_audio_a,
        submitted_parts_a=a_cache_handler.get_parts_result().slice_seconds(slice_start_a, slice_end_a),
        submitted_beats_a=submitted_beats_a,
        best_result_url=mashup_id.song_b,
        best_result_start_bar=mashup_id.song_b_start_bar,
        best_result_transpose=mashup_id.transpose,
        nbars=config.nbars,
        natural_drum_activity_threshold=config.natural_drum_activity_threshold,
        natural_drum_proportion_threshold=config.natural_drum_proportion_threshold,
        natural_vocal_activity_threshold=config.natural_vocal_activity_threshold,
        natural_vocal_proportion_threshold=config.natural_vocal_proportion_threshold,
        natural_window_size=config.natural_window_size,
        mashup_mode=config.mashup_mode,
        left_pan=config.left_pan,
        save_original=config.save_original,
        mashup_id=mashup_id,
        verbose=config._verbose
    )
    return mashup

def mashup_song(link: YouTubeURL,
                config: MashupConfig,
                cache_handler_factory: Callable[[YouTubeURL], CacheHandler] | None = None,
                dataset: SongDataset | None = None):
    """
    Mashup the given audio with the dataset.
    """
    validate_config(config)
    write = lambda x: print(x, flush=True) if config._verbose else lambda x: None
    write(f"Creating mashup for {link}")

    dataset = load_dataset(config)

    if link in dataset._data:
        dataset._data.pop(link)

    if cache_handler_factory is None:
        cache_handler_factory = lambda x: MemoryCache(x)

    write(f"Loaded dataset with {len(dataset)} songs.")

    a_cache_handler = cache_handler_factory(link)

    write(f"Loading audio for song A from {link}...")
    a_audio = a_cache_handler.get_audio()
    write(f"Got audio with duration {a_audio.duration} seconds.")

    write("Analyzing beats for song A...")
    beat_result = a_cache_handler.get_beat_analysis_result(model_path = config._beat_model_path)
    if beat_result.nbars < config.nbars:
        raise InvalidMashup("The audio is too short to mashup with the dataset.")
    write(f"Got beat analysis result with {beat_result.nbars} bars.")

    write("Analyzing chords for song A...")
    a_chord = a_cache_handler.get_chord_analysis_result(model_path = config._chord_model_path)
    write(f"Got chord analysis result.")

    write("Determining slice results...")
    a_beat, slice_start_a, slice_end_a = determine_slice_results(a_audio, beat_result, config)
    assert slice_start_a >= 0, "Starting point must be >= 0"
    write(f"Got slice results with start {slice_start_a} and end {slice_end_a}.")

    # Create the mashup
    write("Performing search...")
    write(str(config))
    scores = perform_search(config, a_chord, a_beat, slice_start_a, slice_end_a, dataset)
    if len(scores) == 0:
        raise InvalidMashup("No suitable songs found in the dataset.")
    write(f"Got {len(scores)} results.")

    if config._skip_mashup:
        return a_audio.slice_seconds(slice_start_a, slice_end_a), scores, "Skipped mashup."

    best_result_idx = 0
    best_result = scores[best_result_idx][1]
    system_messages: list[str] = ["Here are some other mashups you can consider:"]

    # Create some system messages
    for i, (score, result) in enumerate(scores):
        mashup_id = MashupID(
            song_a=link,
            song_a_start_time=slice_start_a,
            song_b=result.url,
            song_b_start_bar=result.start_bar,
            transpose=result.transpose,
        )
        write(f"Result {i + 1}: {result.url} with score {score}. ID: {mashup_id.to_string()}")
        if i == best_result_idx and i < config._max_show_results:
            system_messages = [f"Created mashup with {best_result.title} ({best_result.url}) with score {scores[0][0]} (ID: {mashup_id.to_string()})"] + system_messages
        elif i < config._max_show_results:
            system_messages.append(f"> {result.title} {result.url} with score {score}. ID: {mashup_id.to_string()}")

    # Create mashup
    mashup_id = MashupID(
        song_a=link,
        song_a_start_time=config.starting_point,
        song_b=best_result.url,
        song_b_start_bar=best_result.start_bar,
        transpose=best_result.transpose,
    )

    mashup = create_mash(
        cache_handler_factory, dataset,
        submitted_audio_a=a_audio.slice_seconds(slice_start_a, slice_end_a),
        submitted_parts_a=a_cache_handler.get_parts_result().slice_seconds(slice_start_a, slice_end_a),
        submitted_beats_a=a_beat,
        best_result_url= best_result.url,
        best_result_start_bar=best_result.start_bar,
        best_result_transpose=best_result.transpose,
        nbars=config.nbars,
        natural_drum_activity_threshold=config.natural_drum_activity_threshold,
        natural_drum_proportion_threshold=config.natural_drum_proportion_threshold,
        natural_vocal_activity_threshold=config.natural_vocal_activity_threshold,
        natural_vocal_proportion_threshold=config.natural_vocal_proportion_threshold,
        natural_window_size=config.natural_window_size,
        mashup_mode=config.mashup_mode,
        left_pan=config.left_pan,
        save_original=config.save_original,
        mashup_id=mashup_id,
        verbose=config._verbose
    )

    system_messages = ["Mashup completed!"] + system_messages
    return mashup, scores, "\n".join(system_messages)
