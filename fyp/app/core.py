# Packs the entire application into a single function which can be called by the main script and packaged into a WSGI application.
# Exports the main function mashup_song which is used to mashup the given audio with the dataset.
import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Callable
from ..audio import Audio, DemucsCollection
from ..audio.analysis import (
    BeatAnalysisResult,
    ChordAnalysisResult,
    OnsetFeatures,
    DiscreteLatentFeatures,
    ContinuousLatentFeatures,
)
from ..audio.dataset import SongDataset, DatasetEntry, create_entry
from ..audio.mix import create_mashup, MashabilityResult, calculate_mashability, MashupMode
from ..audio.separation import demucs_separate
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

        mashup_mode: The mode to use for the mashup. Default is MashupMode.NATURAL.

        natural_drum_activity_threshold: The drum activity threshold for the natural mode. Default is 1.

        natural_drum_proportion_threshold: The drum proportion threshold for the natural mode. Default is 0.8.

        natural_vocal_activity_threshold: The vocal activity threshold for the natural mode. Default is 1.

        natural_vocal_proportion_threshold: The vocal proportion threshold for the natural mode. Default is 0.8.

        natural_window_size: The window size for the natural mode. Default is 10.

        left_pan: The left pan for the natural mode. Default is 0.15.

        save_original: Whether to save the original audio in the resources/mashups folder. Default is False.

        append_song_to_dataset: Whether to append the song to the dataset after the search. Default is False.

        load_on_the_fly: Whether to load the entries on the fly instead of loading everything at once. Default is False.

        assert_audio_exists: Whether to assert that the audio exists in the dataset. Default is False.

        use_simplified_chord: Whether to use the simplified chord model and circle-of-fifth distances. Default is True.
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

    mashup_mode: MashupMode = MashupMode.NATURAL
    natural_drum_activity_threshold: float = 1
    natural_drum_proportion_threshold: float = 0.8
    natural_vocal_activity_threshold: float = 1
    natural_vocal_proportion_threshold: float = 0.8
    natural_window_size: int = 10
    left_pan: float = 0.15
    save_original: bool = False  # If true, save the original audio in resources/mashups/<mashup_id>

    append_song_to_dataset: bool = False  # If true, append the song to the dataset after the search
    load_on_the_fly: bool = False  # If true, load the entries on the fly instead of loading everything at once
    assert_audio_exists: bool = False  # If true, assert that the audio exists in the dataset
    use_simplified_chord: bool = True  # If true, use the simplified chord model and circle-of-fifth distances. This can potentially improve the search results.

    # The path of stuff should not be exposed to the user
    dataset_path: str = "resources/dataset"
    max_dataset_size: str = "16GB"
    _beat_model_path: str = "resources/ckpts/beat_transformer.pt"
    _chord_model_path: str = "resources/ckpts/btc_model_large_voca.pt"
    _simple_chord_model_path: str = "resources/ckpts/btc_model.pt"

    _verbose: bool = False
    _skip_mashup: bool = False
    _max_show_results: int = 5  # Maximum number of results to show in the demo message box


_MASHUP_ID_CUSTOM_AUDIO_PREPEND = "custom_aud+"


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
        if self.song_a.is_placeholder:
            id = _MASHUP_ID_CUSTOM_AUDIO_PREPEND
        else:
            id = f"{self.song_a.video_id}"
        id += f"{song_a_start_time_str}{self.song_b.video_id}{song_b_start_bar_str}{transpose_str}"
        assert len(id) == 33
        return id

    @classmethod
    def from_string(cls, st: str):
        assert len(st) == 33, f"Invalid mashup ID length: ({st})"
        if st.startswith(_MASHUP_ID_CUSTOM_AUDIO_PREPEND):
            raise ValueError("Mashup ID is not valid for custom audio.")

        song_a = get_url(st[:11])
        song_a_start_time = int(st[11:17]) / 1000
        song_b = get_url(st[17:28])
        song_b_start_bar = int(st[28:31])
        transpose = int(st[31:33]) - 6
        return cls(song_a, song_a_start_time, song_b, song_b_start_bar, transpose)


def is_regular(downbeats: NDArray[np.float64], range_threshold: float = 0.2, std_threshold: float = 0.1) -> bool:
    """Return true if the downbeats are evenly spaced."""
    downbeat_diffs = downbeats[1:] - downbeats[:-1]
    downbeat_diff = np.mean(downbeat_diffs).item()
    downbeat_diff_std = np.std(downbeat_diffs).item()
    downbeat_diff_range = (np.max(downbeat_diffs) - np.min(downbeat_diffs)) / downbeat_diff
    return (downbeat_diff_range < range_threshold).item() and downbeat_diff_std < std_threshold


def extrapolate_downbeat(downbeats: NDArray[np.float64], t: float, nbars: int):
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
    return new_downbeats.astype(np.float64), downbeat_diff * nbars


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
    dataset = SongDataset(
        config.dataset_path,
        load_on_the_fly=config.load_on_the_fly,
        assert_audio_exists=config.assert_audio_exists,
        max_dir_size=config.max_dataset_size,
    )

    if config.filter_short_song_bar_threshold > 0:
        dataset = dataset.filter(lambda x: len(x.downbeats) >= config.filter_short_song_bar_threshold)

    if config.filter_uneven_bars:
        def filter_func(x: DatasetEntry):
            db_diff = np.diff(x.downbeats.onsets)
            mean_diff = db_diff / np.mean(db_diff)
            return np.all((config.filter_uneven_bars_min_threshold < mean_diff) & (mean_diff < config.filter_uneven_bars_max_threshold)).item()
        dataset = dataset.filter(filter_func)
    return dataset


def determine_slice_results(downbeats: OnsetFeatures, config: MashupConfig) -> tuple[OnsetFeatures, float, float]:
    """
    Determine the slice point for the audio. Returns the sliced chord and beat analysis results.

    returns (sliced downbeats, slice start in seconds, slice end in seconds)
    """
    # If the config starting point is within the range of beats, use the closest downbeat to the starting point
    nbars = len(downbeats)
    if downbeats.onsets[0] - config.search_radius <= config.starting_point <= downbeats.onsets[-1] + config.search_radius:
        index = np.argmin(np.abs(downbeats.onsets - config.starting_point))
        if index >= nbars - config.nbars:
            index = nbars - config.nbars - 1
        start = downbeats.onsets[index]
        end = downbeats.onsets[index + config.nbars]
        return downbeats.slice_seconds(start, end), start, end

    # If the config starting point is not within the range of beats, try to extrapolate using a few heuristics
    if downbeats.onsets[0] > config.starting_point:
        new_downbeats = new_duration = None
        if is_regular(downbeats.onsets[:config.nbars]):
            new_downbeats, new_duration = extrapolate_downbeat(downbeats.onsets[:config.nbars], config.starting_point, config.nbars)

        elif nbars > config.nbars and is_regular(downbeats.onsets[1:config.nbars + 1]):
            new_downbeats, new_duration = extrapolate_downbeat(downbeats.onsets[1:config.nbars + 1], config.starting_point, config.nbars)

        if new_downbeats is not None and new_duration is not None and new_downbeats[0] + new_duration < downbeats.duration:
            downbeats = OnsetFeatures(downbeats.duration, new_downbeats)
            return downbeats, new_downbeats[0], new_downbeats[0] + new_duration

        # Unable to extrapolate the result. Fall through and handle later
        pass

    # There are very few cases where theres still a sufficient amount of audio to slice after the last downbeat
    # If the starting point is after the last downbeat, just pretend it is not a valid starting point
    # In these cases, we were not able to find a valid starting point
    # Our current last resort is to pretend the starting point is the first downbeat
    # This is not ideal but it is the best we can do for now
    # TODO in the future if there are ways to detect music phrase boundaries reliably, we can use that to determine the starting point
    raise InvalidMashup("Unable to find a valid starting point.")


def get_search_result_log(link: YouTubeURL,
                          config: MashupConfig,
                          slice_start_a: float,
                          scores: list[tuple[float, MashabilityResult]],
                          best_result_idx: int,
                          best_result: MashabilityResult,
                          verbose: bool) -> str:
    write = print if verbose else lambda x: None
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
    system_messages = ["Mashup completed!"] + system_messages
    return "\n".join(system_messages)


def create_mash(dataset: SongDataset,
                submitted_audio_a: Audio,
                submitted_entry_a: DatasetEntry,
                submitted_parts_a: DemucsCollection,
                best_result_url: YouTubeURL,
                best_result_start_bar: int,
                best_result_transpose: int,
                config: MashupConfig):
    """Creates the mashup from the given parameters. Returns the mashup and the original audio for track B."""
    write = print if config._verbose else lambda x: None

    write(f"Loading audio for song B from {best_result_url}...")
    b_audio = dataset.get_audio(best_result_url)
    write(f"Got audio with duration {b_audio.duration} seconds.")

    entry = dataset.get_entry(best_result_url)
    if entry is None:
        raise ValueError(f"Song B ({best_result_url}) is not in the dataset.")
    b_downbeats = entry.downbeats.onsets
    nbars = config.nbars
    slice_start_b, slice_end_b = b_downbeats[best_result_start_bar], b_downbeats[best_result_start_bar + nbars]
    submitted_audio_b = b_audio.slice_seconds(slice_start_b, slice_end_b)
    submitted_downbeats_b = entry.downbeats.slice_seconds(slice_start_b, slice_end_b)

    write("Analyzing parts for song B...")
    submitted_parts_b = dataset.get_parts(entry.url).slice_seconds(slice_start_b, slice_end_b)

    print(submitted_audio_a.duration, submitted_audio_b.duration)

    write("Creating mashup...")
    mashup, mode_used = create_mashup(
        submitted_audio_a=submitted_audio_a,
        submitted_parts_a=submitted_parts_a,
        submitted_downbeats_a=submitted_entry_a.downbeats,
        submitted_audio_b=submitted_audio_b,
        submitted_parts_b=submitted_parts_b,
        submitted_downbeats_b=submitted_downbeats_b,
        transpose=best_result_transpose,
        natural_drum_activity_threshold=config.natural_drum_activity_threshold,
        natural_drum_proportion_threshold=config.natural_drum_proportion_threshold,
        natural_vocal_activity_threshold=config.natural_vocal_activity_threshold,
        natural_vocal_proportion_threshold=config.natural_vocal_proportion_threshold,
        mode=config.mashup_mode,
        left_pan=config.left_pan,
        verbose=config._verbose,
    )
    write(f"Got mashup with mode {mode_used}.")
    return submitted_audio_a, submitted_audio_b, mashup


def save_mashup_result(submitted_audio_a: Audio, submitted_audio_b: Audio, mashup: Audio, mashup_id: MashupID, config: MashupConfig):
    """Save the mashup result to the disk. If the folder already exists, it will be overwritten."""
    if not config.save_original:
        return

    save_dir = os.path.join("./", "resources", "mashups", mashup_id.to_string())
    os.makedirs(save_dir, exist_ok=True)

    submitted_audio_a.save(os.path.join(save_dir, "song_a.wav"))
    submitted_audio_b.save(os.path.join(save_dir, "song_b.wav"))
    mashup.save(os.path.join(save_dir, "mashup.wav"))


def mashup_from_id(mashup_id_str: str, config: MashupConfig | None = None, dataset: SongDataset | None = None) -> Audio:
    """Performs a mashup from a mashup ID"""
    mashup_id = MashupID.from_string(mashup_id_str)

    if config is not None:
        # Copy all aspects of the config, except for the starting point
        # where we set it to song a's starting point
        # Use a little hack just don't tell anybody else is ok :)
        config = copy.deepcopy(config)
        object.__setattr__(config, "starting_point", mashup_id.song_a_start_time)
    else:
        config = MashupConfig(starting_point=mashup_id.song_a_start_time)

    validate_config(config)
    def write(x): return print(x, flush=True) if config._verbose else lambda x: None
    write(f"Creating mashup from ID {mashup_id_str}")

    dataset = load_dataset(config) if dataset is None else dataset
    write(f"Loaded dataset with {len(dataset)} songs.")

    write(f"Loading audio for song A from {mashup_id.song_a}...")
    a_audio = dataset.get_audio(mashup_id.song_a)
    write(f"Got audio with duration {a_audio.duration} seconds.")

    write("Analyzing beats for song A...")
    a_entry = dataset.get_or_create_entry(mashup_id.song_a)
    a_downbeats = a_entry.downbeats
    if len(a_downbeats) < config.nbars:
        raise InvalidMashup("The audio is too short to mashup with the dataset.")
    write(f"Got beat analysis result with {len(a_downbeats)} bars.")

    write("Determining slice results...")
    submitted_downbeats_a, slice_start_a, slice_end_a = determine_slice_results(a_downbeats, config)
    assert slice_start_a >= 0, f"Starting point must be >= 0, got {slice_start_a}"
    write(f"Got slice results with start {slice_start_a} and end {slice_end_a}.")

    submitted_audio_a = a_audio.slice_seconds(slice_start_a, slice_end_a)
    submitted_entry_a = create_entry(
        url=mashup_id.song_a,
        beats=a_entry.beats.slice_seconds(slice_start_a, slice_end_a),
        downbeats=submitted_downbeats_a,
        chords=a_entry.chords.slice_seconds(slice_start_a, slice_end_a),
    )

    submitted_audio_a, submitted_audio_b, mashup = create_mash(
        dataset=dataset,
        submitted_audio_a=submitted_audio_a,
        submitted_entry_a=submitted_entry_a,
        submitted_parts_a=dataset.get_parts(mashup_id.song_a).slice_seconds(slice_start_a, slice_end_a),
        best_result_url=mashup_id.song_b,
        best_result_start_bar=mashup_id.song_b_start_bar,
        best_result_transpose=mashup_id.transpose,
        config=config,
    )

    save_mashup_result(submitted_audio_a, submitted_audio_b, mashup, mashup_id, config)

    if config.append_song_to_dataset:
        write("Appending song to dataset...")
        dataset.save_entry(submitted_entry_a)
    return mashup


def mashup_song(link: YouTubeURL, config: MashupConfig, dataset: SongDataset | None = None) -> tuple[Audio, list[tuple[float, MashabilityResult]], str]:
    """
    Mashup the given audio with the dataset.
    """
    validate_config(config)
    def write(x): return print(x, flush=True) if config._verbose else lambda x: None
    write(f"Creating mashup for {link}")

    if dataset is None:
        dataset = load_dataset(config)

    song_a_entry = None
    if link in dataset:
        song_a_entry = dataset.get_entry(link)
        dataset = dataset.filter(lambda x: x.url != link)

    assert isinstance(dataset, SongDataset), f"Expected SongDataset, got {type(dataset)}"
    if len(dataset) == 0:
        raise InvalidMashup("No songs in the dataset.")

    write(f"Loaded dataset with {len(dataset)} songs.")
    write(f"Loading audio for song A from {link}...")
    a_audio = dataset.get_audio(link)
    write(f"Got audio with duration {a_audio.duration} seconds.")

    write("Preparing entry for song A...")
    if song_a_entry is None:
        song_a_entry = create_entry(
            url=link,
            audio=a_audio,
            chord_model_path=config._chord_model_path if not config.use_simplified_chord else config._simple_chord_model_path,
            beat_model_path=config._beat_model_path,
            use_simplified_chord=config.use_simplified_chord,
        )

    assert isinstance(song_a_entry, DatasetEntry), f"Expected DatasetEntry, got {type(song_a_entry)}"

    write("Determining slice results...")
    song_a_downbeats, slice_start_a, slice_end_a = determine_slice_results(song_a_entry.downbeats, config)
    assert slice_start_a >= 0, f"Starting point must be >= 0, got {slice_start_a}"
    write(f"Got slice results with start {slice_start_a} and end {slice_end_a}.")

    song_a_chords = song_a_entry.chords.slice_seconds(slice_start_a, slice_end_a)
    song_a_beats = song_a_entry.beats.slice_seconds(slice_start_a, slice_end_a)

    # Prepare the entry to calculate the mashability
    submitted_entry = create_entry(
        url=link,
        beats=song_a_beats,
        downbeats=song_a_downbeats,
        chords=song_a_chords,
    )

    # Create the mashup
    write("Performing search...")
    write(str(config))
    scores = calculate_mashability(
        submitted_entry,
        dataset,
        max_transpose=config.max_transpose,
        min_music_percentage=config.min_music_percentage,
        delta_bpm=(config.min_delta_bpm, config.max_delta_bpm),
        max_distance=config.max_distance,
        keep_first_k=config.keep_first_k_results,
        filter_top_scores=config.filter_first,
        verbose=config._verbose,
        use_simplified_chord_distance=config.use_simplified_chord,
    )

    if len(scores) == 0:
        raise InvalidMashup("No suitable songs found in the dataset.")
    write(f"Got {len(scores)} results.")

    best_result_idx = 0
    best_result = scores[best_result_idx][1]
    system_messages = get_search_result_log(link, config, slice_start_a, scores, best_result_idx, best_result, verbose=config._verbose)

    if config._skip_mashup:
        return a_audio.slice_seconds(slice_start_a, slice_end_a), scores, system_messages

    # Create mashup
    mashup_id = MashupID(
        song_a=link,
        song_a_start_time=config.starting_point,
        song_b=best_result.url,
        song_b_start_bar=best_result.start_bar,
        transpose=best_result.transpose,
    )

    submitted_audio_a, submitted_audio_b, mashup = create_mash(
        dataset=dataset,
        submitted_audio_a=a_audio.slice_seconds(slice_start_a, slice_end_a),
        submitted_entry_a=submitted_entry,
        submitted_parts_a=dataset.get_parts(link).slice_seconds(slice_start_a, slice_end_a),
        best_result_url=best_result.url,
        best_result_start_bar=best_result.start_bar,
        best_result_transpose=best_result.transpose,
        config=config,
    )

    save_mashup_result(submitted_audio_a, submitted_audio_b, mashup, mashup_id, config)
    if config.append_song_to_dataset:
        write("Appending song to dataset...")
        dataset.save_entry(song_a_entry)
    return mashup, scores, system_messages


def mashup_from_audio(audio: Audio, config: MashupConfig):
    """Performs a mashup from an audio file."""
    def write(x): return print(x, flush=True) if config._verbose else lambda x: None
    write(f"Creating mashup from audio")

    write("Analyzing audio...")
    song_a_entry = create_entry(
        url=YouTubeURL.get_placeholder(),
        audio=audio,
        chord_model_path=config._chord_model_path if not config.use_simplified_chord else config._simple_chord_model_path,
        beat_model_path=config._beat_model_path,
        use_simplified_chord=config.use_simplified_chord,
    )

    write("Determining slice results...")
    song_a_downbeats, slice_start_a, slice_end_a = determine_slice_results(song_a_entry.downbeats, config)
    assert slice_start_a >= 0, f"Starting point must be >= 0, got {slice_start_a}"
    write(f"Got slice results with start {slice_start_a} and end {slice_end_a}.")

    # Prepare the entry to calculate the mashability
    submitted_entry = create_entry(
        url=YouTubeURL.get_placeholder(),
        beats=song_a_entry.beats.slice_seconds(slice_start_a, slice_end_a),
        downbeats=song_a_downbeats,
        chords=song_a_entry.chords.slice_seconds(slice_start_a, slice_end_a),
    )

    dataset = load_dataset(config)

    # Create the mashup
    write("Performing search...")
    write(str(config))
    scores = calculate_mashability(
        submitted_entry,
        dataset,
        max_transpose=config.max_transpose,
        min_music_percentage=config.min_music_percentage,
        delta_bpm=(config.min_delta_bpm, config.max_delta_bpm),
        max_distance=config.max_distance,
        keep_first_k=config.keep_first_k_results,
        filter_top_scores=config.filter_first,
        verbose=config._verbose,
        use_simplified_chord_distance=config.use_simplified_chord,
    )

    if len(scores) == 0:
        raise InvalidMashup("No suitable songs found in the dataset.")
    write(f"Got {len(scores)} results.")

    best_result_idx = 0
    best_result = scores[best_result_idx][1]

    a_audio = audio.slice_seconds(slice_start_a, slice_end_a)

    if config._skip_mashup:
        return a_audio, scores, "Skipped mashup"

    # Create mashup
    submitted_audio_a, submitted_audio_b, mashup = create_mash(
        dataset,
        a_audio,
        song_a_entry,
        demucs_separate(a_audio),
        best_result.url,
        best_result.start_bar,
        best_result.transpose,
        config,
    )

    mashup_id = MashupID(
        song_a=YouTubeURL.get_placeholder(),
        song_a_start_time=config.starting_point,
        song_b=best_result.url,
        song_b_start_bar=best_result.start_bar,
        transpose=best_result.transpose,
    )

    system_messages = get_search_result_log(YouTubeURL.get_placeholder(), config, slice_start_a, scores, best_result_idx, best_result, verbose=config._verbose)

    save_mashup_result(submitted_audio_a, submitted_audio_b, mashup, mashup_id, config)
    if config.append_song_to_dataset:
        write("Appending song to dataset...")
        dataset.save_entry(song_a_entry)
    return mashup, scores, system_messages
