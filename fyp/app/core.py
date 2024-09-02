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
from ..audio.dataset.create import create_entry
from ..audio.search.align import calculate_mashability
from ..audio.dataset.cache import CacheHandler
from ..audio.search.search import filter_first, curve_score
from ..audio.search.search_config import SearchConfig
from ..audio.separation import DemucsAudioSeparator
from ..util import YouTubeURL

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

        max_score: The maximum mashability score allowed for the queried song. Default is infinity.

        filter_first: Whether to include only the best result of each song from the search. This does not affect the runtime of the search since the filtering is done after the search. Default is True.

        cache: Whether to cache the queried song (if the youtube link is used). Default is True.

        keep_first_k_results: The number of results to keep. Set to -1 to keep all results. Default is 10.

        filter_uneven_bars: Whether to filter out songs with uneven bars. Default is True.

        filter_uneven_bars_min_threshold: The minimum threshold for the mean difference between downbeats. Default is 0.9.

        filter_uneven_bars_max_threshold: The maximum threshold for the mean difference between downbeats. Default is 1.1.

        filter_short_song_bar_threshold: The minimum number of bars for a song to be considered long enough. Default is 12.

        filter_uncached: Whether to filter out songs that are not cached. Default is True."""
    starting_point: float
    min_transpose: int = -3
    max_transpose: int = 3
    min_music_percentage: float = 0.8
    max_delta_bpm: float = 1.25
    min_delta_bpm: float = 0.8
    max_score: float = float("inf")
    filter_first: bool = True
    cache: bool = True
    keep_first_k_results: int = 10

    filter_uneven_bars: bool = True
    filter_uneven_bars_min_threshold: float = 0.9
    filter_uneven_bars_max_threshold: float = 1.1
    filter_short_song_bar_threshold: int = 12
    filter_uncached: bool = True

    # The path of stuff should not be exposed to the user
    dataset_path: str = "resources/dataset/audio-infos-v2.1.db"
    beat_model_path: str = "resources/ckpts/beat_transformer.pt"
    chord_model_path: str = "resources/ckpts/btc_model_large_voca.pt"


def load_dataset(config: MashupConfig) -> SongDataset:
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

def mashup_song(audio: Audio, config: MashupConfig, cache_handler: CacheHandler):
    """
    Mashup the given audio with the dataset.
    """
    dataset = load_dataset(config)

    demucs = DemucsAudioSeparator()
    parts_result = demucs.separate(audio)
    chord_result = cache_handler.get_chord_analysis() or analyse_chord_transformer(audio, model_path=config.chord_model_path)
    beat_result = cache_handler.get_beat_analysis() or analyse_beat_transformer(audio, model_path=config.beat_model_path, parts=parts_result)

    # TODO Do some processing to get submitted_chord_result and submitted_beat_result
    # Need to figure out how to do this with a timestamp instead of a bar number
    submitted_chord_result = ...
    submitted_beat_result = ...

    # Perform the search
    scores_ = calculate_mashability(
                        submitted_chord_result=submitted_chord_result,
                        submitted_beat_result=submitted_beat_result,
                        dataset=dataset,
                        max_transpose=(config.min_transpose, config.max_transpose),
                        min_music_percentage=config.min_music_percentage,
                        max_delta_bpm=config.max_delta_bpm,
                        min_delta_bpm=config.min_delta_bpm,
                        max_score=config.max_score,
                        keep_first_k=config.keep_first_k_results,
                        verbose=False,
    )

    if config.filter_first:
        scores_ = filter_first(scores_)

    scores = [(curve_score(x[0]), x[1]) for x in scores_]
    ...
