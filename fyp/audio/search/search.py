# Has convenience functions and classes to use the search pipeline to search for songs.
import os
import numpy as np
from ... import Audio
from ...util import YouTubeURL
from ..analysis import ChordAnalysisResult, BeatAnalysisResult, analyse_beat_transformer, analyse_chord_transformer, analyse_beat_transformer
from ..base import AudioCollection
from ..dataset import SongDataset
from .align import filter_first, curve_score
from ..separation import DemucsAudioSeparator
from .align import calculate_mashability, MashabilityResult
from ..dataset.cache import LocalCache
from ..dataset import DatasetEntry
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class SearchConfig:
    """
    The configuration for the search.

    Attributes:
        max_transpose: The maximum number of semitones to transpose the audio. If a tuple,
            it will represent the range of transposition. If an integer, it will represent
            the maximum transposition (equivalent to (-k, k)). Default is 3.

        min_music_percentage: The minimum percentage of music in the audio. Default is 0.8.

        max_delta_bpm: The maximum bpm deviation allowed for the queried song. Default is 1.25.

        min_delta_bpm: The minimum bpm deviation allowed for the queried song. Default is 0.8.

        keep_first_k: The number of top results to keep. Default is 20. Set to -1 to keep all results.

        max_score: The maximum mashability score allowed for the queried song. Default is infinity.

        pychorus_work_factor: The work factor for the pychorus library. Must be between 10 and 20. The higher the number,
            the less accurate but also the less runtime. This parameter scales (inverse) exponentially to the runtime i.e.
            A work factor of 13 ought to have twice the runtime compared to work factor of 14. Default is 14.

        progress_bar: Whether to show the progress bar during the search. Default is True.

        bar_number: The bar number (according to the beat analysis result) to slice the song. If set to None, the pipeline
            will perform time segmentation and slice the song according to the chorus. Default is None.

        nbars: The number of bars to slice the song. If set to None, the pipeline will perform time segmentation and slice
            the song according to the chorus, and set nbars to 8 bars. Default is None.

        filter_func: A lambda function to be used in the search. The filter should take in a dataset entry and return
            a boolean. If the filter returns True, the entry will be included in the search. When this parameter is set
            to None, the search will include all entries. Default is None.

        filter_first: Whether to include only the best result of each song from the search. This does not affect the
            runtime of the search since the filtering is done after the search. Default is True.

        dataset_path: The path to the dataset. Default is "hkust-fypho2", which is the dataset path on hugging face.
            Feel free to keep this default value because hugging face will handle caching for us.
            This path will be directly passed into SongDataset.load, so it should be a valid path for that function.
            Refer to the SongDataset.load documentation for more information.

        chord_model_path: The path to the chord model. Default is "resources/ckpts/btc_model_large_voca.pt", which is the
            model path on a fresh clone of the repository from the root

        beat_model_path: The path to the beat model. Default is "resources/ckpts/beat_transformer.pt", which is the model
            path on a fresh clone of the repository from the root

        cache_dir: The directory to store the cache. If set to None, will disable caching. Default is "./", which is the current directory.

        cache: Whether to cache the results. If set to False, the pipeline will force recomputation on every search. Default is True.

        verbose: Whether to show the progress bars during the search. Default is False.
    """
    max_transpose: int | tuple[int, int] = 3
    min_music_percentage: float = 0.8
    max_delta_bpm: float = 1.25
    min_delta_bpm: float = 0.8
    keep_first_k: int = 20
    max_score: float = float("inf")
    bar_number: int | None = None
    nbars: int | None = None
    filter_func: Callable[[DatasetEntry], bool] | None = None
    filter_first: bool = True
    chord_model_path: str = "resources/ckpts/btc_model_large_voca.pt"
    beat_model_path: str = "resources/ckpts/beat_transformer.pt"
    dataset_path: str = "HKUST-FYPHO2/audio-infos-filtered"
    cache_dir: str | None = "./"
    cache: bool = True
    verbose: bool = False

    def __post_init__(self):
        if self.cache_dir is not None and not os.path.isdir(self.cache_dir):
            raise ValueError(f"Cache directory not found: {self.cache_dir}")

class SongSearchState:
    """A state object for song searching that caches the results of the search and every intermediate step."""
    def __init__(self, link: YouTubeURL, config: SearchConfig, audio: Audio | None = None, dataset: SongDataset | None = None):
        self._link = link
        self._audio = audio
        self.search_config = config
        self._cache_handler = LocalCache(config.cache_dir, self._link)
        self._dataset = dataset
        self._all_scores: list[tuple[float, MashabilityResult]] = []
        self._raw_chord_result: ChordAnalysisResult | None = None
        self._raw_beat_result: BeatAnalysisResult | None = None
        self._raw_parts_result: AudioCollection | None = None
        self._submitted_chord_result: ChordAnalysisResult | None = None
        self._submitted_beat_result: BeatAnalysisResult | None = None
        self._submitted_audio: Audio | None = None
        self._submitted_parts: AudioCollection | None = None
        self._slice_start_bar: int | None = None
        self._slice_nbar: int | None = None

    @property
    def link(self) -> YouTubeURL:
        return self._link

    @property
    def audio(self) -> Audio:
        print(f"Loading audio: {self._cache_handler.link}")
        if self._audio is None:
            self._audio = self._cache_handler.get_audio()
        if self._audio is None:
            self._audio = Audio.load(self.link)
        return self._audio

    @property
    def dataset(self) -> SongDataset:
        if self._dataset is None:
            self._dataset = SongDataset.load(self.search_config.dataset_path)
        return self._dataset

    @property
    def raw_chord_result(self) -> ChordAnalysisResult:
        """The raw chord result of the user-submitted song without any processing."""
        if self._raw_chord_result is None:
            self._raw_chord_result = analyse_chord_transformer(
                self.audio,
                model_path=self.search_config.chord_model_path,
            )
            self._cache_handler._store_chord_analysis(self._raw_chord_result)
        return self._raw_chord_result

    @property
    def raw_beat_result(self) -> BeatAnalysisResult:
        """The raw beat result of the user-submitted song without any processing."""
        if self._raw_beat_result is None:
            self._raw_beat_result = analyse_beat_transformer(
                parts = self.raw_parts_result,
                model_path=self.search_config.beat_model_path
            )
            self._cache_handler._store_beat_analysis(self._raw_beat_result)
        return self._raw_beat_result

    @property
    def slice_start_bar(self):
        """Returns the start bar of the slice. If not found, it will perform the chorus analysis."""
        if self._slice_start_bar is None:
            self._slice_start_bar, self._slice_nbar = self._chorus_analysis()
        return self._slice_start_bar

    @property
    def slice_nbar(self):
        """Returns the number of bars of the slice. If not found, it will perform the chorus analysis."""
        if self._slice_nbar is None:
            self._slice_start_bar, self._slice_nbar = self._chorus_analysis()
        return self._slice_nbar

    @property
    def slice_start(self) -> float:
        """Returns the start time of the slice in seconds."""
        return self.raw_beat_result.downbeats[self.slice_start_bar]

    @property
    def slice_end(self) -> float:
        """Returns the end time of the slice in seconds."""
        return self.raw_beat_result.downbeats[self.slice_start_bar + self.slice_nbar]

    @property
    def raw_parts_result(self):
        """Returns the raw parts result of the user-submitted song without any processing."""
        if self._raw_parts_result is None:
            demucs = DemucsAudioSeparator()
            self._raw_parts_result = demucs.separate(self.audio)
        return self._raw_parts_result

    @property
    def submitted_chord_result(self):
        """Returns the chord result submitted for database search. It is the slice of the raw chord result."""
        if self._submitted_chord_result is None:
            self._submitted_chord_result = self.raw_chord_result.slice_seconds(self.slice_start, self.slice_end)
        return self._submitted_chord_result

    @property
    def submitted_beat_result(self):
        """Returns the beat result submitted for database search. It is the slice of the raw beat result."""
        if self._submitted_beat_result is None:
            self._submitted_beat_result = self.raw_beat_result.slice_seconds(self.slice_start, self.slice_end)
        return self._submitted_beat_result

    @property
    def submitted_audio(self):
        """Returns the audio submitted for database search. It is the slice of the raw audio."""
        if self._submitted_audio is None:
            self._submitted_audio = self.audio.slice_seconds(self.slice_start, self.slice_end)
        return self._submitted_audio

    @property
    def submitted_parts(self):
        """Returns the parts submitted for database search. It is the slice of the raw parts."""
        if self._submitted_parts is None:
            self._submitted_parts = self.raw_parts_result.slice_seconds(self.slice_start, self.slice_end)
        return self._submitted_parts

    def _chorus_analysis(self):
        assert self.search_config.bar_number is not None, "Chorus detection is not implemented. Bar number must be provided for chorus analysis"
        _slice_start_bar = self.search_config.bar_number
        _slice_nbar = self.search_config.nbars if self.search_config.nbars is not None else 8
        if _slice_start_bar + _slice_nbar >= len(self.raw_beat_result.downbeats):
            raise ValueError(f"Bar number out of bounds: {_slice_start_bar + _slice_nbar} >= {len(self.raw_beat_result.downbeats)}")
        return _slice_start_bar, _slice_nbar

def search_song(state: SongSearchState) -> list[tuple[float, MashabilityResult]]:
    """Searches for songs that match the audio. Returns a list of tuples, where the first element is the score and the second element is the url.

    :param link: The link of the audio
    :param audio: The audio object. If not provided, the audio will be loaded from the link.
    :param reset_states: Whether to reset the states of the pipeline. Default is True."""

    dataset = state.dataset.filter(state.search_config.filter_func)

    scores_ = calculate_mashability(
                        submitted_chord_result=state.submitted_chord_result,
                        submitted_beat_result=state.submitted_beat_result,
                        dataset=dataset,
                        max_transpose=state.search_config.max_transpose,
                        min_music_percentage=state.search_config.min_music_percentage,
                        max_delta_bpm=state.search_config.max_delta_bpm,
                        min_delta_bpm=state.search_config.min_delta_bpm,
                        max_score=state.search_config.max_score,
                        keep_first_k=state.search_config.keep_first_k,
                        verbose=state.search_config.verbose,
    )

    if state.search_config.filter_first:
        scores_ = filter_first(scores_)

    scores = [(curve_score(x[0]), x[1]) for x in scores_]
    state._all_scores = scores
    return scores
