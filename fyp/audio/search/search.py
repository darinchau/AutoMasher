from __future__ import annotations
import os
import numpy as np
from ... import Audio
from ...util import get_video_id, YouTubeURL
from ..analysis import ChordAnalysisResult, BeatAnalysisResult, analyse_beat_transformer, analyse_chord_transformer, analyse_beat_transformer
from ..base import AudioCollection
from ..dataset import SongDataset, DatasetEntry, SongGenre
from ..dataset.create import create_entry
from ..separation import DemucsAudioSeparator
from .align import calculate_mashability, MashabilityResult
from .cache import LocalCache
from .search_config import SearchConfig
from dataclasses import dataclass
from math import exp
from typing import Any

def filter_first(scores: list[tuple[float, MashabilityResult]]) -> list[tuple[float, MashabilityResult]]:
    seen = set()
    result = []
    for score in scores:
        id = score[1].url_id
        if id not in seen:
            seen.add(id)
            result.append(score)
    return result

def curve_score(score: float) -> float:
    """Returns the curve score"""
    return round(100 * exp(-.05 * score), 2)

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
            self._cache_handler.store_chord_analysis(self._raw_chord_result)
        return self._raw_chord_result

    @property
    def raw_beat_result(self) -> BeatAnalysisResult:
        """The raw beat result of the user-submitted song without any processing."""
        if self._raw_beat_result is None:
            self._raw_beat_result = analyse_beat_transformer(
                parts = self.raw_parts_result,
                model_path=self.search_config.beat_model_path
            )
            self._cache_handler.store_beat_analysis(self._raw_beat_result)
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
        if self._submitted_parts is None:
            self._submitted_parts = self.raw_parts_result.slice_seconds(self.slice_start, self.slice_end)
        return self._submitted_parts

    def _chorus_analysis(self):
        """Calculates the start bar and the number of bars of the slice. If the bar number is provided, it will use that instead."""
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
