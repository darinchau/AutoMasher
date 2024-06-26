from __future__ import annotations
import os
import numpy as np
from ..analysis import ChordAnalysisResult, BeatAnalysisResult, analyse_beat_transformer, analyse_chord_transformer, analyse_beat_transformer
from ... import Audio
from typing import Any
from ..dataset import SongDataset, DatasetEntry, SongGenre
from ..dataset.create import create_entry
from ..separation import DemucsAudioSeparator
from .search_config import SearchConfig
from .align import MashabilityResult
from ...util.combine import get_video_id
from ..analysis import extract_chorus
from ..base import AudioCollection
from pytube import YouTube

class SongSearchState:
    """A state object for song searching that caches the results of the search and every intermediate step."""
    def __init__(self, link: str, config: SearchConfig, audio: Audio | None = None, dataset: SongDataset | None = None):
        self._link = link
        self._audio = audio
        self.search_config = config
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
    def link(self) -> str:
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
    def cache_dir(self) -> str | None:
        """Returns the beat cache path. If not found or cache is disabled, return None"""
        if self.search_config.cache_dir is None:
            return None
        if not self.search_config.cache:
            return None
        if not os.path.isdir(self.search_config.cache_dir):
            raise ValueError(f"Cache directory not found: {self.search_config.cache_dir}")
        return self.search_config.cache_dir

    @property
    def beat_cache_path(self) -> str | None:
        """Returns the beat cache path. If not found or cache is disabled, return None"""
        if not self.search_config.cache or self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, f"beat_{get_video_id(self.link)}.cache")

    @property
    def chord_cache_path(self) -> str | None:
        """Returns the chord cache path. If not found or cache is disabled, return None"""
        if not self.search_config.cache or self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, f"chord_{get_video_id(self.link)}.cache")

    @property
    def audio_cache_path(self) -> str | None:
        """Returns the audio cache path. If not found or cache is disabled, return None"""
        if not self.search_config.cache or self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, f"audio_{get_video_id(self.link)}.wav")

    @property
    def raw_chord_result(self) -> ChordAnalysisResult:
        """The raw chord result of the user-submitted song without any processing."""
        if self._raw_chord_result is None:
            self._raw_chord_result = analyse_chord_transformer(
                self.audio,
                model_path=self.search_config.chord_model_path,
                cache_path = self.chord_cache_path
            )
        return self._raw_chord_result

    @property
    def raw_beat_result(self) -> BeatAnalysisResult:
        """The raw beat result of the user-submitted song without any processing."""
        if self._raw_beat_result is None:
            self._raw_beat_result = analyse_beat_transformer(
                parts = lambda: self.raw_parts_result,
                cache_path = self.beat_cache_path,
                model_path=self.search_config.beat_model_path
            )
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
            self._raw_parts_result = demucs.separate_audio(self.audio)
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
        if self.search_config.bar_number is not None:
            _slice_start_bar = self.search_config.bar_number
            _slice_nbar = self.search_config.nbars if self.search_config.nbars is not None else 8
            if _slice_start_bar + _slice_nbar >= len(self.raw_beat_result.downbeats):
                raise ValueError(f"Bar number out of bounds: {_slice_start_bar + _slice_nbar} >= {len(self.raw_beat_result.downbeats)}")
            return _slice_start_bar, _slice_nbar

        slice_ranges = extract_chorus(self.raw_parts_result, self.raw_beat_result, self.audio, work_factor=self.search_config.pychorus_work_factor)
        slice_nbar = 8
        slice_start_bar = 0
        for i in range(len(slice_ranges)):
            if slice_ranges[i] + slice_nbar < len(self.raw_beat_result.downbeats):
                slice_start_bar = slice_ranges[i]
                break

        # Prevent array out of bounds below
        slice_nbar = self.search_config.nbars if self.search_config.nbars is not None else slice_nbar

        assert slice_start_bar + slice_nbar < len(self.raw_beat_result.downbeats)
        return slice_start_bar, slice_nbar

    def to_dataset_entry(self, genre: SongGenre, *,
                         audio_name: str | None = None,
                         playlist: str | None = None,
                         views: int | None = None) -> DatasetEntry:
        """Converts the search state to a dataset entry or future use. Will figure out the audio and views if not provided, and will assume the
        "No playlist" option if playlist not provided."""
        if audio_name is None or views is None:
            yt = YouTube(self.link)
            if audio_name is None:
                audio_name = yt.title
            if views is None:
                views = yt.views

        return create_entry(
            length = self.audio.duration,
            beats = self.raw_beat_result.beats.tolist(),
            downbeats = self.raw_beat_result.downbeats.tolist(),
            chords = self.raw_chord_result.labels.tolist(),
            chord_times = self.raw_chord_result.times.tolist(),
            genre = genre,
            audio_name = audio_name,
            url = self.link,
            playlist = playlist,
            views = views
        )
