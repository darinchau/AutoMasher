# Handles song and result caching

import os
from dataclasses import dataclass
from typing import Any
from abc import ABC, abstractmethod
from .. import Audio
from ..analysis import ChordAnalysisResult, BeatAnalysisResult
from ...util import YouTubeURL

class CacheHandler(ABC):
    @abstractmethod
    def store_audio(self, audio: Audio) -> None:
        pass

    @abstractmethod
    def store_chord_analysis(self, result: ChordAnalysisResult) -> None:
        pass

    @abstractmethod
    def store_beat_analysis(self, result: BeatAnalysisResult) -> None:
        pass

    @abstractmethod
    def get_audio(self) -> Audio | None:
        pass

    @abstractmethod
    def get_chord_analysis(self) -> ChordAnalysisResult | None:
        pass

    @abstractmethod
    def get_beat_analysis(self) -> BeatAnalysisResult | None:
        pass

    @property
    def cached_audio(self) -> bool:
        """Returns whether the audio is cached. This is an inefficient method and should be overridden by subclasses."""
        return self.get_audio() is not None

    @property
    def cached_chord_analysis(self) -> bool:
        """Returns whether the chord analysis is cached. This is an inefficient method and should be overridden by subclasses"""
        return self.get_chord_analysis() is not None

    @property
    def cached_beat_analysis(self) -> bool:
        """Returns whether the beat analysis is cached. This is an inefficient method and should be overridden by subclasses"""
        return self.get_beat_analysis() is not None


class LocalCache(CacheHandler):
    def __init__(self, cache_dir: str | None, link: YouTubeURL) -> None:
        self.cache_dir = cache_dir
        self.link = link

    @property
    def _audio_save_path(self) -> str:
        if self.cache_dir is None:
            raise ValueError("Cache directory is not set")
        return os.path.join(self.cache_dir, f"{self.link.video_id}.mp3")

    @property
    def _chord_save_path(self) -> str:
        if self.cache_dir is None:
            raise ValueError("Cache directory is not set")
        return os.path.join(self.cache_dir, f"chord_{self.link.video_id}.cache")

    @property
    def _beat_save_path(self) -> str:
        if self.cache_dir is None:
            raise ValueError("Cache directory is not set")
        return os.path.join(self.cache_dir, f"beat_{self.link.video_id}.cache")

    def store_audio(self, audio: Audio) -> None:
        if self.cache_dir is None:
            return
        audio.save(self._audio_save_path)

    def store_chord_analysis(self, result: ChordAnalysisResult) -> None:
        if self.cache_dir is None:
            return
        result.save(self._chord_save_path)

    def store_beat_analysis(self, result: BeatAnalysisResult) -> None:
        if self.cache_dir is None:
            return
        result.save(self._beat_save_path)

    def get_audio(self) -> Audio | None:
        if self.cache_dir is None:
            return None
        if not os.path.isfile(self._audio_save_path):
            return None
        return Audio.load(self._audio_save_path)

    def get_chord_analysis(self) -> ChordAnalysisResult | None:
        if self.cache_dir is None:
            return
        if not os.path.isfile(self._chord_save_path):
            return None
        return ChordAnalysisResult.load(self._chord_save_path)

    def get_beat_analysis(self) -> BeatAnalysisResult | None:
        if self.cache_dir is None:
            return
        if not os.path.isfile(self._beat_save_path):
            return None
        return BeatAnalysisResult.load(self._beat_save_path)

    @property
    def cached_audio(self) -> bool:
        return os.path.isfile(self._audio_save_path)

    @property
    def cached_chord_analysis(self) -> bool:
        return os.path.isfile(self._chord_save_path)

    @property
    def cached_beat_analysis(self) -> bool:
        return os.path.isfile(self._beat_save_path)
