# Handles song and result caching

import os
from dataclasses import dataclass
from typing import Any
from abc import ABC, abstractmethod
from .. import Audio
from ..analysis import ChordAnalysisResult, BeatAnalysisResult, analyse_chord_transformer, analyse_beat_transformer
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
    def get_option_audio(self) -> Audio | None:
        pass

    @abstractmethod
    def get_option_chord_analysis(self) -> ChordAnalysisResult | None:
        pass

    @abstractmethod
    def get_option_beat_analysis(self) -> BeatAnalysisResult | None:
        pass

    def get_audio(self, fallback: Callable[[], Audio]) -> Audio:
        audio = self.get_option_audio()
        if audio is not None:
            return audio
        audio = fallback()
        self.store_audio(audio)
        return audio

    def get_chord_analysis_result(self, fallback: Callable[[], ChordAnalysisResult] | None = None) -> ChordAnalysisResult:
        cr = self.get_option_chord_result()
        if cr is not None:
            return cr
        if fallback is not None:
            cr = fallback()
        else:
            cr = analyse_chord_transformer(self.get_audio())
        self.store_chord_result(cr)
        return cr

    def get_beat_analysis_result(self, fallback: Callable[[], BeatAnalysisResult] | None = None) -> BeatAnalysisResult:
        br = self.get_option_beat_result()
        if br is not None:
            return br
        if fallback is not None:
            br = fallback()
        else:
            br = analyse_beat_transformer(self.get_audio())
        self.store_beat_result(br)
        return br

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
        return os.path.join(self.cache_dir, f"{self.link.video_id}.wav")

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
