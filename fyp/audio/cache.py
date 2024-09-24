# Handles song and result caching

import os
from dataclasses import dataclass
from typing import Any
from abc import ABC, abstractmethod
from typing import Callable
from . import Audio, DemucsCollection
from .analysis import ChordAnalysisResult, BeatAnalysisResult, analyse_chord_transformer, analyse_beat_transformer
from ..util import YouTubeURL
from .separation import DemucsAudioSeparator

class CacheHandler(ABC):
    """An abstract class that handles caching of audio, chord analysis, and beat analysis results. The URL of the song must be provided to instantiate this class."""
    def __init__(self, link: YouTubeURL) -> None:
        self.link = link

    @abstractmethod
    def _store_audio(self, audio: Audio) -> None:
        pass

    @abstractmethod
    def _store_chord_analysis(self, result: ChordAnalysisResult) -> None:
        pass

    @abstractmethod
    def _store_beat_analysis(self, result: BeatAnalysisResult) -> None:
        pass

    @abstractmethod
    def _store_parts_result(self, result: DemucsCollection) -> None:
        pass

    @abstractmethod
    def _get_option_audio(self) -> Audio | None:
        pass

    @abstractmethod
    def _get_audio_fallback(self) -> Audio:
        pass

    @abstractmethod
    def _get_option_chord_analysis(self) -> ChordAnalysisResult | None:
        pass

    @abstractmethod
    def _get_option_beat_analysis(self) -> BeatAnalysisResult | None:
        pass

    @abstractmethod
    def _get_option_parts_result(self) -> DemucsCollection | None:
        pass

    def get_audio(self, fallback: Callable[[], Audio] | None = None) -> Audio:
        audio = self._get_option_audio()
        if audio is not None:
            return audio
        fallback = fallback or self._get_audio_fallback
        audio = fallback()
        self._store_audio(audio)
        return audio

    def get_chord_analysis_result(self, fallback: Callable[[], ChordAnalysisResult] | None = None, **kwargs) -> ChordAnalysisResult:
        """Get the chord analysis result. If the result is not cached, the fallback function is called.
        args are passed to the fallback function which is analyse_chord_transformer."""
        cr = self._get_option_chord_analysis()
        if cr is not None:
            return cr
        if fallback is not None:
            cr = fallback()
        else:
            cr = analyse_chord_transformer(self.get_audio(), **kwargs)
        self._store_chord_analysis(cr)
        return cr

    def get_beat_analysis_result(self, fallback: Callable[[], BeatAnalysisResult] | None = None, **kwargs) -> BeatAnalysisResult:
        """Get the beat analysis result. If the result is not cached, the fallback function is called.
        args are passed to the fallback function which is analyse_beat_transformer."""
        br = self._get_option_beat_analysis()
        if br is not None:
            return br
        if fallback is not None:
            br = fallback()
        else:
            br = analyse_beat_transformer(self.get_audio(), parts=self.get_parts_result(), **kwargs)
        self._store_beat_analysis(br)
        return br

    def get_parts_result(self, fallback: Callable[[], DemucsCollection] | None = None, **kwargs) -> DemucsCollection:
        """Get the parts result. If the result is not cached, the fallback function is called.
        kwargs are passed to the fallback function which is DemucsAudioSeparator.separate."""
        pr = self._get_option_parts_result()
        if pr is not None:
            return pr
        if fallback is not None:
            pr = fallback()
        else:
            pr = DemucsAudioSeparator().separate(self.get_audio())
        self._store_parts_result(pr)
        return pr

    @property
    def cached_audio(self) -> bool:
        """Returns whether the audio is cached. This is an inefficient method and should be overridden by subclasses."""
        return self._get_option_audio() is not None

class LocalCache(CacheHandler):
    def __init__(self, cache_dir: str | None, link: YouTubeURL) -> None:
        super().__init__(link)
        self.cache_dir = cache_dir

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

    @property
    def _parts_save_path(self) -> str:
        if self.cache_dir is None:
            raise ValueError("Cache directory is not set")
        return os.path.join(self.cache_dir, f"{self.link.video_id}.demucs")

    def _store_audio(self, audio: Audio) -> None:
        if self.cache_dir is None:
            return
        audio.save(self._audio_save_path)

    def _store_chord_analysis(self, result: ChordAnalysisResult) -> None:
        if self.cache_dir is None:
            return
        result.save(self._chord_save_path)

    def _store_beat_analysis(self, result: BeatAnalysisResult) -> None:
        if self.cache_dir is None:
            return
        result.save(self._beat_save_path)

    def _store_parts_result(self, result: DemucsCollection) -> None:
        if self.cache_dir is None:
            return
        result.save(self._parts_save_path, inner_format="mp3")

    def _get_option_audio(self) -> Audio | None:
        if self.cache_dir is None:
            return None
        if not self.cached_audio:
            return None
        return Audio.load(self._audio_save_path)

    def _get_audio_fallback(self) -> Audio:
        return Audio.load(self.link)

    def _get_option_chord_analysis(self) -> ChordAnalysisResult | None:
        if self.cache_dir is None:
            return None
        if not os.path.isfile(self._chord_save_path):
            return None
        return ChordAnalysisResult.load(self._chord_save_path)

    def _get_option_beat_analysis(self) -> BeatAnalysisResult | None:
        if self.cache_dir is None:
            return
        if not os.path.isfile(self._beat_save_path):
            return None
        return BeatAnalysisResult.load(self._beat_save_path)

    def _get_option_parts_result(self) -> DemucsCollection | None:
        if self.cache_dir is None:
            return
        if not os.path.isfile(self._parts_save_path):
            return None
        return DemucsCollection.load(self._parts_save_path)

    @property
    def cached_audio(self) -> bool:
        return os.path.isfile(self._audio_save_path)


class MemoryCache(CacheHandler):
    _STORAGE = {}

    def _store_audio(self, audio: Audio) -> None:
        self._STORAGE[f"{self.link.video_id}_audio"] = audio

    def _store_chord_analysis(self, result: ChordAnalysisResult) -> None:
        self._STORAGE[f"{self.link.video_id}_chord"] = result

    def _store_beat_analysis(self, result: BeatAnalysisResult) -> None:
        self._STORAGE[f"{self.link.video_id}_beat"] = result

    def _store_parts_result(self, result: DemucsCollection) -> None:
        self._STORAGE[f"{self.link.video_id}_parts"] = result

    def _get_option_audio(self) -> Audio | None:
        return self._STORAGE.get(f"{self.link.video_id}_audio")

    def _get_audio_fallback(self) -> Audio:
        return Audio.load(self.link)

    def _get_option_chord_analysis(self) -> ChordAnalysisResult | None:
        return self._STORAGE.get(f"{self.link.video_id}_chord")

    def _get_option_beat_analysis(self) -> BeatAnalysisResult | None:
        return self._STORAGE.get(f"{self.link.video_id}_beat")

    def _get_option_parts_result(self) -> DemucsCollection | None:
        return self._STORAGE.get(f"{self.link.video_id}_parts")

    @property
    def cached_audio(self) -> bool:
        return f"{self.link.video_id}_audio" in self._STORAGE
