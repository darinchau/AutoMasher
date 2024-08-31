# Handles song and result caching

import os
from dataclasses import dataclass
from typing import Any
from abc import ABC, abstractmethod
from ...audio import Audio
from ...audio.analysis import ChordAnalysisResult, BeatAnalysisResult
from ...util import YouTubeURL

class CacheHandler(ABC):
    @abstractmethod
    def store_audio(self, link: YouTubeURL, audio: Audio) -> None:
        pass

    @abstractmethod
    def store_chord_analysis(self, link: YouTubeURL, result: ChordAnalysisResult) -> None:
        pass

    @abstractmethod
    def store_beat_analysis(self, link: YouTubeURL, result: BeatAnalysisResult) -> None:
        pass

    @abstractmethod
    def get_audio(self, link: YouTubeURL) -> Audio | None:
        pass

    @abstractmethod
    def get_chord_analysis(self, link: YouTubeURL) -> ChordAnalysisResult | None:
        pass

    @abstractmethod
    def get_beat_analysis(self, link: YouTubeURL) -> BeatAnalysisResult | None:
        pass


class LocalCache(CacheHandler):
    def __init__(self, cache_dir: str | None):
        self.cache_dir = cache_dir

    def store_audio(self, link: YouTubeURL, audio: Audio) -> None:
        if self.cache_dir is None:
            return
        audio.save(os.path.join(self.cache_dir, link.video_id))

    def store_chord_analysis(self, link: YouTubeURL, result: ChordAnalysisResult) -> None:
        if self.cache_dir is None:
            return
        result.save(os.path.join(self.cache_dir, f"chord_{link.video_id}.cache"))

    def store_beat_analysis(self, link: YouTubeURL, result: BeatAnalysisResult) -> None:
        if self.cache_dir is None:
            return
        result.save(os.path.join(self.cache_dir, f"beat_{link.video_id}.cache"))

    def get_audio(self, link: YouTubeURL) -> Audio | None:
        if self.cache_dir is None:
            return None
        if not os.path.isfile(os.path.join(self.cache_dir, link.video_id)):
            return None
        return Audio.load(os.path.join(self.cache_dir, link.video_id))

    def get_chord_analysis(self, link: YouTubeURL) -> ChordAnalysisResult | None:
        if self.cache_dir is None:
            return
        if not os.path.isfile(os.path.join(self.cache_dir, f"chord_{link.video_id}.cache")):
            return None
        return ChordAnalysisResult.load(os.path.join(self.cache_dir, f"chord_{link.video_id}.cache"))

    def get_beat_analysis(self, link: YouTubeURL) -> BeatAnalysisResult | None:
        if self.cache_dir is None:
            return
        if not os.path.isfile(os.path.join(self.cache_dir, f"beat_{link.video_id}.cache")):
            return None
        return BeatAnalysisResult.load(os.path.join(self.cache_dir, f"beat_{link.video_id}.cache"))
