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


class LocalCache(CacheHandler):
    def __init__(self, cache_dir: str | None, link: YouTubeURL) -> None:
        self.cache_dir = cache_dir
        self.link = link

    def store_audio(self, audio: Audio) -> None:
        if self.cache_dir is None:
            return
        audio.save(os.path.join(self.cache_dir, self.link.video_id))

    def store_chord_analysis(self, result: ChordAnalysisResult) -> None:
        if self.cache_dir is None:
            return
        result.save(os.path.join(self.cache_dir, f"chord_{self.link.video_id}.cache"))

    def store_beat_analysis(self, result: BeatAnalysisResult) -> None:
        if self.cache_dir is None:
            return
        result.save(os.path.join(self.cache_dir, f"beat_{self.link.video_id}.cache"))

    def get_audio(self) -> Audio | None:
        if self.cache_dir is None:
            return None
        if not os.path.isfile(os.path.join(self.cache_dir, self.link.video_id)):
            return None
        return Audio.load(os.path.join(self.cache_dir, self.link.video_id))

    def get_chord_analysis(self) -> ChordAnalysisResult | None:
        if self.cache_dir is None:
            return
        if not os.path.isfile(os.path.join(self.cache_dir, f"chord_{self.link.video_id}.cache")):
            return None
        return ChordAnalysisResult.load(os.path.join(self.cache_dir, f"chord_{self.link.video_id}.cache"))

    def get_beat_analysis(self) -> BeatAnalysisResult | None:
        if self.cache_dir is None:
            return
        if not os.path.isfile(os.path.join(self.cache_dir, f"beat_{self.link.video_id}.cache")):
            return None
        return BeatAnalysisResult.load(os.path.join(self.cache_dir, f"beat_{self.link.video_id}.cache"))
