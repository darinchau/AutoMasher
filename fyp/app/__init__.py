# Packs the entire application into a single function which can be called by the main script and packaged into a WSGI application.
import os
from dataclasses import dataclass
from typing import Callable
from . import Audio, AudioCollection
from ..audio.dataset import DatasetEntry, SongDataset
from ..audio.analysis import ChordAnalysisResult, BeatAnalysisResult
from ..audio.search import search_song

@dataclass(frozen=True)
class MashupConfig:
    """
    The configuration for the mashup.

    Attributes:
        max_transpose: The maximum number of semitones to transpose the audio. If a tuple,
            it will represent the range of transposition. If an integer, it will represent
            the maximum transposition (equivalent to (-k, k)). Default is 3.

        min_music_percentage: The minimum percentage of music in the audio. Default is 0.8.

        max_delta_bpm: The maximum bpm deviation allowed for the queried song. Default is 1.25.

        min_delta_bpm: The minimum bpm deviation allowed for the queried song. Default is 0.8.

        max_score: The maximum mashability score allowed for the queried song. Default is infinity.

        filter_first: Whether to include only the best result of each song from the search. This does not affect the runtime of the search since the filtering is done after the search. Default is True.

        cache: Whether to cache the queried song (if the youtube link is used). Default is True."""
    max_transpose: int | tuple[int, int] = 3
    min_music_percentage: float = 0.8
    max_delta_bpm: float = 1.25
    min_delta_bpm: float = 0.8
    max_score: float = float("inf")
    filter_first: bool = True
    cache: bool = True
