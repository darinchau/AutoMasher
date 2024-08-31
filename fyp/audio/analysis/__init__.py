# Ideally this module will give us tools to analyse an audio
# i.e. given audio, return tonality and pitch

from .key import (
    analyse_key_center,
    analyse_key_center_chroma
)
from .beat import (
    analyse_beat_transformer,
)
from .chord import (
    analyse_chord_transformer,
)
from .chroma import (
    chroma_cens,
    chroma_cqt,
    chroma_stft,
    ChromaFunction
)

from .key import KeyAnalysisResult
from .base import BeatAnalysisResult, ChordAnalysisResult, TimeSegmentResult

__all__ = [
    "analyse_key_center",
    "analyse_key_center_chroma",
    "analyse_beat_transformer",
    "analyse_chord_transformer",
    "chroma_cens",
    "chroma_cqt",
    "chroma_stft",
    "ChromaFunction",
    "KeyAnalysisResult",
    "BeatAnalysisResult",
    "ChordAnalysisResult",
    "TimeSegmentResult",
]
