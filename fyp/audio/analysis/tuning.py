# Returns the tuning correction coefficient (-0.5 < x < 0.5 semitones)

from typing import Any
from ...audio import Audio
import librosa
from .base import TuningAnalysisResult
import torch
from torch import Tensor

def analyse_tuning(audio: Audio) -> TuningAnalysisResult:
    tuning = librosa.estimate_tuning(y = audio.numpy(), sr = audio.sample_rate)
    return TuningAnalysisResult(tuning)
