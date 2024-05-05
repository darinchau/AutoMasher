import torch
import torchaudio.functional as F
from .base import AudioTransform
from torch import Tensor
from typing import Callable
from enum import Enum
from typing import Any

class PitchShift(AudioTransform):
    def __init__(self, nsteps: int):
        self.nsteps = nsteps

    def apply(self, audio: Tensor, sample_rate: int) -> Tensor:
        return F.pitch_shift(audio, sample_rate, n_steps=self.nsteps)
