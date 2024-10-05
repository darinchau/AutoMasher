import torch
import torchaudio.functional as F
from .base import AudioTransform, Audio
from torch import Tensor
from typing import Callable
from enum import Enum
from typing import Any

class PitchShift(AudioTransform):
    def __init__(self, nsteps: int):
        self.nsteps = nsteps

    def apply(self, audio: Audio) -> Audio:
        y = audio._data[..., 0]
        return Audio(F.pitch_shift(y, int(audio.sample_rate), self.nsteps), audio.sample_rate)
