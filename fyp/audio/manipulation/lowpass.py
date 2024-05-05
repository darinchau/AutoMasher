import torch
import torchaudio.functional as F
from .base import AudioTransform
from torch import Tensor

class LowpassFilter(AudioTransform):
    """Implements the lowpass filter."""
    def __init__(self, cutoff_freq: float, Q: float = 0.707):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.Q = Q

    def apply(self, audio: Tensor, sample_rate: int) -> Tensor:
        return F.lowpass_biquad(audio, sample_rate, self.cutoff_freq, self.Q)

class HighpassFilter(AudioTransform):
    """Implements the highpass filter."""
    def __init__(self, cutoff_freq: float, Q: float = 0.707):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.Q = Q

    def apply(self, audio: Tensor, sample_rate: int) -> Tensor:
        return F.highpass_biquad(audio, sample_rate, self.cutoff_freq, self.Q)
    