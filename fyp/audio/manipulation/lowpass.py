import torch
import torchaudio.functional as F
from .base import AudioTransform, Audio
from torch import Tensor

class LowpassFilter(AudioTransform):
    """Implements the lowpass filter."""
    def __init__(self, cutoff_freq: float, Q: float = 0.707):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.Q = Q

    def apply(self, audio: Audio) -> Audio:
        sample_rate = int(audio.sample_rate)
        y = audio._data[..., 0]
        return Audio(F.lowpass_biquad(y, sample_rate, self.cutoff_freq, self.Q), sample_rate)

class HighpassFilter(AudioTransform):
    """Implements the highpass filter."""
    def __init__(self, cutoff_freq: float, Q: float = 0.707):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.Q = Q

    def apply(self, audio: Audio) -> Audio:
        sample_rate = int(audio.sample_rate)
        y = audio._data[..., 0]
        return Audio(F.highpass_biquad(y, sample_rate, self.cutoff_freq, self.Q), sample_rate)
