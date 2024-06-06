# Speeds up or slows down audio without pitch change
import torch
import torchaudio.functional as F
from .base import AudioTransform
from torch import Tensor
from ... import Audio

PI = 3.1415926535897932

class TimeStretch(AudioTransform):
    """Stretchs the audio by a factor of `delta` without changing the pitch. If delta = 1.1, then this corresponds to making audio 10% faster"""
    def __init__(self, delta: float,
                 n_fft: int = 512,
                 win_length: int | None = None,
                 hop_length: int | None = None,
                 window: Tensor | None = None,
                 ):
        """delta(float): How much to shift the audio"""
        self._delta = delta
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window

    def apply(self, audio: Tensor, sample_rate: int) -> Tensor:
        return Audio(audio, sample_rate).change_speed(self._delta, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, window=self.window)._data
