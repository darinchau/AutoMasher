# Contains an implementation (actually wraps pytorch library) for fade

from enum import Enum
import torch
import torchaudio.transforms as T
from .base import AudioTransform
from torch import Tensor

class FadeType(Enum):
    QUARTER_SINE = "quarter_sine" 
    HALF_SINE = "half_sine"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic" 
    EXPONENTIAL = "exponential"

class FadeExact(AudioTransform):
    """Same as Fade but uses number of frames instead."""
    def __init__(self, fade_in_frames: int, fade_out_frames: int, mode: FadeType = FadeType.LINEAR):
        self.mode = mode
        self.fadein = fade_in_frames
        self.fadeout = fade_out_frames
    
    def apply(self, audio: Tensor, sample_rate: int) -> Tensor:
        fade = T.Fade(fade_in_len = self.fadein, fade_out_len = self.fadeout, fade_shape = self.mode.value)
        return fade(audio)

class Fade(AudioTransform):
    """Makes the audio fade in and fade out by a certain number of seconds. Mode specifies the fade mode."""
    def __init__(self, fade_in_seconds: float, fade_out_seconds: float, mode: FadeType = FadeType.LINEAR):
        self.fadein = fade_in_seconds
        self.fadeout = fade_out_seconds
        self.mode = mode
    
    def apply(self, audio: Tensor, sample_rate: int) -> Tensor:
        fadein = int(self.fadein * sample_rate)
        fadeout = int(self.fadeout * sample_rate)
        return FadeExact(fadein, fadeout, self.mode).apply(audio, sample_rate)
