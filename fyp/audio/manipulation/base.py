from torch import Tensor
from abc import ABCMeta, abstractmethod as abstract
from typing import Any
from ...audio import Audio

class AudioTransform(metaclass=ABCMeta):
    """Interface for audio manipulation tools. An implementation must implement the apply method, which takes
    in and returns an audio. The tensor will be in audio format and the sample rate will be in hertz.
    If the transform changes the sample rate of the audio, it can return a tuple of (audio, sr)."""
    @abstract
    def apply(self, audio: Tensor, sample_rate: int) -> Tensor | tuple[Tensor, int]: ...

    def __call__(self, audio: Audio) -> Audio:
        return audio.apply(self)
