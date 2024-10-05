from torch import Tensor
from abc import ABCMeta, abstractmethod as abstract
from typing import Any
from ...audio import Audio

class AudioTransform(metaclass=ABCMeta):
    """Interface for audio manipulation tools. An implementation must implement the apply method, which takes
    in and returns an audio."""
    @abstract
    def apply(self, audio: Audio) -> Audio: ...
