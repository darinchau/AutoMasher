from torch import Tensor
from abc import ABCMeta, abstractmethod as abstract
from .. import Audio, AudioCollection

class AudioSeparator(metaclass=ABCMeta):
    """Interface for audio separation strategy pattern. An implementation must implement the separate_audio method, 
    which takes in an audio (shape: (nchannels, t)) and outputs dictionary containing the separated audio (shape: (nchannels, t))"""
    @abstract
    def separate_audio(self, audio: Audio) -> AudioCollection: ...
