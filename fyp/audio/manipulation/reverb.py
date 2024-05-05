import torch
import torchaudio.functional as F
from .base import AudioTransform
from torch import Tensor
from typing import Callable
from enum import Enum
from typing import Any
from ...audio import Audio
import torchaudio
from torchaudio.utils import download_asset

class ReverbSettings(Enum):
    """Contains several preset settings for reverb to support fast prototyping.
    
    (Dev) Note: Python does not support using functions or audios as enum values. My suspicion is
    because neither functions nor audios are hashable. Use the get_method function to get the load method for reverb settings
    If you want to add more default settings, it is your responsibility now to add your own enum value
    and the method to the mapping dict inside the get_method dict.
    """
    AUDITORIUM = "auditorium"

    @staticmethod
    def _all_settings():
        return {member.value for member in ReverbSettings}

    def get_audio(self, *args, **kwargs):
        """Gets the audio from enum keys"""
        # Feel free to put the preset load functions here

        # Load fn for auditorium
        def _auditorium(tail_percent: float = 0.):
            assert tail_percent >= 0. and tail_percent <= 1., "Tail percent must be between 0 and 1"
            SAMPLE_RIR = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav", progress=False)
            rir_raw, sample_rate = torchaudio.load(SAMPLE_RIR) # type: ignore
            aux = Audio(rir_raw, sample_rate).slice_seconds(1.01 + 0.29 * tail_percent, 1.3)
            return aux

        # Main function body
        # Using function ptrs is useful for passing in preset evaluation settings that requires us to load the aux audio.
        mapping: dict[str, Callable[[], Audio]] = {
            "auditorium": _auditorium
        }

        assert sorted(self._all_settings()) == sorted(mapping.keys())
        return mapping[self.value](*args, **kwargs)

class Reverb(AudioTransform):
    def __init__(self, aux: ReverbSettings | Any, conv_mode: str = "full"):
        """The init method should take an audio object - please move in the audio ok. `conv_mode` specifies whether to trim the audio 
        if the aux audio is too long. Check Conv1D for the specifications of this arg."""
        assert conv_mode in ('full', 'valid', 'same'), f"Convolution mode must be one of ('full', 'valid', 'same') but got '{conv_mode}'"
        self._mode = conv_mode

        if isinstance(aux, ReverbSettings):
            aux = aux.get_audio()
        elif isinstance(aux, Audio):
            pass
        else:
            raise TypeError(f"aux must be either a ReverbSettings enum or an Audio object but got {type(aux)}")
        self._aux: Audio = aux

    def apply(self, audio: Tensor, sample_rate: int) -> Tensor:
        """Use convolution to perform reverb. Requires an auxiliary input audio (from init) to specify the reverb decay. Use the Reverb class along with
        the ReverbSettings enum class for prespecified settings."""
        assert self._mode in ('full', 'valid', 'same')
        aux = self._aux.resample(target_sr=sample_rate)
        augmented = F.fftconvolve(audio, aux._data, mode=self._mode)
        return augmented
