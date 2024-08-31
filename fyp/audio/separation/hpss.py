## This module implements the HPSS algorithm (actually wraps the librosa implementation)
## which separates an audio into the harmonic components and percussive components

from .. import Audio, AudioCollection
from .base import AudioSeparator
import torch
import librosa

class HPSSAudioSeparator(AudioSeparator):
    """Separates the audio into its harmonic and percussive components."""
    def __init__(self, return_harmonic: bool = True, return_percussive: bool = True):
        self._return_harmonic = return_harmonic
        self._return_percussive = return_percussive

    def separate(self, audio: Audio) -> AudioCollection:
        """Implementes the HPSS algorithm. Return may contain the keys 'harmonic' and 'percussive' depending on the initialization."""
        result = {}

        # If not return anything, skip the thing entirely
        if not self._return_harmonic and not self._return_percussive:
            return AudioCollection({})

        audio_data = audio.get_data()
        stft = torch.stft(audio_data, n_fft=2048, return_complex=True).detach().numpy()
        stft_harm, stft_perc = librosa.decompose.hpss(stft)

        if self._return_harmonic:
            y_harm = torch.istft(torch.as_tensor(stft_harm), n_fft = 2048, length=audio_data.size(1))
            result['harmonic'] = Audio(y_harm, sample_rate=audio.sample_rate)

        if self._return_percussive:
            y_perc = torch.istft(torch.as_tensor(stft_perc), n_fft = 2048, length=audio_data.size(1))
            result['percussive'] = Audio(y_perc, sample_rate=audio.sample_rate)

        return AudioCollection(**result)
