# Functions to calculate the chroma features of the audio signal. The input must be an audio and output a numpy array of shape (12, T)
# Includes a manual implementation of the chroma features. The output is slightly different in the sense it seems much more quiet
from torch import Tensor
import librosa
from ...audio import Audio
import numpy as np
from numpy.typing import NDArray
import math
from typing import Callable

# All chroma functions MUST accept the keyword argument hop, and produce a numpy array of shape (12, T)
# where T = audio.nframes // hop + 1 is the number of time frames
ChromaFunction = Callable[[Audio, int], NDArray[np.float32]]


def chroma_cqt(audio: Audio, hop: int = 512, *, bins_per_octave=24, fmin=27.5, **kwargs):
    """Calculates the chroma features of the audio signal. The input must be an audio and output a numpy array of shape (12, T)"""
    audio_array = audio.numpy()
    chroma = librosa.feature.chroma_cqt(y=audio_array, sr=audio.sample_rate, hop_length=hop, bins_per_octave=bins_per_octave, fmin=fmin, **kwargs)
    assert chroma.shape[0] == 12
    return chroma.astype(np.float32)


def chroma_stft(audio: Audio, hop: int = 512, **kwargs):
    """Calculates the chroma features of the audio signal. The input must be an audio and output a numpy array of shape (12, T)"""
    audio_array = audio.numpy()
    chroma = librosa.feature.chroma_stft(y=audio_array, sr=audio.sample_rate, hop_length=hop, **kwargs)
    assert chroma.shape[0] == 12
    return chroma.astype(np.float32)


def chroma_stft_nfft(audio: Audio, n_fft: int, hop: int = 512, **kwargs):
    audio_array = audio.numpy()
    s = np.abs(librosa.stft(audio_array, n_fft=n_fft))**2
    chroma = librosa.feature.chroma_stft(S=s, sr=audio.sample_rate, hop_length=hop, **kwargs)
    assert chroma.shape[0] == 12
    return chroma.astype(np.float32)


def chroma_cens(audio: Audio, hop: int = 512, *, bins_per_octave=24, fmin=27.5, **kwargs):
    """Calculates the chroma features of the audio signal. The input must be an audio and output a numpy array of shape (12, T)"""
    audio_array = audio.numpy()
    chroma = librosa.feature.chroma_cens(y=audio_array, sr=audio.sample_rate, hop_length=hop, bins_per_octave=bins_per_octave, fmin=fmin, **kwargs)
    assert chroma.shape[0] == 12
    return chroma.astype(np.float32)
