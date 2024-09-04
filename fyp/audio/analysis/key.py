from typing import Any
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
import librosa
from ...audio.analysis.base import KeyAnalysisResult
from ...audio import Audio
from ...audio.separation import HPSSAudioSeparator
import torch
from typing import Callable
from .chroma import chroma_cqt, ChromaFunction

def _get_major_profile():
    """The major profile from Krumhansl-Schmuckler key-finding algorithm"""
    return [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]

def _get_minor_profile():
    """The minor profile from Krumhansl-Schmuckler key-finding algorithm"""
    return [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def _rotate_array(array: np.ndarray, i: int):
    """Rotate an array by i spaces clockwise"""
    return np.concatenate([array[i:], array[:i]])

def analyse_key_center_chroma(audio: Audio, f_or_chromagram: ChromaFunction | NDArray[np.float32], hop = 512) -> KeyAnalysisResult:
    """The base function to calculate a key center from a chromograph function."""
    if callable(f_or_chromagram):
        # Use HPSS to extract the harmonic component
        harmonic_component = HPSSAudioSeparator(return_percussive=False).separate(audio)['harmonic']
        chromograph = f_or_chromagram(harmonic_component, hop)
    else:
        chromograph = f_or_chromagram

    # Run the Krumhansl-Schmuckler key-finding algorithm
    assert chromograph.shape[0] == 12
    chroma = np.sum(chromograph, axis = 1)

    maj_profile = _get_major_profile()
    min_profile = _get_minor_profile()

    # Builds the key cprrelations - i.e. the correlation of the pitch for each key
    correlations = [0.] * 24
    for i in range(12):
        key_test = _rotate_array(chroma, i)
        correlations[i] = float(np.corrcoef(maj_profile, key_test)[1, 0])
        correlations[12 + i] = float(np.corrcoef(min_profile, key_test)[1, 0])
    return KeyAnalysisResult(tuple(correlations), chromograph)

def analyse_key_center(audio: Audio, hop = 512) -> KeyAnalysisResult:
    """Uses the librosa chromograph along with the Krumhansl-Schmuckler key-finding algorithm."""
    return analyse_key_center_chroma(audio, chroma_cqt, hop)
