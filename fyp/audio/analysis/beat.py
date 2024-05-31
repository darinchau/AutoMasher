### Provides an algorithm for bpm detection from librosa.

import os
from typing import Any
from ...audio import Audio, AudioMode
from .base import BeatAnalysisResult
import requests
from typing import Any
from ...audio import Audio, AudioMode
import librosa
from typing import Callable
import numpy as np
import soundfile as sf
import random
from contextlib import contextmanager
from .. import AudioCollection
from ..separation import DemucsAudioSeparator, AudioSeparator
from librosa.core import stft
from scipy.signal.windows import hann
from ...model.beat_transformer import inference    
import warnings

def analyse_beat_transformer(audio: Audio | None = None, 
                                   parts: AudioCollection | Callable[[], AudioCollection] | None = None, 
                                   separator: AudioSeparator | None = None,
                                    do_normalization: bool = False, cache_path: str | None = None,
                                    model_path: str = "../../resources/ckpts/beat_transformer.pt"):
    """Beat transformer but runs locally using Demucs and some uh workarounds
    
    Args:
        audio (Audio, optional): The audio to use. If not provided, parts must be provided. Defaults to None.
        parts (AudioCollection | Callable[[], AudioCollection], optional): The parts to use. 
            If not provided, audio must be provided. If callable, assumes a lazily-evaluated value. Defaults to None.
        separator (AudioSeparator, optional): The separator to use. Defaults to None.
        do_normalization (bool, optional): Whether to normalize the downbeat frames to the closest beat frame. Defaults to False.
        cache_path (str, optional): The path to save the cache. Defaults to None.
        model_path (str, optional): The path to the model. Defaults to "../../resources/ckpts/beat_transformer.pt"."""    
    def calculate_beats():
        # Handle audio/parts case
        duration = -1.
        nonlocal parts
        if audio is None and parts is None:
            raise ValueError("Either audio or parts must be provided")
        
        if parts is None and audio is not None:
            demucs = DemucsAudioSeparator() if separator is None else separator
            parts = demucs.separate_audio(audio)
            duration = audio.duration
        elif parts is not None:
            if callable(parts):
                parts = parts()
            duration = parts.get_duration()

        # Resample as needed
        assert parts is not None
        assert duration > 0
        
        parts = parts.map(lambda x: x.resample(44100).to_nchannels(AudioMode.STEREO))

        # Detect whether the parts is Demucs or Spleeter or something else
        assert parts is not None
        key_set = set(parts.keys())
        expected_key_set = {"vocals", "piano", "bass", "drums", "other"}
        parts_dict = {}
        if key_set == expected_key_set:
            # Spleeter audio
            for k in key_set:
                parts_dict[k] = parts[k]._data.numpy().T
        elif key_set == {"vocals", "bass", "drums", "other"}:
            # Demucs audio
            for k in key_set:
                parts_dict[k] = parts[k]._data.numpy().T
            parts_dict['piano'] = np.zeros_like(parts_dict['vocals'])
        else:
            raise ValueError("Unknown audio type - found the following keys: " + str(key_set))
        
        assert set(parts_dict.keys()) == expected_key_set
        with warnings.catch_warnings(action="ignore"):
            beat_frames, downbeat_frames = inference(parts_dict, model_path=model_path)
        
        # Empirically, the downbeat frames are not always the same as the beat frames
        # So we need to map the downbeat frames to the closest beat frame
        if do_normalization:
            for i, fr in enumerate(downbeat_frames):
                closest_beat = min(beat_frames, key=lambda x: abs(x - fr))
                downbeat_frames[i] = closest_beat

        # Trim the beat frames to the length of the audio
        beat_frames = [x for x in beat_frames if x < duration]
        downbeat_frames = [x for x in downbeat_frames if x < duration]

        result = BeatAnalysisResult(duration, beat_frames, downbeat_frames)
        return result

    if cache_path is not None and os.path.isfile(cache_path):
        return BeatAnalysisResult.load(cache_path)
    
    result = calculate_beats()
    if cache_path is not None:
        result.save(cache_path)
    return result
