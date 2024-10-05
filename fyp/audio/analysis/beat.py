### Provides an algorithm for bpm detection from librosa.

import os
from typing import Any
from ...audio import Audio
from .base import BeatAnalysisResult
from typing import Callable
import numpy as np
from .. import DemucsCollection
from ..separation import DemucsAudioSeparator
from ...model import beats_inference as inference
import warnings

def analyse_beat_transformer(audio: Audio | None = None,
                                parts: DemucsCollection | dict[str, Audio] | None = None,
                                separator: DemucsAudioSeparator | None = None,
                                do_normalization: bool = False,
                                model_path: str = "./resources/ckpts/beat_transformer.pt",
                                use_loaded_model: bool = True) -> BeatAnalysisResult:
    """Beat transformer but runs locally using Demucs and some uh workarounds

    Args:
        audio (Audio, optional): The audio to use. If not provided, parts must be provided. Defaults to None.
        parts (AudioCollection | Callable[[], AudioCollection], optional): The parts to use.
            If not provided, audio must be provided. If callable, assumes a lazily-evaluated value. Defaults to None.
        separator (AudioSeparator, optional): The separator to use. Defaults to None.
        do_normalization (bool, optional): Whether to normalize the downbeat frames to the closest beat frame. Defaults to False.
        cache_path (str, optional): The path to save the cache. Defaults to None.
        model_path (str, optional): The path to the model. Defaults to "./resources/ckpts/beat_transformer.pt"."""
    # Handle audio/parts case
    duration = -1.
    if audio is None and parts is None:
        raise ValueError("Either audio or parts must be provided")

    if parts is None and audio is not None:
        demucs = DemucsAudioSeparator() if separator is None else separator
        parts = demucs.separate(audio)
        duration = audio.duration
    elif parts is not None:
        if isinstance(parts, DemucsCollection):
            duration = parts.get_duration()
            parts = {
                "vocals": parts.vocals,
                "drums": parts.drums,
                "bass": parts.bass,
                "other": parts.other,
            }
        elif isinstance(parts, dict):
            duration = list(parts.values())[0].duration
        else:
            raise ValueError("Unknown parts type")

    # Resample as needed
    assert parts is not None
    assert duration > 0

    parts = {k: v.resample(44100).to_nchannels(2) for k, v in parts.items()}

    # Detect whether the parts is Demucs or Spleeter or something else
    assert parts is not None
    key_set = set(parts.keys())
    expected_key_set = {"vocals", "piano", "bass", "drums", "other"}
    parts_dict = {}
    if key_set == expected_key_set:
        # Spleeter audio
        for k in key_set:
            parts_dict[k] = parts[k].numpy(keep_dims=True).T
    elif key_set == {"vocals", "bass", "drums", "other"}:
        # Demucs audio
        for k in key_set:
            parts_dict[k] = parts[k].numpy(keep_dims=True).T
        parts_dict['piano'] = np.zeros_like(parts_dict['vocals'])
    else:
        raise ValueError("Unknown audio type - found the following keys: " + str(key_set))

    assert set(parts_dict.keys()) == expected_key_set
    with warnings.catch_warnings(action="ignore"):
        beat_frames, downbeat_frames = inference(parts_dict, model_path=model_path, use_loaded_model=use_loaded_model)

    # Empirically, the downbeat frames are not always the same as the beat frames
    # So we need to map the downbeat frames to the closest beat frame
    if do_normalization:
        for i, fr in enumerate(downbeat_frames):
            closest_beat = min(beat_frames, key=lambda x: abs(x - fr))
            downbeat_frames[i] = closest_beat

    # Trim the beat frames to the length of the audio
    beat_frames = [x for x in beat_frames if x < duration]
    downbeat_frames = [x for x in downbeat_frames if x < duration]

    result = BeatAnalysisResult.from_data(duration, beat_frames, downbeat_frames)
    return result
