### Provides an algorithm for bpm detection from librosa.
from __future__ import annotations
import os
from typing import Any
from ...audio import Audio
from .base import OnsetFeatures
from dataclasses import dataclass
import json
from typing import Callable
import numpy as np
from .. import DemucsCollection
from ..separation import DemucsAudioSeparator
from ...model import beats_inference as inference
import warnings

@dataclass(frozen=True)
class BeatAnalysisResult:
    """A class that represents the result of a beat analysis."""
    _beats: OnsetFeatures
    _downbeats: OnsetFeatures

    @property
    def duration(self):
        return self._beats.duration

    @property
    def beats(self):
        return self._beats.onsets

    @property
    def downbeats(self):
        return self._downbeats.onsets

    @property
    def tempo(self):
        return self._beats.tempo

    @property
    def nbars(self):
        """Returns the number of bars in the song"""
        return self._downbeats.nsegments

    @classmethod
    def from_data(cls, duration: float, beats: list[float], downbeats: list[float]):
        _beats = OnsetFeatures(duration, np.array(beats, dtype=np.float32))
        _downbeats = OnsetFeatures(duration, np.array(downbeats, dtype=np.float32))
        return cls(_beats, _downbeats)

    def slice_seconds(self, start: float, end: float) -> BeatAnalysisResult:
        """Slice the beat analysis result by seconds. includes start and excludes end"""
        return BeatAnalysisResult(
            self._beats.slice_seconds(start, end),
            self._downbeats.slice_seconds(start, end)
        )

    def change_speed(self, speed: float) -> BeatAnalysisResult:
        """Change the speed of the beat analysis result"""
        return BeatAnalysisResult(
            self._beats.change_speed(speed),
            self._downbeats.change_speed(speed)
        )

    def save(self, path: str):
        json_dict = {
            "duration": self.duration,
            "beats": self.beats.tolist(),
            "downbeats": self.downbeats.tolist()
        }

        with open(path, "w") as f:
            json.dump(json_dict, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_data(data["duration"], data["beats"], data["downbeats"])

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
    with warnings.catch_warnings():
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
