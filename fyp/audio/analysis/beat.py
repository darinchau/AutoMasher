# Provides an algorithm for bpm detection from librosa.
from __future__ import annotations
import os
from typing import Any
from .base import OnsetFeatures
from dataclasses import dataclass
import json
from typing import Callable
import numpy as np
from .. import DemucsCollection
from ..separation import demucs_separate
from ...model import beats_inference as inference
import warnings
import requests
from typing import Any
from ...audio import Audio, AudioMode
import librosa
import soundfile as sf
import random
from contextlib import contextmanager
import json
import torch
from ...util import YouTubeURL
from ...util.exception import DeadBeatKernel
import typing

if typing.TYPE_CHECKING:
    from ..dataset import SongDataset

BEAT_DATASET_KEY = "beats"


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
        _beats = OnsetFeatures(duration, np.array(beats, dtype=np.float64))
        _downbeats = OnsetFeatures(duration, np.array(downbeats, dtype=np.float64))
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


@contextmanager
# This creates a temp file with a temp file name and cleans it up at the end
def get_temp_file(extension: str):
    def get_random_string():
        s = "temp"
        for _ in range(6):
            s += "qwertyuiopasdfghjklzxcvbnm"[random.randint(0, 25)]
        return s

    def get_unique_filename(name):
        # Check if the file already exists
        if not os.path.isfile(f"{name}.{extension}"):
            return f"{name}.{extension}"

        # If the file already exists, add a number to the end of the filename
        i = 1
        while True:
            new_name = f"{name} ({i}).{extension}"
            if not os.path.isfile(new_name):
                return new_name
            i += 1
    fn = get_unique_filename(get_random_string())
    try:
        with open(fn, 'w+b'):
            pass
        yield fn
    finally:
        if os.path.isfile(fn):
            os.remove(fn)


def analyse_beat_transformer(audio: Audio | None = None,
                             dataset: SongDataset | None = None,
                             url: YouTubeURL | None = None,
                             parts: DemucsCollection | None = None,
                             backend: typing.Literal["demucs", "spleeter"] = "demucs",
                             backend_url: str | None = None,
                             do_normalization: bool = False,
                             model_path: str = "./resources/ckpts/beat_transformer.pt",
                             use_cache: bool = True,
                             device: torch.device | None = None,
                             use_loaded_model: bool = True) -> BeatAnalysisResult:
    """Beat transformer but runs locally using Demucs and some uh workarounds

    The duration will be the duration of the audio if audio is provided, otherwise the duration of the parts.

    Args:

        audio (Audio | None): The audio to analyse. If parts is not None, this is ignored. Defaults to None.
        dataset (SongDataset | None): The dataset to use for caching. Defaults to None.
        url (YouTubeURL | None): The URL of the audio to analyse. Defaults to None.
        parts (DemucsCollection | None): The parts of the audio to analyse. Defaults to None.
        backend (typing.Literal["demucs", "spleeter"]): The backend to use. Defaults to "demucs".
        backend_url (str | None): The URL of the beat transformer backend. Defaults to None.
        do_normalization (bool): Whether to normalize the downbeats to the closest beat. Defaults to False.
        model_path (str): The path to the beat transformer model. Defaults to "./resources/ckpts/beat_transformer.pt".
        use_cache (bool): Whether to use the cache - if False, will not save to or load from cache. Defaults to True.
        use_loaded_model (bool): Whether to use the loaded model. Defaults to True.
        """
    if use_cache and dataset is not None:
        dataset._register("beats", "{video_id}.beats")
        if url is not None and not url.is_placeholder and dataset.has_path("beats", url):
            return BeatAnalysisResult.load(dataset.get_path("beats", url))

    if audio is None and parts is None:
        raise ValueError("Either audio or parts must be provided")

    if parts is None and audio is not None and backend == "demucs":
        parts = demucs_separate(audio)

    if parts is not None:
        if audio is None:
            duration = parts.get_duration()
        else:
            duration = audio.duration  # Unnecessary line of code but here to apease the linter
        parts_dict = {
            "vocals": parts.vocals,
            "drums": parts.drums,
            "bass": parts.bass,
            "other": parts.other,
        }

        assert parts is not None
        assert duration > 0

        parts_dict = {k: v.resample(44100).to_nchannels(2).numpy(keep_dims=True).T for k, v in parts_dict.items()}
        parts_dict['piano'] = np.zeros_like(parts_dict['vocals'])

        with warnings.catch_warnings():
            beat_frames, downbeat_frames = inference(parts_dict, model_path=model_path, use_loaded_model=use_loaded_model, device=device)
    else:
        assert backend == "spleeter"
        assert audio is not None, "Audio must be provided if using spleeter"
        assert backend_url is not None, "Backend URL must be provided if using spleeter"
        duration = audio.duration
        audio_ = audio.resample(44100).to_nchannels(AudioMode.STEREO)

        with get_temp_file("wav") as fn:
            with sf.SoundFile(fn, 'w', samplerate=audio_.sample_rate, channels=1, format='WAV') as f:
                audio_ = audio_.numpy()
                f.write(audio_)

            with open(fn, 'rb') as f:
                wav_bytes = f.read()

        if backend_url[-1] == "/":
            backend_url = backend_url[:-1]

        try:
            preflight = requests.get(backend_url + "/alive")
        except requests.exceptions.ConnectionError:
            raise DeadBeatKernel("Beat transformer backend is dead")

        fail_reason = None
        if preflight.status_code != 200:
            fail_reason = f"Beat transformer preflight failed with status code {preflight.status_code}. Have you forgot to activate the backer end API? Falling back to demucs for now"
        elif isinstance(preflight.json(), dict) and preflight.json().get("alive") != "true":
            fail_reason = "Wrong response from beat transformer. Falling back to demucs for now"

        if fail_reason is not None:
            raise DeadBeatKernel(fail_reason)

        try:
            r = requests.post(backend_url + "/beat", data=wav_bytes)
        except requests.exceptions.ConnectionError:
            raise DeadBeatKernel("Beat transformer backend is dead")

        if r.status_code != 200:
            raise DeadBeatKernel(f"Beat transformer failed with status code {r.status_code}. Have you forgot to activate the backer end API?")

        data = r.json()
        downbeat_frames: list[float] = [int(x) / 44100 for x in data["downbeat_frames"]]
        beat_frames: list[float] = [int(x) / 44100 for x in data["beat_frames"]]

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
    if use_cache and dataset is not None:
        result.save(dataset.get_path("beats", url))
    return result
