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
from ...util.combine import get_video_id

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

def analyse_beat_transformer(audio: Audio, *, 
                             url: str | None = None, 
                             with_fallback: Callable[[Audio], BeatAnalysisResult] | None = None, 
                             do_normalization: bool = False,
                             verbose: bool = False, cache_path: str | None = None) -> BeatAnalysisResult:
    """Performs beat transformer using a separate microservice"""

    if url is None:
        raise ValueError("URL must be provided")

    def calculate_beats():
        request_fail_msg = "Beat transformer failed with status code {}. Have you forgot to activate the backer end API?"
        def try_fallback(e: Exception):
            if with_fallback is not None:
                print("Beat transformer failed. Using fallback...")
                return with_fallback(audio)
            else:
                raise e

        # Use this colab notebook (just set runtime to gpu and run all) to play with this function
        # Ensure 44100 sample rate and stuff
        audio_ = audio.resample(44100).to_nchannels(AudioMode.STEREO)

        with get_temp_file("wav") as fn:
            with sf.SoundFile(fn, 'w', samplerate=audio_.sample_rate, channels=1, format='WAV') as f:
                audio_ = audio_.numpy()
                f.write(audio_)

            with open(fn, 'rb') as f:
                wav_bytes = f.read()

        # Use a preflight response to check if the server is alive (as a sanity check only)
        preflight = requests.get((url[:-1] if url[-1] == "/" else url) + "/alive")
        if preflight.status_code != 200:
            return try_fallback(RuntimeError(request_fail_msg.format(preflight.status_code)))
        
        preflight_response = preflight.json()
        if preflight_response["alive"] != "true":
            try_fallback(RuntimeError("Wrong response"))
        
        if verbose:
            print("Passed preflight check for beat transformer :D")

        # Do the actual request. This will take approximately 3 ages
        r = requests.post(url + "/beat", data=wav_bytes)
        if r.status_code != 200:
            print(r.text)
            try_fallback(RuntimeError(request_fail_msg.format(r.status_code)))
        
        data = r.json()
        
        downbeat_frames_: list[int] = data["downbeat_frames"]
        beat_frames_: list[int] = data["beat_frames"]
        downbeat_frames: list[float] = [int(x) / 44100 for x in downbeat_frames_]
        beat_frames: list[float] = [int(x) / 44100 for x in beat_frames_]

        # Empirically, the downbeat frames are not always the same as the beat frames
        # So we need to map the downbeat frames to the closest beat frame
        if do_normalization:
            for i, fr in enumerate(downbeat_frames):
                closest_beat = min(beat_frames, key=lambda x: abs(x - fr))
                downbeat_frames[i] = closest_beat

        # Trim the beat frames to the length of the audio
        beat_frames = [x for x in beat_frames if x < audio.duration]
        downbeat_frames = [x for x in downbeat_frames if x < audio.duration]

        result = BeatAnalysisResult(audio.duration, beat_frames, downbeat_frames)
        return result

    if cache_path is not None and os.path.isfile(cache_path):
        return BeatAnalysisResult.load(cache_path)
    
    result = calculate_beats()
    if cache_path is not None:
        result.save(cache_path)
    return result

def analyse_beat_transformer_local(audio: Audio | None = None, 
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
