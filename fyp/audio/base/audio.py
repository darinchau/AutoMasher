# Contains the definition for the Audio class
# Note: We try to make it such that audio is only an interface thing.
# The actual implementations will switch back to tensors whereever necessary
# Its just safer to have runtime sanity checks for stuff
# Also we enforce a rule: resample and process the audio outside model objects (nn.Module objects)

from __future__ import annotations
import os
import sys
import librosa
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import tempfile
import threading
import torch
import torchaudio
import torchaudio.functional as F
from ...util import download_audio, is_ipython, YouTubeURL, load_audio
from abc import ABC, abstractmethod
from enum import Enum
from math import pi as PI
from PIL import Image
from torch import nn, Tensor
from torchaudio.transforms import TimeStretch
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)


def get_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        raise RuntimeError("You need to install sounddevice to use the play function")


class AudioMode(Enum):
    """The number of channels of an audio"""
    MONO = 1
    STEREO = 2


class Audio:
    """An audio has a special type of tensor with shape=(nchannels, T) and dtype=float32. We have checks and special methods for audios to facilitate audio processing."""

    def sanity_check(self):
        assert self._sample_rate > 0, "Sample rate must be greater than 0"
        assert isinstance(self._sample_rate, int), f"Sample rate must be an int but found {type(self._sample_rate)}"
        assert len(self._data.shape) == 2, f"Audio data must have 2 dimensions, but found {self._data.shape}"
        assert 1 <= self._data.size(0) <= 2, f"Audio data must have 1 or 2 channels, but found {self._data.size(0)} channels"
        assert isinstance(self._data, torch.Tensor), f"Audio data must be a torch.Tensor but found {type(self._data)}"
        assert self._data.dtype == self.dtype(), f"Audio data must have dtype {self.dtype()} but found {self._data.dtype}"
        assert self._inited, f"Make sure to call __init__ from subclasses of Audio ;)"

    def __init__(self, data: Tensor, sample_rate: int):
        """An audio is a special type of audio features - each feature vector has 1 dimensions"""
        assert len(data.shape) == 2, f"Audio data must have 2 dimensions, but found {data.shape}"
        assert data.dtype == self.dtype(), f"Audio data must have dtype {self.dtype()} but found {data.dtype}"
        assert sample_rate > 0, "Sample rate must be greater than 0"

        self._data = data.detach()
        self._sample_rate = sample_rate
        self._inited = True
        self.sanity_check()

        # For playing audio
        self._stop_audio = False
        self._thread = None

    @staticmethod
    def dtype() -> torch.dtype:
        return torch.float32

    @property
    def sample_rate(self) -> int:
        """Returns the sample rate of the audio"""
        self.sanity_check()
        return self._sample_rate

    @property
    def nframes(self) -> int:
        """Returns the number of frames of the audio"""
        self.sanity_check()
        return self._data.size(1)

    @property
    def nchannels(self) -> AudioMode:
        """Returns the number of channels of the audio"""
        self.sanity_check()
        return AudioMode(self._data.size(0))

    @property
    def duration(self) -> float:
        """Returns the duration of the audio in seconds"""
        self.sanity_check()
        return self.nframes / self.sample_rate

    @property
    def volume(self) -> float:
        """Returns the volume of the audio"""
        self.sanity_check()
        return self._data.square().mean().sqrt().item()

    @property
    def data(self) -> Tensor:
        """Returns the audio data tensor"""
        self.sanity_check()
        return self._data

    def clone(self):
        """Returns an identical copy of self"""
        return Audio(self._data.clone(), self._sample_rate)

    def pad(self, target: int, front: bool = False) -> Audio:
        """Returns a new audio with the given number of frames and the same sample rate as self.
        If n < self.nframes, we will trim the audio; if n > self.nframes, we will perform zero padding
        If front is set to true, then operate on the front instead of on the back"""
        length = self.nframes
        if not front:
            if length > target:
                new_data = self._data[:, :target].clone()
            else:
                new_data = torch.nn.functional.pad(self._data, [0, target - length])
        else:
            if length > target:
                new_data = self._data[:, -target:].clone()
            else:
                new_data = torch.nn.functional.pad(self._data, [target - length, 0])

        assert new_data.size(1) == target

        return Audio(new_data, self._sample_rate)

    def to_nchannels(self, target: AudioMode | int) -> Audio:
        """Return self with the correct target. If you use int, you must guarantee the value is 1 or 2, otherwise you get an error"""
        if not isinstance(target, AudioMode) and not isinstance(target, int):
            raise AssertionError(f"nchannels must be an AudioMode but found trying to set nchannels to {target}")
        self.sanity_check()
        if isinstance(target, int) and target != 1 and target != 2:
            raise RuntimeError(f"Told you if you use int you must have target (={target}) to be 1 or 2")
        elif isinstance(target, int):
            target = AudioMode.MONO if target == 1 else AudioMode.STEREO

        match (self.nchannels, target):
            case (AudioMode.MONO, AudioMode.MONO):
                return self.clone()

            case (AudioMode.STEREO, AudioMode.STEREO):
                return self.clone()

            case (AudioMode.MONO, AudioMode.STEREO):
                return self.mix_to_stereo(left_mix=0.)

            case (AudioMode.STEREO, AudioMode.MONO):
                return Audio(self._data.mean(dim=0, keepdim=True), self._sample_rate)

        assert False, "Unreachable"

    def resample(self, target_sr: int, **kwargs) -> Audio:
        """Performs resampling on the audio and returns the mutated self. **kwargs is that for F.resample"""
        self.sanity_check()
        if self._sample_rate == target_sr:
            return self.clone()

        assert target_sr > 0, "Target sample rate must be greater than 0"

        data = F.resample(self._data, self._sample_rate, target_sr, **kwargs)
        return Audio(data, target_sr)

    def slice_frames(self, start_frame: int = 0, end_frame: int = -1) -> Audio:
        """Takes the current audio and splice the audio between start (frames) and end (frames). Returns a new copy.

        Specify end = -1 to take everything alll the way until the end"""
        assert start_frame >= 0
        assert end_frame == -1 or (end_frame > start_frame and end_frame <= self.nframes)
        data = None

        if end_frame == -1:
            data = self._data[:, start_frame:]
        if end_frame > 0:
            data = self._data[:, start_frame:end_frame]

        assert data is not None
        return Audio(data.clone(), self.sample_rate)

    def slice_seconds(self, start: float = 0, end: float = -1) -> Audio:
        """Takes the current audio and splice the audio between start (seconds) and end (seconds). Returns a new copy.

        Specify end = -1 to take everything alll the way until the end"""
        assert start >= 0
        start_frame = int(start * self._sample_rate)
        end_frame = self.nframes if end == -1 else int(end * self._sample_rate)
        assert start_frame < end_frame <= self.nframes
        if end_frame == self.nframes:
            end_frame = -1
        return self.slice_frames(start_frame, end_frame)

    @classmethod
    def download(cls, link: YouTubeURL, port: int | None = None, *, cache_dir: str | None = None) -> Audio:
        """Downloads the audio from the given link. Exposes the port option for greater flexibility"""
        cache_path = None
        if cache_dir is not None:
            cache_path = os.path.join(cache_dir, link.video_id + ".wav")
            if os.path.isfile(cache_path):
                try:
                    return cls.load(cache_path)
                except Exception as e:
                    logger.warning(f"Error loading the cache file: {e}")
                    logger.warning("Loading from youtube instead")
            os.makedirs(cache_dir, exist_ok=True)

        tempdir = tempfile.gettempdir()
        tmp_audio_path = download_audio(link, tempdir, verbose=False, port=port)
        a = cls.load(tmp_audio_path)

        # Attempt to delete the temporary file created
        try:
            os.remove(tmp_audio_path)
        except Exception as e:
            logger.warning(f"Error deleting the temporary file: {e}")
            logger.warning("You might want to delete the temporary file manually")
            pass

        if cache_path is not None:
            a.save(cache_path)
        return a

    @classmethod
    def load(cls, fpath: str, *, cache_dir: str | None = None) -> Audio:
        """
        Loads an audio file from a given file path, and returns the audio as a tensor. If fpath is a YouTubeURL (a subclass of str),
        then it will download the audio from youtube and return the audio as a tensor.
        """
        try:
            fpath = YouTubeURL(fpath)
        except Exception as e:
            pass

        if isinstance(fpath, YouTubeURL):
            logger.warning(f"The provided fpath is a YouTube URL. Use Audio.download instead. This will be removed in the future")
            return cls.download(fpath, cache_dir=cache_dir)

        wav, sr = load_audio(fpath)
        return cls(wav, sr)

    def play(self, blocking: bool = False, info: list[tuple[str, float]] | None = None):
        """Plays audio in a separate thread. Use the stop() function or wait() function to let the audio stop playing.
        info is a list of stuff you want to print. Each element is a tuple of (str, float) where the float is the time in seconds
        if progress is true, then display a nice little bar that shows the progress of the audio"""
        sd = get_sounddevice()

        def _play(sound, sr, nc, stop_event):
            event = threading.Event()
            x = 0

            def callback(outdata, frames, time, status):
                nonlocal x
                sound_ = sound[x:x+frames]
                x = x + frames

                # Print the info if there are anything
                while info and x/sr > info[0][1]:
                    info_str = info[0][0].ljust(longest_info)
                    print("\r" + info_str, end="")
                    info.pop(0)

                if stop_event():
                    raise sd.CallbackStop

                # Push the audio
                if len(outdata) > len(sound_):
                    outdata[:len(sound_)] = sound_
                    outdata[len(sound_):] = np.zeros((len(outdata) - len(sound_), 1))
                    raise sd.CallbackStop
                else:
                    outdata[:] = sound_[:]

            stream = sd.OutputStream(samplerate=sr, channels=nc, callback=callback, blocksize=1024, finished_callback=event.set)
            with stream:
                event.wait()
                self._stop_audio = True

        if info is not None:
            blocking = True  # Otherwise jupyter notebook will behave weirdly
        else:
            if is_ipython():
                from IPython.display import Audio as IPAudio
                return IPAudio(self.numpy(), rate=self.sample_rate)
            info = []
        info = sorted(info, key=lambda x: x[1])
        longest_info = max([len(x[0]) for x in info]) if info else 0
        sound = self._data.mean(dim=0).unsqueeze(1).detach().cpu().numpy()
        self._thread = threading.Thread(target=_play, args=(sound, self.sample_rate, self.nchannels.value, lambda: self._stop_audio))
        self._stop_audio = False
        self._thread.start()
        if blocking:
            self.wait()

    def stop(self):
        """Attempts to stop the audio that's currently playing. If the audio is not playing, this does nothing."""
        self._stop_audio = True
        self.wait()

    def wait(self):
        """Wait for the audio to stop playing. If the audio is not playing, this does nothing."""
        if self._thread is None:
            return

        if not self._thread.is_alive():
            return

        self._thread.join()
        self._thread = None
        self._stop_audio = False  # Reset the state

    def save(self, fpath: str):
        """Saves the audio at the provided file path. WAV is (almost certainly) guaranteed to work"""
        self.sanity_check()
        data = self._data
        if fpath.endswith(".mp3"):
            try:
                from pydub import AudioSegment
            except ImportError:
                raise RuntimeError("You need to install pydub to save the audio as mp3")
            with tempfile.TemporaryDirectory() as tempdir:
                temp_fpath = os.path.join(tempdir, "temp.wav")
                torchaudio.save(temp_fpath, data, sample_rate=int(self._sample_rate))
                song = AudioSegment.from_wav(temp_fpath)
                song.export(fpath, format="mp3")
            return
        try:
            torchaudio.save(fpath, data, sample_rate=int(self._sample_rate))
            return
        except (ValueError, RuntimeError) as e:  # Seems like torchaudio changed the error type to runtime error in 2.2?
            # or the file path is invalid
            raise RuntimeError(f"Error saving the audio: {e} - {fpath}")

    def plot(self, keep_sr: bool = False):
        """Plots the audio as a waveform. If keep_sr is true, then we plot the audio with the original sample rate. Otherwise we plot the audio with a lower sample rate to save time."""
        audio = self if keep_sr else self.resample(1000)

        waveform = audio.numpy(keep_dims=True)

        num_channels = audio.nchannels.value
        num_frames = audio.nframes

        time_axis = torch.arange(0, num_frames) / audio.sample_rate

        figure, axes = plt.subplots()
        if num_channels == 1:
            axes.plot(time_axis, waveform[0], linewidth=1)
        else:
            axes.plot(time_axis, np.abs(waveform[0]), linewidth=1)
            axes.plot(time_axis, -np.abs(waveform[1]), linewidth=1)
        axes.grid(True)
        plt.show(block=False)

    def join(self, other: Audio) -> Audio:
        """Joins two audio back to back. Two audios must have the same sample rate.

        Returns the new audio with the same sample rate and nframes equal to the sum of both"""
        assert self.sample_rate == other.sample_rate

        nchannels = 1 if self.nchannels == other.nchannels == AudioMode.MONO else 2
        if nchannels == 2:
            newself = self.to_nchannels(AudioMode.STEREO)
            newother = other.to_nchannels(AudioMode.STEREO)
        else:
            newself = self
            newother = other
        newdata = torch.zeros((nchannels, self.nframes + other.nframes), dtype=torch.float32)
        newdata[:, :self.nframes] = newself._data
        newdata[:, self.nframes:] = newother._data
        return Audio(newdata, self.sample_rate)

    def numpy(self, keep_dims: bool = False):
        """Returns the 1D numpy audio format of the audio. If you insist you want the 2D audio, put keep_dims = True"""
        self.sanity_check()
        if keep_dims:
            return self._data.detach().cpu().numpy()
        data = self._data
        if self._data.size(0) == 2:
            data = data.mean(dim=0)
        else:
            data = data[0]
        try:
            return data.numpy()
        except Exception as e:
            return data.detach().cpu().numpy()

    def __repr__(self):
        """
        Prints out the following information about the audio:
        Duration, Sample rate, Num channels, Num frames
        """
        return f"(Audio)\nDuration:\t{self.duration:5f}\nSample Rate:\t{self.sample_rate}\nChannels:\t{self.nchannels}\nNum frames:\t{self.nframes}"

    def mix_to_stereo(self, left_mix: float = 0.) -> Audio:
        """Mix a mono audio to stereo audio. The left_mix is the amount of left pan of the audio.
        Must be -1 <= left_mix <= 1. If -1 then the audio is completely on the left, if 1 then the audio is completely on the right"""
        if self.nchannels == AudioMode.STEREO:
            audio = self.to_nchannels(1)
        else:
            audio = self

        if left_mix < -1 or left_mix > 1:
            raise ValueError("left_mix must be between -1 and 1")

        left_mix = left_mix / 2 + 0.5
        right_mix = 1 - left_mix
        mixer = torch.tensor([[left_mix], [right_mix]], device=audio._data.device)
        return Audio(audio._data * mixer, audio.sample_rate)

    def change_speed(self, speed: float, n_fft: int = 512, win_length: int | None = None, hop_length: int | None = None, window: Tensor | None = None) -> Audio:
        if speed == 1:
            return self.clone()
        if speed < 0:
            data = torch.flip(self._data, dims=[1])
            speed = -speed
        else:
            data = self._data

        audio = data
        audio_length = audio.size(-1)

        if hop_length is None:
            hop_length = n_fft // 4
        if win_length is None:
            win_length = n_fft
        if window is None:
            window = torch.hann_window(window_length=win_length, device=audio.device)

        # Apply stft
        spectrogram = torch.stft(
            input=audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Stretch the audio without modifying pitch - phase vocoder
        phase_advance = torch.linspace(0, PI * hop_length, spectrogram.shape[-2], device=spectrogram.device)[..., None]
        stretched_spectrogram = F.phase_vocoder(spectrogram, speed, phase_advance)
        len_stretch = int(round(audio_length / speed))

        # Inverse the stft
        waveform_stretch = torch.istft(
            stretched_spectrogram,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            length=len_stretch
        )

        return Audio(waveform_stretch, self.sample_rate)

    def mix(self, others: list[Audio]):
        """Mixes the current audio with other audio. The audio must have the same sample rate"""
        audios = [self] + others
        for audio in audios:
            assert audio.sample_rate == self.sample_rate, "All audios must have the same sample rate"
        data = torch.stack([audio._data for audio in audios], dim=0).mean(dim=0)
        return Audio(data, self.sample_rate)
