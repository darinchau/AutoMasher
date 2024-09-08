## Contains the definition for the Audio class
## Note: We try to make it such that audio is only an interface thing.
## The actual implementations will switch back to tensors whereever necessary
## Its just safer to have runtime sanity checks for stuff
## Also we enforce a rule: resample and process the audio outside model objects (nn.Module objects)

from __future__ import annotations
import os
import random
import torch
import torchaudio
import torchaudio.functional as F
from torch import nn, Tensor
import numpy as np
import threading
from enum import Enum
from typing import final
from os import path
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from .time_series import TimeSeries
from typing import Callable
import typing
from torchaudio.transforms import TimeStretch
import librosa
from ...util import is_ipython
from ...util.download import download_audio
import tempfile

def get_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        raise RuntimeError("You need to install sounddevice to use the play function")

if typing.TYPE_CHECKING:
    from ..manipulation.base import AudioTransform

class AudioMode(Enum):
    """The number of channels of an audio"""
    MONO = 1
    STEREO = 2

class Audio(TimeSeries):
    """An audio has a special type of tensor with shape=(nchannels, T) and dtype=float32. We have checks and special methods for audios to facilitate audio processing."""
    @final
    def sanity_check(self):
        # Dont call any properties here as it might lead to infinite recursion
        assert self._sample_rate > 0
        assert len(self._data.shape) == 2
        assert 1 <= self._data.size(0) <= 2
        assert self._data.dtype == torch.float32

    def __init__(self, data: Tensor, sample_rate: int):
        """Data should be moved in - i.e. holding additional references of
        data outside of the class and modifying it could lead to assertion errors"""
        self._data = data.detach()
        self._sample_rate = sample_rate
        self.sanity_check()

        # For playing audio
        self._stop_audio = False
        self._thread = None

    @property
    def sample_rate(self) -> int:
        """The sample rate of the audio. Use the resample() method to change the sample rate"""
        self.sanity_check()
        return self._sample_rate

    @property
    def nchannels(self) -> AudioMode:
        """Number of channels of the audio. Returns an AudioMode enum"""
        self.sanity_check()
        return AudioMode.MONO if self._data.size(0) == 1 else AudioMode.STEREO

    @property
    def duration(self) -> float:
        """Return the duration (s) of the audio"""
        self.sanity_check()
        return self._data.size(1)/self._sample_rate

    @property
    def device(self) -> torch.device:
        """Accessor for the device that the data is sitting on. Currently it is guaranteed to be cpu but we do not make this guarantee in the future"""
        self.sanity_check()
        return self._data.device

    @property
    def nframes(self):
        """The length of the audio in terms of number of frames."""
        return int(self._data.size(1))

    def get_data(self):
        """Returns a copy of the underlying audio data of the Audio object."""
        self.sanity_check()
        return self._data.clone()

    def clone(self):
        """Returns an identical copy of self"""
        return Audio(self.get_data(), self._sample_rate)

    def set_volume(self, volume: float) -> Audio:
        current_volume = self._data.square().mean().sqrt()
        return Audio(self._data * volume / current_volume, self._sample_rate)

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
                return Audio(self._data.mean(dim = 0, keepdim=True), self._sample_rate)

        assert False, "Unreachable"

    def resample(self, target_sr: int, **kwargs) -> Audio:
        """Performs resampling on the audio and returns the mutated self. **kwargs is that for F.resample"""
        self.sanity_check()
        if self._sample_rate == target_sr:
            return self.clone()

        data = F.resample(self._data, self._sample_rate, target_sr, **kwargs)
        return Audio(data, target_sr)

    def slice(self, start_frame: int = 0, end_frame: int = -1) -> Audio:
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
        return self.slice(start_frame, end_frame)

    @classmethod
    def load(cls, fpath: str, *, cache_path: str | None = None) -> Audio:
        """
        Loads an audio file from a given file path, and returns the audio as a tensor.
        Output shape: (channels, N) where N = duration (seconds) x sample rate (hz)

        if channels == 1, then take the mean across the audio
        if channels == audio channels, then leave it alone
        otherwise we will take the mean and duplicate the tensor until we get the desired number of channels

        Explicitly specify cache = False if you do not want to cache the audio when downloaded from youtube
        """
        if cache_path is not None and os.path.isfile(cache_path):
            try:
                return cls.load(cache_path)
            except Exception as e:
                pass

        # Load from youtube if the file path is a youtube url
        if fpath.startswith("http") and "youtu" in fpath:
            tempdir = tempfile.gettempdir()
            tmp_audio_path = download_audio(fpath, tempdir, verbose=False)
            a = cls.load(tmp_audio_path)

            # Attempt to delete the temporary file created
            try:
                os.remove(tmp_audio_path)
            except Exception as e:
                pass

            if cache_path is not None:
                a.save(cache_path)
            return a

        try:
            wav, sr = torchaudio.load(fpath)
        except Exception as e:
            wav, sr = librosa.load(fpath, mono=False)
            sr = int(sr)
            if len(wav.shape) > 1:
                wav = wav.reshape(-1, wav.shape[-1])
            else:
                wav = wav.reshape(1, -1)

            wav = torch.tensor(wav).float()

        if wav.dtype != torch.float32:
            wav = wav.to(dtype = torch.float32)
        return cls(wav, sr)

    @classmethod
    def record(cls, duration: float, sample_rate: int, verbose = True):
        """Records audio for a given duration and sample rate. Returns the audio as a tensor"""
        sd = get_sounddevice()
        channels = 1
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
        if verbose:
            print("Recording")
        sd.wait()
        if verbose:
            print("Finished recording")
        recording = torch.as_tensor(recording, dtype = torch.float32).flatten().unsqueeze(0)
        return cls(recording, sample_rate)

    def play(self, blocking: bool = False,
             callback_fn: Callable[[float], None] | None = None,
             stop_callback_fn: Callable[[], None] | None = None,
             info: list[tuple[str, float]] | None = None):
        """Plays audio in a separate thread. Use the stop() function or wait() function to let the audio stop playing.
        info is a list of stuff you want to print. Each element is a tuple of (str, float) where the float is the time in seconds
        callback fn should take a float t which will be called every time an audio chunk is processed. The float will be the current
        time of the audio. stop_callback_fn will also be called one last time with t = -1 when the audio finished
        """
        sd = get_sounddevice()
        def _play(sound, sr, nc, stop_event):
            event = threading.Event()
            x = 0

            def callback(outdata, frames, time, status):
                nonlocal x
                sound_ = sound[x:x+frames]
                x = x + frames

                t = x/sr

                if callback_fn is not None:
                    callback_fn(t)

                # Print the info if there are anything
                if info is not None:
                    while info and t > info[0][1]:
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
                if stop_callback_fn is not None:
                    stop_callback_fn()

        if info is not None:
            blocking = True # Otherwise jupyter notebook will behave weirdly
        else:
            if is_ipython():
                from IPython.display import Audio as IPAudio
                return IPAudio(self.numpy(), rate = self.sample_rate)
            info = []
        info = sorted(info, key = lambda x: x[1])
        longest_info = max([len(x[0]) for x in info]) if info else 0
        sound = self._data.mean(dim = 0).unsqueeze(1).detach().cpu().numpy()
        self._thread = threading.Thread(target=_play, args=(sound, self.sample_rate, self.nchannels.value, lambda :self._stop_audio))
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
        self._stop_audio = False # Reset the state

    def __add__(self, other: Audio):
        """Adds two audio together. Currently we only support that `self` and `other` have the same duration, same device, and the same sample rate
        For the nchannels - returns an audio with two channels unless both `self` and `other` has one channel only."""
        assert isinstance(other, Audio)
        self.sanity_check()
        other.sanity_check()
        assert self._sample_rate == other._sample_rate
        assert self._data.size(1) == other._data.size(1)

        new_nchannels = AudioMode.MONO if self.nchannels == other.nchannels == AudioMode.MONO else AudioMode.STEREO

        if self.nchannels == other.nchannels == new_nchannels:
            return Audio(self._data + other._data, self._sample_rate)

        # Make a recursive call to save space
        return self.to_nchannels(new_nchannels) + other.to_nchannels(new_nchannels)

    def save(self, fpath: str):
        """Saves the audio at the provided file path. WAV is (almost certainly) guaranteed to work"""
        self.sanity_check()
        if fpath.endswith(".mp3"):
            try:
                from pydub import AudioSegment
            except ImportError:
                raise RuntimeError("You need to install pydub to save the audio as mp3")
            with tempfile.TemporaryDirectory() as tempdir:
                temp_fpath = path.join(tempdir, "temp.wav")
                torchaudio.save(temp_fpath, self._data, sample_rate = self._sample_rate)
                song = AudioSegment.from_wav(temp_fpath)
                song.export(fpath, format="mp3")
            return
        try:
            torchaudio.save(fpath, self._data, sample_rate = self._sample_rate)
            return
        except (ValueError, RuntimeError) as e: # Seems like torchaudio changed the error type to runtime error in 2.2?
            # or the file path is invalid
            raise RuntimeError(f"Error saving the audio: {e} - {fpath}")



    def plot_waveform(self, keep_sr = False):
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

    def add_to_frame(self, other: Audio, nframe: int | float):
        """Adds `other` to `self` starting on frame `nframe`. If `nframe` is negative, then `other` will start before `self`
        `self` and `other` must have the same sample rate"""

        nframe = int(nframe)

        if self.sample_rate != other.sample_rate:
            raise RuntimeError(f"self ({self.sample_rate}) and other ({other.sample_rate}) has a different sample rate")

        if nframe < 0:
            return other.add_to_frame(self, -nframe)

        # see whether need to pad self or pad other
        # ------ self ------
        # (nframes) --- other ---
        other = other.pad(nframe, front=True)

        # Add extra padding if needed
        pad_self = other.nframes - self.nframes
        if pad_self > 0:
            newself = self.pad(other.nframes)
            newother = other
        else:
            newother = other.pad(self.nframes)
            newself = self

        return newself + newother

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

    def numpy(self, keep_dims=False):
        """Returns the 1D numpy audio format of the audio. If you insist you want the 2D audio, put keep_dims = True"""
        self.sanity_check()
        if keep_dims:
            return self._data.detach().cpu().numpy()
        data = self._data
        if self._data.size(0) == 2:
            data = data.mean(dim = 0)
        else:
            data = data[0]
        try:
            return data.numpy()
        except Exception as e:
            return data.detach().cpu().numpy()

    def apply(self, transform: AudioTransform, keep_dims = True) -> Audio:
        """Applies a transform to the audio and returns the transformed audio

        if keep_dims is true, then if the sample rate of the transformed audio is different from the original audio, we will resample the audio to the original sample rate."""
        x = transform.apply(self._data, self._sample_rate)
        if not isinstance(x, tuple):
            x = (x, self._sample_rate)

        audio = Audio(x[0], x[1])
        if not keep_dims:
            return audio

        if x[1] != self._sample_rate:
            audio = audio.resample(self.sample_rate)

        if audio.nframes != self.nframes:
            audio = audio.pad(self.nframes)

        return audio

    def __str__(self):
        """
        Prints out the following information about the audio:
        Duration, Sample rate, Num channels, Num frames
        """
        return f"(Audio)\nDuration:\t{self.duration:5f}\nSample Rate:\t{self.sample_rate}\nChannels:\t{self.nchannels}\nNum frames:\t{self.nframes}"

    def change_speed(self, speed: float, n_fft: int = 512, win_length: int | None = None, hop_length: int | None = None, window: Tensor | None = None) -> Audio:
        """Changes the speed of the audio. speed in absolute value must be greater than 0. If speed = 1, then the audio is unchanged.
        If speed > 1, then the audio is sped up. If speed < 1, then the audio is slowed down. The pitch of the audio is not changed.
        The audio is stretched or compressed in time. The audio is stretched by a factor of `speed`"""
        if speed == 1:
            return self.clone()
        if speed < 0:
            raise ValueError("Speed must be nonnegative")

        PI = 3.1415926535897932

        audio = self._data
        audio_length = audio.size(-1)

        if hop_length is None:
            hop_length = n_fft // 4
        if win_length is None:
            win_length = n_fft
        if window is None:
            window = torch.hann_window(window_length = win_length, device = audio.device)

        # Apply stft
        spectrogram = torch.stft(
            input = audio,
            n_fft = n_fft,
            hop_length = hop_length,
            win_length = win_length,
            window = window,
            center = True,
            pad_mode = "reflect",
            normalized = False,
            onesided = True,
            return_complex = True,
        )

        # Stretch the audio without modifying pitch - phase vocoder
        phase_advance = torch.linspace(0, PI * hop_length, spectrogram.shape[-2], device=spectrogram.device)[..., None]
        stretched_spectrogram = F.phase_vocoder(spectrogram, speed, phase_advance)
        len_stretch = int(round(audio_length / speed))

        # Inverse the stft
        waveform_stretch = torch.istft(
            stretched_spectrogram,
            n_fft = n_fft,
            hop_length = hop_length,
            win_length = win_length,
            window = window,
            length = len_stretch
        )

        return Audio(waveform_stretch, self.sample_rate)

    def get_duration(self) -> float:
        return self.duration

    def show_spectrogram(self):
        plt.style.use('dark_background')
        y = self.numpy()
        sr = self.sample_rate
        fig, ax = plt.subplots()
        hop_length = 1024
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)),
                                    ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
                                x_axis='time', ax=ax)
        ax.set(title='Log-frequency power spectrogram')
        ax.label_outer()
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        return fig

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
        mixer = torch.tensor([[left_mix], [right_mix]], device = audio.device)
        return Audio(audio._data * mixer, audio.sample_rate)
