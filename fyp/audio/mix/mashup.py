# Contains the algorithmic implementation to create a simple mashup
from typing import Any
import torch
from math import isclose
from ..base import Audio, DemucsCollection
from ...util import get_url
from ..analysis import OnsetFeatures
from ..manipulation import PitchShift
from ..mix.align import calculate_onset_boundaries
from .mastering import create_mashup_from_parts
from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy.typing import NDArray
import librosa

class MashupMode(Enum):
    """The swap algorithm to use for the mashup

    Vocals and drum swapping are especially helpful and effective (Cite: X. Wu and A. Horner)"""
    VOCAL_A = "vocal_a"
    """Use the vocals from song A and the rest from song B"""

    VOCAL_B = "vocal_b"
    """Use the vocals from song B and the rest from song A"""

    DRUMS_A = "drums_a"
    """Use the drums from song A and the rest from song B"""

    DRUMS_B = "drums_b"
    """Use the drums from song B and the rest from song A"""

    VOCALS_NATURAL = "vocals_natural"
    """Performs a vocal swap, but algorithmically determines the best vocals to use"""

    DRUMS_NATURAL = "drums_natural"
    """Performs a drum swap, but algorithmically determines the best drums to use"""

    NATURAL = "natural"
    """Algorithmically determines the best parts to use"""

    RANDOM = "random"
    """Randomly swaps the parts. The 'Suprise me' mode"""

def get_volume(audio: Audio, hop: int = 512) -> NDArray[np.float32]:
    y = audio.numpy()
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop)), ref=np.max).astype(np.float32)
    volume = spec.mean(axis = 0)
    return volume

def cross_fade(song1: Audio, song2: Audio, fade_duration: float, cross_fade_mode: str = "linear"):
    """Joins two songs with a cross fade effect with the given fade duration
    cross_fade_mode: "linear" or "sigmoid"
    result: ( ==== Song 1 ==== ) ( == Cross Fade == ) ( ==== Song 2 ====)
    """
    if song2.sample_rate != song1.sample_rate:
        song2 = song2.resample(int(song1.sample_rate))

    if song2.nchannels != song1.nchannels:
        song2 = song2.to_nchannels(song1.nchannels)

    fade_duration_frames = int(fade_duration * song1.sample_rate)
    if cross_fade_mode == "sigmoid":
        fade_in = torch.sigmoid(torch.linspace(-10, 10, fade_duration_frames))
        fade_out = torch.sigmoid(torch.linspace(10, -10, fade_duration_frames))
    elif cross_fade_mode == "linear":
        fade_in = torch.linspace(0, 1, fade_duration_frames)
        fade_out = torch.linspace(1, 0, fade_duration_frames)
    else:
        raise ValueError("cross_fade_mode must be either 'linear' or 'sigmoid' but found " + cross_fade_mode)

    fade_in = fade_in.view(1, -1)
    fade_out = fade_out.view(1, -1)

    song1_fade_out = song1.data[:, -fade_duration_frames:] * fade_out
    song2_fade_in = song2.data[:, :fade_duration_frames] * fade_in
    cross_fade = Audio(data = song1_fade_out + song2_fade_in, sample_rate = song1.sample_rate)

    song1_normal = song1.slice_frames(0, song1.nframes - fade_duration_frames)
    song2_normal = song2.slice_frames(fade_duration_frames, song2.nframes)
    return song1_normal.join(cross_fade).join(song2_normal)

def create_mashup_component(song_a_submitted_bt: OnsetFeatures, song_b_submitted_bt: OnsetFeatures,
                            transpose: int, song_b_submitted_parts: DemucsCollection, song_a_nframes: int, song_a_sr: int, song_b_nframes: int):
    """Creates the song B components that ought to be used for mashup. This includes transposing the parts and aligning song B with song A

    Returns the song B parts that should be ready for mashup"""
    assert isclose(song_b_submitted_parts.get_duration(), song_b_submitted_bt.duration, abs_tol=1/44100), \
        f"Song B parts duration {song_b_submitted_parts.get_duration()} does not match the beat analysis duration {song_b_submitted_bt.duration}"

    factors, boundaries = calculate_onset_boundaries(song_a_submitted_bt, song_b_submitted_bt)

    # Transpose the parts
    trimmed_parts: dict[str, Audio] = {}
    pitchshift = PitchShift(nsteps=transpose)
    for key, value in song_b_submitted_parts.items():
        trimmed_parts[key] = pitchshift.apply(value)

    # Pad the output just in case
    for key, value in trimmed_parts.items():
        trimmed_parts[key] = value.pad(song_b_nframes)

    trimmed_portion = DemucsCollection(**trimmed_parts)

    trimmed_portion = trimmed_portion.align_from_boundaries(factors, boundaries) \
        .map(lambda x: x.resample(song_a_sr)) \
            .map(lambda x: x.pad(song_a_nframes))

    return trimmed_portion

def calculate_average_volume(audio: Audio, window_size: int, hop: int = 512) -> NDArray[np.float32]:
    vol = get_volume(audio, hop)
    pad_length = window_size - vol.shape[0] % window_size
    vol = np.pad(vol, (0, pad_length), mode = "constant", constant_values=0)
    vol = vol.reshape(-1, window_size)
    return vol.mean(axis = 1)


def create_mashup(submitted_audio_a: Audio,
                  submitted_audio_b: Audio,
                  submitted_downbeats_a: OnsetFeatures,
                  submitted_downbeats_b: OnsetFeatures,
                  submitted_parts_a: DemucsCollection,
                  submitted_parts_b: DemucsCollection,
                  transpose: int,
                  mode: MashupMode,
                  volume_hop: int = 512,
                  natural_vocal_activity_threshold: float = 1,
                  natural_vocal_proportion_threshold: float = 0.8,
                  natural_drum_activity_threshold: float = 1,
                  natural_drum_proportion_threshold: float = 0.8,
                  natural_window_size: int = 20,
                  left_pan: float = 0.15,
                  verbose: bool = False) -> tuple[Audio, MashupMode]:
    """Creates a basic mashup with the given components"""
    if mode == MashupMode.RANDOM:
        raise NotImplementedError # TODO: Should be easy to implement

    vocal_a_proportions: float | None = None
    vocal_b_proportions: float | None = None
    drum_a_pass_threshold: bool | None = None
    drum_b_pass_threshold: bool | None = None

    if mode in (MashupMode.NATURAL, MashupMode.VOCALS_NATURAL):
        vocal_a_volume = calculate_average_volume(submitted_parts_a.vocals, natural_window_size, volume_hop)
        vocal_b_volume = calculate_average_volume(submitted_parts_b.vocals, natural_window_size, volume_hop)
        vocal_a_proportions = np.count_nonzero(vocal_a_volume > natural_vocal_activity_threshold) / vocal_a_volume.shape[0]
        vocal_b_proportions = np.count_nonzero(vocal_b_volume > natural_vocal_activity_threshold) / vocal_b_volume.shape[0]
        vocal_a_pass_threshold = vocal_a_proportions > natural_vocal_proportion_threshold
        vocal_b_pass_threshold = vocal_b_proportions > natural_vocal_proportion_threshold

        # Preference for using vocals from B and backing from A
        if vocal_b_pass_threshold:
            mode = MashupMode.VOCAL_B

        # If it is vocal natural, then VOCAL_A is the only other option
        # But if we use natural mode and vocal_a does not pass threshold, then we have further options
        if mode == MashupMode.VOCALS_NATURAL or vocal_a_pass_threshold:
            mode = MashupMode.VOCAL_A

    if mode in (MashupMode.NATURAL, MashupMode.DRUMS_NATURAL):
        drum_a_volume = calculate_average_volume(submitted_parts_a.drums, natural_window_size, volume_hop)
        drum_b_volume = calculate_average_volume(submitted_parts_b.drums, natural_window_size, volume_hop)
        drum_a_pass_threshold = np.count_nonzero(drum_a_volume > natural_drum_activity_threshold) / drum_a_volume.shape[0] > natural_drum_proportion_threshold
        drum_b_pass_threshold = np.count_nonzero(drum_b_volume > natural_drum_activity_threshold) / drum_b_volume.shape[0] > natural_drum_proportion_threshold

        # Preference for using drums from B and backing from A
        if drum_b_pass_threshold:
            mode = MashupMode.DRUMS_B

        # If it is drum natural, then DRUMS_A is the only other option
        # But if we use natural mode and drum_a does not pass threshold, then we have further options
        if mode == MashupMode.DRUMS_NATURAL or drum_a_pass_threshold:
            mode = MashupMode.DRUMS_A

    if mode == MashupMode.NATURAL:
        assert vocal_a_proportions is not None and vocal_b_proportions is not None
        if vocal_a_proportions > vocal_b_proportions:
            mode = MashupMode.VOCAL_A
        else:
            mode = MashupMode.VOCAL_B

    submitted_parts_b = create_mashup_component(
        submitted_downbeats_a,
        submitted_downbeats_b,
        transpose,
        submitted_parts_b,
        submitted_parts_a.vocals.nframes,
        int(submitted_parts_a.vocals.sample_rate),
        submitted_parts_b.vocals.nframes
    )

    if verbose:
        print("Creating mashup with mode", mode)
        print("Vocal A proportions", vocal_a_proportions)
        print("Vocal B proportions", vocal_b_proportions)
        print("Drum A proportions", drum_a_pass_threshold)
        print("Drum B proportions", drum_b_pass_threshold)
        print("Transpose", transpose)
        print(f"Downbeats A: {submitted_downbeats_a}")
        print(f"Downbeats B: {submitted_downbeats_b}")

    mix_bitmask = {
        MashupMode.VOCAL_A: 0b00011110,
        MashupMode.VOCAL_B: 0b11100001,
        MashupMode.DRUMS_A: 0b10000111,
        MashupMode.DRUMS_B: 0b01111000
    }[mode]

    mashup = create_mashup_from_parts(submitted_audio_a, submitted_audio_b, submitted_parts_a, submitted_parts_b, mix_bitmask, left_pan=left_pan)

    return mashup, mode
