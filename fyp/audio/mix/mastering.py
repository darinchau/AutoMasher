# Exports the function `mastering` that takes in an Audio and a reference and performs mastering on the audio.

import os
from typing import Any
import tempfile
import matchering as mg
import torch
from ..base import Audio, DemucsCollection
from ..manipulation import HighpassFilter
import torch.nn.functional as F

def mastering(audio: Audio, reference: Audio) -> Audio:
    """Master the audio using the reference audio"""
    with tempfile.TemporaryDirectory() as temp_dir:
        mashup_path = os.path.join(temp_dir, "mashup.wav")
        ref_path = os.path.join(temp_dir, "ref.wav")
        result_path = os.path.join(temp_dir, "result.wav")
        audio.save(mashup_path)
        reference.save(ref_path)
        mg.process(
            target=os.path.join(temp_dir, "mashup.wav"),
            reference=os.path.join(temp_dir, "ref.wav"),
            results=[
                mg.pcm16(result_path),
            ],
        )
        result = Audio.load(result_path)

    return result


def create_mashup_from_parts(
        submitted_audio_a: Audio,
        submitted_audio_b: Audio,
        submitted_parts_a: DemucsCollection,
        submitted_parts_b: DemucsCollection,
        mix: int,
        left_pan: float = 0.15
    ) -> Audio:
    """Mix is a bitmask of the parts to mix. 0b1000 is drums, 0b0100 is bass, 0b0010 is other, 0b0001 is vocals
    upper 4 bits are for parts_a, lower 4 bits are for parts_b"""
    assert 0 <= mix < 256
    highpass = HighpassFilter(300) #TODO change to smth determined with the spectrogram of song instead of 300

    audios: list[Audio] = []
    vocals_a = None
    vocals_b = None
    if mix & 0b10000000:
        audios.append(submitted_parts_a.drums.mix_to_stereo(left_mix = left_pan * 2))
    if mix & 0b01000000:
        audios.append(submitted_parts_a.bass.mix_to_stereo(left_mix = -left_pan))
    if mix & 0b00100000:
        audios.append(submitted_parts_a.other.mix_to_stereo(left_mix = -left_pan * 2))
    if mix & 0b00010000:
        vocals_a = submitted_parts_a.vocals.mix_to_stereo(left_mix = left_pan)
        vocals_a = highpass.apply(vocals_a)
        # If vocals_a is not None, then reverse the pan to make sure A and B are separated
        left_pan = -left_pan
    if mix & 0b1000:
        audios.append(submitted_parts_b.drums.mix_to_stereo(left_mix = -left_pan * 2))
    if mix & 0b0100:
        audios.append(submitted_parts_b.bass.mix_to_stereo(left_mix = left_pan))
    if mix & 0b0010:
        audios.append(submitted_parts_b.other.mix_to_stereo(left_mix = left_pan * 2))
    if mix & 0b0001:
        vocals_b = submitted_parts_b.vocals.mix_to_stereo(left_mix = -left_pan)
        vocals_b = highpass.apply(vocals_b)

    assert len(audios) > 0, "No parts selected for mixing"

    backing_track_tensor = torch.stack([x.data for x in audios], dim = 0).sum(dim = 0)
    backing_track_volume = backing_track_tensor.square().mean().sqrt().item()
    backing_track_tensor = (backing_track_tensor * 0.1 / backing_track_volume).clamp(-1, 1)

    vocals = None
    if vocals_a is not None and vocals_b is not None:
        vocals = torch.stack([vocals_a.data, vocals_b.data], dim = 0).sum(dim = 0)
    elif vocals_a is not None:
        vocals = vocals_a.data
    elif vocals_b is not None:
        vocals = vocals_b.data
    else:
        vocals = torch.empty_like(submitted_audio_a.data)

    vocals_volume = vocals.square().mean().sqrt().item()
    vocals = (vocals * 0.1 / vocals_volume).clamp(-1, 1)

    mixed = Audio((vocals + backing_track_tensor).clamp(-1, 1), sample_rate = submitted_audio_a.sample_rate)
    mixed = mastering(mixed, submitted_audio_b) if vocals_b is not None else mastering(mixed, submitted_audio_a)
    return mixed
