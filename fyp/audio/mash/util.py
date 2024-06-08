from .. import Audio
import torch
from .. import AudioMode, AudioCollection
from ..manipulation import HighpassFilter
from ..search.align import calculate_boundaries
from ..manipulation import PitchShift
from ..analysis import ChordAnalysisResult, BeatAnalysisResult
from datasets import Dataset
from typing import Any

def mash_two_songs(submitted: AudioCollection, sample: AudioCollection):
	"""Mash two songs using sample vocals and submitted audio"""
	bass = submitted['bass']
	other = submitted['other']
	drums = submitted['drums']
	vocals = sample['vocals']

	# Baseline volume
	baseline_volume = 0.1

	# Make vocals 1.1x louder than others
	vocals = vocals.mix_to_stereo(left_mix = 0.15)
	vocals.volume = baseline_volume * 1.1

	# Assuming the other 3 parts are well mixed
	bass = bass.mix_to_stereo(left_mix = -0.15)
	drums = drums.mix_to_stereo(left_mix = 0.3)
	other = other.mix_to_stereo(left_mix = -0.3)

	backing_track = bass + drums + other
	backing_track.volume = baseline_volume

	highpass = HighpassFilter(300) #TODO change to smth determined with the spectrogram of song instead of 300

	filtered_vocals = vocals.apply(highpass)

	filtered_vocals.volume = baseline_volume * 1.1

	mashup = backing_track + filtered_vocals
	return mashup

def cross_fade(song1: Audio, song2: Audio, fade_duration: float, cross_fade_mode: str = "linear"):
	"""Joins two songs with a cross fade effect with the given fade duration
	cross_fade_mode: "linear" or "sigmoid"
	"""
	if song2.sample_rate != song1.sample_rate:
		song2 = song2.resample(song1.sample_rate)

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

	song1_fade_out = song1._data[:, -fade_duration_frames:] * fade_out
	song2_fade_in = song2._data[:, :fade_duration_frames] * fade_in
	cross_fade = Audio(data = song1_fade_out + song2_fade_in, sample_rate = song1.sample_rate)

	song1_normal = song1.slice(0, song1.nframes - fade_duration_frames)
	song2_normal = song2.slice(fade_duration_frames, song2.nframes)
	return song1_normal.join(cross_fade).join(song2_normal)
