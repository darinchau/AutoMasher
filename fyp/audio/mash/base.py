from abc import ABC, abstractmethod
from typing import Any
import torch
from .. import AudioCollection
from .. import Audio, AudioMode, AudioCollection
from ...util import get_url
from ..analysis import ChordAnalysisResult, BeatAnalysisResult
from ..manipulation import HighpassFilter, PitchShift
from ..search.align import MashabilityResult, calculate_boundaries
from ..search.search import SongSearchState

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

class MashupMaker(ABC):
    """A pipeline for creating various kinds of mashup"""
    def __init__(self, search_state: SongSearchState, score: MashabilityResult):
        self._submitted_song = search_state
        self._sample_song = SongSearchState(
            link=get_url(score.url_id),
            config=SearchConfig(
                bar_number=score.start_bar,
                nbars=search_state.slice_nbar,
            ),
            dataset=search_state.dataset
        )
        self._score = score

    @property
    def sample_url_id(self):
        return self._score.url_id

    @property
    def sample_start_idx(self):
        return self._score.start_bar

    @property
    def sample_transpose(self):
        return self._score.transpose

    def create_mashup_components(self):
        """Create the song components for the mashup from the pipeline and the score id"""
        factors, boundaries = calculate_boundaries(self._submitted_song.submitted_beat_result, self._sample_song.submitted_beat_result)

        # Transpose the parts
        trimmed_parts: dict[str, Audio] = {}
        pitchshift = PitchShift(self.sample_transpose)
        for key, value in self._sample_song.submitted_parts.items():
            trimmed_parts[key] = value.apply(pitchshift)
        trimmed_parts["original"] = self._sample_song.submitted_audio.apply(pitchshift)

        # Pad the output just in case
        for key, value in trimmed_parts.items():
            trimmed_parts[key] = value.pad(trimmed_parts["original"].nframes)
        trimmed_portion = AudioCollection(**trimmed_parts)
        trimmed_portion = trimmed_portion.align_from_boundaries(factors, boundaries) \
            .map(lambda x: x.resample(self._submitted_song.submitted_audio.sample_rate)) \
                .map(lambda x: x.pad(self._submitted_song.submitted_audio.nframes))

        return self._submitted_song.submitted_parts, trimmed_portion

    @abstractmethod
    def create(self) -> Audio:
        pass
