from tqdm.auto import trange
from typing import Any
import enum
import numpy as np
from .. import Audio
from .. import AudioCollection
from ..analysis import ChordAnalysisResult, BeatAnalysisResult
from ..manipulation import PitchShift, HighpassFilter
from ..search import calculate_boundaries
from .base import MashupMaker
from .base import mash_two_songs, cross_fade

class SameBPMMode(enum.Enum):
    NORMAL = 0
    FORCE_SUBMITTED_VOCALS = 1
    FORCE_TRIMMED_VOCALS = 2

def vocals_insufficient(vocals: Audio) -> bool:
    """Check if the volume of the vocals is insufficient."""
    return vocals.volume < 0.05

class BasicMashupMaker(MashupMaker):
    def create(self, mode: SameBPMMode = SameBPMMode.NORMAL):
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

        submitted = self._submitted_song.submitted_parts
        sample = trimmed_portion

        use_sample_as_backing = mode == SameBPMMode.FORCE_SUBMITTED_VOCALS or (mode == SameBPMMode.NORMAL and vocals_insufficient(sample["vocals"]))
        if use_sample_as_backing:
            submitted, sample = sample, submitted
        mashup = mash_two_songs(submitted, sample)
        return mashup

class TheseTwoSongsHaveTheSameBPM(MashupMaker):
    def create(self, mode: SameBPMMode = SameBPMMode.NORMAL, fade_mode: str = "linear"):
        """Create the mashup using the trimmed sample and the submitted audio
        trimmed_vocals_volume_threshold: If the volume of the trimmed vocals is less than this threshold,
        the trimmed audio will be used as the backing track instead of the submitted audio. Default is 0.05.
        Set it to -1 if you want to always use the submitted audio as the backing track.
        Set it to a very high value if you want to always use the trimmed audio as the backing track."""
        mashupper = BasicMashupMaker(self._submitted_song, self._score)
        mashup = mashupper.create(mode)

        mashup_start_idx = self._submitted_song.slice_start_bar
        mashup_nbars = self._submitted_song.slice_nbar

        if mashup_start_idx < mashup_nbars:
            raise NotImplementedError("Mashup start index is less than mashup n bars")

        # Get the prev n bars
        start_time = self._submitted_song.raw_beat_result.downbeats[mashup_start_idx - mashup_nbars]
        end_time = self._submitted_song.raw_beat_result.downbeats[mashup_start_idx + 1]
        prev_nbars_of_sumbitted_audio = self._submitted_song.audio.slice_seconds(start_time, end_time)
        fade_time = end_time - self._submitted_song.raw_beat_result.downbeats[mashup_start_idx]

        mashup_result = cross_fade(prev_nbars_of_sumbitted_audio, mashup, fade_time, cross_fade_mode=fade_mode)
        return mashup_result
