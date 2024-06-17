from ..separation import DemucsAudioSeparator
from .. import Audio
from ..analysis import BeatAnalysisResult, ChordAnalysisResult
from ...util.combine import get_url
from ..manipulation import PitchShift
from .. import AudioCollection
from abc import ABC, abstractmethod
from ..search.align import SearchConfig, MashabilityResult, calculate_boundaries
from ..search.search import SongSearchState

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
