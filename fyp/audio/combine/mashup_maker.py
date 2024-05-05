from ..search import SongSearcher
from ..search.align import calculate_boundaries
from ..separation import DemucsAudioSeparator
from .. import Audio
from ..analysis import BeatAnalysisResult, ChordAnalysisResult
from ...util.combine import get_entry_from_database, get_url
from ..manipulation import PitchShift
from .. import AudioCollection
from abc import ABC, abstractmethod

class MashupMaker(ABC):
    """A pipeline for creating various kinds of mashup"""
    def __init__(self, initial_search_result: SongSearcher, score_id: str):
        self._pipeline = initial_search_result
        self.set_sample_score_id(score_id)

    @property
    def pipeline(self) -> SongSearcher:
        return self._pipeline

    @property
    def sample_score_id(self) -> str:
        if not hasattr(self, "_score_id"):
            raise ValueError("No score id found")
        return self._score_id
    
    def set_sample_score_id(self, value: str):
        self._score_id = value
        self._sample_url_id, self._sample_start_idx, self._sample_transpose = value.split('/')[:3]

    @property
    def sample_url_id(self):
        if not hasattr(self, "_sample_url_id"):
            self.sample_score_id
        return self._sample_url_id
    
    @property
    def sample_start_idx(self):
        if not hasattr(self, "_sample_start_idx"):
            self.sample_score_id
        return int(self._sample_start_idx)
    
    @property
    def sample_transpose(self):
        if not hasattr(self, "_sample_transpose"):
            self.sample_score_id
        return int(self._sample_transpose)
    
    @property
    def sample_nbars(self):
        return self.pipeline.slice_nbar
    
    @property
    def sample_entry(self):
        if not hasattr(self, "_sample_entry"):
            self._sample_entry = get_entry_from_database(self.pipeline.dataset, get_url(self.sample_url_id))
        return self._sample_entry
    
    @property
    def raw_sample_beat_result(self) -> BeatAnalysisResult:
        if not hasattr(self, "_sample_beat_result"):
            self._sample_beat_result = BeatAnalysisResult.from_data_entry(self.sample_entry)
        return self._sample_beat_result
    
    @property
    def raw_sample_chord_result(self) -> ChordAnalysisResult:
        if not hasattr(self, "_sample_chord_result"):
            self._sample_chord_result = ChordAnalysisResult.from_data_entry(self.sample_entry)
        return self._sample_chord_result
    
    @property
    def trimmed_sample_beat_result(self) -> BeatAnalysisResult:
        if not hasattr(self, "_trimmed_sample_beat_result"):
            start_downbeat = self.raw_sample_beat_result.downbeats[self.sample_start_idx]
            end_downbeat = self.raw_sample_beat_result.downbeats[self.sample_start_idx + self.sample_nbars]
            self._start_downbeat = start_downbeat
            self._end_downbeat = end_downbeat
            self._trimmed_sample_beat_result = self.raw_sample_beat_result.slice_seconds(start_downbeat, end_downbeat)
        return self._trimmed_sample_beat_result
    
    @property
    def sample_start_downbeat(self):
        if not hasattr(self, "_start_downbeat"):
            self.trimmed_sample_beat_result
        return self._start_downbeat
    
    @property
    def sample_end_downbeat(self):
        if not hasattr(self, "_end_downbeat"):
            self.trimmed_sample_beat_result
        return self._end_downbeat
    
    @property
    def sample_audio(self):
        if not hasattr(self, "_sample_audio"):
            url = get_url(self.sample_url_id)
            self._sample_audio = Audio.load(url)
        return self._sample_audio
        
    @property
    def trimmed_sample_audio(self):
        if not hasattr(self, "_trimmed_sample_audio"):
            self._trimmed_sample_audio = self.sample_audio.slice_seconds(self.sample_start_downbeat, self.sample_end_downbeat)
        return self._trimmed_sample_audio
    
    @property
    def sample_parts(self):
        if not hasattr(self, "_sample_parts"):
            demucs = DemucsAudioSeparator()
            self._sample_parts = demucs.separate_audio(self.trimmed_sample_audio)
        return self._sample_parts
    
    @property
    def trimmed_sample_parts(self):
        if not hasattr(self, "_trimmed_sample_parts"):
            if hasattr(self, "_sample_parts"):
                self._trimmed_sample_parts = self.sample_parts.slice_seconds(self.sample_start_downbeat, self.sample_end_downbeat)
            else:
                demucs = DemucsAudioSeparator()
                self._trimmed_sample_parts = demucs.separate_audio(self.trimmed_sample_audio)
        return self._trimmed_sample_parts

    def create_mashup_components(self):
        """Create the song components for the mashup from the pipeline and the score id"""
        factors, boundaries = calculate_boundaries(self.pipeline.submitted_beat_result, self.trimmed_sample_beat_result)

        # Transpose the parts
        trimmed_parts: dict[str, Audio] = {}
        pitchshift = PitchShift(self.sample_transpose)
        for key, value in self.trimmed_sample_parts.items():
            trimmed_parts[key] = value.apply(pitchshift)
        trimmed_parts["original"] = self.trimmed_sample_audio.apply(pitchshift)

        # Pad the output just in case
        for key, value in trimmed_parts.items():
            trimmed_parts[key] = value.pad(trimmed_parts["original"].nframes)
        trimmed_portion = AudioCollection(**trimmed_parts)
        trimmed_portion = trimmed_portion.align_from_boundaries(factors, boundaries) \
            .map(lambda x: x.resample(self.pipeline.submitted_audio.sample_rate)) \
                .map(lambda x: x.pad(self.pipeline.submitted_audio.nframes))
        
        return self.pipeline.submitted_parts, trimmed_portion

    @abstractmethod
    def create(self) -> Audio:
        pass
