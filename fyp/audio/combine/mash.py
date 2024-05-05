from .mashup_maker import MashupMaker
from ..search import calculate_boundaries, SearchConfig, SongSearcher
from ..search.searcher import curve_score
from .. import Audio
from ..manipulation import PitchShift, HighpassFilter
from .. import AudioCollection
from .util import mash_two_songs, cross_fade
from ..analysis import ChordAnalysisResult, BeatAnalysisResult
from datasets import Dataset
import numpy as np
from tqdm.auto import trange
from ...util.combine import get_url as get_video_url, get_entry_from_database
import enum
from typing import Any
from .util import restrictive_search

class SameBPMMode(enum.Enum):
    NORMAL = 0
    FORCE_SUBMITTED_VOCALS = 1
    FORCE_TRIMMED_VOCALS = 2

def vocals_insufficient(vocals: Audio) -> bool:
    """Check if the volume of the vocals is insufficient."""
    return vocals.volume < 0.05

class BasicMashupMaker(MashupMaker):
    def create(self, mode: SameBPMMode = SameBPMMode.NORMAL):        
        factors, boundaries = calculate_boundaries(self.pipeline.submitted_beat_result, self.trimmed_sample_beat_result)

        # Transpose the parts
        print("Demixing and transposing parts", flush=True)
        trimmed_parts: dict[str, Audio] = {}
        pitchshift = PitchShift(self.sample_transpose)
        for key, value in self.trimmed_sample_parts.items():
            trimmed_parts[key] = value.apply(pitchshift)
        trimmed_parts["original"] = self.trimmed_sample_audio.apply(pitchshift)

        # Pad the output just in case
        print("Aligning parts", flush=True)
        for key, value in trimmed_parts.items():
            trimmed_parts[key] = value.pad(trimmed_parts["original"].nframes)
        trimmed_portion = AudioCollection(**trimmed_parts)
        trimmed_portion = trimmed_portion.align_from_boundaries(factors, boundaries) \
            .map(lambda x: x.resample(self.pipeline.submitted_audio.sample_rate)) \
                .map(lambda x: x.pad(self.pipeline.submitted_audio.nframes))

        # Now determine the volume and see if we want to change the backing track
        print("Creating final mashup")
        submitted = self.pipeline.submitted_parts
        trimmed = trimmed_portion

        
        if mode == SameBPMMode.FORCE_SUBMITTED_VOCALS or (mode == SameBPMMode.NORMAL and vocals_insufficient(trimmed["vocals"])):
            # Used trimmed as backing track and submitted as vocals
            print("Using trimmed as backing track and submitted as vocals")
            submitted, trimmed = trimmed, submitted
        mashup = mash_two_songs(submitted, trimmed)
        return mashup

class TheseTwoSongsHaveTheSameBPM(MashupMaker):
    def create(self, mode: SameBPMMode = SameBPMMode.NORMAL, fade_mode: str = "linear"):
        """Create the mashup using the trimmed sample and the submitted audio
        trimmed_vocals_volume_threshold: If the volume of the trimmed vocals is less than this threshold, 
        the trimmed audio will be used as the backing track instead of the submitted audio. Default is 0.05.
        Set it to -1 if you want to always use the submitted audio as the backing track.
        Set it to a very high value if you want to always use the trimmed audio as the backing track."""
        mashupper = BasicMashupMaker(self.pipeline, self.sample_score_id)
        mashup = mashupper.create(mode)

        mashup_start_idx = self.pipeline.slice_start_bar
        mashup_nbars = self.pipeline.slice_nbar

        if mashup_start_idx < mashup_nbars:
            raise NotImplementedError("Mashup start index is less than mashup n bars")

        # Get the prev n bars
        start_time = self.pipeline.raw_beat_result.downbeats[mashup_start_idx - mashup_nbars]
        end_time = self.pipeline.raw_beat_result.downbeats[mashup_start_idx + 1]
        prev_nbars_of_sumbitted_audio = self.pipeline.audio.slice_seconds(start_time, end_time)
        fade_time = end_time - self.pipeline.raw_beat_result.downbeats[mashup_start_idx]

        mashup_result = cross_fade(prev_nbars_of_sumbitted_audio, mashup, fade_time, cross_fade_mode=fade_mode)
        return mashup_result

class FordFulkersonMashupper(MashupMaker):
    def construct_table(self, nbars: int, *, verbose: bool = False):
        # Use a database with one entry so that we can reuse our code
        video_url = get_video_url(self.sample_url_id)
        query_dataset = self.pipeline.dataset.filter(lambda x: x['url'] == video_url)
        if len(query_dataset) != 1:
            raise ValueError(f"Found {len(query_dataset)} entry in the dataset with the given video id.")

        # Construct the table
        beat_result = self.pipeline.raw_beat_result
        nrows = beat_result.downbeats.shape[0] - nbars

        # Fill the table
        scores: list[list[float]] = []
        for i in trange(nrows - nbars, desc="Constructing restrictive search table", disable=not verbose):
            score = restrictive_search(
                chord_result=self.pipeline.raw_chord_result,
                beat_result=beat_result,
                dataset=query_dataset,
                start_bar_number=i,
                transpose=self.sample_transpose,
                nbars=nbars
            )
            scores.append(score)

        score_table = np.array(scores)
        return score_table
    
    def create(self, nbars: int | None = None):
        if nbars is None:
            nbars = self.pipeline.slice_nbar
        print(f"Creating FF mashup with {nbars} bars")
        score_table = self.construct_table(nbars)
        raise NotImplementedError("Ford Fulkerson is not implemented yet.")

class GlobalBestSameBPM(MashupMaker):
    def create(self, mode: SameBPMMode = SameBPMMode.NORMAL, fade_mode: str = "linear"):
        nbars = self.pipeline.slice_nbar
        print(f"Creating GBSB mashup with {nbars} bars")
        ff = FordFulkersonMashupper(self.pipeline, self.sample_score_id)
        table = ff.construct_table(nbars)
        table[:nbars, :] = 0 # We don't want to use the first nbars to avoid an error later on
        best_score = np.max(table)
        submitted_bar, trimmed_bar = np.where(table == best_score)
        submitted_bar, trimmed_bar = submitted_bar[0], trimmed_bar[0]
        
        # Create the new result
        new_search_config = SearchConfig(
            nbars=nbars,
            bar_number=submitted_bar,
        )
        new_pipeline = SongSearcher(
            search_config=new_search_config, 
            dataset = self.pipeline.dataset,
        )
        new_pipeline.set_link(self.pipeline.link)
        new_pipeline._audio = self.pipeline._audio
        new_pipeline._raw_beat_result = self.pipeline.raw_beat_result
        new_pipeline._raw_chord_result = self.pipeline.raw_chord_result
        new_pipeline._raw_parts_result = self.pipeline.raw_parts_result

        self._result_position = {
            "submitted_bar": submitted_bar,
            "trimmed_bar": trimmed_bar,
            "best_score": best_score,
        }
        new_song_id = f"{self.sample_url_id}/{self._result_position['trimmed_bar']}/{self.sample_transpose}"
        self._result_position["revised_score_id"] = new_song_id
        self._mashup_maker = TheseTwoSongsHaveTheSameBPM(self.pipeline, new_song_id)
        return self._mashup_maker.create(mode, fade_mode)
