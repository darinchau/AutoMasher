from dataclasses import dataclass
from typing import Callable
from .. import Audio, AudioCollection
from ..dataset import DatasetEntry
from ..analysis import ChordAnalysisResult, BeatAnalysisResult
from tqdm.auto import tqdm
from typing import Any
from copy import deepcopy

@dataclass(frozen=True)
class SearchConfig:
    """
    The configuration for the search.

    Attributes:
        max_transpose: The maximum number of semitones to transpose the audio. If a tuple, 
            it will represent the range of transposition. If an integer, it will represent 
            the maximum transposition (equivalent to (-k, k)). Default is 3.
        
        min_music_percentage: The minimum percentage of music in the audio. Default is 0.8.

        max_delta_bpm: The maximum bpm deviation allowed for the queried song. Default is 1.25.

        min_delta_bpm: The minimum bpm deviation allowed for the queried song. Default is 0.8.

        keep_first_k: The number of top results to keep. Default is 20. Set to -1 to keep all results.

        min_score: The minimum similarity percentage that the song should get to be included in the results. Default is 0.

        max_score: The maximum similarity percentage that the song should get to be included in the results. Default is 100.

        extra_info: Extra information to be included in the score_id for better display. We will use format string
            to include the extra information in the search. Available format strings are title, timestamp, genre, views.
            Default is an empty string.

        pychorus_work_factor: The work factor for the pychorus library. Must be between 10 and 20. The higher the number,
            the less accurate but also the less runtime. This parameter scales (inverse) exponentially to the runtime i.e.
            A work factor of 13 ought to have twice the runtime compared to work factor of 14. Default is 14.

        progress_bar: Whether to show the progress bar during the search. Default is True.

        bar_number: The bar number (according to the beat analysis result) to slice the song. If set to None, the pipeline
            will perform time segmentation and slice the song according to the chorus. Default is None.

        nbars: The number of bars to slice the song. If set to None, the pipeline will perform time segmentation and slice
            the song according to the chorus, and set nbars to 8 bars. Default is None.

        filter_func: A lambda function to be used in the search. The filter should take in a dataset entry and return
            a boolean. If the filter returns True, the entry will be included in the search. When this parameter is set
            to None, the search will include all entries. Default is None.

        filter_first: Whether to include only the best result of each song from the search. This does not affect the
            runtime of the search since the filtering is done after the search. Default is True.

        dataset_path: The path to the dataset. Default is "hkust-fypho2", which is the dataset path on hugging face.
            Feel free to keep this default value because hugging face will handle caching for us.

        chord_model_path: The path to the chord model. Default is "resources/ckpts/btc_model_large_voca.pt", which is the
            model path on a fresh clone of the repository from the root

        beat_model_path: The path to the beat model. Default is "resources/ckpts/beat_transformer.pt", which is the model
            path on a fresh clone of the repository from the root

        cache_dir: The directory to store the cache. If set to None, will disable caching. Default is "./", which is the current directory.

        cache: Whether to cache the results. If set to False, the pipeline will force recomputation on every search. Default is True.

        verbose: Whether to show the progress bars during the search. Default is False.
    """
    max_transpose: int | tuple[int, int] = 3
    min_music_percentage: float = 0.8
    max_delta_bpm: float = 1.25
    min_delta_bpm: float = 0.8
    keep_first_k: int = 20
    min_score: int = 0
    max_score: int = 100
    extra_info: str = ""
    pychorus_work_factor: int = 14
    progress_bar: bool = True
    bar_number: int | None = None
    nbars: int | None = None
    filter_func: Callable[[DatasetEntry], bool] | None = None
    filter_first: bool = True
    chord_model_path: str = "resources/ckpts/btc_model_large_voca.pt"
    beat_model_path: str = "resources/ckpts/beat_transformer.pt"
    dataset_path: str = "HKUST-FYPHO2/audio-infos-filtered"
    cache_dir: str | None = "./"
    cache: bool = True
    verbose: bool = False
    
    def clone(self, **kwargs: Any):
        """Clones the SearchConfig with the new attributes"""
        new_kwargs = deepcopy(vars(self))
        new_kwargs.update(kwargs)
        new_config = SearchConfig(**new_kwargs)
        return new_config
