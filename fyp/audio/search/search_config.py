from dataclasses import dataclass
from typing import Callable
from .. import Audio, AudioCollection
from ..analysis import ChordAnalysisResult, BeatAnalysisResult
from tqdm.auto import tqdm
from typing import Any
from copy import deepcopy

def has_madmom():
    """Check if madmom is installed"""
    try:
        import madmom
        return True
    except ImportError:
        return False
    
HAS_MADMOM = has_madmom()

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

        skip_search: Whether to skip the search and only perform the analysis. This parameter is useful in some internal
            use cases. Default is False.

        filter_dataset: A lambda function to be used in the search. The filter should take in a dataset entry and return
            a boolean. If the filter returns True, the entry will be included in the search. When this parameter is set
            to None, the search will include all entries. Default is None.

        filter_first: Whether to include only the best result of each song from the search. This does not affect the
            runtime of the search since the filtering is done after the search. Default is True.

        backend_url: The URL of the beat transformer backer-end. Defaults to None, and is only used when use_request_beat_transformer
            is set to True.

        chord_model_path: The path to the chord model. Default is "resources/ckpts/btc_model_large_voca.pt", which is the
            model path on a fresh clone of the repository from the root

        beat_model_path: The path to the beat model. Default is "resources/ckpts/beat_transformer.pt", which is the model
            path on a fresh clone of the repository from the root

        cache_dir: The directory to store the cache. If set to None, will disable caching. Default is "./", which is the current directory.

        cache: Whether to cache the results. If set to False, the pipeline will force recomputation on every search. Default is True.

        verbose_progress: Whether to show the debug progress bars during the search. Default is False.

        use_request_beat_transformer: Whether to use the request beat transformer. If set to True, the pipeline will use the
            beat transformer backer-end. Default is True if madmom is not installed, False otherwise.
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
    skip_search: bool = False
    filter_dataset: Callable[[dict], bool] | None = None
    filter_first: bool = True
    backend_url: str | None = "https://unique-closing-gecko.ngrok-free.app"
    chord_model_path: str = "resources/ckpts/btc_model_large_voca.pt"
    beat_model_path: str = "resources/ckpts/beat_transformer.pt"
    cache_dir: str | None = "./"
    cache: bool = True
    verbose_progress: bool = False
    use_request_beat_transformer: bool = not HAS_MADMOM

    @staticmethod
    def parse(cmd: str):
        """Parses the command line arguments"""
        return SearchConfig() ## TODO: implement
        args = cmd.split()
        config = SearchConfig()
        for arg in args:
            if "=" not in arg:
                continue
            key, value = arg.split("=")
            # Check if key is an attribute of SearchConfig
            if hasattr(config, key):
                # Potentially unsafe eval but oh well
                setattr(config, key, eval(value))
        return config
    
    def clone(self, **kwargs: Any):
        """Clones the SearchConfig with the new attributes"""
        new_kwargs = deepcopy(vars(self))
        new_kwargs.update(kwargs)
        new_config = SearchConfig(**new_kwargs)
        return new_config

class SongSearchCallbackHandler:
    def on_search_start(self, link: str):
        """Called when the search is about to start. Link is the link of the audio."""
        pass

    def on_search_end(self, scores: list[tuple[float, str]]):
        """Called when the search is done. Scores is a list of tuples, where the first element is the score and the second element is the url."""
        pass

    def on_chord_transformer_start(self, audio: Audio):
        """Called when the chord transformer is about to start transforming the audio"""
        print("Chord transformer started", flush=True)

    def on_chord_transformer_end(self, chord_result: ChordAnalysisResult):
        """Called when the chord transformer is done transforming the audio"""
        print("Chord transformer ended", flush=True)

    def on_beat_transformer_start(self, audio: Audio):
        """Called when the beat transformer is about to start transforming the audio"""
        print("Beat transformer started", flush=True)

    def on_beat_transformer_end(self, beat_result: BeatAnalysisResult):
        """Called when the beat transformer is done transforming the audio"""
        print("Beat transformer ended", flush=True)

    def on_demucs_start(self, audio: Audio):
        """Called when the demucs audio separator is about to start separating the audio"""
        pass

    def on_demucs_end(self, parts: AudioCollection):
        """Called when the demucs audio separator is done separating the audio"""
        pass

    def on_load(self, audio: Audio | None, link: str | None):
        """Called when the audio is loaded from the link"""
        pass

    def on_chorus_start(self, audio: Audio):
        """Called when the chorus extraction is about to start"""
        print("Time segmentation extraction started", flush=True)

    def on_chorus_end(self, downbeat: float):
        """Called when the chorus extraction is done"""
        print("Time segmentation extraction ended", flush=True)

    def on_search_database_entry(self, progress: int, total: int):
        self._tqdm.update(1)

    def on_search_database_start(self, total: int):
        self._tqdm = tqdm(total=total, desc="Searching database", unit="entry")

    def on_search_database_end(self, total: int):
        self._tqdm.close()

class Shutup(SongSearchCallbackHandler):
    """A callback handler that silences all the messages."""
    def on_chord_transformer_start(self, audio: Audio):
        pass

    def on_chord_transformer_end(self, chord_result: ChordAnalysisResult):
        pass

    def on_beat_transformer_start(self, audio: Audio):
        pass

    def on_beat_transformer_end(self, beat_result: BeatAnalysisResult):
        pass
        
    def on_chorus_start(self, audio: Audio):
        """Called when the chorus extraction is about to start"""
        pass

    def on_chorus_end(self, downbeat: float):
        """Called when the chorus extraction is done"""
        pass
        
    def on_search_database_entry(self, progress: int, total: int):
        pass

    def on_search_database_start(self, total: int):
        pass

    def on_search_database_end(self, total: int):
        pass
