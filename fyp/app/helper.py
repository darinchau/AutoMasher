# Contains out-of-the-box code for some common operations.
# Might not be available in the demo but are useful for
# viewing intermediate results and serves as example
# for performing common operations.

from fyp.audio.analysis import BeatAnalysisResult
from fyp import Audio, YouTubeURL, get_url
from fyp.audio.cache import CacheHandler
from fyp.audio.analysis import analyse_beat_transformer

def get_click_track(yt: YouTubeURL, *,
                    cache_handler: CacheHandler | None = None,
                    model_path: str = "./resources/ckpts/beat_transformer.pt"
                    ) -> Audio:
    """Creates a click track for the given YouTube URL."""
    audio = Audio.load(yt)
    if cache_handler is not None:
        beat_result = cache_handler.get_beat_analysis_result()
    else:
        beat_result = analyse_beat_transformer(audio, model_path=model_path, use_loaded_model=True)
    click_track = beat_result.make_click_track(audio)
    return click_track
