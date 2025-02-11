# Isolate a single point of failure to madmom
# This allows us to remove madmom dependency in the future if needed

import numpy as np
import itertools


def require_madmom():
    try:
        from madmom.features.beats import DBNBeatTrackingProcessor
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    except ImportError:
        raise ImportError("Please install madmom to use this function. Install madmom from `pip install git+https://github.com/darinchau/madmom`")


def unpack_beats(activations: np.ndarray, *,
                 min_bpm: float, max_bpm: float, fps: float, threshold: float,
                 num_tempi: int | None = None,
                 transition_lambda: float = 100,
                 observation_lambda: float = 6
                 ) -> np.ndarray:
    from madmom.features import DBNBeatTrackingProcessor
    beat = DBNBeatTrackingProcessor(min_bpm=min_bpm, max_bpm=max_bpm, fps=fps, threshold=threshold,
                                    num_tempi=num_tempi, transition_lambda=transition_lambda, observation_lambda=observation_lambda)
    return beat(activations)


def unpack_downbeats(activations: np.ndarray, beats_per_bar: list[int], *,
                     min_bpm: float, max_bpm: float, fps: float, threshold: float,
                     num_tempi: int | None = None,
                     transition_lambda: float = 100,
                     observation_lambda: float = 6
                     ) -> np.ndarray:
    from madmom.features import DBNDownBeatTrackingProcessor
    downbeat = DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar, min_bpm=min_bpm, max_bpm=max_bpm, fps=fps, threshold=threshold,
                                            num_tempi=num_tempi, transition_lambda=transition_lambda, observation_lambda=observation_lambda)
    return downbeat(activations)
