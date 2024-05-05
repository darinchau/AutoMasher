# Ideally this module will give us tools to analyse an audio
# i.e. given audio, return tonality and pitch

from .key import (
    analyse_key_center,
    analyse_key_center_chroma
)
from .beat import (
    analyse_beat_transformer,
    analyse_beat_transformer_local,
)
from .chord import (
    analyse_chord_transformer,
)
from .chroma import (
    chroma_cens,
    chroma_cqt,
    chroma_stft,
    ChromaFunction
)
from .beat import BeatAnalysisResult
from .key import KeyAnalysisResult
from .tuning import analyse_tuning
from .tuning import TuningAnalysisResult

from .base import *
from ...audio import Audio
from typing import Callable
from .time_seg import pychorus, top_k_edge_detection, top_k_maxpool, top_k_rms, top_k_stft

## Utility functions to compare BeatAnalyser and the smoothener that I am too reluctant to throw away
## Returns a list of the results because its a waste to throw them away either
def compare_bpm_trend(audio: Audio, beatanalysers: list[Callable[[Audio], BeatAnalysisResult]], names: list[str] | None = None) -> list[BeatAnalysisResult]:
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    if names is None:
        names = [b.__class__.__name__ for b in beatanalysers]
    else:
        assert len(beatanalysers) == len(names) > 0

    results = []

    plt.figure()

    for analyser, name in zip(beatanalysers, names):
        t1 = time.time()
        result = analyser(audio)
        t2 = time.time()

        beats = np.array(result.beats)
        bpm_diff = 1 / (beats[1:] - beats[:-1]) * 60
        bpm_diff_x = (beats[1:] + beats[:-1]) / 2

        print(f"({name}) Mean: {bpm_diff.mean()}")

        if hasattr(result, "smooth_heuristics"):
            print(f"({name}) Best smoothery: {result.smooth_heuristics}") #type: ignore

        print(f"({name}) Time: {round(t2 - t1, 4)}s")
        print()

        results.append(result)

        # Plot the x-y pairs
        plt.plot(bpm_diff_x, bpm_diff, label=name)
    
    # Add labels and legends
    plt.xlabel("Frame")
    plt.ylabel("BPM")
    plt.legend()

    # Show the plot
    plt.show()

    return results
