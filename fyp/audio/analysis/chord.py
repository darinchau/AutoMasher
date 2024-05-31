import os
import numpy as np
import librosa
from .base import ChordAnalysisResult
from .. import Audio
from typing import Callable
from ...model import chord_inference as inference
from ...util.note import get_idx2voca_chord
from ...util.note import get_inv_voca_map

def analyse_chord_transformer(audio: Audio, *, model_path: str = "../../resources/ckpts/btc_model_large_voca.pt", 
                              use_loaded_model: bool = True,
                              cache_path: str | None = None) -> ChordAnalysisResult:    
    def calculate_chord():
        results = inference(audio, model_path=model_path, use_loaded_model=use_loaded_model)
        chords = get_idx2voca_chord()
        times = [r[0] for r in results]
        inv_voca = get_inv_voca_map()
        labels = [inv_voca[chords[r[1]]] for r in results]

        cr = ChordAnalysisResult(
            audio.duration,
            labels = labels,
            times = times,
        )
        return cr

    if cache_path is not None and os.path.isfile(cache_path):
        return ChordAnalysisResult.load(cache_path)
    
    cr = calculate_chord()
    if cache_path is not None:
        cr.save(cache_path)

    return cr
