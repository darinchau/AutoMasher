import os
import numpy as np
import librosa
from .base import ChordAnalysisResult
from .. import Audio
from typing import Callable
from ...model.chord_btc import inference, get_idx2chord, get_idx2voca_chord
from ...util.note import get_inv_voca_map
from ...util.combine import get_video_id

def analyse_chord_transformer(audio: Audio, *, model_path: str = "../../resources/ckpts/btc_model.pt", 
                              use_voca: bool = True, 
                              device = None,
                              cache_path: str | None = None) -> ChordAnalysisResult:    
    def calculate_chord():
        results = inference(audio, model_path=model_path, use_voca = use_voca, device = device)
        chords = get_idx2chord() if not use_voca else get_idx2voca_chord()
        times = [r[0] for r in results]
        if not use_voca:
            inv_voca = get_inv_voca_map()
            labels = [inv_voca[chords[r[1]]] for r in results]
        else:
            labels = [r[1] for r in results]

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
