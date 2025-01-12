# Ideally this module will give us tools to analyse an audio
# i.e. given audio, return tonality and pitch

from .beat import (
    analyse_beat_transformer,
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

from .base import (
    OnsetFeatures,
    DiscreteLatentFeatures,
    ContinuousLatentFeatures,
    dist_discrete_latent_features,
    dist_continuous_latent_features,
)

from .beat import BeatAnalysisResult, DeadBeatKernel
from .chord import ChordAnalysisResult, SimpleChordAnalysisResult
