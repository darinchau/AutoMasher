from ... import Audio
from .key import analyse_key_center, KeyAnalysisResult
from .beat import analyse_beat_transformer, BeatAnalysisResult


def analyse_cadence(audio: Audio, hop = 512) -> BeatAnalysisResult:
    """Analyse the cadence of an audio file"""
    return analyse_beat_transformer(audio, hop)
def analyse_cadence(audio: Audio, bar_number: int, bt: BeatAnalysisResult | None = None, hop = 512) -> CadenceAnalysisResult:
    """Analyse the cadence of an audio file. Bar number must be at least 2 and less than the number of bars in the song."""
    if bt is None:
        bt = analyse_beat_transformer(audio)

    if bar_number < 2 or bar_number >= bt.nbars:
        raise ValueError(f"Invalid bar number (2 <= bar_number < {bt.nbars})")

    key = analyse_key_center(audio.slice_seconds(max(0, bt.downbeats[bar_number] - 10), bt.downbeats[bar_number + 1]), hop=hop)

    two_bar_length = bt.downbeats[bar_number] - bt.downbeats[bar_number - 2]
