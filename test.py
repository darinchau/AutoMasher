from fyp.audio.analysis import *
from fyp.audio.separation import DemucsAudioSeparator
from fyp.audio.dataset.cache import LocalCache, YouTubeURL
from fyp.audio.search.align import distance_of_chord_results
from fyp import Audio
from fyp.util.note import get_keys
import numpy as np
import librosa

def get_volume(audio: Audio, hop: int = 512):
    hop = 512
    y = audio.numpy()
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop)), ref=np.max)
    volume = spec.mean(axis = 0)

    time_arr = np.arange(volume.shape[0]) * hop / audio.sample_rate
    infos = []
    max_bar_len = 20
    for i in range(0, volume.shape[0], 10):
        bar_len = int((volume[i] + 80) / 80 * max_bar_len)
        infos.append((str(bar_len).ljust(3) + "=" * bar_len, time_arr[i]))
    return infos

def main():
    c = LocalCache("resources/cache", YouTubeURL("https://www.youtube.com/watch?v=4Q3eqJZxJPs"))
    audio = c.get_audio() or Audio.load(c.link)
    # bt = c.get_beat_analysis() or analyse_beat_transformer(audio)
    # ct = c.get_chord_analysis() or analyse_chord_transformer(audio)
    # c.store_beat_analysis(bt)
    # c.store_chord_analysis(ct)
    c.store_audio(audio)

    vocals = DemucsAudioSeparator().separate(audio)["vocals"]
    infos = get_volume(vocals)
    audio.play(info = infos)

if __name__ == "__main__":
    main()
