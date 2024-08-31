import matplotlib.pyplot as plt
import librosa
import numpy as np
from .url import YouTubeURL

def show_audio_spectrogram(audio, link: YouTubeURL):
    from .. import Audio
    assert isinstance(audio, Audio)

    plt.style.use('dark_background')
    y = audio.numpy()
    sr = audio.sample_rate
    fig, ax = plt.subplots()
    hop_length = 1024
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time', ax=ax)
    ax.set(title=f'{link.title}')
    ax.label_outer()
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return fig
