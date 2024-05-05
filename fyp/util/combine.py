from datasets import Dataset
from typing import Any, Callable
from pytube import YouTube
import matplotlib.pyplot as plt
import librosa
import numpy as np

DatasetType = list[dict[str, Any]] | Dataset

def filter_dataset(dataset: DatasetType, filter_func: Callable[[dict[str, Any]], bool] | None):
    if filter_func is None:
        return dataset
    if isinstance(dataset, Dataset):
        return dataset.filter(filter_func)
    return [e for e in dataset if filter_func(e)]

def get_entry_from_database(dataset: DatasetType, video_id_or_url: str):
    """Given a youtube link or video id, returns the entry from the database. Raises ValueError if not found."""
    url = video_id_or_url if video_id_or_url.startswith("https://") else f"https://www.youtube.com/watch?v={video_id_or_url}"
    entry: dict[str, Any] = {}
    for e in dataset:
        if e['url'] == url: #type: ignore
            for key in ("length", "beats", "downbeats", "audio_name", "views"):
                entry[key] = e[key]  #type: ignore
            entry['url'] = url
            break
    
    if entry is None:
        raise ValueError("Entry not found")
    return entry

def get_video_id(link_or_video_id: str):
    return YouTube(get_url(link_or_video_id)).video_id

def get_video_title(link_or_video_id: str):
    return YouTube(get_url(link_or_video_id)).title

def get_url(link_or_video_id: str):
    if link_or_video_id.startswith("https://"):
        return link_or_video_id
    return f"https://www.youtube.com/watch?v={link_or_video_id}"

def show_audio_spectrogram(audio, link: str):
    from .. import Audio
    assert isinstance(audio, Audio)
    
    plt.style.use('dark_background')
    y = audio.numpy()
    sr = audio.sample_rate
    fig, ax = plt.subplots()
    hop_length = 1024
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time', ax=ax)
    ax.set(title=f'{get_video_title(link)}')
    ax.label_outer()
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return fig

def prepare_output(score: float, score_id: str, path: str):
    from .. import Audio
    def four_digit_escape(string):
        # converts characters to unicode, ie "¥" to "\u00a5"
        # return u''.join(char if 32 <= ord(char) <= 126 else u'\\u%04x'%ord(char) for char in string)
        return u''.join(u'%04x'%ord(char) for char in string)
    url_id, start_idx, transpose = score_id.split('/')[:3]
    title = get_video_title(url_id)
    audio_path = path.format(score_id = score_id.replace("/", "+"))
    audio = Audio.load(get_url(url_id))
    audio.save(audio_path)
    return f"{url_id}\n({round(score, 2)}% match) {four_digit_escape(title)}\n{audio_path}\n"
