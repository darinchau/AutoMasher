import os
from pytubefix import YouTube
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import random
from contextlib import contextmanager
import gc
import re

_VIDEO_ID = re.compile(r"[A-Za-z0-9-_]{11}")
_URL = re.compile(r"^https:\/\/www\.youtube\.com\/watch\?v=[A-Za-z0-9-_]{11}$")

class YouTubeURL(str):
    URL_PREPEND = "https://www.youtube.com/watch?v="

    def __new__(cls, value: str):
        if isinstance(value, cls):
            return value
        if len(value) == 11 and _VIDEO_ID.match(value):
            value = cls.URL_PREPEND + value
        assert value.startswith(cls.URL_PREPEND), f"Invalid YouTube URL: {value}"
        assert _URL.match(value), f"URL might not be normalized. Use get_url instead: {value}"
        assert len(value[len(cls.URL_PREPEND):]) == 11, f"Invalid video id: {value}"
        return super().__new__(cls, value)

    @property
    def video_id(self):
        """Gets the video id. Example: dQw4w9WgXcQ"""
        video_id = self[len(self.URL_PREPEND):]
        assert len(video_id) == 11, f"Invalid video id: {video_id}"
        return video_id

    @property
    def title(self):
        """Gets the title of the video."""
        return to_youtube(self).title

    def get_length(self):
        """Gets the length of the video in seconds."""
        yt = to_youtube(self)
        vid_info = yt.vid_info
        if vid_info is None:
            return None
        video_details = vid_info.get("videoDetails")
        if video_details is None:
            return None
        length = video_details.get("lengthSeconds")
        return int(length) if length is not None else None

    def get_views(self):
        """Gets the number of views of the video."""
        yt = to_youtube(self)
        vid_info = yt.vid_info
        if vid_info is None:
            return None
        video_details = vid_info.get("videoDetails")
        if video_details is None:
            return None
        views = video_details.get("viewCount")
        return int(views) if views is not None else None

def to_youtube(link_or_video_id: str):
    link_or_video_id = link_or_video_id.strip()
    if _VIDEO_ID.match(link_or_video_id):
        return YouTube(f"https://www.youtube.com/watch?v={link_or_video_id}")
    return YouTube(link_or_video_id)

def get_video_id(link_or_video_id: str):
    """Gets the video id. Example: dQw4w9WgXcQ"""
    url_id = to_youtube(link_or_video_id).video_id
    assert url_id is not None and len(url_id) == 11
    return url_id

def get_url(link_or_video_id: str) -> YouTubeURL:
    """Gets the url in the form of https://www.youtube.com/watch?v=dQw4w9WgXcQ

    This essentially normalizes the input to a YouTube URL."""
    if isinstance(link_or_video_id, YouTubeURL):
        return link_or_video_id
    try:
        url_id = to_youtube(link_or_video_id).video_id
    except Exception as e:
        raise ValueError(f"Invalid YouTube URL or video id: {link_or_video_id}") from e
    assert url_id is not None and len(url_id) == 11
    url = f"{YouTubeURL.URL_PREPEND}{url_id}"
    return YouTubeURL(url)

# A little function to clear cuda cache. Put the import inside just in case we do not need torch, because torch import takes too long
def clear_cuda():
    import torch
    gc.collect()
    torch.cuda.empty_cache()

def is_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_posix():
    return os.name == 'posix'

# Convert video file to .wav format with 48kHz sample rate
def convert_to_wav(video_path, output_path, sr=48000, timeout=120, verbose = True):
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        assert audio is not None
        wav_filename: str = os.path.splitext(video_path)[0] + ".wav"

        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)

        logger = 'bar' if verbose else None
        audio.write_audiofile(os.path.join(output_path, wav_filename), fps=sr, verbose=verbose, logger=logger)
        return wav_filename
    except Exception as e:
        return "Error converting to wav", e

# Download video from YouTube given a URL and an output path
def download_video(yt: YouTube, output_path: str, verbose=True, timeout=120):
    def progress_callback(stream, chunk, bytes_remaining):
        nonlocal progress_bar
        progress_bar.update(len(chunk))

    try:
        # Remake the youtube object for the progess bar
        if verbose:
            print(f"Downloading: {yt.title}")

        video = YouTube(yt.watch_url, on_progress_callback=progress_callback).streams.filter(file_extension='mp4').get_lowest_resolution()
        assert video is not None
        progress_bar = tqdm(total=video.filesize, unit='B', unit_scale=True, ncols=100, disable=not verbose)
        video.download(output_path=output_path, timeout=timeout)
        return os.path.join(output_path, video.default_filename)
    except Exception as e:
        return "Error downloading video", e

def download_audio_with_yt_dlp(link: str, output_dir: str, verbose=True):
    try:
        from yt_dlp.YoutubeDL import YoutubeDL
    except ImportError:
        raise ImportError("Please install yt-dlp to use this function")
    ydl = YoutubeDL({
        "format": "bestaudio/best",
        "output": "%(title)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": not verbose,
        "windowsfilenames": not is_posix(),
        "paths": {
            "home": output_dir
        }
    })
    retcode = ydl.download([link])
    title = ydl.extract_info(link, download=False)['title']

    # Find the downloaded file
    for file in os.listdir(output_dir):
        if title in file and file.endswith('.wav'):
            return os.path.join(output_dir, file)
    raise FileNotFoundError(f"Could not find downloaded file for {link}: {retcode}")

def download_audio_with_pytube(link: str, output_dir: str, verbose=True, timeout=120) -> str:
    yt = YouTube(link)
    video_path = download_video(yt, output_dir, verbose=verbose, timeout=timeout)
    if isinstance(video_path, tuple):
        raise RuntimeError(f"Error downloading video: {video_path[1]}")

    audio_path = convert_to_wav(video_path, output_dir, verbose = False)
    if isinstance(audio_path, tuple):
        raise RuntimeError(f"Error converting to wav: {audio_path[1]}")
    return audio_path

# More often than not we only want the audio so here is one combined function
def download_audio(link: str, output_dir: str, verbose=True, timeout=120):
    try:
        return download_audio_with_pytube(link, output_dir, verbose=verbose, timeout=timeout)
    except Exception as e:
        return download_audio_with_yt_dlp(link, output_dir, verbose=verbose)
