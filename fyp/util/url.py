import os
from tqdm import tqdm
import random
from contextlib import contextmanager
import gc
import re

_VIDEO_ID = re.compile(r"[A-Za-z0-9-_]{11}")
_URL = re.compile(r"^https:\/\/www\.youtube\.com\/watch\?v=[A-Za-z0-9-_]{11}$")


class YouTubeURL(str):
    URL_PREPEND = "https://www.youtube.com/watch?v="

    @staticmethod
    def get_placeholder():
        """Gets a placeholder YouTube URL for when you don't have/need a real one."""
        return YouTubeURL("https://www.youtube.com/watch?v=placeholder")

    @property
    def is_placeholder(self):
        return self == self.get_placeholder()

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
    def video_title(self):
        """Gets the title of the video."""
        try:
            return to_youtube(self).title
        except Exception as e:
            print(f"Error getting title: {e}")
            return "*Could not get title*"

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
    from pytubefix import YouTube
    link_or_video_id = link_or_video_id.strip()
    if _VIDEO_ID.match(link_or_video_id):
        return YouTube(f"{YouTubeURL.URL_PREPEND}{link_or_video_id}")
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


def clear_cuda():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def is_posix():
    return os.name == 'posix'


def is_file_in_directory(path: str, dir: str) -> bool:
    normalized_path = os.path.abspath(path)
    normalized_dir = os.path.abspath(dir)
    if os.path.isfile(normalized_path):
        file_dir = os.path.dirname(normalized_path)
        return file_dir == normalized_dir
    return False


def download_audio_with_yt_dlp(link: YouTubeURL, output_dir: str, port: int | None = None, verbose=True, max_retries=3):
    try:
        from yt_dlp.YoutubeDL import YoutubeDL
    except ImportError:
        raise ImportError("Please install yt-dlp to use this function")

    args = {
        "format": "bestaudio",  # Get the best available audio quality
        "output": "%(id)s.%(ext)s",  # Save the file with its original format extension
        "quiet": not verbose,
        "windowsfilenames": not is_posix(),
        "paths": {
            "home": output_dir
        }
    }

    if port is not None:
        args["proxy"] = f"http://localhost:{port}"

    retries = 0
    while retries < max_retries:
        try:
            ydl = YoutubeDL(args)
            retcode = ydl.download([link])
            if retcode == 0:  # Check if download was successful
                # Find the downloaded file
                for file in os.listdir(output_dir):
                    if link.video_id in file:
                        return os.path.join(output_dir, file)
            retries += 1
        except Exception as e:
            print(f"Attempt {retries+1} failed: {e}")
            retries += 1
            if retries >= max_retries:
                raise Exception("Maximum retries reached, download failed.")

    raise FileNotFoundError(f"Could not find downloaded file for {link} after {max_retries} attempts.")


def download_audio(link: str | YouTubeURL, output_dir: str, verbose=True, timeout=120, port: int | None = None):
    """Downloads the audio from a YouTube link using yt-dlp. Returns the file name of the downloaded audio file."""
    link = get_url(link)
    path = download_audio_with_yt_dlp(link, output_dir, verbose=verbose, port=port)
    # Make sure path points to a file in output_dir
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Downloaded file not found: {path}. Ensure yt-dlp is installed and the link is valid.")
    if not is_file_in_directory(path, output_dir):
        raise ValueError(f"Downloaded file {path} is not in the specified output directory {output_dir}.")
    path = os.path.basename(path)
    return path
