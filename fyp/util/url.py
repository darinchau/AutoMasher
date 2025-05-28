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


def is_posix():
    return os.name == 'posix'
