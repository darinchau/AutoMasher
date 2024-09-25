import re
from pytube import YouTube

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
        return super().__new__(cls, value)

    @property
    def video_id(self):
        return self[len(self.URL_PREPEND):]

    @property
    def title(self):
        return to_youtube(self).title

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
    url = f"https://www.youtube.com/watch?v={get_video_id(link_or_video_id)}"
    return YouTubeURL(url)
