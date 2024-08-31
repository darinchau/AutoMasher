import re
from pytube import YouTube

_VIDEO_ID = re.compile(r"[A-Za-z0-9-_]{11}")

class YouTubeURL(str):
    URL_PREPEND = "https://www.youtube.com/watch?v="

    def __new__(cls, value: str):
        if isinstance(value, cls):
            return value
        if len(value) == 11 and _VIDEO_ID.match(value):
            value = cls.URL_PREPEND + value
        assert value.startswith(cls.URL_PREPEND)
        assert _VIDEO_ID.match(value[len(cls.URL_PREPEND):])
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

def get_video_title(link_or_video_id: str):
    return to_youtube(link_or_video_id).title

def get_url(link_or_video_id: str) -> YouTubeURL:
    """Gets the url in the form of https://www.youtube.com/watch?v=dQw4w9WgXcQ"""
    url = f"https://www.youtube.com/watch?v={get_video_id(link_or_video_id)}"
    try:
        from ..audio.dataset import DatasetEntry
        assert url.startswith(DatasetEntry.get_url_prepend())
    except ImportError:
        pass
    return YouTubeURL(url)
