import re
from pytube import YouTube

_VIDEO_ID = re.compile(r"[A-Za-z0-9-_]{11}")

def to_youtube(link_or_video_id: str):
    link_or_video_id = link_or_video_id.strip()
    if _VIDEO_ID.match(link_or_video_id):
        return YouTube(f"https://www.youtube.com/watch?v={link_or_video_id}")
    return YouTube(link_or_video_id)

def get_video_id(link_or_video_id: str):
    url_id = to_youtube(link_or_video_id).video_id
    assert url_id is not None and len(url_id) == 11
    return url_id

def get_video_title(link_or_video_id: str):
    return to_youtube(link_or_video_id).title

def get_url(link_or_video_id: str):
    url = f"https://www.youtube.com/watch?v={get_video_id(link_or_video_id)}"
    try:
        from ..audio.dataset import DatasetEntry
        assert url.startswith(DatasetEntry.get_url_prepend())
    except ImportError:
        pass
    return url
