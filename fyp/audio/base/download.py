# Exports a single function: get_audio(YouTubeURL) -> Audio that downloads the audio

import os
from pytubefix import YouTube
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from ...util import is_posix, YouTubeURL
import random
from contextlib import contextmanager
import gc
import tempfile
import re
from .audio import Audio


def convert_to_wav(video_path, output_path, sr=48000, timeout=120, verbose=True):
    # Convert video file to .wav format with 48kHz sample rate
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


def download_video(yt: YouTube, output_path: str, verbose=True, timeout=120):
    # Download video from YouTube given a URL and an output path
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


def download_ytdlp_inner(link: str, proxy: str, output_dir: str, verbose=True):
    try:
        from yt_dlp.YoutubeDL import YoutubeDL, DownloadError
    except ImportError:
        raise ImportError("Please install yt-dlp to use this function")
    ydl_opts = {
        "format": "bestaudio/best",
        "output": "%(id)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": not verbose,
        "windowsfilenames": not is_posix(),
        "paths": {
            "home": output_dir
        },
    }
    with YoutubeDL(ydl_opts) as ydl:
        try:
            result = ydl.extract_info(link, download=True)
            video = result.get('entries', [result])[0]
            file_path = ydl.prepare_filename(video)
            return file_path
        except DownloadError as e:
            raise RuntimeError(f"Could not find downloaded file for {link}: {e}")


def download_pytube_inner(link: str, output_dir: str, verbose=True, timeout=120) -> str:
    yt = YouTube(link)
    video_path = download_video(yt, output_dir, verbose=verbose, timeout=timeout)
    if isinstance(video_path, tuple):
        raise RuntimeError(f"Error downloading video: {video_path[1]}")

    audio_path = convert_to_wav(video_path, output_dir, verbose=False)
    if isinstance(audio_path, tuple):
        raise RuntimeError(f"Error converting to wav: {audio_path[1]}")

    return audio_path


def download_pytube(link: YouTubeURL):
    # Download to a temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            audio_path = download_pytube_inner(link, temp_dir, timeout=120)
            audio = Audio.load(audio_path)
        except Exception as e:
            raise RuntimeError(f"Error downloading audio: {e}")
    return audio


def download_ytdlp(link: YouTubeURL):
    # Download to a temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            audio_path = download_ytdlp_inner(link, temp_dir)
            audio = Audio.load(audio_path)
        except Exception as e:
            raise RuntimeError(f"Error downloading audio: {e}")
    return audio


def get_audio(link: str) -> Audio:
    trials = [
        lambda: download_pytube(link),
        lambda: download_ytdlp(link)
    ]
    for trial in trials:
        try:
            return trial()
        except Exception as e:
            print(f"Error: {e}")
            continue
    raise RuntimeError(f"Failed to download audio from {link} after {len(trials)} attempts.")
