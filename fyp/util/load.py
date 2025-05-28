import os
import sys
import audiofile
import torch
import warnings
import numpy as np
from .url import get_url, YouTubeURL, to_youtube, is_posix


def load_audio(path: str, sr: int | None = None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        wav, sr = audiofile.read(path, always_2d=True)
    except ImportError:
        if sys.version_info >= (3, 13):
            raise RuntimeError(f"You might need to install aifc from the deadlib repository to make this work: `pip install standard-aifc standard-sunau`") from e
        raise
    # Do some basic checks
    assert isinstance(wav, np.ndarray), f"Audio file {path} is not a valid numpy array (type={type(wav)})"
    assert wav.ndim == 2, f"Audio file {path} is not stereo or mono (ndim={wav.ndim})"
    if not (wav.max() <= 1.0 and wav.min() >= -1.0):
        msg = f"Audio file {path} has values outside the range [-1.0, 1.0] (max={wav.max()}, min={wav.min()})"
        warnings.warn(msg, UserWarning)
    assert wav.dtype in [np.float32, np.float64], f"Audio file {path} has an unsupported dtype {wav.dtype}. Expected float32 or float64."
    wav = torch.from_numpy(wav).float()
    return wav, sr


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
