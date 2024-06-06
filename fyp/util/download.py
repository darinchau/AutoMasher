import os
from pytube import YouTube
from tqdm import tqdm
from moviepy.editor import VideoFileClip

# Convert video file to .wav format with 48kHz sample rate
def convert_to_wav(video_path, output_path, sr=48000, timeout=120, verbose = True):
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
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
        "windowsfilenames": True,
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
