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
        audio.write_audiofile(os.path.join(output_path, wav_filename), fps=sr, verbose=verbose, logger=logger) #type: ignore
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
        progress_bar = tqdm(total=video.filesize, unit='B', unit_scale=True, ncols=100, disable=not verbose) #type:ignore
        video.download(output_path=output_path, timeout=timeout) #type: ignore
        return os.path.join(output_path, video.default_filename) #type: ignore
    except Exception as e:
        return "Error downloading video", e
