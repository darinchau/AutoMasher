import os
from pytube import YouTube
from pytube import Playlist
from fyp import Audio
from fyp.audio.separation import HPSSAudioSeparator
from fyp.audio.analysis import analyse_beat_transformer, analyse_chord_transformer
from fyp.util.download import download_video, convert_to_wav
import concurrent.futures
import gc
from datasets import Dataset, load_from_disk
import glob
import time
import traceback
from tqdm.auto import tqdm

columns = [
    "chords", # list[int], index of the chord
    "chord_times",  # list[float], start time of the nth chord
    "beats", # list[float], start time of the nth beat
    "downbeats", # list[float], start time of the nth downbeat
    "sample_rate", # int, sample rate of the audio
    "genre", # str, genre of the audio
    "audio_name", # str, name of the audio (rather, the name of the video on YT from which the audio is extracted)
    "url", # str, url of the video
    "playlist", # str, url of the playlist
    "time_accessed", # int, time when the data is accessed
    "views", # int, number of views
    "length", # int, length of the video in seconds
    "rating", # float, rating of the video. Can be None
    "age_restricted" # bool, whether the video is age restricted
]

dataset_path = "./audio-infos"
GPU_THRESHOLD = 210 # seconds of audio such that we use gpu host. Anything longer will use cpu host
FORCE_CPU = False # Force CPU usage for beat detection

# Return true if this song should be processed, false otherwise
# Duplicate handling should not be queried here
def filter_song(yt: YouTube):
    if yt.length > 600 or yt.length < 120:
        return False
    
    if yt.age_restricted:
        return False
    
    if yt.views < 5e5:
        return False
    
    return True

def clear_output():
    try:
        get_ipython
        from IPython.display import clear_output as ip_clear_output
        ip_clear_output()
    except (ImportError, NameError) as e:
        os.system("cls" if "nt" in os.name else "clear")

def write_error(error: str, exec: Exception):
    with open("error.txt", "a") as file:
        file.write(f"{error}: {exec}\n")
        file.write("".join(traceback.format_exception(exec)))
        file.write("\n\n")
        print("ERROR: " + error)

def get_video_urls(playlist_url):
    playlist = Playlist(playlist_url)
    return playlist.video_urls

def is_youtube_playlist(link):
    return "playlist?list=" in link

def get_data_from_video_url(yt: YouTube, sr: int, timeout = 120):
    video_path = download_video(yt, "./", timeout=timeout)

    if isinstance(video_path, tuple):
        write_error(video_path[0], video_path[1])
        return
    
    audio_path = convert_to_wav(video_path, "./", sr = sr, timeout=timeout)
    if isinstance(audio_path, tuple):
        write_error(audio_path[0], audio_path[1])
        return
    
    gc.collect() 
    
    # Remove both files
    try:
        os.remove(video_path)
    except OSError as e:
        write_error(f"Error removing video file", e)
        pass

    return audio_path

def log(message: str, verbose: bool = True):
    if verbose:
        print(message)

# Calculate chord and beat info for a given audio file
def calculate_chord_info(audio_path: str, verbose: bool = True):
    log(f"Calculating chord info for {audio_path}", verbose=verbose)

    audio = Audio.load(audio_path)
    separator = HPSSAudioSeparator(return_percussive=False)
    separate_results = separator.separate_audio(audio)

    log(f"Separation complete", verbose=verbose)

    chord_results = analyse_chord_transformer(separate_results["harmonic"])

    log(f"Chord analysis complete", verbose=verbose)
    return chord_results.grouped_labels, chord_results.grouped_times

# Analyse with GPU if audio is short enough
def analyse_beat_with_cpu_fallback(audio: Audio, verbose: bool = True):
    def analyse_with_cpu(a: Audio):
        print(">>> Using CPU instead...")
        return analyse_beat_transformer(a, url="http://localhost:8123", verbose=verbose, do_normalization=False)

    if audio.duration < GPU_THRESHOLD and not FORCE_CPU:
        print(">>> Audio is short enough, using GPU...")
        try:
            return analyse_beat_transformer(audio, url="http://localhost:8124", verbose=verbose, with_fallback=analyse_with_cpu, do_normalization=False)
        except Exception as e:
            # Allows us to actively repair the GPU API if it crashes without compromising the CPU side
            write_error(f"Error analysing with GPU", e)
            return analyse_with_cpu(audio)
    return analyse_with_cpu(audio)

def calculate_beat_info(audio_path: str, verbose: bool = True):
    log(f"Calculating beat info for {audio_path}", verbose=verbose)

    audio = Audio.load(audio_path)
    beat_results = analyse_beat_with_cpu_fallback(audio)
    
    log(f"Beat analysis complete", verbose=verbose)
    beat_frames = beat_results.beats
    downbeat_frames = beat_results.downbeats
    return beat_frames, downbeat_frames

def process_video_url(dataset: dict[str, list], video_url: str, playlist_url: str, sr: int, verbose: bool = True, timeout=120, genre="pop", use_low_resource_mode: bool = False):
    # Make the YouTube object
    yt = YouTube(video_url)

    # If too long or too short then return
    if not filter_song(yt):
        print(f"Video filtered: {video_url}")
        return
    
    # Get audio data
    audio_path = get_data_from_video_url(yt, sr = sr, timeout=timeout)
    if audio_path is None:
        return False
    
    # Calculate chord and beat info
    try:
        if use_low_resource_mode:
            # Use single thread
            chord_info = calculate_chord_info(audio_path, verbose=verbose)
            beat_info = calculate_beat_info(audio_path, verbose=verbose)
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                chord_future = executor.submit(calculate_chord_info, audio_path, verbose=verbose)
                beat_future = executor.submit(calculate_beat_info, audio_path, verbose=verbose)
            
            chord_info = chord_future.result()
            beat_info = beat_future.result()
    except Exception as e:
        write_error(f"Error getting results", e)
        return False
    
    labels, times = chord_info
    beats, downbeats = beat_info    
    gc.collect()

    # Remove all WAV files in the directory
    wav_files = glob.glob("*.wav")
    for file in wav_files:
        try:
            os.remove(file)
        except OSError as e:
            write_error(f"Error removing audio file", e)
            pass

    # Remove all MP4 files in the directory
    wav_files = glob.glob("*.mp4")
    for file in wav_files:
        try:
            os.remove(file)
        except OSError as e:
            write_error(f"Error removing video file", e)
            pass

    # Save to dataset
    dataset["chords"].append(labels)
    dataset["chord_times"].append(times)
    dataset["beats"].append(beats)
    dataset["downbeats"].append(downbeats)
    dataset["sample_rate"].append(sr)
    dataset["genre"].append(genre)
    dataset["audio_name"].append(str(yt.title))
    dataset["url"].append(video_url)
    dataset["playlist"].append(playlist_url)
    dataset["time_accessed"].append(time.monotonic_ns())
    dataset["views"].append(yt.views)
    dataset["length"].append(yt.length)
    dataset["rating"].append(str(yt.rating))
    dataset["age_restricted"].append(yt.age_restricted)
    return True

# Calculates features for an entire playlist. Returns false if the calculation fails at any point
def calculate_playlist(playlist_url: str, batch_genre_name: str, use_low_resource_mode: bool = False):
    clear_output()

    # Get and confirm playlist url
    if not is_youtube_playlist(playlist_url):
        playlist_url = f"https://www.youtube.com/playlist?list={playlist_url}"

    try:
        title = Playlist(playlist_url).title
    except Exception as e:
        raise RuntimeError(f"It seems like what you entered is not a valid playlist: {e}")

    # Initialize dataset
    dataset = {key: [] for key in columns}

    # Copy dataset if necessary
    if os.path.exists(dataset_path):
        existing_ds = load_from_disk(dataset_path, keep_in_memory = True)
        for entry in tqdm(existing_ds, desc="Copying dataset"):
            for key in dataset:
                dataset[key].append(entry[key])
        del existing_ds
        gc.collect()

    # Get all video url datas
    t = time.time()
    urls = []
    processed_urls = set(dataset["url"])
    for url in tqdm(get_video_urls(playlist_url), desc="Getting URLs from playlist..."):
        if url not in processed_urls:
            urls.append(url)
            processed_urls.add(url)
        
    for i, url in enumerate(urls):
        clear_output()
        print(f"Current number of entries: {len(dataset['url'])} {i}/{len(urls)} for current playlist.")
        print(f"Playlist title: {title}, Genre: {batch_genre_name}, time elapsed: {time.time() - t:.2f}s")
        if use_low_resource_mode:
            print("Using low resource mode...")

        success = process_video_url(dataset, url, playlist_url, sr=22050, timeout=120, genre=batch_genre_name, use_low_resource_mode=use_low_resource_mode)
        if not success:
            continue

        hub_dataset = Dataset.from_dict(dataset)
        hub_dataset.save_to_disk(dataset_path)

# Deletes the first line of a file
def delete_first_line(path: str):
    with open(path, "r") as file:
        lines = file.readlines()
    with open(path, "w") as file:
        file.writelines(lines[1:])

# Puts the first line at the back of the file
def put_first_line_at_back(path: str):
    with open(path, "r") as file:
        lines = file.readlines()
    with open(path, "w") as file:
        file.writelines(lines[1:] + lines[0:1])

# Gets the playlist info from the playlist queue
def get_playlist_info(queue_path: str):
    # Read the queue file. The queue file consists of playlist url and genre name separated by a space
    try:
        with open(queue_path, "r") as file:
            first_row = file.readline().strip()
    except FileNotFoundError:
        print("No playlist queue file found, exiting...")
        return "stop now"
    
    if not first_row:
        return
    
    if first_row.startswith("### stop now"):
        return "stop now"
    
    if first_row.startswith("###"):
        put_first_line_at_back(queue_path)
        return
    
    elements = first_row.split(" ")
    if len(elements) != 2:
        put_first_line_at_back(queue_path)
        return
    
    playlist_url = elements[0]
    genre_name = elements[1]
    return playlist_url, genre_name,  first_row

# Indicates the playlist has been processed and appends a message to the queue
def append_to_queue(path: str, first_row: str, txt: str):
    with open(path, "r") as file:
            first_row_ = file.readline().strip()
    if first_row.strip() == first_row_.strip():
        delete_first_line(path)
    with open(path, "a") as file:
        file.write(f"{txt}\n")

# Main function
def main():
    queue_path = "playlist_queue.txt"
    use_low_resource_mode = input("Use low resource mode? (y/[n]): ").lower() == "y"
    while True:
        result = get_playlist_info(queue_path)
        if result == "stop now":
            break
        if result is None:
            time.sleep(0.5)
            continue
        playlist_url, genre_name, first_row = result
        try:
            calculate_playlist(playlist_url, genre_name, use_low_resource_mode=use_low_resource_mode)
            append_to_queue(queue_path, first_row, f"### Processed: {playlist_url} {genre_name}")
        except Exception as e:
            er = f"{e}".replace("\n", " ")
            append_to_queue(queue_path, first_row, f"### Failed: {playlist_url} ({genre_name}) due to {er}")

if __name__ == "__main__":
    main()
    