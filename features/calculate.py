import os
from pytube import YouTube
from pytube import Playlist
from fyp import Audio
from fyp.audio.separation import HPSSAudioSeparator
from fyp.audio.analysis import analyse_beat_transformer, analyse_chord_transformer
from fyp.audio.analysis import ChordAnalysisResult, BeatAnalysisResult
from fyp.audio.dataset import DatasetEntry, SongDataset, load_song_dataset, save_song_dataset
from fyp.util import is_ipython, clear_cuda
import gc
from datasets import Dataset
import glob
import time
import traceback
from tqdm.auto import tqdm
from fyp.audio.search.align import get_normalized_chord_result, get_music_duration

# Return true if this song should be processed, false otherwise
# Duplicate handling should not be queried here
def filter_song(yt: YouTube) -> bool:
    if yt.length > 600 or yt.length < 120:
        return False
    
    if yt.age_restricted:
        return False
    
    if yt.views < 5e5:
        return False
    
    return True

def clear_output():
    try:
        if is_ipython():    
            from IPython.display import clear_output as ip_clear_output
            ip_clear_output()
        else:
            os.system("cls" if "nt" in os.name else "clear")
    except (ImportError, NameError) as e:
        os.system("cls" if "nt" in os.name else "clear")

def write_error(error: str, exec: Exception, error_file: str = "./features/error.txt"):
    with open(error_file, "a") as file:
        file.write(f"{error}: {exec}\n")
        file.write("".join(traceback.format_exception(exec)))
        file.write("\n\n")
        print("ERROR: " + error)

def get_video_urls(playlist_url: str):
    playlist = Playlist(playlist_url)
    return playlist.video_urls

def is_youtube_playlist(link: str):
    return "playlist?list=" in link

def log(message: str, verbose: bool = True):
    if verbose:
        print(message)

# Postprocessing filter dataset. If false means the data point is not valid
def filter_dataset(beats: list[float], downbeats: list[float], chord_times: list[float], length: float) -> bool:
    if not beats or beats[-1] > length:
        return False
    
    if not downbeats or downbeats[-1] > length:
        return False
    
    if not chord_times or chord_times[-1] > length:
        return False
    
    return True

# Create a dataset entry from the given data
def create_entry(length: float, beats: list[float], downbeats: list[float], chords: list[int], chord_times: list[float],
                    *, genre: str, audio_name: str, url: str, playlist: str, views: int):
    chord_result = ChordAnalysisResult(length, chords, chord_times)
    beat_result = BeatAnalysisResult(length, beats, downbeats)
    normalized_cr = get_normalized_chord_result(chord_result, beat_result)

    # For each bar, calculate its music duration
    music_duration: list[float] = []
    for i in range(len(downbeats)):
        bar_cr = normalized_cr.slice_seconds(i, i + 1)
        music_duration.append(get_music_duration(bar_cr))
    
    return DatasetEntry(
        chords=chords,
        chord_times=chord_times,
        downbeats=downbeats,
        beats=beats,
        genre=genre,
        audio_name=audio_name,
        url=url,
        playlist=playlist,
        views=views,
        length=length,
        normalized_chord_times=normalized_cr.times,
        music_duration=music_duration
    )

def process_video_url(video_url: str, playlist_url: str, genre="pop") -> DatasetEntry | None:
    # Make the YouTube object
    yt = YouTube(video_url)

    # If too long or too short then return
    if not filter_song(yt):
        print(f"Video filtered: {video_url}")
        return None
    
    print(f"Downloading audio... {yt.title}")
    audio = Audio.load(video_url)

    print(f"Analysing chords...")
    chord_result = analyse_chord_transformer(audio, model_path="./resources/ckpts/btc_model_large_voca.pt", use_loaded_model=True)

    print(f"Analysing beats...")
    beat_result = analyse_beat_transformer(audio, model_path="./resources/ckpts/beat_transformer.pt", use_loaded_model=True)

    print("Postprocessing...")
    labels = chord_result.grouped_labels
    times = chord_result.grouped_times
    beats: list[float] = beat_result.beats.tolist()
    downbeats: list[float] = beat_result.downbeats.tolist()

    if len(labels) != len(times):
        print(f"Length mismatch: {video_url}")
        return None

    if not filter_dataset(beats, downbeats, times, audio.duration):
        print(f"Dataset filtered: {video_url}")
        return None
    
    return create_entry(
        length = audio.duration,
        beats = beats,
        downbeats = downbeats,
        chords = labels,
        chord_times = times,
        genre = genre,
        audio_name = yt.title,
        url = video_url,
        playlist = playlist_url,
        views = yt.views
    )

# Calculates features for an entire playlist. Returns false if the calculation fails at any point
def calculate_playlist(playlist_url: str, batch_genre_name: str, dataset_path: str):
    clear_output()

    # Get and confirm playlist url
    if not is_youtube_playlist(playlist_url):
        playlist_url = f"https://www.youtube.com/playlist?list={playlist_url}"

    try:
        title = Playlist(playlist_url).title
    except Exception as e:
        raise RuntimeError(f"It seems like what you entered is not a valid playlist: {e}")

    # Initialize dataset
    if os.path.exists(dataset_path):
        dataset = load_song_dataset(dataset_path)
        gc.collect()
    else:
        dataset = SongDataset()

    # Get all video url datas
    t = time.time()
    last_t = None
    urls = []
    processed_urls = set(dataset._data.keys())
    for url in tqdm(get_video_urls(playlist_url), desc="Getting URLs from playlist..."):
        if url not in processed_urls:
            urls.append(url)
            processed_urls.add(url)
        
    for i, url in enumerate(urls):
        clear_output()
        
        last_entry_process_time = round(time.time() - last_t, 2) if last_t else None
        last_t = time.time()
        print(f"Current number of entries: {len(dataset)} {i}/{len(urls)} for current playlist.")
        print(f"Playlist title: {title}")
        print(f"Last entry process time: {last_entry_process_time} seconds")
        print(f"Current entry: {url}")
        print(f"Time elapsed: {round(time.time() - t, 2)} seconds")
        print(f"Genre: {batch_genre_name}")
        
        clear_cuda()

        entry = process_video_url(url, playlist_url, genre=batch_genre_name)
        if not entry:
            continue
        
        dataset[url] = entry
        save_song_dataset(dataset, dataset_path)

#### Driver code and functions ####
def get_next_playlist_to_process(queue_path: str) -> tuple[str, str] | None:
    try:
        with open(queue_path, "r") as file:
            lines = file.readlines()
            if not lines:
                return None
    except FileNotFoundError:
        return None
    
    for line in lines:
        if not line.startswith("###"):
            elements = line.strip().split(" ")
            playlist_id, genre = elements[0], elements[1]
            return playlist_id, genre
    return None

def update_playlist_process_queue(success: bool, playlist_url: str, genre_name: str, queue_path: str, error: Exception | None = None):
    with open(queue_path, "r") as file:
        lines = file.readlines()
    
    with open(queue_path, "w") as file:
        for line in lines:
            if line.startswith("###"):
                file.write(line)
            else:
                if line.strip() == f"{playlist_url} {genre_name}":
                    if success:
                        file.write(f"### Processed: {playlist_url} {genre_name}\n")
                    else:
                        file.write(f"### Failed: {playlist_url} {genre_name} ({error})\n")
                else:
                    file.write(line)

# Main function
def main():
    queue_path = "./features/playlist_queue.txt"
    dataset_path = "./resources/dataset/audio-infos-v2"
    error_file = "./features/error.txt"

    # Sanity check
    if not os.path.exists(queue_path):
        print("No playlist queue found.")
        return

    while True:
        next_playlist = get_next_playlist_to_process(queue_path)
        if not next_playlist:
            print("No more playlists to process.")
            break
        
        playlist_url, genre_name = next_playlist
        try:
            calculate_playlist(playlist_url, genre_name, dataset_path)
            update_playlist_process_queue(True, playlist_url, genre_name, queue_path)
        except Exception as e:
            write_error(f"Failed to process playlist: {playlist_url} {genre_name}", e, error_file)
            update_playlist_process_queue(False, playlist_url, genre_name, queue_path, error=e)

if __name__ == "__main__":
    main()
    