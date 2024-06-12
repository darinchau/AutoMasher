# python -m scripts.calculate from root directory to run the script

import os
from pytube import YouTube
from pytube import Playlist
from fyp import Audio
from fyp.audio.separation import DemucsAudioSeparator
from fyp.audio.analysis import analyse_beat_transformer, analyse_chord_transformer
from fyp.audio.dataset import DatasetEntry, SongGenre
from fyp.audio.dataset.compress import DatasetEntryEncoder
from fyp.audio.dataset.create import create_entry
from fyp.util import is_ipython, clear_cuda
from fyp.util.combine import get_video_id, get_url
import time
import traceback
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

def filter_song(yt: YouTube) -> bool:
    """Returns True if the song should be processed, False otherwise."""
    if yt.length >= 600 or yt.length < 120:
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

def write_error(error: str, exec: Exception, error_file: str = "./scripts/error.txt"):
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

_DEMUCS = None
def get_demucs():
    global _DEMUCS
    if not _DEMUCS:
        _DEMUCS = DemucsAudioSeparator()
    return _DEMUCS

def cleanup_temp_dir():
    clear_output()
    current_dir = tempfile.gettempdir()
    for filename in os.listdir(current_dir):
        if filename.endswith('.wav') or filename.endswith('.mp4'):
            file_path = os.path.join(current_dir, filename)
            try:
                os.remove(file_path)
                print(f"Deleted {filename}")
            except Exception as e:
                print(f"Failed to delete file: {file_path}")

def get_processed_urls(dataset_path: str) -> set[str]:
    processed_urls = set()
    for file in os.listdir(dataset_path):
        if file.endswith(".data"):
            processed_urls.add(file[:-5])
    return processed_urls

def process_audio(audio: Audio, video_url: str, playlist_url: str, genre: SongGenre) -> DatasetEntry | None:
    print(f"Audio length: {audio.duration} ({YouTube(video_url).length})")
    length = audio.duration

    print(f"Analysing chords...")
    chord_result = analyse_chord_transformer(audio, model_path="./resources/ckpts/btc_model_large_voca.pt", use_loaded_model=True)

    cr = chord_result.group()
    labels = cr.labels
    chord_times = cr.times
    if len(labels) != len(chord_times):
        print(f"Length mismatch: {video_url}")
        time.sleep(1)
        return None

    if not chord_times or chord_times[-1] > length:
        print(f"Chord times error: {video_url}")
        return None

    print("Separating audio...")
    parts = get_demucs().separate_audio(audio)

    print(f"Analysing beats...")
    beat_result = analyse_beat_transformer(parts=parts, model_path="./resources/ckpts/beat_transformer.pt", use_loaded_model=True)

    print("Postprocessing...")
    beats: list[float] = beat_result.beats.tolist()
    downbeats: list[float] = beat_result.downbeats.tolist()

    if not beats or beats[-1] > length:
        print(f"Beats error: {video_url}")
        time.sleep(1)
        return None

    if not downbeats or downbeats[-1] > length:
        print(f"Downbeats error: {video_url}")
        time.sleep(1)
        return None

    yt = YouTube(video_url)

    print("Creating entry...")
    return create_entry(
        length = audio.duration,
        beats = beats,
        downbeats = downbeats,
        chords = labels,
        chord_times = chord_times,
        genre = genre,
        audio_name = yt.title,
        url = video_url,
        playlist = playlist_url,
        views = yt.views
    )

def download_audio(urls: list[str]):
    def download_audio_single(url: str):
        if not filter_song(YouTube(url)):
            return None
        audio = Audio.load(url)
        return audio

    # Downloads the things concurrently and yields them one by one
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_audio_single, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                audio = future.result()
                yield audio, url
            except Exception as e:
                write_error(f"Failed to download audio (skipping): {url}", e)
                yield None, url

def save_dataset_entry(entry: DatasetEntry, dataset_path: str):
    encoder = DatasetEntryEncoder()
    b = encoder.encode(entry)
    with open(os.path.join(dataset_path, f"{get_video_id(entry.url)}.data"), "wb") as file:
        file.write(bytes(b))

def calculate_url_list(urls: list[str], genre: SongGenre, dataset_path: str, playlist_url: str, title: str):
    if len(urls) > 300:
        calculate_url_list(urls[:300], genre, dataset_path, playlist_url, title)
        calculate_url_list(urls[300:], genre, dataset_path, playlist_url, title)
        return

    t = time.time()
    last_t = None
    urls = [get_url(url) for url in urls]
    for i, (audio, url) in enumerate(download_audio(urls)):
        if not audio:
            continue

        clear_output()

        last_entry_process_time = round(time.time() - last_t, 2) if last_t else None
        last_t = time.time()
        print(f"Current number of entries: {len(os.listdir(dataset_path))} {i}/{len(urls)} for current playlist.")
        print(f"Playlist title: {title}")
        print(f"Last entry process time: {last_entry_process_time} seconds")
        print(f"Current entry: {url}")
        print(f"Time elapsed: {round(time.time() - t, 2)} seconds")
        print(f"Genre: {genre.value}")

        clear_cuda()

        try:
            entry = process_audio(audio, url, playlist_url, genre=genre)
            print(f"Entry processed: {url}")
        except Exception as e:
            write_error(f"Failed to process video: {url}", e)
            continue

        if not entry:
            continue

        save_dataset_entry(entry, dataset_path)
    cleanup_temp_dir()

# Calculates features for an entire playlist. Returns false if the calculation fails at any point
def calculate_playlist(playlist_url: str, batch_genre: SongGenre, dataset_path: str):
    clear_output()

    # Get and confirm playlist url
    if not is_youtube_playlist(playlist_url):
        playlist_url = f"https://www.youtube.com/playlist?list={playlist_url}"

    print(f"Processing playlist: {playlist_url}")

    try:
        title = Playlist(playlist_url).title
    except Exception as e:
        raise RuntimeError(f"It seems like what you entered is not a valid playlist: {e}")
    print("Playlist title: " + title)

    # Initialize dataset
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    processed_urls = set()
    for file in os.listdir(dataset_path):
        if file.endswith(".data"):
            processed_urls.add(file[:-5])

    # Debug only
    print("Processed URLs:")
    for url in processed_urls:
        print(url)

    # Get all video url datas
    urls: list[str] = []
    for url in tqdm(get_video_urls(playlist_url), desc="Getting URLs from playlist..."):
        url = get_video_id(url)
        if url not in processed_urls:
            urls.append(url)
            processed_urls.add(url)

    # Calculate features
    calculate_url_list(urls, batch_genre, dataset_path, playlist_url, title)

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

def clean_playlist_queue(queue_path: str):
    with open(queue_path, "r") as f:
        playlist = f.readlines()

    playlist = sorted(set([x.strip() for x in playlist]))

    with open(queue_path, "w") as f:
        f.write("\n".join(playlist))

# Main function
def main():
    queue_path = "./scripts/playlist_queue.txt"
    dataset_path = "./resources/dataset/audio-infos-v2"
    error_file = "./scripts/error.txt"

    # Sanity check
    if not os.path.exists(queue_path):
        print("No playlist queue found.")
        return

    clean_playlist_queue(queue_path)

    while True:
        next_playlist = get_next_playlist_to_process(queue_path)
        if not next_playlist:
            print("No more playlists to process.")
            break

        playlist_url, genre_name = next_playlist
        try:
            calculate_playlist(playlist_url, SongGenre(genre_name), dataset_path)
            update_playlist_process_queue(True, playlist_url, genre_name, queue_path)
        except Exception as e:
            write_error(f"Failed to process playlist: {playlist_url} {genre_name}", e, error_file)
            update_playlist_process_queue(False, playlist_url, genre_name, queue_path, error=e)

if __name__ == "__main__":
    main()
