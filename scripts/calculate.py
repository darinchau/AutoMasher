# python -m scripts.calculate from root directory to run the script
# This creates a dataset of audio features from a playlist of songs

import os
from fyp import Audio
from fyp.audio.dataset import DatasetEntry, SongGenre
from fyp.audio.dataset.v3 import DatasetEntryEncoder
from fyp.audio.dataset.create import process_audio_
from fyp.util import is_ipython, clear_cuda
from fyp.util import get_video_id, get_url, YouTubeURL
import time
import traceback
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

try:
    from pytube import Playlist, YouTube, Channel
except ImportError:
    try:
        from pytubefix import Playlist, YouTube, Channel
    except ImportError:
        raise ImportError("Please install the pytube library to download the audio. You can install it using `pip install pytube` or `pip install pytubefix`")

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

def is_youtube_playlist(link: str):
    return "playlist?list=" in link

def log(message: str, verbose: bool = True):
    if verbose:
        print(message)

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
        if file.endswith(".dat3"):
            processed_urls.add(file[:-5])
    return processed_urls

def process_audio(audio: Audio, video_url: YouTubeURL, genre: SongGenre):
    processed = process_audio_(audio, video_url, genre, verbose=True)
    if isinstance(processed, str):
        print(processed)
        with open("scripts/rejected.txt", "a") as file:
            file.write(f"{video_url} {processed}\n")
        time.sleep(1)
        return None
    return processed

def download_audio(urls: list[YouTubeURL]):
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
    with open(os.path.join(dataset_path, f"{entry.url.video_id}.dat3"), "wb") as file:
        file.write(bytes(b))

def calculate_url_list(urls: list[YouTubeURL], genre: SongGenre, dataset_path: str, title: str):
    if len(urls) > 300:
        calculate_url_list(urls[:300], genre, dataset_path, title)
        calculate_url_list(urls[300:], genre, dataset_path, title)
        return

    t = time.time()
    last_t = None
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
            entry = process_audio(audio, url, genre=genre)
            print(f"Entry processed: {url}")
        except Exception as e:
            write_error(f"Failed to process video: {url}", e)
            continue

        if not entry:
            continue

        save_dataset_entry(entry, dataset_path)
    cleanup_temp_dir()

def get_playlist_title_and_video(key: str) -> tuple[str, list[YouTube]]:
    # Messy code dont look D:
    e1 = e2 = None
    try:
        if not "playlist?list=" in key:
            pl = Playlist(f"https://www.youtube.com/playlist?list={key}")
        else:
            pl = Playlist(key)
        urls = list(pl.video_urls) # Un-defer the generator to make sure any errors are raised here
        if urls:
            try:
                return pl.title, urls
            except Exception as e:
                return f"Playlist {key}", urls
    except Exception as e:
        e1 = e
        pass

    try:
        if not "channel/" in key:
            ch = Channel(f"https://www.youtube.com/channel/{key}")
        else:
            ch = Channel(key)
        urls = list(ch.video_urls)
        if urls:
            try:
                return ch.title, urls
            except Exception as e:
                return f"Channel {key}", urls
    except Exception as e:
        e2 = e
        pass

    # Format error message
    raise ValueError(f"Invalid channel or playlist: {key} (Playlist error: {e1}, Channel error: {e2})")

# Calculates features for an entire playlist. Returns false if the calculation fails at any point
def calculate_playlist(playlist_url: str, batch_genre: SongGenre, dataset_path: str, queue_path: str):
    clear_output()

    # Get and confirm playlist url
    title, video_ids = get_playlist_title_and_video(playlist_url)
    print("Playlist title: " + title)

    # Initialize dataset
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    processed_video_ids = set()
    for file in os.listdir(dataset_path):
        if file.endswith(".dat3"):
            processed_video_ids.add(file[:-6])

    # Debug only
    print(f"Number of processed URLs: {len(processed_video_ids)}")

    # Get all video url datas
    urls: list[YouTubeURL] = []
    for yt in tqdm(video_ids, desc="Getting URLs from playlist..."):
        video_url = get_url(yt.watch_url) if isinstance(yt, YouTube) else get_url(yt)

        if video_url.video_id not in processed_video_ids:
            urls.append(video_url)
            processed_video_ids.add(video_url.video_id)

        # Be aggressive with the number of songs and add all the channels' songs into it
        # Trying to assume that if a channel has a song in the playlist, all of its uploads will be songs
        try:
            channel_id = yt.channel_id
            if channel_id is None or not channel_id or channel_id.lower().strip() == "none":
                continue
            # The cleanup function will automatically remove duplicates so we don't need to worry about that
            # inefficient but convenient
            add_playlist_to_queue(channel_id, batch_genre.value, queue_path)
        except Exception as e:
            pass

    # Calculate features
    calculate_url_list(urls, batch_genre, dataset_path, title)

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
            elif line.strip() == f"{playlist_url} {genre_name}":
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
    playlist = [x for x in playlist if len(x.strip()) > 0]

    with open(queue_path, "w") as f:
        f.write("\n".join(playlist))
        f.write("\n")

def add_playlist_to_queue(playlist_url: str, genre_name: str, queue_path: str):
    with open(queue_path, "a") as file:
        file.write(f"\n{playlist_url} {genre_name}\n")
    clean_playlist_queue(queue_path)

# Main function
def main():
    queue_path = "./scripts/playlist_queue.txt"
    dataset_path = "./resources/dataset/audio-infos-v3"
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
            calculate_playlist(playlist_url, SongGenre(genre_name), dataset_path, queue_path)
            update_playlist_process_queue(True, playlist_url, genre_name, queue_path)
        except Exception as e:
            write_error(f"Failed to process playlist: {playlist_url} {genre_name}", e, error_file)
            update_playlist_process_queue(False, playlist_url, genre_name, queue_path, error=e)

if __name__ == "__main__":
    main()
