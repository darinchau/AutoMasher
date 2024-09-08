import os
import gc
import tempfile
from threading import Thread, Lock
from fyp import Audio, SongDataset, DatasetEntry
from tqdm.auto import tqdm
import random

def cleanup_temp_dir():
    current_dir = tempfile.gettempdir()
    for filename in os.listdir(current_dir):
        if filename.endswith('.wav') or filename.endswith('.mp4') or filename.endswith('.mp3'):
            file_path = os.path.join(current_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                pass
    gc.collect()

print_lock = Lock()

def worker_thread(dataset: SongDataset):
    cache_path = "./resources/cache"
    while True:
        entry = random.choice(dataset)
        with print_lock:
            print(f"Downloading audio: {entry.url.video_id}... ({len(os.listdir(cache_path))})")
        try:
            audio = entry.get_audio()
            cleanup_temp_dir()
        except Exception as e:
            with print_lock:
                print(f"Failed to download audio: {entry.url.video_id}: {e}")

def download_audio(dataset: SongDataset):
    workers = [Thread(target=worker_thread, args=(dataset,)) for _ in range(8)]
    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

def main():
    ds = SongDataset.load("resources/dataset/audio-infos-v2.1.db")
    download_audio(ds)

if __name__ == "__main__":
    main()
