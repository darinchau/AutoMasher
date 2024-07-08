from concurrent.futures import ThreadPoolExecutor, as_completed
from fyp import Audio, SongDataset, DatasetEntry
from tqdm.auto import tqdm

def download_audio(dataset: SongDataset):
    def download_audio_single(entry: DatasetEntry):
        return entry.get_audio()

    # Downloads the things concurrently and yields them one by one
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_audio_single, entry): entry for entry in dataset}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                audio = future.result()
                yield audio
            except Exception as e:
                yield f"Failed to download audio (skipping): {entry.url_id}: {e}"

def main():
    ds = SongDataset.load("resources/dataset/audio-infos-v2.1.db")
    progress_bar = tqdm(desc="Downloading audio...", total=len(ds))
    for audio in download_audio(ds):
        if isinstance(audio, str):
            progress_bar.write(audio)
            continue

        progress_bar.update(1)

if __name__ == "__main__":
    main()
