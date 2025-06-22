from fyp.audio.dataset import SongDataset
from fyp.util import YouTubeURL, get_url
import shutil
import os
import json
from tqdm import tqdm


def migrate_audios(ds: SongDataset, old_dataset_path: str):
    audio_infos = os.path.join(old_dataset_path, "audio", "info.json")
    if not os.path.isfile(audio_infos):
        raise FileNotFoundError(f"Audio info file not found at {audio_infos}")
    with open(audio_infos, "r", encoding='utf-8') as f:
        audio_info = json.load(f)

    print("Migrating audio files...")
    for url, info in tqdm(audio_info.items(), desc="Audio files"):
        if not isinstance(url, str):
            raise ValueError(f"Invalid URL in audio info: {url}")
        url = get_url(url)
        old_path = os.path.join(old_dataset_path, "audio", info)
        if not os.path.isfile(old_path):
            print(f"Audio file not found at {old_path}, skipping...")
            continue
        new_path = ds.get_path("audio", url, os.path.basename(old_path))
        if os.path.isfile(new_path):
            print(f"Audio file already exists at {new_path}, skipping...")
            continue
        shutil.copy2(old_path, new_path)

    data_paths = os.listdir(os.path.join(old_dataset_path, "datafiles"))
    print("Migrating data files...")
    for data_path in tqdm(data_paths, desc="Data files"):
        old_data_path = os.path.join(old_dataset_path, "datafiles", data_path)
        if not os.path.isfile(old_data_path):
            print(f"Data file not found at {old_data_path}, skipping...")
            continue
        new_data_path = ds.get_path("datafiles", get_url(data_path), data_path)
        if os.path.isfile(new_data_path):
            print(f"Data file already exists at {new_data_path}, skipping...")
            continue
        shutil.copy2(old_data_path, new_data_path)

    print("Writing info.json...")
    old_info_path = os.path.join(old_dataset_path, "info.json")
    new_info_path = ds.get_path("info")
    with open(old_info_path, "r", encoding='utf-8') as f:
        old_info = json.load(f)
    with open(new_info_path, "w", encoding='utf-8') as f:
        json.dump(old_info, f, ensure_ascii=False, indent=4)


def main():
    old_path = "E:/audio-dataset-v3"
    new_path = "E:/audio-dataset-v4"
    os.makedirs(new_path, exist_ok=True)
    ds = SongDataset(new_path)
    migrate_audios(ds, old_path)


if __name__ == "__main__":
    main()
