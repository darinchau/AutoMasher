# python -m scripts.pack_dataset from the root directory
# Packs the audio-infos-v2 dataset into a single, compressed dataset file

import os
from tqdm.auto import tqdm
from fyp.audio.dataset import DatasetEntry, SongDataset, SongGenre
from fyp.audio.dataset.compress import DatasetEntryEncoder, SongDatasetEncoder

def main():
    path_in = "./resources/dataset/audio-infos-v2"
    path_out = "./resources/dataset/audio-infos-v2.db"

    audio_datas = os.listdir(path_in)
    dataset = SongDataset()

    entry_encoder = DatasetEntryEncoder()
    dataset_encoder = SongDatasetEncoder()

    for audio_data in tqdm(audio_datas):
        audio_data_path = os.path.join(path_in, audio_data)
        try:
            dataset_entry = entry_encoder.read_from_path(audio_data_path)
        except Exception as e:
            print(f"Error reading {audio_data_path}: {e}")
            continue
        dataset.add_entry(dataset_entry)

    dataset_encoder.write_to_path(dataset, path_out)
    print(f"Dataset packed to {path_out} ({len(dataset)} entries)")

    # Verify dataset
    read_dataset = dataset_encoder.read_from_path(path_out)
    print(f"Read dataset from {path_out} ({len(read_dataset)} entries)")
    for url, entry in tqdm(dataset._data.items(), total=len(dataset)):
        read_entry = read_dataset.get_by_url(url)
        if read_entry is None:
            raise ValueError(f"Entry {entry} not found in read dataset")
        if entry != read_entry:
            print(f"Entry {entry} mismatch")
            raise ValueError(f"Entry {entry} mismatch")

    print("Dataset verified :D")

if __name__ == "__main__":
    main()
