import os
from fyp.audio.dataset.legacy.v2 import DatasetEntryEncoder as DatasetEntryEncoder_v2
from fyp.audio.dataset.v3 import DatasetEntryEncoder as DatasetEntryEncoder
from fyp.audio.dataset.base import SongDataset, DatasetEntry

def main():
    path = "./resources/dataset/audio-infos-v3"
    for entry in os.listdir(path):
        if not entry.endswith(".data"):
            continue
        entry_path = os.path.join(path, entry)
        try:
            dataset_entry = DatasetEntryEncoder_v2().read_from_path(entry_path)
        except Exception as e:
            print(f"Error reading {entry_path}: {e}")
            os.remove(entry_path)
            continue

        new_path = entry_path.replace(".data", ".dat3")
        DatasetEntryEncoder().write_to_path(dataset_entry, new_path)
        os.remove(entry_path)

    print("Migration complete :D")

if __name__ == "__main__":
    main()
