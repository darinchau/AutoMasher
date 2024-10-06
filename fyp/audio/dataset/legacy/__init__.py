import os
from .. import DatasetEntry, SongDataset
from .v2 import SongDatasetEncoder, FastSongDatasetEncoder
from .v2 import DatasetEntryEncoder
from .v1 import load_dataset_v1

def load_dataset_legacy(dataset_path: str) -> SongDataset:
        if os.path.isfile(dataset_path):
            try:
                return SongDatasetEncoder().read_from_path(dataset_path)
            except Exception as e:
                try:
                    return FastSongDatasetEncoder().read_from_path(dataset_path)
                except Exception as e:
                    pass

        try:
            return load_dataset_v1(dataset_path)
        except Exception as e:
            pass

        assert os.path.exists(dataset_path) and os.path.isdir(dataset_path), f"Invalid dataset path: {dataset_path}"

        # If .data file exists, we assume that the dataset is a directory containing compressed DatasetEntry files
        data_files = [f for f in os.listdir(dataset_path) if f.endswith(".data")]
        if data_files:
            dataset = SongDataset()
            encoder = DatasetEntryEncoder()
            for data_file in data_files:
                try:
                    entry = encoder.read_from_path(os.path.join(dataset_path, data_file))
                except Exception as e:
                    print(f"Error reading {data_file}: {e}")
                    continue
                dataset.add_entry(entry)
            return dataset

        raise ValueError(f"Invalid dataset path: {dataset_path}")
