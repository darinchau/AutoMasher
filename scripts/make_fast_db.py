from fyp import SongDataset, DatasetEntry

# This reencodes the dataset to a more efficient format

# cProfile.run("SongDataset.load('./resources/dataset/audio-infos-v2.db')", sort="cumulative")

from fyp.audio.dataset.compress import FastSongDatasetEncoder

def main():
    ds = SongDataset.load("./resources/dataset/audio-infos-v2.db")
    FastSongDatasetEncoder().write_to_path(ds, "./resources/dataset/audio-infos-v2.1.db")

if __name__ == "__main__":
    main()
