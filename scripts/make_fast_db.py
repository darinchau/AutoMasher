# python -m scripts.make_fast_db from the root directory
# Packs the audio-infos-v3 compressed dataset file into a pickle which supports faster loading
from fyp.audio.dataset.v3 import DatasetEntryEncoder, SongDatasetEncoder, FastSongDatasetEncoder

def main():
    path = "./resources/dataset/audio-infos-v3.0.db"
    ds = SongDatasetEncoder().read_from_path(path)
    FastSongDatasetEncoder().write_to_path(ds, path.replace(".db", ".fast.db"))

if __name__ == "__main__":
    main()
