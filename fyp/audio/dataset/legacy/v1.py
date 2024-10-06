# Contains all the code to load and save to the v1 dataset
from ..base import SongDataset, DatasetEntry, SongGenre

def _get_genre_map() -> dict[str, SongGenre]:
    """Gets a genre map thats used for loading the dataset for loading the legacy v1 dataset"""
    genre_map = {genre.value: genre for genre in SongGenre}
    genre_map["jp-anime"] = SongGenre.ANIME
    genre_map["country"] = SongGenre.POP
    genre_map["hip-hop"] = SongGenre.POP
    genre_map["dance-pop"] = SongGenre.POP
    genre_map["rock"] = SongGenre.POP
    genre_map["folk"] = SongGenre.POP
    genre_map["reggaeton"] = SongGenre.POP
    return genre_map

def load_dataset_v1(dataset_path: str) -> SongDataset:
    """Load the song dataset v1 from hugging face. The dataset path can be either a local path or a remote path."""
    try:
        from datasets import load_dataset, Dataset
    except ImportError:
        raise ImportError("Please install the datasets library to save the dataset. You can install it using `pip install datasets==2.18.0` and `pip install huggingface-hub==0.22.2`")
    try:
        dataset = load_dataset(dataset_path, split="train")
    except ValueError as e:
        expected_message = "You are trying to load a dataset that was saved using `save_to_disk`. Please use `load_from_disk` instead."
        if e.args[0] == expected_message:
            dataset = Dataset.load_from_disk(dataset_path)
        else:
            raise e
    genre_map = _get_genre_map()
    song_dataset = SongDataset()
    for entry in dataset:
        entry = DatasetEntry(
            chords=entry["chords"],
            chord_times=entry["chord_times"],
            downbeats=entry["downbeats"],
            beats=entry["beats"],
            genre=genre_map[entry["genre"].strip()],
            url=entry["url"],
            views=entry["views"],
            length=entry["length"],
            normalized_chord_times=entry["normalized_chord_times"],
            music_duration=entry["music_duration"]
        )
        song_dataset.add_entry(entry)

    return song_dataset

def save_dataset_v1(dataset: SongDataset, dataset_path: str):
    raise NotImplementedError("Saving to v1 dataset is not supported. Please use v3 dataset instead.")
