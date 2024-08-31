# Contains all the code to load and save to the v1 dataset
from .base import SongDataset, DatasetEntry, SongGenre
from datasets import load_dataset, Dataset

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
            audio_name=entry["audio_name"],
            url=entry["url"],
            playlist=entry["playlist"],
            views=entry["views"],
            length=entry["length"],
            normalized_chord_times=entry["normalized_chord_times"],
            music_duration=entry["music_duration"]
        )
        song_dataset.add_entry(entry)

    return song_dataset

def save_dataset_v1(dataset: SongDataset, dataset_path: str):
    ds = Dataset.from_dict({
        "chords": [entry.chords for entry in dataset],
        "chord_times": [entry.chord_times for entry in dataset],
        "downbeats": [entry.downbeats for entry in dataset],
        "beats": [entry.beats for entry in dataset],
        "genre": [entry.genre.value for entry in dataset],
        "audio_name": [entry.audio_name for entry in dataset],
        "url": [entry.url for entry in dataset],
        "playlist": [entry.playlist for entry in dataset],
        "views": [entry.views for entry in dataset],
        "length": [entry.length for entry in dataset],
        "normalized_chord_times": [entry.normalized_chord_times for entry in dataset],
        "music_duration": [entry.music_duration for entry in dataset]
    })

    ds.save_to_disk(dataset_path)
