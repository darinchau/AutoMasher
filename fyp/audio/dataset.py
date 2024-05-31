# Provides a function to load our dataset as a list of dataset entries
# This additional data structure facilitates getting by url
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from typing import Any, Callable, Optional
from copy import deepcopy
from ..util.combine import get_url

@dataclass(frozen=True)
class DatasetEntry:
    chords: list[int]
    chord_times: list[float]
    downbeats: list[float]
    beats: list[float]
    genre: str
    audio_name: str
    url: str
    playlist: str
    views: int
    length: float
    normalized_chord_times: list[float]
    music_duration: list[float]

    def __post_init__(self):
        assert len(self.chords) == len(self.chord_times) == len(self.normalized_chord_times)

    @property
    def url_id(self):
        return self.url[-11:]

class SongDataset:
    """Use a hashmap for now. lmk if there are more efficient ways to do this."""
    def __init__(self):
        self._data: dict[str, DatasetEntry] = {}

    def get_by_url(self, url: str) -> DatasetEntry | None:
        return self._data.get(url, None)
    
    def __getitem__(self, url: str) -> DatasetEntry:
        return self._data[url]
    
    def __setitem__(self, url: str, entry: DatasetEntry):
        self._data[url] = entry

    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data.values())
    
    def filter(self, filter_func: Callable[[DatasetEntry], bool] | None):
        """Returns a new dataset with the entries that satisfy the filter function. If filter_func is None, return yourself"""
        if filter_func is None:
            return self
        new_dataset = SongDataset()
        for entry in self:
            if filter_func(entry):
                new_dataset[entry.url] = entry
        return new_dataset

def load_song_dataset(dataset_path: str) -> SongDataset:
    try:
        dataset = load_dataset(dataset_path, split="train")
    except ValueError as e:
        expected_message = "You are trying to load a dataset that was saved using `save_to_disk`. Please use `load_from_disk` instead."
        if e.args[0] == expected_message:
            dataset = Dataset.load_from_disk(dataset_path)
        else:
            raise e
    song_dataset = SongDataset()
    for entry in dataset:
        song_dataset[entry["url"]] = DatasetEntry(
            chords=entry["chords"],
            chord_times=entry["chord_times"],
            downbeats=entry["downbeats"],
            beats=entry["beats"],
            genre=entry["genre"],
            audio_name=entry["audio_name"],
            url=entry["url"],
            playlist=entry["playlist"],
            views=entry["views"],
            length=entry["length"],
            normalized_chord_times=entry["normalized_chord_times"],
            music_duration=entry["music_duration"]
        )

    return song_dataset

def save_song_dataset(dataset: SongDataset, dataset_path: str):
    ds = Dataset.from_dict({
        "chords": [entry.chords for entry in dataset],
        "chord_times": [entry.chord_times for entry in dataset],
        "downbeats": [entry.downbeats for entry in dataset],
        "beats": [entry.beats for entry in dataset],
        "genre": [entry.genre for entry in dataset],
        "audio_name": [entry.audio_name for entry in dataset],
        "url": [entry.url for entry in dataset],
        "playlist": [entry.playlist for entry in dataset],
        "views": [entry.views for entry in dataset],
        "length": [entry.length for entry in dataset],
        "normalized_chord_times": [entry.normalized_chord_times for entry in dataset],
        "music_duration": [entry.music_duration for entry in dataset]
    })

    ds.save_to_disk(dataset_path)
