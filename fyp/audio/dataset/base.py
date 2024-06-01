# Provides a function to load our dataset as a list of dataset entries
# This additional data structure facilitates getting by url
from __future__ import annotations
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from typing import Any, Callable, Optional
from copy import deepcopy
from ...util.combine import get_url
from enum import Enum

class SongGenre(Enum):
    POP = "pop"
    CANTOPOP = "cantopop"
    VOCALOID = "vocaloid"
    KPOP = "kpop"
    NIGHTCORE = "nightcore"
    ANIME = "anime"
    BROADWAY = "broadway"
    JPOP = "jp-pop"

    @property
    def mapping(self):
        return {
            SongGenre.POP: 0,
            SongGenre.CANTOPOP: 1,
            SongGenre.VOCALOID: 2,
            SongGenre.KPOP: 3,
            SongGenre.NIGHTCORE: 4,
            SongGenre.ANIME: 5,
            SongGenre.BROADWAY: 6,
            SongGenre.JPOP: 7
        }

    def to_int(self) -> int:
        return self.mapping[self]
    
    @staticmethod
    def from_int(i: int) -> SongGenre:
        return list(SongGenre)[i]

@dataclass(frozen=True)
class DatasetEntry:
    chords: list[int]
    chord_times: list[float]
    downbeats: list[float]
    beats: list[float]
    genre: SongGenre
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
    
    def __getitem__(self, url: str | int) -> DatasetEntry:
        if isinstance(url, str):
            return self._data[url]
        elif isinstance(url, int):
            return list(self._data.values())[url]
        else:
            raise TypeError(f"Invalid type for url: {type(url)}")
    
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
    
    def __repr__(self):
        return f"SongDataset({len(self)} entries)"

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
            genre=SongGenre(entry["genre"].strip()),
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
