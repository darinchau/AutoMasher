# Provides a function to load our dataset as a list of dataset entries
# This additional data structure facilitates getting by url
from __future__ import annotations
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from typing import Any, Callable, Optional
from copy import deepcopy
from ...util.combine import get_video_id
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

def _is_sorted(ls: list[float]):
    return all(a <= b for a, b in zip(ls, ls[1:]))

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

    @staticmethod
    def get_playlist_prepend():
        return "https://www.youtube.com/playlist?list="

    def __post_init__(self):
        assert len(self.chords) == len(self.chord_times) == len(self.normalized_chord_times)
        assert all(0 <= c <= 600 for c in self.chord_times)
        assert all(0 <= c <= 170 for c in self.chords)
        assert all(0 <= c <= 600 for c in self.downbeats)
        assert all(0 <= c <= 600 for c in self.beats)
        assert self.playlist.startswith(self.get_playlist_prepend())
        assert self.views >= 0
        assert self.length > 0
        assert _is_sorted(self.chord_times)
        assert _is_sorted(self.downbeats)
        assert _is_sorted(self.beats)

    @property
    def url_id(self):
        # return self.url[-11:]
        return get_video_id(self.url)

    def __repr__(self):
        return f"DatasetEntry({self.audio_name} [{self.url_id}])"

    def equal(self, value: DatasetEntry, *, eps: float = 1e-5) -> bool:
        """Check if the given value is equal to this entry. This is useful for testing purposes.
        This is a bit more lenient than __eq__ as it allows for some floating point error."""
        if self.url != value.url:
            return False
        if self.audio_name != value.audio_name:
            return False
        if self.genre != value.genre:
            return False
        if self.playlist != value.playlist:
            return False
        if self.views != value.views:
            return False
        if abs(self.length - value.length) > eps:
            return False
        if self.chords != value.chords:
            return False
        if len(self.chord_times) != len(value.chord_times):
            return False
        if any(abs(a - b) > eps for a, b in zip(self.chord_times, value.chord_times)):
            return False
        if len(self.downbeats) != len(value.downbeats):
            return False
        if any(abs(a - b) > eps for a, b in zip(self.downbeats, value.downbeats)):
            return False
        if len(self.beats) != len(value.beats):
            return False
        if any(abs(a - b) > eps for a, b in zip(self.beats, value.beats)):
            return False
        return True

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

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data.values())

    def add_entry(self, entry: DatasetEntry):
        self._data[entry.url] = entry

    def filter(self, filter_func: Callable[[DatasetEntry], bool] | None):
        """Returns a new dataset with the entries that satisfy the filter function. If filter_func is None, return yourself"""
        if filter_func is None:
            return self
        new_dataset = SongDataset()
        for entry in self:
            if filter_func(entry):
                new_dataset.add_entry(entry)
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
        entry = DatasetEntry(
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
        song_dataset.add_entry(entry)

    return song_dataset

def save_song_dataset(dataset: SongDataset, dataset_path: str):
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
