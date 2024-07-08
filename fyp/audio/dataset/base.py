# Provides a function to load our dataset as a list of dataset entries
# This additional data structure facilitates getting by url
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional
from copy import deepcopy
from ... import Audio
from ...util import get_video_id, get_url
from enum import Enum
import numpy as np
from typing import Iterable

class SongGenre(Enum):
    POP = "pop"
    CANTOPOP = "cantopop"
    VOCALOID = "vocaloid"
    KPOP = "kpop"
    NIGHTCORE = "nightcore"
    ANIME = "anime"
    BROADWAY = "broadway"
    JPOP = "jp-pop"
    UNKNOWN = "unknown"

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
            SongGenre.JPOP: 7,
            SongGenre.UNKNOWN: 8
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
    """A single entry in the dataset. This is a dataclass that represents a single entry in the dataset.
    Typically we use the create_entry function to create a DatasetEntry object.

    chords: list[int] - The chord progression of the song
    chord_times: list[float] - The time in seconds where the chord changes
    downbeats: list[float] - The time in seconds where the downbeats are
    beats: list[float] - The time in seconds where the beats are
    genre: SongGenre - The genre of the song
    audio_name: str - The name of the audio file
    url: str - The url of the youtube video
    playlist: str - The url of the youtube playlist this song is taken from
    views: int - The number of views of the video at the time of scraping
    length: float - The length of the song in seconds
    normalized_chord_times: list[float] - The normalized chord times
    music_duration: list[float] - The percentage of the music at each bar"""
    chords: list[int]
    chord_times: list[float]
    downbeats: list[float]
    beats: list[float]
    genre: SongGenre
    audio_name: str
    url: str
    playlist: str | None
    views: int
    length: float
    normalized_chord_times: list[float]
    music_duration: list[float]

    @staticmethod
    def get_playlist_prepend():
        return "https://www.youtube.com/playlist?list="

    @staticmethod
    def get_url_prepend():
        return "https://www.youtube.com/watch?v="

    def __post_init__(self):
        assert len(self.chords) == len(self.chord_times) == len(self.normalized_chord_times), f"{len(self.chords)} != {len(self.chord_times)} != {len(self.normalized_chord_times)}"
        assert len(self.downbeats) == len(self.music_duration), f"{len(self.downbeats)} != {len(self.music_duration)}"
        assert all(0 <= c < 600 for c in self.chord_times), f"Invalid chord times: {self.chord_times}"
        assert all(0 <= c < 170 for c in self.chords), f"Invalid chords: {self.chords}"
        assert all(0 <= c < 600 for c in self.downbeats), f"Invalid downbeats: {self.downbeats}"
        assert all(0 <= c < 600 for c in self.beats), f"Invalid beats: {self.beats}"
        assert self.playlist is None or self.playlist.startswith(self.get_playlist_prepend())
        assert self.url.startswith(self.get_url_prepend())
        assert len(self.url) > 11
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

    def get_valid_starting_points(self, nbars: int, min_music_percentage: float) -> Iterable[int]:
        """Returns a list of valid starting points based on the min_music_percentage and nbars.
        Fundamentally this is the following code, just optimized:

        [i for i in range(len(self.downbeats) - nbars) if sum(self.music_duration[i:i + nbars]) >= min_music_percentage * nbars]"""
        cumulative_music_durations = np.cumsum(self.music_duration)
        cumulative_music_durations[nbars:] = cumulative_music_durations[nbars:] - cumulative_music_durations[:-nbars]
        cumulative_music_durations = cumulative_music_durations[nbars-1:-1]
        min_music_duration = min_music_percentage * nbars

        # Find the indices where the sum of the music duration is greater than the minimum music duration
        valid_indices = np.where(cumulative_music_durations >= min_music_duration)[0]
        return valid_indices

    def get_audio(self):
        return Audio.load(self.url, cache_path=f"resources/cache/{self.url_id}.mp3")

    @staticmethod
    def from_url(url: str, playlist: str | None = None, genre: SongGenre = SongGenre.UNKNOWN):
        from .create import process_audio_
        url = get_url(url)
        audio = Audio.load(url)
        entry = process_audio_(audio, url, playlist, genre, verbose=False)
        if isinstance(entry, str):
            raise ValueError(f"Failed to process audio: {entry}")
        return entry

class SongDataset:
    """A data structure that holds a bunch of dataset entries for query. Use a hashmap for now. lmk if there are more efficient ways to do this."""
    def __init__(self):
        self._data: dict[str, DatasetEntry] = {}

    def get_by_url(self, url: str) -> DatasetEntry | None:
        return self._data.get(url, None)

    def __getitem__(self, url: str | int) -> DatasetEntry:
        if isinstance(url, str):
            url = get_url(url)
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

    @property
    def urls(self):
        return sorted(self._data.keys())

    @staticmethod
    def load(dataset_path: str):
        """Load the dataset from hugging face. The dataset can be in one of the following formats:
        - A local directory with a hugging face dataset
        - A hugging face remote path
        - A local directory containing a bunch of compressed DatasetEntry files
        - A file saved using the SongDatasetEncoder"""

        if os.path.isfile(dataset_path):
            from .compress import SongDatasetEncoder, FastSongDatasetEncoder
            try:
                return SongDatasetEncoder().read_from_path(dataset_path)
            except Exception as e:
                try:
                    return FastSongDatasetEncoder().read_from_path(dataset_path)
                except Exception as e:
                    pass

        try:
            from .v1 import load_dataset_v1
            return load_dataset_v1(dataset_path)
        except Exception as e:
            pass

        assert os.path.exists(dataset_path) and os.path.isdir(dataset_path), f"Invalid dataset path: {dataset_path}"

        # If .data file exists, we assume that the dataset is a directory containing compressed DatasetEntry files
        data_files = [f for f in os.listdir(dataset_path) if f.endswith(".data")]
        if data_files:
            from .compress import DatasetEntryEncoder
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

    def __repr__(self):
        return f"SongDataset({len(self)} entries)"

    def keys(self) -> list[str]:
        return sorted(self._data.keys())
