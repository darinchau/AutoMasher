# Provides a function to load our dataset as a list of dataset entries
# This additional data structure facilitates getting by url
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional
from copy import deepcopy
from ..base import Audio
from ...util import get_url
from ...util import YouTubeURL
from enum import Enum
import os
import json
import traceback
from typing import Callable

from tqdm.auto import tqdm
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
    url: YouTubeURL
    views: int
    length: float
    normalized_chord_times: list[float]
    music_duration: list[float]

    def __post_init__(self):
        assert len(self.chords) == len(self.chord_times) == len(self.normalized_chord_times), f"{len(self.chords)} != {len(self.chord_times)} != {len(self.normalized_chord_times)}"
        assert len(self.downbeats) == len(self.music_duration), f"{len(self.downbeats)} != {len(self.music_duration)}"
        assert all(0 <= c < 600 for c in self.chord_times), f"Invalid chord times: {self.chord_times}"
        assert all(0 <= c < 170 for c in self.chords), f"Invalid chords: {self.chords}"
        assert all(0 <= c < 600 for c in self.downbeats), f"Invalid downbeats: {self.downbeats}"
        assert all(0 <= c < 600 for c in self.beats), f"Invalid beats: {self.beats}"
        assert self.views >= 0
        assert self.length > 0
        assert _is_sorted(self.chord_times)
        assert _is_sorted(self.downbeats)
        assert _is_sorted(self.beats)

    def __repr__(self):
        return f"DatasetEntry([{self.url.video_id}])"

    def equal(self, value: DatasetEntry, *, eps: float = 1e-5) -> bool:
        """Check if the given value is equal to this entry. This is useful for testing purposes.
        This is a bit more lenient than __eq__ as it allows for some floating point error."""
        if self.url != value.url:
            return False
        if self.genre != value.genre:
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

    @property
    def _cache_handler(self):
        from ..cache import LocalCache
        return LocalCache("resources/cache", self.url)

    def get_audio(self) -> Audio:
        return self._cache_handler.get_audio()

    @property
    def cached(self):
        return self._cache_handler.cached_audio

    @staticmethod
    def from_url(url: YouTubeURL, genre: SongGenre = SongGenre.UNKNOWN):
        from .create import process_audio
        audio = Audio.load(url)
        entry = process_audio(audio, url, genre, verbose=False)
        if isinstance(entry, str):
            raise ValueError(f"Failed to process audio: {entry}")
        return entry

class SongDataset:
    """A data structure that holds a bunch of dataset entries for query. Use a hashmap for now. lmk if there are more efficient ways to do this."""
    def __init__(self):
        self._data: dict[YouTubeURL, DatasetEntry] = {}

    def get_by_url(self, url: YouTubeURL) -> DatasetEntry | None:
        return self._data.get(url, None)

    def __getitem__(self, url: str | YouTubeURL | int) -> DatasetEntry:
        if isinstance(url, YouTubeURL):
            return self._data[url]
        if isinstance(url, str):
            return self._data[get_url(url)]
        if isinstance(url, int):
            return list(self._data.values())[url]
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
        """Loads a v3 dataset from the given path. Only .db, .fast.db, and a folder of .dat3 files are supported"""
        from .v3 import SongDatasetEncoder, DatasetEntryEncoder, FastSongDatasetEncoder

        if os.path.isfile(dataset_path) and "fast.db" in dataset_path:
            try:
                return FastSongDatasetEncoder().read_from_path(dataset_path)
            except Exception as e:
                raise ValueError(f"Error reading dataset: {e}")

        if os.path.isfile(dataset_path):
            try:
                return SongDatasetEncoder().read_from_path(dataset_path)
            except Exception as e:
                raise ValueError(f"Error reading dataset: {e}")

        data_files = [f for f in os.listdir(dataset_path) if f.endswith(".dat3")]
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

    def __repr__(self):
        return f"SongDataset({len(self)} entries)"

    def pack(self, path_out: str = "dataset_v3.db"):
        from .v3 import SongDatasetEncoder
        dataset_encoder = SongDatasetEncoder()
        dataset_encoder.write_to_path(self, path_out)

        read_dataset = dataset_encoder.read_from_path(path_out)
        print(f"Read dataset from {path_out} ({len(read_dataset)} entries)")
        for url, entry in tqdm(read_dataset._data.items(), total=len(read_dataset)):
            read_entry = self.get_by_url(url)
            if read_entry is None:
                raise ValueError(f"Entry {entry} not found in read dataset")
            if entry != read_entry:
                print(f"Entry {entry} mismatch")
                raise ValueError(f"Entry {entry} mismatch")

        print("Dataset packed successfully")

# Provides a unified directory structure and API that defines a AutoMasher v3 dataset
class LocalSongDataset(SongDataset):
    """New in v3

    Provides a unified definition of a directory structure that ought to define a v3 dataset. This is useful particularly for loading datasets from a directory.

    Supports reading on-the-fly from a directory structure. The directory structure should be as follows:
    - audio-infos-v3/
        - audio/
            - <youtube_id>.mp3
        - datafiles/
            - <youtube_id>.dat3
        - error_logs.txt
        - log.json

    Where:
    - <youtube_id> is the youtube video id
    - <youtube_id>.mp3 is the audio file
    - <youtube_id>.dat3 is the datafile containing the song information
    - error_logs.txt is a file containing error logs
    - log.json is a file containing information about calculations done on the dataset
    """
    def __init__(self, root: str, *, load_on_the_fly: bool = False):
        from .v3 import DatasetEntryEncoder

        if not os.path.exists(root):
            raise FileNotFoundError(f"Directory {root} does not exist")

        self.root = root
        self.load_on_the_fly = load_on_the_fly

        self.init_directory_structure()
        directory_invalid_reason = self._check_directory_structure()
        if directory_invalid_reason is not None:
            raise ValueError(f"Invalid directory structure: {directory_invalid_reason}")

        self._encoder = DatasetEntryEncoder()

        super(SongDataset, self).__init__()

        if not load_on_the_fly:
            self._load_from_directory()

    def init_directory_structure(self):
        """Checks if the directory structure is correct"""
        if not os.path.exists(self.datafile_path):
            os.makedirs(self.datafile_path)

        if not os.path.exists(self.audio_path):
            os.makedirs(self.audio_path)

        if not os.path.exists(self.error_logs_path):
            with open(self.error_logs_path, "w") as f:
                f.write("")

        if not os.path.exists(self.info_path):
            with open(self.info_path, "w") as f:
                json.dump({}, f)

    def _check_directory_structure(self) -> str | None:
        """Checks if the files in the respective directories are correct"""
        for file in os.listdir(self.datafile_path):
            if not file.endswith(".dat3"):
                return f"Invalid datafile: {file}"

        for file in os.listdir(self.audio_path):
            if not file.endswith(".mp3"):
                return f"Invalid audio: {file}"

    def _load_from_directory(self):
        for file in os.listdir(self.datafile_path):
            if not file.endswith(".dat3"):
                continue
            try:
                entry = self._encoder.read_from_path(os.path.join(self.datafile_path, file))
            except Exception as e:
                self.write_error(f"Error reading {file}", e)
                continue
            self._data[entry.url] = entry

    def write_error(self, error: str, e: Exception | None = None):
        try:
            with open(self.error_logs_path, "a") as f:
                f.write(error + "\n")
                if e:
                    f.write(str(e) + "\n")
                    f.write(traceback.format_exc() + "\n")
                    f.write("\n")
        except Exception as e2:
            print(f"Error : {e2}")
            print(f"Error writing error: {error}")
            print(f"Error writing error: {e}")
            print(f"Error writing error: {traceback.format_exc()}")

    @property
    def datafile_path(self):
        return os.path.join(self.root, "datafiles")

    @property
    def audio_path(self):
        return os.path.join(self.root, "audio")

    @property
    def error_logs_path(self):
        return os.path.join(self.root, "error_logs.txt")

    @property
    def info_path(self):
        return os.path.join(self.root, "log.json")

    def get_data_path(self, url: YouTubeURL):
        """Return the path to the datafile of the given url"""
        return os.path.join(self.datafile_path, f"{url.video_id}.dat3")

    def get_audio_path(self, url: YouTubeURL):
        """Return the path to the audio file of the given url"""
        return os.path.join(self.audio_path, f"{url.video_id}.mp3")

    def get_by_url(self, url: YouTubeURL) -> DatasetEntry | None:
        if url in self._data:
            return self._data[url]
        if self.load_on_the_fly:
            file = f"{url.video_id}.dat3"
            try:
                entry = self._encoder.read_from_path(os.path.join(self.datafile_path, file))
                return entry
            except Exception as e:
                self.write_error(f"Error reading {file}", e)
                return None
        return None

    def __getitem__(self, url: str | YouTubeURL | int) -> DatasetEntry:
        if isinstance(url, YouTubeURL):
            entry = self.get_by_url(url)
            if entry is None:
                raise KeyError(f"URL {url} not found")
            return entry
        if isinstance(url, str):
            entry = self.get_by_url(get_url(url))
            if entry is None:
                raise KeyError(f"URL {url} not found")
            return entry
        if isinstance(url, int):
            file = os.listdir(self.datafile_path)[url]
            entry = self._encoder.read_from_path(os.path.join(self.datafile_path, file))
            return entry
        raise TypeError(f"Invalid type for url: {type(url)}")

    def __len__(self):
        return len(os.listdir(self.datafile_path)) if self.load_on_the_fly else len(self._data)

    def __iter__(self):
        if self.load_on_the_fly:
            for file in os.listdir(self.datafile_path):
                try:
                    entry = self._encoder.read_from_path(os.path.join(self.datafile_path, file))
                    yield entry
                except Exception as e:
                    self.write_error(f"Error reading {file}", e)
                    continue
        else:
            yield from self._data.values()

    def add_entry(self, entry: DatasetEntry):
        """This adds an entry to the dataset and checks for the presence of audio"""
        audio_path = self.get_audio_path(entry.url)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found")
        if not self.load_on_the_fly:
            self._data[entry.url] = entry
        path = os.path.join(self.datafile_path, f"{entry.url.video_id}.dat3")
        if not os.path.isfile(path):
            self._encoder.write_to_path(entry, path)

    def filter(self, filter_func: Callable[[DatasetEntry], bool] | None) -> SongDataset:
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
        if self.load_on_the_fly:
            return [get_url(file[:-5]) for file in os.listdir(self.datafile_path)]
        return list(self._data.keys())

    def __repr__(self):
        return f"LocalSongDataset(at: {self.root}, {len(self)} entries)"

    def write_info(self, key: str, value: YouTubeURL, desc: str | None = None):
        with open(self.info_path, "r") as f:
            info = json.load(f)
        if key not in info:
            info[key] = []

        if desc is not None:
            assert all(isinstance(x, list) and len(x) == 2 for x in info[key]), f"Invalid info format: key: {key} should contain description and url"
            if value.video_id in [x[1] for x in info[key]]:
                return
            info[key].append([desc, value.video_id])
        else:
            assert all(isinstance(x, str) for x in info[key]), f"Invalid info format: key: {key} should contain only urls"
            if value.video_id in info[key]:
                return
            info[key].append(value.video_id)

        with open(self.info_path, "w") as f:
            json.dump(info, f)

    def read_info(self, key: str) -> list[YouTubeURL] | dict[YouTubeURL, str] | None:
        with open(self.info_path, "r") as f:
            info = json.load(f)
        if not key in info:
            return None
        if all(isinstance(x, str) for x in info[key]):
            return [get_url(x) for x in info[key]]
        return {get_url(x[1]): x[0] for x in info[key]}

    def read_info_urls(self, key: str) -> set[YouTubeURL]:
        infos = self.read_info(key)
        if infos is None:
            return set()
        if isinstance(infos, dict):
            return set(infos.keys())
        return set(infos)
