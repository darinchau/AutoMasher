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
from ..analysis import ChordAnalysisResult, OnsetFeatures
from enum import Enum
import json
import traceback
from typing import Callable
from numpy.typing import NDArray
import numpy as np
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
    """The main data structure in AutoMasher. This represents a single entry in the dataset.
    Typically we use the create_entry function to create a DatasetEntry object."""
    chords: ChordAnalysisResult
    downbeats: OnsetFeatures
    beats: OnsetFeatures
    genre: SongGenre
    url: YouTubeURL
    views: int
    normalized_times: NDArray[np.float64]
    music_duration: list[float]

    @property
    def duration(self):
        return self.chords.duration

    def __post_init__(self):
        assert len(self.chords.features) == len(self.normalized_times), f"{len(self.chords.features)} != {len(self.normalized_times)}"

        assert len(self.downbeats) == len(self.music_duration), f"{len(self.downbeats)} != {len(self.music_duration)}"
        assert np.all(self.normalized_times < len(self.downbeats)), f"Normalized times out of bounds: {self.normalized_times} >= {len(self.downbeats)}"
        assert np.all(self.normalized_times >= 0), f"Normalized times out of bounds: {self.normalized_times} >= {len(self.downbeats)}"

        assert self.chords.duration == self.downbeats.duration == self.beats.duration, f"Duration mismatch: {self.chords.duration} != {self.downbeats.duration} != {self.beats.duration}"
        assert self.chords.duration <= 600, f"Duration too long: {self.chords.duration}"
        assert self.views >= 0 or self.views == -1, f"Invalid views: {self.views}"

    def __repr__(self):
        return f"DatasetEntry([{self.url.video_id}])"

# Provides a unified directory structure and API that defines a AutoMasher v3 dataset
#TODO avoid saving the placeholder URL
class SongDataset:
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
        - pack.db

    Where:
    - <youtube_id> is the youtube video id
    - <youtube_id>.mp3 is the audio file
    - <youtube_id>.dat3 is the datafile containing the song information
    - error_logs.txt is a file containing error logs
    - log.json is a file containing information about calculations done on the dataset
    - pack.db is a file containing the dataset in a compressed format
    """
    def __init__(self, root: str, *, load_on_the_fly: bool = False, assert_audio_exists: bool = False):
        from .v3 import DatasetEntryEncoder, SongDatasetEncoder

        if not os.path.exists(root):
            raise FileNotFoundError(f"Directory {root} does not exist")

        self.root = root
        self.load_on_the_fly = load_on_the_fly
        self.assert_audio_exists = assert_audio_exists

        self.init_directory_structure()
        directory_invalid_reason = self._check_directory_structure()
        if directory_invalid_reason is not None:
            raise ValueError(f"Invalid directory structure: {directory_invalid_reason}")

        self._encoder = DatasetEntryEncoder()
        self._data: dict[YouTubeURL, DatasetEntry] = {}

        # There may be extra data in the dataset in places other than the packed db - but that shouldnt matter
        if os.path.exists(self.pack_path):
            self._data = SongDatasetEncoder().read_from_path(os.path.join(self.root, "pack.db"))

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

    def clean_directory(self) -> list[str]:
        """Ensures the all audios have corresponding datafiles and vice versa. Returns a list of paths to remove"""
        audio_files = {file[:-4] for file in os.listdir(self.audio_path)}
        data_files = {file[:-5] for file in os.listdir(self.datafile_path)}

        paths_to_remove = []

        for file in audio_files - data_files:
            paths_to_remove.append(os.path.join(self.audio_path, file + ".mp3"))
        for file in data_files - audio_files:
            paths_to_remove.append(os.path.join(self.datafile_path, file + ".dat3"))

        return paths_to_remove

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
            url = get_url(file[:-5])
            entry = self.get_by_url(url)
            if entry is None:
                continue
            self._data[entry.url] = entry

    def write_error(self, error: str, e: Exception | None = None):
        print(f"Error: {error}")
        if e:
            print(f"Error: {e}")
            print(f"Error: {traceback.format_exc()}")
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

    @property
    def pack_path(self):
        return os.path.join(self.root, "pack.db")

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
            file = self.get_data_path(url)
            if not file.endswith(".dat3"):
                return None
            file_url = get_url(file[:-5])
            if self.assert_audio_exists:
                audio_path = self.get_audio_path(file_url)
                if not os.path.isfile(audio_path):
                    return None
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

    def filter(self, filter_func: Callable[[DatasetEntry], bool] | None):
        """Returns a new dataset with the entries that satisfy the filter function. If filter_func is None, return yourself"""
        raise NotImplementedError   # TODO

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

def get_normalized_times(unnormalized_times: NDArray[np.float64], br: OnsetFeatures) -> NDArray[np.float64]:
    """Normalize the chord result with the beat result. This is done by retime the chord result as the number of downbeats."""
    # For every time stamp in the chord result, retime it as the number of downbeats.
    # For example, if the time stamp is half way between downbeat[1] and downbeat[2], then it should be 1.5
    # If the time stamp is before the first downbeat, then it should be 0.
    # If the time stamp is after the last downbeat, then it should be the number of downbeats.
    downbeats = br.onsets.tolist() + [br.duration]
    new_chord_times = []
    curr_downbeat, curr_downbeat_idx, next_downbeat = 0, 0, downbeats[1]
    for chord_times in unnormalized_times:
        while chord_times > next_downbeat:
            curr_downbeat_idx += 1
            curr_downbeat = next_downbeat
            next_downbeat = downbeats[curr_downbeat_idx + 1]
        normalized_time = curr_downbeat_idx + (chord_times - curr_downbeat) / (next_downbeat - curr_downbeat)
        new_chord_times.append(normalized_time)
    return np.array(new_chord_times, dtype=np.float64)
