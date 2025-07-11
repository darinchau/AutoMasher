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
from ..separation import DemucsCollection, demucs_separate
from ..analysis import BeatAnalysisResult, ChordAnalysisResult, analyse_beat_transformer, analyse_chord_transformer
from ...util import get_inv_voca_map
from enum import Enum
import json
import traceback
from typing import Callable
from numpy.typing import NDArray
import numpy as np
from functools import lru_cache
from tqdm.auto import tqdm
from typing import Iterable
import pickle
import numba
import warnings
import tempfile
import shutil
import typing
import zipfile

PURGE_ERROR_LIMIT_BYTES = 1 << 32  # 4GB


@dataclass(frozen=True)
class DatasetEntry:
    """The main data structure in AutoMasher. This represents a single entry in the dataset.
    Typically we use the create_entry function to create a DatasetEntry object."""
    chords: ChordAnalysisResult
    downbeats: OnsetFeatures
    beats: OnsetFeatures
    url: YouTubeURL
    normalized_times: NDArray[np.float64]
    music_duration: NDArray[np.float64]
    source: str = ""

    @property
    def duration(self):
        return self.chords.duration

    def __post_init__(self):
        assert len(self.chords.features) == len(self.normalized_times), f"{len(self.chords.features)} != {len(self.normalized_times)}"

        assert len(self.downbeats) == len(self.music_duration), f"{len(self.downbeats)} != {len(self.music_duration)}"
        assert np.all(self.normalized_times < len(self.downbeats)), f"Normalized times out of bounds: entry cannot be larger than number of downbeats {self.normalized_times[-1]} >= {len(self.downbeats)}"
        assert np.all(self.normalized_times >= 0), f"Normalized times out of bounds: entry cannot be less than 0 = {self.normalized_times[0]} < {len(self.downbeats)}"

        assert self.chords.duration == self.downbeats.duration == self.beats.duration, f"Duration mismatch: {self.chords.duration} != {self.downbeats.duration} != {self.beats.duration}"
        assert self.chords.duration <= 600, f"Duration too long: {self.chords.duration}"

    def __repr__(self):
        return f"DatasetEntry([{self.url.video_id}])"

    def _to_dict(self) -> dict[str, Any]:
        return {
            "chords": {
                "labels": self.chords.features.tolist(),
                "times": self.chords.times.tolist(),
            },
            "downbeats": self.downbeats.onsets.tolist(),
            "beats": self.beats.onsets.tolist(),
            "duration": self.chords.duration,
            "url": str(self.url),
            "source": self.source,
        }

    @staticmethod
    def _from_dict(data: dict[str, Any]):
        chords = ChordAnalysisResult(
            features=np.array(data["chords"]["labels"], dtype=np.uint32),
            times=np.array(data["chords"]["times"]),
            duration=data["duration"],
        )
        downbeats = OnsetFeatures(
            duration=data["duration"],
            onsets=np.array(data["downbeats"]),
        )
        beats = OnsetFeatures(
            duration=data["duration"],
            onsets=np.array(data["beats"]),
        )
        url = get_url(data["url"])
        return create_entry(
            url,
            chords=chords,
            beats=beats,
            downbeats=downbeats,
            source=data["source"],
        )

    def save(self, path: str):
        """Save the entry to a file"""
        _safe_write_json(self._to_dict(), path)

    @staticmethod
    def load(path: str) -> DatasetEntry:
        """Load the entry from a file"""
        with open(path, "r") as f:
            data = json.load(f)
        return DatasetEntry._from_dict(data)


class SongDataset:
    """New in v3

    Provides a unified definition of a directory structure that ought to define a v3 dataset. This is useful particularly for loading datasets from a directory.

    Supports reading on-the-fly from a directory structure. The directory structure should be as follows:
    - audio-infos-v3/
        - audio/
            - <youtube_id>.mp3
        - datafiles/
            - <youtube_id>.dat3
        - parts/
            - <youtube_id>.demucs
        - error_logs.txt
        - log.json
        - pack.data
        - dataset.pkl
        - .db

    Where:
    - <youtube_id> is the youtube video id
    - <youtube_id>.mp3 is the audio file
    - <youtube_id>.dat3 is the datafile containing the song information
    - <youtube_id>.demucs is the parts file containing the separated parts of the audio
    - error_logs.txt is a file containing error logs
    - log.json is a file containing information about calculations done on the dataset
    - pack.data is a file containing the dataset in a compressed format
    - .db is a file containing the metadata of the dataset in json format

    Has the following attributes
    - root: str: The root directory of the dataset
    - load_on_the_fly: bool: Whether to load the dataset on the fly or not
    - assert_audio_exists: bool: Whether to assert that the audio file exists or not

    Has the following methods
    - init_directory_structure: None: Initializes the directory structure
    - add_key: None: Adds a key to the dataset
    - _check_directory_structure: str | None: Checks if the directory structure is correct
    - get_path: str: Gets the path for the given key and url
    - list_files: list[str]: Lists all the files in the given key
    - load_from_directory: None: Reloads the dataset from the directory into memory
    - write_error: None: Writes an error to the error log
    - __len__: int: Returns the length of the dataset
    - __iter__: Iterable[DatasetEntry]: Returns an iterator over the dataset
    - __contains__: bool: Checks if the dataset contains the given url
    - save_entry: None: Saves the entry to the dataset
    - filter: SongDataset: Returns a filtered version of the dataset
    - write_info: None: Writes information to the info file
    - read_info: list[YouTubeURL] | dict[YouTubeURL, str] | None: Reads information from the info file
    - read_info_urls: set[YouTubeURL]: Reads information from the info file and returns the urls
    - get_audio: Audio: Gets the audio for the given url
    - get_parts: DemucsCollection: Gets the parts for the given url
    - get_or_create_entry: DatasetEntry: Gets or creates the entry for the given url
    - pack: None: Packs the dataset into a single file
    - pickle: None: Pickles the dataset into a single file

    The dataset can be loaded from hugging face. Make sure to have hf datasets installed. Then put "hf://<dataset_name>" as the root directory.
    The dataset will be downloaded and unpacked into the directory: './resources/dataset'.
    """

    def __init__(
        self,
        root: str, *,
        load_on_the_fly: bool = False,
    ):
        self._data: dict[YouTubeURL, DatasetEntry] = {}
        self.root = root
        self.load_on_the_fly = load_on_the_fly
        self.filters: list[Callable[[DatasetEntry], bool]] = []

        if root.startswith("hf://"):
            hf_dataset_name = root[5:]
            self._maybe_load_from_hf(hf_dataset_name)
            loaded_from_hf = True
        else:
            loaded_from_hf = False

        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Directory {self.root} does not exist")

        self.init_directory_structure()
        directory_invalid_reason = self._check_directory_structure()
        if directory_invalid_reason is not None:
            raise ValueError(f"Invalid directory structure: {directory_invalid_reason}")

        if not self.list_files("datafiles") and not loaded_from_hf:
            warnings.warn(f"Dataset {self.root} is empty. This could be an error. If you are sure this is correct, please ignore this warning.")

        # There may be extra data in the dataset in places other than the packed db - but that shouldnt matter
        if not self.load_on_the_fly:
            self.load_from_directory()

        # Make backup of infos
        with open(self.get_path("info"), "r") as f:
            _info = json.load(f)
        _safe_write_json(_info, self.get_path("info") + ".bak")

    @classmethod
    def from_packed(cls, root: str):
        """Load the dataset from a packed file"""
        # Skip the initial load from the directory but load the packed file instead
        ds = cls(root, load_on_the_fly=True)
        ds.load_on_the_fly = False
        if not os.path.exists(ds.get_path("pack")):
            raise FileNotFoundError(f"Packed file {ds.get_path('pack')} does not exist")
        with open(ds.get_path("pack"), "rb") as f:
            data = json.load(f)
        assert isinstance(data, dict), f"Invalid packed data format: {type(data)}"
        for k, v in data.items():
            try:
                entry = create_entry(
                    get_url(k),
                    chords=ChordAnalysisResult(
                        features=np.array(v["chords"]["labels"], dtype=np.uint32),
                        times=np.array(v["chords"]["times"]),
                        duration=v["duration"],
                    ),
                    downbeats=OnsetFeatures(
                        duration=v["duration"],
                        onsets=np.array(v["downbeats"]),
                    ),
                    beats=OnsetFeatures(
                        duration=v["duration"],
                        onsets=np.array(v["beats"]),
                    ),
                    source=v.get("source", ""),
                )
                ds._data[k] = entry
            except KeyError as e:
                ds.write_error(f"Error loading {k}: {e}")
                continue
        return ds

    def _maybe_load_from_hf(self, dataset_name: str):
        """Load the dataset from hugging face"""
        try:
            import datasets
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets to load from hugging face: pip install datasets")

        if not os.path.exists("./resources/dataset"):
            os.makedirs("./resources/dataset")
        dataset = load_dataset(dataset_name, split="train")
        self.root = "./resources/dataset"

        for entry in tqdm(dataset, desc="Loading dataset from hf...", disable=False):
            url = get_url(entry["url"])
            try:
                entry = create_entry(
                    url,
                    chords=ChordAnalysisResult(
                        features=np.array(entry["chords"], dtype=np.uint32),
                        times=np.array(entry["chord_times"]),
                        duration=entry["length"],
                    ),
                    downbeats=OnsetFeatures(
                        duration=entry["length"],
                        onsets=np.array(entry["downbeats"]),
                    ),
                    beats=OnsetFeatures(
                        duration=entry["length"],
                        onsets=np.array(entry["beats"]),
                    ),
                    source="fyp",
                )
            except Exception as e:
                tqdm.write(f"Error loading {url}: {e}")
                continue
            self._data[url] = entry
            path = self.get_path("datafiles", url)
            if not os.path.isfile(path):
                entry.save(path)
            else:
                warnings.warn(f"File {path} already exists - skipping")

    def init_directory_structure(self):
        """Checks if the directory structure is correct"""
        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, "w") as f:
                json.dump({}, f)

        self._register("audio")
        self._register("parts", "{video_id}.wav.demucs")
        self._register("datafiles", "{video_id}.dat3")
        self._register("info", "info.json", initial_data="{}")
        self._register("error", "error_logs.txt")
        self._register("pack", "pack.data", create=False)

    def _register(self, key: str, file_format: str = "", *, create: bool = True, initial_data: str | None = None):
        """Add a type of file to the dataset. The file format is a string that describes the format of the file (e.g. "{video_id}.dat3)

        Leave the file format blank for a variable directory structure, where a json file will be created in the key directory
        to map the video id to the file name on the fly.

        The file format should contain the string "{video_id}" which will be replaced by the video id of the url
        If it does not, then a file is created in the root directory

        If the file format is empty, then a variable directory structure is created, where a json file is created in the key directory"""
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        if not "file_structure" in metadata:
            metadata["file_structure"] = {}
        metadata["file_structure"][key] = file_format
        _safe_write_json(metadata, self.metadata_path)
        if file_format == "":
            # Variable directory structure, create a json file in the key directory
            if not os.path.exists(os.path.join(self.root, key)):
                os.makedirs(os.path.join(self.root, key))
            if not os.path.isfile(os.path.join(self.root, key, "info.json")):
                with open(os.path.join(self.root, key, "info.json"), "w") as f:
                    json.dump({}, f)
        elif "{video_id}" in file_format and not os.path.exists(os.path.join(self.root, key)):
            # id-indexed directory structure, create a directory for the key
            os.makedirs(self.root + "/" + key)
        elif create and "{video_id}" not in file_format and not os.path.isfile(os.path.join(self.root, file_format)):
            # Fixed file format, create the file in the root directory
            with open(self.root + "/" + file_format, "w") as f:
                f.write(initial_data or "")
        directory_invalid_reason = self._check_directory_structure()
        if directory_invalid_reason is not None:
            raise ValueError(f"Invalid directory structure: {directory_invalid_reason}")

    def _check_directory_structure(self) -> str | None:
        """Checks if the files in the respective directories are correct"""
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        for key, file_format in metadata["file_structure"].items():
            if file_format == "":
                # Variable directory structure, check if the info.json file exists
                # TODO add more checks if necessary
                if not os.path.exists(os.path.join(self.root, key, "info.json")):
                    return f"File {key}/info.json does not exist"
            if "{video_id}" in file_format:
                if not os.path.exists(self.root + "/" + key):
                    return f"File {key} does not exist"
        return None

    def get_path(self, key: str, url: YouTubeURL | None = None, filename: str | None = None) -> str:
        """Get the (absolute) file path for the given key and url"""
        if url is not None and url.is_placeholder:
            raise ValueError("Cannot get data path for placeholder url")
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        if key not in metadata["file_structure"]:
            raise ValueError(f"Key {key} not registered")
        file_format: str = metadata["file_structure"][key]

        # Something like error.txt or info.json matches here
        if file_format != "" and "{video_id}" not in file_format:
            if url is not None:
                print(f"URL provided for {key} is not required and will be ignored")
            return os.path.join(self.root, file_format)

        if file_format == "":
            if url is None:
                raise ValueError(f"Invalid file format for {key} (variable file format) - a URL is expected")
            with open(os.path.join(self.root, key, "info.json"), "r") as f:
                info = json.load(f)
            if url.video_id in info:
                if filename is not None:
                    print(f"File already exists for {key} and url {url}, ignoring provided filename {filename}")
                filename = info[url.video_id]
            elif filename is None:
                raise ValueError(f"URL {url} not found in info.json for key {key} (variable file format)")
        else:
            if url is None:
                raise ValueError(f"Invalid file format for {key}: {file_format} - a URL is expected")
            filename = file_format.format(video_id=url.video_id)
        if filename is None:
            raise ValueError(f"Cannot resolve filename for key {key} and url {url} (variable file format)?")
        subindex = url.video_id[:2]
        path = os.path.join(self.root, key, subindex, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def has_path(self, key: str, url: YouTubeURL) -> bool:
        """Check if the file path for the given key and url exists"""
        try:
            path = self.get_path(key, url)
        except ValueError as e:
            if "(variable file format)" in str(e):
                return False
            raise e
        return os.path.isfile(path)

    def set_path(self, key: str, url: YouTubeURL, path: str):
        """Set the file path for the given key and url. This is useful for variable directory structures"""
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        if key not in metadata["file_structure"]:
            raise ValueError(f"Key {key} not registered")
        json_path = os.path.join(self.root, key, "info.json")
        if metadata["file_structure"][key] == "":
            # Variable directory structure, write to info.json
            with open(json_path, "r") as f:
                info = json.load(f)
            info[url.video_id] = os.path.basename(path)
            _safe_write_json(info, json_path)
            return
        raise ValueError(f"Cannot set path for {key} - it is not a variable directory structure")

    def list_files(self, key: str) -> list[str]:
        """List all the files in the given key. Returns a list of filenames (not full paths)"""
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        if key not in metadata["file_structure"]:
            raise ValueError(f"Key {key} not registered")
        return os.listdir(os.path.join(self.root, key))

    def list_urls(self, key: str) -> list[YouTubeURL]:
        """List all the urls in the given key"""
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        if key not in metadata["file_structure"]:
            raise ValueError(f"Key {key} not registered")
        file_format: str = metadata["file_structure"][key]
        if file_format == "":
            # Variable directory structure, read the info.json file
            with open(os.path.join(self.root, key, "info.json"), "r") as f:
                info = json.load(f)
            return [YouTubeURL(f"{YouTubeURL.URL_PREPEND}{url_id}") for url_id in info.keys()]
        if "{video_id}" not in file_format:
            raise ValueError(f"Invalid file format for {key}: {file_format} - a video_id is expected")
        len_prepend = len(file_format.split("{video_id}")[0])
        len_append = len(file_format.split("{video_id}")[1])
        return [get_url(file[len_prepend:-len_append]) for file in self.list_files(key)]

    def load_from_directory(self, verbose: bool = True):
        """Reloads the dataset from the directory into memory"""
        for file in tqdm(self.list_files("datafiles"), desc="Loading dataset", disable=not verbose):
            url = get_url(file[:-5])
            entry = self.get_entry(url, _skip_filecheck=True)
            if entry is None:
                continue
            self._data[entry.url] = entry

    def write_error(self, error: str, e: Exception | None = None, print_fn: Callable[[str], Any] | None = print):
        def print_fn_(x): return print_fn(x) if print_fn is not None else None
        print_fn_(f"Error: {error}")
        if e:
            print_fn_(f"Error: {e}")
            print_fn_(f"Error: {traceback.format_exc()}")
        try:
            with open(self.error_logs_path, "a") as f:
                f.write(error + "\n")
                if e:
                    f.write(str(e) + "\n")
                    f.write(traceback.format_exc() + "\n")
                    f.write("\n")
        except Exception as e2:
            print_fn_(f"Error : {e2}")
            print_fn_(f"Error writing error: {error}")
            print_fn_(f"Error writing error: {e}")
            print_fn_(f"Error writing error: {traceback.format_exc()}")

    @property
    def metadata_path(self):
        return os.path.join(self.root, ".db")

    @property
    def error_logs_path(self):
        return self.get_path("error")

    def get_entry(self, url: YouTubeURL, *, _skip_filecheck: bool = False) -> DatasetEntry | None:
        """Try to get the entry by url. If it does not exist, return None"""
        if url.is_placeholder:
            return None
        if url in self._data and all(f(self._data[url]) for f in self.filters):
            return self._data[url]
        file = self.get_path("datafiles", url)
        file_url = get_url(file[-16:-5])
        if not _skip_filecheck and not os.path.isfile(file):
            return None
        try:
            entry = DatasetEntry.load(file)
            if not all(f(entry) for f in self.filters):
                return None
            return entry
        except Exception as e:
            self.write_error(f"Error reading {file}", e)
            return None

    def __len__(self):
        return len(self.list_files("datafiles")) if self.load_on_the_fly else len(self._data)

    def __iter__(self):
        if self.load_on_the_fly:
            urls = self.list_urls("datafiles")
            for url in urls:
                entry = self.get_entry(url)
                if entry is not None:
                    yield entry
        else:
            yield from self._data.values()

    def __contains__(self, url: YouTubeURL):
        return self.get_entry(url) is not None

    def save_entry(self, entry: DatasetEntry):
        """This adds an entry to the dataset and checks for the presence of audio"""
        if not self.load_on_the_fly:
            self._data[entry.url] = entry
        path = self.get_path("datafiles", entry.url)
        if not os.path.isfile(path):
            entry.save(path)

    def filter(self, filter_func: Callable[[DatasetEntry], bool] | None):
        """Returns self with the filter applied lazily"""
        if filter_func is None:
            return self
        if self.load_on_the_fly:
            self.filters.append(filter_func)
        else:
            self._data = {url: entry for url, entry in self._data.items() if filter_func(entry)}
        return self

    def __repr__(self):
        return f"LocalSongDataset(at: {self.root}, {len(self)} entries)"

    def write_info(self, key: str, value: YouTubeURL, desc=None, *, indent: int | str | None = None):
        with open(self.get_path("info"), "r") as f:
            info = json.load(f)
        if key not in info and desc is None:
            info[key] = []
        elif key not in info:
            info[key] = {}

        if desc is not None:
            assert isinstance(info[key], dict), f"Invalid info format: key: {key} should contain description and url"
            if value.video_id in info[key]:
                return
            info[key][value.video_id] = desc
        else:
            assert isinstance(info[key], list), f"Invalid info format: key: {key} should contain only urls"
            if value.video_id in info[key]:
                return
            info[key].append(value.video_id)

        _safe_write_json(info, self.get_path("info"))

    def read_info(self, key: str) -> list[YouTubeURL] | dict[YouTubeURL, str] | None:
        with open(self.get_path("info"), "r") as f:
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

    def get_audio(self, url: YouTubeURL) -> Audio:
        if url.is_placeholder:
            raise ValueError("Cannot get audio for placeholder url")
        path = self.get_path("audio", url)
        if os.path.isfile(path):
            return Audio.load(path)
        # Save and reload to ensure consistency
        audio = Audio.download(url)
        audio.save(path)
        audio = Audio.load(path)
        return audio

    def get_parts(self, url: YouTubeURL) -> DemucsCollection:
        if url.is_placeholder:
            raise ValueError("Cannot get parts for placeholder url")
        path = self.get_path("parts", url)
        if os.path.isfile(path):
            return DemucsCollection.load(path)
        # Save and reload to ensure consistency
        parts = demucs_separate(self.get_audio(url))
        parts.save(path)
        return parts

    def get_or_create_entry(self, url: YouTubeURL, **kwargs) -> DatasetEntry:
        """Gets the entry from the dataset, or if it doesn't exist, create one. Any excess kwargs will be passed to create_entry"""
        if url.is_placeholder:
            raise ValueError("Cannot get or create entry for placeholder url")
        entry = self.get_entry(url)
        if entry is None:
            audio = self.get_audio(url)
            entry = create_entry(url, dataset=self, audio=audio, **kwargs)
        return entry

    def pack(self):
        """Packs the dataset into a single file"""
        data: dict[YouTubeURL, DatasetEntry] = {}
        if self.load_on_the_fly:
            for entry in self:
                data[entry.url] = entry
        else:
            data = self._data
        data_ = {k: v._to_dict() for k, v in data.items()}
        _safe_write_json(data_, self.get_path("pack"))


def _safe_write_json(data, filename):
    temp_fd, temp_path = tempfile.mkstemp()

    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
            json.dump(data, temp_file, indent=4, ensure_ascii=False)
        shutil.move(temp_path, filename)
    except Exception as e:
        print(f"Failed to write data: {e}")
        os.unlink(temp_path)


def get_normalized_times(unnormalized_times: NDArray[np.float64], br: OnsetFeatures) -> NDArray[np.float64]:
    """Normalize the chord result with the beat result. This is done by retime the chord result as the number of downbeats."""
    # For every time stamp in the chord result, retime it as the number of downbeats.
    # For example, if the time stamp is half way between downbeat[1] and downbeat[2], then it should be 1.5
    # If the time stamp is before the first downbeat, then it should be 0.
    # If the time stamp is after the last downbeat, then it should be the number of downbeats.
    # Note: This is somehow faster than jitting the function because we are not using computation-intensive functions like np.searchsorted
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


@numba.njit
def _get_music_durations(times: NDArray[np.float64], features: NDArray[np.uint32], duration: float, no_chord_idx: int, total_nbars: int) -> NDArray[np.float64]:
    music_durations = np.zeros((total_nbars,), dtype=np.float64)
    start_idx = 0
    for i in range(total_nbars):
        end_idx = np.searchsorted(times, i + 1, side='right')
        new_times = times[start_idx:end_idx] - i
        new_times[0] = 0.
        music_duration = 0.
        new_features = features[start_idx:end_idx]
        for j in range(len(new_times) - 1):
            if new_features[j] != no_chord_idx:
                music_duration += new_times[j + 1] - new_times[j]
        if new_features[-1] != no_chord_idx:
            music_duration += duration - new_times[-1]
        music_durations[i] = music_duration
        start_idx = end_idx - 1
    return music_durations


# Returns None if the chord result is valid, otherwise returns an error message


def verify_chord_result(cr: ChordAnalysisResult, audio_duration: float, video_url: YouTubeURL | None = None) -> str | None:
    labels = cr.features
    chord_times = cr.times

    if len(labels) != len(chord_times):
        return f"Length mismatch: labels({len(labels)}) != chord_times({len(chord_times)})"

    # Check if the chord times are sorted monotonically
    if len(chord_times) == 0 or chord_times[-1] > audio_duration:
        return f"Chord times error: {video_url}"

    if len(chord_times) < 12:
        return f"Too few chords: {video_url} ({len(chord_times)})"

    # Check if the chord times are sorted monotonically
    if not all([t1 < t2 for t1, t2 in zip(chord_times, chord_times[1:])]):
        return f"Chord times not sorted monotonically: {video_url}"

    # New in v3: Check if there are enough chords
    no_chord_duration = 0.
    for t1, t2, c in zip(chord_times, chord_times[1:], labels):
        if c == get_inv_voca_map()["No chord"]:
            no_chord_duration += t2 - t1
    if no_chord_duration > 0.5 * audio_duration:
        return f"Too much no chord: {video_url} ({no_chord_duration}) (Proportion: {no_chord_duration / audio_duration}) - probably this is not a song"

    chord_times = np.round(chord_times * 10.8).astype(int)
    diffs = np.diff(chord_times)
    if np.any(diffs < 1):
        return f"Chord times too close: {video_url} ({diffs})"

    return None


def verify_parts_result(parts: DemucsCollection, mean_vocal_threshold: float, video_url: YouTubeURL | None = None) -> str | None:
    # New in v3: Check if there are enough vocals
    mean_vocal_volume = parts.vocals.volume
    if mean_vocal_volume < mean_vocal_threshold:
        return f"Too few vocals: {video_url} ({mean_vocal_volume})"
    return None


def verify_beats_result(br: BeatAnalysisResult, audio_duration: float, video_url: YouTubeURL | None = None, reject_weird_meter: bool = True, bad_alignment_threshold: float = 0.1) -> str | None:
    """Verify the beat result. If strict is True, then it will reject songs with weird meters."""
    # New in v3: Reject if there are too few downbeats
    if len(br.downbeats) < 12:
        return f"Too few downbeats: {video_url} ({len(br.downbeats)})"

    # New in v3: Reject songs with weird meters
    # Remove all songs with 3/4 meter as well because 96% of the songs are 4/4
    # This is rejecting way too many songs. Making this optional
    beat_align_idx = np.abs(br.beats[:, None] - br.downbeats[None, :]).argmin(axis=0)
    nbeat_in_bar = beat_align_idx[1:] - beat_align_idx[:-1]
    if reject_weird_meter and not np.all(nbeat_in_bar == 4):
        return f"Weird meter: {video_url} ({nbeat_in_bar})"

    # New in v3: Reject songs with a bad alignment
    beat_alignment = np.abs(br.beats[:, None] - br.downbeats[None, :]).min(axis=0)
    if np.max(beat_alignment) > bad_alignment_threshold:
        return f"Bad alignment: {video_url} ({np.max(beat_alignment)})"

    # Check if beats and downbeats make sense
    if len(br.beats) == 0 or br.beats[-1] >= audio_duration:
        return f"Beats error: {video_url}"

    if len(br.downbeats) == 0 or br.downbeats[-1] >= audio_duration:
        return f"Downbeats error: {video_url}"

    if not all([b1 < b2 for b1, b2 in zip(br.beats, br.beats[1:])]):
        return f"Beats not sorted monotonically: {video_url}"

    if not all([d1 < d2 for d1, d2 in zip(br.downbeats, br.downbeats[1:])]):
        return f"Downbeats not sorted monotonically: {video_url}"

    return None


def create_entry(url: YouTubeURL, *,
                 dataset: SongDataset | None = None,
                 audio: Audio | None = None,
                 chords: ChordAnalysisResult | None = None,
                 chord_times: list[float] | None = None,
                 chord_labels: list[int] | None = None,
                 beats: OnsetFeatures | None = None,
                 downbeats: OnsetFeatures | None = None,
                 beats_list: list[float] | None = None,
                 downbeats_list: list[float] | None = None,
                 duration: float | None = None,
                 chord_model_path: str | None = None,
                 beat_model_path: str | None = None,
                 chord_regularizer: float = 0.5,
                 beat_backend: typing.Literal["demucs", "spleeter"] = "demucs",
                 beat_backend_url: str | None = None,
                 use_simplified_chord: bool = False,
                 use_chord_cache: bool = True,
                 use_beat_cache: bool = True,
                 source: str = "",
                 strict: bool = False) -> DatasetEntry:
    """Creates the dataset entry from the data - performs normalization and music duration postprocessing

    If the dataset is provided, then the audio, chords, beats and downbeats can be None. In this case, the audio, chords, beats and downbeats will be loaded from the dataset.
    If the dataset is not provided, then the audio, chords, beats and downbeats must be provided.
    If the duration is not provided, then the audio, chords, beats or downbeats must be provided to calculate the duration.
    If the chord model path is provided, then the chords will be calculated using the chord model path.
    If the beat model path is provided, then the beats and downbeats will be calculated using the beat model path.
    If the use_simplified_chord is True, then the simplified chord model will be used.
    If the strict is True, then it will raise an error on bad entries"""
    if dataset is not None:
        entry = dataset.get_entry(url)
        if entry is not None:
            return entry

    if duration is None:
        if audio is not None:
            duration = audio.duration
        elif chords is not None:
            duration = chords.duration
        elif beats is not None:
            duration = beats.duration
        elif downbeats is not None:
            duration = downbeats.duration
        elif dataset is not None:
            audio = dataset.get_audio(url)
            duration = audio.duration
        else:
            raise ValueError("If duration is not provided, then either audio, chords, beats or downbeats must be provided to calculate the duration")

    if chords is None and (chord_times is None or chord_labels is None):
        assert audio is not None, "Either chords or audio or (chord_times, chord_labels) must be provided"
        if chord_model_path is not None:
            chords = analyse_chord_transformer(
                audio,
                dataset=dataset,
                url=url,
                regularizer=chord_regularizer,
                model_path=chord_model_path,
                use_large_voca=not use_simplified_chord,
                use_cache=use_chord_cache
            )
        else:
            chords = analyse_chord_transformer(
                audio,
                dataset=dataset,
                url=url,
                regularizer=chord_regularizer,
                use_large_voca=not use_simplified_chord,
                use_cache=use_chord_cache
            )
        assert audio.duration == chords.duration, f"Duration mismatch: {audio.duration} != {chords.duration}"
    elif chords is None:
        assert chord_times is not None and chord_labels is not None, "Either chords or audio or (chord_times, chord_labels) must be provided"
        chords = ChordAnalysisResult.from_data(duration, chord_labels, chord_times)

    assert chords is not None
    if strict:
        chord_fail_reason = verify_chord_result(chords, duration, url)
        if chord_fail_reason is not None:
            raise ValueError(f"Chord verification failed: {chord_fail_reason}")

    if (beats is None and beats_list is None) or (downbeats is None and downbeats_list is None):
        assert audio is not None, "Either beats or downbeats or audio must be provided"
        if beat_backend == "demucs":
            parts = dataset.get_parts(url) if dataset is not None else demucs_separate(audio)
        else:
            parts = None
        if beat_model_path is not None:
            bt = analyse_beat_transformer(
                audio,
                dataset=dataset,
                url=url,
                parts=parts,
                backend=beat_backend,
                backend_url=beat_backend_url,
                model_path=beat_model_path,
                use_cache=use_beat_cache
            )
        else:
            bt = analyse_beat_transformer(
                audio,
                dataset=dataset,
                url=url,
                parts=parts,
                backend=beat_backend,
                backend_url=beat_backend_url,
                use_cache=use_beat_cache
            )
        beats = bt._beats
        downbeats = bt._downbeats
        assert audio.duration == beats.duration, f"Duration mismatch: {audio.duration} != {beats.duration}"
        assert audio.duration == downbeats.duration, f"Duration mismatch: {audio.duration} != {downbeats.duration}"

    if beats is None:
        assert beats_list is not None, "Either beats or beats_list must be provided"
        beats = OnsetFeatures(duration, np.array(beats_list, dtype=np.float64))

    if downbeats is None:
        assert downbeats_list is not None, "Either downbeats or downbeats_list must be provided"
        downbeats = OnsetFeatures(duration, np.array(downbeats_list, dtype=np.float64))

    assert beats is not None
    assert downbeats is not None
    if strict:
        beat_fail_reason = verify_beats_result(BeatAnalysisResult(beats, downbeats), duration, url)
        if beat_fail_reason is not None:
            raise ValueError(f"Beat verification failed: {beat_fail_reason}")

    normalized_times = get_normalized_times(chords.times, downbeats)
    normalized_cr = ChordAnalysisResult(
        duration=duration,
        features=chords.features,
        times=normalized_times
    )

    # For each bar, calculate its music duration
    no_chord_idx = get_inv_voca_map()["No chord"]
    music_duration = _get_music_durations(normalized_cr.times, normalized_cr.features, normalized_cr.duration, no_chord_idx, len(downbeats))
    return DatasetEntry(
        chords=chords,
        downbeats=downbeats,
        beats=beats,
        url=url,
        normalized_times=normalized_times,
        music_duration=music_duration,
        source=source
    )
