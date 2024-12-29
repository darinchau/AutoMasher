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
from ..separation import DemucsAudioSeparator, DemucsCollection
from ..analysis import BeatAnalysisResult, ChordAnalysisResult, analyse_beat_transformer, analyse_chord_transformer
from ...util import get_inv_voca_map
from enum import Enum
import json
import traceback
from typing import Callable
from numpy.typing import NDArray
import numpy as np
from tqdm.auto import tqdm
from typing import Iterable
import pickle
import numba

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
    music_duration: NDArray[np.float64]

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
    """
    def __init__(self, root: str, *, load_on_the_fly: bool = False, assert_audio_exists: bool = False):
        from .compress import DatasetEntryEncoder, SongDatasetEncoder

        if not os.path.exists(root):
            raise FileNotFoundError(f"Directory {root} does not exist")

        self.root = root
        self.load_on_the_fly = load_on_the_fly
        self.assert_audio_exists = assert_audio_exists
        self.filters: list[Callable[[DatasetEntry], bool]] = []
        self._keys: set[str] = set()

        self.init_directory_structure()
        directory_invalid_reason = self._check_directory_structure()
        if directory_invalid_reason is not None:
            raise ValueError(f"Invalid directory structure: {directory_invalid_reason}")

        self._data: dict[YouTubeURL, DatasetEntry] = {}

        # There may be extra data in the dataset in places other than the packed db - but that shouldnt matter
        if not self.load_on_the_fly:
            if os.path.exists(self.pickle_path):
                with open(self.pickle_path, "rb") as f:
                    self._data = pickle.load(f)
            elif os.path.exists(self.pack_path):
                self._data = SongDatasetEncoder().read_from_path(self.pack_path)
                self.pickle()
            else:
                self.load_from_directory()

    def init_directory_structure(self):
        """Checks if the directory structure is correct"""
        if not os.path.exists(self.error_logs_path):
            with open(self.error_logs_path, "w") as f:
                f.write("")

        if not os.path.exists(self.info_path):
            with open(self.info_path, "w") as f:
                json.dump({}, f)

        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, "w") as f:
                json.dump({}, f)

    def add_key(self, key: str, file_format: str):
        """Add a type of file to the dataset. The file format is a string that describes the format of the file (e.g. "{video_id}.dat3)

        The file format should contain the string "{video_id}" which will be replaced by the video id of the url"""
        if key in self._keys:
            return
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        if not "file_structure" in metadata:
            metadata["file_structure"] = {}
        metadata["file_structure"][key] = file_format
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f)
        self._keys.add(key)
        directory_invalid_reason = self._check_directory_structure()
        if directory_invalid_reason is not None:
            raise ValueError(f"Invalid directory structure: {directory_invalid_reason}")

    def _check_directory_structure(self) -> str | None:
        """Checks if the files in the respective directories are correct"""
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        for key, file_format in metadata["file_structure"].items():
            if not os.path.exists(self.root + "/" + key):
                return f"File {key} does not exist"
            if "{video_id}" not in file_format:
                return f"Invalid file format for {key}: {file_format}"
            for file in self.list_files(key):
                if not len(file) == len(file_format.format(video_id="")) + 11:
                    return f"Invalid file format for {key}: {file} in {self.root}/{key}"
        return None

    def get_path(self, key: str, url: YouTubeURL) -> str:
        """Get the file path for the given key and url"""
        if url.is_placeholder:
            raise ValueError("Cannot get data path for placeholder url")
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        file_format: str = metadata[key]
        return os.path.join(self.root, key, file_format.format(video_id=url.video_id))

    def has_path(self, key: str, url: YouTubeURL) -> bool:
        """Check if the file path for the given key and url exists"""
        return os.path.isfile(self.get_path(key, url))

    def list_files(self, key: str) -> list[str]:
        """List all the files in the given key"""
        return os.listdir(os.path.join(self.root, key))

    def load_from_directory(self, verbose: bool = True):
        """Reloads the dataset from the directory into memory"""
        for file in tqdm(self.list_files("datafiles"), desc="Loading dataset", disable=not verbose):
            url = get_url(file[:-5])
            entry = self.get_by_url(url)
            if entry is None:
                continue
            self._data[entry.url] = entry

        if not os.path.isfile(self.pack_path):
            self.pack()

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
    def metadata_path(self):
        return os.path.join(self.root, ".db")

    @property
    def error_logs_path(self):
        return os.path.join(self.root, "error_logs.txt")

    @property
    def info_path(self):
        return os.path.join(self.root, "log.json")

    @property
    def pack_path(self):
        return os.path.join(self.root, "pack.db")

    @property
    def pickle_path(self):
        return os.path.join(self.root, "dataset.pkl")

    @property
    def encoder(self):
        if not hasattr(self, "_encoder"):
            from .compress import DatasetEntryEncoder
            self._encoder = DatasetEntryEncoder()
        return self._encoder

    def get_by_url(self, url: YouTubeURL) -> DatasetEntry | None:
        """Try to get the entry by url. If it does not exist, return None"""
        if url.is_placeholder:
            return None
        if url in self._data and all(f(self._data[url]) for f in self.filters):
            return self._data[url]
        file = self.get_path("datafiles", url)
        file_url = get_url(file[-16:-5])
        if not os.path.isfile(file):
            return None
        if self.assert_audio_exists:
            audio_path = self.get_path("audio", file_url)
            if not os.path.isfile(audio_path):
                return None
        try:
            entry = self.encoder.read_from_path(file)
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
            urls = [get_url(file[:-5]) for file in self.list_files("datafiles")]
            for url in urls:
                entry = self.get_by_url(url)
                if entry is not None:
                    yield entry
        else:
            yield from self._data.values()

    def __contains__(self, url: YouTubeURL):
        return self.get_by_url(url) is not None

    def save_entry(self, entry: DatasetEntry):
        """This adds an entry to the dataset and checks for the presence of audio"""
        audio_path = self.get_path("audio", entry.url)
        if self.assert_audio_exists and not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found")
        if not self.load_on_the_fly:
            self._data[entry.url] = entry
        path = self.get_path("datafiles", entry.url)
        if not os.path.isfile(path):
            self.encoder.write_to_path(entry, path)

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

    def get_audio(self, url: YouTubeURL) -> Audio:
        if url.is_placeholder:
            raise ValueError("Cannot get audio for placeholder url")
        path = self.get_path("audio", url)
        if os.path.isfile(path):
            return Audio.load(path)
        # Save and reload to ensure consistency
        audio = Audio.load(url)
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
        parts = get_demucs().separate(self.get_audio(url))
        parts.save(path)
        parts = DemucsCollection.load(path)
        return parts

    def get_or_create_entry(self, url: YouTubeURL) -> DatasetEntry:
        if url.is_placeholder:
            raise ValueError("Cannot get or create entry for placeholder url")
        entry = self.get_by_url(url)
        if entry is None:
            audio = self.get_audio(url)
            entry = create_entry(url, dataset=self, audio=audio)
        return entry

    def pack(self):
        """Packs the dataset into a single file"""
        from .compress import SongDatasetEncoder
        data: dict[YouTubeURL, DatasetEntry] = {}
        if self.load_on_the_fly:
            for entry in self:
                data[entry.url] = entry
        else:
            data = self._data
        SongDatasetEncoder().write_to_path(data, self.pack_path)

    def pickle(self):
        """Pickle the dataset into a single file"""
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self._data, f)

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

@numba.njit
def _get_music_duration(times: NDArray[np.float64], features: NDArray[np.uint32], duration: float, no_chord_idx: int, bar_idx: int) -> float:
    start_idx = np.searchsorted(times, bar_idx, side='right') - 1
    end_idx = np.searchsorted(times, bar_idx + 1, side='right')
    new_times = times[start_idx:end_idx] - bar_idx
    new_times[0] = 0.
    music_duration = 0.
    new_features = features[start_idx:end_idx]
    for i in range(len(new_times) - 1):
        if new_features[i] != no_chord_idx:
            music_duration += new_times[i + 1] - new_times[i]
    if new_features[-1] != no_chord_idx:
        music_duration += duration - new_times[-1]
    return music_duration

_DEMUCS = None
def get_demucs():
    global _DEMUCS
    if not _DEMUCS:
        _DEMUCS = DemucsAudioSeparator()
    return _DEMUCS

# Returns None if the chord result is valid, otherwise returns an error message
def verify_chord_result(cr: ChordAnalysisResult, audio_duration: float, video_url: YouTubeURL | None = None) -> str | None:
    labels = cr.features
    chord_times = cr.times

    if len(labels) != len(chord_times):
        return f"Length mismatch: labels({len(labels)}) != chord_times({len(chord_times)})"

    # Check if the chord times are sorted monotonically
    if len(chord_times) == 0 or chord_times[-1] > audio_duration:
        return f"Chord times error: {video_url}"

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
    beat_align_idx = np.abs(br.beats[:, None] - br.downbeats[None, :]).argmin(axis = 0)
    nbeat_in_bar = beat_align_idx[1:] - beat_align_idx[:-1]
    if reject_weird_meter and not np.all(nbeat_in_bar == 4):
        return f"Weird meter: {video_url} ({nbeat_in_bar})"

    # New in v3: Reject songs with a bad alignment
    beat_alignment = np.abs(br.beats[:, None] - br.downbeats[None, :]).min(axis = 0)
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

# Create a dataset entry from the given data
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
                 genre: SongGenre = SongGenre.UNKNOWN,
                 views: int | None = None,
                 chord_model_path: str | None = None,
                 beat_model_path: str | None = None,
                 use_simplified_chord: bool = False) -> DatasetEntry:
    """Creates the dataset entry from the data - performs normalization and music duration postprocessing"""
    if dataset is not None:
        entry = dataset.get_by_url(url)
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
            chords = analyse_chord_transformer(audio, model_path=chord_model_path, use_large_voca=not use_simplified_chord)
        else:
            chords = analyse_chord_transformer(audio, use_large_voca=not use_simplified_chord)
        chord_fail_reason = verify_chord_result(chords, duration, url)
        if chord_fail_reason is not None:
            raise ValueError(f"Chord verification failed: {chord_fail_reason}")
    elif chords is None:
        assert chord_times is not None and chord_labels is not None, "Either chords or audio or (chord_times, chord_labels) must be provided"
        chords = ChordAnalysisResult.from_data(duration, chord_labels, chord_times)

    if (beats is None and beats_list is None) or (downbeats is None and downbeats_list is None):
        assert audio is not None, "Either beats or downbeats or audio must be provided"
        parts = dataset.get_parts(url) if dataset is not None else get_demucs().separate(audio)
        if beat_model_path is not None:
            bt = analyse_beat_transformer(audio, parts, model_path=beat_model_path)
        else:
            bt = analyse_beat_transformer(audio, parts)
        beat_fail_reason = verify_beats_result(bt, duration, url)
        if beat_fail_reason is not None:
            raise ValueError(f"Beat verification failed: {beat_fail_reason}")
        beats = bt._beats
        downbeats = bt._downbeats

    if beats is None:
        assert beats_list is not None, "Either beats or beats_list must be provided"
        beats = OnsetFeatures(duration, np.array(beats_list, dtype=np.float64))

    if downbeats is None:
        assert downbeats_list is not None, "Either downbeats or downbeats_list must be provided"
        downbeats = OnsetFeatures(duration, np.array(downbeats_list, dtype=np.float64))

    normalized_times = get_normalized_times(chords.times, downbeats)
    normalized_cr = ChordAnalysisResult(
        duration=duration,
        features=chords.features,
        times=normalized_times
    )

    # For each bar, calculate its music duration
    music_duration: list[float] = []
    no_chord_idx = get_inv_voca_map()["No chord"]
    for i in range(len(downbeats)):
        bar_music_duration = _get_music_duration(normalized_cr.times, normalized_cr.features, normalized_cr.duration, no_chord_idx, i)
        music_duration.append(bar_music_duration)

    return DatasetEntry(
        chords=chords,
        downbeats=downbeats,
        beats=beats,
        genre=genre,
        url=url,
        views=views if views is not None else -1,
        normalized_times=normalized_times,
        music_duration=np.array(music_duration, dtype=np.float64)
    )
