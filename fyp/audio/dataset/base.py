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
        - pack.db
        - dataset.pkl

    Where:
    - <youtube_id> is the youtube video id
    - <youtube_id>.mp3 is the audio file
    - <youtube_id>.dat3 is the datafile containing the song information
    - <youtube_id>.demucs is the parts file containing the separated parts of the audio
    - error_logs.txt is a file containing error logs
    - log.json is a file containing information about calculations done on the dataset
    - pack.db is a file containing the dataset in a compressed format
    """
    def __init__(self, root: str, *, load_on_the_fly: bool = False, assert_audio_exists: bool = False):
        from .v3 import DatasetEntryEncoder, SongDatasetEncoder

        if not os.path.exists(root):
            raise FileNotFoundError(f"Directory {root} does not exist")

        print(f"Initializing dataset at {root}")
        self.root = root
        self.load_on_the_fly = load_on_the_fly
        self.assert_audio_exists = assert_audio_exists
        self.filters: list[Callable[[DatasetEntry], bool]] = []

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
                self._data = SongDatasetEncoder().read_from_path(os.path.join(self.root, "pack.db"))
                self.pickle()
            else:
                self.load_from_directory()

        print(f"Dataset initialized at {root}")

    def init_directory_structure(self):
        """Checks if the directory structure is correct"""
        if not os.path.exists(self.datafile_path):
            os.makedirs(self.datafile_path)

        if not os.path.exists(self.audio_path):
            os.makedirs(self.audio_path)

        if not os.path.exists(self.error_logs_path):
            with open(self.error_logs_path, "w") as f:
                f.write("")

        if not os.path.exists(self.parts_path):
            os.makedirs(self.parts_path)

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

        for file in os.listdir(self.parts_path):
            if not file.endswith(".demucs"):
                return f"Invalid parts: {file}"

    def load_from_directory(self, verbose: bool = True):
        """Reloads the dataset from the directory into memory"""
        for file in tqdm(os.listdir(self.datafile_path), desc="Loading dataset", disable=not verbose):
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

    @property
    def pickle_path(self):
        return os.path.join(self.root, "dataset.pkl")

    @property
    def parts_path(self):
        return os.path.join(self.root, "parts")

    @property
    def encoder(self):
        if not hasattr(self, "_encoder"):
            from .v3 import DatasetEntryEncoder
            self._encoder = DatasetEntryEncoder()
        return self._encoder

    def get_data_path(self, url: YouTubeURL):
        """Return the path to the datafile of the given url"""
        if url.is_placeholder:
            raise ValueError("Cannot get data path for placeholder url")
        return os.path.join(self.datafile_path, f"{url.video_id}.dat3")

    def get_audio_path(self, url: YouTubeURL):
        """Return the path to the audio file of the given url"""
        if url.is_placeholder:
            raise ValueError("Cannot get audio path for placeholder url")
        return os.path.join(self.audio_path, f"{url.video_id}.mp3")

    def get_parts_path(self, url: YouTubeURL):
        """Return the path to the parts file of the given url"""
        if url.is_placeholder:
            raise ValueError("Cannot get parts path for placeholder url")
        return os.path.join(self.parts_path, f"{url.video_id}.demucs")

    def get_by_url(self, url: YouTubeURL) -> DatasetEntry | None:
        """Try to get the entry by url. If it does not exist, return None"""
        if url.is_placeholder:
            return None
        if url in self._data and all(f(self._data[url]) for f in self.filters):
            return self._data[url]
        file = self.get_data_path(url)
        file_url = get_url(file[-16:-5])
        if self.assert_audio_exists:
            audio_path = self.get_audio_path(file_url)
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
        return len(os.listdir(self.datafile_path)) if self.load_on_the_fly else len(self._data)

    def __iter__(self):
        urls = [get_url(file[:-5]) for file in os.listdir(self.datafile_path)] if self.load_on_the_fly else self._data.keys()
        for url in urls:
            entry = self.get_by_url(url)
            if entry is not None and all(f(entry) for f in self.filters):
                yield entry

    def __contains__(self, url: YouTubeURL):
        return self.get_by_url(url) is not None

    def save_entry(self, entry: DatasetEntry):
        """This adds an entry to the dataset and checks for the presence of audio"""
        audio_path = self.get_audio_path(entry.url)
        if self.assert_audio_exists and not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found")
        if not self.load_on_the_fly:
            self._data[entry.url] = entry
        path = os.path.join(self.datafile_path, f"{entry.url.video_id}.dat3")
        if not os.path.isfile(path):
            self.encoder.write_to_path(entry, path)

    def filter(self, filter_func: Callable[[DatasetEntry], bool] | None):
        """Returns self with the filter applied lazily"""
        if filter_func is None:
            return self
        self.filters.append(filter_func)
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
        if os.path.isfile(self.get_audio_path(url)):
            return Audio.load(self.get_audio_path(url))
        # Save and reload to ensure consistency
        audio = Audio.load(url)
        audio.save(self.get_audio_path(url))
        audio = Audio.load(self.get_audio_path(url))
        return audio

    def get_parts(self, url: YouTubeURL) -> DemucsCollection:
        if url.is_placeholder:
            raise ValueError("Cannot get parts for placeholder url")
        if os.path.isfile(self.get_parts_path(url)):
            return DemucsCollection.load(self.get_parts_path(url))
        # Save and reload to ensure consistency
        parts = get_demucs().separate(self.get_audio(url))
        parts.save(self.get_parts_path(url))
        parts = DemucsCollection.load(self.get_parts_path(url))
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
        from .v3 import SongDatasetEncoder
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

# This is now not needed during the search step because we have precalculated it
def get_music_duration(chord_result: ChordAnalysisResult):
    """Get the duration of actual music in the chord result. This is calculated by summing the duration of all chords that are not "No chord"."""
    music_duration = 0.
    times = chord_result.group().times + [chord_result.duration]
    no_chord_idx = ChordAnalysisResult.map_feature_name("No chord")
    for chord, start, end in zip(chord_result.features, times[:-1], times[1:]):
        if chord != no_chord_idx:
            music_duration += end - start
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
                 beat_model_path: str | None = None) -> DatasetEntry:
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
            chords = analyse_chord_transformer(audio, model_path=chord_model_path)
        else:
            chords = analyse_chord_transformer(audio)
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
    for i in range(len(downbeats)):
        bar_cr = normalized_cr.slice_seconds(i, i + 1)
        music_duration.append(get_music_duration(bar_cr))

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
