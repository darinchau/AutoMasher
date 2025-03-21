# This module contains code that compresses the SongDataset into a binary format

import numpy as np
import re
import struct
from abc import ABC, abstractmethod
from typing import Iterator, TypeVar, Generic
from .base import DatasetEntry, SongDataset
from ...util import get_url, YouTubeURL
from collections import Counter
import numpy as np
from math import ceil, exp
from typing import Iterator
import struct
from .base import create_entry
import zlib
from functools import lru_cache
from tqdm.auto import tqdm
import pickle

T = TypeVar('T')


class BitsEncoder(ABC, Generic[T]):
    @abstractmethod
    def encode(self, data: T) -> Iterator[int]:
        """Encodes the data into a stream of bits. (int between 0 and 255)"""
        pass

    @abstractmethod
    def decode(self, data: Iterator[int]) -> T:
        """Returns the decoded data and the number of bits read from the input stream."""
        pass

    def read_from_path(self, path: str) -> T:
        def file_stream():
            with open(path, "rb") as f:
                stream = f.read()
                yield from stream
        return self.decode(file_stream())

    def write_to_path(self, obj: T, path: str):
        with open(path, "wb") as f:
            f.write(bytes(self.encode(obj)))


def make_huffman_tree(counter: dict[int, int]) -> tuple:
    # Reserve 0xF for padding and EOS
    # So we want the 0xF least common elements
    # Since the length of the counter decreases by 14 each iteration
    # To ensure each level has exactly 15 elements, the first merge
    # should operate on len(counter) % 14 elements
    def merge(counter: dict, n_elems: int):
        least_common = sorted([(k, v) for k, v in counter.items()], key=lambda x: x[1])[:n_elems]
        cumulative = 0
        elems = []
        for k, v in least_common:
            del counter[k]
            cumulative += v
            elems.append(k)
        counter[tuple(elems)] = cumulative
        return counter

    if len(counter) > 0xF:
        first_merge_n_elems = len(counter) % 14
        first_merge_n_elems = 14 if first_merge_n_elems == 0 else first_merge_n_elems
        counter = merge(counter, first_merge_n_elems)
        while len(counter) > 0xF:
            counter = merge(counter, 0xF)

    tree = tuple(counter.keys())
    assert len(tree) == 15

    # Turn the tree into a table
    table = {}

    def traverse(tree, path):
        if isinstance(tree, int):
            table[tree] = path
        else:
            for i, branch in enumerate(tree):
                traverse(branch, path + [i])
    traverse(tree, [])
    return tree, table


@lru_cache(maxsize=1)
def get_chord_time_label_codebook(resolution: float = 10.8):
    max_time = ceil(600 * resolution) + 1
    # The distribution of chord time diffs roughly follows a power law
    # So use that to build a huffman tree and then we have a context free codebook
    counter = {n: int(exp(-0.05 * n) * max_time) + 1 for n in range(1, max_time)}
    tree, table = make_huffman_tree(counter)
    return tree, table


@lru_cache(maxsize=1)
def get_beat_time_label_codebook(resolution: float = 44100/1024):
    max_time = int(resolution * 600) + 1
    scores = np.arange(1, max_time + 1)
    # Determined using scipy optimization on a subset of data
    # because the frequencies plot looks like a bimodal distribution
    # which makes sense since the datas is a combination of beat and downbeat times
    # I dont think this has to be crazily optimized
    # but if someone finds better constants feel free to open a PR
    x = [23.379, 21.324, 36902, 87.321, 70.501, 3560.3]
    scores = np.exp(-.5 * (scores - x[0])**2 / x[1]**2) * x[2] + np.exp(-.5 * (scores - x[3])**2 / x[4]**2) * x[5]
    scores = np.ceil(scores).astype(int)
    counter = {i: scores[i] for i in range(1, max_time)}
    tree, table = make_huffman_tree(counter)
    return tree, table


class Int32Encoder(BitsEncoder[int]):
    def encode(self, data: int) -> Iterator[int]:
        assert 0 <= data < 2**32
        yield from struct.pack('I', data)

    def decode(self, data: Iterator[int]) -> int:
        b: list[int] = []
        for i in range(4):
            b.append(next(data))
        return struct.unpack('I', bytes(b))[0]


class FourBitEncoder(BitsEncoder[list[int]]):
    def __init__(self):
        self.i32encoder = Int32Encoder()

    def encode(self, data: list[int]) -> Iterator[int]:
        yield from self.i32encoder.encode(len(data))
        elems = [b for b in data]
        if len(elems) % 2 != 0:
            elems.append(0)
        for i in range(0, len(elems), 2):
            yield (elems[i] << 4) | elems[i + 1]

    def decode(self, data: Iterator[int]) -> list[int]:
        length = self.i32encoder.decode(data)
        b = []
        num_elems = length // 2 if length % 2 == 0 else (length // 2) + 1
        for _ in range(num_elems):
            byte = next(data)
            b.append(byte >> 4)
            b.append(byte & 0xF)
        b = b[:length]
        return b


class ChordTimesEncoder(BitsEncoder[list[float]]):
    def __init__(self, resolution: float = 10.8):
        self.tree, self.table = get_chord_time_label_codebook(resolution)
        self.fourbitencoder = FourBitEncoder()
        self.resolution = resolution

    def encode(self, data: list[float]) -> Iterator[int]:
        assert data and data[0] == 0
        chord_times = np.round(np.array(data) * self.resolution).astype(int)
        diffs = np.diff(chord_times)
        encoded = []
        for diff in diffs:
            encoded.extend(self.table[diff])
        encoded.append(0xF)
        yield from self.fourbitencoder.encode(encoded)

    def decode(self, data: Iterator[int]) -> list[float]:
        decoded = self.fourbitencoder.decode(data)
        chord_times = [0]

        def decode_bits(data_iter, tree):
            index = data_iter.pop(0)
            if index == 0xF:
                return
            entry = tree[index]
            if isinstance(entry, int):
                return entry
            else:
                return decode_bits(data_iter, entry)
        while decoded:
            diff = decode_bits(decoded, self.tree)
            if diff is None:
                break
            chord_times.append(chord_times[-1] + diff)
        return (np.array(chord_times) / self.resolution).tolist()


class BeatTimesEncoder(BitsEncoder[list[float]]):
    def __init__(self, resolution: float = 44100/1024):
        self.fourbitencoder = FourBitEncoder()
        self.timestamp_encoder = ChordTimesEncoder(resolution)
        self.timestamp_encoder.tree, self.timestamp_encoder.table = get_beat_time_label_codebook(resolution)
        self.resolution = resolution

    def encode(self, data: list[float]) -> Iterator[int]:
        new_data = [0] + data
        yield from self.timestamp_encoder.encode(new_data)

    def decode(self, data: Iterator[int]) -> list[float]:
        new_data = self.timestamp_encoder.decode(data)
        return new_data[1:]


class ChordLabelsEncoder(BitsEncoder[list[int]]):
    def encode(self, data: list[int]) -> Iterator[int]:
        chord_labels = np.array(data, dtype=np.uint8)
        yield from chord_labels
        yield 0xFF

    def decode(self, data: Iterator[int]) -> list[int]:
        chord_labels = []
        # The upper bound comes from 600 seconds * 10.8 chord resolution
        # Just to be safe, we will break if we see more than 6480 labels
        for _ in range(6480):
            label = next(data)
            if label == 0xFF:
                break
            if label >= 170:
                raise ValueError(f"Invalid chord label {label}")
            chord_labels.append(label)
        else:
            raise ValueError("Too many chord labels")
        return chord_labels


class StringEncoder(BitsEncoder[str]):
    def __init__(self, limit: int = 1000):
        self.limit = limit

    def encode(self, data: str) -> Iterator[int]:
        for byte in data.encode('utf-8'):
            yield byte
        yield 0

    def decode(self, data: Iterator[int]) -> str:
        i = 0
        b: list[int] = []
        for _ in range(self.limit):
            byte = next(data)
            i += 1
            if byte == 0:
                break
            b.append(byte)
        else:
            raise ValueError("String possibly too long")

        return bytes(b).decode('utf-8')


class Float32Encoder(BitsEncoder[float]):
    def encode(self, data: float) -> Iterator[int]:
        yield from struct.pack('f', data)

    def decode(self, data: Iterator[int]) -> float:
        b: list[int] = []
        for i in range(4):
            b.append(next(data))
        return struct.unpack('f', bytes(b))[0]


class Int64Encoder(BitsEncoder[int]):
    def encode(self, data: int) -> Iterator[int]:
        assert 0 <= data < 2**64
        yield from struct.pack('Q', data)

    def decode(self, data: Iterator[int]) -> int:
        b: list[int] = []
        for i in range(8):
            b.append(next(data))
        return struct.unpack('Q', bytes(b))[0]


class DatasetEntryEncoder(BitsEncoder[DatasetEntry]):
    """Format (bytes, in order):
    - Chord labels: a list of bytes, each byte is a chord label (0-170)
    - A EOS label (-1)
    - Chord times: The difference array of (t * chord time resolution = 10.8) in a Huffman-like encoding
    - Downbeats: The difference array of (t * beat_times_resolution = 44100/1024) in a Huffman-like encoding
    - Beats: Same as downbeats
    - YouTube ID: Null-terminated unicode string in bytes array format
    - Song Genre: There are only 8 genres but lets use one byte for it
    - Length: Signed 32 bit floating point number
    """

    def __init__(self, chord_time_resolution: float = 10.8, beat_time_resolution: float = 44100/1024):
        self.chord_time_encoder = ChordTimesEncoder(chord_time_resolution)
        self.beat_time_encoder = BeatTimesEncoder(beat_time_resolution)
        self.chord_labels_encoder = ChordLabelsEncoder()
        self.string_encoder = StringEncoder()
        self.float32_encoder = Float32Encoder()
        self.int64_encoder = Int64Encoder()

    def encode(self, data: DatasetEntry) -> Iterator[int]:
        yield from self.chord_labels_encoder.encode(data.chords.features.tolist())
        yield from self.chord_time_encoder.encode(data.chords.times.tolist())
        yield from self.beat_time_encoder.encode(data.downbeats.onsets.tolist())
        yield from self.beat_time_encoder.encode(data.beats.onsets.tolist())
        yield from self.string_encoder.encode(data.url.video_id)
        yield from self.float32_encoder.encode(data.duration)

    def decode(self, data: Iterator[int]) -> DatasetEntry:
        chords = self.chord_labels_encoder.decode(data)
        chord_times = self.chord_time_encoder.decode(data)
        downbeats = self.beat_time_encoder.decode(data)
        beats = self.beat_time_encoder.decode(data)
        youtube_id = self.string_encoder.decode(data)
        length = self.float32_encoder.decode(data)

        entry = create_entry(
            duration=length,
            beats_list=beats,
            downbeats_list=downbeats,
            chord_labels=chords,
            chord_times=chord_times,
            url=get_url(youtube_id),
        )

        return entry


class SongDatasetEncoder(BitsEncoder[dict[YouTubeURL, DatasetEntry]]):
    def __init__(self, chord_time_resolution: float = 10.8, beat_time_resolution: float = 44100/1024, progress_bar: bool = True, old: bool = False):
        """If old is True, the encoder will use the old format which includes the views and genre of the song"""
        self.entry_encoder: BitsEncoder[DatasetEntry] = DatasetEntryEncoder(chord_time_resolution, beat_time_resolution)
        self.int64_encoder = Int64Encoder()
        self.checksum_encoder = Int32Encoder()
        self.progress_bar = progress_bar

    def encode(self, data: dict[YouTubeURL, DatasetEntry]) -> Iterator[int]:
        def inner():
            yield from self.int64_encoder.encode(len(data))
            for entry in data.values():
                yield from self.entry_encoder.encode(entry)
        data_binary = bytes(inner())
        data_compressed = zlib.compress(data_binary, level=9)
        checksum = zlib.adler32(data_compressed)
        yield from self.checksum_encoder.encode(checksum)
        yield from data_compressed

    def decode(self, data: Iterator[int]) -> dict[YouTubeURL, DatasetEntry]:
        checksum = self.checksum_encoder.decode(data)
        data_compressed = bytes(data)
        if checksum != zlib.adler32(data_compressed):
            raise ValueError("Checksum mismatch")
        data_binary = zlib.decompress(data_compressed)
        data_binary = iter(tqdm(data_binary, desc="Decompressing", unit="B", leave=False, disable=not self.progress_bar))
        dataset: dict[YouTubeURL, DatasetEntry] = {}
        length = self.int64_encoder.decode(data_binary)
        for _ in range(length):
            entry = self.entry_encoder.decode(data_binary)
            dataset[entry.url] = entry
        return dataset
