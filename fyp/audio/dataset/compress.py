# This module contains code that compresses the SongDataset into a binary format
# Assumes a couple of things which all the sanity checks are done in the post init method of DatasetEntry

import numpy as np
import re
import struct
from abc import ABC, abstractmethod
from typing import Iterator, TypeVar, Generic
from .base import DatasetEntry, SongDataset, SongGenre
from ...util.combine import get_url
from .create import create_entry
from collections import Counter
import struct
from typing import Iterator
import numpy as np

def build_tree(int_arrays: Iterator[list[int]]):
    counter = Counter()

    for array in int_arrays:
        assert array[0] == 0
        diffs = np.diff(array)
        counter.update(diffs)

    # Reserve 0xF for padding and EOS
    # So we want the 0xF least common elements
    # Since the length of the counter decreases by 14 each iteration
    # To ensure each level has exactly 15 elements, the first merge
    # should operate on len(counter) % 14 elements

    def merge(counter: Counter, n_elems: int):
        least_common = counter.most_common()[-n_elems:]
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
    return tree


T = TypeVar('T')
class BitsEncoder(ABC, Generic[T]):
    @abstractmethod
    def encode(self, data: T) -> Iterator[int]:
        """Encodes the data into a stream of bits. (int between 0 and 255)"""
        pass

    @abstractmethod
    def decode(self, data: Iterator[int]) -> T:
        """Returns the decoded data"""
        pass

class TimeStampEncoder(BitsEncoder[list[float]]):
    def __init__(self, resolution: float):
        self.resolution = resolution

    def encode(self, data: list[float]) -> Iterator[int]:
        timestamps = np.array(data)
        assert np.all(0 <= timestamps) and np.all(timestamps <= 600)
        timestamps =  np.round(timestamps * self.resolution).astype(np.uint16)
        for time in timestamps:
            yield time & 0xFF
            yield (time >> 8) & 0xFF
        yield 0xFF
        yield 0xFF

    def decode(self, data: Iterator[int]) -> tuple[list[float], int]:
        beats = []
        while True:
            time = 0
            for i in range(2):
                time |= next(data) << (i * 8)
            if time == 0xFFFF:
                break
            beats.append(time / self.resolution)
        return beats, len(beats) * 2 + 2
    
class ChordLabelsEncoder(BitsEncoder[list[int]]):
    def encode(self, data: list[int]) -> Iterator[int]:
        chord_labels = np.array(data, dtype=np.uint8)
        yield from chord_labels
        yield 0xFF

    def decode(self, data: Iterator[int]) -> list[int]:
        chord_labels = []
        while True:
            label = next(data)
            if label == 0xFF:
                break
            chord_labels.append(label)
        return chord_labels
    
class GenreEncoder(BitsEncoder[SongGenre]):
    def encode(self, data: SongGenre) -> Iterator[int]:
        yield data.to_int()

    def decode(self, data: Iterator[int]) -> SongGenre:
        genre = SongGenre.from_int(next(data))
        return genre
    
class StringEncoder(BitsEncoder[str]):
    def encode(self, data: str) -> Iterator[int]:
        for byte in data.encode('utf-8'):
            yield byte
        yield 0

    def decode(self, data: Iterator[int]) -> str:
        i = 0
        b: list[int] = []
        while True:
            byte = next(data)
            i += 1
            if byte == 0:
                break
            b.append(byte)

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
        yield from struct.pack('q', data)

    def decode(self, data: Iterator[int]) -> int:
        b: list[int] = []
        for i in range(8):
            b.append(next(data))
        return struct.unpack('q', bytes(b))[0]
    
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

    def encode(self, bits: list[int]) -> Iterator[int]:
        yield from self.i32encoder.encode(len(bits))
        elems = [b for b in bits]
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
    
class HuffmanTableEncoder(BitsEncoder[list[int]]):
    def __init__(self):
        self.encoder = FourBitEncoder()

    def encode(self, tree: tuple) -> Iterator[int]:
        def encode_tree(node: tuple | int):
            if isinstance(node, tuple):
                assert len(node) <= 15
                yield 0xF
                yield len(node) - 1
                for i, child in enumerate(node):
                    yield from encode_tree(child)
            else:
                yield 0xF
                yield 0xF
                for i in range(4):
                    yield node & 0xF
                    node >>= 4
        encoded_tree = list(encode_tree(tree))
        yield from self.encoder.encode(encoded_tree)

    def decode(self, data: Iterator[int]) -> tuple:
        encoded_tree = self.encoder.decode(data)
        def decode_tree(data: Iterator[int]):
            elem = next(data)
            assert elem == 0xF
            elem = next(data)
            if elem == 0xF:
                node = 0
                for i in range(4):
                    node |= next(data) << (i * 4)
                return node
            else:
                children = []
                for _ in range(elem + 1):
                    children.append(decode_tree(data))
                return tuple(children)
        return decode_tree(iter(encoded_tree))

class ChordTimesEncoder(BitsEncoder[list[float]]):
    def __init__(self, resolution: float = 10.8):
        self.four_bit_encoder = FourBitEncoder()
        self._mapping = None
        self.resolution = resolution
    
    def train(self, dataset: SongDataset):
        def iterate_chord_times(dataset: SongDataset) -> Iterator[list[int]]:
            for entry in dataset:
                yield np.round(np.array(entry.chord_times) * self.resolution).astype(int).tolist()
        self._mapping = build_tree(iterate_chord_times(dataset))

    def encode(self, data: list[float]) -> Iterator[int]:
        assert self._mapping is not None, "Encoder not trained"
        chord_times = np.round(np.array(data) * self.resolution).astype(int)
        diffs = np.diff(chord_times)
        mapping = {}
        def get_mapping(node: tuple, path: list[int]):
            if isinstance(node, tuple):
                for i, child in enumerate(node):
                    get_mapping(child, path + [i])
            else:
                mapping[node] = path
        get_mapping(self._mapping, [])
        for diff in diffs:
            yield from mapping[diff]
        yield 0xF

    def decode(self, data: Iterator[int]) -> list[float]:
        assert self._mapping is not None, "Encoder not trained"
        def get_next_element(data: Iterator[int], mapping):
            elem = next(data)
            if elem == 0xF:
                return None
            elem = mapping[elem]
            if isinstance(elem, tuple):
                return get_next_element(data, elem)
            return elem
        chord_times = [0]
        while True:
            next_elem = get_next_element(data, self._mapping)
            if next_elem is None:
                break
            chord_times.append(next_elem)
        return (np.cumsum(chord_times) / self.resolution).tolist()
    
class DatasetEntryEncoder(BitsEncoder[DatasetEntry]):
    """Format (bytes, in order):
    - Chord labels: a list of bytes, each byte is a chord label (0-170)
    - A EOS label (-1)
    - Chord times: a list of 2-byte integers, each representing the time t * chord_times_resolution in seconds
        This list will have the same length as the chord labels list
        The integers are in little-endian format
        EOS marker 0xFFFF
    - Downbeats: a list of 2-byte integers, each representing the time t * beat_times_resolution in seconds
        This makes sense because the maximum length of a song is 600 seconds,
        so the maximum label would be 600 * 44100 / 1024 = 25839 < 2^16
        In fact the highest bit will never be set so we will use FF FF as the EOS marker
        Little-endian format
    - A EOS marker 0xFFFF
    - Beats: A list of 2-byte integers, each representing the time t * beat_times_resolution in seconds
        Same format as downbeats really
    - A EOS marker 0xFFFF
    - YouTube ID: a 11-character string
    - Song Genre: There are only 8 genres but lets use one byte for it
    - Views: unsigned 64 bit integer format
    - Length: Signed 32 bit floating point number
    - Playlist ID: Null-terminated unicode string in bytes array format
    - Title: Null-terminated unicode string in bytes array format
    """
    def __init__(self, chord_times_encoder: BitsEncoder[list[float]], beats_encoder: BitsEncoder[list[float]]):
        self.chord_time_encoder = chord_times_encoder
        self.beat_time_encoder = beats_encoder
        self.chord_labels_encoder = ChordLabelsEncoder()
        self.genre_encoder = GenreEncoder()
        self.string_encoder = StringEncoder()
        self.float32_encoder = Float32Encoder()
        self.int64_encoder = Int64Encoder()

    def encode(self, data: DatasetEntry) -> Iterator[int]:
        playlist_id = data.playlist[len(DatasetEntry.get_playlist_prepend()):]

        yield from self.chord_labels_encoder.encode(data.chords)
        yield from self.chord_time_encoder.encode(data.chord_times)
        yield from self.beat_time_encoder.encode(data.downbeats)
        yield from self.beat_time_encoder.encode(data.beats)
        yield from self.string_encoder.encode(data.url_id)
        yield from self.genre_encoder.encode(data.genre)
        yield from self.int64_encoder.encode(data.views)
        yield from self.float32_encoder.encode(data.length)
        yield from self.string_encoder.encode(playlist_id)
        yield from self.string_encoder.encode(data.audio_name)

    def decode(self, data: Iterator[int]) -> DatasetEntry:
        chords = self.chord_labels_encoder.decode(data)
        chord_times = self.chord_time_encoder.decode(data)
        downbeats = self.beat_time_encoder.decode(data)
        beats = self.beat_time_encoder.decode(data)
        youtube_id = self.string_encoder.decode(data)
        genre = self.genre_encoder.decode(data)
        views = self.int64_encoder.decode(data)
        length = self.float32_encoder.decode(data)
        playlist_id = self.string_encoder.decode(data)
        audio_name = self.string_encoder.decode(data)

        entry = create_entry(
            length=length,
            beats=beats,
            downbeats=downbeats,
            chords=chords,
            chord_times=chord_times,
            genre=genre,
            views=views,
            audio_name=audio_name,
            url=get_url(youtube_id),
            playlist=f"{DatasetEntry.get_playlist_prepend()}{playlist_id}"
        )
        return entry

class SongDatasetEncoder(BitsEncoder[SongDataset]):
    def __init__(self, chord_time_resolution: float = 10.8, beat_time_resolution: float = 44100/1024):
        self.chord_times_encoder = ChordTimesEncoder(chord_time_resolution)
        self.beat_times_encoder = TimeStampEncoder(beat_time_resolution)
        self.tree_encoder = HuffmanTableEncoder()
        self.entry_encoder = DatasetEntryEncoder(self.chord_times_encoder, self.beat_times_encoder)
        self.int64_encoder = Int64Encoder()

    def encode(self, data: SongDataset) -> Iterator[int]:
        self.chord_times_encoder.train(data)
        mapping = self.chord_times_encoder._mapping
        assert mapping is not None
        yield from self.int64_encoder.encode(len(data))
        yield from self.tree_encoder.encode(mapping)
        for entry in data:
            yield from self.entry_encoder.encode(entry)

    def decode(self, data: Iterator[int]) -> SongDataset:
        dataset_len = self.int64_encoder.decode(data)
        tree = self.tree_encoder.decode(data)
        self.chord_times_encoder._mapping = tree
        dataset = SongDataset()
        for _ in range(dataset_len):
            entry = self.entry_encoder.decode(data)
            dataset.add_entry(entry)
        return dataset
