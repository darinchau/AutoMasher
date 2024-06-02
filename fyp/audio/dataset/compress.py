# This module contains code that compresses the SongDataset into a binary format

import numpy as np
import re
import struct
from abc import ABC, abstractmethod
from typing import Iterator, TypeVar, Generic
from .base import DatasetEntry, SongDataset, SongGenre
from ...util.combine import get_url
from .create import create_entry

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

    def decode(self, data: Iterator[int]) -> list[float]:
        beats = []
        while True:
            time = 0
            for i in range(2):
                time |= next(data) << (i * 8)
            if time == 0xFFFF:
                break
            beats.append(time / self.resolution)
        return beats
    
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
    def __init__(self, chord_time_resolution: float = 10.8, beat_time_resolution: float = 44100/1024):
        self.chord_time_encoder = TimeStampEncoder(chord_time_resolution)
        self.beat_time_encoder = TimeStampEncoder(beat_time_resolution)
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
        self.entry_encoder = DatasetEntryEncoder(chord_time_resolution, beat_time_resolution)
        self.int64_encoder = Int64Encoder()

    def encode(self, data: SongDataset) -> Iterator[int]:
        yield from self.int64_encoder.encode(len(data))
        for entry in data:
            yield from self.entry_encoder.encode(entry)

    def decode(self, data: Iterator[int]) -> SongDataset:
        dataset_len = self.int64_encoder.decode(data)
        dataset = SongDataset()
        for _ in range(dataset_len):
            entry = self.entry_encoder.decode(data)
            dataset.add_entry(entry)
        return dataset
