# This module contains code that compresses the SongDataset into a binary format

import numpy as np
import re
import struct
from ...util.combine import get_video_id
from .base import DatasetEntry, SongDataset, SongGenre

_IS_YOUTUBE_ID = re.compile(r'^[A-Za-z0-9_-]{11}$')

def encode_beats(beats: list[float], resolution: float):
    beats_ = np.array(beats)
    assert np.all(0 <= beats_) and np.all(beats_ <= 600)
    beats_ =  np.round(beats_ * resolution).astype(np.uint16)
    for time in beats_:
        yield time & 0xFF
        yield (time >> 8) & 0xFF
    yield 0xFF
    yield 0xFF

def str_to_bytes(st: str):
    for byte in st.encode('utf-8'):
        yield byte
    yield 0

def float_to_bytes(f):
    bytes_array = struct.pack('f', f)
    for byte in bytes_array:
        yield byte

def encode_data_entry_generate_bytes(data_entry: DatasetEntry, *, chord_times_resolution: float = 10.8, beat_times_resolution: float = 44100/1024):
    """Format (bytes, in order):
    - Chord labels: a list of bytes, each byte is a chord label (0-170)
    - A EOS label (-1)
    - Chord times: a list of 2-byte integers, each representing the time t * chord_times_resolution in seconds
        This list will have the same length as the chord labels list
        The integers are in little-endian format
    - Downbeats: a list of 2-byte integers, each representing the time t * beat_times_resolution in seconds
        This makes sense because the maximum length of a song is 600 seconds,
        so the maximum label would be 600 * 44100 / 1024 = 25839 < 2^16
        In fact the highest bit will never be set so we will use FF FF as the EOS marker
        Little-endian format
    - A EOS marker (FF FF)
    - Beats: A list of 2-byte integers, each representing the time t * beat_times_resolution in seconds
        Same format as downbeats really
    - A EOS marker (FF FF)
    - YouTube ID: a 11-character string
    - Song Genre: There are only 8 genres but lets use one byte for it
    - Views: unsigned 64 bit integer format
    - Length: Signed 32 bit floating point number
    - Playlist: Null-terminated unicode string in bytes array format
    - Title: Null-terminated unicode string in bytes array format
    """
    # Sanity check about beat times resolution and chord times resolution
    assert chord_times_resolution > 0
    assert chord_times_resolution * 600 < 2**16
    assert beat_times_resolution > 0
    assert beat_times_resolution * 600 < 2**16

    # Encode chord labels
    chord_labels = np.array(data_entry.chords, dtype=np.uint8)
    assert np.all(0 <= chord_labels) and np.all(chord_labels <= 170)
    yield from chord_labels
    yield 0xFF

    # Encode chord times
    chord_times = np.array(data_entry.chord_times)
    assert np.all(0 <= chord_times) and np.all(chord_times <= 600)
    chord_times = np.round(chord_times * chord_times_resolution).astype(np.uint16)
    for time in chord_times:
        yield time & 0xFF
        yield (time >> 8) & 0xFF

    yield from encode_beats(data_entry.downbeats, beat_times_resolution)
    yield from encode_beats(data_entry.beats, beat_times_resolution)

    # Encode YouTube ID and song genre
    youtube_id = get_video_id(data_entry.url)
    assert _IS_YOUTUBE_ID.match(youtube_id)
    for byte in youtube_id.encode('utf-8'):
        yield byte
    
    genre = data_entry.genre.to_int()
    assert 0 <= genre <= 255
    yield genre

    # Encode views
    views = data_entry.views
    assert 0 <= views < 2**64
    for i in range(8):
        yield (views >> (i * 8)) & 0xFF

    # Encode length
    yield from float_to_bytes(data_entry.length)

    # Only save the playlist ID
    # Use a regex to extract the playlist ID
    check_str = "https://www.youtube.com/playlist?list="
    assert data_entry.playlist.startswith(check_str), f"Playlist URL is not a valid YouTube playlist URL: {data_entry.playlist}"
    playlist_id = data_entry.playlist[len(check_str):]
    yield from str_to_bytes(playlist_id)

    # Encode title
    yield from str_to_bytes(data_entry.audio_name)

def encode_song_dataset(dataset: SongDataset):
    def _encoder():
        for entry in dataset:
            yield from encode_data_entry_generate_bytes(entry)
    return bytes(_encoder())

def decode_beats(beats: bytes, resolution: float):
    i = 0
    while i < len(beats):
        time = beats[i] + (beats[i + 1] << 8)
        i += 2
        if time == 0xFFFF:
            break
        yield time / resolution

def decode_data_entry_generate_bytes(data_entry: bytes, *, chord_times_resolution: float = 10.8, beat_times_resolution: float = 44100/1024):
    i = 0
    # Decode chord labels
    chord_labels = []
    while data_entry[i] != 0xFF:
        chord_labels.append(data_entry[i])
        i += 1
    i += 1

    # Decode chord times
    chord_times = []
    while i < len(data_entry):
        time = data_entry[i] + (data_entry[i + 1] << 8)
        i += 2
        if time == 0xFFFF:
            break
        chord_times.append(time / chord_times_resolution)

    downbeats = list(decode_beats(data_entry[i:], beat_times_resolution))
    beats = list(decode_beats(data_entry[i:], beat_times_resolution))
    return chord_labels, chord_times, downbeats, beats

def decode_song_dataset(data: bytes):
    i = 0
    dataset = SongDataset()
    while i < len(data):
        chord_labels, chord_times, downbeats, beats = decode_data_entry_generate_bytes(data[i:])
        i += len(chord_labels) + len(chord_times) * 2 + len(downbeats) * 2 + len(beats) * 2 + 5 # 5 is the number of EOS markers
        youtube_id = data[i:i + 11].decode('utf-8')
        i += 11
        genre = data[i]
        i += 1
        views = 0
        for j in range(8):
            views |= data[i + j] << (j * 8)
        i += 8
        length = struct.unpack('f', data[i:i + 4])[0]
        i += 4
        playlist_id = []
        while data[i] != 0:
            playlist_id.append(data[i])
            i += 1
        i += 1
        title = []
        while data[i] != 0:
            title.append(data[i])
            i += 1
        i += 1
        dataset[youtube_id] = DatasetEntry(
            chords=chord_labels,
            chord_times=chord_times,
            downbeats=downbeats,
            beats=beats,
            genre=SongGenre.from_int(genre),
            audio_name=title.decode('utf-8'),
            url=f"https://www.youtube.com/watch?v={youtube_id}",
            playlist=f"https://www.youtube.com/playlist?list={playlist_id.decode('utf-8')}",
            views=views,
            length=length,
            normalized_chord_times=[],
            music_duration=[]
        )
    return dataset
