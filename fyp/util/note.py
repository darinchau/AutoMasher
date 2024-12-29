# Contains useful utilities for manipulating notes
# This place contains a few getters:
# get_idx2voca_chord() -> list[str]: Gets the list of all 170 chord names
# get_inv_voca_map() -> dict[str, int]: Gets a mapping from chord names to indices. The inverse of get_idx2voca_chord()
# get_pitch_names() -> list[str]: Gets all the pitch names in a 12-tone equal temperament system
# get_keys() -> list[str]: Gets all 24 major and minor keys
# get_chord_notes() -> dict[str, frozenset[str]]: Gets a dictionary of chord notes
# get_chord_note_inv() -> dict[frozenset[str], str]: Gets a dictionary of chord notes, inverse of get_chord_notes()
# get_chord_quality(chord: str) -> tuple[str, str]: Gets the quality of a chord. Returns (note, quality) string tuple
# small_voca_to_chord(x: int) -> str: Gets the large voca index from the small voca index
# small_voca_to_large_voca(x: int) -> int: Gets the large voca index from the small voca index
# large_voca_to_small_voca_map() -> dict[int, int]: Gets the mapping from large voca index to small voca index (works for only small voca chords)
# simplify_chord(x: int) -> int: Simplify a large voca chord to a small voca chord (works for each large voca chord)

from typing import Any
from functools import lru_cache

@lru_cache(maxsize=1)
def _get_quality_list():
    return ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']

@lru_cache(maxsize=1)
def get_idx2voca_chord() -> list[str]:
    """A more comprehensive chord list"""
    root_list = get_pitch_names()
    quality_list = _get_quality_list()
    idx2voca_chord = [""] * 170
    idx2voca_chord[169] = 'No chord'
    idx2voca_chord[168] = 'Unknown'
    for i in range(168):
        root = i // 14
        root = root_list[root]
        quality = i % 14
        quality = quality_list[quality]
        if i % 14 != 1:
            chord = root + ':' + quality
        else:
            chord = root
        idx2voca_chord[i] = chord

    assert idx2voca_chord[-1] == "No chord"
    return idx2voca_chord

@lru_cache(maxsize=1)
def get_inv_voca_map() -> dict[str, int]:
    """Get a mapping from chord names to indices. The inverse of get_idx2voca_chord()"""
    idx2voca_chord = get_idx2voca_chord()
    inv_voca_map = {v: i for i, v in enumerate(idx2voca_chord)}
    return inv_voca_map

@lru_cache(maxsize=25)
def small_voca_to_chord(x: int) -> str:
    """Get the large voca index from the small voca index"""
    if not (0 <= x < 25):
        return "Unknown"
    if x == 24:
        return "No chord"

    minmaj = x % 2
    root = x // 2

    return idx_to_notes(root) + ('' if minmaj == 0 else ':min')

@lru_cache(maxsize=25)
def small_voca_to_large_voca(x: int) -> int:
    """Get the large voca index from the small voca index"""
    if not (0 <= x < 25):
        return 168
    if x == 24:
        return 169

    minmaj = x % 2
    root = x // 2

    return root * 14 + (1 - minmaj)

@lru_cache(maxsize=1)
def large_voca_to_small_voca_map() -> dict[int, int]:
    """Get the mapping from large voca index to small voca index"""
    return {small_voca_to_large_voca(i): i for i in range(25)}

@lru_cache(maxsize=170)
def simplify_chord(x: int) -> int:
    """Simplify a large voca chord to a small voca chord"""
    mapping = ['min', 'maj', 'min', 'maj', 'min', 'maj', 'min', 'min', 'maj', 'maj', 'min', 'min', 'maj', 'maj']
    if x in [168, 169]:
        return 24
    root = x // 14
    quality = x % 14
    return root * 2 + (1 if mapping[quality] == 'min' else 0)

def notes_to_idx(note: str):
    """Return the index given the note name. The index is the number of semitones from C."""
    mapping = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }
    return mapping[note]

def idx_to_notes(idx: int):
    """Return the note name given the index. The index is the number of semitones from C."""
    mapping = {
        0: "C",
        1: "C#",
        2: "D",
        3: "D#",
        4: "E",
        5: "F",
        6: "F#",
        7: "G",
        8: "G#",
        9: "A",
        10: "A#",
        11: "B",
    }
    return mapping[idx % 12]

def move_semitone(note: str, semitone: int):
    """Move a note by a number of semitones. Returns the new note. If the note is a black key, then the note is always sharp instead of flat."""
    return idx_to_notes(notes_to_idx(note) + semitone)

def get_pitch_names():
    """All the pitch names in a 12-tone equal temperament system."""
    return ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

@lru_cache(maxsize=1)
def get_chord_notes() -> dict[str, frozenset[str]]:
    """Get a dictionary of chord notes. The keys are the voca chords and the values are the notes in the chord."""
    base_chord_notes = {
        "C:min": ["C", "Eb", "G"],
        "C": ["C", "E", "G"],
        "C:dim": ["C", "Eb", "Gb"],
        "C:aug": ["C", "E", "G#"],
        "C:min6": ["C", "Eb", "G", "A"],
        "C:maj6": ["C", "E", "G", "A"],
        "C:min7": ["C", "Eb", "G", "Bb"],
        "C:minmaj7": ["C", "Eb", "G", "B"],
        "C:maj7": ["C", "E", "G", "B"],
        "C:7": ["C", "E", "G", "Bb"],
        "C:dim7": ["C", "Eb", "Gb", "A"],
        "C:hdim7": ["C", "Eb", "Gb", "Bb"],
        "C:sus2": ["C", "D", "G"],
        "C:sus4": ["C", "F", "G"],
    }

    chord_notes = {}
    for i in range(12):
        note = idx_to_notes(i)
        for chord, notes in base_chord_notes.items():
            new_chord_name = note + ":" + chord[2:] if ":" in chord else note
            new_notes = frozenset(idx_to_notes(notes_to_idx(n) + i) for n in notes)
            chord_notes[new_chord_name] = new_notes

    chord_notes["Unknown"] = frozenset()
    chord_notes["No chord"] = frozenset()

    # Sanity check
    assert set(chord_notes.keys()) == set(get_idx2voca_chord())
    for _, value in chord_notes.items():
        for note in value:
            assert note in get_pitch_names()
    return chord_notes

@lru_cache(maxsize=1)
def get_chord_note_inv() -> dict[frozenset[str], str]:
    """Get a dictionary of chord notes. The inverse of get_chord_notes()"""
    chord_notes_map = get_chord_notes()
    chord_notes_inv = {v: k for k, v in chord_notes_map.items()}
    return chord_notes_inv

@lru_cache(maxsize=None)
def get_chord_quality(chord: str) -> tuple[str, str]:
    """Get the quality of a chord. Returns (note, quality) string tuple"""
    assert chord in get_chord_notes(), f"{chord} not a recognised chord"
    if chord in ["No chord", "Unknown"]:
        return "", chord

    if ":" in chord:
        note, quality = chord.split(":")
        return note, quality

    return chord, "maj"

@lru_cache(maxsize=None)
def transpose_chord(chord: str, semitone: int) -> str:
    """Transpose a chord by a number of semitones. Returns the new chord.
    The chord is in the format of "root:quality" or "root".
    If the chord is "No chord" or "Unknown", it will return the same chord.

    The chord does not have to be one of the voca chords. It can be like Eb:7
    """
    if chord in ["No chord", "Unknown"]:
        return chord

    if ":" in chord:
        root, quality = chord.split(":")
        transposed = f"{move_semitone(root, semitone)}:{quality}"
        if transposed not in get_idx2voca_chord():
            raise ValueError(f"Invalid chord: {chord} (Transposed: {transposed})")
        return transposed

    return move_semitone(chord, semitone)
