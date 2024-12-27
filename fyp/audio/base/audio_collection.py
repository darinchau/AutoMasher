from __future__ import annotations
import os
from .audio import Audio
from typing import Callable
import zipfile
import tempfile
import torch

class DemucsCollection:
    def __init__(self, bass: Audio, drums: Audio, other: Audio, vocals: Audio):
        assert bass.nframes == drums.nframes == other.nframes == vocals.nframes, "All audios must have the same number of frames"
        assert bass.sample_rate == drums.sample_rate == other.sample_rate == vocals.sample_rate, "All audios must have the same sample rate"
        self._bass = bass
        self._drums = drums
        self._other = other
        self._vocals = vocals

    @property
    def bass(self):
        return self._bass

    @property
    def drums(self):
        return self._drums

    @property
    def other(self):
        return self._other

    @property
    def vocals(self):
        return self._vocals

    def get_duration(self) -> float:
        return self._bass.duration

    def change_speed(self, speed: float) -> DemucsCollection:
        return DemucsCollection(
            bass=self._bass.change_speed(speed),
            drums=self._drums.change_speed(speed),
            other=self._other.change_speed(speed),
            vocals=self._vocals.change_speed(speed)
        )

    def slice_seconds(self, start: float, end: float) -> DemucsCollection:
        return DemucsCollection(
            bass=self._bass.slice_seconds(start, end),
            drums=self._drums.slice_seconds(start, end),
            other=self._other.slice_seconds(start, end),
            vocals=self._vocals.slice_seconds(start, end)
        )

    def join(self, other: DemucsCollection) -> DemucsCollection:
        return DemucsCollection(
            bass=self._bass.join(other.bass),
            drums=self._drums.join(other.drums),
            other=self._other.join(other.other),
            vocals=self._vocals.join(other.vocals)
        )

    @property
    def sample_rate(self) -> int:
        return int(self._bass.sample_rate)

    @property
    def nframes(self) -> int:
        return self._bass.nframes

    def keys(self):
        return ['bass', 'drums', 'other', 'vocals']

    def items(self):
        return [('bass', self._bass), ('drums', self._drums), ('other', self._other), ('vocals', self._vocals)]

    def map(self, func: Callable[[Audio], Audio]):
        return DemucsCollection(
            bass=func(self._bass),
            drums=func(self._drums),
            other=func(self._other),
            vocals=func(self._vocals)
        )

    def save(self, path: str, inner_format: str = "wav"):
        """Save the collection to a zip file."""
        assert inner_format in ["wav", "mp3"], "Invalid inner format"
        with zipfile.ZipFile(path, 'w') as z:
            with tempfile.TemporaryDirectory() as tmpdirname:
                for k, v in self.items():
                    v.save(os.path.join(tmpdirname, f"{k}.{inner_format}"))
                for root, dirs, files in os.walk(tmpdirname):
                    for file in files:
                        z.write(os.path.join(root, file), file)
            z.writestr("format.txt", inner_format)

    @staticmethod
    def load(path: str) -> DemucsCollection:
        """Load a collection from a zip file."""
        with zipfile.ZipFile(path, 'r') as z:
            with tempfile.TemporaryDirectory() as tmpdirname:
                z.extractall(tmpdirname)
                inner_format = z.read("format.txt").decode("utf-8")
                return DemucsCollection(
                    bass=Audio.load(os.path.join(tmpdirname, "bass." + inner_format)),
                    drums=Audio.load(os.path.join(tmpdirname, "drums." + inner_format)),
                    other=Audio.load(os.path.join(tmpdirname, "other." + inner_format)),
                    vocals=Audio.load(os.path.join(tmpdirname, "vocals." + inner_format))
                )

    @staticmethod
    def join_all(segments: list[DemucsCollection]) -> DemucsCollection:
        """Join all the segments together"""
        if len(segments) == 0:
            raise ValueError("Cannot join an empty list of segments")
        if len(segments) == 1:
            return segments[0]
        result = segments[0]
        duration = result.get_duration()
        for segment in segments[1:]:
            result = result.join(segment)
            duration += segment.get_duration()
        return result

    def align_from_boundaries(self, factors: list[float], boundaries: list[float]):
        """Align the segments to the boundaries"""
        assert len(factors) == len(boundaries)
        result = self
        boundaries = [0] + boundaries
        segments = []
        for i in range(len(factors)):
            segment = result.slice_seconds(boundaries[i], boundaries[i + 1])
            segment = segment.change_speed(factors[i])
            segments.append(segment)
        result = self.join_all(segments)
        return result

    @property
    def data(self):
        """Gets the underlying data of the collection in (4, channels, nframes) format, in VDIB format."""
        return torch.stack([self.vocals.data, self.drums.data, self.other.data, self.bass.data])

class HPSSCollection:
    def __init__(self, harmonic: Audio | None, percussive: Audio | None):
        assert not harmonic or not percussive or harmonic.nframes == percussive.nframes, "All audios must have the same number of frames"
        assert not harmonic or not percussive or harmonic.sample_rate == percussive.sample_rate, "All audios must have the same sample rate"
        self._harmonic = harmonic
        self._percussive = percussive

    @property
    def harmonic(self):
        return self._harmonic

    @property
    def percussive(self):
        return self._percussive

    def get_duration(self) -> float:
        return self._harmonic.duration if self._harmonic else self._percussive.duration if self._percussive else 0

    def change_speed(self, speed: float) -> HPSSCollection:
        return HPSSCollection(
            harmonic=self._harmonic.change_speed(speed) if self._harmonic else None,
            percussive=self._percussive.change_speed(speed) if self._percussive else None
        )

    def slice_seconds(self, start: float, end: float) -> HPSSCollection:
        return HPSSCollection(
            harmonic=self._harmonic.slice_seconds(start, end) if self._harmonic else None,
            percussive=self._percussive.slice_seconds(start, end) if self._percussive else None
        )

    def join(self, other: HPSSCollection) -> HPSSCollection:
        return HPSSCollection(
            harmonic=self._harmonic.join(other._harmonic) if self._harmonic and other._harmonic else None,
            percussive=self._percussive.join(other._percussive) if self._percussive and other._percussive else None
        )

    @property
    def sample_rate(self) -> int:
        sr = self._harmonic.sample_rate if self._harmonic else self._percussive.sample_rate if self._percussive else 0
        return int(sr)

    @property
    def nframes(self) -> int:
        return self._harmonic.nframes if self._harmonic else self._percussive.nframes if self._percussive else 0

    def map(self, func: Callable[[Audio], Audio]):
        return HPSSCollection(
            harmonic=func(self._harmonic) if self._harmonic else None,
            percussive=func(self._percussive) if self._percussive else None
        )
