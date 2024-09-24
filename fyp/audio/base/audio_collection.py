from __future__ import annotations
from .audio import Audio
from .time_series import TimeSeries
from typing import Callable

class DemucsCollection(TimeSeries):
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
        return self._bass.get_duration()

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
        return self._bass.sample_rate

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

class HPSSCollection(TimeSeries):
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
        return self._harmonic.get_duration() if self._harmonic else self._percussive.get_duration() if self._percussive else 0

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
        return self._harmonic.sample_rate if self._harmonic else self._percussive.sample_rate if self._percussive else 0

    @property
    def nframes(self) -> int:
        return self._harmonic.nframes if self._harmonic else self._percussive.nframes if self._percussive else 0

    def map(self, func: Callable[[Audio], Audio]):
        return HPSSCollection(
            harmonic=func(self._harmonic) if self._harmonic else None,
            percussive=func(self._percussive) if self._percussive else None
        )
