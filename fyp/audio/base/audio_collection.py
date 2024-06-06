from __future__ import annotations
from .audio import Audio
from .time_series import TimeSeries
from typing import Callable

class AudioCollection(TimeSeries, dict[str, Audio]):
    def __new__(cls, *args, **kwargs):
        dur = -1
        sr = -1
        for k, v in kwargs.items():
            if dur == -1 and isinstance(v, Audio):
                dur = v.get_duration()
                sr = v.sample_rate
            elif isinstance(v, Audio):
                assert v.get_duration() == dur, f"Duration mismatch: {v.get_duration()} != {dur}"
                assert v.sample_rate == sr, f"Sample rate mismatch: {v.sample_rate} != {sr}"
            else:
                raise ValueError(f"Expected Audio but found {type(v)}")
        return super().__new__(cls, *args, **kwargs)

    def slice_seconds(self, start: float, end: float) -> AudioCollection:
        """Slice whatever we have between start and end seconds. After the slice, start becomes t=0"""
        return self.map(lambda x: x.slice_seconds(start, end))

    def change_speed(self, speed: float) -> AudioCollection:
        return self.map(lambda x: x.change_speed(speed))

    def join(self, other: AudioCollection) -> AudioCollection:
        assert set(self.keys()) == set(other.keys()), "The keys of the two audio collections must be the same"
        return AudioCollection(**{k: self[k].join(other[k]) for k in self.keys()})

    def get_duration(self) -> float:
        if len(self) == 0:
            raise NotImplementedError("Cannot get the duration of an empty audio collection")
        return list(self.values())[0].get_duration()

    def __setitem__(self, __key: str, __value: Audio) -> None:
        if len(self) > 0:
            # assert __value.get_duration() == self.get_duration(), f"Duration mismatch: {__value.get_duration()} != {self.get_duration()}"
            if __value.get_duration() != self.get_duration():
                __value = __value.pad(self.nframes)
            assert __value.sample_rate == self.sample_rate, f"Sample rate mismatch: {__value.sample_rate} != {self.sample_rate}"
        return super().__setitem__(__key, __value)

    @property
    def sample_rate(self):
        if len(self) == 0:
            raise NotImplementedError("Cannot get the sample rate of an empty audio collection")
        return list(self.values())[0].sample_rate

    @property
    def nframes(self):
        if len(self) == 0:
            raise NotImplementedError("Cannot get the nframes of an empty audio collection")
        return list(self.values())[0].nframes

    def map(self, func: Callable[[Audio], Audio]):
        new_dict = {}
        dur, sr = -1, -1
        for k, v in self.items():
            new_audio = func(v)
            assert isinstance(new_audio, Audio), f"Expected Audio but found {type(new_audio)}"
            if dur == -1:
                dur = new_audio.get_duration()
                sr = new_audio.sample_rate
            else:
                assert new_audio.get_duration() == dur, f"Duration mismatch: {new_audio.get_duration()} != {dur}"
                assert new_audio.sample_rate == sr, f"Sample rate mismatch: {new_audio.sample_rate} != {sr}"
            new_dict[k] = new_audio
        return AudioCollection(**new_dict)
