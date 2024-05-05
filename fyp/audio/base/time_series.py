# An abstract class that represents a time series (see final report)

from abc import ABCMeta, abstractmethod as abstract
from typing import TypeVar

T = TypeVar('T', bound='TimeSeries')
class TimeSeries(metaclass=ABCMeta):
    """A mixin for anything that has an element of time to it. Mainly used in Audio, Chord and Beat analysis results"""
    @abstract
    def slice_seconds(self: T, start: float, end: float) -> T:
        """Slice whatever we have between start and end seconds. After the slice, start becomes t=0"""
        pass

    @abstract
    def change_speed(self: T, speed: float) -> T:
        pass

    @abstract
    def join(self: T, other: T) -> T:
        pass

    @abstract
    def get_duration(self) -> float:
        pass

    @staticmethod
    def join_all(segments: list[T]) -> T:
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

    def slice_and_change_speed(self: T, start: float, end: float, speed: float) -> T:
        """Change in place the segment between start and end seconds to the new speed. The rest of the segment is left unchanged."""
        assert start < end and start >= 0
        segments: list[T] = []
        if start > 0:
            starting = self.slice_seconds(0, start)
            segments.append(starting)

        mid = self.slice_seconds(start, end)
        mid = mid.change_speed(speed)
        segments.append(mid)
        if self.get_duration() > end:
            ending = self.slice_seconds(end, self.get_duration())
            segments.append(ending)

        return self.join_all(segments)
    
    def align_from_boundaries(self: T, factors: list[float], boundaries: list[float]) -> T:
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
