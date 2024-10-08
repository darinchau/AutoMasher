# I will try to collect as many test cases here
# Please add more test cases as you see fit
# Please open a PR if you could add more test cases :D
import unittest
from fyp.audio.search.align import get_valid_starting_points
import numpy as np

class AutoMasherTests(unittest.TestCase):
    def test_get_valid_starting_points(self):
        md = [1., 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
        downbeats = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.float64)
        beats = np.array([0, 0.25, 0.5, 0.75, 1,
                    1.25, 1.5, 1.75, 2,
                    2.25, 2.5, 2.75, 3,
                    3.25, 3.5, 3.75, 4,
                    4.25, 4.5, 5,
                    5.25, 5.5, 5.75, 6,
                    6.25, 6.5, 6.75, 7,
                    7.25, 7.5, 7.75, 8,
                    8.25, 8.5, 8.75, 9,
                    9.25, 9.5, 9.75, 10,
                    10.25, 10.5, 10.75, 11,
                    11.25, 11.5, 11.75, 12], dtype=np.float64)

        assert len(md) == len(downbeats)
        result = get_valid_starting_points(md, downbeats, beats, 4, 0.99)
        assert result == [0, 5, 6]
