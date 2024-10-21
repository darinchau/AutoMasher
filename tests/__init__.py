# I will try to collect as many test cases here
# Please add more test cases as you see fit
# Please open a PR if you could add more test cases :D
import unittest
import torch
import numpy as np
from fyp import Audio
from fyp.audio.mix.align import get_valid_starting_points
from fyp.app.core import extrapolate_downbeat
from fyp.util.url import get_url
import numpy as np
from math import isclose


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

    def test_downbeat_extrapolation(self):
        x, start = extrapolate_downbeat(
            np.array([107.13, 110.31, 113.40, 116.51, 119.62, 122.74, 125.87, 129.01]),
            70,
            8
        )
        self.assertTrue(isclose(x[0], 69.62143, abs_tol=1e-4))
        self.assertTrue(start > 0)

class TestYouTubeURL(unittest.TestCase):
    def test_youtube_url(self):
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=RDdQw4w9WgXcQ&start_radio=1",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://music.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstleyVEVO",
        ]
        for url in urls:
            yt = get_url(url)
            self.assertEqual(yt.video_id, "dQw4w9WgXcQ", msg=f"Failed at {url}")

class TestAudio(unittest.TestCase):
    def test_init_audio(self):
        # Mono audio
        mono_tensor = torch.randn(1, 1000, dtype=torch.float32)
        mono_audio = Audio(mono_tensor, 44100)
        self.assertTrue(mono_audio.nchannels.value == 1)

        # Stereo audio
        stereo_tensor = torch.randn(2, 1000, dtype=torch.float32)
        stereo_audio = Audio(stereo_tensor, 44100)
        self.assertTrue(stereo_audio.nchannels.value == 2)

        # Invalid audio
        with self.assertRaises(AssertionError):
            invalid_tensor = torch.randn(3, 1000, dtype=torch.float32)
            invalid_audio = Audio(invalid_tensor, 44100)

        # Invalid audio - wrong dtype
        with self.assertRaises(AssertionError):
            invalid_tensor = torch.randn(2, 1000, dtype=torch.float64)
            invalid_audio = Audio(invalid_tensor, 44100)

    def test_slice_seconds(self):
        tensor = torch.randn(2, 44100, dtype=torch.float32)
        audio = Audio(tensor, 44100)
        sliced = audio.slice_seconds(0, 0.5)
        self.assertTrue(sliced.duration == 0.5)
        self.assertTrue(sliced.nchannels.value == 2)
        self.assertTrue(sliced.sample_rate == 44100)
        self.assertTrue(sliced.nframes == 22050)
        self.assertTrue(torch.equal(sliced.data, tensor[:, :22050]))

    def test_pad(self):
        tensor = torch.randn(2, 1000, dtype=torch.float32)
        audio = Audio(tensor, 44100)
        padded = audio.pad(2000, front=False)
        self.assertTrue(padded.nchannels.value == 2)
        self.assertTrue(padded.sample_rate == 44100)
        self.assertTrue(padded.nframes == 2000)
        self.assertTrue(torch.equal(padded.data[:, :1000], tensor))

        padded_front = audio.pad(2000, front=True)
        self.assertTrue(padded_front.nchannels.value == 2)
        self.assertTrue(padded_front.sample_rate == 44100)
        self.assertTrue(padded_front.nframes == 2000)
        self.assertTrue(torch.equal(padded_front.data[:, -1000:], tensor))

        padded_short_front = audio.pad(500, front=True)
        self.assertTrue(padded_short_front.nframes == 500)
        self.assertTrue(torch.equal(padded_short_front.data, tensor[:, -500:]))

        padded_short_back = audio.pad(500, front=False)
        self.assertTrue(padded_short_back.nframes == 500)
        self.assertTrue(torch.equal(padded_short_back.data, tensor[:, :500]))

    def test_change_speed(self):
        tensor = torch.randn(2, 44100, dtype=torch.float32)
        audio = Audio(tensor, 44100)
        faster = audio.change_speed(2)
        self.assertTrue(faster.duration == 0.5)
        self.assertTrue(faster.nchannels.value == 2)
        self.assertTrue(faster.sample_rate == 44100)
        self.assertTrue(faster.nframes == 22050)

        slower = audio.change_speed(0.5)
        self.assertTrue(slower.duration == 2)
        self.assertTrue(slower.nchannels.value == 2)
        self.assertTrue(slower.sample_rate == 44100)
        self.assertTrue(slower.nframes == 88200)

    def test_resample(self):
        tensor = torch.randn(2, 44100, dtype=torch.float32)
        audio = Audio(tensor, 44100)
        resampled = audio.resample(22050)
        self.assertTrue(resampled.duration == audio.duration)
        self.assertTrue(resampled.nchannels.value == 2)
        self.assertTrue(resampled.sample_rate == 22050)
        self.assertTrue(resampled.nframes == 22050)

    def test_to_nchannels(self):
        tensor = torch.randn(2, 1000, dtype=torch.float32)
        audio = Audio(tensor, 44100)
        mono = audio.to_nchannels(1)
        self.assertTrue(mono.nchannels.value == 1)
        self.assertTrue(mono.sample_rate == 44100)
        self.assertTrue(mono.nframes == 1000)

        stereo = audio.to_nchannels(2)
        self.assertTrue(stereo.nchannels.value == 2)
        self.assertTrue(stereo.sample_rate == 44100)
        self.assertTrue(stereo.nframes == 1000)

    def test_slice_frames(self):
        tensor = torch.randn(2, 1000, dtype=torch.float32)
        audio = Audio(tensor, 44100)
        sliced = audio.slice_frames(100, 200)
        self.assertTrue(sliced.nframes == 100)
        self.assertTrue(torch.equal(sliced.data, tensor[:, 100:200]))

        sliced2 = audio.slice_frames(100, -1)
        self.assertTrue(sliced2.nframes == 900)
        self.assertTrue(torch.equal(sliced2.data, tensor[:, 100:]))

        sliced3 = audio.slice_frames(0, 100)
        self.assertTrue(sliced3.nframes == 100)
        self.assertTrue(torch.equal(sliced3.data, tensor[:, :100]))

        # Invalid slice
        with self.assertRaises(AssertionError):
            audio.slice_frames(100, 50)

        # Invalid slice - OOB
        with self.assertRaises(AssertionError):
            audio.slice_frames(100, 1001)

        # Valid slice - till the end
        sliced4 = audio.slice_frames(100, 1000)
        self.assertTrue(sliced4.nframes == 900)
        self.assertTrue(torch.equal(sliced4.data, tensor[:, 100:]))

    def test_join(self):
        tensor1 = torch.randn(2, 1000, dtype=torch.float32)
        tensor2 = torch.randn(2, 1000, dtype=torch.float32)
        audio1 = Audio(tensor1, 44100)
        audio2 = Audio(tensor2, 44100)
        joined = audio1.join(audio2)
        self.assertTrue(joined.nframes == 2000)
        self.assertTrue(torch.equal(joined.data, torch.cat([tensor1, tensor2], dim=1)))
