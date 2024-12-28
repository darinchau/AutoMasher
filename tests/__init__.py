# These test cases are the unsorted ones
# Please add more test cases as you see fit
# Please open a PR if you could add more test cases :D
import unittest
import torch
import numpy as np
from fyp import Audio
from fyp.audio.mix.align import get_valid_starting_points
from fyp.app.core import extrapolate_downbeat
from fyp.audio.analysis import OnsetFeatures
from fyp.util.url import get_url
import numpy as np
from math import isclose

class AutoMasherTests(unittest.TestCase):
    def test_get_valid_starting_points(self):
        music_duration = np.array([1., 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=np.float64)
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

        assert len(music_duration) == len(downbeats)
        result = get_valid_starting_points(music_duration, downbeats, beats, 4, 0.99)
        assert np.all(result == np.array([0, 5, 6]))

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

    def test_youtube_url_fail(self):
        urls = [
            "https://www.youtube.com/",
            "dQw4w9WgXc",
            "my-invalid-url",
            "C:/Users/username/Music/My%20Music/My%20Playlist/My%20Song.mp3",
            "https://www.youtube.com/@RickAstleyYT",
            "https://www.youtube.com/playlist?list=PLlaN88a7y2_plecYoJxvRFTLHVbIVAOoc",
        ]
        for url in urls:
            try:
                yt = get_url(url)
                self.fail(f"Should have failed at {url}")
            except Exception as e:
                pass

# Tests the audio class specifically
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

class TestOnsetFeatures(unittest.TestCase):
    def test_valid_initialization(self):
        # Test case 1: No onsets
        features = OnsetFeatures(duration=120.0, onsets=np.array([], dtype=np.float64))
        self.assertEqual(features.duration, 120.0)
        self.assertEqual(len(features.onsets), 0)

        # Test case 2: Multiple sorted onsets
        onsets = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        features = OnsetFeatures(duration=50.0, onsets=onsets)
        self.assertTrue(np.array_equal(features.onsets, onsets))
        self.assertEqual(features.duration, 50.0)

    def test_invalid_initialization(self):
        # Test case 1: Duration less than last onset
        with self.assertRaises(AssertionError):
            OnsetFeatures(duration=25.0, onsets=np.array([10, 20, 30, 40], dtype=np.float64))

        # Test case 2: Unsorted onsets
        with self.assertRaises(AssertionError):
            OnsetFeatures(duration=60.0, onsets=np.array([10, 30, 20, 40], dtype=np.float64))

        # Test case 3: Incorrect dtype
        with self.assertRaises(AssertionError):
            OnsetFeatures(duration=60.0, onsets=np.array([10, 20, 30, 40], dtype=np.int32)) #type: ignore

        # Test case 4: Not a numpy array
        with self.assertRaises(AssertionError):
            OnsetFeatures(duration=60.0, onsets=[10.0, 20.0, 30.0, 40.0]) #type: ignore

    def test_immutability_of_onsets(self):
        onsets = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        features = OnsetFeatures(duration=50.0, onsets=onsets)
        with self.assertRaises(ValueError):
            features.onsets[0] = 15.0

    def test_slice_on_exact_onsets(self):
        # Slicing exactly at onset times to check boundary inclusion
        features = OnsetFeatures(duration=100.0, onsets=np.array([0, 30, 60, 90], dtype=np.float64))
        sliced_features = features.slice_seconds(30, 90)
        self.assertEqual(sliced_features.duration, 60.0)
        self.assertTrue(np.array_equal(sliced_features.onsets, np.array([0, 30], dtype=np.float64)))

    def test_slice_beyond_last_onset(self):
        # Slicing exactly at onset times to check boundary inclusion
        features = OnsetFeatures(duration=100.0, onsets=np.array([0, 30, 60, 90], dtype=np.float64))
        sliced_features = features.slice_seconds(30, 95)
        self.assertEqual(sliced_features.duration, 65.0)
        self.assertTrue(np.array_equal(sliced_features.onsets, np.array([0, 30, 60], dtype=np.float64)))

    def test_slice_at_end(self):
        # Slicing exactly at onset times to check boundary inclusion
        features = OnsetFeatures(duration=90.0, onsets=np.array([0, 30, 60, 90], dtype=np.float64))
        sliced_features = features.slice_seconds(30, -1)
        self.assertEqual(sliced_features.duration, 60.0)
        self.assertTrue(np.array_equal(sliced_features.onsets, np.array([0, 30], dtype=np.float64)))


# Tests every step of the search process
from fyp.audio.mix.align import (
    calculate_onset_boundaries,
    calculate_mashability,
    get_valid_starting_points,
)
from fyp.audio.analysis.base import _dist_discrete_latent
from fyp.audio.analysis.chord import NO_CHORD_PENALTY, UNKNOWN_CHORD_PENALTY, ChordAnalysisResult

def dist_chord(a: ChordAnalysisResult, b: ChordAnalysisResult):
    assert a.duration == b.duration
    dist_array = ChordAnalysisResult.get_dist_array()
    return _dist_discrete_latent(a.times, b.times, a.features, b.features, dist_array, a.duration)

def chords_to_features(chords: list[str]):
    from fyp.util.note import get_inv_voca_map
    voca_map = get_inv_voca_map()
    return np.array([voca_map[chord] for chord in chords], dtype=np.uint32)

class TestAlign(unittest.TestCase):
    def test_calculate_onset_boundaries(self):
        onset1 = OnsetFeatures(duration=8, onsets=np.array([0, 2, 4, 6], dtype=np.float64))
        onset2 = OnsetFeatures(duration=8, onsets=np.array([0, 2, 4, 6], dtype=np.float64))
        factors, boundaries = calculate_onset_boundaries(onset1, onset2)
        self.assertTrue(np.array_equal(factors, np.array([1, 1, 1, 1], dtype=np.float64)), msg=f"Factors: {factors}")
        self.assertTrue(np.array_equal(boundaries, np.array([2, 4, 6, 8], dtype=np.float64)), msg=f"Boundaries: {boundaries}")

        onset1 = OnsetFeatures(duration=8, onsets=np.array([0, 2, 4, 6], dtype=np.float64))
        onset2 = OnsetFeatures(duration=8, onsets=np.array([0, 2, 3, 6], dtype=np.float64))
        factors, boundaries = calculate_onset_boundaries(onset1, onset2)
        self.assertTrue(np.array_equal(factors, np.array([1, 0.5, 1.5, 1], dtype=np.float64)), msg=f"Factors: {factors}")
        self.assertTrue(np.array_equal(boundaries, np.array([2, 3, 6, 8], dtype=np.float64)), msg=f"Boundaries: {boundaries}")

    def test_dist_chord_analysis_result_1(self):
        # Chords are the same, distance should be 0
        chord1 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "E:min", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "E:min", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        self.assertTrue(np.isclose(dist, 0), msg=f"Distance: {dist}")

    def test_dist_chord_analysis_result_2(self):
        # (Unknown chord penalty = 3) * 2 seconds = 6
        chord1 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "Unknown", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "Unknown", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        self.assertTrue(np.isclose(dist, UNKNOWN_CHORD_PENALTY * 2), msg=f"Distance: {dist}")

    def test_dist_chord_analysis_result_3(self):
        # (Unknown chord penalty = 3) * 2 seconds = 6
        chord1 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "Unknown", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "Unknown", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        self.assertTrue(np.isclose(dist, UNKNOWN_CHORD_PENALTY * 2), msg=f"Distance: {dist}")

    def test_dist_chord_analysis_result_4(self):
        # (No chord penalty = 3) * 2 second = 6
        chord1 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "E:min", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "No chord", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        self.assertTrue(np.isclose(dist, NO_CHORD_PENALTY * 2), msg=f"Distance: {dist}")

    def test_dist_chord_analysis_result_5(self):
        # (Unknown chord penalty = 3) * 2 second = 6
        chord1 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "E:min", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "Unknown", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        self.assertTrue(np.isclose(dist, UNKNOWN_CHORD_PENALTY * 2), msg=f"Distance: {dist}")

    def test_dist_chord_analysis_result_6(self):
        # Em and G distance should be 1
        # 1 * 2 = 4
        chord1 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "E:min", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "G", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        target = 2 * ChordAnalysisResult.fdist("E:min", "G")
        self.assertTrue(np.isclose(dist, target), msg=f"Distance: {dist}")

    def test_dist_chord_analysis_result_7(self):
        # Em and Em7 has two notes different, distance should be 1
        # 1 * 2 = 4
        chord1 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "E:min", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "E:min7", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        target = 2 * ChordAnalysisResult.fdist("E:min", "E:min7")
        self.assertTrue(np.isclose(dist, target), msg=f"Distance: {dist}")

    def test_dist_chord_analysis_result_8(self):
        chord1 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "E:min", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "D:min", "E:min", "F"]),
            times=np.array([0, 2, 3, 6], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        target = ChordAnalysisResult.fdist("D:min", "E:min")
        self.assertTrue(np.isclose(dist, target), msg=f"Distance: {dist}")

    def test_dist_chord_analysis_result_9(self):
        # No chord penalty * 8
        chord1 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["No chord", "D:min", "No chord", "F"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "No chord", "E:min", "No chord"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        self.assertTrue(np.isclose(dist, NO_CHORD_PENALTY * 8), msg=f"Distance: {dist}")

    def test_dist_chord_analysis_result_10(self):
        # 0-1: C and C -> 0
        # 1-2: C and G7 -> 7
        # 2-3: G and G7 -> 0
        # 3-4: G and G7 -> 0
        # 4-5: D and G7 -> 3
        # 5-6: D and F#m -> 1
        # 6-7: A and F#m -> 1
        # 7-8: A and E -> 4
        chord1 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "G", "D", "A"]),
            times=np.array([0, 2, 4, 6], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=8,
            features=chords_to_features(["C", "G:7", "F#:min", "E"]),
            times=np.array([0, 1, 5, 7], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        self.assertTrue(np.isclose(dist, sum([
            ChordAnalysisResult.fdist("C", "C"),
            ChordAnalysisResult.fdist("C", "G:7"),
            ChordAnalysisResult.fdist("G", "G:7"),
            ChordAnalysisResult.fdist("G", "G:7"),
            ChordAnalysisResult.fdist("D", "G:7"),
            ChordAnalysisResult.fdist("D", "F#:min"),
            ChordAnalysisResult.fdist("A", "F#:min"),
            ChordAnalysisResult.fdist("A", "E"),
        ])), msg=f"Distance: {dist}")

    def test_dist_chord_analysis_result_11(self):
        chord1 = ChordAnalysisResult(
            duration=4,
            features=chords_to_features(["C", "C", "C", "C"]),
            times=np.array([0, 1, 2, 3], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=4,
            features=chords_to_features(["C", "C", "C", "F"]),
            times=np.array([0, 1, 2, 3], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        target = ChordAnalysisResult.fdist("C", "F")
        self.assertTrue(np.isclose(dist, target), msg=f"Distance: {dist}")

    def test_dist_chord_analysis_result_12(self):
        chord1 = ChordAnalysisResult(
            duration=4,
            features=chords_to_features(["C", "C", "C", "C"]),
            times=np.array([0, 1, 2, 3], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=4,
            features=chords_to_features(["C", "C", "F", "C"]),
            times=np.array([0, 1, 2, 3], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        target = ChordAnalysisResult.fdist("C", "F")
        self.assertTrue(np.isclose(dist, target), msg=f"Distance: {dist}")

    def test_dist_chord_analysis_result_13(self):
        chord1 = ChordAnalysisResult(
            duration=4,
            features=chords_to_features(["C", "C", "C", "C"]),
            times=np.array([0, 1, 2, 3], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=4,
            features=chords_to_features(["C", "F", "C", "C"]),
            times=np.array([0, 1, 2, 3], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        target = ChordAnalysisResult.fdist("C", "F")
        self.assertTrue(np.isclose(dist, target), msg=f"Distance: {dist}")

    def test_dist_chord_analysis_result_14(self):
        chord1 = ChordAnalysisResult(
            duration=4,
            features=chords_to_features(["C", "C", "C", "C"]),
            times=np.array([0, 1, 2, 3], dtype=np.float64),
        )
        chord2 = ChordAnalysisResult(
            duration=4,
            features=chords_to_features(["F", "C", "C", "C"]),
            times=np.array([0, 1, 2, 3], dtype=np.float64),
        )
        dist = dist_chord(chord1, chord2)
        target = ChordAnalysisResult.fdist("C", "F")
        self.assertTrue(np.isclose(dist, target), msg=f"Distance: {dist}")
