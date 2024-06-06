"""
    This part of the project is derived from pychrous's source code
    Original repo: https://github.com/vivjay30/pychorus/tree/master
"""
import numpy as np
from scipy.signal import argrelextrema
from .chroma import chroma_stft_nfft
from .. import Audio
from numpy.typing import NDArray
import numpy as np
# import librosa
"""
    This part of the project is derived from pychrous's source code
    Original repo: https://github.com/vivjay30/pychorus/tree/master
"""
from abc import ABCMeta, abstractmethod
from math import sqrt
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from .base import BeatAnalysisResult
from .base import Audio
from .base import TimeSegmentResult
from scipy import stats
import librosa
from torch import nn
import torch


def local_maxima_rows(denoised_time_lag):
    """Find rows whose normalized sum is a local maxima"""
    row_sums = np.sum(denoised_time_lag, axis=1)
    divisor = np.arange(row_sums.shape[0], 0, -1)
    normalized_rows = row_sums / divisor
    local_minima_rows = argrelextrema(normalized_rows, np.greater)
    return local_minima_rows[0]


def detect_lines(denoised_time_lag, rows, min_length_samples: int, decay: float, thershold: float, num_iter: int, min_lines: int):
    """Detect lines in the time lag matrix. Reduce the threshold until we find enough lines"""
    cur_threshold = thershold
    for _ in range(num_iter):
        line_segments = detect_lines_helper(denoised_time_lag, rows,
                                            cur_threshold, min_length_samples)
        if len(line_segments) >= min_lines:
            return line_segments
        cur_threshold *= decay
    return line_segments


def detect_lines_helper(denoised_time_lag, rows, threshold: float,
                        min_length_samples: int):
    """Detect lines where at least min_length_samples are above threshold"""
    num_samples = denoised_time_lag.shape[0]
    line_segments = []
    cur_segment_start = None
    for row in rows:
        if row < min_length_samples:
            continue
        for col in range(row, num_samples):
            if denoised_time_lag[row, col] > threshold:
                if cur_segment_start is None:
                    cur_segment_start = col
            else:
                if (cur_segment_start is not None
                   ) and (col - cur_segment_start) > min_length_samples:
                    line_segments.append(Line(cur_segment_start, col, row))
                cur_segment_start = None
    return line_segments


def count_overlapping_lines(lines, margin, min_length_samples):
    """Look at all pairs of lines and see which ones overlap vertically and diagonally"""
    line_scores = {}
    for line in lines:
        line_scores[line] = 0

    # Iterate over all pairs of lines
    for line_1 in lines:
        for line_2 in lines:
            # If line_2 completely covers line_1 (with some margin), line_1 gets a point
            lines_overlap_vertically = (
                line_2.start < (line_1.start + margin)) and (
                    line_2.end > (line_1.end - margin)) and (
                        abs(line_2.lag - line_1.lag) > min_length_samples)

            lines_overlap_diagonally = (
                (line_2.start - line_2.lag) < (line_1.start - line_1.lag + margin)) and (
                    (line_2.end - line_2.lag) > (line_1.end - line_1.lag - margin)) and (
                        abs(line_2.lag - line_1.lag) > min_length_samples)

            if lines_overlap_vertically or lines_overlap_diagonally:
                line_scores[line_1] += 1

    # Sorting the scores first by chorus matches, then by duration
    res = []

    for l in line_scores:
        res.append((l, line_scores[l], l.end - l.start))

    # The best result is at the top
    res.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return res


def line_scores(audio: Audio, clip_length: float, decay: float, n_fft: int, smoothing_size: float, margin: float, thershold: float, num_iter: int, min_lines: int):
    """
    Find the lines scores

    Args:
        chroma: 12 x n frequency chromogram
        sr: sample rate of the song, usually 22050
        song_length_sec: length in seconds of the song (lost in processing chroma)
        clip_length: minimum length in seconds we want our chorus to be (at least 10-15s)

    Returns:
        line_scores: valid line scores computed,
        chorma_sr: is used to find each chorus section start by line_scores[0][i].start/chorma_sr
    """
    chroma = chroma_stft_nfft(audio, n_fft)

    num_samples = chroma.shape[1]

    time_time_similarity = Time2Time(chroma)
    time_lag_similarity = TimeLag(chroma)

    # Denoise the time lag matrix
    chroma_sr = num_samples / (audio.duration)
    smoothing_size_samples = int(smoothing_size * chroma_sr)
    time_lag_similarity.denoise(time_time_similarity.matrix,
                                smoothing_size_samples)

    # Detect lines in the image
    clip_length_samples = int(clip_length * chroma_sr)
    candidate_rows = local_maxima_rows(time_lag_similarity.matrix)
    lines = detect_lines(time_lag_similarity.matrix, candidate_rows,
                         clip_length_samples, decay, thershold, num_iter, min_lines)
    if len(lines) == 0:
        print("No choruses were detected. Try a smaller search duration")
        return [], chroma_sr
    line_scores = count_overlapping_lines(
        lines, margin * clip_length_samples,
        clip_length_samples)
    best_chorus = line_scores[0][0]
    # return best_chorus.start / chroma_sr, line_scores
    return line_scores, chroma_sr


def valid_start(t, starts, seg_len):
    for c in starts:
        if abs(c-t) < seg_len:
            return False
        else:
            continue
    return True


def bin_search(t, segs):
    if len(segs) == 2:
        l_dist = abs(t-segs[0])
        r_dist = abs(t-segs[1])
        res = segs[0] if l_dist < r_dist else segs[1]

        return res

    elif len(segs) == 1:
        return segs[0]

    else:
        mid = len(segs)//2
        res = bin_search(t, segs[:mid]) if t < segs[mid] else bin_search(t, segs[mid+1:])
        return res


def find_first_downbeat(first_down: float, all: NDArray[np.floating]) -> int:
    return int(np.argmin(np.abs(all-first_down)))


class SimilarityMatrix(object):
    """Abstract class for our time-time and time-lag similarity matrices"""

    __metaclass__ = ABCMeta

    def __init__(self, chroma):
        self.chroma = chroma
        self.matrix = self.compute_similarity_matrix(chroma)

    @abstractmethod
    def compute_similarity_matrix(self, chroma):
        """"
        The specific type of similarity matrix we want to compute

        Args:
            chroma: 12 x n numpy array of musical notes present at every time step
        """
        pass


class Time2Time(SimilarityMatrix):
    """
    Class for the time time similarity matrix where sample (x,y) represents how similar
    are the song frames x and y
    """

    def compute_similarity_matrix(self, chroma):
        """Optimized way to compute the time-time similarity matrix with numpy broadcasting"""
        broadcast_x = np.expand_dims(chroma, 2)  # (12 x n x 1)
        broadcast_y = np.swapaxes(np.expand_dims(chroma, 2), 1,
                                  2)  # (12 x 1 x n)
        time_time_matrix = 1 - (np.linalg.norm(
            (broadcast_x - broadcast_y), axis=0) / sqrt(12))
        return time_time_matrix

    def compute_similarity_matrix_slow(self, chroma):
        """Slow but straightforward way to compute time time similarity matrix"""
        num_samples = chroma.shape[1]
        time_time_similarity = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                # For every pair of samples, check similarity
                time_time_similarity[i, j] = 1 - (
                    np.linalg.norm(chroma[:, i] - chroma[:, j]) / sqrt(12))

        return time_time_similarity

def convolve2d(matrix, kernel, mode):
    """
    Convolve a 2d matrix with a kernel
    """
    mat = torch.tensor(matrix.copy())
    ker = torch.tensor(kernel.copy(), dtype=torch.float32)
    torch_results = F.conv2d(mat.unsqueeze(0).unsqueeze(0), ker.unsqueeze(0).unsqueeze(0), padding=(kernel.shape[0] - 1, kernel.shape[1] - 1))[0, 0]
    torch_results = torch_results.numpy()
    return torch_results

class TimeLag(SimilarityMatrix):
    """
    Class to hold the time lag similarity matrix where sample (x,y) represents the
    similarity of song frames x and (x-y)
    """

    def compute_similarity_matrix(self, chroma):
        """Optimized way to compute the time-lag similarity matrix"""
        num_samples = chroma.shape[1]
        broadcast_x = np.repeat(
            np.expand_dims(chroma, 2), num_samples + 1, axis=2)

        # We create the lag effect by tiling the samples but reshaping with an extra column
        # so that subsequent rows are offset by one each time
        circulant_y = np.tile(chroma, (1, num_samples + 1)).reshape(
            12, num_samples, num_samples + 1)
        time_lag_similarity = 1 - (np.linalg.norm(
            (broadcast_x - circulant_y), axis=0) / sqrt(12))
        time_lag_similarity = np.rot90(time_lag_similarity, k=1, axes=(0, 1))
        return time_lag_similarity[:num_samples, :num_samples]

    def denoise(self, time_time_matrix, smoothing_size):
        """
        Emphasize horizontal lines by suppressing vertical and diagonal lines. We look at 6
        moving averages (left, right, up, down, upper diagonal, lower diagonal). For lines, the
        left or right average should be much greater than the other ones.

        Args:
            time_time_matrix: n x n numpy array to quickly compute diagonal averages
            smoothing_size: smoothing size in samples (usually 1-2 sec is good)
        """
        n = self.matrix.shape[0]

        # Get the horizontal strength at every sample
        horizontal_smoothing_window = np.ones(
            (1, smoothing_size)) / smoothing_size
        horizontal_moving_average = convolve2d(
            self.matrix, horizontal_smoothing_window, mode="full")
        left_average = horizontal_moving_average[:, 0:n]
        right_average = horizontal_moving_average[:, smoothing_size - 1:]
        max_horizontal_average = np.maximum(left_average, right_average)

        # Get the vertical strength at every sample
        vertical_smoothing_window = np.ones((smoothing_size,
                                             1)) / smoothing_size
        vertical_moving_average = convolve2d(
            self.matrix, vertical_smoothing_window, mode="full")
        down_average = vertical_moving_average[0:n, :]
        up_average = vertical_moving_average[smoothing_size - 1:, :]

        # Get the diagonal strength of every sample from the time_time_matrix.
        # The key insight is that diagonal averages in the time lag matrix are horizontal
        # lines in the time time matrix
        diagonal_moving_average = convolve2d(
            time_time_matrix, horizontal_smoothing_window, mode="full")
        ur_average = np.zeros((n, n))
        ll_average = np.zeros((n, n))
        for x in range(n):
            for y in range(x):
                ll_average[y, x] = diagonal_moving_average[x - y, x]
                ur_average[y, x] = diagonal_moving_average[x - y,
                                                           x + smoothing_size - 1]

        non_horizontal_max = np.maximum.reduce([down_average, up_average, ll_average, ur_average])
        non_horizontal_min = np.minimum.reduce([up_average, down_average, ll_average, ur_average])

        # If the horizontal score is stronger than the vertical score, it is considered part of a line
        # and we only subtract the minimum average. Otherwise subtract the maximum average
        suppression = (max_horizontal_average > non_horizontal_max) * non_horizontal_min + (
            max_horizontal_average <= non_horizontal_max) * non_horizontal_max

        # Filter it horizontally to remove any holes, and ignore values less than 0
        denoised_matrix = gaussian_filter1d(
            np.triu(self.matrix - suppression), smoothing_size, axis=1)
        denoised_matrix = np.maximum(denoised_matrix, 0)
        denoised_matrix[0:5, :] = 0

        self.matrix = denoised_matrix

class Line(object):
    def __init__(self, start, end, lag):
        self.start = start
        self.end = end
        self.lag = lag

def get_all_segments(beat_res: BeatAnalysisResult, audio: Audio, work_factor: int = 13,
                 smoothing_size: float = 2.5,
                 n_fft: int = 2**14,
                 line_theshold: float = 0.15,
                 min_lines: int = 8,
                 num_iter: int = 8,
                 overlap_margin: float = 0.2,
                 decay: float = 0.95):
        required_length = 60 / beat_res.tempo * 4 * 16

        if required_length > 30:
            required_length /= 2
        elif required_length < 15:
            required_length *=2

        # Find the chorus by using the line scores
        scores, ratio = line_scores(audio, required_length, decay, n_fft,
                             smoothing_size, overlap_margin,
                             line_theshold, num_iter,
                             min_lines)

        if len(scores) == 0:
            return [0, audio.duration]

        # Determine all starting positions
        starts: list[float] = []
        for m in scores:
            starts.append(m[0].start/ratio)

        segs: list[float] = []
        for t in starts:
            if valid_start(t, segs, required_length):
                segs.append(t)

        # Prepare the results
        res: list[float] = []
        for t in segs:
            res.append(t)
            res.append(t + required_length)
        res.append(0)
        res.append(audio.duration)
        res = sorted(res)
        return res

def pychorus(beat_result: BeatAnalysisResult, audio: Audio, work_factor: int = 13) -> TimeSegmentResult:
    # work factor is the number of seconds we want our chorus to be
    if work_factor < 10 or work_factor > 20:
        raise ValueError("work factor should be between 10 and 20")
    segs = get_all_segments(beat_result, audio, work_factor=work_factor)

    # Extract the start and end times
    starts = segs[1:-1:2]
    # ends = segs[2:-1:2]

    # closest_start_dbs = np.abs(np.array(starts).reshape(-1, 1) - beat_result.downbeats).argmin(axis=1)
    # closest_end_dbs = np.abs(np.array(ends).reshape(-1, 1) - beat_result.downbeats).argmin(axis=1)
    # chorus_bar_length = stats.mode(closest_end_dbs - closest_start_dbs).mode
    return TimeSegmentResult(starts, beat_result.get_duration())

def top_k_stft(aud: Audio, k: int = 5, HOP_LENGTH: int = 4096) -> TimeSegmentResult:
	"""
	Takes in audio (mono stereo both ok), returns top k timestamps with largest increase in volume.
	Computed using short term fourier transform together with root mean square.
	--- Parameters ---
	Audio aud: The input audio, mono stereo both ok
	int k: How many timestamps to return
	int HOP_LENGTH: bin size, used for calculation in RMS and STFT
	---- Returns ---
	np.ndarray: the top k timestamps, where largest difference comes first
	"""
	# change the audio to MONO and cast to numpy
	aud_np = aud.to_nchannels(1).numpy()

	# result from using RMS with STFT spectrogram
	stft, phase = librosa.magphase(librosa.stft(aud_np, hop_length=HOP_LENGTH))
	result_stft = librosa.feature.rms(S=stft, hop_length=HOP_LENGTH).reshape(-1)

	# get derivative and find top 5 timestamps with largest derivative
	arg_sorted_stft = np.diff(result_stft).argsort()
	top_5_diff_stft = arg_sorted_stft[arg_sorted_stft.shape[0]-5: arg_sorted_stft.shape[0]][np.arange(4, -1, -1)]

	# print top 5 derivative timestamps (in seconds)
	return TimeSegmentResult(top_5_diff_stft * HOP_LENGTH / aud.sample_rate, aud.duration)

def top_k_rms(aud: Audio, k: int = 5, HOP_LENGTH: int = 4096) -> TimeSegmentResult:
	"""
	Takes in audio (mono stereo both ok), returns top k timestamps with largest increase in volume.
	Computed using root mean square.
	--- Parameters ---
	Audio aud: The input audio, mono stereo both ok
	int k: How many timestamps to return
	int HOP_LENGTH: bin size, used for calculation in RMS
	---- Returns ---
	np.ndarray: the top k timestamps, where largest difference comes first
	"""
	# change the audio to MONO and cast to numpy
	aud_np = aud.to_nchannels(1).numpy()

	# result from using RMS without STFT spectrogram
	result_rms = librosa.feature.rms(y=aud_np, hop_length=HOP_LENGTH).reshape(-1)

	# get derivative and find top 5 timestamps with largest derivative
	arg_sorted_rms = np.diff(result_rms).argsort()
	top_5_diff_rms = arg_sorted_rms[arg_sorted_rms.shape[0]-5: arg_sorted_rms.shape[0]][np.arange(4, -1, -1)]

	# print top 5 derivative timestamps (in seconds)
	return TimeSegmentResult(top_5_diff_rms * HOP_LENGTH / aud.sample_rate, aud.duration)

def pick_top_k_timestamps(timestamps: torch.Tensor, sliding_max_diff: torch.Tensor, sorted_args: torch.Tensor, k: int) -> list[float]:
    if timestamps.size(0) == 1:
        return timestamps.tolist()

    elements = []
    for i in range(k):
        if timestamps[0, i] == timestamps[1, i]:
            elements.append(timestamps[0, i].item())
            if len(elements) >= k:
                break
            continue

        dif1 = sliding_max_diff[0, sorted_args[0, i]]
        dif2 = sliding_max_diff[1, sorted_args[1, i]]
        ts1, ts2 = timestamps[0, i].item(), timestamps[1, i].item()
        ts1, ts2 = (ts1, ts2) if dif1 > dif2 else (ts2, ts1)
        elements.append(ts1)
        if len(elements) >= k:
            break
        elements.append(ts2)
        if len(elements) >= k:
            break
    return elements

def top_k_maxpool(vocals: Audio, kernel_size: int | None = None, stride: int = 2048, padding: int | None = None, k: int = 5) -> TimeSegmentResult:
    if kernel_size is None:
        kernel_size = vocals.sample_rate // 8
    if padding is None:
        padding = kernel_size // 2
    kernel = nn.MaxPool1d(kernel_size, stride = stride, padding=kernel_size//2)
    sliding_max = kernel(vocals._data.abs())
    sliding_max_diff = sliding_max[:, 1:] - sliding_max[:, :-1]
    sliding_max_diff[sliding_max_diff < 0] = 0

    sorted_diffs = torch.argsort(sliding_max_diff, dim=1, descending=True)
    timestamps = sorted_diffs[:, :k] * stride / vocals.sample_rate
    elements = pick_top_k_timestamps(timestamps, sliding_max_diff, sorted_diffs, k)
    return TimeSegmentResult(elements, vocals.duration)

def top_k_edge_detection(vocals: Audio, kernel_size: int | None = None, stride: int = 2048, padding: int | None = None, k: int = 5) -> TimeSegmentResult:
    if kernel_size is None:
        kernel_size = vocals.sample_rate // 8
    if padding is None:
        padding = kernel_size // 2
    if not vocals.nchannels == 2:
        vocals = vocals.to_nchannels(2)
    kernel = nn.Conv1d(2, 2, kernel_size, stride = stride, padding=padding//2, bias=False, padding_mode='zeros')
    kr = torch.ones_like(kernel.weight)
    kr[:, :, kernel_size//2:] = -1
    kernel.weight = nn.Parameter(kr)
    sliding_max = kernel(vocals._data.abs())
    sliding_max_diff = sliding_max[:, 1:] - sliding_max[:, :-1]
    sliding_max_diff[sliding_max_diff < 0] = 0

    sorted_diffs = torch.argsort(sliding_max_diff, dim=1, descending=True)
    timestamps = sorted_diffs[:, :k] * stride / vocals.sample_rate
    elements = pick_top_k_timestamps(timestamps, sliding_max_diff, sorted_diffs, k)
    return TimeSegmentResult(elements, vocals.duration)
