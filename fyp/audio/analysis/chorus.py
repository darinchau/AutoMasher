from .. import AudioCollection
from .. import Audio
from .beat import BeatAnalysisResult
from .base import TimeSegmentResult
from .time_seg import pychorus, top_k_edge_detection, top_k_maxpool, top_k_rms, top_k_stft
from collections import Counter


def extract_chorus(parts: AudioCollection, beat_result: BeatAnalysisResult, audio: Audio, work_factor: int) -> list[int]:
	"""Extract the chorus position from the beat result"""
	try:
		pychorus_result = pychorus(beat_result, audio, work_factor = work_factor)
	except Exception as e:
		print("Error in pychorus:", e)
		pychorus_result = TimeSegmentResult([], duration=audio.duration)
	
	vocals = parts['vocals']
	time_segment_results = [
		pychorus_result,
		top_k_edge_detection(vocals),
		top_k_stft(vocals),
		top_k_maxpool(vocals),
		top_k_rms(vocals)
	]

	s = []
	for tx in time_segment_results:
		s.extend(tx.align_with_closest_downbeats(beat_result))
	# s is a list of integers
	# Sort the unique elements in s according to their frequency
	# The most frequent elements will be at the beginning

	c = Counter(s)
	sorted_s = sorted(set(s), key=lambda x: (-c[x], s.index(x)))

	return sorted_s
