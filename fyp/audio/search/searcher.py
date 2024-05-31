from __future__ import annotations
import os
from ..analysis import ChordAnalysisResult, BeatAnalysisResult, analyse_beat_transformer, analyse_chord_transformer, analyse_beat_transformer_local
from ..analysis import TimeSegmentResult
from ... import Audio
from .. import AudioCollection
from typing import Any
from datasets import load_from_disk, Dataset, load_dataset
from ..separation import DemucsAudioSeparator
from fyp.audio.analysis import pychorus, top_k_stft, top_k_edge_detection, top_k_maxpool, top_k_rms
from math import exp
from collections import Counter
from fyp.audio import AudioCollection
from .search_config import SearchConfig, SongSearchCallbackHandler
from .align import search_database, calculate_boundaries
from ...util.combine import get_video_id, filter_dataset

NULL_FN = lambda *args, **kwargs: None

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

def filter_first(scores: list[tuple[float, str]]) -> list[tuple[float, str]]:
	seen = set()
	result = []
	for score in scores:
		id = score[1][:11]
		if id not in seen:
			seen.add(id)
			result.append(score)
	return result

def curve_score(score: float) -> float:
	"""Returns the curve score"""
	# # Unnormalize the score back to its original duration
	# score_ = score * bar_length
	return round(100 * exp(-.05 * score), 2)

class SongSearcher:
	"""The main pipeline, which wraps the search, create_mashup and other functions, and handles internal states for you."""
	def __init__(self, callback_handler: SongSearchCallbackHandler | None = None, 
				 search_config: SearchConfig | None = None, 
				 dataset: Dataset | list[dict[str, Any]] | None = None,
				 dataset_path: str = "./backend/features/audio-infos-filtered"):
		self.search_config = search_config or SearchConfig()
		if dataset is None:
			if os.path.isdir(dataset_path):
				dataset = load_from_disk(dataset_path=dataset_path, keep_in_memory=True)
			else:
				dataset = load_dataset("HKUST-FYPHO2/audio-infos-filtered", split="train")
				dataset.save_to_disk(dataset_path)
				dataset = load_from_disk(dataset_path=dataset_path, keep_in_memory=True)
		assert isinstance(dataset, Dataset) or isinstance(dataset, list), dataset
		self.dataset = dataset
		self._all_scores: list[tuple[float, str]] = []
		self.callback_handler = callback_handler if callback_handler is not None else SongSearchCallbackHandler()
		
	@property
	def cache_dir(self) -> str | None:
		"""Returns the beat cache path. If not found or cache is disabled, return None"""
		if self.search_config.cache_dir is None:
			return None
		if not self.search_config.cache:
			return None
		if not os.path.isdir(self.search_config.cache_dir):
			os.makedirs(self.search_config.cache_dir)
		return self.search_config.cache_dir
	
	@property
	def beat_cache_path(self) -> str | None:
		"""Returns the beat cache path. If not found or cache is disabled, return None"""
		if not self.search_config.cache or self.cache_dir is None:
			return None
		return os.path.join(self.cache_dir, f"beat_{get_video_id(self.link)}.cache")
	
	@property
	def chord_cache_path(self) -> str | None:
		"""Returns the chord cache path. If not found or cache is disabled, return None"""
		if not self.search_config.cache or self.cache_dir is None:
			return None
		return os.path.join(self.cache_dir, f"chord_{get_video_id(self.link)}.cache")
	
	@property
	def audio_cache_path(self) -> str | None:
		"""Returns the audio cache path. If not found or cache is disabled, return None"""
		if not self.search_config.cache or self.cache_dir is None:
			return None
		return os.path.join(self.cache_dir, f"audio_{get_video_id(self.link)}.wav")

	@property
	def link(self) -> str:
		"""Returns the link of the audio. Raises ValueError if not found."""
		if not hasattr(self, "_link"):
			raise ValueError("No link found")
		return self._link
	
	def set_link(self, value: str, *, reset_states: bool = True):
		"""Sets the link of the audio. If reset_states is True, the states of the pipeline will be reset."""
		self._link = value
		# Unset the audio and the chord result
		if reset_states:
			for attrs in ("_audio", "_raw_chord_result", "_raw_beat_result", "_raw_parts_result", 
					"_submitted_chord_result", "_submitted_beat_result", "_submitted_audio", "_submitted_parts"):
				if hasattr(self, attrs):
					delattr(self, attrs)
	
	@property
	def audio(self) -> Audio:
		"""Returns the audio object. Raises ValueError if not found. If link is found, the audio will be loaded from the link"""
		if not hasattr(self, "_audio"):
			if not hasattr(self, "_link"):
				raise ValueError("No link and no audio found")
			self._audio = Audio.load(self.link, cache_path=self.audio_cache_path)
			self.callback_handler.on_load(self._audio, self.link)
		return self._audio
	
	def set_audio(self, value: Audio):
		"""Sets the audio object."""
		self._audio = value.clone()

	@property
	def raw_chord_result(self) -> ChordAnalysisResult:
		"""The raw chord result of the user-submitted song without any processing."""
		if not hasattr(self, "_raw_chord_result"):
			self.callback_handler.on_chord_transformer_start(self.audio)
			self._raw_chord_result = analyse_chord_transformer(
				self.audio, 
				model_path=self.search_config.chord_model_path,
				cache_path = self.chord_cache_path
			)
			self.callback_handler.on_chord_transformer_end(self._raw_chord_result)
		return self._raw_chord_result
	
	def set_raw_chord_result(self, value: ChordAnalysisResult):
		self._raw_chord_result = value

	@property
	def raw_beat_result(self) -> BeatAnalysisResult:
		"""The raw beat result of the user-submitted song without any processing."""
		if not hasattr(self, "_raw_beat_result"):
			self.callback_handler.on_beat_transformer_start(self.audio)
			if self.search_config.use_request_beat_transformer:
				br = analyse_beat_transformer(
					self.audio, 
					url = self.search_config.backend_url, 
					cache_path = self.beat_cache_path,
				)
			else:
				br = analyse_beat_transformer_local(
					parts = lambda: self.raw_parts_result, 
					cache_path = self.beat_cache_path,
					model_path=self.search_config.beat_model_path
				)
			self._raw_beat_result = br
			self.callback_handler.on_beat_transformer_end(self._raw_beat_result)
		return self._raw_beat_result

	@property
	def slice_start_bar(self):
		"""Returns the start bar of the slice. If not found, it will perform the chorus analysis."""
		try:
			return self._slice_start_bar
		except AttributeError:
			self._slice_start_bar, self._slice_nbar = self._chorus_analysis()
			return self._slice_start_bar
	
	@property
	def slice_nbar(self):
		"""Returns the number of bars of the slice. If not found, it will perform the chorus analysis."""
		if hasattr(self, "_slice_nbar"):
			return self._slice_nbar
		if hasattr(self, "_submitted_beat_result"):
			return len(self.submitted_beat_result.downbeats)
		self._slice_start_bar, self._slice_nbar = self._chorus_analysis()
		return self._slice_nbar
	
	@property
	def slice_start(self) -> float:
		"""Returns the start time of the slice in seconds."""
		return self.raw_beat_result.downbeats[self.slice_start_bar]
	
	@property
	def slice_end(self) -> float:
		"""Returns the end time of the slice in seconds."""
		return self.raw_beat_result.downbeats[self.slice_start_bar + self.slice_nbar]
	
	@property
	def raw_parts_result(self):
		"""Returns the raw parts result of the user-submitted song without any processing."""
		if not hasattr(self, "_raw_parts_result"):
			demucs = DemucsAudioSeparator()
			self.callback_handler.on_demucs_start(self.audio)
			self._raw_parts_result = demucs.separate_audio(self.audio)
			self.callback_handler.on_demucs_end(self._raw_parts_result)
		return self._raw_parts_result
	
	@property
	def submitted_chord_result(self):
		"""Returns the chord result submitted for database search. It is the slice of the raw chord result."""
		if not hasattr(self, "_submitted_chord_result"):
			self._submitted_chord_result = self.raw_chord_result.slice_seconds(self.slice_start, self.slice_end)
		return self._submitted_chord_result
	
	@property
	def submitted_beat_result(self):
		"""Returns the beat result submitted for database search. It is the slice of the raw beat result."""
		if not hasattr(self, "_submitted_beat_result"):
			self._submitted_beat_result = self.raw_beat_result.slice_seconds(self.slice_start, self.slice_end)
		return self._submitted_beat_result
	
	@property
	def submitted_audio(self):
		"""Returns the audio submitted for database search. It is the slice of the raw audio."""
		if not hasattr(self, "_submitted_audio"):
			self._submitted_audio = self.audio.slice_seconds(self.slice_start, self.slice_end)
		return self._submitted_audio
	
	@property
	def submitted_parts(self):
		if not hasattr(self, "_submitted_parts"):
			self._submitted_parts = self.raw_parts_result.slice_seconds(self.slice_start, self.slice_end)
		return self._submitted_parts

	def _chorus_analysis(self):
		"""Extracts the chorus and returns the start bar and the number of bars"""
		if self.search_config.bar_number is not None:
			self._slice_start_bar = self.search_config.bar_number
			self._slice_nbar = self.search_config.nbars if self.search_config.nbars is not None else 8

			if self._slice_start_bar + self._slice_nbar >= len(self.raw_beat_result.downbeats):
				raise ValueError(f"Bar number out of bounds: {self._slice_start_bar + self._slice_nbar} >= {len(self.raw_beat_result.downbeats)}")
			
			return self._slice_start_bar, self._slice_nbar
		self.callback_handler.on_chorus_start(self.audio)
		slice_ranges = extract_chorus(self.raw_parts_result, self.raw_beat_result, self.audio,
										work_factor=self.search_config.pychorus_work_factor)
		slice_nbar = 8
		slice_start_bar = 0
		for i in range(len(slice_ranges)):
			if slice_ranges[i] + slice_nbar < len(self.raw_beat_result.downbeats):
				slice_start_bar = slice_ranges[i]
				break

		# Prevent array out of bounds below
		slice_nbar = self.search_config.nbars if self.search_config.nbars is not None else slice_nbar
		self.callback_handler.on_chorus_end(self.raw_beat_result.downbeats[slice_start_bar])

		assert slice_start_bar + slice_nbar < len(self.raw_beat_result.downbeats)
		self._slice_start_bar = slice_start_bar
		self._slice_nbar = slice_nbar
		return slice_start_bar, slice_nbar
	
	def search(self, link: str, audio: Audio | None = None, *, reset_states: bool = True):
		"""Searches for songs that match the audio. Returns a list of tuples, where the first element is the score and the second element is the url.
		
		:param link: The link of the audio
		:param audio: The audio object. If not provided, the audio will be loaded from the link.
		:param reset_states: Whether to reset the states of the pipeline. Default is True."""
		if link:
			self.set_link(link, reset_states=reset_states)
		if audio:
			self.set_audio(audio)
		
		if self._all_scores:
			return self._all_scores

		# Perform the search
		if self.search_config.skip_search:
			return []
		
		self.callback_handler.on_search_start(str(link))

		dataset = filter_dataset(self.dataset, self.search_config.filter_dataset)
		
		on_search_progress = self.callback_handler.on_search_database_entry if self.search_config.progress_bar else NULL_FN
		on_search_start = self.callback_handler.on_search_database_start if self.search_config.progress_bar else NULL_FN
		on_search_end = self.callback_handler.on_search_database_end if self.search_config.progress_bar else NULL_FN
		
		scores_ = search_database(
							submitted_chord_result=self.submitted_chord_result, 
							submitted_beat_result=self.submitted_beat_result, 
							dataset=dataset,
							search_config=self.search_config,
							on_search_progress=on_search_progress,
							on_search_start=on_search_start,
							on_search_end=on_search_end)

		if self.search_config.filter_first:
			scores_ = filter_first(scores_)

		# bar_length = 60 / self.submitted_beat_result.tempo

		## Curve the scores
		scores = [(curve_score(x[0]), x[1]) for x in scores_]
		self.callback_handler.on_search_end(scores)
		self._all_scores = scores
		scores = [s for s in scores if self.search_config.min_score <= s[0] <= self.search_config.max_score]
		return scores
