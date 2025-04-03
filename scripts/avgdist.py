# This script calculates the average distance between chord features and saves the results to a file.
# It uses our fyp library to load audio files and analyze their chord features.

import os
import time
import numpy as np
import numba
import torch
from tqdm import tqdm, trange
from fyp import YouTubeURL, get_url, Audio
from fyp.audio.analysis.chord import get_chord_result, analyse_chord_features, analyse_chord_transformer
from fyp import SongDataset


@numba.njit
def calculate_distances(labels: np.ndarray, distances: np.ndarray, distances_sum: np.ndarray, count: np.ndarray, fs: int):
    for i in range(fs):
        for j in range(fs):
            class_i = labels[i]
            class_j = labels[j]
            d = distances[i, j]
            distances_sum[class_i, class_j, 0] += d
            distances_sum[class_i, class_j, 1] += d * d
            count[class_i, class_j] += 1


class DistanceCalculator:
    def __init__(self, num_classes: int):
        self.distances_sum = np.zeros((num_classes, num_classes, 2), dtype=np.float64)
        self.count = np.zeros((num_classes, num_classes), dtype=np.uint64)
        self.num_classes = num_classes

        os.makedirs("resources/cache", exist_ok=True)
        self.cache_path = "resources/distance_calculator.npz"

        if os.path.exists(self.cache_path):
            data = np.load(self.cache_path, allow_pickle=True)
            self.distances_sum = data['distances_sum']
            self.count = data['count']
            assert self.distances_sum.shape == (num_classes, num_classes, 2), "Invalid shape for distances_sum"
            assert self.count.shape == (num_classes, num_classes), "Invalid shape for count"
            print("Loaded cached distances and counts from", self.cache_path)

    def update(self, features: torch.Tensor, logits: torch.Tensor):
        t1 = time.time()
        labels = np.argmax(logits.numpy(), axis=1)
        distances = torch._euclidean_dist(features, features).numpy()
        calculate_distances(labels, distances, self.distances_sum, self.count, features.size(0))
        tqdm.write(f"Number of distances calculated: {np.sum(self.count)}, mean dist: {np.mean(distances)}, took {time.time() - t1:.2f} seconds")
        np.savez(self.cache_path, distances_sum=self.distances_sum, count=self.count)


def main(path: str):
    sd = SongDataset(path, load_on_the_fly=True)
    sd.register("chords", "{video_id}.json")
    dist_calc = DistanceCalculator(num_classes=170)
    audios = sd.list_urls("audio")
    for url in tqdm(audios, desc="Processing audio files"):
        path = sd.get_path("audio", url)
        try:
            audio = Audio.load(path)
        except Exception as e:
            tqdm.write(f"Error loading audio file {path}: {e}")
            continue

        try:
            result = get_chord_result(audio, use_cache=False)
            result.save(sd.get_path("chords", url))
            dist_calc.update(result.features, result.logits)
        except Exception as e:
            tqdm.write(f"Error processing audio file {path}: {e}")
            continue


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.avgdist <path_to_audio_files>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        sys.exit(1)
    main(path)
