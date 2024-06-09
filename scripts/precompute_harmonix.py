# We calculate 4 things: the chord result, beat result, the beat similarity and the downbeat similarity
# All of them havent been sped up
import numpy as np
from fyp import Audio
from fyp.audio.analysis import analyse_chord_transformer, analyse_beat_transformer

import os
from fyp.util import clear_cuda
import re
from tqdm import tqdm
from fyp.audio.search.search import calculate_self_similarity, calculate_self_similarity_beat

# harmonix might contain something like "(needs to be 3.05% sped-up)"
# Use a regex to extract the percentage
def extract_speedup_percentage(harmonix: str):
    percentage = re.search(r"\(needs to be (-?\d+\.\d+)% sped-up\)", harmonix)
    if percentage is None:
        return 0., harmonix
    # Remove the bracketed part from the string
    harmonix = harmonix[:percentage.start()].strip()
    return float(percentage.group(1)), harmonix

def get_urls():
    urls_path = "resources/harmonixset/dataset/youtube_urls.csv"
    entries = {}
    with open(urls_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("File"):
                continue
            name, entry = line.split(",")
            entries[name] = entry

    speedups = {}
    for name, entry in entries.items():
        factor, url = extract_speedup_percentage(entry)
        speedups[url] = (factor, name)

    return speedups

def process_audio(url: str, name: str, factor: float):
    ct_path = os.path.join("resources/harmonixset/dataset/data", name + ".ct")
    if os.path.exists(ct_path):
        return
    try:
        audio = Audio.load(os.path.join("resources/harmonixset/dataset/audio", name + ".wav"))
    except Exception as e:
        audio = Audio.load(url)
    ct = analyse_chord_transformer(audio)
    bt = analyse_beat_transformer(audio)
    bs4 = calculate_self_similarity_beat(ct, bt, 4)
    bs8 = calculate_self_similarity_beat(ct, bt, 8)
    ds4 = calculate_self_similarity(ct, bt, 4)
    ds8 = calculate_self_similarity(ct, bt, 8)
    clear_cuda()
    ct.save(ct_path)
    bt.save(os.path.join("resources/harmonixset/dataset/data", name + ".bt"))
    np.save(os.path.join("resources/harmonixset/dataset/data", name + ".bs4"), bs4)
    np.save(os.path.join("resources/harmonixset/dataset/data", name + ".bs8"), bs8)
    np.save(os.path.join("resources/harmonixset/dataset/data", name + ".ds4"), ds4)
    np.save(os.path.join("resources/harmonixset/dataset/data", name + ".ds8"), ds8)

def main():
    if not os.path.exists("resources/harmonixset/dataset/data"):
        os.makedirs("resources/harmonixset/dataset/data")

    urls: dict[str, tuple[float, str]] = get_urls()
    for url, (factor, name) in tqdm(urls.items()):
        try:
            process_audio(url, name, factor)
        except Exception as e:
            print(f"Failed to process {name}: {e}")

if __name__ == "__main__":
    main()
