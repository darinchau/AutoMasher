import os
from fyp import Audio
import re
from tqdm import tqdm

# harmonix might contain something like "(needs to be 3.05% sped-up)"
# Use a regex to extract the percentage
def extract_speedup_percentage(harmonix: str):
    percentage = re.search(r"\(needs to be (-?\d+\.\d+)% sped-up\)", harmonix)
    if percentage is None:
        return 0, harmonix
    # Remove the bracketed part from the string
    harmonix = harmonix[:percentage.start()].strip()
    return float(percentage.group(1)), harmonix

def load_audio_from_harmonix(harmonix: str):
    speedup, url = extract_speedup_percentage(harmonix)
    audio = Audio.load(url)
    return audio.change_speed(speedup)

def main():
    urls_path = "resources/harmonixset/dataset/youtube_urls.csv"
    urls = {}
    with open(urls_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("File"):
                continue
            name, url = line.split(",")
            urls[name] = url

    if not os.path.exists("resources/harmonixset/dataset/audio"):
        os.makedirs("resources/harmonixset/dataset/audio")

    for name, url in tqdm(urls.items()):
        try:
            audio = load_audio_from_harmonix(url)
            audio.save(f"resources/harmonixset/dataset/audio/{name}.wav")
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue
        print(f"Saved {name}")

if __name__ == "__main__":
    main()
