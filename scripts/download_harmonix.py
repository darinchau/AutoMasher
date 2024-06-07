import os
from fyp import Audio
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_audio(urls: list[str]):
    def download_audio_single(url: str):
        audio = Audio.load(url)
        return audio

    # Downloads the things concurrently and yields them one by one
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_audio_single, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                audio = future.result()
                yield audio, url
            except Exception as e:
                print(f"Failed to download audio (skipping): {url}", e)
                yield None, url

# harmonix might contain something like "(needs to be 3.05% sped-up)"
# Use a regex to extract the percentage
def extract_speedup_percentage(harmonix: str):
    percentage = re.search(r"\(needs to be (-?\d+\.\d+)% sped-up\)", harmonix)
    if percentage is None:
        return 0., harmonix
    # Remove the bracketed part from the string
    harmonix = harmonix[:percentage.start()].strip()
    return float(percentage.group(1)), harmonix

def main():
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

    if not os.path.exists("resources/harmonixset/dataset/audio"):
        os.makedirs("resources/harmonixset/dataset/audio")

    speedups = {}
    for name, entry in entries.items():
        factor, url = extract_speedup_percentage(entry)
        speedups[url] = (factor, name)

    for audio, entry in tqdm(download_audio(list(speedups.keys())), total=len(entries)):
        if audio is None:
            continue
        factor, name = speedups[entry]
        audio.save(f"resources/harmonixset/dataset/audio/{name}.wav")

if __name__ == "__main__":
    main()
