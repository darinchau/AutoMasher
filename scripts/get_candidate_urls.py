import os
import sys
from datasets import load_dataset, load_from_disk
from langdetect import detect, DetectorFactory, LangDetectException
from langdetect import detector_factory
from tqdm.auto import tqdm
import json

from fyp import SongDataset, get_url
from fyp.constants import CANDIDATE_URLS


def get_views(x):
    s = x["views"]
    for key in ["views", "plays", "play"]:
        s = s.replace(key, "").strip()
    try:
        x["views"] = float(s)
        return x
    except Exception as e:
        pass
    try:
        if s[-1] == "K":
            x["views"] = float(s[:-1]) * 1e3
        if s[-1] == "M":
            x["views"] = float(s[:-1]) * 1e6
        if s[-1] == "B":
            x["views"] = float(s[:-1]) * 1e9
        return x
    except Exception as e:
        pass
    print(f"Cannot map : {x['views']}")
    x["views"] = -1
    return x


DetectorFactory.seed = 0

mapping = {
    "zh-cn": "zh",
    "zh-tw": "zh",
}


def detect_language(x):
    try:
        language = detect(x["title"])
        x["language"] = mapping.get(language, language)
    except LangDetectException:
        x["language"] = "unk"  # unknown
    return x


keys_to_write = ['title', 'artist_names', 'album_name', 'views', "language"]


def main(path: str):
    song_ds = SongDataset(path)
    song_ds.register("metadata", "metadata.json", initial_data="{}")

    if not os.path.exists(os.path.join(path, "filtered_ds")):
        ds = load_dataset("laion/LAION-DISCO-12M")
        ds = ds["train"].map(get_views, num_proc=8)  # type: ignore
        # filtered_ds = filtered_ds.filter(lambda x: x["views"] > 500000, num_proc=8)
        ds = ds.filter(lambda x: x["isExplicit"] is False, num_proc=8)
        ds = ds.filter(lambda x: x["duration"] < 600 and x["duration"] > 120, num_proc=8)
        ds = ds.filter(lambda x: len(x["artist_ids"]) > 0, num_proc=8)
        ds = ds.map(detect_language)
        # ds = ds.filter(lambda x: x["language"] in ["en", "ja", "ko", "zh"], num_proc=8)
        ds = ds.remove_columns(["isExplicit"])

        ds.save_to_disk(os.path.join(path, "filtered_ds"))
    else:
        ds = load_from_disk(os.path.join(path, "filtered_ds"))

    urls = []
    metadatas = {}

    for entry in tqdm(ds, total=len(ds), ncols=75):
        try:
            url = get_url(entry["song_id"])  # type: ignore
        except Exception as e:
            print(f"Cannot get url for {entry}")
            continue
        urls.append(url.video_id)
        metadata = {k: entry[k] for k in keys_to_write}  # type: ignore
        metadatas[url.video_id] = metadata

    with open(song_ds.get_path("info"), "r") as f:
        info = json.load(f)

    info[CANDIDATE_URLS] = urls

    with open(song_ds.get_path("info"), "w") as f:
        json.dump(info, f)

    with open(song_ds.get_path("metadata"), "w") as f:
        json.dump(metadatas, f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.get_candidate_urls <path_to_dataset>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Making directory {path}")
    main(path)
