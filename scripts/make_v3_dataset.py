# This script consolidates the v3 dataset.
# python -m scripts.make_v3_dataset from the root directory

import os
import numpy as np
import base64
import zipfile
from math import isclose
from typing import Literal
import time
import traceback
from tqdm.auto import tqdm, trange
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import argparse
import shutil
from dataclasses import dataclass, field
import random
import datetime
from PIL import Image
from threading import Thread

try:
    from pytubefix import Playlist, YouTube, Channel
except ImportError:
    try:
        from pytube import Playlist, YouTube, Channel  # type: ignore
    except ImportError:
        raise ImportError("Please install the pytube library to download the audio. You can install it using `pip install pytube` or `pip install pytubefix`")

from fyp.audio.dataset import DatasetEntry, SongDataset, create_entry
from fyp import Audio
from fyp.audio.analysis import BeatAnalysisResult, DeadBeatKernel
from fyp.util import (
    clear_cuda,
    YouTubeURL,
    download_audio as download_audio_inner,
    DownloadError,
    Colors
)
from fyp.constants import (
    CANDIDATE_URLS,
    PROCESSED_URLS,
    REJECTED_URLS,
)
from itertools import zip_longest
import json

LIST_SPLIT_SIZE = 300
MAX_ERRORS = 10
MAX_ERRORS_TIME = 20
RANDOM_WAIT_TIME_MIN = 15
RANDOM_WAIT_TIME_MAX = 60
REST_EVERY_N_VIDEOS = 1200


class FatalError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    root: str
    port: int | None = None

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(description="Config for dataset creation")
        parser.add_argument(
            "--root",
            type=str,
            help="Root directory for the dataset",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=None,
            help="Port to use for downloading audio. If not specified, the default port will be used.",
        )
        args = parser.parse_args()
        return cls(
            root=args.root,
            port=args.port,
        )


def download_audio(ds: SongDataset, urls: list[YouTubeURL], port: int | None = None, antiban: bool = False):
    """Downloads the audio from the URLs. Yields the path to audio (in the dataset) and the URL."""
    def download_audio_single(url: YouTubeURL) -> str:
        if ds.has_path("audio", url):
            return ds.get_path("audio", url)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio_path = download_audio_inner(url, temp_dir, port=port)
            audio_destination = os.path.join(ds.root, "audio")
            audio = os.path.join(audio_destination, os.path.basename(temp_audio_path))
            shutil.copy2(temp_audio_path, audio)
        if antiban:
            # Wait for a random amount of time to avoid getting blacklisted
            time.sleep(random.uniform(RANDOM_WAIT_TIME_MIN, RANDOM_WAIT_TIME_MAX))
        ds.set_path("audio", url, audio)
        return audio

    # Downloads the things concurrently and yields them one by one
    # If more than MAX_ERRORS fails in MAX_ERRORS_TIME seconds, then we assume YT has blacklisted our IP or our internet is down or smth and we stop
    error_logs = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(download_audio_single, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                audio = future.result()
                tqdm.write(f"Downloaded audio: {url}")
                yield audio, url
            except Exception as e:
                if isinstance(e, DownloadError):
                    if "Video not downloadable" in str(e):
                        ds.write_info(REJECTED_URLS, url)
                        tqdm.write(f"{Colors.UNDERLINE}Rejected URL{Colors.END}: {str(e)}")
                        continue
                    if "Tunnel connection failed: 502 Proxy Error" in str(e):
                        raise FatalError(f"Proxy error: {url}. 502 Proxy Error") from e
                tqdm.write(f"{Colors.RED}Failed to download audio: {url}{Colors.END}")
                tqdm.write(f"Error: {e}")
                ds.write_error(f"Failed to download audio: {url}", e, print_fn=tqdm.write)
                error_logs.append((time.time(), e))
                while len(error_logs) >= MAX_ERRORS:
                    if time.time() - error_logs[0][0] < MAX_ERRORS_TIME:
                        tqdm.write(f"Too many errors in a short time, has YouTube blacklisted us?")
                        for t, e in error_logs:
                            tqdm.write(f"Error ({t}): {e}")
                            tqdm.write("=" * os.get_terminal_size().columns)
                        # Stop all the other downloads
                        for future in futures:
                            future.cancel()
                        raise FatalError(f"Too many errors in a short time, has YouTube blacklisted us?")
                    error_logs.pop(0)


def process_batch(ds: SongDataset, urls: list[YouTubeURL], port: int | None = None):
    audios = download_audio(ds, urls, port=port)
    t = time.time()
    last_t = None

    for i, (audio_path, url) in tqdm(enumerate(audios), total=len(urls)):
        if audio_path is None:
            continue

        last_entry_process_time = round(time.time() - last_t, 2) if last_t else None
        last_t = time.time()
        tqdm.write("")
        tqdm.write("\u2500" * os.get_terminal_size().columns)
        tqdm.write(f"Current time: {datetime.datetime.now()}")
        tqdm.write("Recalculating entries")
        tqdm.write(f"Last entry process time: {last_entry_process_time} seconds")
        tqdm.write(f"Current entry: {url}")
        tqdm.write(f"Time elapsed: {round(time.time() - t, 2)} seconds")
        tqdm.write(f"Entries processed: {i + 1} / {len(urls)}")
        tqdm.write(f"Number of entries in dataset: {len(ds.list_files('audio'))}")
        tqdm.write("\u2500" * os.get_terminal_size().columns)
        tqdm.write("")

        clear_cuda()

        audio = Audio.load(audio_path)

        try:
            dataset_entry = create_entry(
                url=url,
                dataset=ds,
                audio=audio,
                beat_backend="spleeter",
                beat_backend_url="http://localhost:8123",
                use_beat_cache=False,
                use_chord_cache=False,
                source="laion-12m"
            )
        except DeadBeatKernel as e:
            raise FatalError(f"Beat kernel is unresponsive: {url}") from e
        except Exception as e:
            ds.write_error(f"Failed to create entry: {url}", e, print_fn=tqdm.write)
            ds.write_info(REJECTED_URLS, url)
            tqdm.write(f"Failed to create entry: {url}")
            continue

        tqdm.write(f"Writing entry to {ds.get_path('datafiles', url)}")

        try:
            dataset_entry.save(ds.get_path("datafiles", url))
            ds.write_info(PROCESSED_URLS, url)
        except Exception as e:
            ds.write_error(f"Failed to write entry: {url}", e, print_fn=tqdm.write)
            ds.write_info(REJECTED_URLS, url)
            tqdm.write(f"Failed to write entry: {url}")
            continue
        tqdm.write(f"Waiting for the next entry...")


def get_candidate_urls(ds: SongDataset) -> list[YouTubeURL]:
    # TODO remove this - temporarily only use the audios that we already have around
    candidates = ds.read_info(CANDIDATE_URLS)
    assert candidates is not None
    finished = ds.read_info_urls(PROCESSED_URLS) | ds.read_info_urls(REJECTED_URLS)
    metadata = ds.get_path("metadata")
    candidates = [c for c in candidates if c not in finished]

    # Sort by views, but intersperse them according to language info
    with open(metadata, "r") as f:
        metadatas = json.load(f)
    language: dict[str, list[YouTubeURL]] = {
        "en": [],
        "ja": [],
        "ko": [],
        "zh": [],
    }
    for c in candidates:
        try:
            lang = metadatas[c.video_id]["language"]
            language[lang].append(c)
        except KeyError:
            continue

    for lang in language:
        language[lang].sort(key=lambda x: metadatas[x.video_id]["views"], reverse=True)

    result: list[YouTubeURL] = []
    for items in zip_longest(
        language["en"],
        language["ja"],
        language["ko"],
        language["zh"],
        fillvalue=None
    ):
        # Filter out None values and extend the result list
        result.extend(item for item in items if item is not None)

    return result


def main(config: Config | None = None):
    """Packs the audio-infos-v3 dataset into a single, compressed dataset file."""
    if config is None:
        config = Config.parse_args()

    ds = SongDataset(config.root, load_on_the_fly=True)

    candidate_urls = get_candidate_urls(ds)
    print(f"Loading dataset from {ds} ({len(candidate_urls)} candidate entries)")
    process_bar = tqdm(desc="Processing candidates", total=len(candidate_urls))
    n_videos_before_rest = REST_EVERY_N_VIDEOS

    while True:
        # Get the dict with the first LIST_SPLIT_SIZE elements sorted by key
        url_batch = sorted(candidate_urls, key=lambda x: x[0])[:LIST_SPLIT_SIZE]
        if not url_batch:
            break
        try:
            process_batch(ds, url_batch, config.port)
        except FatalError as e:
            print(e)
            break
        except Exception as e:
            ds.write_error("Failed to process batch", e)
            traceback.print_exc()

        nbefore = len(candidate_urls)
        candidate_urls = get_candidate_urls(ds)
        nafter = len(candidate_urls)
        process_bar.update(nbefore - nafter)

        n_videos_before_rest -= nbefore - nafter
        if n_videos_before_rest <= 0:
            print("Taking a rest")
            for _ in trange(60 * 60 * 5, desc="Resting..."):
                time.sleep(1)
            n_videos_before_rest = REST_EVERY_N_VIDEOS


if __name__ == "__main__":
    main()
