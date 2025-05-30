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
import argparse
from dataclasses import dataclass, field
import datetime
from fyp.audio.dataset import DatasetEntry, SongDataset, create_entry
from fyp.util import (
    YouTubeURL,
)
from .make_v3_dataset import download_audio, REJECTED_URLS

LIST_SPLIT_SIZE = 300
MAX_ERRORS = 10
MAX_ERRORS_TIME = 20
RANDOM_WAIT_TIME_MIN = 15
RANDOM_WAIT_TIME_MAX = 60
REST_EVERY_N_VIDEOS = 999999999


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
            required=True,
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

        tqdm.write(f"Waiting for the next entry...")


def get_candidate_urls(ds: SongDataset) -> list[YouTubeURL]:
    datafiles = ds.list_urls("datafiles")
    audios = ds.list_urls("audio")
    rejected = ds.read_info_urls(REJECTED_URLS)
    result = [link for link in datafiles if link not in audios and link not in rejected]
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
