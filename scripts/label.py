# Runs a little widget that allows one to label the dataset
# python -m scripts.label
# - Press spacebar to start playing a song
# - If you don't like the song, skip it by pressing L
# - Restart the song by pressing K
# - Press P to remove the last (presumably faulty) label
# - At each valid 8-bar phrase starter, press spacebar to label it as such
# - After the song, press K to replay the song with beep sounds at each label points
# - Press spacebar to start the next song and automatically save the song to save_dir

import os
import sounddevice as sd
import numpy as np
import torch
import time
import tkinter as tk
import time
import torchaudio
from fyp import Audio
from fyp.audio.dataset import SongDataset
from fyp.audio.analysis import BeatAnalysisResult
from fyp.util.combine import get_video_id
from collections import defaultdict
from threading import Lock
import json

DATA_PATH = "resources/labels.json"
CACHE_PATH = "resources/cache"

class AudioKeyBinder:
    def __init__(self, win: tk.Tk, label: tk.Label, dataset: SongDataset):
        self.win = win
        self.label = label
        self.play_time = -1.
        self._audio = None
        self._url = None
        self.labels: dict[str, list[float]] = defaultdict(list)
        with open(DATA_PATH, "r") as f:
            self.labels.update(json.load(f))
        self._lock = Lock()
        self.dataset = dataset

    def update_text(self, text: str):
        print(text)
        self.label.config(text=text)

        self.win.update()
        window_width = self.win.winfo_width()
        window_height = self.win.winfo_height()
        label_width = self.label.winfo_reqwidth()
        label_height = self.label.winfo_reqheight()
        x = (window_width - label_width) // 2
        y = (window_height - label_height) // 2

        self.label.place(x=x, y=y)

    @property
    def playing(self):
        return self.play_time >= 0. and self._audio is not None and self._url is not None

    def play(self):
        # Gets the next song and plays it
        with self._lock:
            self.update_text("Preparing song")
            if self._url is None:
                for link in self.dataset.keys():
                    if link not in self.labels:
                        self._url = link
                        break
                else:
                    self.update_text("No more songs")
                    return

            self._audio = Audio.load(self._url, cache_path=os.path.join(CACHE_PATH, f"{get_video_id(self._url)}.wav"))
            self._audio = BeatAnalysisResult.from_data_entry(self.dataset[self._url]).make_click_track(self._audio)

            def audio_callback_fn(t: float):
                self.play_time = t

            def stop_callback_fn():
                self.play_time = -1
                with open(DATA_PATH, "w") as f:
                    json.dump(self.labels, f)
                self.update_text("Song stopped. Press SPACEBAR to start the next song or K to replay the song")
            self._audio.play(callback_fn=audio_callback_fn, stop_callback_fn=stop_callback_fn)


    def stop(self):
        if self._audio is not None:
            self._audio._stop_audio = True
            self.play_time = -1

    def __call__(self, event):
        c = event.char

        if self._lock.locked():
            return

        if not isinstance(c, str):
            return

        if not self.playing and c == " ":
            print("Spacebar pressed while not playing")
            self._audio = self._url = None
            self.play()
            self.update_text(f"Playing started. Press SPACEBAR to perform labelling. \nPress L to skip this song. Press K to restart the labelling.")
            return

        if not self.playing and c == "k":
            print("K pressed while not playing")
            if self._audio is not None:
                # TODO make audio thing with click track
                return
            self.update_text(f"No songs to replay. Press SPACEBAR to perform labelling.")
            return

        if not self.playing:
            return

        assert self._audio is not None and self._url is not None

        if c == " ":
            print("Spacebar pressed while playing")
            self.labels[self._url].append(self.play_time)
            self.update_text(f"Added label at t={round(self.play_time, 2)}")
            return

        if c == "l":
            # Immediately stop the current song and move on to the next. Delete the current labels
            print("L pressed while playing")
            self.stop()
            print("Song stopped.")
            self._audio = None
            self.labels[self._url].append(-1)
            self._url = None
            self.update_text("Song skipped. Preparing next song.")
            self.play()
            return

        if c == "k":
            # Immediately stop the current song and restart it. Delete the current labels
            print("K pressed while playing")
            self.stop()
            if self._url in self.labels:
                del self.labels[self._url]
            return

        if c == "p" and self._url in self.labels:
            self.labels[self._url].pop()
            return

        print(f"Unrecognized key: {c}")


def main():
    win = tk.Tk()
    win.geometry("500x100")
    win.resizable(False, False)
    win.title("Dataset labeller")

    label = tk.Label(win)

    ds = SongDataset.load("./resources/dataset/audio-infos-v2.1.db")
    ds = ds.filter(lambda e: len(e.downbeats) > 32).filter(lambda e: e.length < 300)

    keybinder = AudioKeyBinder(win, label, ds)

    win.bind("<Key>", keybinder)

    keybinder.update_text("Press SPACEBAR to start the labelling session")

    def on_close():
        keybinder.update_text("Window is being closed")
        if keybinder.playing:
            keybinder.stop()
            keybinder.update_text("Song stopped.")
        if keybinder._audio is not None:
            del keybinder._audio
            keybinder._audio = None
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)

    win.mainloop()

    with open(DATA_PATH, "w") as f:
        json.dump(keybinder.labels, f)

if __name__ == "__main__":
    main()
