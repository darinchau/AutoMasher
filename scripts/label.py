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
from scripts.calculate import download_audio
from fyp.audio.dataset import SongDataset
from collections import defaultdict

class AudioKeyBinder:
    def __init__(self, win: tk.Tk, label: tk.Label, links: list[str]):
        self.win = win
        self.label = label
        self.play_time = -1.
        self.downloader = download_audio(links)
        self._audio = None
        self._url = None
        self.labels: dict[str, list[float]] = defaultdict(list)

    def update_text(self, text: str):
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
        self.update_text("Playing song")
        if self._audio is None or self._url is None:
            try:
                self._audio, self._url = next(self.downloader)
                while self._audio is None:
                    self._audio, self._url = next(self.downloader)
            except StopIteration:
                self.update_text("No more songs to play")
                return
        def audio_callback_fn(t: float):
            self.play_time = t
            if t == -1:
                self.update_text("Song stopped. Press SPACEBAR to start the next song or K to replay the song")
        self._audio.play(callback_fn=audio_callback_fn)


    def stop(self):
        if self._audio is not None:
            self._audio.stop()
            self.play_time = -1

    def __call__(self, event):
        c = event.char

        if not isinstance(c, str):
            return

        if not self.playing and c == " ":
            self._audio = self._url = None
            self.play()
            self.update_text(f"Playing started. Press SPACEBAR to perform labelling. \nPress L to skip this song. Press K to restart the labelling.")
            return

        if not self.playing and c == "K":
            if self._audio is not None:
                # TODO make audio thing with click track
                return
            self.update_text(f"No songs to replay. Press SPACEBAR to perform labelling.")
            return

        if not self.playing:
            return

        assert self._audio is not None and self._url is not None

        if c == " ":
            self.labels[self._url].append(self.play_time)
            self.update_text(f"Added label at t={round(self.play_time, 2)}")
            return

        if c == "L":
            # Immediately stop the current song and move on to the next. Delete the current labels
            self.stop()
            self._audio = None
            if self._url in self.labels:
                del self.labels[self._url]
            self._url = None
            self.play()
            return

        if c == "K":
            # Immediately stop the current song and restart it. Delete the current labels
            self.stop()
            if self._url in self.labels:
                del self.labels[self._url]
            self.play()
            return

        if c == "P" and self._url in self.labels:
            self.labels[self._url].pop()
            return


def main():
    win = tk.Tk()
    win.geometry("500x100")
    win.resizable(False, False)
    win.title("Dataset labeller")

    label = tk.Label(win)

    ds = SongDataset.load("./resources/dataset/audio-infos-v2.db")
    ds = ds.filter(lambda e: len(e.downbeats) > 32).filter(lambda e: e.length < 300)
    links = list(ds._data.keys())

    keybinder = AudioKeyBinder(win, label, links)

    win.bind("<Key>", keybinder)

    keybinder.update_text("Press SPACEBAR to start the labelling session")

    win.mainloop()

if __name__ == "__main__":
    main()
