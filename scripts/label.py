# Runs a little widget that allows one to label the dataset
# python -m scripts.label
# - Press spacebar to start playing a song
# - If you don't like the song, skip it by pressing L
# - Restart the song by pressing K
# - Pause and come back to it later by pressing J (Same as K but with no auto restart)
# - Press N to remove the last (presumably faulty) label
# - Press M to remove the last (presumably faulty) label and label the current time
# - Press H twice to stkip straight to the next song, keeping the current labels
#       This should be used when you notice that the metronome beats is screwed up anyways
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
from fyp.audio.dataset import SongDataset, SongGenre
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
        self.dataset = dataset
        self.playback_playing = False

        self.labels: dict[str, list[float]] = defaultdict(list)
        if os.path.isfile(DATA_PATH):
            with open(DATA_PATH, "r") as f:
                self.labels.update(json.load(f))

        self._audio = None
        self._url = None
        self._playback_audio = None
        self._lock = Lock()
        self._last_h_press = -1
        self._calibrate_timestamps: list[float] = []

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
        return self.play_time >= 0. and self._audio is not None

    @property
    def calibrating(self):
        return self._calibrate_timestamps and self.play_time >= 0. and self._audio is not None

    def save(self):
        with open(DATA_PATH, "w") as f:
            json.dump(self.labels, f, indent=4)

    def play(self):
        # Gets the next song and plays it
        with self._lock:
            self.update_text(f"Preparing song")
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
            self.update_text(f"Song loaded: {self._url}")

            def audio_callback_fn(t: float):
                self.play_time = t

            def stop_callback_fn():
                self.play_time = -1
                self.update_text("Song stopped. Press SPACEBAR to start the next song or K to replay the song")
            self._audio.play(callback_fn=audio_callback_fn, stop_callback_fn=stop_callback_fn)

    def playback(self):
        # Creates the playback with clicktracks. This will do nothing if there is no song
        if self._audio is None:
            return

        if self._url is None:
            return

        with self._lock:
            bt = BeatAnalysisResult.from_data(self._audio.get_duration(), beats = [], downbeats = self.labels[self._url])
            self._playback_audio = bt.make_click_track(self._audio)
            self.update_text("Playback started. Press SPACEBAR to stop playback")
            self.playback_playing = True

            def stop_callback_fn():
                self.playback_playing = False
                self.update_text("Playback stopped. Press SPACEBAR to start the next song or K to replay the song")
            self._playback_audio.play(stop_callback_fn=stop_callback_fn)

    def stop(self):
        if self._audio is not None:
            self._audio._stop_audio = True
            self.play_time = -1

        if self._playback_audio is not None:
            self._playback_audio._stop_audio = True

    def calibrate(self):
        # Start the calibrate routine
        import torch

        calibration_duration_seconds = 20
        num_ticks = int((calibration_duration_seconds+2)/4)
        with self._lock:
            self._calibrate_timestamps = [-1.]
            self._audio = Audio(torch.zeros(2, 44100 * calibration_duration_seconds), 44100)
            self._audio = BeatAnalysisResult.from_data(
                duration = calibration_duration_seconds,
                beats = list(range(0, calibration_duration_seconds)),
                downbeats = list(range(2, calibration_duration_seconds, 4))
            ).make_click_track(self._audio)

            def audio_callback_fn(t: float):
                self.play_time = t

            def stop_callback_fn():
                self.play_time = -1
                num_calibrations = len(self._calibrate_timestamps) - 1
                if num_calibrations != num_ticks:
                    self.update_text("Calibration stopped. Wrong number of downbeats detected")
                    return
                calibration_time = sum([
                    self._calibrate_timestamps[i + 1] - 2 - 4 * i for i in range(num_calibrations)
                ]) / num_calibrations
                self.update_text(f"Calibration stopped. Time = {calibration_time}")
                self._calibrate_timestamps = []
            self._audio.play(callback_fn=audio_callback_fn, stop_callback_fn=stop_callback_fn)

    def __call__(self, event):
        c = event.char

        if self._lock.locked():
            return

        if not isinstance(c, str):
            return

        if self.playback_playing and c == " ":
            print("Spacebar pressed while playing back")
            self.stop()
            return

        if self.playback_playing:
            return

        if not self.playing and c == " ":
            print("Spacebar pressed while not playing")
            self._audio = self._url = None
            self.play()
            self.update_text(f"Playing started. Press SPACEBAR to perform labelling. \nPress L to skip this song. Press K to restart the labelling.")
            return

        if not self.playing and c == "c":
            print("Spacebar pressed while not playing")
            self._audio = self._url = None
            self.calibrate()
            self.update_text(f"Calibration started. Press SPACEBAR at each downbeat. Press K to restart the labelling.")
            return

        if not self.playing and c == "k":
            print("K pressed while not playing")
            if self._url is not None and self._audio is not None:
                self.playback()
                return
            self.update_text(f"No songs to replay. Press SPACEBAR to perform labelling.")
            return

        if not self.playing:
            return

        if self.calibrating and c == " ":
            print("Spacebar pressed while calibrating")
            self._calibrate_timestamps.append(self.play_time)
            return


        assert self._audio is not None and self._url is not None

        if c == " ":
            print("Spacebar pressed while playing")
            assert self._url is not None
            self.labels[self._url].append(self.play_time)
            self.update_text(f"Added label at t={round(self.play_time, 2)}")
            return

        if c in ("j", "k", "l"):
            # Immediately stop the current song and restart it. Delete the current labels
            print(f"{c.upper()} pressed while playing")
            self.stop()
            if c == "l":
                self._audio = None
                self.labels[self._url].append(-1)
                self._url = None
            elif self._url in self.labels:
                del self.labels[self._url]
            self.save()
            if c in ("k", "l"):
                self.play()
            return

        if c in ("m", "n") and self._url in self.labels:
            print(f"{c.upper()} pressed while playing")
            try:
                self.labels[self._url].pop()
            except IndexError:
                pass
            if c == "m":
                self.labels[self._url].append(self.play_time)
                self.update_text(f"Deleted last label and added label at t={round(self.play_time, 2)}")
            else:
                print("Deleted last label")
            return

        if c == "h":
            print("H pressed while playing")
            t = time.time()
            if self._last_h_press < 0 or t - self._last_h_press > 1:
                self._last_h_press = t
                return

            self.stop()
            print("Song stopped.")
            self._audio = None
            self._url = None
            self.save()
            self.update_text("Song skipped. Preparing next song.")
            self.play()
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
    print(ds)

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

    keybinder.save()

if __name__ == "__main__":
    main()
