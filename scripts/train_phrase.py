# Trains the Phrase model

from fyp import SongDataset
from typing import Literal
import torch
import numpy as np
from numpy.typing import NDArray
from torch import nn
from fyp.audio.analysis.phrase import get_input, PhraseModel
from fyp.audio.analysis import ChordAnalysisResult
from torch.utils.data import DataLoader, Dataset
import random
import torch.utils
import wandb
import os
import torch.nn.functional as F
from tqdm.auto import tqdm
import json
import dotenv

dotenv.load_dotenv()

class PhraseDataset(Dataset):
    def __init__(self, entries: SongDataset, *, nbar_phrases: int = 8, n_chords: int = 10):
        idx_entries = []
        for url_id, entry in entries._data.items():
            if len(entry.downbeats) < 12:
                continue
            for bar_number in range(len(entry.downbeats) - nbar_phrases):
                idx_entries.append((url_id, bar_number))
        self.data = idx_entries
        self.song_dataset = entries
        self.nbar_phrases = nbar_phrases
        self.n_chords = n_chords

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        url_id, bar_number = self.data[idx]
        entry = self.song_dataset[url_id]
        ct = ChordAnalysisResult.from_data(
            duration = len(entry.downbeats),
            labels = entry.chords,
            times = entry.normalized_chord_times
        )
        anchor_vector = get_input(ct, slice_time=bar_number, n=self.n_chords, return_format="pt")

        # Get a positive example. Idea: Chord transpositions should belong to the same class because they are the same progression
        transposition = random.randint(1, 11)
        positive_vector = get_input(ct.transpose(transposition), slice_time=bar_number, n=self.n_chords, return_format="pt")

        # Get a negative example. Idea: Bar X and X+1/X+2/X+3 will not be both valid phrase starting points, because we assume 8 bar phrases
        # Let's say X+4 might be so let's avoid picking X+4
        valid_choices = [1, 2, 3, -1, -2, -3]
        valid_choices = [x for x in valid_choices if 0 <= bar_number + x < len(entry.downbeats) - self.nbar_phrases]
        shift = random.choice(valid_choices)
        transposition = random.randint(1, 11)
        negative_vector = get_input(ct.transpose(transposition), slice_time=bar_number + shift, n=self.n_chords, return_format="pt")

        return anchor_vector, positive_vector, negative_vector

CONFIG = {
    "dataset_id": "./resources/dataset/audio-infos-v2.db",
    "hidden_sizes": [300, 256, 256],
    "output_size": 256,
    "dropouts": [0.1, 0.1, 0.1],
    "margin": 1,
    "epochs": 20,
    "save_dir": "./resources/models",
    "log_every": 50,
    "save_every": 10000,
    "use_wandb": True,
    "nbar_phrases": 8,
    "n_chords": 20,
    "distance_function": "cosine"
}

def main():
    ds = SongDataset.load(CONFIG["dataset_id"])
    ds = ds.filter(lambda e: len(e.downbeats) > 32)

    model = PhraseModel(CONFIG)
    dataset = PhraseDataset(ds, nbar_phrases=CONFIG["nbar_phrases"], n_chords=CONFIG["n_chords"])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    save_path = CONFIG["save_dir"]
    i = 0
    while os.path.exists(save_path):
        save_path = f"{CONFIG['save_dir']}_{i}"
        i += 1
    os.makedirs(save_path)

    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(CONFIG, f)

    if CONFIG["use_wandb"]:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        with wandb.init(
            project = f"automasher-phrase",
            name = f"Trial {i + 1}",
            config = CONFIG
        ) as run:
            train_model(model, dataloader, optimizer, log_every=CONFIG["log_every"], save_every=CONFIG["save_every"], epochs=CONFIG["epochs"], save_path=save_path, run=run)
    else:
        train_model(model, dataloader, optimizer, log_every=CONFIG["log_every"], save_every=CONFIG["save_every"], epochs=CONFIG["epochs"], save_path=save_path, run=None)



def train_model(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, *,
                epochs: int, log_every: int, save_every: int, save_path: str, run):
    model = model.cuda()
    losses = []
    samples = 0
    for epoch in range(epochs):
        for anchor, positive, negative in tqdm(dataloader, total=len(dataloader), desc="Training"):
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()
            optimizer.zero_grad()
            loss = model(anchor, positive, negative)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            samples += 1

            if len(losses) % log_every == 0:
                if run is not None:
                    run.log({"loss": np.mean(losses)})
                losses.clear()

            if samples % save_every == 0 and samples > 0:
                torch.save(model.state_dict(), os.path.join(save_path, f"phrase_model_{samples}.pt"))

        torch.save(model.state_dict(), os.path.join(save_path, f"phrase_model_{samples}.pt"))

if __name__ == "__main__":
    main()
