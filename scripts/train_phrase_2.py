# Trains the Phrase model

from fyp import SongDataset
from typing import Literal
import torch
import numpy as np
from numpy.typing import NDArray
from torch import nn
from fyp.audio.analysis.phrase import get_input, PhraseModel, PhraseModelAdapter
from fyp.audio.analysis import ChordAnalysisResult, BeatAnalysisResult
from torch.utils.data import DataLoader, Dataset
import random
import torch.utils
import wandb
import os
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
import json
import dotenv

dotenv.load_dotenv()

class PhraseDataset(Dataset):
    def __init__(self, config: dict, entries: SongDataset):
        self.song_dataset = entries
        self.nbar_phrases = config["nbar_phrases"]
        self.n_chords = config["n_chords"]

        with open(config["labels_path"], "r") as f:
            labels = json.load(f)

        self.labels = []
        for url, label in labels.items():
            if not label or label[-1] == -1:
                continue

            label = np.array(label) - config['calibration']
            label = label[label >= 0]
            downbeats = BeatAnalysisResult.from_data_entry(entries[url]).downbeats
            label_downbeats = [np.argmin(np.abs(downbeats - x)) for x in label]
            label_downbeats_time = np.abs(label - downbeats[label_downbeats])
            label = [(url, x) for x, t in zip (label, label_downbeats_time) if t < config["calibrated_tolerance"]]

            self.labels.extend(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        url, downbeat = self.labels[idx]
        entry = self.song_dataset[url]
        ct = ChordAnalysisResult.from_data_entry(entry)
        if random.random() > 0.5:
            # Positive sample
            transpose = random.randint(0, 11)
            ct = ct.transpose(transpose)
            return get_input(ct, downbeat, n = self.n_chords, return_format="pt"), 1

        # Negative sample
        downbeat = random.randint(1, self.nbar_phrases - 1) + downbeat
        return get_input(ct, downbeat, n = self.n_chords, return_format="pt"), 0

CONFIG = {
    "dataset_id": "./resources/dataset/audio-infos-v2.1.db",
    "labels_path": "./resources/labels.json",
    "initial_model_path": "./resources/models_0/phrase_model_507900.pt",
    "freeze_backbone": True,
    "epochs": 10000,
    "save_dir": "./resources/models",
    "log_every": 50,
    "save_every": 1000,
    "use_wandb": True,
    "calibration": 0.21678004535147374,
    "calibrated_tolerance": 0.5,
    "nbar_phrases": 8,
    "n_chords": 20
}

def main():
    ds = SongDataset.load(CONFIG["dataset_id"])
    ds = ds.filter(lambda e: len(e.downbeats) > 32)

    # Sanity Check
    with open(os.path.join(os.path.dirname(CONFIG["initial_model_path"]), "config.json"), "r") as f:
        initial_config = json.load(f)
    assert initial_config["n_chords"] == CONFIG["n_chords"]
    assert initial_config["nbar_phrases"] == CONFIG["nbar_phrases"]

    model = PhraseModel(initial_config)
    sd = torch.load(CONFIG["initial_model_path"])
    model.load_state_dict(sd)
    model = PhraseModelAdapter(model)

    dataset = PhraseDataset(CONFIG, ds)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(len(dataset), " labels")

    if CONFIG["freeze_backbone"]:
        for param in model.model.parameters():
            param.requires_grad = False

    save_path = CONFIG["save_dir"]
    i = 0
    while os.path.exists(save_path):
        save_path = f"{CONFIG['save_dir']}_{i}"
        i += 1
    os.makedirs(save_path)
    print(f"Saving to {save_path}")

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
    for epoch in trange(epochs):
        for sample, label in dataloader:
            sample = sample.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            y = model(sample).squeeze()
            loss = F.binary_cross_entropy(y, label.float())
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
