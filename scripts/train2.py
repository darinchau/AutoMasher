# Trains the Phrase model

from fyp import SongDataset, DatasetEntry
from typing import Literal
import torch
import numpy as np
from numpy.typing import NDArray
from torch import nn
from fyp.audio.analysis.phrase import AudioEncoderModel, AudioEncoderModelConfig, preprocess, HarmonicFeaturesExtractor
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
from dataclasses import dataclass, asdict

dotenv.load_dotenv()

@dataclass
class TrainingConfig:
    dataset_id: str = "./resources/dataset/audio-infos-v2.1.db"
    labels_path: str = "./resources/labels.json"
    initial_model_path: str = "./resources/models_5/phrase_model_34500.pt"
    freeze_backbone: bool = True
    epochs: int = 100
    save_dir: str = "./resources/models"
    log_every: int = 50
    save_every: int = 1000
    use_wandb: bool = True
    calibration: float = 0.21678004535147374
    calibrated_tolerance: float = 0.5
    nbar_phrases: int = 8

    model_config: AudioEncoderModelConfig = AudioEncoderModelConfig()

class PhraseDataset(Dataset):
    def __init__(self, config: TrainingConfig, entries: SongDataset, device: torch.device):
        self.song_dataset = entries
        self.nbar_phrases = config.nbar_phrases

        with open(config.labels_path, "r") as f:
            labels: dict[str, list[float]] = json.load(f)

        self.labels = []
        for url, label in labels.items():
            if not label or label[-1] == -1:
                continue

            label = np.array(label) - config.calibration
            label = label[label >= 0]
            downbeats = BeatAnalysisResult.from_data_entry(entries[url]).downbeats
            label_downbeats_idxs = [np.argmin(np.abs(downbeats - x)) for x in label]
            label_downbeats_diff = np.abs(label - downbeats[label_downbeats_idxs])
            label = [(url, x) for x, t in zip (label_downbeats_idxs, label_downbeats_diff) if t < config.calibrated_tolerance]

            self.labels.extend(label)

        self.extractor = HarmonicFeaturesExtractor(device = device)
        self.config = config

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            url, downbeat = self.labels[idx]
            entry = self.song_dataset[url]
            features, anchors, anchor_valid_lengths = preprocess(
                extractor=self.extractor,
                entry=entry,
                audio=entry.get_audio(),
                augment=False,
                nbars=self.config.nbar_phrases,
            )
            if random.random() > 0.5:
                # Make a negative sample
                choices = [1, 2, 3, 4, 5, 6, 7] if downbeat == 0 else [-1, -1, 1, 2, 3, 4, 5, 6, 7]
                downbeat = random.choice(choices)
                return anchors[downbeat], anchor_valid_lengths[downbeat], 0

            return anchors[downbeat], anchor_valid_lengths[downbeat], 1

        except Exception as e:
            print(f"Failed to get data for idx={idx}: {e}")
            r = random.randrange(0, len(self))
            return self.__getitem__(r)

class PhraseModel(nn.Module):
    def __init__(self, model: nn.Module, model_output_dims: int):
        super().__init__()
        self.model = model
        self.aux = nn.Linear(model_output_dims, 1)

    def forward(self, x, lengths):
        x = self.model(x, lengths)
        x = F.normalize(x)
        x = self.aux(x)
        x = F.sigmoid(x)
        return x


def collate_fn(tensors):
    lengths = torch.stack([t[1] for t in tensors])
    max_len = max([t[0].size(0) for t in tensors])
    x = torch.stack([F.pad(t[0], (0, 0, 0, max_len - t[0].size(0))) for t in tensors])
    labels = torch.tensor([t[2] for t in tensors])
    return x, lengths, labels

def get_save_path(config: TrainingConfig):
    import json

    save_path = f"{config.save_dir}_0"
    i = 1
    while os.path.exists(save_path):
        save_path = f"{config.save_dir}_{i}"
        i += 1
    os.makedirs(save_path)

    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(asdict(config), f)
    return save_path

def load_song_dataset(config: TrainingConfig):
    import numpy as np

    ds = SongDataset.load(config.dataset_id)

    def filter_func(x: DatasetEntry):
        if not (12 < len(x.downbeats) < 100):
            return False

        db = np.array(x.downbeats)
        db_diff = np.diff(db)
        mean_diff = db_diff / np.mean(db_diff)
        if not np.all((0.9 < mean_diff) & (mean_diff < 1.1)):
            return False

        if not x.cached:
            return False

        # try:
        #     x.get_audio()
        # except Exception as e:
        #     return False

        return True

    ds.filter(filter_func)
    return ds

def main():
    print("Loading dataset...")
    config = TrainingConfig()
    ds = load_song_dataset(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sanity Check
    print("Creating model...")
    model_path = config.initial_model_path
    with open(os.path.dirname(model_path) + "/config.json", "r") as f:
        model_config = AudioEncoderModelConfig(**json.load(f)["model_config"])

    config.model_config = model_config

    model = AudioEncoderModel(model_config)
    sd = torch.load(model_path)
    model.load_state_dict(sd)

    model = PhraseModel(model, config.model_config.encoder_embed_dim)

    model = model.to(device)

    dataset = PhraseDataset(config, ds, device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(len(dataset), " labels")

    if config.freeze_backbone:
        print("Freezing backbone...")
        for param in model.model.parameters():
            param.requires_grad = False

    save_path = get_save_path(config)

    model = model.cuda()
    losses = []
    samples = 0
    for epoch in trange(config.epochs):
        for sample, lengths, label in dataloader:
            sample = sample.cuda()
            label = label.cuda()
            lengths = lengths.cuda()

            optimizer.zero_grad()
            y = model(sample, lengths).squeeze()
            loss = F.binary_cross_entropy(y, label.float())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            samples += 1

            if len(losses) % config.log_every == 0:
                print(sum(losses) / len(losses))
                losses.clear()

            if samples % config.save_every == 0 and samples > 0:
                torch.save(model.state_dict(), os.path.join(save_path, f"phrase_model_{samples}.pt"))

        torch.save(model.state_dict(), os.path.join(save_path, f"phrase_model_{samples}.pt"))

if __name__ == "__main__":
    main()
