import os
from fyp.audio.analysis import analyse_beat_transformer, analyse_chord_transformer, BeatAnalysisResult, ChordAnalysisResult
from fyp import Audio
from fyp import SongDataset
from fyp.audio.analysis import analyse_beat_transformer, analyse_chord_transformer, BeatAnalysisResult, ChordAnalysisResult
from fyp.audio.analysis.phrase import AudioEncoderModelConfig, HarmonicFeaturesExtractor
from dataclasses import dataclass, asdict
import torch
import torch.nn.functional as F
import random
import json
from typing import Literal
from fyp import DatasetEntry
from fyp.audio.manipulation import PitchShift
from fyp.audio.analysis.phrase import AudioEncoderModel
from fyp.audio.search import calculate_self_similarity_entry
import wandb
from tqdm.auto import tqdm
import gc
import tempfile
from itertools import chain

def cleanup():
    current_dir = tempfile.gettempdir()
    for filename in os.listdir(current_dir):
        if filename.endswith('.wav') or filename.endswith('.mp4') or filename.endswith('.mp3'):
            file_path = os.path.join(current_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                pass
    gc.collect()

@dataclass(frozen=True)
class TrainingConfig:
    # Augmentation parameters
    augment: bool = False # Whether to augment the dataset
    min_speedup: float = 0.9 # Minimum speedup factor
    max_speedup: float = 1.1 # Maximum speedup factor
    min_pitch_shift: int = 0 # Minimum pitch shift
    max_pitch_shift: int = 11 # Maximum pitch shift

    model_config: AudioEncoderModelConfig = AudioEncoderModelConfig()

    # Training parameters
    dataset_id: str = "./resources/dataset/audio-infos-v2.1.db"
    epochs: int = 1
    save_dir: str = "./resources/models"
    save_every: int = 100
    nbar_phrases: int = 8
    existing_weights: str | None = "./resources/models_0/phrase_model_7300.pt"

    def get_random_tranpose(self):
        return random.randint(self.min_pitch_shift, self.max_pitch_shift)

    def get_random_speedup(self):
        return random.uniform(self.min_speedup, self.max_speedup)

    def get_negative_phrase(self):
        return random.randint(1, self.nbar_phrases - 1)

class PhraseDataset:
    def __init__(self, entries: SongDataset, *, config: TrainingConfig, device: torch.device | None = None):
        idx_entries: list[DatasetEntry] = []
        for url_id, entry in entries._data.items():
            if len(entry.downbeats) < 12 or len(entry.downbeats) > 100: # Filter out songs with too many downbeats because my poor GPu can't handle it
                continue
            idx_entries.append(entry)
        self.data = idx_entries
        random.shuffle(self.data)
        self.config = config
        self.extractor = HarmonicFeaturesExtractor(device=device)

    def get_data(self, entry: DatasetEntry):
        # Augment by speedup and transposition
        audio = entry.get_audio()

        if self.config.augment:
            speedup = self.config.get_random_speedup()
            transpose = self.config.get_random_tranpose()
            audio = audio.change_speed(speedup)
            audio = audio.apply(PitchShift(transpose))
        else:
            speedup = 1.

        features = self.extractor(audio)
        sliced_features = []
        downbeat_slice_idxs = [int(downbeat / speedup * 10.8) for downbeat in entry.downbeats]
        nbars = self.config.nbar_phrases

        for i in range(len(entry.downbeats) - nbars):
            start_downbeat_idx = downbeat_slice_idxs[i]
            end_downbeat_idx = downbeat_slice_idxs[i + nbars]
            feat_ = features[start_downbeat_idx:end_downbeat_idx]
            sliced_features.append(feat_)

        anchor_valid_lengths = torch.tensor([len(x) for x in sliced_features]).to(self.extractor.device)

        max_length = int(anchor_valid_lengths.max().item())
        anchors = torch.stack([F.pad(x, (0, 0, 0, max_length - len(x))) for x in sliced_features])

        return entry, features, anchors, anchor_valid_lengths

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for idx in range(len(self)):
            entry = self.data[idx]
            try:
                yield self.get_data(entry)
            except Exception as e:
                print(f"Failed to process entry {entry.url_id}: {e}")

def get_save_path(config: TrainingConfig):
    save_path = f"{config.save_dir}_0"
    i = 1
    while os.path.exists(save_path):
        save_path = f"{config.save_dir}_{i}"
        i += 1
    os.makedirs(save_path)
    return save_path

def login_to_wandb():
    import dotenv
    dotenv.load_dotenv()
    wandb.login(key=os.environ["WANDB_API_KEY"])

def main():
    print("Loading data...")
    config = TrainingConfig()

    ds = SongDataset.load(config.dataset_id)

    print("Creating dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = PhraseDataset(ds, config=config, device=device)

    print("Creating model...")
    model = AudioEncoderModel(config=config.model_config)
    model = model.to(device)
    if config.existing_weights:
        sd = torch.load(config.existing_weights)
        model.load_state_dict(sd)

    print("Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    save_path = get_save_path(config)

    login_to_wandb()

    data_iter = chain(*[data for _ in range(config.epochs)])
    progress_bar = tqdm(desc="Training")
    run_cnt = 0

    with wandb.init(
        project = f"automasher-phrase-2",
        name = f"Trial {save_path.split('_')[-1]}",
        config=asdict(config)
    ) as run:
        for i, (entry, features, x, valid_lengths) in enumerate(data_iter):
            cleanup()
            ssim = torch.tensor(calculate_self_similarity_entry(entry, config.nbar_phrases) / 100, device = device, dtype = torch.float32)

            for j in range(valid_lengths.size(0)):
                y: torch.Tensor = model(x, valid_lengths)
                y = F.sigmoid(y)

                dists = F.cosine_similarity(y.unsqueeze(1), y.unsqueeze(0), dim=-1)

                loss = F.mse_loss(dists[j], ssim[j])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.update(1)

                run.log({
                    "Loss": loss.item()
                })

                run_cnt += 1

                if run_cnt % config.save_every == 0:
                    torch.save(model.state_dict(), os.path.join(save_path, f"phrase_model_{run_cnt}.pt"))

        torch.save(model.state_dict(), os.path.join(save_path, f"phrase_model_{run_cnt}.pt"))

if __name__ == "__main__":
    main()
