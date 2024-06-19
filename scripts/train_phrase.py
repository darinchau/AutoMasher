# Trains the Phrase model

from fyp import SongDataset, DatasetEntry
from fyp.util.note import get_chord_notes, get_idx2voca_chord, get_chord_note_inv, get_inv_voca_map, notes_to_idx, idx_to_notes
from typing import Literal
import torch
import numpy as np
from fyp.audio.search.search import calculate_self_similarity, calculate_self_similarity_beat
from fyp.audio.analysis import ChordAnalysisResult, BeatAnalysisResult
from fyp import Audio
from numpy.typing import NDArray
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random
import torch.utils
import wandb
import os
import torch.nn.functional as F
from tqdm.auto import tqdm
import dotenv

dotenv.load_dotenv()

class PhraseModelCore(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, *, dropouts: list[float] | None = None):
        super().__init__()
        if dropouts is None:
            dropouts = [0.1] * len(hidden_sizes)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.layers = nn.ModuleList()
        prev_size = input_size
        for size, drop in zip(hidden_sizes, dropouts):
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(drop))
            prev_size = size

        self.layers.append(nn.Linear(prev_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# A HuggingFace-compatible model format with built-in training
class PhraseModel(nn.Module):
    """The phrase model. inputs are (batch, D) where D is the input size.
    The output is (batch, output_size) where output_size is the output size.
    The output is the embedding of the input.
    If positive and negative are provided, the model will calculate the triplet loss with cosine similarity."""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.core = PhraseModelCore(
            input_size = config["input_size"],
            hidden_sizes = config["hidden_sizes"],
            output_size = config["output_size"],
            dropouts = config["dropouts"]
        )
        self.loss_fct =  nn.TripletMarginWithDistanceLoss(
            margin=config["margin"],
            distance_function=lambda x, y: 1 - F.cosine_similarity(x, y, dim=-1)
        )

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor | None = None, negative: torch.Tensor | None = None):
        anchor = self.core(anchor)
        if positive is not None and negative is not None:
            # Calculate the loss
            positive = self.core(positive)
            negative = self.core(negative)
            return self.loss_fct(anchor, positive, negative)

        return anchor

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
        valid_choices = [1, 2, 3, 5, 6, 7, -1, -2, -3, -5, -6, -7]
        valid_choices = [x for x in valid_choices if 0 <= bar_number + x < len(entry.downbeats) - self.nbar_phrases]
        shift = random.choice(valid_choices)
        negative_vector = get_input(ct, slice_time=bar_number + shift, n=self.n_chords, return_format="pt")

        return anchor_vector, positive_vector, negative_vector

def get_next_input(ct: ChordAnalysisResult, slice_time: int, *, n: int):
    chord_notes = [get_chord_notes()[chord] for chord in get_idx2voca_chord()]

    # Get next n chords
    next_chords: list[tuple[frozenset[str], float]] = []
    t = 0.
    ct_next = ct.slice_seconds(slice_time, ct.get_duration())
    end_times, labels = ct_next.grouped_end_times_labels()
    for end_time, label in zip(end_times, labels):
        notes = chord_notes[label]
        duration = end_time - t
        next_chords.append((notes, duration))
        t = end_time

    # Pad with empty chords
    no_chord_notes = chord_notes[-1]
    if len(next_chords) < n:
        next_chords += [(no_chord_notes, -1)] * (n - len(next_chords))

    return next_chords[:n]


def get_prev_input(ct: ChordAnalysisResult, slice_time: int, *, n: int):
    chord_notes = [get_chord_notes()[chord] for chord in get_idx2voca_chord()]

    # Get previous n chords. Prepad the first n chords with empty chords
    no_chord_notes = chord_notes[-1]
    prev_chords: list[tuple[frozenset[str], float]] =  [(no_chord_notes, -1)] * n
    if slice_time <= 0:
        return prev_chords
    t = 0.
    ct_prev = ct.slice_seconds(0, slice_time)
    end_times, labels = ct_prev.grouped_end_times_labels()
    for end_time, label in zip(end_times, labels):
        notes = chord_notes[label]
        duration = end_time - t
        prev_chords.append((notes, duration))
        t = end_time

    return prev_chords[-n:]

def vectorize_chords(chords: list[tuple[frozenset[str], float]], return_format: Literal["pt", "np", "py"] = "pt"):
    """Turns a list of chord notes into encoded vectors. The return format can be either a pytorch tensor, numpy array or python list.
    The format is: [12 * one hot encoded notes, no_chord indicator, duration]
    so the shape should be (14 * len(chords),)"""
    # Maybe in the future we can try setting the root note = 2 or something to emphasize it
    # Each chord entry is the following: [(12 * one hot encoded notes), no_chord indicator, duration]
    # no chord indicator means we have hit end of the song or beginning of the song

    chord_note_vector = []

    for chord_notes, duration in chords:
        if duration < 0:
            chord_note_vector.extend([0] * 12 + [1] + [0])
            continue

        one_hot = [0] * 12 + [0, duration]
        for note in chord_notes:
            one_hot[notes_to_idx(note)] = 1

        chord_note_vector.extend(one_hot)

    if return_format == "pt":
        return torch.tensor(chord_note_vector, dtype=torch.float32)
    elif return_format == "np":
        return np.array(chord_note_vector, dtype=np.float32)
    else:
        return chord_note_vector

def get_input(ct: ChordAnalysisResult, slice_time: int, *, n: int, return_format: Literal["pt", "np", "py"] = "pt"):
    """Get the vectorized input at the bar number slice_time. The input is the previous n chords and the next n chords. The return format can be either a pytorch tensor, numpy array or python list."""
    prev_chords = get_prev_input(ct, slice_time, n=n)
    next_chords = get_next_input(ct, slice_time, n=n)
    return vectorize_chords(prev_chords + next_chords, return_format=return_format)

def unvectorize_input(inputs: list[float] | torch.Tensor | NDArray[np.float32]):
    """Turns the vectorized input back into a list of chords. Mostly for debug"""
    # Convert inputs to numpy array
    chord_array: NDArray[np.float32]
    if isinstance(inputs, torch.Tensor):
        chord_array = inputs.cpu().numpy()
    elif isinstance(inputs, list):
        chord_array = np.array(inputs, dtype=np.float32)

    # Reconstructs the chord analysis result
    chord_array = chord_array.reshape(-1, 14)
    cumtime = 0.
    chords: list[int] = []
    chord_times: list[float] = []
    for chord_vector in chord_array:
        if chord_vector[12] == 1:
            continue
        chord_notes = frozenset([idx_to_notes(x) for x in range(12) if chord_vector[x] == 1])
        chord = get_chord_note_inv()[chord_notes]
        chord_idx = get_inv_voca_map()[chord]
        duration = chord_vector[13]
        chords.append(chord_idx)
        chord_times.append(cumtime)
        cumtime += duration

    ct = ChordAnalysisResult.from_data(
        cumtime,
        labels = chords,
        times = chord_times
    )
    return ct

CONFIG = {
    "dataset_id": "./resources/dataset/audio-infos-v2.db",
    "input_size": 14 * 2 * 10,
    "hidden_sizes": [256, 256, 512],
    "output_size": 512,
    "dropouts": [0.1, 0.1, 0.1],
    "margin": 0.5,
    "save_dir": "./resources/models",
    "log_every": 50,
    "save_every": 1000,
    "use_wandb": True,
}

def main():
    ds = SongDataset.load(CONFIG["dataset_id"])
    ds = ds.filter(lambda e: len(e.downbeats) > 32)

    model = PhraseModel(CONFIG)
    dataset = PhraseDataset(ds)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    save_path = CONFIG["save_dir"]
    os.makedirs(save_path, exist_ok=True)

    if CONFIG["use_wandb"]:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        with wandb.init(
            project = f"automasher-phrase",
            name = f"Trial 1",
            config = CONFIG
        ) as run:
            train_model(model, dataloader, optimizer, log_every=CONFIG["log_every"], save_every=CONFIG["save_every"], save_path=save_path, run=run)
    else:
        train_model(model, dataloader, optimizer, log_every=CONFIG["log_every"], save_every=CONFIG["save_every"], save_path=save_path, run=None)



def train_model(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, *,
                log_every: int, save_every: int, save_path: str, run):
    model = model.cuda()
    losses = []
    samples = 0
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
