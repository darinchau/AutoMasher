from typing import Literal, Callable
import torch
from torch import Tensor
import numpy as np
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F
from fyp.audio.search.search import calculate_self_similarity, calculate_self_similarity_beat
from fyp.audio.analysis import ChordAnalysisResult, BeatAnalysisResult
from fyp import Audio
from fyp.audio.dataset import DatasetEntry
from fyp.util.note import get_chord_notes, get_idx2voca_chord, get_chord_note_inv, get_inv_voca_map, notes_to_idx, idx_to_notes

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

class PhraseModel(nn.Module):
    """The phrase model. inputs are (batch, D) where D is the input size.
    The output is (batch, output_size) where output_size is the output size.
    The output is the embedding of the input.
    If positive and negative are provided, the model will calculate the triplet loss with cosine similarity."""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.core = PhraseModelCore(
            input_size = 15 * 2 * config["n_chords"],
            hidden_sizes = config["hidden_sizes"],
            output_size = config["output_size"],
            dropouts = config["dropouts"]
        )
        distances = {
            "cosine": lambda x, y: 1 - F.cosine_similarity(x, y, dim=-1),
            "euclidean": lambda x, y: F.pairwise_distance(x, y, p=2),
        }
        self.loss_fct =  nn.TripletMarginWithDistanceLoss(
            margin=config["margin"],
            distance_function=distances[config["distance_function"]]
        )

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor | None = None, negative: torch.Tensor | None = None):
        anchor = self.core(anchor)
        if positive is not None and negative is not None:
            # Calculate the loss
            positive = self.core(positive)
            negative = self.core(negative)
            return self.loss_fct(anchor, positive, negative)

        return anchor

    @property
    def distance_function(self) -> Callable[[Tensor, Tensor], Tensor]:
        fn = self.loss_fct.distance_function
        assert fn is not None
        return fn

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

def vectorize_chords(chords: list[tuple[frozenset[str], float]], starting_time: float, return_format: Literal["pt", "np", "py"] = "pt"):
    """Turns a list of chord notes into encoded vectors. The return format can be either a pytorch tensor, numpy array or python list.
    The format is: [12 * one hot encoded notes, no_chord indicator, duration, cum time (relative to the cut. Starting time is used to offset the prev chords time in the opposite direction)]
    so the shape should be (15 * len(chords),)"""
    # Maybe in the future we can try setting the root note = 2 or something to emphasize it
    # Each chord entry is the following: [(12 * one hot encoded notes), no_chord indicator, duration, cum time (relative to the cut)]
    # no chord indicator means we have hit end of the song or beginning of the song

    chord_note_vector = []

    cum_time = starting_time
    for chord_notes, duration in chords:
        if duration < 0:
            chord_note_vector.extend([0] * 12 + [1, 0, cum_time])
            continue

        one_hot = [0] * 12 + [0, duration, cum_time]
        for note in chord_notes:
            one_hot[notes_to_idx(note)] = 1

        chord_note_vector.extend(one_hot)
        cum_time += duration

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
    starting_time = -sum([x[1] for x in prev_chords if x[1] > 0])
    return vectorize_chords(prev_chords + next_chords, starting_time=starting_time, return_format=return_format)

def unvectorize_input(inputs: list[float] | torch.Tensor | NDArray[np.float32]):
    """Turns the vectorized input back into a list of chords. Mostly for debug"""
    # Convert inputs to numpy array
    chord_array: NDArray[np.float32]
    if isinstance(inputs, torch.Tensor):
        chord_array = inputs.cpu().numpy()
    elif isinstance(inputs, list):
        chord_array = np.array(inputs, dtype=np.float32)

    # Reconstructs the chord analysis result
    chord_array = chord_array.reshape(-1, 15)
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
