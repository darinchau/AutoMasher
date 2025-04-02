# Contains the definition of the Chord BTC model
# Code originally taken from https://github.com/jayg996/BTC-ISMIR19/tree/master

import os
import torch
import librosa
from dataclasses import dataclass
from typing import Any
from ...audio import Audio
from .chord_modules import *
import warnings


@dataclass
class Hyperparameters:
    mp3: dict[str, float]
    feature: dict[str, Any]
    model: dict[str, Any]


def get_default_config() -> Hyperparameters:
    return Hyperparameters(
        mp3={
            'song_hz': 22050,
            'inst_len': 10.0,
            'skip_interval': 5.0
        },
        feature={
            'n_bins': 144,
            'bins_per_octave': 24,
            'hop_length': 2048,
            'large_voca': False
        },
        model={
            'feature_size': 144,
            'timestep': 108,
            'num_chords': 25,
            'input_dropout': 0.2,
            'layer_dropout': 0.2,
            'attention_dropout': 0.2,
            'relu_dropout': 0.2,
            'num_layers': 8,
            'num_heads': 4,
            'hidden_size': 128,
            'total_key_depth': 128,
            'total_value_depth': 128,
            'filter_size': 128,
            'loss': 'ce',
            'probs_out': False
        },
    )


def get_model(model_path: str, device: torch.device, use_voca: bool) -> tuple[BTCModel, Hyperparameters, Any, Any]:
    # Init config
    config = get_default_config()

    if use_voca:
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170
        if "large_voca" not in model_path:
            model_path = model_path.replace("model.pt", "model_large_voca.pt")

    # Load the model
    model = BTCModel(config=config.model).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    mean = checkpoint['mean']
    std = checkpoint['std']
    try:
        model.load_state_dict(checkpoint['model'])
    except Exception as e:
        raise ValueError("Model cannot load checkpoint. Perhaps you provided the large voca model for small voca or vice versa.") from e
    del model.output_layer.lstm
    return model, config, mean, std


@dataclass
class ChordModelOutput:
    logits: torch.Tensor
    features: torch.Tensor
    duration: float
    time_resolution: float  # In Hz (number of features per second)

    def save(self, path: str):
        torch.save({
            'logits': self.logits,
            'features': self.features,
            'time_resolution': self.time_resolution,
            'duration': self.duration
        }, path)

    @staticmethod
    def load(path: str) -> 'ChordModelOutput':
        file = torch.load(path)
        return ChordModelOutput(
            duration=file['duration'],
            logits=file['logits'],
            features=file['features'],
            time_resolution=file['time_resolution']
        )

    def __post_init__(self):
        if self.time_resolution >= 1:
            warnings.warn(f"Time resolution is greater than 1 Hz (found {self.time_resolution}). This is likely a bug.")


def inference(audio: Audio, model_path: str, *, use_voca: bool = True) -> ChordModelOutput:
    """Main entry point. We will give you back list of triplets: (start, chord)"""
    # Handle audio and resample to the requied sr
    audio_duration = audio.duration
    original_wav: np.ndarray = audio.resample(22050).numpy()
    sr = 22050

    # Load the model
    # TODO: Profile this function to see if model caching is necessary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, mean, std = get_model(model_path, device, use_voca)

    # Compute audio features
    currunt_sec_hz = 0
    feature = np.array([])
    while len(original_wav) > currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len']:
        start_idx = int(currunt_sec_hz)
        end_idx = int(currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len'])
        tmp = librosa.cqt(original_wav[start_idx:end_idx], sr=sr, n_bins=config.feature['n_bins'], bins_per_octave=config.feature['bins_per_octave'], hop_length=config.feature['hop_length'])
        if start_idx == 0:
            feature = tmp
        else:
            feature = np.concatenate((feature, tmp), axis=1)
        currunt_sec_hz = end_idx

    # Concatenate the last part of the audio onto the feature
    tmp = librosa.cqt(original_wav[currunt_sec_hz:], sr=sr, n_bins=config.feature['n_bins'], bins_per_octave=config.feature['bins_per_octave'], hop_length=config.feature['hop_length'])
    if currunt_sec_hz == 0:
        feature = tmp
    else:
        feature = np.concatenate((feature, tmp), axis=1)

    feature = np.log(np.abs(feature) + 1e-6)
    feature_per_second = config.mp3['inst_len'] / config.model['timestep']

    # Process features
    feature = feature.T
    feature = (feature - mean) / std
    n_timestep = config.model['timestep']
    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feature.shape[0] // n_timestep

    # Inference
    features = []
    logits = []
    with torch.no_grad():
        model.eval()
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        for t in range(num_instance):
            self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
            features.append(self_attn_output)
            prediction, logit = model.output_layer(self_attn_output)
            prediction = prediction.squeeze()
            logits.append(logit)
    features = torch.cat(features, dim=1)[0].cpu().numpy()
    logits = torch.cat(logits, dim=1)[0].cpu().numpy()
    return ChordModelOutput(
        duration=audio_duration,
        logits=torch.tensor(logits),
        features=torch.tensor(features),
        time_resolution=feature_per_second
    )
