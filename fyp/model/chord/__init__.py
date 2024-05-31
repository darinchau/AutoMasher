# Contains the definition of the Chord BTC model
# Code originally taken from https://github.com/jayg996/BTC-ISMIR19/tree/master

import os
import torch
import librosa
from dataclasses import dataclass
from typing import Any
from ...audio import Audio
from .chord_modules import *

@dataclass
class Hyperparameters:
    mp3: dict[str, float]
    feature: dict[str, Any]
    model: dict[str, Any]

def get_default_config() -> Hyperparameters:
    return Hyperparameters(
        mp3 = {
            'song_hz': 22050,
            'inst_len': 10.0,
            'skip_interval': 5.0
        },
        feature = {
            'n_bins': 144,
            'bins_per_octave': 24,
            'hop_length': 2048,
            'large_voca': False
        },
        model = {
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

_BTC_MODEL = None
def get_model(model_path: str, device: torch.device, use_loaded_model: bool) -> tuple[BTCModel, Hyperparameters, Any, Any]:
    global _BTC_MODEL
    if _BTC_MODEL is not None and use_loaded_model:
        return _BTC_MODEL

    # Init config
    config = get_default_config()

    config.feature['large_voca'] = True
    config.model['num_chords'] = 170
    if "large_voca" not in model_path:
       raise ValueError(f"The small model has been deprecated. Please use the large model. Perhaps incorrect path detected? {model_path}")

    # Load the model
    model = BTCModel(config = config.model).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    mean = checkpoint['mean']
    std = checkpoint['std']
    model.load_state_dict(checkpoint['model'])
    _BTC_MODEL = (model, config, mean, std)
    return _BTC_MODEL

def inference(audio: Audio, model_path: str, *, use_loaded_model: bool = True) -> list[tuple[float, int]]:
    """Main entry point. We will give you back list of triplets: (start, chord)"""
    # Handle audio and resample to the requied sr
    original_wav: np.ndarray = audio.resample(22050).numpy()
    sr = 22050

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, mean, std = get_model(model_path, device, use_loaded_model)

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
    time_unit = feature_per_second
    n_timestep = config.model['timestep']
    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feature.shape[0] // n_timestep

    # Inference
    start_time = 0.0
    lines: list[tuple[float, int]] = []
    with torch.no_grad():
        model.eval()
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        prev_chord: int = -1
        for t in range(num_instance):
            self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
            prediction, _ = model.output_layer(self_attn_output)
            prediction = prediction.squeeze()
            for i in range(n_timestep):
                if t == 0 and i == 0:
                    prev_chord = prediction[i].item()
                    continue
                if prediction[i].item() != prev_chord:
                    lines.append((start_time, prev_chord))
                    start_time = time_unit * (n_timestep * t + i)
                    prev_chord = prediction[i].item()
                if t == num_instance - 1 and i + num_pad == n_timestep:
                    if start_time != time_unit * (n_timestep * t + i):
                        lines.append((start_time, prev_chord))
                    break
    return lines
