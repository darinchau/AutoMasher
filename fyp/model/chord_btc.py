# Contains the definition of the Chord BTC model
# Code originally taken from https://github.com/jayg996/BTC-ISMIR19/tree/master

import os
import torch
import librosa
from dataclasses import dataclass
from typing import Any
from ..audio import Audio
from ..util.note import get_idx2chord, get_idx2voca_chord
from .modules import *
from .modules import _gen_timing_signal, _gen_bias_mask

@dataclass
class Hyperparameters:
    mp3: dict[str, float]
    feature: dict[str, Any]
    model: dict[str, Any]


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, attention_map=False):
        super(SelfAttentionBlock, self).__init__()

        self.attention_map = attention_map
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,hidden_size, num_heads, bias_mask, attention_dropout, attention_map)
        self.positionwise_convolution = PositionwiseFeedForward(hidden_size, filter_size, hidden_size, layer_config='cc', padding='both', dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x = inputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        if self.attention_map is True:
            y, weights = self.multi_head_attention(x_norm, x_norm, x_norm)
        else:
            y = self.multi_head_attention(x_norm, x_norm, x_norm)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_convolution(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        if self.attention_map is True:
            return y, weights #type: ignore
        return y

class BidirectionalSelfAttention(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, max_length,
                 layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):

        super(BidirectionalSelfAttention, self).__init__()

        self.weights_list = list()

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  True)

        self.attn_block = SelfAttentionBlock(*params)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  torch.transpose(_gen_bias_mask(max_length), dim0=2, dim1=3),
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  True)

        self.backward_attn_block = SelfAttentionBlock(*params)

        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, inputs):
        x, list = inputs

        # Forward Self-attention Block
        encoder_outputs, weights = self.attn_block(x)
        # Backward Self-attention Block
        reverse_outputs, reverse_weights = self.backward_attn_block(x)
        # Concatenation and Fully-connected Layer
        outputs = torch.cat((encoder_outputs, reverse_outputs), dim=2)
        y = self.linear(outputs)

        # Attention weights for Visualization
        self.weights_list = list
        self.weights_list.append(weights)
        self.weights_list.append(reverse_weights)
        return y, self.weights_list

class BidirectionalSelfAttentionLayers(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0):
        super(BidirectionalSelfAttentionLayers, self).__init__()

        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  max_length,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.self_attn_layers = nn.Sequential(*[BidirectionalSelfAttention(*params) for l in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # A Stack of Bi-directional Self-attention Layers
        y, weights_list = self.self_attn_layers((x, []))

        # Layer Normalization
        y = self.layer_norm(y)
        return y, weights_list

class BTCModel(nn.Module):
    def __init__(self, config):
        super(BTCModel, self).__init__()

        self.timestep = config['timestep']
        self.probs_out = config['probs_out']

        params = (config['feature_size'],
                  config['hidden_size'],
                  config['num_layers'],
                  config['num_heads'],
                  config['total_key_depth'],
                  config['total_value_depth'],
                  config['filter_size'],
                  config['timestep'],
                  config['input_dropout'],
                  config['layer_dropout'],
                  config['attention_dropout'],
                  config['relu_dropout'])

        self.self_attn_layers = BidirectionalSelfAttentionLayers(*params)
        self.output_layer = SoftmaxOutputLayer(hidden_size=config['hidden_size'], output_size=config['num_chords'], probs_out=config['probs_out'])

    def forward(self, x, labels):
        labels = labels.view(-1, self.timestep)
        # Output of Bi-directional Self-attention Layers
        self_attn_output, weights_list = self.self_attn_layers(x)

        # return logit values for CRF
        if self.probs_out is True:
            logits = self.output_layer(self_attn_output)
            return logits

        # Output layer and Soft-max
        prediction,second = self.output_layer(self_attn_output)
        prediction = prediction.view(-1)
        second = second.view(-1)

        # Loss Calculation
        loss = self.output_layer.loss(self_attn_output, labels)
        return prediction, loss, weights_list, second

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

def inference(audio: Audio, model_path: str, *, use_voca = False, device = None) -> list[tuple[float, int]]:
    """Main entry point. We will give you back list of triplets: (start, chord)"""
    # Handle audio and resample to the requied sr
    original_wav: np.ndarray = audio.resample(22050).numpy()
    sr = 22050

    # Device
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init config
    config = get_default_config()

    if use_voca:
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170
        if "large_voca" not in model_path:
            model_path = model_path.replace("model.pt", "model_large_voca.pt")

    # Load the model
    model = BTCModel(config = config.model).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    mean = checkpoint['mean']
    std = checkpoint['std']
    model.load_state_dict(checkpoint['model'])

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
