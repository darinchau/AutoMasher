import torch
from torch import nn, Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from fyp.model.chord import get_model, Hyperparameters
import numpy as np
import librosa
from numpy.typing import NDArray
import os
from fyp.audio.dataset import SongDataset, DatasetEntry
from fyp import Audio
from torch.utils.data import Dataset, DataLoader
from typing import Literal
import random
from fyp.audio.manipulation import PitchShift
import json

class HarmonicFeaturesExtractor:
    def __init__(self, model_path: str = "resources/ckpts/btc_model_large_voca.pt", device: torch.device | None = None, use_loaded_model: bool = True, cache_dir: str = "resources/cache"):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model, self.config, self.mean, self.std = get_model(model_path, device, use_loaded_model)
        self.cache_dir = cache_dir

    def __call__(self, audio: Audio) -> torch.Tensor:
        features = calculate_features(audio, config=self.config, mean=self.mean, std=self.std)
        harmonic_features = get_harmonic_features(features, model=self.model, config=self.config, device=self.device)
        return harmonic_features

def preprocess(entry: DatasetEntry, extractor: HarmonicFeaturesExtractor | None = None, audio: Audio | None = None, nbars: int = 8, augment: bool = False, speedup: float = 1., transpose: int = 0):
    """Returns features, anchors, anchor_valid_lengths"""
    # Augment by speedup and transposition
    if audio is None:
        audio = entry.get_audio()

    if extractor is None:
        extractor = HarmonicFeaturesExtractor(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if augment:
        audio = audio.change_speed(speedup)
        audio = audio.apply(PitchShift(transpose))
    else:
        speedup = 1.

    features = extractor(audio)
    sliced_features = []
    downbeat_slice_idxs = [int(downbeat / speedup * 10.8) for downbeat in entry.downbeats]

    for i in range(len(entry.downbeats) - nbars):
        start_downbeat_idx = downbeat_slice_idxs[i]
        end_downbeat_idx = downbeat_slice_idxs[i + nbars]
        feat_ = features[start_downbeat_idx:end_downbeat_idx]
        sliced_features.append(feat_)

    anchor_valid_lengths = torch.tensor([len(x) for x in sliced_features]).to(extractor.device)

    max_length = int(anchor_valid_lengths.max().item())
    anchors = torch.stack([F.pad(x, (0, 0, 0, max_length - len(x))) for x in sliced_features])

    return features, anchors, anchor_valid_lengths

class SelfAttention(nn.Module):
    """Multihead Self Attention module"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = head_dim

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). " f"Found {x.shape}.")

        batch_size, length, embed_dim = x.size()
        shape = (batch_size, 1, length, length)
        if attention_mask.size() != shape:
            raise ValueError(f"The expected attention mask shape is {shape}. " f"Found {attention_mask.size()}.")

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)
        k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)
        v = self.v_proj(x).view(*shape).transpose(2, 1)
        # scale down q to avoid value overflow.
        weights = (self.scaling * q) @ k
        weights += attention_mask

        # This avoids overflow
        # Section 3.3 https://arxiv.org/abs/2112.08778
        weights = weights - weights.max(dim=-1, keepdim=True)[0]

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v
        output = output.transpose(2, 1).reshape(batch_size, length, embed_dim)

        output = self.out_proj(output)
        return output

class FeedForward(nn.Module):
    """The feed forward layer in a transformer. Expects x: (batch, len_sequence, nfeatures) and returns x: (batch, len_sequence, nfeatures)"""
    def __init__(self, io_features: int, intermediate_features: int, intermediate_dropout: float, output_dropout: float):
        super().__init__()
        self.intermediate_dense = nn.Linear(io_features, intermediate_features)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = nn.Linear(intermediate_features, io_features)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x: Tensor):
        x = self.intermediate_dense(x)
        x = F.gelu(x)
        x = self.intermediate_dropout(x)
        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x

class EncoderLayer(nn.Module):
    """A layer unit in encoder with multihead attention and feed forward layer plus residual

    x: has shape (batch, sequence_length, embed_dim)
    position_bias: has shape (batch_size * num_heads, src_len, src_len)"""
    def __init__(self, attention: SelfAttention, dropout: float, layer_norm_first: bool, feed_forward: FeedForward):
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(attention.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(attention.embed_dim)

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        residual = x

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.attention(x, attention_mask=attention_mask)

        x = self.dropout(x)
        x = residual + x

        if self.layer_norm_first:
            x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x)
            x = self.final_layer_norm(x + self.feed_forward(x))
        return x

class FeatureProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x

class ConvolutionalPositionalEmbedding(nn.Module):
    """Positional embedding which is placed at the beginning of Transformer.

    x: has shape (N, L, nfeatures)"""
    def __init__(self, embed_dim: int, kernel_size: int, groups: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size

        conv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=groups)
        self.conv = nn.utils.weight_norm(conv, name="weight", dim=2)

    def forward(self, x: Tensor):
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if not (self.kernel_size & 1):
            x = x[..., : -1]
        x = F.gelu(x)
        x = x.transpose(-2, -1)
        return x

class Transformer(nn.Module):
    def __init__(self, conv_position_embedding: ConvolutionalPositionalEmbedding, dropout_rate: float, encoder_layers: nn.ModuleList, layer_norm_first: bool, layer_drop: float):
        super().__init__()
        self.pos_conv_embed = conv_position_embedding
        self.layer_norm = nn.LayerNorm(conv_position_embedding.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = encoder_layers

    def preprocess(self, x: Tensor):
        x = x + self.pos_conv_embed(x)
        if self.layer_norm_first:
            x = self.layer_norm(x)
        x = self.dropout(x)
        return x

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        x = self.preprocess(x)
        for layer in self.layers:
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                x = layer(x, attention_mask)

        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x

class Encoder(nn.Module):
    """Forward accepts x of shape (batch, nfeatres, embed dims)"""
    def __init__(self, feature_projection: FeatureProjection, transformer: Transformer):
        super().__init__()
        self.feature_projection = feature_projection
        self.transformer = transformer

    def preprocess(self, features: Tensor, lengths: Tensor):
        x = self.feature_projection(features)

        # Create a bit mask to zero out the regions that doesnt make sense (as suggested by length parameter)
        # Then extend the mask to attention shape and set weight
        batch_size, max_len, _ = x.shape
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
        x[mask] = 0.0
        mask = -10000.0 * mask[:, None, None, :].to(dtype=features.dtype)
        mask = mask.expand(batch_size, 1, max_len, max_len)
        return x, mask

    def forward(self, features: Tensor, valid_lengths: Tensor) -> Tensor:
        x, mask = self.preprocess(features, valid_lengths)
        x = self.transformer(x, attention_mask=mask)
        return x

@dataclass(frozen=True)
class AudioEncoderModelConfig:
    input_features: int = 128
    encoder_embed_dim: int = 128
    encoder_projection_dropout: float = 0.1
    encoder_pos_conv_kernel: int = 64
    encoder_pos_conv_groups: int = 16
    encoder_num_layers: int = 12
    encoder_num_heads: int = 8
    encoder_attention_dropout: float = 0.1
    encoder_ff_interm_features: int = 256
    encoder_ff_interm_dropout: float = 0.1
    encoder_dropout: float = 0.1
    encoder_layer_norm_first: bool = True
    encoder_layer_drop: float = 0.0
    normalize_input: bool = True

class AudioEncoderModel(nn.Module):
    """An audio transformer encoder implemented in a very specific way - to load model weights and not shit my pants
    forward accepts x (shape: batch, nchannels, time) and lengths (shape: batch) and returns (shape: batch, time, embedding ndims)"""
    def __init__(self, config: AudioEncoderModelConfig):
        super().__init__()

        self.encoder = self.get_encoder(config)
        self.config = config

    def get_encoder(self, config: AudioEncoderModelConfig):
        proj_features = config.input_features
        feature_projection = FeatureProjection(proj_features, config.encoder_embed_dim, config.encoder_projection_dropout)
        pos_conv = ConvolutionalPositionalEmbedding(config.encoder_embed_dim, config.encoder_pos_conv_kernel, config.encoder_pos_conv_groups)

        # Transformer encoder
        encoder_layers = nn.ModuleList()
        for _ in range(config.encoder_num_layers):
            attention = SelfAttention(embed_dim=config.encoder_embed_dim, num_heads=config.encoder_num_heads, dropout=config.encoder_attention_dropout)
            feed_forward = FeedForward(
                io_features=config.encoder_embed_dim,
                intermediate_features=config.encoder_ff_interm_features,
                intermediate_dropout=config.encoder_ff_interm_dropout,
                output_dropout=config.encoder_dropout
            )
            encoder_layer = EncoderLayer(attention=attention, dropout=config.encoder_dropout, layer_norm_first=config.encoder_layer_norm_first, feed_forward=feed_forward)
            encoder_layers.append(encoder_layer)

        transformer = Transformer(
            conv_position_embedding = pos_conv,
            dropout_rate = config.encoder_dropout,
            encoder_layers = encoder_layers,
            layer_norm_first = not config.encoder_layer_norm_first,
            layer_drop = config.encoder_layer_drop,
        )

        encoder = Encoder(feature_projection, transformer)
        return encoder

    @property
    def encoded_dim(self):
        return self.config.encoder_embed_dim

    def forward(self, x: Tensor, valid_length: Tensor) -> Tensor:
        # x must have shape (N, 2, L, nfeatures)
        # valid_length must have shape (N, 2)
        assert len(x.shape) == 3 and x.shape[-1] == self.config.input_features, f"x must have 3 dimensions (N, L, nfeatures={self.encoded_dim}), found {x.shape}"
        assert len(valid_length.shape) == 1, f"valid_length must have 1 dimensions (N,), found {valid_length.shape}"
        assert x.shape[0] == valid_length.shape[0], f"Batch size mismatch: x.shape[0] ({x.shape[0]}) != valid_length.shape[0] ({valid_length.shape[0]})"
        if self.config.normalize_input:
            x = F.layer_norm(x, x.shape)
        x = self.encoder(x, valid_length)
        x = F.adaptive_avg_pool1d(x.transpose(-2, -1), 1).squeeze(-1)
        return x

def calculate_features(audio: Audio, *, config: Hyperparameters, mean: float, std: float, sr: int = 22050):
    original_wav: np.ndarray = audio.resample(sr).numpy()

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
    return feature

def get_harmonic_features(feature: NDArray[np.float32], *, model: nn.Module, config: Hyperparameters, device: torch.device) -> torch.Tensor:
    n_timestep = config.model['timestep']
    num_pad = n_timestep - (feature.shape[0] % n_timestep)

    feature_ = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feature_.shape[0] // n_timestep

    # This batched inference is practically the same as the original inference
    feature_ = torch.tensor(feature_, dtype=torch.float32).to(device)
    feature_ = torch.stack([feature_[i * n_timestep:(i + 1) * n_timestep, :] for i in range(num_instance)], dim=0)

    # Inference
    with torch.no_grad():
        model.eval()
        predictions, _ = model.self_attn_layers(feature_)

    # Unstack the predictions
    predictions = predictions.reshape(-1, predictions.shape[-1])

    # start_time = 0.0
    # predictions2 = []
    # with torch.no_grad():
    #     model.eval()
    #     feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
    #     prev_chord: int = -1
    #     for t in range(num_instance):
    #         self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
    #         self_attn_output = self_attn_output.squeeze(0)
    #         predictions2.append(self_attn_output)

    # predictions2 = torch.cat(predictions2, dim=0)

    # torch.abs(predictions - predictions2).max()
    return predictions # (L, nfeatures=128)
