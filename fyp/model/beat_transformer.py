# Referenced from https://github.com/zhaojw1998/Beat-Transformer


import torch
import numpy as np
import librosa
from typing import Any
from scipy.signal.windows import hann
from librosa.core import stft
from torch.nn import TransformerEncoderLayer as torchTransformerEncoderLayer
import math
import torch 
import torch.nn.functional as F 
from torch import nn
from torch.nn.modules.normalization import LayerNorm
from .beat_tracker import unpack_beats, unpack_downbeats

class DemixedDilatedTransformerModel(nn.Module):
    def __init__(self, attn_len=5, instr=5, ntoken=2, dmodel=128, nhead=2, d_hid=512, nlayers=9, norm_first=True, dropout=.1):
        super(DemixedDilatedTransformerModel, self).__init__()
        self.nhead = nhead
        self.nlayers = nlayers
        self.attn_len = attn_len
        self.head_dim = dmodel // nhead
        self.dmodel = dmodel
        assert self.head_dim * nhead == dmodel, "embed_dim must be divisible by num_heads"

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 3), stride=1, padding=(2, 0))#126
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#42
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 12), stride=1, padding=(0, 0))#31
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#10
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=dmodel, kernel_size=(3, 6), stride=1, padding=(1, 0))#5
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))#1
        self.dropout3 = nn.Dropout(p=dropout)
        
        self.Transformer_layers = nn.ModuleDict({})
        for idx in range(nlayers):
            self.Transformer_layers[f'time_attention_{idx}'] = DilatedTransformerLayer(dmodel, nhead, d_hid, dropout, Er_provided=False, attn_len=attn_len, norm_first=norm_first)
            if (idx >= 3) and (idx <= 5):
                self.Transformer_layers[f'instr_attention_{idx}'] = torchTransformerEncoderLayer(dmodel, nhead, d_hid, dropout, batch_first=True, norm_first=norm_first)
            
        self.out_linear = nn.Linear(dmodel, ntoken)

        self.dropout_t = nn.Dropout(p=.5)
        self.out_linear_t = nn.Linear(dmodel, 300)
        
    def forward(self, x):
        #x: (batch, instr, time, dmodel), FloatTensor
        batch, instr, time, melbin = x.shape
        x = x.reshape(-1, 1, time, melbin)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = torch.relu(x)
        x = self.dropout3(x)    #(batch*instr, channel, time, 1)

        x = x.reshape(-1, self.dmodel, time).transpose(1, 2)    #(batch*instr, time, channel=dmodel)
        t = []

        for layer in range(self.nlayers):
            x, skip = self.Transformer_layers[f'time_attention_{layer}'](x, layer=layer)
            skip = skip.reshape(batch, instr, time, self.dmodel)
            t.append(skip.mean(1))
  
            if (layer >= 3) and (layer <= 5):
                x = x.reshape(batch, instr, time, self.dmodel)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(-1, instr, self.dmodel)

                x = self.Transformer_layers[f'instr_attention_{layer}'](x)

                x = x.reshape(batch, time, instr, self.dmodel)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(-1, time, self.dmodel)
            
        x = torch.relu(x)
        x = x.reshape(batch, instr, time, self.dmodel)
        x = x.mean(1)
        x = self.out_linear(x)

        t = torch.stack(t, axis=-1).sum(dim=-1)
        t = torch.relu(t)
        t = self.dropout_t(t)
        t = t.mean(dim=1) #(batch, dmodel)
        t = self.out_linear_t(t)

        return x, t

    def inference(self, x):
        #x: (batch, instr, time, dmodel), FloatTensor
        #This inference method also outputs the cumulative attention matrix
        batch, instr, time, melbin = x.shape
        x = x.reshape(-1, 1, time, melbin)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = torch.relu(x)
        x = self.dropout3(x)    #(batch*instr, channel, time, 1)

        x = x.reshape(-1, self.dmodel, time).transpose(1, 2)    #(batch*instr, time, channel=dmodel)
        t = []

        attn = [torch.eye(time, device=x.device).repeat(batch, self.nhead, 1, 1)]

        for layer in range(self.nlayers):
            x, skip, layer_attn = self.Transformer_layers[f'time_attention_{layer}'].inference(x, layer=layer)
            skip = skip.reshape(batch, instr, time, self.dmodel)
            t.append(skip.mean(1))

            attn.append(torch.matmul(attn[-1], layer_attn.transpose(-2, -1)))
  
            if (layer >= 3) and (layer <= 5):
                x = x.reshape(batch, instr, time, self.dmodel)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(-1, instr, self.dmodel)

                x = self.Transformer_layers[f'instr_attention_{layer}'](x)

                x = x.reshape(batch, time, instr, self.dmodel)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(-1, time, self.dmodel)
            
        x = torch.relu(x)
        x = x.reshape(batch, instr, time, self.dmodel)
        x = x.mean(1)
        x = self.out_linear(x)

        t = torch.stack(t, axis=-1).sum(dim=-1)
        t = torch.relu(t)
        t = self.dropout_t(t)
        t = t.mean(dim=1) #(batch, dmodel)
        t = self.out_linear_t(t)

        return x, t, attn


class DilatedMultiheadSelfAttentionWithRelativePositionalEmbedding(nn.Module):
    def __init__(self, dmodel, num_heads, dropout=0., Er_provided=False, attn_len=5):
        super(DilatedMultiheadSelfAttentionWithRelativePositionalEmbedding, self).__init__()
        self.attn_len = attn_len
        self.dmodel = dmodel
        self.num_heads = num_heads
        self.head_dim = dmodel // num_heads
        assert self.head_dim * num_heads == dmodel, "embed_dim must be divisible by num_heads"

        self.key = nn.Linear(dmodel, dmodel)
        self.value = nn.Linear(dmodel, dmodel)
        self.query = nn.Linear(dmodel, dmodel)
        self.dropout = nn.Dropout(dropout)
        self.Er_provided = Er_provided
        
        if not Er_provided:
            self.Er = nn.Parameter(torch.randn(num_heads, self.head_dim, attn_len))


    def forward(self, query, key, value, layer=0):
        #query, key, and value: (batch, time, dmodel), float tensor

        batch, time, d_model = query.shape

        q = self.query(query).reshape(batch, time, self.num_heads, 1, self.head_dim).transpose(1, 2)  #(batch, num_head, time, 1, head_dim)
        k = self.key(key).reshape(batch, time, self.num_heads, 1, self.head_dim).transpose(1, 2)  #(batch, num_head, time, 1, head_dim)
        v = self.value(value).reshape(batch, time, self.num_heads, 1, self.head_dim).transpose(1, 2)  #(batch, num_head, time, 1, head_dim)

        k = torch.cat(
                        (
                        self.kv_roll(k[:, 0: 4], layer, padding_value=0, shift=0),
                        self.kv_roll(k[:, 4: 5], layer, padding_value=0, shift=-2),
                        self.kv_roll(k[:, 5: 6], layer, padding_value=0, shift=-1),
                        self.kv_roll(k[:, 6: 7], layer, padding_value=0, shift=1),
                        self.kv_roll(k[:, 6: 7], layer, padding_value=0, shift=2)   
                        ),
                    dim=1
                    )   #we define 4 symmetrical heads and 4 skewed heads
                        #The last line should be k[:, 7: 8]. This is a bug in my code. 
                        #This bug should not have impacted model performance though.

        v = torch.cat(
                        (
                        self.kv_roll(v[:, 0: 4], layer, padding_value=0, shift=0),
                        self.kv_roll(v[:, 4: 5], layer, padding_value=0, shift=-2),
                        self.kv_roll(v[:, 5: 6], layer, padding_value=0, shift=-1),
                        self.kv_roll(v[:, 6: 7], layer, padding_value=0, shift=1),
                        self.kv_roll(v[:, 7: 8], layer, padding_value=0, shift=2)
                        ),
                    dim=1
                    )   #we define 4 symmetrical heads and 4 skewed heads
        
        Er_t = self.Er.unsqueeze(1).unsqueeze(0)  #(1, num_head, 1, head_dim, attn_len)

        qk = torch.matmul(q, k.transpose(-2, -1))
        attn_mask = torch.zeros_like(qk).masked_fill_((qk==0), float('-inf'))
        attn = (qk + torch.matmul(q, Er_t)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn + attn_mask, dim=-1)

        out = torch.matmul(attn, v) #(batch, num_head, time, 1, head_dim)
        out = out.squeeze(-2).transpose(1, 2).reshape(batch, time, d_model)

        return self.dropout(out), attn

    def kv_roll(self, tensor, layer, padding_value=0, shift=1):
        #tensor: (batch, num_head, time, 1, head_dim)
        batch, num_head, time, _, head_dim = tensor.shape

        tensor = F.pad(tensor, (0, 0, 0, 0, (2**layer)*(self.attn_len//2), (2**layer)*(self.attn_len//2)), mode='constant', value=padding_value) 
        #(batch, num_head, time+(2**layer)*(self.attn_len//2), 1, head_dim)

        tensor = torch.cat([torch.roll(tensor, shifts=-i*(2**layer), dims=2) for i in range(shift, self.attn_len+shift)], dim=-2) 
        #(batch, num_head, time+(2**layer)*(self.attn_len//2), attn_len, head_dim)

        return tensor[:, :, :time, :, :]    #(batch, num_head, time, attn_len, head_dim)

class DilatedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, Er_provided=False, attn_len=5, norm_first=False, layer_norm_eps=1e-5):
        super(DilatedTransformerLayer, self).__init__()
        self.self_attn = DilatedMultiheadSelfAttentionWithRelativePositionalEmbedding(d_model, nhead, dropout, Er_provided, attn_len)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu


    def forward(self, x, layer=0):
        #x: (batch, time, dmodel)
        if self.norm_first:
            x_ = self._sa_block(self.norm1(x), layer)[0]
            x = x + x_
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, layer)[0])
            x = self.norm2(x + self._ff_block(x))
        return x, x_


    def inference(self, x, layer=0):
        #x: (batch, time, dmodel)
        if self.norm_first:
            x_, attn = self._sa_block(self.norm1(x), layer)
            x = x + x_
            x = x + self._ff_block(self.norm2(x))
        else:
            x_, attn = self._sa_block(x, layer)
            x = self.norm1(x + x_)
            x = self.norm2(x + self._ff_block(x))


        attn = attn.squeeze(-2) #batch, num_head, time, attn_len
        batch, num_head, time, attn_len = attn.shape
        padded_attn_len = (attn_len-1) * (2**layer) + 1
        tmp_output = torch.zeros(batch, num_head, time, padded_attn_len, device=x.device)
        for i, j in enumerate(range(0, padded_attn_len, 2**layer)):
            tmp_output[:, :, :, j] = attn[:, :, :, i]

        attn = torch.zeros(batch, num_head, time, time+(padded_attn_len-1)*2, device=x.device)
        for i in range(time):
            attn[:, :, i, i: i+padded_attn_len] = tmp_output[:, :, i]

        center = (padded_attn_len-1)
        attn = torch.cat(
                            [
                                attn[:, 0: 4, :,  center - (2**layer) * 2:  center - (2**layer) * 2 + time],
                                attn[:, 4: 5, :,  center - (2**layer) * 1:  center - (2**layer) * 1 + time],
                                attn[:, 5: 6, :,  center - (2**layer) * 0:  center - (2**layer) * 0 + time],
                                attn[:, 6: 7, :,  center - (2**layer) * 3:  center - (2**layer) * 3 + time],
                                attn[:, 7: 8, :,  center - (2**layer) * 4:  center - (2**layer) * 4 + time]
                            ],
                            dim=1
                        )   #restore the square attention matrix from dilated self-attention

        return x, x_, attn


    # self-attention block
    def _sa_block(self, x, layer=0):
        x, attn = self.self_attn(x, x, x, layer)
        return self.dropout1(x), attn


    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

def separator_stft(data: np.ndarray) -> np.ndarray:
    data = np.asfortranarray(data)
    N = 4096
    H = 1024
    win = hann(N, sym=False)
    n_channels = data.shape[-1]
    out = []
    for c in range(n_channels):
        d = np.concatenate((np.zeros((N,)), data[:, c], np.zeros((N,))))
        s = stft(d, hop_length=H, window=win, center=False, n_fft = N)
        s = np.expand_dims(s.T, 2)
        out.append(s)
    if len(out) == 1:
        return out[0]
    return np.concatenate(out, axis=2)

def inference(parts: dict[str, np.ndarray], model_path: str, *, min_bpm: float = 55.0, max_bpm: float = 215.0) -> tuple[list[float], list[float]]:
    """Beat transformer inference code copied from backer-end"""
    try:
        from madmom.features.beats import DBNBeatTrackingProcessor
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    except ImportError:
        raise ImportError("Please install madmom to use this function. Install madmom from `pip install git+https://github.com/darinchau/madmom`")
    assert set(parts.keys()) == {'vocals', 'piano', 'drums', 'bass', 'other'}
    model = DemixedDilatedTransformerModel(attn_len=5, instr=5, ntoken=2, dmodel=256, nhead=8, d_hid=1024, nlayers=9, norm_first=True)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])
    
    downbeat_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=44100/1024, 
                                                    transition_lambda=100, observation_lambda=6, num_tempi=None, threshold=0.2) #type: ignore
    

    mel_f = librosa.filters.mel(sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000).T
    x = np.stack([
        np.dot(np.abs(np.mean(separator_stft(parts[k]), axis=-1))**2, mel_f) for k in (
            'vocals', 'piano', 'drums', 'bass', 'other'
        )
    ])
    x = np.transpose(x, (0, 2, 1))
    x = np.stack([librosa.power_to_db(x[i], ref=np.max) for i in range(len(x))])
    x = np.transpose(x, (0, 2, 1))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        model_input = torch.from_numpy(x).unsqueeze(0).float().to(device)
        activation, _ = model(model_input)

    beat_activation = torch.sigmoid(activation[0, :, 0]).detach().cpu().numpy()
    downbeat_activation = torch.sigmoid(activation[0, :, 1]).detach().cpu().numpy()
    dbn_beat_pred = unpack_beats(beat_activation, min_bpm=55.0, max_bpm=215.0, fps=44100/1024, threshold=0.2)

    combined_act = np.concatenate((np.maximum(beat_activation - downbeat_activation,
                                            np.zeros(beat_activation.shape)
                                            )[:, np.newaxis],
                                downbeat_activation[:, np.newaxis]
                                ), axis=-1)   #(T, 2)
    dbn_downbeat_pred = unpack_downbeats(combined_act, beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=44100/1024, threshold=0.2)
    dbn_downbeat_pred = dbn_downbeat_pred[dbn_downbeat_pred[:, 1] == 1][:, 0]
    return dbn_beat_pred, dbn_downbeat_pred
