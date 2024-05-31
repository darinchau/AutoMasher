# Referenced from https://github.com/zhaojw1998/Beat-Transformer

import torch
import numpy as np
import librosa
from scipy.signal.windows import hann
from librosa.core import stft
import torch
from .modules import DemixedDilatedTransformerModel
from .tracker import unpack_beats, unpack_downbeats, require_madmom

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

_BEAT_MODEL = None
def get_model(model_path: str, device: torch.device, use_loaded_model: bool = True):
    global _BEAT_MODEL
    if _BEAT_MODEL is not None and use_loaded_model:
        return _BEAT_MODEL
    model = DemixedDilatedTransformerModel(attn_len=5, instr=5, ntoken=2, dmodel=256, nhead=8, d_hid=1024, nlayers=9, norm_first=True)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])
    model.to(device)
    model.eval()
    _BEAT_MODEL = model
    return model

def inference(parts: dict[str, np.ndarray], model_path: str, use_loaded_model: bool = True) -> tuple[list[float], list[float]]:
    """Beat transformer inference code copied from backer-end"""
    require_madmom()
    assert set(parts.keys()) == {'vocals', 'piano', 'drums', 'bass', 'other'}

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
    model = get_model(model_path, device, use_loaded_model)

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
