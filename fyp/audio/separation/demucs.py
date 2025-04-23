from .. import DemucsCollection, Audio
import os
import subprocess
import tempfile
import torch
from torch import Tensor
from pathlib import Path
from enum import Enum
import torchaudio.functional as F
from .. import Audio, AudioMode, DemucsCollection


class DemucsAudioSeparator:
    def __init__(self, model_name: str = "htdemucs", repo: Path | None = None, segment: float | None = None, compile: bool = False):
        """ Preloads the model

        segment (float): duration of the chunks of audio to ideally evaluate the model on.
            This is used by `demucs.apply.apply_model`.

        compile (bool): whether to compile the model. If set to False, the model will be loaded in eval mode."""
        try:
            from demucs.pretrained import get_model
            from demucs.repo import ModelLoadingError
            from demucs.apply import apply_model, BagOfModels
        except ImportError as e:
            raise ImportError("Please install demucs to use this class") from e

        try:
            model = get_model(model_name, repo)
        except ModelLoadingError as error:
            raise RuntimeError(f"Failed to get model from args {error}")

        if segment is not None and segment < 8:
            raise RuntimeError("Segment must greater than 8.")

        if isinstance(model, BagOfModels):
            if segment is not None:
                for sub in model.models:
                    sub.segment = segment
        else:
            if segment is not None:
                model.segment = segment

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        # Not explicitly tested - this worked slower than default on my WSL machine.
        if compile:
            if isinstance(model, BagOfModels):
                for sub in model.models:
                    sub.compile()
            else:
                model.compile()
        self.model = model

    @property
    def sample_rate(self) -> int:
        """Sample rate of the audio the model expects."""
        return self.model.samplerate

    @property
    def nchannels(self) -> int:
        """Number of channels that the model expects"""
        return self.model.audio_channels

    def pipeline(self,
                 audio: Tensor,
                 shifts: int = 1,
                 split: bool = True,
                 jobs: int = 0,
                 overlap: float = 0.25,
                 show_progress: bool = False):
        """Performs the demucs audio separation pipeline.

        Feel free to play around with different hyperparameters
        audio: Tensor of shape (nchannels, T) representing an audio with channels equal to self.nchannels,
            and sample rate equal to self.sample_rate
        name: name of the model structure. Use the DemucsModelStructure to get all the different model structures.
        shifts (int): if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
            and apply the oppositve shift to the output. This is repeated `shifts` time and
            all predictions are averaged. This effectively makes the model time equivariant
            and improves SDR by up to 0.2 points.
        split (bool): if True, the input will be broken down in 8 seconds extracts
            and predictions will be performed individually on each and concatenated.
            Useful for model with large memory footprint like Tasnet.
        jobs (int): If the model is evaluated on the CPU, then we might want to use multiple threads :D
        overlap (float): Amount of overlap (seconds) between the splits.
        show_progress (bool) Whether to display the progress bar.
        force_cpu (bool): Force the pipeline to use cpu. If set to false, will use ..device"""
        from demucs.apply import apply_model
        assert audio.size(0) == self.nchannels
        assert len(audio.shape) == 2

        model = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ref = audio.mean(0)
        wav = (audio - ref.mean()) / ref.std()
        components = apply_model(model, wav[None],
                                 shifts=shifts,
                                 split=split,
                                 overlap=overlap,
                                 progress=show_progress,
                                 num_workers=jobs,
                                 device=device)[0]
        components = components * ref.std() + ref.mean()

        # Get the name indices - i.e. components[name[i]] is the audio for the `name` component
        name_indices: dict[str, int] = {name: i for i, name in enumerate(model.sources)}
        return components, name_indices

    def separate(self, audio: Audio, **kwargs) -> DemucsCollection:
        """Performs the demucs audio separation pipeline.
        Play with hyperparameters with the pipeline() method.
        All kwargs will be forwarded to pipeline.

        Returns: a demucs audio collection. The returned audio is guaranteed to have the same sample rate as the original audio"""
        audio_ = audio.resample(self.sample_rate).to_nchannels(AudioMode.MONO if self.nchannels == 1 else AudioMode.STEREO)
        components, name_indices = self.pipeline(audio_.data, **kwargs)
        dct = {k: Audio(components[v].clone(), self.sample_rate).resample(int(audio.sample_rate)).pad(audio.nframes) for k, v in name_indices.items()}
        return DemucsCollection(
            vocals=dct['vocals'],
            bass=dct['bass'],
            other=dct['other'],
            drums=dct['drums']
        )


def demucs_separate(
    audio: Audio, *,
    use_gpu: bool | None = None,
    model: str = "htdemucs",
    verbose: bool = False,
) -> DemucsCollection:
    with tempfile.TemporaryDirectory() as tempdir:
        audio_path = os.path.join(tempdir, "audio.wav")
        audio.save(audio_path)
        cmd = ["demucs"]
        if use_gpu is True:
            cmd += ["-d", "cuda"]
        elif use_gpu is False:
            cmd += ["-d", "cpu"]
        if verbose:
            cmd += ["-v"]
        cmd += ["-o", tempdir]
        cmd += ["-n", model, audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout and verbose:
            print("stdout:", result.stdout)
        if result.stderr:
            print("stderr:", result.stderr)
        out_dir = os.path.join(tempdir, model, "audio")

        def pad_or_raise(audio: Audio, target_length: int, part_name: str) -> Audio:
            current_length = audio.nframes
            if abs(current_length - target_length) > 100:
                raise RuntimeError(
                    f"Expected {target_length} frames ({part_name}), got {current_length}."
                )
            return audio.pad(target_length)
        drums = Audio.load(os.path.join(out_dir, "drums.wav"))
        bass = Audio.load(os.path.join(out_dir, "bass.wav"))
        other = Audio.load(os.path.join(out_dir, "other.wav"))
        vocals = Audio.load(os.path.join(out_dir, "vocals.wav"))
        return DemucsCollection(
            drums=pad_or_raise(drums, audio.nframes, "drums"),
            bass=pad_or_raise(bass, audio.nframes, "bass"),
            other=pad_or_raise(other, audio.nframes, "other"),
            vocals=pad_or_raise(vocals, audio.nframes, "vocals"),
        )
