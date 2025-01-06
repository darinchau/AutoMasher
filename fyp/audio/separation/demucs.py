from .. import DemucsCollection, Audio
import os
import subprocess
import tempfile

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
