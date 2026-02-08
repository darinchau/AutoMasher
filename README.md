<img
    style="display: block;
           margin-left: auto;
           margin-right: auto;
           width: 30%;"
    src="resources/assets/auto_mashup.png"
    alt="Auto Masher">
</img>

# What is Auto Masher?
Auto Masher is a research project that aims to create pop song mashups with AI-assisted music information retrieval and analysis. In this project, we have compiled a dataset of pop songs and their corresponding chords and beats. The user will submit a song from YouTube, and the pipeline will automatically find the best song to mashup with the user's song.

Our technical paper "Retrieval-based automatic mashup generation with deep learning-guided features" has been accepted by the 25th International Congress on Acoustics/188th Meeting of the Acoustical
Society of America (ICA2025 New Orleans)

Our paper [DOI: https://doi.org/10.1121/2.0002071](https://doi.org/10.1121/2.0002071) has been awarded Best Student Paper by Proceedings of Meetings on Acoustics (POMA)


Auto Masher has been awarded the second-runner up in Best Final Year Project Award in the Department of Computer Science, HKUST in the year 2023-2024.

# Installation

## Requirements
- Python 3.12 (Ideal version, should theoretically work with Python 3.10 and 3.11, beyond 3.13 the aifc libraries are removed so librosa kinda doesn't work)
- A decent GPU with some VRAM (>= 4GB) if possible
- C++14 compatible compiler (for `madmom` library)
- `ffmpeg` installed on your system and added to your PATH

## Minimal Run
1. Clone the repository:
```bash
git clone ...
cd AutoMasher
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Install `mediainfo` for metadata extraction: refer to [https://mediaarea.net/en/MediaInfo](https://mediaarea.net/en/MediaInfo) for installation instructions. The binary `mediainfo` should be accessible in your system's PATH.

4. `python main.py`
