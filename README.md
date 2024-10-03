# Audio Analysis Toolkit

This repository contains a set of Python scripts for various audio analysis tasks, including speaker diarization, YouTube audio downloading, and phoneme/formant analysis.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Scripts](#scripts)
5. [Dependencies](#dependencies)
6. [License](#license)

## Overview

The Audio Analysis Toolkit is a collection of Python scripts designed to perform various audio processing and analysis tasks. It includes tools for:

- Speaker diarization (identifying who spoke when in an audio file)
- Downloading audio from YouTube videos
- Analyzing phonemes and formants in speech

These tools can be used individually or combined for more complex audio analysis workflows.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/audio-analysis-toolkit.git
   cd audio-analysis-toolkit
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Additional system dependencies:
   - Install espeak:
     ```
     sudo apt-get install espeak  # On Ubuntu/Debian
     ```
   - For other systems, please refer to the espeak documentation.

## Usage

Each script in this toolkit can be used independently. Here's a brief overview of how to use each tool:

### Speaker Diarization

```
python speaker_diarization_processor.py
```

This script processes all MP3 files in the `audio` directory and outputs speaker labels in CSV format in the `speaker_labels` directory.

### YouTube Audio Downloader

```
python youtube_audio_downloader.py
```

Edit the `VIDEO_IDs` list in the script to include the YouTube video IDs you want to download. The script will download the audio as MP3 files.

### Phoneme Formant Analyzer

```
python phoneme_formant_analyzer.py
```

This script analyzes phonemes and formants in audio files. It requires speaker label CSV files (generated by the diarization script) and corresponding audio files.

## Scripts

1. `speaker_diarization_processor.py`: Performs speaker diarization on audio files.
2. `youtube_audio_downloader.py`: Downloads audio from YouTube videos.
3. `phoneme_formant_analyzer.py`: Analyzes phonemes and formants in speech.

For detailed information about each script, please refer to the comments within the script files.

## Dependencies

Main dependencies include:

- PyTorch
- Transformers
- librosa
- pyannote.audio
- pytubefix
- pydub
- praat-parselmouth
- phonemizer

For a complete list of dependencies, see the `requirements.txt` file.
