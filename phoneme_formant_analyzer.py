'''
!gdown 16_urp9tksq4jTNRmtEfwyCVhIjSMUuSD
!gdown 1K-lubLO5cGYTwYUWqDOEm2eI3q4yoHOF

!unzip audio.zip -d ./
!unzip speaker_labels.zip -d ./

!apt-get install -y espeak
!pip install phonemizer
!pip install praat-parselmouth
!pip install transformers torch librosa phonemizer


'''
import parselmouth
from parselmouth.praat import call
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2PhonemeCTCTokenizer
import tqdm
import csv
import os
import pickle

def analyze_vowel_formants(audio_np, sample_rate):
    """
    Analyze formants of a vowel segment using Praat.
    
    Args:
    audio_np (np.array): Audio data as a numpy array
    sample_rate (int): Sampling rate of the audio
    
    Returns:
    dict: Dictionary containing vowel onset, mean pitch, mean F1, and mean F2
    """
    # Ensure minimum duration for analysis
    min_duration = 0.10666666666666667  # 6.4 / 60 Hz
    current_duration = len(audio_np) / sample_rate
    if current_duration < min_duration:
        repetitions = int(np.ceil(min_duration / current_duration)) * 10
        audio_np = np.tile(audio_np, repetitions)

    # Create Praat Sound object and process
    sound = parselmouth.Sound(audio_np, sampling_frequency=sample_rate)
    if sound.sampling_frequency > 11025:
        sound = sound.resample(11025)
    
    # Filter and analyze intensity
    sound_filtered = call(sound, "Filter (one formant)", 1000, 500)
    intensity = sound_filtered.to_intensity(minimum_pitch=60, time_step=0.01)
    intensity_values = intensity.values[0]
    intensity_times = intensity.xs()

    # Find vowel onsets
    intensity_derivative = np.gradient(intensity_values, intensity_times)
    peaks = np.where((intensity_derivative[:-1] > 0) & (intensity_derivative[1:] <= 0))[0]
    rises = np.where((intensity_derivative[:-2] < intensity_derivative[1:-1]) &
                     (intensity_derivative[1:-1] > intensity_derivative[2:]))[0]
    
    max_intensity = np.max(intensity_values)
    threshold = 8
    min_intensity = max_intensity - threshold

    vowel_onsets = []
    for peak in peaks:
        if intensity_values[peak] > min_intensity:
            rise = rises[rises < peak][-1] if any(rises < peak) else peak
            onset_time = (intensity_times[rise] + intensity_times[peak]) / 2
            vowel_onsets.append(onset_time)

    if not vowel_onsets:
        return None

    # Analyze the first vowel onset
    vowel_start = vowel_onsets[0]
    vowel_end = min(vowel_start + 0.07, sound.xmax)
    vowel = sound.extract_part(from_time=vowel_start, to_time=vowel_end)

    # Pitch and formant analysis
    pitch = vowel.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    formants = vowel.to_formant_burg(time_step=0.025, max_number_of_formants=5,
                                     maximum_formant=5500, window_length=0.025, pre_emphasis_from=50)
    mean_f1 = call(formants, "Get mean", 1, 0, 0, "Hertz")
    mean_f2 = call(formants, "Get mean", 2, 0, 0, "Hertz")

    return {
        "vowel_onset": vowel_start,
        "mean_pitch": mean_pitch,
        "mean_f1": mean_f1,
        "mean_f2": mean_f2
    }

def timestamp_to_seconds(time_str):
    """
    Convert timestamp string to seconds.
    
    Args:
    time_str (str): Timestamp in format 'HH:MM:SS'
    
    Returns:
    int: Total seconds
    """
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def parse_speaker_timestamps(file):
    """
    Parse CSV file containing speaker timestamps.
    
    Args:
    file (str): Path to the CSV file
    
    Returns:
    dict: Dictionary with speaker IDs as keys and lists of timestamp ranges as values
    """
    data_set = {}
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            speaker_id = row[0]
            start_time = timestamp_to_seconds(row[1])
            end_time = timestamp_to_seconds(row[2])
            if speaker_id not in data_set:
                data_set[speaker_id] = []
            data_set[speaker_id].append([start_time, end_time])
    return data_set

def extract_phoneme_formants(data_set, audio, model, processor, tokenizer, device):
    """
    Extract formants for phonemes in the audio.
    
    Args:
    data_set (dict): Speaker timestamps
    audio (np.array): Audio data
    model (Wav2Vec2ForCTC): Pretrained model
    processor (Wav2Vec2Processor): Audio processor
    tokenizer (Wav2Vec2PhonemeCTCTokenizer): Phoneme tokenizer
    device (torch.device): Computation device (CPU/GPU)
    
    Returns:
    tuple: (vowel_collection, phoneme_set)
    """
    phoneme_set = set()
    vowel_collection = {}
    
    for speaker in data_set:
        for time_stamp in tqdm.tqdm(data_set[speaker]):
            start_time, end_time = time_stamp
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio_segment = audio[start_sample:end_sample]
            
            input_values = processor(audio_segment, return_tensors="pt", sampling_rate=16000).input_values.to(device)
            
            try:
                with torch.no_grad():
                    logits = model(input_values).logits.cpu()

                predicted_ids = torch.argmax(logits, dim=-1)
                phonemes = tokenizer.batch_decode(predicted_ids)
                frame_phonemes = [tokenizer.decode(id) for id in predicted_ids[0]]

                current_phoneme = frame_phonemes[0]
                start_frame = 0
                frame_duration_ms = len(audio_segment) / len(frame_phonemes)

                for i, phoneme in enumerate(frame_phonemes):
                    if phoneme != current_phoneme:
                        start_time = int(start_frame * frame_duration_ms)
                        end_time = int(i * frame_duration_ms)
                        phoneme_audio = audio_segment[start_time:end_time]

                        if phoneme in audio_inventory:
                            result = analyze_vowel_formants(phoneme_audio, sample_rate=16000)
                            if result:
                                vowel_collection.setdefault(speaker, {}).setdefault(phoneme, []).append([start_time, end_time, result])

                        current_phoneme = phoneme
                        start_frame = i

                phoneme_set.update(phonemes[0])

            except Exception as e:
                print(f"An error occurred: {e}")

    return vowel_collection, list(phoneme_set)

# Main execution
if __name__ == "__main__":
    # Initialize model and processors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft").to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

    # Define audio inventory (list of phonemes to analyze)
    audio_inventory = ['æ', 'o ː', 'ɪ', 'o ɪ', 'i', 'ɑ 5', 'ɨ', 'ə ɜ', 'i 5', 'i ː',
                       'ɔ ː', 'a 5', 'ɑ ː ɹ', 'e ɪ', 'ɔ', 'a', 'ɔ ː ɹ', 'o ː ɹ', 'ɐ',
                       'ɑ', 'u ː', 'ʊ', 'ɔ ɨ', 'ɑ ː', 'a ʊ', 'ə', 'o ʊ', 'o ɜ', 'a ɪ',
                       'ə 2', 'ə l', 'ɜ ː', 'i ɜ', 'ʌ', 'ɪ ɹ', 'u ɨ', 'o', 'u', 'ə 5',
                       'ɚ', 'e', 'u 5', 'ɛ', 'ɛ ɹ']

    # Process audio files
    ROOT = 'audio/'
    for file in os.listdir(ROOT):
        audio, sr = librosa.load(ROOT + file, sr=16000)
        speakers = parse_speaker_timestamps('speaker_labels/' + file.split('.')[0] + '.csv')
        vowels, phonemes = extract_phoneme_formants(speakers, audio, model, processor, tokenizer, device)

        # Save results
        with open(file.split('.')[0] + '.pkl', 'wb') as f:
            pickle.dump(vowels, f)

        print(f"Processed {file}")

# Note: This script assumes that the necessary libraries are installed and
# the required audio files and speaker label CSVs are in the correct directories.