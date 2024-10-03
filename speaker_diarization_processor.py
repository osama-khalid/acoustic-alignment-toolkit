#source python10/bin/activate

# Import required libraries
import csv
import os
import torch
from pyannote.audio import Pipeline
import tqdm
import datetime

def setup_device():
    """
    Set up and return the appropriate device (GPU if available, otherwise CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def initialize_diarization_pipeline(device):
    """
    Initialize and return the speaker diarization pipeline.
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=""
    )
    pipeline.to(device)
    return pipeline

def convert_seconds_to_timestamp(seconds):
    """
    Convert seconds to HH:MM:SS format.
    """
    return str(datetime.timedelta(seconds=seconds)).split('.')[0]

def process_audio_file(pipeline, file_path):
    """
    Process a single audio file and return diarization results.
    """
    print(f"Starting diarization for {file_path}...")
    return pipeline(file_path)

def save_diarization_results(results, output_file):
    """
    Save diarization results to a CSV file.
    """
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["speaker", "start", "end"])
        for turn, _, speaker in results.itertracks(yield_label=True):
            start_time = convert_seconds_to_timestamp(turn.start)
            end_time = convert_seconds_to_timestamp(turn.end)
            print(f"[{start_time} --> {end_time}] {speaker}")
            writer.writerow([speaker, start_time, end_time])

def main():
    # Set up the device
    device = setup_device()

    # Initialize the diarization pipeline
    pipeline = initialize_diarization_pipeline(device)

    # Create output directory if it doesn't exist
    os.makedirs('speaker_labels', exist_ok=True)

    # Process all MP3 files in the 'audio' directory
    audio_files = [f for f in os.listdir('audio') if f.endswith('.mp3')]
    for audio_file in tqdm.tqdm(audio_files):
        input_path = os.path.join('audio', audio_file)
        output_path = os.path.join('speaker_labels', audio_file.replace('.mp3', '.csv'))

        # Process the audio file
        diarization_results = process_audio_file(pipeline, input_path)

        # Save the results
        save_diarization_results(diarization_results, output_path)

if __name__ == "__main__":
    main()