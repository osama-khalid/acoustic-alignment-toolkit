'''
!pip install pytubefix pydub
'''

# Required libraries
import csv
import os
from pytubefix import YouTube
from pydub import AudioSegment
from google.colab import files

# Install required packages
!pip install pytubefix pydub

def download_audio_from_youtube(video_id):
    """
    Downloads audio from a YouTube video and converts it to MP3.
    
    Args:
    video_id (str): The YouTube video ID.
    
    Returns:
    str: The filename of the downloaded MP3.
    """
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    yt = YouTube(video_url)
    
    # Get the first audio stream
    audio_stream = yt.streams.filter(only_audio=True).first()
    
    # Download the audio file
    temp_audio_file = audio_stream.download()
    
    # Convert to MP3
    audio = AudioSegment.from_file(temp_audio_file)
    mp3_filename = f"{video_id}.mp3"
    audio.export(mp3_filename, format="mp3")
    
    # Remove the temporary file
    os.remove(temp_audio_file)
    
    return mp3_filename

def process_video_list(video_ids):
    """
    Process a list of YouTube video IDs, downloading their audio as MP3.
    
    Args:
    video_ids (list): List of YouTube video IDs.
    """
    for video_id in video_ids:
        mp3_file = download_audio_from_youtube(video_id)
        print(f"Downloaded and converted: {mp3_file}")

def download_mp3_files():
    """
    Downloads all MP3 files in the current directory using Google Colab's file download feature.
    """
    for file in os.listdir('.'):
        if file.endswith('.mp3'):
            files.download(file)

# Main execution
if __name__ == "__main__":
    # List of YouTube video IDs to process
    VIDEO_IDs = ['qlaum72JNRA', 'pjW6WKpSCeQ', 'GdSDngmDLmY']
    
    # Process the videos
    process_video_list(VIDEO_IDs)
    
    # Download the resulting MP3 files
    download_mp3_files()