import json

import requests
import pyaudio
import io
import wave
import numpy as np
import audioop
import time

# Constants
SILENCE_THRESHOLD = 350
SILENCE_DURATION_THRESHOLD = 1.2
SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 16000

CHUNK_SIZE = 2048
SILENT_FRAMES_THRESHOLD = int(SILENCE_DURATION_THRESHOLD * SAMPLE_RATE / CHUNK_SIZE)
AGENT_NAME = 'Sophia'


# Function to check if audio data is silence
def is_silence(data, threshold=SILENCE_THRESHOLD):
    rms = audioop.rms(data, 2)
    return rms < threshold


# Function to play the received audio data
def play_audio(audio_data):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=OUTPUT_SAMPLE_RATE, output=True)
    stream.write(audio_data)
    stream.stop_stream()
    stream.close()
    p.terminate()


# Function to send audio to the server
def send_audio(audio_data, agent_name):
    url = 'http://localhost:5000/audio'
    files = {'audio': ('audio.wav', audio_data, 'audio/wav')}
    data = {'agent_name': agent_name}
    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        # Get the response audio and JSON data
        response_audio = response.content
        # Get the response JSON data from the response headers
        response_json = json.loads(response.headers.get('X-JSON'))

        # Play the response audio
        play_audio(response_audio)

        # Print the response JSON data
        print(response_json)
    else:
        print(f"Request failed with status code: {response.status_code}")


# Function to send text to the server
def send_text(text, agent_name):
    url = 'http://localhost:5000/text'
    data = {'text': text, 'agent_name': agent_name}
    response = requests.post(url, json=data)

    if response.status_code == 200:
        response_json = response.json()
        print(response_json)
    else:
        print(f"Request failed with status code: {response.status_code}")


# Main function
def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    print("Listening...")

    non_silence_present = False
    silent_frames_count = 0
    speech_frames = []

    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

        if is_silence(data):
            silent_frames_count += 1
            speech_frames.append(data)
            if silent_frames_count >= SILENT_FRAMES_THRESHOLD and non_silence_present:
                stream.stop_stream()
                print("Silence detected. Processing speech...")
                audio_data = b''.join(speech_frames)

                send_audio(audio_data, AGENT_NAME)  # Send the audio data to the server

                # resume listening
                stream.start_stream()
                non_silence_present = False
                speech_frames = []
                silent_frames_count = 0
        else:
            silent_frames_count = 0
            if not non_silence_present:
                non_silence_present = True
                # get rid of the silence at the beginning of the speech
                speech_frames = []
            speech_frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == '__main__':
    main()