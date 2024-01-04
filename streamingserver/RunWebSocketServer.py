import asyncio
import websockets
import pyaudio
import wave
import os
import sys
from pydub import AudioSegment
import numpy as np
import whisper
import audioop
import torch

import simpleaudio as sa


# Global variable to determine the mode of operation
TEST_MODE = "--test" in sys.argv
#TEST_MODE = True

# if we are not in test mode, we need to load whisper before any connections are made
if not TEST_MODE:
    whisper_model = whisper.load_model('base')


# Function to stream MP3 file audio in test mode
async def stream_mp3(websocket, mp3_file):
    stereo_audio = AudioSegment.from_mp3(mp3_file)
    mono_audio = stereo_audio.set_channels(1)
    raw_data = mono_audio.raw_data
    chunk_size = 2048

    try:
        while True:  # Loop to continuously stream the MP3 file
            for i in range(0, len(raw_data), chunk_size):
                await websocket.send(raw_data[i:i + chunk_size])
    except websockets.ConnectionClosed:
        print("Connection for MP3 streaming closed")

# Function to generate a placeholder tone (or silence) for outgoing audio
def generate_audio_data(duration_seconds=0.1, sample_rate=44100, tone=False):
    if tone:
        # Generate a simple tone
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
        tone_freq = 440  # Tone frequency in Hz (A4)
        audio_signal = np.sin(tone_freq * t * 2 * np.pi)
        audio_signal *= 32767 / np.max(np.abs(audio_signal))  # Normalize to 16-bit range
        audio_data = audio_signal.astype(np.int16).tobytes()
    else:
        # Generate silence
        audio_data = bytes(int(sample_rate * duration_seconds) * 2)  # 2 bytes per sample (16-bit audio)
    return audio_data

def is_silence(data, threshold=350):
    # Calculate the RMS value
    rms = audioop.rms(data, 2)  # Assuming 16-bit audio (2 bytes per sample)
    return rms < threshold

# Function to play an audio buffer
def play_audio_buffer(audio_buffer):
    # Convert the buffer to a NumPy array and normalize
    audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
    # Convert it back to int16 for playback
    audio_int16 = (audio_np * 32768).astype(np.int16)
    # Play the audio
    play_obj = sa.play_buffer(audio_int16, 1, 2, 16000)  # 1 channel, 2 bytes/sample, 44100 Hz
    play_obj.wait_done()  # Wait for playback to finish

# Coroutine for handling incoming audio data
async def handle_incoming_audio(websocket):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    if TEST_MODE:
        output_file = "output_audio_file.wav"
        # Prepare the output file
        wf = wave.open(output_file, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)

        try:
            while True:
                data = await websocket.recv()
                wf.writeframes(data)
        except websockets.ConnectionClosed:
            print("Connection for incoming audio closed")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
    else:
        # Define time-based thresholds in seconds
        SOME_SPEECH_THRESHOLD = 1.00
        SOME_SILENCE_THRESHOLD = 0.67
        OVERLAP = 0.5
        sr = 16000

        # Calculate byte lengths based on time durations
        bytes_per_sample = 2  # 16-bit audio
        speech_threshold_bytes = int(SOME_SPEECH_THRESHOLD * sr * bytes_per_sample)
        silence_threshold_bytes = int(SOME_SILENCE_THRESHOLD * sr * bytes_per_sample)
        overlap_bytes = int(OVERLAP * sr * bytes_per_sample)

        try:
            speech_frames = []
            total_speech_duration = 0
            silence_duration = 0

            while True:
                data = await websocket.recv()

                if is_silence(data):
                    silence_duration += len(data)
                    # Check for end of speech segment
                    if total_speech_duration >= speech_threshold_bytes and silence_duration >= silence_threshold_bytes:
                        # Process accumulated speech frames
                        audio_data = b''.join(speech_frames)
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        result = whisper_model.transcribe(audio_np)
                        print(result["text"])

                        # Reset buffers and counters
                        speech_frames = []
                        total_speech_duration = 0
                    # Continue accumulating silence if not enough speech yet
                else:
                    # Reset silence duration and accumulate speech
                    silence_duration = 0
                    speech_frames.append(data)
                    total_speech_duration += len(data)

        except websockets.ConnectionClosed:
            print("Connection for incoming audio closed")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

# Coroutine for handling outgoing audio data
async def handle_outgoing_audio(websocket):
    if TEST_MODE:
        mp3_file = "your_audio_file.mp3"
        await stream_mp3(websocket, mp3_file)
    else:
        while True:
            audio_data = generate_audio_data(tone=False)  # Set 'tone=True' to send a tone instead of silence
            try:
                await websocket.send(audio_data)
            except websockets.ConnectionClosed:
                print("Connection for outgoing audio closed")
                break
            await asyncio.sleep(0.1)  # Sleep for 0.1 second, adjust as needed

# WebSocket server handler
async def audio_handler(websocket, path):
    print("New connection established")
    incoming_task = asyncio.ensure_future(handle_incoming_audio(websocket))
    outgoing_task = asyncio.ensure_future(handle_outgoing_audio(websocket))

    await asyncio.gather(incoming_task, outgoing_task)
    print("Connection handler completed")


async def main():
    async with websockets.serve(audio_handler, "localhost", 8765):
        print("Server started. Awaiting connections...")
        await asyncio.Future()  # This will keep the server running indefinitely

asyncio.run(main())
