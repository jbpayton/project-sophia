import asyncio
import websockets
import pyaudio
import wave
import os
import sys
from pydub import AudioSegment
import numpy as np

# Global variable to determine the mode of operation
TEST_MODE = "--test" in sys.argv
#TEST_MODE = True

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

# Coroutine for handling incoming audio data
async def handle_incoming_audio(websocket):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    output_file = "output_audio_file.wav"

    # Prepare the output file
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)

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
