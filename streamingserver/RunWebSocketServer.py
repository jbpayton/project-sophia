import asyncio
import queue
import threading
import json
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

# we need to define a header for our outgoing websocket messages
# this header will be used to determine the type of message being sent
# and how it should be handled

# the header is a 4-byte string
# the first byte is the message type
# the remaining 3 bytes are undefined for now
# the message type is one of the following:
#   0 - text
#   1 - audio
#   2 - avatar action
#   3 - client message


# Global variable to determine the mode of operation
from NewTypeAgent import NewTypeAgent
from audio import TTSClient

from util import load_secrets
load_secrets("../secrets.json")

TEST_MODE = "--test" in sys.argv
#TEST_MODE = True

# if we are not in test mode, we need to load whisper before any connections are made
if not TEST_MODE:
    agent = NewTypeAgent("Sophia")
    whisper_model = whisper.load_model('base')

text_queue = queue.Queue()  # Queue for text to be synthesized
incoming_message_queue = queue.Queue()  # Queue for incoming text messages

audio_queue = queue.Queue()  # Queue for synthesized audio data
action_queue = queue.Queue()  # Queue for avatar actions
client_message_queue = queue.Queue()  # Queue for text to be sent to the client


# Function to stream MP3 file audio in test mode
async def stream_mp3(websocket, mp3_file):
    stereo_audio = AudioSegment.from_mp3(mp3_file)
    mono_audio = stereo_audio.set_channels(1)
    raw_data = mono_audio.raw_data
    chunk_size = 2048

    try:
        while True:  # Loop to continuously stream the MP3 file
            for i in range(0, len(raw_data), chunk_size):
                # append the header to the chunk
                header = bytearray([1, 0, 0, 0])
                await websocket.send(header + raw_data[i:i + chunk_size])
                # await websocket.send(raw_data[i:i + chunk_size])
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
async def handle_incoming_data(websocket):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

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
        SOME_SPEECH_THRESHOLD = 0.75
        SOME_SILENCE_THRESHOLD = 0.67
        sr = 16000

        # Calculate byte lengths based on time durations
        bytes_per_sample = 2  # 16-bit audio
        speech_threshold_bytes = int(SOME_SPEECH_THRESHOLD * sr * bytes_per_sample)
        silence_threshold_bytes = int(SOME_SILENCE_THRESHOLD * sr * bytes_per_sample)

        try:
            speech_frames = []
            total_speech_duration = 0
            silence_duration = 0

            while True:
                data = await websocket.recv()

                # look at the first byte to determine what type of data it is
                data_type = data[0]
                if data_type == 1:
                    incoming_audio = data[4:]

                    if is_silence(incoming_audio):
                        silence_duration += len(incoming_audio)
                        # Check for end of speech segment
                        if total_speech_duration >= speech_threshold_bytes and silence_duration >= silence_threshold_bytes:
                            # Process accumulated speech frames
                            audio_data = b''.join(speech_frames)
                            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                            result = whisper_model.transcribe(audio_np)
                            print(result["text"])
                            text_queue.put(result["text"])

                            # Reset buffers and counters
                            speech_frames = []
                            total_speech_duration = 0
                        # Continue accumulating silence if not enough speech yet
                    else:
                        # Reset silence duration and accumulate speech
                        silence_duration = 0
                        speech_frames.append(incoming_audio)
                        total_speech_duration += len(incoming_audio)
                elif data_type == 3:
                    # Put the text in the incoming message queue
                    decoded_string = data[4:].decode("utf-8")
                    incoming_message_queue.put(decoded_string)
                else:
                    print("Unknown data type received")

        except websockets.ConnectionClosed:
            print("Connection for incoming audio closed")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


# Coroutine for handling outgoing audio data
async def handle_outgoing_data(websocket):
    if TEST_MODE:
        mp3_file = "your_audio_file.mp3"
        await stream_mp3(websocket, mp3_file)
    else:
        CHUNK_SIZE = 2048  # Size of each audio chunk (in bytes)
        SAMPLE_RATE = 16000  # Sampling rate of the audio (in Hz)
        FRAME_DURATION = CHUNK_SIZE / SAMPLE_RATE  # Duration of each audio chunk (in seconds)
        #adjust for latency
        FRAME_DURATION = FRAME_DURATION - 0.03

        while True:

            if not audio_queue.empty():
                audio_data = audio_queue.get()
                # Split audio_data into smaller chunks
                for i in range(0, len(audio_data), CHUNK_SIZE):
                    # send audio data with the header

                    header = bytearray([1, 0, 0, 0])
                    await websocket.send(header + audio_data[i:i+CHUNK_SIZE])
                    #await asyncio.sleep(FRAME_DURATION)  # Control the rate of sending
            if not action_queue.empty():
                action = action_queue.get()
                header = bytearray([2, 0, 0, 0])
                await websocket.send(header + action.encode())
            if not client_message_queue.empty():
                message = client_message_queue.get()
                header = bytearray([0, 0, 0, 0])
                await websocket.send(header + message.encode())
            else:
                # Generate and send filler audio when there's no TTS data
                #filler_audio = generate_audio_data(duration_seconds=FRAME_DURATION/2, sample_rate=16000, tone=False)
                #await websocket.send(filler_audio)
                await asyncio.sleep(FRAME_DURATION/2)

# WebSocket server handler
async def audio_handler(websocket, path):
    print("New connection established")
    incoming_task = asyncio.ensure_future(handle_incoming_data(websocket))
    outgoing_task = asyncio.ensure_future(handle_outgoing_data(websocket))

    await asyncio.gather(incoming_task, outgoing_task)
    print("Connection handler completed")


def tts_processor():
    azure_speech = TTSClient(voice=agent.profile['voice'], silent=True)
    while True:
        text = text_queue.get()  # Wait for text from the STT process
        if text is None:
            break  # None is used as a signal to stop the thread

        response, mood, inner_monologue, actions = agent.send(text)
        print(response)

        # create a json object to send to the client
        if mood is None:
            mood = "neutral"
        emote_json = json.dumps({"actionName": "Emote", "emoteName": mood})
        action_queue.put(emote_json)

        for action in actions:
            print("Sending action: " + str(action) + " to client")
            # make sure the action is a string
            action = json.dumps(action)
            action_queue.put(action)

        client_message_queue.put(response)

        audio_data = azure_speech.speak_to_stream(response, mood=mood)
        if audio_data is not None:
            audio_queue.put(audio_data)  # Place the TTS audio into the outgoing queue

# Function to start the TTS processor thread
def start_tts_processor():
    thread = threading.Thread(target=tts_processor)
    thread.start()
    return thread

def observation_processor():
    while True:
        observation_text = incoming_message_queue.get()  # Wait for text from client
        if observation_text is None:
            break  # None is used as a signal to stop the thread
        print("New observation: " + observation_text)
        agent.accept_observation(observation_text)

def start_observation_processor():
    thread = threading.Thread(target=observation_processor)
    thread.start()
    return thread

async def main():
    tts_thread = start_tts_processor()  # Start the TTS processor thread
    observation_thread = start_observation_processor()
    #async with websockets.serve(audio_handler, "localhost", 8765)
    #listen on all interfaces
    async with websockets.serve(audio_handler, None, 8765):
        print("Server started. Awaiting connections...")
        await asyncio.Future()  # This will keep the server running indefinitely
    tts_thread.join()  # Wait for the TTS processor thread to complete

asyncio.run(main())
