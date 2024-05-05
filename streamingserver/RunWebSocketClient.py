import asyncio
import time

import websockets
import pyaudio
import json
import numpy as np
import audioop

def is_silence(data, threshold=350):
    # Calculate the RMS value
    rms = audioop.rms(data, 2)  # Assuming 16-bit audio (2 bytes per sample)
    return rms < threshold

# Function to handle sending and receiving audio
async def handle_audio(uri):
    async with websockets.connect(uri, ping_timeout=None, ping_interval=None) as websocket:
        # Set up audio input (microphone)
        p = pyaudio.PyAudio()
        input_stream = p.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=16000,
                              input=True,
                              frames_per_buffer=1024)

        # Set up audio output (speaker)
        output_stream = p.open(format=pyaudio.paInt16,
                               channels=1,
                               rate=24000,
                               output=True)

        # Define time-based thresholds in seconds
        SILENCE_DURATION_THRESHOLD = 1.2  # Adjust this value based on your requirements
        sr = 16000

        # Calculate byte lengths based on time durations
        silent_frames_threshold = int(SILENCE_DURATION_THRESHOLD * sr / 1024)

        speech_frames = []
        silent_frames_count = 0

        print("Recording and playing back...")

        listening = True
        non_silence_present = False

        try:
            while True:
                if listening:
                    # Read data from microphone
                    input_data = input_stream.read(1024, exception_on_overflow=False)
                    if is_silence(input_data):
                        silent_frames_count += 1
                        speech_frames.append(input_data)

                        # Check for end of speech segment
                        if silent_frames_count >= silent_frames_threshold and non_silence_present:
                            # Send accumulated speech frames to the server
                            header = bytearray([1, 0, 0, 0])
                            await websocket.send(header + b''.join(speech_frames))
                            print("Sending speech data" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

                            # Reset buffers and counters
                            speech_frames = []
                            silent_frames_count = 0
                            non_silence_present = False
                        # Continue accumulating silence if not enough speech yet
                    else:
                        # Reset silence duration and accumulate speech
                        silent_frames_count = 0
                        if not non_silence_present:
                            non_silence_present = True
                            # get rid of the silence at the beginning of the speech
                            speech_frames = []
                        speech_frames.append(input_data)

                '''
                # Receive data from server
                output_data = await websocket.recv()

                # Extract the message type from the header
                message_type = output_data[0]

                if message_type == 1:  # Audio data
                    audio_data = output_data[4:]
                    # Pause listening during playback
                    if listening:
                        print("Speaking, pausing listening" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                        listening = False
                    # Play received audio data
                    output_stream.write(audio_data)
                elif message_type == 0:  # Text message
                    message = output_data[4:].decode("utf-8")
                    print(f"Received text message: {message}")
                elif message_type == 2:  # Avatar action
                    action = output_data[4:].decode("utf-8")
                    action_data = json.loads(action)
                    print(f"Received avatar action: {action_data}")
                elif message_type == 3:  # Keepalive message
                    print(f"Received client message")
                elif message_type == 4:  # Keepalive message
                    # Resume listening after playback
                    if not listening:
                        listening = True
                        print("Resuming listening")
                else:
                    print(f"Received unknown message type: {message_type}")
                '''

        except websockets.ConnectionClosed:
            print("Connection to server closed")
        finally:
            # Stop and close the streams and PyAudio
            input_stream.stop_stream()
            input_stream.close()
            output_stream.stop_stream()
            output_stream.close()
            p.terminate()

# The URI should match the one used by your server
uri = "ws://localhost:8765"

# Connect to the WebSocket server and handle audio
asyncio.get_event_loop().run_until_complete(handle_audio(uri))