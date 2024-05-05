import asyncio
import time

import websockets
import pyaudio
import json

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

        print("Recording and playing back...")

        listening = True

        try:
            while True:
                if listening:
                    # Read data from microphone
                    input_data = input_stream.read(1024, exception_on_overflow=False)
                    # Append header to the input data
                    header = bytearray([1, 0, 0, 0])
                    await websocket.send(header + input_data)

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