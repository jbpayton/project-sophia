import asyncio
import websockets
import pyaudio

# Function to handle sending and receiving audio
async def handle_audio(uri):
    async with websockets.connect(uri) as websocket:
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
                               rate=44100,
                               output=True)

        print("Recording and playing back...")

        try:
            while True:
                # Read data from microphone
                input_data = input_stream.read(1024, exception_on_overflow=False)
                # Send data to server
                await websocket.send(input_data)

                # Receive data from server
                output_data = await websocket.recv()
                # Play received data
                output_stream.write(output_data)
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
