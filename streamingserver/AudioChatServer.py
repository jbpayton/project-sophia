from flask import Flask, request, jsonify, send_file, make_response, Response
import io
import numpy as np
import whisper
from queue import Queue
from threading import Thread
from NewTypeAgent import NewTypeAgent
import requests
import os
import json
import tempfile
import soundfile as sf

app = Flask(__name__)

# Load the Whisper model
whisper_model = whisper.load_model('base')

# Initialize the NewTypeAgent
agent_dict = {}

# Create queues for text and audio
text_queue = Queue()
audio_queue = Queue()

TEST_MODE = True

@app.route('/text', methods=['POST'])
def handle_text():
    data = request.get_json()
    text = data['text']
    agent_name = request.form['agent_name']

    if agent_name not in agent_dict:
        agent_dict[agent_name] = NewTypeAgent(agent_name)

    agent = agent_dict[agent_name]

    # Process the text using the NewTypeAgent
    response, mood, inner_monologue, actions = agent.send(text)

    # Create a JSON object to send back to the client
    response_data = {
        'response': response,
        'mood': mood,
        'inner_monologue': inner_monologue,
        'actions': actions
    }

    return jsonify(response_data)

@app.route('/audio', methods=['POST'])
def handle_audio():
    audio_data = request.files['audio'].read()
    agent_name = request.form['agent_name']

    if TEST_MODE:
        # Save the audio data to a temporary WAV file
        # Create a file-like object from the audio data
        audio_io = io.BytesIO(audio_data)

        # Convert the audio data to a numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe the audio using the Whisper model
        result = whisper_model.transcribe(audio_np)
        text = result["text"]

        # Create a JSON object to send back to the client
        response_data = {
            'response': text,
            'mood': "neutral",
            'inner_monologue': "",
            'actions': ""
        }

        # Create a response object with the audio data
        response = Response(audio_io.getvalue(), mimetype='audio/wav')
        response.headers.set('Content-Disposition', 'attachment', filename='response.wav')

        # Attach the JSON data to the response headers
        response.headers['X-JSON'] = json.dumps(response_data)

        return response


    # check if the agent is already initialized
    if agent_name not in agent_dict:
        agent_dict[agent_name] = NewTypeAgent(agent_name)

    agent = agent_dict[agent_name]
    voice_name = agent.profile['voice']

    # Convert the audio data to a numpy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    # Transcribe the audio using the Whisper model
    result = whisper_model.transcribe(audio_np)
    text = result["text"]

    # Process the text using the NewTypeAgent
    response, mood, inner_monologue, actions = agent.send(text)

    # Generate the response audio using the TTS API
    response_audio = generate_response_audio(response, voice_name, mood)

    # Create a JSON object to send back to the client
    response_data = {
        'response': response,
        'mood': mood,
        'inner_monologue': inner_monologue,
        'actions': actions
    }

    # Create a file-like object from the response audio data
    audio_io = io.BytesIO(response_audio)

    # Create a response object with the audio data
    response = Response(audio_io.getvalue(), mimetype='audio/wav')
    response.headers.set('Content-Disposition', 'attachment', filename='response.wav')

    # Attach the JSON data to the response headers
    response.headers['X-JSON'] = json.dumps(response_data)

    return response

def generate_response_audio(response, speaker_name, mood="neutral"):
    # Check if a mood-specific WAV file exists
    mood_wav_file = f"{speaker_name}_{mood}.wav"
    if os.path.exists(mood_wav_file):
        speaker_wav = mood_wav_file
    else:
        speaker_wav = f"{speaker_name}.wav"

    # Make a POST request to the TTS API
    api_url = os.environ['LOCAL_TTS_API_BASE'] + "/tts_to_audio/"
    payload = {
        "text": response,
        "speaker_wav": speaker_wav,
        "language": "en"
    }
    api_response = requests.post(api_url, json=payload)

    if api_response.status_code == 200:
        # Get the generated audio data from the response
        audio_data = api_response.content

        # Save the audio data to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        # Read the WAV file using soundfile
        audio_frames, sample_rate = sf.read(temp_file_path, dtype='float32')

        # Convert float32 samples to int16
        audio_frames = (audio_frames * 32767).astype('int16')

        # Convert audio frames to a byte array
        audio_bytes = audio_frames.tobytes()

        return audio_bytes
    else:
        print(f"TTS API request failed with status code: {api_response.status_code}")
        return None

if __name__ == '__main__':
    app.run()