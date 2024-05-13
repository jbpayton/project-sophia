import sys

from flask import Flask, request, jsonify, Response
import io
import numpy as np
import whisper
import requests
import os
import json
import tempfile
import soundfile as sf
import util
from AgentManager import AgentManager

app = Flask(__name__)

# Load the Whisper model
whisper_model = whisper.load_model('base')

TEST_MODE = False


@app.route('/text', methods=['POST'])
def handle_text():
    data = request.get_json()
    text = data['text']
    agent_name = data['agent_name']
    user_name = data['sender']
    # if data contains the property 'sound_response', the server should return an audio response
    audio_response = 'audio_response' in data

    if not AgentManager.does_agent_exist(agent_name):
        return jsonify({'error': 'Agent does not exist'})

    # Process the text using the AIAgent
    message = AgentManager.send_to_agent(agent_name, text, user_name)

    return formulate_response(agent_name, audio_response, message)


def formulate_response(agent_name, audio_response, message):
    response, mood, inner_monologue, actions = message
    # Strip emojis from the response text
    response = util.remove_emojis(response)
    # Generate the response audio using the TTS API
    if not audio_response:
        # Create a JSON object to send back to the client (but it must be in the response headers)
        response_data = {
            'response': response,
            'mood': mood,
            'inner_monologue': inner_monologue,
            'actions': actions
        }

        return jsonify(response_data)
    response_audio = generate_response_audio(response, AgentManager.get_agent_voice(agent_name), mood)
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
    http_response = Response(audio_io.getvalue(), mimetype='audio/wav')
    http_response.headers.set('Content-Disposition', 'attachment', filename='response.wav')
    # Attach the JSON data to the response headers
    http_response.headers['X-JSON'] = json.dumps(response_data)
    return http_response


@app.route('/audio', methods=['POST'])
def handle_audio():
    audio_data = request.files['audio'].read()
    agent_name = request.form['agent_name']
    user_name = request.form['sender']

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
    if not AgentManager.does_agent_exist(agent_name):
        return jsonify({'error': 'Agent does not exist'})

    # Convert the audio data to a numpy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    # Transcribe the audio using the Whisper model
    result = whisper_model.transcribe(audio_np)
    text = result["text"]

    # Process the text using the AIAgent
    response, mood, inner_monologue, actions = AgentManager.send_to_agent(agent_name, text, user_name)

    # Strip emojis from the response text
    response = util.remove_emojis(response)

    # Generate the response audio using the TTS API
    response_audio = generate_response_audio(response, AgentManager.get_agent_voice(agent_name), mood)

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


@app.route('/getqueued', methods=['POST'])
def get_queued():
    data = request.get_json()
    agent_name = data['agent_name']
    # if data contains the property 'sound_response', the server should return an audio response
    audio_response = 'audio_response' in data

    if not AgentManager.does_agent_exist(agent_name):
        return jsonify({'error': 'Agent does not exist'})

    queued = AgentManager.get_queued_message(agent_name)

    if queued is None:
        return jsonify({'information': 'No queued message'})

    sender_agent, message = queued

    return formulate_response(sender_agent, audio_response, message)


def generate_response_audio(response, speaker_name, mood="neutral"):
    # Check if a mood-specific WAV file exists
    mood_wav_file = f"../xtts-api-server/speakers/{speaker_name}_{mood}.wav"
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
    # handle a single parameter for the port number, 5000 is the default
    if len(sys.argv) > 1:
        print(f"Port number provided: {sys.argv[1]}")
        port = int(sys.argv[1])
    else:
        print("No port number provided, defaulting to 5000")
        port = 5000

    util.load_secrets("secrets.json")
    app.run(host='0.0.0.0', port=port)
