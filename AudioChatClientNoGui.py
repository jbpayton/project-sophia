import json
import requests
import pyaudio
import audioop
import configparser
import threading

# Constants
SILENCE_THRESHOLD = 350
SILENCE_DURATION_THRESHOLD = 1.2
SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000

CHUNK_SIZE = 2048
SILENT_FRAMES_THRESHOLD = int(SILENCE_DURATION_THRESHOLD * SAMPLE_RATE / CHUNK_SIZE)
is_running = False
audio_thread = None

# Load settings from configuration file
config = configparser.ConfigParser()
config.read('config.ini')

if 'Settings' in config:
    SERVER_ADDRESS = config['Settings'].get('ServerAddress', 'http://localhost:5000')
    SPEAKER_NAME = config['Settings'].get('SpeakerName', 'Joey')
    AGENT_NAME = config['Settings'].get('AgentName', 'Sophia')
    INPUT_DEVICE = config['Settings'].get('InputDevice', '')
    OUTPUT_DEVICE = config['Settings'].get('OutputDevice', '')
else:
    SERVER_ADDRESS = 'http://localhost:5000'
    SPEAKER_NAME = 'Joey'
    AGENT_NAME = 'Sophia'
    INPUT_DEVICE = ''
    OUTPUT_DEVICE = ''

# Function to check if audio data is silence
def is_silence(data, threshold=SILENCE_THRESHOLD):
    rms = audioop.rms(data, 2)
    return rms < threshold

# Function to send text to the server
def send_text(text, server_url, agent_name, user_name='User', audio_response=False):
    url = f"{server_url}/text"
    data = {'text': text, 'agent_name': agent_name, 'user_name': user_name}
    if audio_response:
        data['audio_response'] = True

    response = requests.post(url, json=data)

    if response.status_code == 200:
        # Get the response audio and JSON data
        response_content = response.content

        if not audio_response:
            # Play the response audio
            response_json = json.loads(response_content)
            # Print the response JSON data
            print(response_json)
            return response_json, None

        # Get the response JSON data from the response headers
        response_json = json.loads(response.headers.get('X-JSON'))

        # Print the response JSON data
        print(response_json)
        return response_json, response_content
    else:
        return f"Request failed with status code: {response.status_code}"
