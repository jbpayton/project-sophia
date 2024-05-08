import json
import requests
import pyaudio
import audioop
import tkinter as tk
from tkinter import ttk
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

# GUI window
window = tk.Tk()
window.title("Audio Client")

# Server address entry
server_label = ttk.Label(window, text="Server Address:")
server_label.grid(row=0, column=0, padx=5, pady=5)
server_entry = ttk.Entry(window)
server_entry.insert(0, SERVER_ADDRESS)
server_entry.grid(row=0, column=1, padx=5, pady=5)

# Speaker name entry
speaker_label = ttk.Label(window, text="Speaker Name:")
speaker_label.grid(row=1, column=0, padx=5, pady=5)
speaker_entry = ttk.Entry(window)
speaker_entry.insert(0, SPEAKER_NAME)
speaker_entry.grid(row=1, column=1, padx=5, pady=5)

# Agent name entry
agent_label = ttk.Label(window, text="Agent Name:")
agent_label.grid(row=2, column=0, padx=5, pady=5)
agent_entry = ttk.Entry(window)
agent_entry.insert(0, AGENT_NAME)
agent_entry.grid(row=2, column=1, padx=5, pady=5)

# Get the list of input devices
p = pyaudio.PyAudio()
input_devices = []
input_device_dict = {}
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    if device_info["maxInputChannels"] > 0:
        new_index = len(input_devices)
        input_device_dict[new_index] = i
        input_devices.append(device_info["name"])

p.terminate()

# Get the list of output devices
p = pyaudio.PyAudio()
output_devices = []
output_device_dict = {}
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    if device_info["maxOutputChannels"] > 0:
        new_index = len(output_devices)
        output_device_dict[new_index] = i
        output_devices.append(device_info["name"])
p.terminate()

# Input device dropdown
input_label = ttk.Label(window, text="Input Device:")
input_label.grid(row=3, column=0, padx=5, pady=5)
input_var = tk.StringVar()
input_dropdown = ttk.Combobox(window, textvariable=input_var)
input_dropdown['values'] = input_devices

# Check if the saved input device starts with the same characters as any available device
input_device_found = False
for device in input_devices:
    if device.startswith(INPUT_DEVICE):
        input_dropdown.current(input_devices.index(device))
        input_device_found = True
        break

if not input_device_found:
    input_dropdown.current(0)

input_dropdown.grid(row=3, column=1, padx=5, pady=5)

# Output device dropdown
output_label = ttk.Label(window, text="Output Device:")
output_label.grid(row=4, column=0, padx=5, pady=5)
output_var = tk.StringVar()
output_dropdown = ttk.Combobox(window, textvariable=output_var)
output_dropdown['values'] = output_devices

# Check if the saved output device starts with the same characters as any available device
output_device_found = False
for device in output_devices:
    if device.startswith(OUTPUT_DEVICE):
        output_dropdown.current(output_devices.index(device))
        output_device_found = True
        break

if not output_device_found:
    output_dropdown.current(0)

output_dropdown.grid(row=4, column=1, padx=5, pady=5)


# Text entry field
text_entry = ttk.Entry(window)
text_entry.grid(row=5, column=0, padx=5, pady=5)

# Send button
def send_text_message():
    global AGENT_NAME, USER_NAME, SERVER_URL

    SERVER_URL = server_entry.get()
    USER_NAME = speaker_entry.get()
    AGENT_NAME = agent_entry.get()

    text = text_entry.get()
    if text:
        response, audio = send_text(text, SERVER_URL, AGENT_NAME, USER_NAME, audio_response=True)
        log_text.insert(tk.END, f"{AGENT_NAME}: {response}\n")
        play_audio(audio)
        text_entry.delete(0, tk.END)

send_button = ttk.Button(window, text="Send", command=send_text_message)
send_button.grid(row=5, column=1, padx=5, pady=5)

# Log area
log_text = tk.Text(window, height=10, width=50)
log_text.grid(row=6, column=0, columnspan=2, padx=5, pady=5)


# Start button
def start_client():
    global SERVER_URL, USER_NAME, AGENT_NAME, INPUT_DEVICE, OUTPUT_DEVICE, is_running, audio_thread
    SERVER_URL = server_entry.get()
    USER_NAME = speaker_entry.get()
    AGENT_NAME = agent_entry.get()
    INPUT_DEVICE = input_var.get()
    OUTPUT_DEVICE = output_var.get()

    # Save settings to configuration file
    config = configparser.ConfigParser()
    config['Settings'] = {
        'ServerAddress': SERVER_URL,
        'SpeakerName': USER_NAME,
        'AgentName': AGENT_NAME,
        'InputDevice': INPUT_DEVICE,
        'OutputDevice': OUTPUT_DEVICE
    }
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    if is_running:
        is_running = False
        if audio_thread is not None:
            print("Stopping audio thread...")
            audio_thread.join()
            print("Audio thread stopped.")

    if is_running is False:
        is_running = True
        print("Starting audio thread...")
        audio_thread = threading.Thread(target=audio_loop)
        audio_thread.start()
        print("Audio thread started.")


def stop_client():
    global is_running, audio_thread
    if is_running:
        is_running = False
        if audio_thread is not None:
            print("Stopping audio thread...")
            audio_thread.join()
            print("Audio thread stopped.")


start_button = ttk.Button(window, text="Start", command=start_client)
start_button.grid(row=7, column=0, padx=5, pady=5)

# Stop button
stop_button = ttk.Button(window, text="Stop", command=stop_client)
stop_button.grid(row=7, column=1, padx=5, pady=5)

# Update window closing behavior
def on_closing():
    global is_running
    is_running = False
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)

# Function to check if audio data is silence
def is_silence(data, threshold=SILENCE_THRESHOLD):
    rms = audioop.rms(data, 2)
    return rms < threshold

# Function to play the received audio data
def play_audio(audio_data):
    output_device = output_var.get()
    output_device_index = output_devices.index(output_device)
    p = pyaudio.PyAudio()
    real_index = output_device_dict[output_device_index]

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=OUTPUT_SAMPLE_RATE,
                    output=True, output_device_index=real_index)
    stream.write(audio_data)
    stream.stop_stream()
    stream.close()
    p.terminate()

# Function to send audio to the server
def send_audio(audio_data, agent_name, user_name):
    url = f"{SERVER_URL}/audio"
    files = {'audio': ('audio.wav', audio_data, 'audio/wav')}
    data = {'agent_name': agent_name, 'user_name': user_name}
    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        # Get the response audio and JSON data
        response_audio = response.content
        # Get the response JSON data from the response headers
        response_json = json.loads(response.headers.get('X-JSON'))

        # Play the response audio
        play_audio(response_audio)

        # Print the response JSON data
        log_text.insert(tk.END, f"Response: {response_json}\n")
        print(response_json)
    else:
        log_text.insert(tk.END, f"Request failed with status code: {response.status_code}\n")
        print(f"Request failed with status code: {response.status_code}")


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

# Main function
def audio_loop():
    global is_running
    input_device = input_var.get()

    p = pyaudio.PyAudio()
    input_device_index = input_devices.index(input_device)
    real_index = input_device_dict[input_device_index]
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True,
                    frames_per_buffer=CHUNK_SIZE, input_device_index=real_index)

    print("Listening...")

    non_silence_present = False
    silent_frames_count = 0
    speech_frames = []

    while is_running:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

        if is_silence(data):
            silent_frames_count += 1
            speech_frames.append(data)
            if silent_frames_count >= SILENT_FRAMES_THRESHOLD and non_silence_present:
                stream.stop_stream()
                print("Silence detected. Processing speech...")
                audio_data = b''.join(speech_frames)

                send_audio(audio_data, AGENT_NAME, USER_NAME)  # Send the audio data to the server

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
    window.mainloop()