import speech_recognition as sr
import whisper
import torch
import numpy as np
import random
from audio.AudioUtilities import find_input_device


class WhisperSTTEngine:
    def __init__(self, input_audio_device="Default", model='base', english=False, energy=350, pause=0.8, dynamic_energy=False):

        # there are no english models for large
        if model != "large" and english:
            model = model + ".en"
        self.audio_model = whisper.load_model(model)

        self.english = english
        self.r = sr.Recognizer()
        self.r.energy_threshold = energy
        self.r.pause_threshold = pause
        self.r.dynamic_energy_threshold = dynamic_energy
        self.mic_source = sr.Microphone(device_index=find_input_device(selected_device_name=input_audio_device), sample_rate=16000)
        with self.mic_source as source:
            self.r.adjust_for_ambient_noise(source, duration=0.5)

        print("Systems Loaded")

    def listen(self):
        while True:
            idle_response = False

            with self.mic_source as source:
                try:
                    audio = self.r.listen(source, timeout=random.randrange(5, 30))
                except sr.exceptions.WaitTimeoutError as e:
                    idle_response = True

            if not idle_response:
                torch_audio = torch.from_numpy(
                    np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                audio_data = torch_audio

                if self.english:
                    result = self.audio_model.transcribe(audio_data, language='english')
                else:
                    result = self.audio_model.transcribe(audio_data)

                predicted_text = result["text"]

                if sum(char.isalnum() for char in predicted_text) > 0 and predicted_text.strip() != 'you':
                    return predicted_text
                else:
                    print("Threshold exceeded, but no speech detected")
            else:
                return ""


if __name__ == "__main__":
    tts_engine = WhisperSTTEngine()
    print("Listening:")
    print(tts_engine.listen())
    print("Listening (2):")
    print(tts_engine.listen())
    print("Listening (3):")
    print(tts_engine.listen())
