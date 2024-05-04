import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

voice_path = 'babyshort.wav'

# List available 🐸TTS models
print(TTS().list_models())

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

# Init TTS
tts = TTS(model_name).to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text="Hello world!", speaker_wav=voice_path, language="en")
# Text to speech to a file
tts.tts_to_file(text="Hello, Rachel, how are you doing?", speaker_wav=voice_path, emotion="Happy", language="en",
                file_path="../speech_circuss.wav")
tts.tts_to_file(text="Maybe we should clone another voice? No!", speaker_wav=voice_path, language="en",
                file_path="../speech2_circuss.wav")
tts.tts_to_file(text="The tickle spiders are the cutest! They protect, they attack, but mostly... they want a snack! Nom Nom!", speaker_wav=voice_path, language="en",
                file_path="../speech3_circuss.wav")

wav = tts.tts(text="Hello world, I hope it is friendly!", speaker_wav=voice_path, language="en")
