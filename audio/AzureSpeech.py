import azure.cognitiveservices.speech as speechsdk
import os
from audio.AudioUtilities import find_windows_output_device
import wave
import io


class TTSClient:
    def __init__(self, voice="en-US-JaneNeural", device_name=None, silent=False):
        speech_key = os.environ["AZURE_SPEECH_KEY"]
        service_region = os.environ["AZURE_SPEECH_REGION"]

        self.voice = voice
        selected_device = find_windows_output_device(selected_device_name=device_name)
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        speech_config.speech_synthesis_voice_name = voice

        if silent:
            audio_config = None
        elif selected_device is not None:
            audio_config = speechsdk.audio.AudioOutputConfig(device_name=selected_device)
        else:
            audio_config = speechsdk.audio.AudioOutputConfig(device_name=selected_device, use_default_speaker=True)

        # use the default speaker as audio output.
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    def speak(self, text_input):
        result = self.speech_synthesizer.speak_text_async(text_input).get()
        # Check result
        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))

    @staticmethod
    def parse_wav_data(audio_bytes):
        """
        Parses WAV audio byte data and returns metadata and audio frames.

        :param audio_bytes: Byte string containing the WAV file data.
        :return: A tuple containing metadata and audio frames.
        """
        with io.BytesIO(audio_bytes) as audio_file:
            with wave.open(audio_file, 'rb') as wav_file:
                metadata = {
                    "channels": wav_file.getnchannels(),
                    "sample_width": wav_file.getsampwidth(),
                    "frame_rate": wav_file.getframerate(),
                    "num_frames": wav_file.getnframes(),
                    "compression_type": wav_file.getcomptype(),
                    "compression_name": wav_file.getcompname()
                }

                audio_frames = wav_file.readframes(wav_file.getnframes())

        return metadata, audio_frames


    def speak_to_stream(self, text_input, mood="cheerful", style_degree=2, filename="output.wav"):
        # Construct SSML with the specified mood
        ssml_text = f"""
                <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
                    <voice name="{self.voice}">
                        <mstts:express-as style="{mood}" styledegree="{style_degree}">
                            {text_input}
                        </mstts:express-as>
                    </voice>
                </speak>
                """

        # Synthesize the SSML and save to a file
        #speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
        result = self.speech_synthesizer.speak_ssml_async(ssml_text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            if filename is not None:
                with open(filename, "wb") as file:
                    file.write(audio_data)
                print(f"Audio saved to {filename}")
            metadata, frames = TTSClient.parse_wav_data(audio_data)
            return frames
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))

        return None

    def speak_async(self, text_input):
        self.speech_synthesizer.speak_text_async(text_input)


if __name__ == "__main__":
    from util import load_secrets
    load_secrets("../secrets.json")
    azureSpeech = TTSClient(silent=True)
    audio = azureSpeech.speak_to_stream("I'm big billy, the biggest wet willy, i gotta go clearly... hmmm", mood="sad")


    azureSpeech.speak_to_stream("I'm big billy, the biggest wet willy, i gotta go clearly... hmmm", mood="hopeful",
                              filename="houtput.wav")
    azureSpeech.speak_to_stream("I'm big billy, the biggest wet willy, i gotta go clearly... hmmm", mood="angry",
                              filename="aoutput.wav")