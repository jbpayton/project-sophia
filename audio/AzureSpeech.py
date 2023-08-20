import azure.cognitiveservices.speech as speechsdk
import os
from AudioUtilities import find_windows_output_device


class TTSClient:
    def __init__(self, voice="en-US-AmberNeural", device_name=None):
        speech_key = os.environ["AZURE_SPEECH_KEY"]
        service_region = os.environ["AZURE_SPEECH_REGION"]

        selected_device = find_windows_output_device(selected_device_name=device_name)
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        speech_config.speech_synthesis_voice_name = voice

        if selected_device is not None:
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


if __name__ == "__main__":
    from util import load_secrets
    load_secrets("../secrets.json")
    pollyClient = TTSClient()
    pollyClient.speak("I'm big billy, the biggest wet willy, i gotta go clearly")
    pollyClient.speak("hmmmm!")