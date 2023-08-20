import pyaudio
from pydub import AudioSegment
import comtypes
from pycaw.pycaw import AudioUtilities, IMMDeviceEnumerator, EDataFlow, DEVICE_STATE
from pycaw.constants import CLSID_MMDeviceEnumerator


def list_audio_devices():
    audio = pyaudio.PyAudio()
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    devices = []
    for i in range(0, numdevices):
        device_info = audio.get_device_info_by_host_api_device_index(0, i)
        devices.append((i, device_info.get('name')))
    return devices


def find_input_device(selected_device_name=None):
    devices = list_audio_devices()

    if selected_device_name is not None:
        for idx, device_name in devices:
            if selected_device_name in device_name:
                print("STT Input Device " + device_name + " found.")
                return idx
    else:
        selected_device_name = "<NOT SPECIFIED>"

    print("STT Input Device " + selected_device_name + " not found. Defaulting to index 1 - " + devices[1][1])
    return 1


def find_output_device(selected_device_name=None):
    devices = list_audio_devices()
    if selected_device_name is not None:
        for idx, device_name in devices:
            if selected_device_name in device_name or device_name in selected_device_name:
                print("TTS Output Device " + device_name + " found.")
                return idx

    else:
        selected_device_name = "<NOT SPECIFIED>"

    print("TTS Output Device " + selected_device_name + " not found. Defaulting to index 0 - " + devices[0][1])


def find_windows_output_device(selected_device_name=None):
    devices = []

    if selected_device_name is None:
        return None

    deviceEnumerator = comtypes.CoCreateInstance(
        CLSID_MMDeviceEnumerator,
        IMMDeviceEnumerator,
        comtypes.CLSCTX_INPROC_SERVER)
    if deviceEnumerator is None:
        devices = []
        raise ValueError("Couldn't find any devices.")

    collection = deviceEnumerator.EnumAudioEndpoints(EDataFlow.eRender.value, DEVICE_STATE.ACTIVE.value)
    if collection is None:
        devices = []
        raise ValueError("Couldn't find any devices.")

    count = collection.GetCount()
    for i in range(count):
        dev = collection.Item(i)
        if dev is not None:
            if not ": None" in str(AudioUtilities.CreateDevice(dev)):
                devices.append(AudioUtilities.CreateDevice(dev))

    for device in devices:
        if selected_device_name in device.FriendlyName:
            print("Found input device.")
            print(device.FriendlyName)
            return device.id


def play_mp3_to_device(mp3_file, device_index):
    audio_segment = AudioSegment.from_mp3(mp3_file)
    audio_data = audio_segment.raw_data
    audio_format = pyaudio.get_format_from_width(audio_segment.sample_width)
    audio_channels = audio_segment.channels
    audio_rate = audio_segment.frame_rate

    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio_format,
                        channels=audio_channels,
                        rate=audio_rate,
                        output=True,
                        output_device_index=device_index)

    stream.write(audio_data)
    stream.stop_stream()
    stream.close()
    audio.terminate()


if __name__ == "__main__":
    find_windows_output_device()
    voicemeeter_device_index = find_output_device("VoiceMeeter Aux Input (VB-Audio VoiceMeeter AUX VAIO)")
    if voicemeeter_device_index is not None:
        print(f"Found input device at index {voicemeeter_device_index}")
    else:
        print("Input device not found.")
