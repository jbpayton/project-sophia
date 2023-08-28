from typing import List

from Agents import DialogueAgent, UserAgent, get_avatar_agent
from DialogSimulator import DialogueSimulator

from audio.AzureSpeech import TTSClient
from audio.WhisperSTTEngine import WhisperSTTEngine

import util

if __name__ == "__main__":

    profile_name = "Sophia"
    input_audio_device = None
    output_audio_device = None

    util.load_secrets()

    # read profile name and input/output audio device from a config file
    # if not specified, use the default values
    config = util.load_config_file()

    # if the config file is not found, (avatar not found in profile) create a new one
    if 'Avatar' not in config:
        config = util.create_config_file()

    profile_name = config['Avatar']['profile_name']
    input_audio_device = config['Audio']['input_audio_device']
    output_audio_device = config['Audio']['output_audio_device']

    profile = util.load_profile(profile_name)

    avatar_tts = TTSClient(voice=profile['voice'], device_name=output_audio_device)
    avatar = get_avatar_agent(profile, avatar_tts)

    # Define a round-robin selection function
    def round_robin(step: int, agents: List[DialogueAgent]) -> int:
        return step % len(agents)

    # Initialize the User agent
    user_stt = WhisperSTTEngine(input_audio_device=input_audio_device)
    user_agent = UserAgent(name="User", stt_engine=user_stt)

    agent_list = [user_agent, avatar]

    # Create your simulator
    simulator = DialogueSimulator(agents=agent_list, selection_function=round_robin)

    while True:
        for _ in range(len(agent_list)):
            speaker, message = simulator.step()
