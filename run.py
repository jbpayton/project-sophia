from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from Agents import DialogueAgent, UserAgent, DialogueAgentWithTools
from DialogSimulator import DialogueSimulator
from langchain.tools import DuckDuckGoSearchRun
from tools.LLMBrowsingTools import paged_web_browser

from audio.AzureSpeech import TTSClient
from audio.WhisperSTTEngine import WhisperSTTEngine

import util

if __name__ == "__main__":

    profile_name = "Sophia"
    input_audio_device = None
    output_audio_device = None

    # read profile name and input/output audio device from a config file
    # if not specified, use the default values
    config = util.load_config_file()

    profile_name = config['Avatar']['profile_name']
    input_audio_device = config['Audio']['input_audio_device']
    output_audio_device = config['Audio']['output_audio_device']

    util.load_secrets()

    profile = util.load_profile(profile_name)

    # Define system prompts for our two agents
    system_prompt_avatar = SystemMessage(role=profile['name'],
                                         content=profile['personality'])

    tools = [DuckDuckGoSearchRun(),
             paged_web_browser]

    avatar_tts = TTSClient(voice=profile['voice'], device_name=output_audio_device)

    # Initialize our agents with their respective roles and system prompts
    avatar = DialogueAgentWithTools(name=profile['name'],
                                    system_message=system_prompt_avatar,
                                    model=ChatOpenAI(model_name='gpt-4', streaming=True,
                                                     callbacks=[StreamingStdOutCallbackHandler()]),
                                    tools=tools,
                                    TTSEngine=avatar_tts)

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
