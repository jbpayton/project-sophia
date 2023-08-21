from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.tools import Tool

from Agents import DialogueAgent, UserAgent, DialogueAgentWithTools
from DialogSimulator import DialogueSimulator
from langchain.utilities import GoogleSearchAPIWrapper
from tools.LLMBrowsingTools import query_website, paged_web_browser

from audio.AzureSpeech import TTSClient

import util

if __name__ == "__main__":

    profile_name = "Sophia"

    util.load_secrets()

    profile = util.load_profile(profile_name)

    # Define system prompts for our two agents
    system_prompt_avatar = SystemMessage(role=profile['name'],
                                         content=profile['personality'])
    # initialie search API
    search = GoogleSearchAPIWrapper()


    def top10_results(query):
        return search.results(query, 10)


    GoogleSearchTool = Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=top10_results,
    )

    tools = [GoogleSearchTool,
             paged_web_browser]

    avatar_tts = TTSClient(voice=profile['voice'], device_name=None)

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
    user_agent = UserAgent(name="User")

    agent_list = [user_agent, avatar]

    # Create your simulator
    simulator = DialogueSimulator(agents=agent_list, selection_function=round_robin)

    while True:
        for _ in range(len(agent_list)):
            speaker, message = simulator.step()
