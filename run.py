from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from Agents import DialogueAgent, UserAgent
from DialogSimulator import DialogueSimulator

import util

if __name__ == "__main__":

    profile_name = "Sophia"

    util.load_secrets()

    profile = util.load_profile(profile_name)

    # Define system prompts for our two agents
    system_prompt_avatar = SystemMessage(role=profile,
                                         content=profile['personality'])

    # Initialize our agents with their respective roles and system prompts
    avatar = DialogueAgent(name=profile['name'],
                           system_message=system_prompt_avatar,
                           model=ChatOpenAI(model_name='gpt-4', streaming=True,
                                            callbacks=[StreamingStdOutCallbackHandler()]))

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

