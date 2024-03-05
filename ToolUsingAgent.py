import datetime
import re

from openai import OpenAI
import util
from ActionInterpreter import ActionInterpreter
from ToolLoader import load_tools_from_file, load_tools_from_files


class ToolUsingAgent:
    def __init__(self, tools=None, name = None):
        self.name = name
        if name is None:
            self.name = "ToolUsingAgent"
            self.profile = {
                "personality": "You are a helpful agent with an assistant that can use tools to do things for you."
            }
        else:
            # load profile
            self.profile = util.load_profile(name)

        if tools is not None:
            self.action_interpreter = ActionInterpreter(tools)
        else:
            self.action_interpreter = ActionInterpreter()

        self.observation = ""
        self.observation_updated = False

        # load prompt from the file ToolUsingAgentPrompt.txt using python file I/O
        with open("ToolUsingAgentPrompt.txt", "r") as file:
            self.AGENT_PROMPT = file.read()

        self.system_prompt = self.profile['personality'] + self.AGENT_PROMPT
        self.client = OpenAI(
            api_key="sk-111111111111111111111111111111111111111111111111",
            base_url="http://192.168.2.94:5000/v1"
        )

        self.messages = [
            {"role": "user", "content": self.system_prompt}
        ]

    @staticmethod
    def prepend_timestamp(text):
        # first format the date+timestamp
        timestamp_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp_string}] {text}"

    def send(self, message, max_actions=10):
        content, emotion, monologue, actions = self.inner_send(message)
        processed_actions = 0

        # Using a while loop to process actions as a queue
        while actions and processed_actions < max_actions:
            print(f"Processing action {processed_actions + 1} of {max_actions}")

            action = actions.pop(0)  # Get the first action from the list
            if len(action) == 0:
                continue
            print(f"Action: {action}")
            response, success = self.action_interpreter.send(action)

            print(f"Action Assistant Response: {response}")
            processed_actions += 1

            if success:
                # Send response to the agent and get new actions
                new_content, new_emotion, new_monologue, new_actions = self.inner_send(response, "ACTION_AGENT")
                content = new_content  # Or handle this as per your logic
                emotion = new_emotion  # Update emotion
                monologue = new_monologue  # Update monologue

                # Insert new actions right after the current action
                for new_action in reversed(new_actions):
                    if len(new_action) > 0:
                        actions.insert(0, new_action)

        return content, emotion, monologue, actions

    def parse_response(self, text):
        # Define markers
        markers = [
            "EMOTION:", "THOUGHTS:", "ACTION_THOUGHTS:",
            "MESSAGE[ACTION_AGENT]:", "MESSAGE[USER]:",
            "OBSERVATION[USER]:", "OBSERVATION[ACTION_AGENT]:"
        ]

        # Initialize a dictionary to hold the parsed content
        parsed_content = {}

        # Split the text into lines for easier parsing
        lines = text.split('\n')

        # don't let the LLM Fake OBSERVATIONS! if you see a line starting with OBSERVATION,
        # remove it and everything after it
        for i in range(len(lines)):
            if lines[i].startswith("OBSERVATION"):
                lines = lines[:i]
                break

        current_marker = None
        for line in lines:
            # Check if the line starts with any of the markers
            for marker in markers:
                if line.startswith(marker):
                    current_marker = marker
                    parsed_content[current_marker] = line[len(marker):].strip()
                    break
            else:
                if current_marker:
                    # Append line to current marker content
                    parsed_content[current_marker] += " " + line.strip()

        return parsed_content

    def inner_send(self, message, origin="USER"):
        if self.observation_updated:
            self.observation_updated = False
            self.messages.append({"role": "user", "content": self.observation})

        message = self.prepend_timestamp(message)

        message = f"OBSERVATION[{origin}]: {message}"

        self.messages.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=self.messages,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        role = response.choices[0].message.role
        content = response.choices[0].message.content
        print(role + ":\n" + content)

        self.messages.append({"role": role, "content": content})

        parse_response = self.parse_response(content)
        emotion = parse_response.get("EMOTION:", "")
        monologue = parse_response.get("THOUGHTS:", "")
        actions_thoughts = parse_response.get("ACTION_THOUGHTS:", "").split("\n")
        actions = parse_response.get("MESSAGE[ACTION_AGENT]:", "").split("\n")
        message_user = parse_response.get("MESSAGE[USER]:", "")

        return message_user, emotion, monologue, actions

    def accept_observation(self, observation):
        self.observation = self.prepend_timestamp(observation)
        self.observation_updated = True


if __name__ == "__main__":


    WEATHER_TEST = False
    INTERACTIVE_TEST = True
    ALUCARD_TEST = False

    if WEATHER_TEST:
        tools = load_tools_from_file("DuckDuckGo.py")
        agent = ToolUsingAgent(tools)
        response, emotion, monologue, actions = agent.send("Can you check the weather in Fredericksburg, VA for 2/25/2024?")
        print(emotion)
        print(monologue)
        print(response)

    if ALUCARD_TEST:
        tools = load_tools_from_files()
        agent = ToolUsingAgent(tools)
        response, emotion, monologue, actions = agent.send("Can you tell me who Rachel Alucard is (according to the internet)?")
        print(emotion)
        print(monologue)
        print(response)
        response, emotion, monologue, actions = agent.send("Write this info to a file called Alucard.txt")
        print(emotion)
        print(monologue)
        print(response)

    if INTERACTIVE_TEST:
        tools = load_tools_from_files()
        agent = ToolUsingAgent(tools)
        while True:
            # prompt user for input
            message = input("Enter a message: ")
            # send message to agent
            response, emotion, monologue, actions = agent.send(message)
            # print emotion
            print(emotion)
            # print monologue
            print(monologue)
            # print response
            print(response)
