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

        self.MONOLOGUE_PROMPT = "Before everything you say, include internal monologue in curly brackets."
        self.ACTION_PROMPT = "If you want to perform an action that you don't you can do alone, put it in square " \
                             "brackets with the word Action in your response (e.g., [Action: <action>] to pass to the " \
                             "Action Assistant, do this before responding to the user. "
        self.EMOTION_PROMPT = "Express your emotions by leading a sentence with parenthesis with your emotional " \
                              "state (e.g., (<emotion>)). Valid emotional states are as follows: Default, Angry, " \
                              "Cheerful, Excited, " \
                              "Friendly, Hopeful, Sad, Shouting, Terrified, Unfriendly, Whispering."

        self.system_prompt = self.profile['personality'] + self.MONOLOGUE_PROMPT + self.EMOTION_PROMPT + self.ACTION_PROMPT
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
            print(f"Action: {action}")
            response, success = self.action_interpreter.send(action)

            print(f"Action Assistant Response: {response}")
            processed_actions += 1

            if success:
                # Send response to the agent and get new actions
                new_content, new_emotion, new_monologue, new_actions = self.inner_send(response)
                content += new_content  # Or handle this as per your logic
                emotion = new_emotion  # Update emotion
                monologue = new_monologue  # Update monologue

                # Insert new actions right after the current action
                for new_action in reversed(new_actions):
                    actions.insert(0, new_action)

        content = self.messages[-1]["content"]  # Update content from the last message
        return content, emotion, monologue, actions

    def inner_send(self, message):
        if self.observation_updated:
            self.observation_updated = False
            self.messages.append({"role": "user", "content": self.observation})

        message = self.prepend_timestamp(message)

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
        self.messages.append({"role": role, "content": content})

        # example content, parse content into internal monologue, emotion, and the rest:
        # '{Curiosity piqued, time to clarify} (Friendly) FIRE as in Finance term, or literal fire?'
        if "{" in content:
            monologue_pattern = r'\{(.*?)\}'
            monologue = re.findall(monologue_pattern, content)[0]
            content = re.sub(monologue_pattern, '', content)
            print("Monologue:" + monologue)
        else:
            monologue = None

        if "(" in content:
            emotion_pattern = r'\((.*?)\)'
            emotion = re.findall(emotion_pattern, content)[0]
            content = re.sub(emotion_pattern, '', content)
            print("Emotion:" + emotion)
        else:
            emotion = None

        if "[" in content:
            action_pattern = r'\[Action:(.*?)\]'

            # Find all actions
            actions = re.findall(action_pattern, content)

            # Remove all occurrences of the action from the content
            content = re.sub(action_pattern, '', content)

            print("Actions:" + str(actions))

        else:
            actions = []

        content = content.strip()
        print("Content:" + content)

        return content, emotion, monologue, actions

    def accept_observation(self, observation):
        self.observation = self.prepend_timestamp(observation)
        self.observation_updated = True


if __name__ == "__main__":
    tools = load_tools_from_file("DuckDuckGo.py")
    agent = ToolUsingAgent(tools)

    WEATHER_TEST = True
    INTERACTIVE_TEST = False

    if WEATHER_TEST:
        response, emotion, monologue, actions = agent.send("Can you check the weather in Fredericksburg, VA for 2/25/2024?")
        print(emotion)
        print(monologue)
        print(response)

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
