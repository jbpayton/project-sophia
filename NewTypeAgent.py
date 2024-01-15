import datetime

from openai import OpenAI
import util
import json
import re


class NewTypeAgent:
    def __init__(self, name):
        self.name = name
        # load profile
        self.profile = util.load_profile(name)

        self.observation = ""
        self.observation_updated = False

        self.MONOLOGUE_PROMPT = "Before everything you say, include internal monologue in curly brackets."
        self.EMOTION_PROMPT = "Express your emotions by leading a sentence with parenthesis with your emotional " \
                              "state. Valid emotional states are as follows: Default, Angry, Cheerful, Excited, " \
                              "Friendly, Hopeful, Sad, Shouting, Terrified, Unfriendly, Whispering."

        self.system_prompt = self.profile['personality'] + self.MONOLOGUE_PROMPT + self.EMOTION_PROMPT
        self.client = OpenAI()

        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    @staticmethod
    def extract_json_objects(text):
        # Remove triple backticks
        text = text.replace('```json', '')
        text = text.replace('```', '')
        text_without_json = text

        pattern = r'\{.*?\}'
        matches = re.findall(pattern, text, re.DOTALL)

        json_objects = []
        for match in matches:
            try:
                json_obj = json.loads(match)
                if "actionName" in json_obj:
                    json_objects.append(json_obj)
                # remove the JSON object from the original text
                text_without_json = text_without_json.replace(match, "")
            except json.JSONDecodeError:
                continue  # Skip invalid JSON

        print("JSON Objects: " + str(json_objects))
        print("Text without JSON: " + text_without_json)
        return json_objects, text_without_json

    @staticmethod
    def prepend_timestamp(text):
        # first format the date+timestamp
        timestamp_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp_string}] {text}"

    def send(self, message):
        if self.observation_updated:
            self.observation_updated = False
            self.messages.append({"role": "system", "content": self.observation})

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

        print("Content, before Command Removal:" + content)

        actions, content = self.extract_json_objects(content)
        content = content.strip()

        print("Content, after Command Removal:" + content)

        # example content, parse content into internal monologue, emotion, and the rest:
        # '{Curiosity piqued, time to clarify} (Friendly) FIRE as in Finance term, or literal fire?'
        if "{" in content:
            monologue = content.split("}")[0][1:]
            content = content.split("}")[1]
            print("Monologue:" + monologue)
        else:
            monologue = None

        if "(" in content:
            emotion = content.split(")")[0].split("(")[1]
            content = content.split(")")[1]
            print("Emotion:" + emotion)
        else:
            emotion = None

        content = content.strip()
        print("Content:" + content)

        return content, emotion, monologue, actions

    def accept_observation(self, observation):
        self.observation = self.prepend_timestamp(observation)
        self.observation_updated = True


if __name__ == "__main__":
    util.load_secrets()
    agent = NewTypeAgent("Sophia")
    while True:
        # prompt user for input
        message = input("Enter a message: ")
        # send message to agent
        response, emotion, monologue, actions = agent.send(message)
        # print response
        print(response)


