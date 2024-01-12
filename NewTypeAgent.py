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

        self.generate_system_prompt()
        self.client = OpenAI()

        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def generate_system_prompt(self, observation=None):
        self.MONOLOGUE_PROMPT = "Before everything you say, include internal monologue in curly brackets."
        self.EMOTION_PROMPT = "Express your emotions by leading a sentence with parenthesis with your emotional " \
                              "state. Valid emotional states are as follows: Default, Angry, Cheerful, Excited, " \
                              "Friendly, Hopeful, Sad, Shouting, Terrified, Unfriendly, Whispering."
        if observation is not None:
            self.system_prompt = self.profile['personality'] + self.MONOLOGUE_PROMPT + self.EMOTION_PROMPT + observation
        else:
            self.system_prompt = self.profile['personality'] + self.MONOLOGUE_PROMPT + self.EMOTION_PROMPT

        return self.system_prompt

    @staticmethod
    def extract_json_objects(text):
        # Remove triple backticks
        text = text.replace('```json', '')
        text = text.replace('```', '')

        pattern = r'\{.*?\}'
        matches = re.findall(pattern, text, re.DOTALL)

        json_objects = []
        for match in matches:
            try:
                json_obj = json.loads(match)
                if "actionName" in json_obj:
                    json_objects.append(json_obj)
            except json.JSONDecodeError:
                continue  # Skip invalid JSON

        # Optional: Remove the JSON objects from the original text
        text_without_json = re.sub(pattern, "", text, flags=re.DOTALL).strip()

        return json_objects, text_without_json

    def send(self, message):
        if self.observation_updated:
            self.generate_system_prompt(self.observation)
            self.observation_updated = False
            self.messages.append({"role": "system", "content": self.system_prompt})

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
        monologue = content.split("}")[0][1:]
        emotion = content.split(")")[0].split("(")[1]
        response = content.split(")")[1]
        # trim whitespace from response
        response = response.strip()

        actions, response = self.extract_json_objects(response)
        response = response.strip()

        return response, emotion, monologue, actions

    def accept_observation(self, observation):
        self.observation = observation
        self.observation_updated = True


if __name__ == "__main__":
    util.load_secrets()
    agent = NewTypeAgent("Sophia")
    while True:
        # prompt user for input
        message = input("Enter a message: ")
        # send message to agent
        response, emotion, monologue = agent.send(message)
        # print response
        print(response)


