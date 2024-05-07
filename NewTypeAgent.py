import datetime
import os

from openai import OpenAI
import util
import json
import re

from ConversationLogger import ConversationFileLogger


class NewTypeAgent:
    def __init__(self, name):
        self.name = name
        # load profile
        self.profile = util.load_profile(name)

        self.conversation_logger = ConversationFileLogger(f"{self.name}_logs")

        self.observation = ""
        self.observation_updated = False

        self.MONOLOGUE_PROMPT = "Before everything you say, include internal monologue in curly brackets."
        self.EMOTION_PROMPT = "Express your emotions by leading a sentence with parenthesis with your emotional " \
                              "state. Valid emotional states are as follows: Default, Angry, Cheerful, Excited, " \
                              "Friendly, Hopeful, Sad, Shouting, Terrified, Unfriendly, Whispering."

        self.system_prompt = self.profile['personality'] + self.EMOTION_PROMPT

        self.client = OpenAI(
            api_key="sk-111111111111111111111111111111111111111111111111",
            base_url=os.environ['LOCAL_TEXTGEN_API_BASE']
        )

        self.messages = [
            {"role": "user", "content": self.system_prompt}
        ]

        self.conversation_logger.append_last_lines_to_messages(200, self.messages)

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
        #print("Text without JSON: " + text_without_json)
        return json_objects, text_without_json

    @staticmethod
    def prepend_timestamp(text, name):
        # first format the date+timestamp
        timestamp_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp_string}] {name}:{text}"

    def send(self, message, user_name="User"):
        if self.observation_updated:
            self.observation_updated = False
            self.messages.append({"role": "system", "content": self.observation})

        message = self.prepend_timestamp(message, user_name)

        print(message)

        self.messages.append({"role": "user", "content": message})
        self.conversation_logger.log_message("user", message)

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

        #print("Raw Content from LLM:" + content)

        # get rid of the timestamp, if present, with regex
        content = re.sub(r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]', '', content)

        #print("Content after timestamp removal:" + content)

        # remove the contents of self.name + colon from the response, if present, with regex
        content = re.sub(rf'{self.name}:', '', content)

        #print("Content after name removal:" + content)

        # if we find {user_name}: anywhere in the response, remove it and everything after it
        if user_name + ":" in content:
            content = content.split(user_name + ":")[0]

        stored_content = content
        stored_content = self.prepend_timestamp(stored_content, self.name)
        print(stored_content)
        self.messages.append({"role": role, "content": stored_content})
        self.conversation_logger.log_message(role, stored_content)

        #print("Content, before Command Removal:" + content)

        actions, content = self.extract_json_objects(content)
        content = content.strip()

        #print("Content, after Command Removal:" + content)

        # example content, parse content into internal monologue, emotion, and the rest:
        # '{Curiosity piqued, time to clarify} (Friendly) FIRE as in Finance term, or literal fire?'
        # let use regex and let's remove it from the content
        monologue = None
        for match in re.finditer(r'\{.*?\}', content):
            monologue = match.group(0)[1:-1]
            content = content.replace(match.group(0), "")
            print("Monologue:" + monologue)
            break

        #find the first instance of an emotion, which is the first word we find in parenthesis.
        # We cannot rely on it being the first character, so let use regex and let's remove it from the content
        emotion = None
        for match in re.finditer(r'\(.*?\)', content):
            emotion = match.group(0)[1:-1]
            content = content.replace(match.group(0), "")
            print("Emotion:" + emotion)
            break

        # now get rid of the rest of the emotions in the content
        for match in re.finditer(r'\(.*?\)', content):
            content = content.replace(match.group(0), "")


        content = content.strip()
        print("Content:" + content)

        return content, emotion, monologue, actions

    def accept_observation(self, observation):
        self.observation = self.prepend_timestamp(observation, "Observation")
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
