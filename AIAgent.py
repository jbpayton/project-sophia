import datetime
import os

from openai import OpenAI
import util
import json
import re

from ConversationLogger import ConversationFileLogger

class AIAgent:
    def __init__(self, name):
        self.name = name
        self.agent_type = "AI"
        # load profile
        self.profile = util.load_profile(name)

        self.conversation_logger = ConversationFileLogger(f"{self.name}_logs")

        self.EMOTION_PROMPT = "Express your emotions by leading a sentence with parenthesis with your emotional " \
                              "state."

        self.SPEAKER_TARGETING_PROMPT = "You can specify the speaker and target by leading a sentence with " \
                                        "the speaker's name followed by '->' and the target's name. For example, " \
                                        "'Alice->Bob: (Happy) Hello!'" \
                                        "Remember, only the intended recipient can hear your messages."

        self.system_prompt = self.profile['personality'] + self.EMOTION_PROMPT + self.SPEAKER_TARGETING_PROMPT

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
        # print("Text without JSON: " + text_without_json)
        return json_objects, text_without_json

    @staticmethod
    def prepend_timestamp(text, speaker_name, target_name):
        timestamp_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp_string}] {speaker_name}->{target_name}:{text.strip()}"

    @staticmethod
    def parse_names(input_string):
        # Strip the input string of leading and trailing whitespaces
        input_string = input_string.strip()

        # Regular expression to match the input string
        pattern = re.compile(r"^(?:(?P<speaker>\w+)(->(?P<target>\w+))?:(?P<message>.*))$")
        match = pattern.match(input_string)

        if match:
            speaker = match.group("speaker")
            target = match.group("target")
            message = match.group("message").strip()
        else:
            # Handle case where no speaker or target is specified
            speaker = None
            target = None
            message = input_string.strip()

        return speaker, target, message

    def who_should_be_the_next_speaker(self, message_count=5):
        # Prepare the context from the last messages
        context = " ".join([msg["content"] for msg in self.messages[-message_count:]])

        # Create a prompt for the LLM to determine the next speaker
        prompt = f"Based on the following conversation, who should be the next speaker? Provide only the name or identifier of the next speaker without any additional text.\n\n{context}"

        # Use the LLM to generate a suggestion
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",  # Ensure you use the correct model identifier
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Adjust temperature to control randomness
            max_tokens=10,  # Adjust max_tokens to limit the response length
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Parse the response to extract the suggested next speaker
        next_speaker = response.choices[0].message.content.strip()
        print("Suggested next speaker:", next_speaker)
        return next_speaker

    def inner_send(self, sender="User"):
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

        # print("Raw Content from LLM:" + content)

        prompt_token_count = response.usage.prompt_tokens
        print(f"Prompt token count: {prompt_token_count}")

        total_token_count = response.usage.total_tokens
        print(f"Total token count: {total_token_count}")

        # get rid of the timestamp, if present, with regex
        content = re.sub(r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]', '', content)

        # print("Content after timestamp removal:" + content)

        # parse out the names
        parsed_speaker, parsed_target, content = self.parse_names(content)

        if parsed_speaker is None:
            parsed_speaker = self.name

        if parsed_speaker != self.name:
            # this should never, happen, so we need to bail out
            print("ERROR: Speaker is not the AI agent")
            return None, None, None, None, None

        if parsed_target is None:
            parsed_target = sender

        stored_content = content
        stored_content = self.prepend_timestamp(stored_content, self.name, parsed_target)
        print(stored_content)
        self.messages.append({"role": role, "content": stored_content})
        self.conversation_logger.log_message(role, stored_content)

        # print("Content, before Command Removal:" + content)

        actions, content = self.extract_json_objects(content)
        content = content.strip()

        # print("Content, after Command Removal:" + content)

        # TODO: We we need to handle the internal monologue here, might do this as a self targeted chat completion
        monologue = ""

        # find the first instance of an emotion, which is the first word we find in parenthesis.
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

        return content, emotion, monologue, actions, parsed_target

    def send(self, message, sender="User"):
        # if is isn't in the agent manager, attempt to add it as a human agent
        from AgentManager import AgentManager
        AgentManager.add_human_agent(sender)

        message = self.prepend_timestamp(message, sender, self.name)
        print(message)
        self.messages.append({"role": "user", "content": message})
        self.conversation_logger.log_message("user", message)

        content, emotion, monologue, actions, parsed_target = self.inner_send(sender)

        if parsed_target is not sender:
            # if the message from the AI is not intended for the one making the request, send it to the intended target
            print(f"Sending to {parsed_target}")
            agent_response = AgentManager.send_to_agent(parsed_target, content, self.name)

        next_speaker = self.who_should_be_the_next_speaker()

        while next_speaker == self.name:
            print("AI should be the next speaker")
            content, emotion, monologue, actions, parsed_target = self.inner_send(parsed_target)

            if parsed_target != sender:
                print(f"Sending to {parsed_target}")
                agent_response = AgentManager.send_to_agent(parsed_target, content, self.name)

            next_speaker = self.who_should_be_the_next_speaker()

        return content, emotion, monologue, actions


if __name__ == "__main__":
    util.load_secrets()
    agent = AIAgent("Sophia")
    while True:
        # prompt user for input
        in_message = input("Enter a message: ")
        # send message to agent
        test_response, test_emotion, test_monologue, test_actions = agent.send(in_message)
        # print response
        print(test_response)
