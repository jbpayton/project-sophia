from openai import OpenAI
import demjson
import re
import ToolLoader


class ActionInterpreter:
    def __init__(self, tools=None):
        self.INSTRUCTION_PROMPT = "You are a translator between the intents of a user and tools. These tools " \
                                  "represent actions of the user. Seek to understand the user's intent, and select " \
                                  "the appropriate tool and give the correct format in JSON. Use some common sense " \
                                  "and guess if need be. If a user explicitly asks, let them know the names and" \
                                  "parameter names (NO JSON!). " \
                                  "If a user does not give enough information to use a tool, "\
                                  "please ask the user again. If you are confident that you understand the user's " \
                                  "intent, include no other text after the JSON."

        self.TOOL_PROMPT_INTRO = "The following actions are available (use only the tools you have):"

        self.system_prompt = self.INSTRUCTION_PROMPT + self.TOOL_PROMPT_INTRO

        self.client = OpenAI(
            api_key="sk-111111111111111111111111111111111111111111111111",
            base_url="http://192.168.2.94:5000/v1"
        )

        self.tools = tools
        if tools:
            self.set_tools(tools)
        else:
            self.messages = [
                {"role": "user", "content": self.system_prompt}
            ]

    def set_tools(self, tools):
        self.tools = tools
        self.refresh_tool_prompt()

    def format_tool_prompt(self, tools):
        # Create a list to hold the tool definitions
        tool_list = []

        # Iterate over the tools and construct the tool definitions
        for actionName, tool_info in tools.items():
            tool_dict = {"actionName": actionName}
            for param in tool_info["params"]:
                tool_dict[param] = f"<{param}>"
            tool_list.append(actionName + " - " + tool_info["description"] + "\n" + str(tool_dict))

        return tool_list

    def refresh_tool_prompt(self):
        tools_strings = self.format_tool_prompt(self.tools)
        self.TOOL_PROMPT = "\n".join(tools_strings)
        self.system_prompt = self.INSTRUCTION_PROMPT + self.TOOL_PROMPT_INTRO + self.TOOL_PROMPT
        self.reset_messages()

    def reset_messages(self):
        self.messages = [
            {"role": "user", "content": self.system_prompt}
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
                json_obj = demjson.decode(match)
            except demjson.JSONDecodeError:
                continue  # Skip invalid JSON

            if "actionName" in json_obj:
                json_objects.append(json_obj)
            # remove the JSON object from the original text
            text_without_json = text_without_json.replace(match, "")
        # print("JSON Objects: " + str(json_objects))

        return json_objects, text_without_json

    def send(self, message, max_retries=3, verbose=False):
        actions, content = self.inner_send(message, verbose)

        successfully_executed = False
        for action in actions:
            retries = max_retries
            tool_response, tool_success = ToolLoader.execute_tool(action, self.tools)

            print(f"Action: {action['actionName']}, Response: {tool_response}, Success: {tool_success}")

            # on the first successful execution, reset the content to an empty string
            if tool_success:
                if successfully_executed is False:
                    content = ""

                content += f"\n{action['actionName']}: {tool_response}"
                successfully_executed = True
            else:
                while retries > 0:
                    retries -= 1
                    print(f"Retrying {action['actionName']}... {retries} retries left.")
                    tool_response += "\nLets try again."
                    actions, tool_response = self.inner_send(tool_response, verbose)
                    tool_response, tool_success = ToolLoader.execute_tool(actions[0], self.tools)
                    if tool_success:
                        content += f"\n{action['actionName']}: {tool_response}"
                        successfully_executed = True
                        break

        if successfully_executed:
            self.reset_messages()

        return content, successfully_executed

    def inner_send(self, message, verbose):
        if verbose:
            print(f"User: {message}")
        self.messages.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=self.messages,
            temperature=0.0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        if verbose:
            print(f"{response.choices[0].message.role}: {response.choices[0].message.content}")
        role = response.choices[0].message.role
        content = response.choices[0].message.content
        self.messages.append({"role": role, "content": content})
        actions, content = self.extract_json_objects(content)
        content = content.strip()
        return actions, content


if __name__ == "__main__":
    tools = ToolLoader.load_tools_from_files()

    agent = ActionInterpreter(tools)

    response, success = agent.send("Search duck duck go for 4 results about bacon", verbose=True)
    print(response)
    print(success)

    response, success = agent.send("find me some moon facts", verbose=True)
    print(response)
    print(success)

    response, success = agent.send("Search the web for weather in Fredericksburg, VA", verbose=True)
    print(response)
    print(success)

'''
    response, success = agent.send("Save a the following text to a file called 'test.txt': 'Hello, World!'", verbose=True)
    print(response)
    print(success)

    response, success = agent.send("What actions can I do?", verbose=True)
    print(response)
    print(success)

    response, success = agent.send("What files are in the current directory?", verbose=True)
    print(response)
    print(success)
'''