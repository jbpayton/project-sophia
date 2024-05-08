import os

from openai import OpenAI
from ConversationLogger import ConversationFileLogger

def summarize_messages(client, messages, start_index, end_index):
    # Extract the specified range of messages using slice notation
    messages_to_summarize = messages[start_index:end_index]

    # Prepare the prompt for summarization
    summary_prompt = "Please provide a concise summary of the following conversation:\n\n"
    for message in messages_to_summarize:
        role = message['role']
        content = message['content']
        summary_prompt += f"{role.capitalize()}: {content}\n"

    summary_prompt += "\nSummary:"

    # Send the summarization prompt to the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.5,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Extract the summary from the API response
    summary = response.choices[0].message.content.strip()

    return summary

def summarize_messages_verbose(client, messages, start_index, end_index):
    # Extract the specified range of messages using slice notation
    messages_to_summarize = messages[start_index:end_index]

    # Prepare the prompt for summarization
    summary_prompt = "Please provide a detailed summary of the following conversation, capturing the main points and key details:\n\n"
    for message in messages_to_summarize:
        role = message['role']
        content = message['content']
        summary_prompt += f"{role.capitalize()}: {content}\n"

    summary_prompt += "\nDetailed Summary:"

    # Send the summarization prompt to the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.1,
        max_tokens=750,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Extract the summary from the API response
    summary = response.choices[0].message.content.strip()

    return summary

if __name__ == "__main__":
    # Test the message summarization functions
    from util import load_secrets

    load_secrets("secrets.json")

    client = OpenAI(
        api_key="sk-111111111111111111111111111111111111111111111111",
        base_url=os.environ['LOCAL_TEXTGEN_API_BASE']
    )

    # create an empty list to store messages
    messages = []

    # use the ConversationLogger class to load the messages
    agent_logs = ConversationFileLogger("Sophia_logs")
    agent_logs.append_last_lines_to_messages(200, messages)

    start_index = 0
    end_index = -1
    summary = summarize_messages(client, messages, start_index, end_index)
    print("Concise Summary:")
    print(summary)

    detailed_summary = summarize_messages_verbose(client, messages, start_index, end_index)
    print("\nDetailed Summary:")
    print(detailed_summary)