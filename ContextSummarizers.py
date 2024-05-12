import os

from openai import OpenAI
from ConversationLogger import ConversationFileLogger


def msgs2string(messages, start_index, end_index):
    formatted_messages = []
    for message in messages[start_index:end_index]:
        content = message['content']
        formatted_message = f"{content}"
        formatted_messages.append(formatted_message)

    return "\n".join(formatted_messages)

def summarize_messages(client, messages, start_index, end_index):
    # Extract the specified range of messages using slice notation
    messages_to_summarize = messages[start_index:end_index]

    # Prepare the prompt for summarization
    summary_prompt = "Please provide a concise summary of the following conversation:\n\n"

    messages_string = msgs2string(messages, start_index, end_index)
    summary_prompt += messages_string

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
    summary_prompt = "Please provide a comprehensive and detailed analysis of the following conversation, capturing " \
                     "all the key points, topics, and the speakers' perspectives, preferences, and opinions. Include " \
                     "relevant details and examples to ensure a thorough understanding of the conversation:\n\n "

    messages_string = msgs2string(messages, start_index, end_index)
    summary_prompt += messages_string

    summary_prompt += "\nComprehensive Analysis:"

    # Send the summarization prompt to the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Extract the summary from the API response
    summary = response.choices[0].message.content.strip()

    return summary


def summarize_messages_tuples(client, messages, start_index, end_index, summarize_first=True):
    if summarize_first:
        input_text = summarize_messages_verbose(client, messages, start_index, end_index)
    else:
        input_text = msgs2string(messages, start_index, end_index)

    summary_prompt = '''
    Extract factual information from the provided text as (topic, subject, relationship, object, source) tuples. Follow these guidelines:
    1. Focus on fundamental relationships between entities, including the preferences and opinions of the speakers.
    2. Treat the speakers as subjects when they express preferences, opinions, or thoughts about topics or entities.
    3. Exclude conversational elements like greetings, thanks, or other exchanges that don't contribute to the main topics.
    4. Ensure that each tuple is a complete thought and doesn't require additional context.
    5. Topic should be a broad category that encompasses the subject. The topic of the conversation containing the subject.
    6. Choose an accurate relationship to convey the core meaning.
    7. The object should be a single entity or concept that is directly related to the subject and relationship.
    8. Keep the object concise and clear, focusing on the most essential information.
    9. Use the speakers' names instead of pronouns when referring to them in the tuples.
    10. Make sure the source is the speaker's name or a referenced source.
    11. Capture inferred identities, such as the type or category of an entity (e.g., Mobile Suit Gundam is an anime).
    12. Separate objects into individual tuples when multiple objects are mentioned in relation to a subject and relationship.

    Here are a few examples of tuples:
    {
      "topic": "Favorite anime",
      "subject": "Alice",
      "relationship": "likes",
      "object": "Naruto",
      "source": "Alice"
    },
    {
      "topic": "Anime genre",
      "subject": "Naruto",
      "relationship": "is",
      "object": "shonen",
      "source": "Alice"
    },
    {
      "topic": "Hobby",
      "subject": "Bob",
      "relationship": "enjoys",
      "object": "cosplaying",
      "source": "Bob"
    },
    {
      "topic": "Favorite character",
      "subject": "Alice",
      "relationship": "admires",
      "object": "Kakashi",
      "source": "Alice"
    }

    Now, extract tuples from the following text and format the output as a JSON array of tuple objects:
    :\n\n'''

    prompt = summary_prompt + input_text

    prompt += "\nTuples:"

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

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

    start_index = -30
    end_index = -1

    original_text = msgs2string(messages, start_index, end_index)
    print("Original Text:")
    print(original_text)

    summary = summarize_messages(client, messages, start_index, end_index)
    print("Concise Summary:")
    print(summary)

    detailed_summary = summarize_messages_verbose(client, messages, start_index, end_index)
    print("\nDetailed Summary:")
    print(detailed_summary)

    triples = summarize_messages_tuples(client, messages, start_index, end_index)
    print("\nExtracted Tuples (1, with summary):")
    print(triples)



