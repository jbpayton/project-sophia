import json
import os
import re
from time import sleep

from openai import OpenAI
from typing import List, Dict, Union, Tuple
from ContextSummarizers import summarize_messages_tuples_simpler, msgs2string


class PromptEngineer:
    def __init__(self, client: OpenAI):
        self.client = client

    def generate_prompt(self, conversations: List[str], current_prompt: str, feedback: str) -> str:
        system_message = """You are an expert prompt engineer. Your task is to create or improve a prompt for an AI model 
        that will extract key facts from conversations as JSON objects with 'topic', 'subject', 'relationship', 'object', 
        and 'source' fields. Consider the following:
        1. The current prompt and its performance across multiple conversations
        2. The feedback provided on the current output
        3. Prompt conciseness, clarity, and completeness
        
        This is the crieria you that will be used to evaluate your prompt:
        1. Accuracy of the extracted information
        2. Completeness (is any important information missing?)
        3. Relevance of the extracted information
        4. Clarity and conciseness of the JSON objects
        5. Proper use of the JSON format and field names
        6. Any parsing errors or issues with the JSON format
        7. Subject, relationship, and object should make sense without the subject or source field
        8. Favor shorter, prompts that are more likely to be accurate over longer, more complex prompts

        Provide a new or improved prompt that addresses any shortcomings and aims to capture all relevant information 
        in the conversations accurately as JSON objects."""

        user_message = f"""Current prompt:
        {current_prompt}

        Feedback on current output:
        {feedback}

        Please provide an improved prompt:"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=4000
        )

        return response.choices[0].message.content.strip()


class Evaluator:
    def __init__(self, client: OpenAI):
        self.client = client

    def evaluate_output(self, conversation: str, prompt: str, json_str: str, parsed_objects: Union[List[Dict], str]) -> Tuple[
        str, float]:
        system_message = """You are an expert in information extraction and summarization. Your task is to evaluate 
        the quality of JSON objects extracted from a conversation. Each object should have 'topic', 'subject', 
        'relationship', 'object', and 'source' fields. Consider the following:
        1. Accuracy of the extracted information
        2. Completeness (is any important information missing?)
        3. Relevance of the extracted information
        4. Clarity and conciseness of the JSON objects
        5. Proper use of the JSON format and field names
        6. Any parsing errors or issues with the JSON format
        7. Subject, relationship, and object should make sense without the subject or source field
        8. Favor shorter, prompts that are more likely to be accurate over longer, more complex prompts

        Provide detailed feedback on the quality of the extraction, pointing out any issues or areas for improvement.
        Do not include suggested output, but provide guidance on how to improve the extraction.
        At the end of your feedback, provide a score from 0 to 10, where 10 is perfect extraction and 0 is completely incorrect or failed extraction. Be a harsh grader.
        Format the score line exactly as follows: "Score: X.X" (where X.X is the numerical score)."""

        user_message = f"""
        Prompt:
        {prompt}
        
        Conversation:
        {conversation}

        Raw JSON output:
        {json_str}

        Parsed JSON objects or error message:
        {parsed_objects}

        Please evaluate the quality of the extraction, provide feedback, and give a score (0-10):"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=4000
        )

        feedback = response.choices[0].message.content.strip()

        # Extract the score from the feedback using regex
        score_match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", feedback)
        if score_match:
            score = float(score_match.group(1))
        else:
            print("Warning: Could not extract score from feedback. Defaulting to 0.")
            score = 0.0

        return feedback, score


def parse_json_objects(json_str: str) -> Union[List[Dict], str]:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {str(e)}"


def iterative_prompt_improvement(client: OpenAI, trainer_client: OpenAI, conversations: List[str], initial_prompt: str, iterations: int) -> \
Tuple[str, float]:
    prompt_engineer = PromptEngineer(trainer_client)
    evaluator = Evaluator(trainer_client)

    current_prompt = initial_prompt
    best_prompt = current_prompt
    best_score = 0.0

    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")

        total_score = 0.0
        all_feedback = []

        for j, conversation in enumerate(conversations):
            # print the conversation in blue
            print(f"Conversation:")
            print(f"\033[94m{conversation}\033[00m")
            print(f"Evaluating conversation {j + 1}/{len(conversations)}")

            json_str = summarize_messages_tuples_simpler(client, conversation, summarize_first=False,
                                                         custom_prompt=current_prompt)

            #print the raw json text in green
            print(f"Raw JSON output:")
            print(f"\033[92m{json_str}\033[00m")


            parsed_objects = parse_json_objects(json_str)

            feedback, score = evaluator.evaluate_output(conversation, current_prompt, json_str, parsed_objects)
            print(f"Feedback: {feedback}\n")
            print(f"Score: {score}\n")

            total_score += score
            all_feedback.append(feedback)

        avg_score = total_score / len(conversations)
        print(f"Average score for iteration: {avg_score:.2f}\n")

        if avg_score > best_score:
            best_score = avg_score
            best_prompt = current_prompt

        # Generate a new prompt based on all feedback
        new_prompt = prompt_engineer.generate_prompt(conversations, current_prompt, "\n\n".join(all_feedback))
        print(f"New prompt:")

        # print new prompt in yellow
        print(f"\033[93m{new_prompt}\033[00m")

        current_prompt = new_prompt

    print(f"Best prompt: {best_prompt}")
    print(f"Best score: {best_score}")
    return best_prompt, best_score


def optimize_json_extraction_prompt(sample_conversations: List[List[str]], iterations: int = 3) -> str:
    client = OpenAI(
        api_key="sk-111111111111111111111111111111111111111111111111",
        base_url=os.environ['LOCAL_TEXTGEN_API_BASE']
    )

    load_secrets()
    client_2 = OpenAI(
        api_key=os.environ['OPENAI_API_KEY']
    )

    initial_prompt = '''
        You are a smart assistant. Your task is to break down a conversation into key facts. Each fact should be in the form of a tuple (topic, subject, relationship, object, source). Here’s how to do it:

        1. **Topic**: What is the general category or theme of the fact?
        2. **Subject**: Who or what is the main focus of the fact?
        3. **Relationship**: What is the connection or relationship between the subject and the object?
        4. **Object**: What is the subject connected to? (An entity, concept, or another subject)
        5. **Source**: Who provided this fact?

        Only include clear, factual information. Ignore greetings, small talk, and irrelevant details.

        **Examples**:
        ```json
        [
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
            "object": "shonen anime",
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
        ]
        ```

        Now, extract the key facts from the following conversation and format them as a JSON array of tuples:
        '''

    # Convert sample conversations to strings
    conversations = [msgs2string(conv) for conv in sample_conversations]

    best_prompt, best_score = iterative_prompt_improvement(client, client_2, conversations, initial_prompt, iterations)

    print(f"\nBest prompt (score: {best_score}):\n{best_prompt}")

    # Test the optimized prompt on all conversations
    for i, conversation in enumerate(conversations):
        print(f"\nTesting optimized prompt on conversation {i + 1}:")
        print(conversation)
        json_str = summarize_messages_tuples_simpler(client, conversation, summarize_first=True,
                                                     custom_prompt=best_prompt)
        print("\nRaw JSON output:")
        print(json_str)
        parsed_objects = parse_json_objects(json_str)
        print("\nParsed JSON objects or error message:")
        print(json.dumps(parsed_objects, indent=2) if isinstance(parsed_objects, list) else parsed_objects)

    return best_prompt


# Example usage
if __name__ == "__main__":
    from util import load_secrets
    load_secrets()

    sample_conversations = [
        [
            "Alice: I would like talk about the characters in Evangelion.",
            "Bob: Evangelion is one of the most popular anime series of all time.",
            "Alice: I think I saw it on my birthday.",
            "Bob: When is your birthday?",
            "Alice: MY birthday is May 2nd, when is yours?",
            "Bob: Mine is in October. But back to Evangelion, what do you think about the characters?",
            "Alice: I think Rei might not come off as a very complex character but she seems to symbolize something.",
            "Bob: That's an interesting observation. Do you think the other characters are symbolic too?",
            "Alice: I wonder about that. What do you think?"
        ],
        [
            "Charlie: Have you ever watched Cowboy Bebop?",
            "David: Yes, it's a classic anime from the late 90s.",
            "Charlie: What do you know about the main character?",
            "David: The main character, Spike Spiegel, is a former hitman turned bounty hunter.",
            "Charlie: That sounds interesting. What genre would you say it is?",
            "David: The show blends genres like sci-fi, western, and film noir.",
            "Charlie: Wow, that's quite a mix. What do you think about anime that mix different genres?",
            "David: I find them really creative and engaging. How about you?",
            "Charlie: I agree, it keeps things fresh and unpredictable."
        ],
        [
            "Alice: Hey Bob, did you hear about the new AI that can write poetry?",
            "Bob: Oh great, another 'creative' AI. I'm sure it writes beautiful odes to binary code and error messages.",
            "Alice: Actually, it's quite impressive. It even won a literary award last month.",
            "Bob: Wait, seriously? I was just kidding. What's next, AI composing symphonies?",
            "Alice: Funny you should mention that. The London Philharmonic just performed a piece composed by an AI.",
            "Bob: You're pulling my leg now, aren't you?",
            "Alice: Not at all! It's called 'Symphony in B Major: The Electronic Dream'. Critics are divided, though.",
            "Bob: I can imagine. So, what do you think about all this AI creativity? Good thing? Bad thing?",
            "Alice: It's complicated. On one hand, it's fascinating to see what AI can do. On the other, I worry about human artists.",
            "Bob: Yeah, I get that. Hey, speaking of art, are you still painting?",
            "Alice: I haven't touched a brush in months. Been too busy with work.",
            "Bob: That's a shame. Your last piece was amazing. The one with the, uh, swirly things and the bird.",
            "Alice: You mean 'Tempest in Azure'? Thanks, but it wasn't a bird, it was supposed to be a dragon."
        ],
        [
            "Alex: Hey Sam, did you watch the game last night?",
            "Sam: You know I don't follow sports. I was busy with my new project.",
            "Alex: Oh right, the one with the things and the stuff. How's that going?",
            "Sam: It's not 'things and stuff'. I'm developing an app that translates cat meows.",
            "Alex: Seriously? I thought you were joking about that last time."
        ]
    ]

    optimized_prompt = optimize_json_extraction_prompt(sample_conversations, iterations=10)
    print(f"\nFinal optimized prompt:\n{optimized_prompt}")