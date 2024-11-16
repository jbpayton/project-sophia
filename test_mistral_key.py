import os
from mistralai import Mistral, models
import util


util.load_secrets()
# Load the API key from the environment
api_key = os.environ["MISTRAL_API_KEY"]

model = "open-mixtral-8x7b"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ],
    presence_penalty=None,
    frequency_penalty=None
)


print(chat_response.choices[0].message.content)