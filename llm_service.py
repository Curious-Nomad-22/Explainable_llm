
import time
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# NVIDIA API configuration
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


def call_llm(messages, temperature=0.3, max_tokens=500, top_p=1):

    max_retries = 3

    for attempt in range(max_retries):

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )

            # Validate the model response
            answer = response.choices[0].message.content

            if not answer or not answer.strip():
                raise ValueError(
                    "The model returned an empty response."
                )

            return response

        except Exception as e:

            if "503" in str(e) or "ResourceExhausted" in str(e):

                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue

            raise
