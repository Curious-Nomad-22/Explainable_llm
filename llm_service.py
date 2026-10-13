
import time
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")

if not API_KEY:
    raise ValueError(
        "NVIDIA_API_KEY is missing. Check your local .env file."
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
    timeout=60.0,
    max_retries=0
)

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


def call_llm(messages, temperature=0.3, max_tokens=500, top_p=1):

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )

            if not response.choices:
                raise ValueError(
                    "The API returned no choices."
                )

            choice = response.choices[0]
            answer = choice.message.content

            if not isinstance(answer, str) or not answer.strip():
                finish_reason = choice.finish_reason
                refusal = getattr(
                    choice.message, "refusal", None
                )

                raise ValueError(
                    "Empty model response. "
                    f"Finish reason: {finish_reason}; "
                    f"Refusal: {refusal}"
                )

            return response

        except Exception as e:
            last_error = e
            error_text = str(e).lower()

            retryable = (
                "503" in error_text
                or "resourceexhausted" in error_text
                or "empty model response" in error_text
                or "returned no choices" in error_text
            )

            if retryable and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue

            raise

    raise RuntimeError(
        f"LLM request failed after retries: {last_error}"
    )
