
import streamlit as st
import time

from llm_service import call_llm, MODEL
from session_store import initialize_session
from chat_ui import display_chat_history



st.title("🤖 Explainable LLM")


initialize_session()

display_chat_history()

user_input = st.chat_input("Ask something...")


if user_input:

    # Start latency timer
    start_time = time.perf_counter()

    # Prepare conversation messages
    messages = [
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in st.session_state.messages
    ] + [
        {
            "role": "user",
            "content": user_input
        }
    ]

    try:

        # Call NVIDIA LLM
        response = call_llm(
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=2048
        )

        # Calculate latency
        latency = time.perf_counter() - start_time

        # Get assistant answer
        answer = response.choices[0].message.content

        # -----------------------------
        # Token Information
        # -----------------------------

        input_tokens = None
        output_tokens = None
        total_tokens = None

        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

        # -----------------------------
        # Save User Message
        # -----------------------------

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # -----------------------------
        # Save Assistant Message
        # -----------------------------

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,

            "inspect": {
                "model": MODEL,
                "latency": f"{latency:.2f} seconds",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        })

        # Refresh the application
        st.rerun()

    except Exception as e:
        st.error(f"Unable to generate a response: {e}")
