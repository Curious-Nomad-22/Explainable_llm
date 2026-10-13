
import streamlit as st

from llm_service import call_llm


def render_inspect(message, index):

    # Create a separate Inspect conversation for this response
    if index not in st.session_state.inspect_chats:
        st.session_state.inspect_chats[index] = []

    inspect_history = st.session_state.inspect_chats[index]

    with st.expander("🔍 Inspect"):

        inspect = message["inspect"]

        # Inspection information
        st.write("**Model:**", inspect["model"])
        st.write("**Latency:**", inspect["latency"])

        if inspect["input_tokens"] is not None:
            st.write("**Input tokens:**", inspect["input_tokens"])
            st.write("**Output tokens:**", inspect["output_tokens"])
            st.write("**Total tokens:**", inspect["total_tokens"])

        st.divider()

        st.write("### Ask about this response")

        # Display previous Inspect conversation
        for inspect_message in inspect_history:
            with st.chat_message(inspect_message["role"]):
                st.write(inspect_message["content"])

        # Inspect input
        inspect_question = st.text_input(
            "Ask something about this response",
            key=f"inspect_question_{index}_{len(inspect_history)}"
        )

        if st.button("Ask", key=f"inspect_button_{index}"):

            if inspect_question.strip():

                # Build Inspect conversation
                inspect_messages = [
                    {
                        "role": "system",
                        "content": """
You are the Inspect Assistant for an Explainable LLM.

Explain a specific assistant response and its technical information.

You have access to:
- The original assistant response
- Model information
- Latency
- Token usage
- Previous questions and answers from this Inspect session

Maintain conversational context throughout the Inspect session.

Do not invent technical information that is unavailable.
If something cannot be determined, say so clearly.
"""
                    },
                    {
                        "role": "user",
                        "content": f"""
Here is the assistant response being inspected:

{message["content"]}

Inspection information:

Model: {inspect["model"]}
Latency: {inspect["latency"]}
Input tokens: {inspect["input_tokens"]}
Output tokens: {inspect["output_tokens"]}
Total tokens: {inspect["total_tokens"]}
"""
                    }
                ]

                # Add previous Inspect conversation
                inspect_messages.extend(inspect_history)

                # Add current question
                inspect_messages.append({
                    "role": "user",
                    "content": inspect_question
                })

                # Ask the LLM
                try:
                    inspect_response = call_llm(
                        messages=inspect_messages,
                        temperature=0.3,
                        max_tokens=500
                    )

                    choice = inspect_response.choices[0]
                    inspect_answer = choice.message.content
                    
                    if not inspect_answer or not inspect_answer.strip():
                        st.error(
                            "The model returned an empty response. "
                            f"Finish reason: {choice.finish_reason}"
                        )
                        st.stop()

                    # Save Inspect conversation
                    inspect_history.append({
                        "role": "user",
                        "content": inspect_question
                    })

                    inspect_history.append({
                        "role": "assistant",
                        "content": inspect_answer
                    })

                    st.rerun()

                except Exception as e:
                    st.error(f"Inspect request failed: {e}")
