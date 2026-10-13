
import streamlit as st

from inspect_ui import render_inspect


def display_chat_history():

    for index, message in enumerate(st.session_state.messages):

        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Show Inspect only for assistant messages
            if (
                message["role"] == "assistant"
                and "inspect" in message
            ):
                render_inspect(
                    message=message,
                    index=index
                )
