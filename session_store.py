
import streamlit as st


def initialize_session():

    # Main chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Separate conversation history for each Inspect session
    if "inspect_chats" not in st.session_state:
        st.session_state.inspect_chats = {}