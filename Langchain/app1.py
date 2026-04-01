import os
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from streamlit.errors import StreamlitSecretNotFoundError

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant."


def load_local_env() -> None:
    """Load environment variables from the working directory or the app directory."""
    if load_dotenv is None:
        return

    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def get_openai_api_key() -> str:
    """Load the OpenAI API key from Streamlit secrets, the environment, or the UI."""
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except StreamlitSecretNotFoundError:
        secret_key = ""

    return (
        secret_key
        or os.getenv("OPENAI_API_KEY", "")
        or st.session_state.get("openai_api_key", "")
    ).strip()


def build_message_history() -> list[SystemMessage | HumanMessage | AIMessage]:
    """Convert Streamlit session messages into LangChain chat messages."""
    history: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]

    for message in st.session_state.messages:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        else:
            history.append(AIMessage(content=message["content"]))

    return history


load_local_env()

st.set_page_config(page_title="Conversational Chatbot", page_icon=":robot_face:")
st.title("Conversational Chatbot with LangChain and Streamlit")
st.caption("Chat with OpenAI using LangChain.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Add your OpenAI API key in the sidebar, then start chatting.",
        }
    ]

with st.sidebar:
    st.header("Settings")
    entered_api_key = st.text_input(
        "OpenAI API key",
        value=st.session_state.get("openai_api_key", ""),
        type="password",
        help="You can also set OPENAI_API_KEY in your environment or Streamlit secrets.",
    )
    st.session_state.openai_api_key = entered_api_key.strip()

    model_name = st.text_input(
        "OpenAI model",
        value=st.session_state.get("model_name", DEFAULT_MODEL),
        help="Change this if your account uses a different model name.",
    ).strip()
    st.session_state.model_name = model_name or DEFAULT_MODEL

    if st.button("Clear chat history"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat history cleared. Ask a new question when you're ready.",
            }
        ]
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

openai_api_key = get_openai_api_key()
if not openai_api_key:
    st.info(
        "Enter a valid `OPENAI_API_KEY` in the sidebar, Streamlit secrets, or a local `.env` file."
    )
    st.stop()

prompt = st.chat_input("Ask something")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            llm = ChatOpenAI(
                api_key=openai_api_key,
                model=st.session_state.model_name,
                temperature=0.8,
            )
            response = llm.invoke(build_message_history())
            answer = response.content.strip() if response.content else "No response returned."
            st.markdown(answer)
        except Exception as exc:
            answer = (
                "The request failed. Check your API key, selected model, and network access.\n\n"
                f"Details: `{exc}`"
            )
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
