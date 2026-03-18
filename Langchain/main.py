import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from langchain_groq import ChatGroq


def get_groq_api_key() -> str | None:
    """Load the Groq API key from Streamlit secrets or environment variables."""
    try:
        return st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    except StreamlitSecretNotFoundError:
        return os.getenv("GROQ_API_KEY")


groq_api_key = get_groq_api_key()
if not groq_api_key:
    st.error(
        "Missing `GROQ_API_KEY`. Add it to Streamlit secrets for deployment "
        "or export it as an environment variable locally."
    )
    st.stop()


# streamlit framework

st.title("Groq API with Langchain")
# create a chat instance
input_text = st.text_input("Enter your query:")

## LLM integration with groq api ChatGroq()

llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.1-8b-instant",
    temperature=0.8,
)

if input_text:
    chat_groq_response = llm.invoke(input_text)
    st.write("Response from Groq API:")
    st.write(chat_groq_response.content)
    
