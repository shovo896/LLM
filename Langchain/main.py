## integrate  our code with api
import os
from constants import groqapi_key
from langchain_groq import ChatGroq

import streamlit as st

os.environ['GROQ_API_KEY'] = groqapi_key


# streamlit framework

st.title("Groq API with Langchain")
# create a chat instance
input_text = st.text_input("Enter your query:")

## LLM integration with groq api ChatGroq()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.8,
)

if input_text:
    chat_groq_response = llm.invoke(input_text)
    st.write("Response from Groq API:")
    st.write(chat_groq_response.content)
    
