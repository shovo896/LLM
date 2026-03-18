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

ChatGroq = ChatGroq(temperature=0.7)

if input_text:
    

