## conversational Q&A chatbot 

import streamlit as st 

from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain.chat_models import ChatOpenAI


## streamlit app 
st.set_page_config(page_title="Conversational Chatbot", page_icon=":robot_face:")
st.title("Conversational Chatbot with Langchain and Streamlit")
st.header("Ask me anything!")



## load dot env file 
from dotenv import load_dotenv
load_dotenv()
import os 

## initialize chat model
chat_model = ChatOpenAI(temperature=2.5,model="gpt-3.5-turbo")

if 'flowmessages' not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a helpful assistant. Answer the user's questions to the best of your ability.")]





# function to load openAI model to get responses 


def get_openai_response(question):
       st.session_state.messages.append({"role": "user", "content": question})
       response = chat_model([HumanMessage(content=question)])
       return response


