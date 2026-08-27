from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langraph import StateGraph, START, END 

import os 
load_dotenv() 

class AgentState(TypedDict): 
    messages: List[BaseMessage] 
    
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct:free",  
    api_key=os.getenv("OPENROUTER_API_KEY"),         
    base_url="https://openrouter.ai/api/v1",          
    default_headers={
        "HTTP-Referer": "http://localhost",          
        "X-Title": "AgentBot",                       
    }
)

def process(state:AgentState) -> AgentState: 
    response= llm.invoke(state["messages"]) 
    state["messages"].append(response)
    return state
    