import os
from pathlib import Path
from typing import List, TypedDict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to first_agent/.env.")

class AgentState(TypedDict): 
    messages: List[BaseMessage] 
    
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct:free",  
    api_key=api_key,
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
    
