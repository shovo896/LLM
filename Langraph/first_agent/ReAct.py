from typing import Annotated, List, TypedDict,Sequence
from dotenv import load_dotenv 
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage,ToolMessage,SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import Tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode 


load_dotenv()

def get_llm() -> ChatOpenAI: 
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to first_agent/.env.")
    return ChatOpenAI(
        model="nvidia/nemotron-3.5-lightning:free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",          
        default_headers={
            "HTTP-Referer": "http://localhost",          
            "X-Title": "AgentBot",                       
        }
    ) 
    
def get_tools() -> List[Tool]:
    return [
        Tool(
            name="search",
            description="useful for when you need to answer questions about current events or the current state of the world",
            func=lambda query: f"Search results for '{query}'"
        ),
        Tool(
            name="calculator",
            description="useful for when you need to answer questions about math",
            func=lambda expression: f"Result of '{expression}'"
        )
    ]