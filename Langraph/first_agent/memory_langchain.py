import os 
from typing import List, TypedDict,Union 
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage 
from langchain_openai import ChatOpenAI 
from langgraph.graph import END, START, StateGraph 
from dotenv import load_dotenv 
load_dotenv()





class AgentState(TypedDict):
    #messages: List[HumanMessage]
    #messages_ai:List[Union[AIMessage]] 
    messages: Union[HumanMessage, AIMessage]
    
    
llm = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AgentBot",
    }
)

def process(state: AgentState) -> AgentState:
    """this node will solve request you input and return the response from the LLM"""
    
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state 



    
    


