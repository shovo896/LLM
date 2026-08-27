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
    
    

    
    


