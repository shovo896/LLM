from typing import Annotated, List, TypedDict,Sequence
from dotenv import load_dotenv 
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage,ToolMessage,SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import Tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode 


load_dotenv()


email=Annotated[str, "The email address of the user"] 
print("Email:", email)
