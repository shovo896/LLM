from typing import Annotated ,Sequence,TypedDict 
from dotenv import load_dotenv 
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage,BaseMessage 
from langchain_openai import ChatOpenAI
from langgraph.graph.message import tool 
from langgraph.graph.message import add_messages 
from  langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv() 

document_content=""

class AgentState(TypedDict):
       messages : Annotated[Sequence[BaseMessage], add_messages]


@tool 

def update(content:str) -> str : 
       """updated the document with the provided content"""

       global document_content 
       document_content = content

       return f" Document has been updated successfully ! The current content is : \n{document_content}"

@tool 
def save(filename:str)  -> str :
       """ save the document to the text file and finish the process.
       Args : Name for the text file
       
       """
       global document_content 
       if not filename.endwith('.txt'):
              filename = filename + '.txt'

       try : 
              with open(filename,'w')  as file : 
                     file.write(document_content)
              return f"Document has been saved successfully to {filename} !"
       except Exception as e :
              return f"An error occurred while saving the document: {str(e)}"   

tools = [update,save]








