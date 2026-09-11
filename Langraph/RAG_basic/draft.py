import dotenv import load_dotenv 
import os 
from langgraph.graph import StateGraph,END 
from typing import TypedDict , Annotated ,Sequence 
from langchain_core.messages import BaseMessages,SystemMessage,HumanMessage,ToolMessage 
from operator import add as add_messages 
from langchain_openai import  ChatOpenAI 
from langchain_openai import OpenAIEmbeddings 
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_core.tools import tool


load_dotenv()

llm = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AgentBot",
    }
)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 


pdf_path = "../ABUIABA9GAAghIK0ugYowM2h3QY.pdf"

if not os.path.exists(pdf_path): 
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")
pdf_loader=PyPDFLoader(pdf_path)


try : 
    pages= pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages ")
    
    
except as e : 
    print(f ' Error loading PDF : {e}')
    
    raise  


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap = 200 
    
)







