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
    model="nvidia/nemotron-3.5-lightning:free",
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

graph = StateGraph(AgentState) 
graph.add_node("agent", process) 
graph.add_edge(START,'agent')
graph.add_edge('agent',END) 
agent_app = graph.compile() 

if __name__ == "__main__":
    messages: List[BaseMessage] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat ended.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Chat ended.")
            break
        if not user_input:
            continue

        try:
            result = agent_app.invoke(
                {"messages": [*messages, HumanMessage(content=user_input)]}
            )
            messages = result["messages"]
            print(f"Agent: {messages[-1].content}")
        except Exception as error:
            print(f"Request failed: {type(error).__name__}: {error}")



    
