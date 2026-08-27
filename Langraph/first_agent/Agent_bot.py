from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import os

# ✅ OpenRouter দিয়ে Free OSS Model (Llama 3, Mistral, Gemma etc.)
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct:free",  
    api_key=os.getenv("OPENROUTER_API_KEY"),         
    base_url="https://openrouter.ai/api/v1",          
    default_headers={
        "HTTP-Referer": "http://localhost",          
        "X-Title": "AgentBot",                       
    }
)


class AgentState(TypedDict):
    messages: List[BaseMessage]


def agent_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    return state

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

app = graph.compile()


if __name__ == "__main__":
    user_input = input("তুমি: ")
    initial_state = {
        "messages": [HumanMessage(content=user_input)]
    }
    result = app.invoke(initial_state)
    last_message = result["messages"][-1]
    print(f"\nAgent: {last_message.content}")