import os
from typing import List, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[BaseMessage]

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
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAgent: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("agent", process)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)
agent = graph.compile()

conversation_history = []
user_input = input("You: ")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    state = {"messages": conversation_history}
    result = agent.invoke(state)
    conversation_history = result["messages"]
    user_input = input("You: ")

with open("logging.txt", "a") as log_file:
    log_file.write("Your conversation logged here:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            log_file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            log_file.write(f"Agent: {message.content}\n")
    log_file.write("\n")

print("Conversation logged in logging.txt")
