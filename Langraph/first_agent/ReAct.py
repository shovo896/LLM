from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import os

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    email: Annotated[str, "The email address of the user"]

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

tools = [add]

model = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AgentBot",
    }
).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my helpful assistant. You will answer the user's questions and provide assistance as needed.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response], "email": state["email"]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "our_agent")
graph.add_conditional_edges("our_agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s[list(s.keys())[-1]]["messages"][-1]
        print(f"{message.type}: {message.content}")
        if hasattr(message, "tool_calls") and message.tool_calls:
            print(f"Tool Calls: {message.tool_calls}")


inputs = {"messages": [HumanMessage(content="Hello! How are you?")], "email": "sjxbhj@gmail.com"}
stream = app.stream(inputs)
print_stream(stream)
