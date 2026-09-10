import os
from pathlib import Path
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv(Path(__file__).resolve().parents[2] / "vectordatabase_langchain" / ".env")

document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Update the document with the provided content."""
    global document_content
    document_content = content
    return (
        "Document has been updated successfully! The current content is:\n"
        f"{document_content}"
    )


@tool
def save(filename: str) -> str:
    """Save the document to a text file."""
    global document_content

    if not filename.endswith(".txt"):
        filename += ".txt"

    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(document_content)
        return f"Document has been saved successfully to {filename}!"
    except Exception as e:
        return f"An error occurred while saving the document: {e}"


tools = [update, save]

model = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AgentBot",
    },
).bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""You are a drafter. You help the user update and modify documents.

- If the user wants to update or modify content, use the update tool with the complete updated content.
- If the user wants to save the document, use the save tool.
- Always allow modifications.
- The current document is:

{document_content}
"""
    )

    if not state["messages"]:
        user_input = HumanMessage(
            content="I am ready to help you update a document. What would you like to create?"
        )
    else:
        user_text = input("\nWhat would you like to do with the document? ")
        print(f"\nUser: {user_text}")
        user_input = HumanMessage(content=user_text)

    all_messages = [system_prompt, *state["messages"], user_input]
    response = model.invoke(all_messages)

    print(f"\nAgent: {response.content}")
    if response.tool_calls:
        print(f"Using tools: {[tool_call['name'] for tool_call in response.tool_calls]}")

    return {"messages": [user_input, response]}


def print_messages(messages):
    """Print saved tool messages in a readable format."""
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            content = str(message.content)
            if "saved" in content.lower():
                print(f"\nTool: {content}")


graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()
