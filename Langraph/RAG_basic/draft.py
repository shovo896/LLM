"""Interactive PDF RAG assistant backed by Chroma and OpenRouter."""

import hashlib
import os
import shutil
from operator import add as add_messages
from pathlib import Path
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "ABUIABA9GAAghIK0ugYowM2h3QY.pdf"
PERSIST_DIRECTORY = Path(__file__).resolve().parent / "db"
COLLECTION_NAME = "stock_market"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Put OPENROUTER_API_KEY in a .env file at the project root (LLM/.env), or
# export it in the terminal before running this file.
load_dotenv(PROJECT_ROOT / ".env")


def require_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to LLM/.env or export it in the terminal."
        )
    return api_key


def build_rag_agent():
    api_key = require_api_key()
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

    llm = ChatOpenAI(
        model="nvidia/nemotron-3.5-lightning:free",
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "AgentBot"},
        timeout=30,
        max_retries=1,
    )
    # OpenRouter offers an OpenAI-compatible embeddings endpoint, so the same
    # project key is used for retrieval as for chat.
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "AgentBot"},
        timeout=30,
        max_retries=1,
    )

    try:
        pages = PyPDFLoader(str(PDF_PATH)).load()
        print(f"PDF loaded: {len(pages)} pages")
    except Exception as error:
        raise RuntimeError(f"Could not load PDF: {error}") from error

    source_hash = hashlib.sha256(PDF_PATH.read_bytes()).hexdigest()
    marker_path = PERSIST_DIRECTORY / ".source_sha256"
    has_current_index = (
        marker_path.is_file() and marker_path.read_text().strip() == source_hash
    )

    try:
        if has_current_index:
            vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(PERSIST_DIRECTORY),
            )
            print("Using existing Chroma vector store")
        else:
            # The index belongs exclusively to this script. Rebuild it when the
            # PDF changes instead of appending duplicate chunks on each launch.
            shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
            PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            pages_split = text_splitter.split_documents(pages)
            vectorstore = Chroma.from_documents(
                documents=pages_split,
                embedding=embeddings,
                persist_directory=str(PERSIST_DIRECTORY),
                collection_name=COLLECTION_NAME,
            )
            marker_path.write_text(source_hash + "\n")
            print("Chroma vector store created")
    except Exception as error:
        raise RuntimeError(f"Could not create Chroma vector store: {error}") from error

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    @tool
    def retriever_tool(query: str) -> str:
        """Search the loaded PDF and return relevant passages."""
        documents = retriever.invoke(query)
        if not documents:
            return "No relevant information was found in the PDF."
        return "\n\n".join(
            f"Document {index}:\n{document.page_content}"
            for index, document in enumerate(documents, start=1)
        )

    tools = [retriever_tool]
    tool_by_name = {rag_tool.name: rag_tool for rag_tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def should_continue(state: AgentState) -> bool:
        last_message = state["messages"][-1]
        return bool(getattr(last_message, "tool_calls", None))

    system_prompt = (
        "You are a helpful AI assistant. Answer questions using the PDF when "
        "relevant, and clearly say when the PDF does not contain the answer."
    )

    def call_llm(state: AgentState) -> dict[str, list[BaseMessage]]:
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def take_action(state: AgentState) -> dict[str, list[ToolMessage]]:
        tool_calls = state["messages"][-1].tool_calls
        results: list[ToolMessage] = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            query = tool_call["args"].get("query", "")
            print(f"Calling {tool_name} with query: {query or 'No query provided'}")
            if tool_name not in tool_by_name:
                result = f"Unknown tool: {tool_name}"
            else:
                result = tool_by_name[tool_name].invoke(tool_call["args"])
            results.append(
                ToolMessage(
                    tool_call_id=tool_call["id"], name=tool_name, content=str(result)
                )
            )
        return {"messages": results}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges(
        "llm", should_continue, {True: "retriever_agent", False: END}
    )
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")
    return graph.compile()


def run_agent() -> None:
    rag_agent = build_rag_agent()
    print("RAG assistant is ready. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        result = rag_agent.invoke({"messages": [HumanMessage(content=user_input)]})
        print(f"Assistant: {result['messages'][-1].content}")


if __name__ == "__main__":
    run_agent()
