from typing import Annotated, List, TypedDict,Sequence
from dotenv import load_dotenv 
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage,ToolMessage,SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import Tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode 
import os 




load_dotenv()


''''email=Annotated[str, "The email address of the user"] 
print("Email:", email)
###print(email.__metadata__) 


#without a reducer 

state={"messages": ["Hi!"]}
update= {"messages":["Nice to meet you!"]}
new_state={"messages": ["Nice to meet you!"]}




#with a reducer 
state={"messages": ["Hi!"]}
update= {"messages":["Nice to meet you!"]}
new_state={"messages": ["Hi!", "Nice to meet you!"]} '''

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The list of messages exchanged between the agent and the user"]
    email: Annotated[str, "The email address of the user"]
    
@tool 
def add(a:int,b:int)->int:
    """Add two numbers together."""
    return a+b

tools=[add]
model=ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AgentBot",
    }
).bind_tools(tools)


def model_call(state:AgentState)->AgentState: 
    system_prompt=SystemMessage(content="You are my helpful assistant. You will answer the user's questions and provide assistance as needed.")
    response=model([system_prompt]+state["messages"]) 
    return {"messages": state["messages"]+[response], "email": state["email"]} 


def should_continue(state:AgentState)->bool:
    last_message=state['messages'][-1]
    if not last_message.tool_calls:
        return "end"
    else : 
        return "continue" 

graph=StateGraph(AgentState) 
graph.add_node(START, model_call)
graph.add_node(END, lambda state: state)
graph.add_edge(START, model_call, should_continue)



tool_node=ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_entry_point("our_agent")

graph.add_clonditional_edges("our_agent",should_continue, {"continue": "tools", "end": END},)
graph.add_edge("tools","our_agent")



app=graph.compile()


def print_stream(stream): 
    for s in stream: 
        print(s)
        message=s["message"][-1]
        print(f"{message.type}: {message.content}") 
        if isinstance(message,tuple): 
            print(f"Tool Calls: {message.tool_calls}")
        else: 
            print(f"Tool Calls: {message.tool_calls}")
        
        












