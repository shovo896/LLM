import os 
from typing import List, TypedDict,Union 
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage 
from langchain_openai import ChatOpenAI 
from langgraph.graph import END, START, StateGraph 
from dotenv import load_dotenv 
load_dotenv()





class AgentState(TypedDict):
    #messages: List[HumanMessage]
    #messages_ai:List[Union[AIMessage]] 
    messages: Union[HumanMessage, AIMessage]
    
    
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
    """this node will solve request you input and return the response from the LLM"""
    
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAgent: {response.content}") 
    
    return state 

with open("logging.txt", "a") as log_file:
    log_file.write("Your conversation logged here:\n") 
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            log_file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            log_file.write(f"Agent: {message.content}\n")
    file.write("\n")  # Add a newline after each conversation for separation    
    
print("Conversation logged in logging.txt")


graph = StateGraph(AgentState) 
graph.add_node("agent", process)
graph.add_edge(START, 'agent')
graph.add_edge('agent', END)
agent=graph.compile() 


conversation_history=[]
user_input = input("You: ") 


while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    state = {"messages": conversation_history}
    result = agent.invoke(state)
    conversation_history = result["messages"]
    user_input = input("You: ")





    
    


