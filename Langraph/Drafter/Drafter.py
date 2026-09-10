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

model = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "AgentBot",
    }
).bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    document_content = state.get("document_content", "")

    system_prompt = SystemMessage(
        content=f"""You are a drafter. You help the user update and modify documents.

- If the user wants to update or modify content, use the update tool with the complete updated content.
- If the user wants to finish or terminate, finish or terminate.
- Always allow modifications.
- The current document is:

{document_content}
"""
    )

   
    return state




    if not state["messages"]: 
           user_input = " I am ready to help you update a documennt. what would you like to create! "

           user_input = HumanMessage(content=user_input)

    else : 
                 user_input = input("\n what would you like to do with the document ??")

                 print(f"\n User : {user_input}")
                 user_input = HumanMessage(content=user_input)
    all_messages= [system_prompt]+list(state["messages"])+[user_input]



    response=model.invoke(all_messages)



    print(f"\n Agent : {response.content}")
    if hasattr(response,"tool_calls") and response.tool_calls :
           print(f"using tools : {[tc["tool_name"] for tc in response.tool_calls]}")

    return {"messages": state["messages"]+[user_input,response],"document_content":document_content}



    ## conditional edge  function 

    def should_continue(state: AgentState) -> str :
           """ determine if we should continue or end of the converstation based on the last message in the state. If the last message is a tool call, we continue, otherwise we end. """

           messages = state['messages'] 
           if not messages: 
                  return "continue"
           for message in reversed(messages): 
                  if (isinstance(message,ToolMessage) and "saved" in message.content.lower()) or (isinstance(message,AIMessage) and "finish" in message.content.lower()): 
              
                         return "end"


           return "continue"



           
           














    


