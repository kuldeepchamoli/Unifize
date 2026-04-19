#ReAct Agent reasoning and acting agent

#Learn how to craete Tools in LangGraph
#how to crearte a ReAct Graph + work with different types of messages such as tool messages
# test out robustness of our graph

#GOAL: create a robust ReAct agent that can reason and act using tools in LangGraph. This agent will be able to handle different types of messages, including tool messages, and maintain a conversation history to provide context for its interactions.

from typing import Annotated , Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage #foundational class for all messages tyypes in LangGraph
from langchain_core.messages import ToolMessage #Passes data back to LLM after it calls a tool such as the content and the  
from langchain_core.messages import SystemMessage #message for providing instruction to LLM
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

#reducer function
#rule that control how updates from nodes are combioned with existing state
# tells us how to merge new data into current state

#without a reducer updates would have replaced exiosting value entirely

#without a reducer
# state = {"messages": ["Hi!"]}
# update={"messages":["Nice to meet you!"]}
# new_state={"messages": ["Nice to meet you!"]}

# #with a reducer
# state={"messages": ["Hi!"]}
# update={"messages": ["Nice to meet you!"]}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int , b: int):
    """This is an addition function that adds 2 numbers together"""

    return a+b

@tool
def subtract(a: int , b: int):
    """This is an subtract function that subtracts 2 numbers together"""

    return a-b

@tool
def multiply(a: int , b: int):
    """This is an multiplication function that multiplies 2 numbers together"""

    return a*b

tools=[add, subtract, multiply]

# Uses GROQ_API_KEY from .env (loaded above). temperature=0 for reliable tool calls.
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
                                  "You are my AI assistant, please answer my query to best of your ability")
    response=model.invoke([system_prompt]+ state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages=state["messages"]
    last_message=messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph= StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node=ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    },
)
graph.add_edge("tools","our_agent")

app=graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


def main():
    inputs = {"messages": [("user", "Add 3 + 4.")]}
    print_stream(app.stream(inputs, stream_mode="values"))


# Don't run the demo query on import -- only when executed directly.
if __name__ == "__main__":
    main()
