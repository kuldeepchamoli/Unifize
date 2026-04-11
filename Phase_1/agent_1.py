
#"develop simple bot=> define state structure with list of HumanMessage objects+ initialize a Llama model via Groq (ChatGroq) +sending and handle diff types of messgaes + building and compiling graph of the agent"


#"GOAL: integrate LLms in the graph"


from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # used to store secret stuff like API keys in a .env file"

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

# Uses GROQ_API_KEY from .env automatically via load_dotenv() above.
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)


def process(state: AgentState) -> AgentState:
    response=llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph= StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent=graph.compile()

def main():
    user_input = input("Enter :")
    agent.invoke({"messages": [HumanMessage(content=user_input)]})


# Only run the interactive loop when this file is executed directly
# (`python agent_1.py`), NOT when it is imported from another module.
# Without this guard, `import agent_1` would pop a prompt and block forever.
if __name__ == "__main__":
    main()



