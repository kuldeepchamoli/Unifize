# use different messgaes types=> Human Message and AI message
# maintain a full converstaion history using both msg types
# use a Llama model via Groq (ChatGroq)
# Create a sophisticated converstaion loop

#Goal: create a form of memory for our agent
import os
from typing import TypedDict, List, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    # `add_messages` is a reducer -- a rule that tells LangGraph "when a node
    # returns new messages, APPEND them to the existing list instead of
    # replacing it." Lets nodes return just the new message and LangGraph
    # handles the merging.
    messages: Annotated[List[Union[HumanMessage, AIMessage]], add_messages]

# Uses GROQ_API_KEY from .env automatically via load_dotenv() above.
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

def process(state: AgentState) -> dict:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")

    # Return ONLY the new message (not the whole list). The `add_messages`
    # reducer on AgentState will append it to the existing history for us.
    # This is the LangGraph-idiomatic pattern: nodes return partial updates,
    # not mutated state.
    return {"messages": [AIMessage(content=response.content)]}

graph=StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent=graph.compile()

# Sentinels that mean "quit the chat". An empty line also quits so the user
# can just hit Enter on a blank prompt to exit cleanly.
EXIT_COMMANDS = {"exit", "quit", "q", ":q", ""}


def get_user_input() -> str | None:
    """
    Prompt the user for input.

    Returns:
        The user's message as a string, OR
        None if the user wants to quit (typed an exit command, pressed
        Ctrl+D which raises EOFError, or pressed Ctrl+C which raises
        KeyboardInterrupt).
    """
    try:
        text = input("Enter: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if text.lower() in EXIT_COMMANDS:
        return None

    return text


def main():
    conversation_history = []

    while True:
        user_input = get_user_input()
        if user_input is None:
            break

        conversation_history.append(HumanMessage(content=user_input))

        result = agent.invoke({"messages": conversation_history})

        print(result["messages"])
        conversation_history = result["messages"]

    # This block runs no matter how the loop exited (exit command, Ctrl+D,
    # Ctrl+C). The old code crashed on Ctrl+D and lost the log entirely.
    with open("logging.txt", "w") as file:
        file.write("Conversation Log:\n")
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                file.write(f"You: {message.content}\n")
            elif isinstance(message, AIMessage):
                file.write(f"AI: {message.content}\n\n")
        file.write("End of the converstaion")

    print("Conversation log saved to logging.txt")


# Don't run the chat loop on import -- only when executed directly.
if __name__ == "__main__":
    main()