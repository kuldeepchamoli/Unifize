# project name is called drafter 

# our company noit working efficiently way too much time drfating documents and this needs to be fixed

#create an ai agent sysytem that can speed up drafting docs, emails etc. AI agentic system 
# shud have Human AI collab meaning humna. shud be able to provide continuous feedback and AI agent shud stop 
# when human is happy with the draft. The system shud be fast and be able to save the drafts

from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage ,SystemMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.types import Command

load_dotenv()

# NOTE: document content used to live in a module-level global `document_content`.
# That was a bug: a global means every conversation shares the same document,
# which breaks multi-user usage, testing, checkpointing, and parallel runs.
# Document content now lives IN the graph state (see AgentState.document below)
# so each conversation has its own isolated document.

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document: str   # the current document content, per-conversation

@tool
def update(
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Updates the doc with the provided content"""
    # `Command` lets a tool return BOTH a tool message (for the LLM) AND a
    # state update (for the graph). Here we update the `document` key so
    # the next turn of the agent sees the new content via state, not a global.
    return Command(update={
        "document": content,
        "messages": [
            ToolMessage(
                content=f"Document has been updated successfully! The current content is: \n{content}",
                tool_call_id=tool_call_id,
            )
        ],
    })

@tool
def save(
    filename: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """save the current document to text file and finish the process
    Args:
        filename: Name for the text file.
    """
    # `InjectedState` asks LangGraph to pass the current graph state into
    # this tool at call time. The LLM does not see this parameter -- it is
    # filled in by the framework. That is how we read `document` without a
    # global variable.
    document_content = state.get("document", "")

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n Document has been saved to: {filename}")
        message = f"Document has been saved successfully to '{filename}'"
    except Exception as e:
        message = f"Error saving document: {str(e)}"

    return Command(update={
        "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
    })
    
tools=[update, save]

model = ChatGroq(model="openai/gpt-oss-20b", temperature=0).bind_tools(tools)

def our_agent(state: AgentState) -> dict:
    """
    Pure function of state: build the prompt, call the LLM, return the
    LLM's reply as a state update. NO I/O (no input(), no file reads).

    The user's message has already been appended to state["messages"] by
    the outer run loop BEFORE the graph is invoked. That separation is
    what makes this node testable, streamable, and reusable outside of
    a CLI context.
    """
    document_content = state.get("document", "")

    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You help the user create,
    update, and save text documents through an interactive conversation.

    You have exactly two tools available:
      - `update(content: str)`: replaces the entire document with the given
        content. Always pass the COMPLETE new document text, not a diff.
      - `save(filename: str)`: saves the current document to a .txt file
        and ends the session. Only call this when the user says they are
        finished.

    Rules you MUST follow:
      1. On every turn you MUST call exactly one tool (`update` or `save`).
         Do not reply with plain text.
      2. If the user asks to change/add/write content, call `update` with
         the full new document.
      3. If the user says they are done / want to save / want to exit,
         call `save` with a sensible filename.
      4. After an `update` call, briefly describe what you changed so the
         user knows the current state of the document.

    The current document content is:
    ---
    {document_content}
    ---
    """)

    all_messages = [system_prompt] + list(state["messages"])
    response = model.invoke(all_messages)

    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"Using Tools: {[tc['name'] for tc in response.tool_calls]}")

    # Only return the new message. The add_messages reducer appends it.
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """
    Router used on BOTH outgoing edges (agent -> ? and tools -> ?).

    The last message in state tells us which edge called us:
      - If it's an AIMessage, we were called from the `agent` edge.
      - If it's a ToolMessage, we were called from the `tools` edge.

    Return "continue" to take the main path, "end" to terminate.
    The actual destination of "continue" vs "end" is defined per-edge
    in the graph wiring below.
    """
    messages = state["messages"]

    if not messages:
        return "continue"

    last = messages[-1]

    # --- Called from the `tools` edge ---
    # A ToolMessage means a tool just ran. If it was a successful `save`,
    # we're done drafting; otherwise loop back to the agent.
    if isinstance(last, ToolMessage):
        if ("saved" in last.content.lower() and
            "document" in last.content.lower()):
            return "end"        # save succeeded -> END
        return "continue"       # other tool -> back to agent

    # --- Called from the `agent` edge ---
    # An AIMessage with tool_calls means the LLM wants to run a tool.
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "continue"       # -> tools

    # AIMessage with NO tool_calls: the LLM replied in plain text instead
    # of calling `update` or `save`. In this app the prompt expects a tool
    # call every turn, so a plain reply means the LLM got confused. Stop
    # rather than forwarding to `tools` (which would have nothing to run).
    return "end"

def print_messages(messages):
    """Function I made to print the messages in more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n Tool Result: {message.content}")

graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

# Conditional edge out of `agent`:
# The LLM's job is to decide whether to call a tool. We must respect that
# decision -- if there are no tool_calls on the agent's reply, going to
# `tools` anyway would be wrong (ToolNode would have nothing to execute).
# So we route conditionally based on the agent's latest output.
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",  # LLM asked for a tool -> run it
        "end": END,            # LLM replied in plain text -> stop
    },
)

# Edge out of `tools`: always END.
# One graph run == one conversational turn. After the tool runs, control
# returns to the outer run loop, which collects the next user input and
# invokes the graph again. This keeps the graph itself a pure function
# of state -- the I/O (input()) lives outside.
graph.add_edge("tools", END)

app = graph.compile()

EXIT_COMMANDS = {"exit", "quit", "q", ":q", ""}


def run_document_agent():
    """
    Outer run loop. All I/O lives here -- the graph itself is pure.

    Flow per turn:
      1. Prompt the human for input (outside the graph).
      2. Append HumanMessage to state.
      3. Invoke the graph for ONE turn (agent -> tools -> END).
      4. Inspect the result for a successful save; if found, exit.
    """
    print("\n === DRAFTER ===")
    print("\n AI: I'm ready to help you update a document. What would you like to create?")

    state = {"messages": [], "document": ""}

    while True:
        try:
            user_input = input("\nWhat would you like to do with the document? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in EXIT_COMMANDS:
            break

        print(f"\n USER: {user_input}")

        # Attach the new user message to state BEFORE invoking the graph.
        # This is the clean boundary: stdin -> state -> graph.
        state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]

        # One graph run == one turn. Returns updated state.
        state = app.invoke(state)

        print_messages(state["messages"])

        # If the last tool message is a successful save, we are done.
        last = state["messages"][-1]
        if (isinstance(last, ToolMessage) and
            "saved" in last.content.lower() and
            "document" in last.content.lower()):
            break

    print("\n === DRAFTER FINISHED===")

if __name__ == "__main__":
    run_document_agent()
