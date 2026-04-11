# agent.py

from dotenv import load_dotenv
import os
from pathlib import Path
from typing import TypedDict, Annotated, Sequence

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# =========================
# 🔧 CONFIG (SAFE)
# =========================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

PERSIST_DIRECTORY = str(BASE_DIR / "chroma_db")
PDF_PATH = str(BASE_DIR / "Stock_Market_Performance_2024.pdf")


# =========================
# 🧠 STATE
# =========================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


SYSTEM_PROMPT = """
You are an AI assistant for Stock Market 2024.
Always use the retriever tool and cite sources.
"""


# =========================
# 🚀 MAIN BUILDER (IMPORTANT)
# =========================

def build_rag_agent():
    """
    ✅ ALL heavy operations happen here
    ✅ Nothing runs at import time
    """

    # -------------------------
    # 1. LLM
    # -------------------------
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    # -------------------------
    # 2. Load PDF
    # -------------------------
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    # -------------------------
    # 3. Split
    # -------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = splitter.split_documents(pages)

    # -------------------------
    # 4. Embeddings
    # -------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------------------------
    # 5. Vector DB
    # -------------------------
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name="stock_market",
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    # -------------------------
    # 6. Tool
    # -------------------------
    @tool
    def retriever_tool(query: str) -> str:
        """
        Search and return relevant information from the Stock Market Performance 2024 document.
        """
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant info found"

        return "\n\n".join(
            f"[Doc {i+1}] {d.page_content}"
            for i, d in enumerate(docs)
        )

    tools = [retriever_tool]
    tools_dict = {t.name: t for t in tools}

    llm = llm.bind_tools(tools)

    # -------------------------
    # 7. Graph Logic
    # -------------------------

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return hasattr(last, "tool_calls") and len(last.tool_calls) > 0

    def call_llm(state: AgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        response = llm.invoke(messages)
        return {"messages": [response]}

    def take_action(state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for call in tool_calls:
            name = call["name"]
            query = call["args"].get("query", "")

            if name not in tools_dict:
                result = "Invalid tool"
            else:
                result = tools_dict[name].invoke(query)

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=name,
                    content=str(result),
                )
            )

        return {"messages": results}

    # -------------------------
    # 8. Graph
    # -------------------------
    graph = StateGraph(AgentState)

    graph.add_node("llm", call_llm)
    graph.add_node("tools", take_action)

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {True: "tools", False: END},
    )

    graph.add_edge("tools", "llm")
    graph.set_entry_point("llm")

    return graph.compile()


# =========================
# 🧪 RUNNER (SAFE)
# =========================

def main():
    agent = build_rag_agent()

    while True:
        q = input("\nAsk: ").strip()
        if q in ["exit", "quit", ""]:
            break

        result = agent.invoke({
            "messages": [HumanMessage(content=q)]
        })

        print("\nAnswer:")
        print(result["messages"][-1].content)


# =========================
# 🔒 ENTRY GUARD
# =========================

if __name__ == "__main__":
    main()