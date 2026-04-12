"""RAG agent for answering questions about Stock Market Performance 2024."""

import logging
from pathlib import Path
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

logger = logging.getLogger(__name__)

# -- Config --
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "Stock_Market_Performance_2024.pdf"
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"
COLLECTION_NAME = "stock_market"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 5

SYSTEM_PROMPT = (
    "You are an AI assistant for Stock Market 2024. "
    "Always use the retriever tool and cite sources."
)


# -- State --
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# -- Builder --
def build_rag_agent():
    """Build and return a compiled RAG agent graph.

    All heavy operations (PDF loading, embedding, vector store creation)
    happen here — nothing runs at import time.
    """
    # LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # Load PDF
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pages = PyPDFLoader(str(PDF_PATH)).load()
    logger.info("Loaded PDF with %d pages", len(pages))

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = splitter.split_documents(pages)
    logger.info("Split into %d chunks", len(docs))

    # Embeddings + Vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIRECTORY),
        collection_name=COLLECTION_NAME,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    # Tool
    @tool
    def retriever_tool(query: str) -> str:
        """Search and return relevant information from the Stock Market Performance 2024 document."""
        results = retriever.invoke(query)
        if not results:
            return "No relevant information found in the document."
        return "\n\n".join(
            f"[Doc {i + 1}]\n{doc.page_content}" for i, doc in enumerate(results)
        )

    tools = [retriever_tool]
    tools_dict = {t.name: t for t in tools}
    bound_llm = llm.bind_tools(tools)

    # Graph nodes
    def should_continue(state: AgentState) -> bool:
        last = state["messages"][-1]
        return hasattr(last, "tool_calls") and len(last.tool_calls) > 0

    def call_llm(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        response = bound_llm.invoke(messages)
        return {"messages": [response]}

    def take_action(state: AgentState) -> dict:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for tc in tool_calls:
            name = tc["name"]
            query = tc["args"].get("query", "")
            logger.info("Calling tool %s with query: %s", name, query)

            if name not in tools_dict:
                content = f"Unknown tool '{name}'. Available: {list(tools_dict)}"
            else:
                content = str(tools_dict[name].invoke(query))

            results.append(
                ToolMessage(tool_call_id=tc["id"], name=name, content=content)
            )
        return {"messages": results}

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", take_action)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {True: "tools", False: END},
    )
    graph.add_edge("tools", "llm")

    return graph.compile()


# -- Entry point --
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    agent = build_rag_agent()

    while True:
        try:
            question = input("\nAsk (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("exit", "quit"):
            break

        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        print("\nAnswer:")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()