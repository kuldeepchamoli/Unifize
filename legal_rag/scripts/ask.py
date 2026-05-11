"""Tiny CLI for the RAG pipeline.

Usage:
    python -m scripts.ask "What is the role of UIDAI under the Aadhaar Act?"
    python -m scripts.ask --k 8 "Penalties for impersonation under the Aadhaar Act?"
"""
import argparse

from src.rag import ask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+", help="Your question (in quotes).")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    question = " ".join(args.question)
    result = ask(question, k=args.k)

    print("\n=== Answer ===")
    print(result.answer)

    print("\n=== Sources ===")
    for i, h in enumerate(result.hits, 1):
        print(f"[{i}] dist={h.distance:.4f}  {h.title}, chunk #{h.chunk_index}")


if __name__ == "__main__":
    main()
