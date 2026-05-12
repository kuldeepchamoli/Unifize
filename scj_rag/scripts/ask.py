import argparse
from src.rag import answer

ap = argparse.ArgumentParser()
ap.add_argument("question", help="Your question, in quotes")
ap.add_argument("--k", type=int, default=5)
args = ap.parse_args()

print(answer(args.question, k=args.k))
