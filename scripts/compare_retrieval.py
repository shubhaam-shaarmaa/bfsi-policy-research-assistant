"""CLI Tool to Compare Retrieval Modes Side-by-Side.

Executes Dense Vector, BM25 Keyword, and Hybrid (RRF + Reranker) search for any query
and presents comparative analysis showing ranks, scores, sources, and citations.

Usage:
    python scripts/compare_retrieval.py --query "What is the portfolio cap on DLG?"
    python scripts/compare_retrieval.py --query "RBI/2024-25/SYNTH-88" --rerank
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from config.settings import settings
from src.embeddings.factory import get_embedding_provider
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.engine import RetrievalEngine
from src.retrieval.mock_reranker import MockReranker
from src.vectorstore.base import SearchResult
from src.vectorstore.factory import get_vector_store

console = Console()


def render_results_table(title: str, results: list[SearchResult]) -> None:
    """Render Rich table for a specific retrieval mode."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Rank", justify="center", style="dim", width=6)
    table.add_column("Score", justify="right", style="green", width=10)
    table.add_column("Source", style="magenta", width=10)
    table.add_column("Citation / Document", style="yellow", width=32)
    table.add_column("Content Snippet", style="white")

    if not results:
        table.add_row("-", "0.0000", "-", "No matches found", "")
    else:
        for r in results:
            citation = f"{r.chunk.document_title}\n[dim]Pg {r.chunk.page_number} | Clause: {r.chunk.clause_number or 'N/A'}[/dim]"
            snippet = r.chunk.content.replace("\n", " ")[:140] + "..."
            table.add_row(str(r.rank), f"{r.score:.4f}", r.source, citation, snippet)

    console.print(table)
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval modes side by side")
    parser.add_argument("--query", type=str, default="What are the requirements for credit underwriting governance?", help="Search query")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results per mode")
    parser.add_argument("--rerank", action="store_true", help="Apply Cross-Encoder reranker to hybrid results")
    args = parser.parse_args()

    console.print(f"\n[bold yellow]Query:[/bold yellow] [bold white]'{args.query}'[/bold white]")
    console.print("[dim]Comparing: 1) Dense Vector Search  2) BM25 Keyword Search  3) Hybrid RRF Search[/dim]\n")

    # Initialize components
    embedder = get_embedding_provider()
    vstore = get_vector_store(dimension=embedder.dimension)
    reranker = MockReranker() if args.rerank else None

    engine = RetrievalEngine(
        vector_store=vstore,
        embedding_provider=embedder,
        reranker=reranker,
    )

    # 1. Pure Dense
    dense_hits = engine.search(query=args.query, mode="dense", top_k=args.top_k)
    render_results_table("1. Dense Vector Search (Cosine Similarity)", dense_hits)

    # 2. Pure BM25
    bm25_hits = engine.search(query=args.query, mode="bm25", top_k=args.top_k)
    render_results_table("2. BM25 Keyword Search (Exact Lexical Match)", bm25_hits)

    # 3. Hybrid RRF
    hybrid_hits = engine.search(
        query=args.query,
        mode="hybrid",
        use_reranker=args.rerank,
        top_k=args.top_k,
    )
    title = "3. Hybrid Search (RRF + Reranker)" if args.rerank else "3. Hybrid Search (Reciprocal Rank Fusion)"
    render_results_table(title, hybrid_hits)


if __name__ == "__main__":
    main()
