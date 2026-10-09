"""CLI Tool for Grounded Regulatory Policy Q&A.

Executes the full RAG pipeline:
    Query -> Hybrid Retrieval -> Context Budgeting -> Grounded Synthesis -> Citation Verification

Usage:
    python scripts/ask.py "What are the rules for Default Loss Guarantee under RBI?"
    python scripts/ask.py "What is the settlement cutoff for payment processing?"
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import settings
from src.embeddings.factory import get_embedding_provider
from src.generation.generator import GroundedGenerator
from src.retrieval.engine import RetrievalEngine
from src.vectorstore.factory import get_vector_store

console = Console()


def ask_question(query: str, top_k: int = 4, show_chunks: bool = False) -> None:
    console.print(Panel(f"[bold cyan]QUESTION:[/bold cyan] {query}", title="BFSI Research Assistant", border_style="blue"))

    # 1. Initialize Retrieval Engine
    embedder = get_embedding_provider()
    vstore = get_vector_store(dimension=embedder.dimension)
    engine = RetrievalEngine(vector_store=vstore, embedding_provider=embedder)

    # 2. Retrieve Relevant Chunks via Hybrid RRF
    with console.status("[bold green]Retrieving relevant regulatory circulars (Hybrid RRF)...[/bold green]"):
        retrieved_hits = engine.search(query=query, mode="hybrid", top_k=top_k)

    if not retrieved_hits:
        console.print("[yellow]No relevant chunks retrieved from vector database.[/yellow]")
        return

    # 3. Grounded Synthesis & Verification
    generator = GroundedGenerator()
    with console.status("[bold green]Synthesizing grounded answer and verifying citations...[/bold green]"):
        grounded_result = generator.generate_answer(query=query, retrieved_results=retrieved_hits)

    # 4. Render Answer Panel
    border_color = "green" if grounded_result.has_sufficient_context else "yellow"
    console.print(
        Panel(
            grounded_result.answer,
            title="[bold green]Grounded Answer[/bold green]",
            subtitle=f"[dim]Latency: {grounded_result.latency_ms:.0f}ms | Model: {grounded_result.model_used}[/dim]",
            border_style=border_color,
        )
    )

    # 5. Render Citations Table
    if grounded_result.citations:
        c_table = Table(title="Statutory Citations & Verification Status", show_header=True, header_style="bold magenta")
        c_table.add_column("Ref / Circular", style="green", width=20)
        c_table.add_column("Document Title", style="cyan", width=30)
        c_table.add_column("Location", style="yellow", width=16)
        c_table.add_column("Status", justify="center", width=12)
        c_table.add_column("Supporting Excerpt (Quote)", style="white")

        for c in grounded_result.citations:
            status_badge = "[bold green][VERIFIED][/bold green]" if c.verified else "[bold red][UNVERIFIED][/bold red]"
            loc = f"Pg {c.page_number} | {c.clause_number or c.section_heading}"
            c_table.add_row(
                c.circular_number or "Internal Doc",
                c.document_title[:28] + ("..." if len(c.document_title) > 28 else ""),
                loc,
                status_badge,
                c.quote,
            )
        console.print(c_table)

    console.print(f"[dim]Confidence Note: {grounded_result.confidence_note}[/dim]\n")

    # 6. Optional: Show Retrieved Chunks Inspector
    if show_chunks:
        inspect_table = Table(title="Audit Inspector: Retrieved Chunks Fed to LLM Context", header_style="bold blue")
        inspect_table.add_column("Rank", justify="center", width=6)
        inspect_table.add_column("Source", width=8)
        inspect_table.add_column("Score", justify="right", width=8)
        inspect_table.add_column("Snippet", style="dim")

        for r in retrieved_hits:
            inspect_table.add_row(
                str(r.rank),
                r.source,
                f"{r.score:.4f}",
                r.chunk.content.replace("\n", " ")[:120] + "...",
            )
        console.print(inspect_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query BFSI Policy Research Assistant")
    parser.add_argument("query", type=str, help="Compliance or regulatory question")
    parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved chunks")
    parser.add_argument("--show-chunks", action="store_true", help="Display retrieved chunk audit table")
    args = parser.parse_args()

    ask_question(query=args.query, top_k=args.top_k, show_chunks=args.show_chunks)
