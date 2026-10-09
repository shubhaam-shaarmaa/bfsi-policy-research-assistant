"""CLI Script to embed and index chunks into Vector Database.

Usage:
    python scripts/index_vectors.py [--provider chromadb|qdrant] [--query "What is the cap on DLG?"]
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
from src.vectorstore.factory import get_vector_store
from src.vectorstore.pipeline import VectorIndexingPipeline

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorize and index regulatory chunks")
    parser.add_argument("--vector-db", choices=["qdrant", "chromadb"], default=None, help="Vector store")
    parser.add_argument("--embedding", choices=["local", "mock", "openai", "voyage"], default=None, help="Embedding provider")
    parser.add_argument("--force", action="store_true", help="Force re-indexing ignoring manifest")
    parser.add_argument("--query", type=str, default=None, help="Optional search test query")
    args = parser.parse_args()

    v_db = args.vector_db or settings.VECTOR_DB_PROVIDER
    e_prov = args.embedding or settings.EMBEDDING_PROVIDER

    console.print(
        f"[bold blue]BFSI Vector Indexer[/bold blue] | Vector DB: [cyan]{v_db}[/cyan] | Embedding: [cyan]{e_prov}[/cyan]"
    )

    embedder = get_embedding_provider(provider_type=e_prov)
    vstore = get_vector_store(provider=v_db, dimension=embedder.dimension)
    pipeline = VectorIndexingPipeline(vector_store=vstore, embedding_provider=embedder)

    indexed = pipeline.index_all_chunk_files()
    total_in_db = vstore.count()

    console.print(f"[green][OK] Indexing completed: {indexed} new chunk(s) stored. Total in DB: {total_in_db}[/green]")

    if args.query:
        console.print(f"\n[bold yellow]Testing Query:[/bold yellow] '{args.query}'")
        results = pipeline.search(args.query, top_k=3)
        table = Table(title="Top Dense Retrieval Hits")
        table.add_column("Rank", justify="center", style="dim")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Doc / Section", style="cyan")
        table.add_column("Content Snippet", style="white")

        for r in results:
            table.add_row(
                str(r.rank),
                f"{r.score:.4f}",
                f"{r.chunk.document_title}\nClause: {r.chunk.clause_number or 'N/A'}",
                r.chunk.content[:150] + "...",
            )
        console.print(table)


if __name__ == "__main__":
    main()
