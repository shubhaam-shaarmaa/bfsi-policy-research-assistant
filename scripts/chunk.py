"""CLI script to chunk processed regulatory documents into structured chunks.

Usage:
    python scripts/chunk.py [--strategy structure_aware|recursive|fixed] [--chunk-size 500]
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
from src.chunking.pipeline import ChunkingPipeline
from src.core.logging import setup_logger

logger = setup_logger("cli.chunk")
console = Console()


def run_chunking_cli(strategy: str | None = None, chunk_size: int | None = None) -> None:
    strat = strategy or settings.CHUNKING_STRATEGY
    c_size = chunk_size or settings.CHUNK_SIZE

    console.print(
        f"[bold blue]BFSI Document Chunker[/bold blue] | Strategy: [cyan]{strat}[/cyan] | Target Size: [cyan]{c_size}[/cyan] tokens"
    )

    pipeline = ChunkingPipeline(strategy=strat, chunk_size=c_size)  # type: ignore
    results = pipeline.chunk_all_processed()

    if not results:
        console.print("[yellow]No processed documents found in data/processed/. Run `python scripts/ingest.py` first.[/yellow]")
        return

    table = Table(title="Document Chunking Summary", show_header=True, header_style="bold magenta")
    table.add_column("Doc ID", style="dim", width=24)
    table.add_column("Chunks Generated", justify="right", style="green")

    total_chunks = sum(results.values())
    for doc_id, count in results.items():
        table.add_row(doc_id, str(count))

    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{total_chunks}[/bold]")

    console.print(table)
    console.print(f"[green][OK] Saved chunks to: {settings.DATA_CHUNKS_DIR}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk processed regulatory documents")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "structure_aware"],
        default=None,
        help="Chunking strategy to apply",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Target maximum token count per chunk",
    )
    args = parser.parse_args()
    run_chunking_cli(strategy=args.strategy, chunk_size=args.chunk_size)
