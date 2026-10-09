"""CLI Script for Document Ingestion.

Usage:
    python scripts/ingest.py --input-dir data/raw
    python scripts/ingest.py --input-dir data/synthetic
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
from src.ingestion.pipeline import IngestionPipeline

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest regulatory documents into processed format")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=settings.DATA_RAW_DIR,
        help="Input directory containing documents (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.DATA_PROCESSED_DIR,
        help="Target output directory for parsed JSON (default: data/processed)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion of already processed documents (disable hash check)",
    )

    args = parser.parse_args()

    console.print(f"[bold cyan]BFSI Policy Ingestion Engine[/bold cyan]")
    console.print(f"Reading from: [yellow]{args.input_dir}[/yellow]")
    console.print(f"Output to:    [green]{args.output_dir}[/green]\n")

    pipeline = IngestionPipeline()
    try:
        documents = pipeline.ingest_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            skip_existing=not args.force,
        )
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    if not documents:
        console.print("[yellow]No documents were ingested. Ensure the input directory has .pdf, .docx, or .txt files.[/yellow]")
        return

    # Render summary table
    table = Table(title="Ingested Policy Documents Summary")
    table.add_column("Doc ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Circular No", style="green")
    table.add_column("Date", style="magenta")
    table.add_column("Pages", justify="right", style="blue")
    table.add_column("Sections", justify="right", style="gold1")
    table.add_column("Format", style="dim")

    for doc in documents:
        table.add_row(
            doc.metadata.doc_id,
            doc.metadata.title[:45] + ("..." if len(doc.metadata.title) > 45 else ""),
            doc.metadata.circular_number or "N/A",
            doc.metadata.date or "N/A",
            str(doc.page_count),
            str(len(doc.sections)),
            doc.metadata.file_format,
        )

    console.print(table)
    console.print(f"\n[bold green]Success![/bold green] Ingested {len(documents)} document(s) successfully.")


if __name__ == "__main__":
    main()
