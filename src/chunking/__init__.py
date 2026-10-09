"""Chunking module exposing strategies and pipeline."""

from src.chunking.chunker_base import BaseChunker
from src.chunking.fixed_chunker import FixedSizeChunker
from src.chunking.pipeline import ChunkingPipeline, get_chunker
from src.chunking.recursive_chunker import RecursiveCharacterChunker
from src.chunking.structure_chunker import StructureAwareChunker

__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveCharacterChunker",
    "StructureAwareChunker",
    "ChunkingPipeline",
    "get_chunker",
]
