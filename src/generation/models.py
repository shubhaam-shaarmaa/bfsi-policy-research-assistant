"""Data Models for Grounded Generation, Citations, and Hallucination Verification."""

from typing import Any
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A verifiable citation back to a specific regulatory passage."""

    document_title: str = Field(..., description="Title of cited document")
    circular_number: str | None = Field(None, description="Official circular reference (e.g., RBI/2023-24/41)")
    page_number: int = Field(..., ge=1, description="Page number where quote appears")
    section_heading: str = Field(..., description="Section or chapter heading")
    clause_number: str | None = Field(None, description="Clause or paragraph identifier")
    quote: str = Field(..., description="Exact supporting text excerpt from source document")
    verified: bool = Field(False, description="True if quote was verified against retrieved chunks by post-check")
    similarity_match: float = Field(0.0, description="Verification match score (0.0 to 1.0)")


class GroundedAnswer(BaseModel):
    """Structured response object returned by the RAG generation pipeline."""

    query: str = Field(..., description="Original user prompt or compliance question")
    answer: str = Field(..., description="Grounded answer synthesized strictly from context")
    citations: list[Citation] = Field(default_factory=list, description="List of granular citations")
    confidence_note: str = Field(
        ...,
        description="Confidence assessment, context completeness, or conflict/temporal warnings",
    )
    has_sufficient_context: bool = Field(
        True,
        description="False if the model triggered the 'insufficient context' refusal path",
    )
    retrieved_chunks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved chunks provided in prompt context for auditability",
    )
    model_used: str = Field(..., description="LLM model identifier used for synthesis")
    prompt_tokens: int = Field(0, description="Tokens consumed by system + context prompt")
    completion_tokens: int = Field(0, description="Tokens generated in completion")
    latency_ms: float = Field(0.0, description="Total generation latency in milliseconds")
    verification_passed: bool = Field(
        True,
        description="True if all citations passed post-check verification against chunks",
    )
