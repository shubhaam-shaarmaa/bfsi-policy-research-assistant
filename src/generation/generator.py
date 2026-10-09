"""Grounded Generator Orchestrator.

Combines ContextBuilder, BaseLLMProvider, JSON parser, and CitationVerifier
to produce production-grade, audited, hallucination-checked regulatory answers.
"""

import json
import re
import time
from typing import Any

from config.settings import settings
from src.core.logging import app_logger
from src.generation.context_builder import ContextBuilder
from src.generation.llm_base import BaseLLMProvider
from src.generation.mock_provider import MockLLMProvider
from src.generation.models import Citation, GroundedAnswer
from src.generation.prompts import GROUNDED_SYSTEM_PROMPT, format_user_prompt
from src.generation.verifier import CitationVerifier
from src.vectorstore.base import SearchResult


class GroundedGenerator:
    """Master generation engine executing context assembly, LLM synthesis, and citation verification."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        context_builder: ContextBuilder | None = None,
        verifier: CitationVerifier | None = None,
    ) -> None:
        if llm_provider:
            self.llm_provider = llm_provider
        elif settings.ANTHROPIC_API_KEY:
            from src.generation.anthropic_provider import AnthropicClaudeProvider

            self.llm_provider = AnthropicClaudeProvider(
                api_key=settings.ANTHROPIC_API_KEY,
                model_name=settings.LLM_MODEL,
            )
        else:
            app_logger.info("No ANTHROPIC_API_KEY provided; using MockLLMProvider.")
            self.llm_provider = MockLLMProvider()

        self.context_builder = context_builder or ContextBuilder(max_context_tokens=3500)
        self.verifier = verifier or CitationVerifier(min_similarity_threshold=0.65)

    def _extract_json_payload(self, text: str) -> dict[str, Any]:
        """Robustly extract and parse JSON from model output, handling potential markdown fences."""
        cleaned = text.strip()
        # Strip markdown fences if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback regex search for JSON object {...}
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Could not parse valid JSON from LLM output: {text[:200]}")

    def generate_answer(
        self,
        query: str,
        retrieved_results: list[SearchResult],
    ) -> GroundedAnswer:
        """Synthesize a grounded answer with citations from retrieved results."""
        start_time = time.time()

        # 1. Build context and token budget
        context_str, included_results = self.context_builder.build_context(retrieved_results)
        included_chunks = [r.chunk for r in included_results]

        # 2. Build user prompt
        user_prompt = format_user_prompt(query, context_str)
        messages = [{"role": "user", "content": user_prompt}]

        # 3. Call LLM
        llm_response = self.llm_provider.generate(
            messages=messages,
            system_prompt=GROUNDED_SYSTEM_PROMPT,
            temperature=settings.LLM_TEMPERATURE,
        )

        # 4. Parse JSON
        try:
            parsed = self._extract_json_payload(llm_response.content)
            answer_text = parsed.get("answer", "")
            raw_citations = parsed.get("citations", [])
            confidence_note = parsed.get("confidence_note", "")
            has_sufficient_context = parsed.get("has_sufficient_context", True)
        except Exception as e:
            app_logger.error("Failed to parse JSON response: %s", e)
            answer_text = llm_response.content
            raw_citations = []
            confidence_note = "Model returned non-structured response."
            has_sufficient_context = True

        citations = [Citation.model_validate(c) for c in raw_citations]

        # 5. Citation Verification Post-Check
        verified_citations, all_passed = self.verifier.verify_all(citations, included_chunks)
        if not all_passed:
            confidence_note += " [WARNING: One or more citations failed source verification!]"

        latency_ms = round((time.time() - start_time) * 1000, 2)

        # 6. Format retrieved chunks for auditability
        audit_chunks = [
            {
                "chunk_id": r.chunk.chunk_id,
                "document_title": r.chunk.document_title,
                "circular_number": r.chunk.circular_number,
                "page_number": r.chunk.page_number,
                "clause_number": r.chunk.clause_number,
                "score": r.score,
                "source": r.source,
                "snippet": r.chunk.content[:200] + "...",
            }
            for r in included_results
        ]

        return GroundedAnswer(
            query=query,
            answer=answer_text,
            citations=verified_citations,
            confidence_note=confidence_note,
            has_sufficient_context=has_sufficient_context,
            retrieved_chunks=audit_chunks,
            model_used=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            latency_ms=latency_ms,
            verification_passed=all_passed,
        )
