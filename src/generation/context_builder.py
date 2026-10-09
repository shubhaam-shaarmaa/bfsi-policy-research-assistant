"""Context Builder and Token Budgeting for Regulatory RAG.

Formats retrieved search results into a clean, deduplicated, token-budgeted prompt
context block with clear source identifiers for LLM citation generation.
"""

from src.core.logging import app_logger
from src.vectorstore.base import SearchResult


class ContextBuilder:
    """Formats and token-budgets retrieved chunks for LLM consumption."""

    def __init__(self, max_context_tokens: int = 3500) -> None:
        self.max_context_tokens = max_context_tokens

    def build_context(self, search_results: list[SearchResult]) -> tuple[str, list[SearchResult]]:
        """Deduplicate, budget, and assemble context string.

        Args:
            search_results: Ranked list of retrieved SearchResult objects.

        Returns:
            Tuple of (formatted_context_string, list_of_included_results).
        """
        if not search_results:
            return "No relevant regulatory documents were retrieved.", []

        seen_hashes: set[str] = set()
        included_results: list[SearchResult] = []
        context_blocks: list[str] = []
        current_token_estimate = 0

        for idx, res in enumerate(search_results, start=1):
            chunk = res.chunk
            if chunk.content_hash in seen_hashes:
                continue

            chunk_tokens = chunk.token_count or max(1, len(chunk.content.split()))
            if current_token_estimate + chunk_tokens > self.max_context_tokens and included_results:
                app_logger.info("Context token budget reached (%d tokens). Stopping.", current_token_estimate)
                break

            seen_hashes.add(chunk.content_hash)
            included_results.append(res)
            current_token_estimate += chunk_tokens

            # Format source block
            header = (
                f"[SOURCE #{idx} | Doc: {chunk.document_title} | "
                f"Ref: {chunk.circular_number or 'None'} | "
                f"Page: {chunk.page_number} | "
                f"Section: {chunk.section_heading} | "
                f"Clause: {chunk.clause_number or 'None'}]"
            )
            block = f"{header}\n{chunk.content}"
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)
        return formatted_context, included_results
