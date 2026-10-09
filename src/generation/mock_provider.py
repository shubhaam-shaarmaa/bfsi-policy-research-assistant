"""Deterministic Mock LLM Provider for offline testing and CI.

Parses retrieved context chunks from prompt text and constructs valid JSON responses
with citations. If query keywords are missing from context, triggers refusal path.
"""

import json
import re
from src.generation.llm_base import BaseLLMProvider, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """Generates structured grounded JSON answers deterministically from prompt context."""

    def __init__(self, model_name: str = "mock-claude-3-5-sonnet") -> None:
        super().__init__(model_name=model_name)

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        user_msg = messages[-1]["content"] if messages else ""

        # Extract Question and Context from message
        q_match = re.search(r"QUESTION:\s*(.*?)(?=\n\nCONTEXT:|\Z)", user_msg, re.DOTALL)
        query = q_match.group(1).strip() if q_match else user_msg

        # Extract source chunks from context: [SOURCE #1 | Doc: ... | Page: ... | Clause: ...]\nContent
        chunk_pattern = re.compile(
            r"\[SOURCE #(\d+) \| Doc: (.*?) \| Ref: (.*?) \| Page: (\d+) \| Section: (.*?) \| Clause: (.*?)\]\n(.*?)(?=\n\n\[SOURCE #|\Z)",
            re.DOTALL,
        )
        matches = list(chunk_pattern.finditer(user_msg))

        # Check if context exists
        if not matches:
            response_json = {
                "answer": "The provided regulatory documents do not contain sufficient information to answer this question.",
                "citations": [],
                "confidence_note": "No relevant regulatory context was retrieved for this query.",
                "has_sufficient_context": False,
            }
        else:
            # Check if query terms overlap with any chunk
            q_words = set(re.findall(r"[A-Za-z0-9%]+", query.lower()))
            best_chunk = None
            max_overlap = 0

            for m in matches:
                chunk_text = m.group(7).strip()
                c_words = set(re.findall(r"[A-Za-z0-9%]+", chunk_text.lower()))
                overlap = len(q_words.intersection(c_words))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_chunk = m

            # If overlap is trivial (e.g. only stop words), trigger refusal path
            content_words = [w for w in q_words if w not in {"what", "is", "the", "for", "and", "in", "of", "to", "are"}]
            if not best_chunk or max_overlap < min(2, len(content_words)):
                response_json = {
                    "answer": "The provided regulatory documents do not contain information to answer this specific question.",
                    "citations": [],
                    "confidence_note": "Retrieved regulatory passages lack direct statutory rules addressing the prompt.",
                    "has_sufficient_context": False,
                }
            else:
                doc_title = best_chunk.group(2).strip()
                ref = best_chunk.group(3).strip()
                page = int(best_chunk.group(4).strip())
                section = best_chunk.group(5).strip()
                clause = best_chunk.group(6).strip()
                body = best_chunk.group(7).strip()

                # Extract first sentence as quote
                sentences = [s.strip() for s in body.split("\n") if s.strip() and not s.startswith("[")]
                quote_text = sentences[0] if sentences else body[:120]

                answer_text = (
                    f"According to {doc_title} ({ref}), Section '{section}' "
                    f"(Clause {clause}, Page {page}): {quote_text}"
                )

                response_json = {
                    "answer": answer_text,
                    "citations": [
                        {
                            "document_title": doc_title,
                            "circular_number": ref if ref != "None" else None,
                            "page_number": page,
                            "section_heading": section,
                            "clause_number": clause if clause != "None" else None,
                            "quote": quote_text,
                        }
                    ],
                    "confidence_note": "Grounded directly in retrieved RBI regulatory text.",
                    "has_sufficient_context": True,
                }

        content_str = json.dumps(response_json, indent=2)
        return LLMResponse(
            content=content_str,
            model=self.model_name,
            prompt_tokens=len(user_msg.split()),
            completion_tokens=len(content_str.split()),
        )
