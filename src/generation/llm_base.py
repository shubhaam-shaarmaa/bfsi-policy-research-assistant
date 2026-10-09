"""Abstract Base LLM Provider and Response Schemas."""

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Raw response output from an LLM completion."""

    content: str = Field(..., description="Generated text content (e.g. JSON string)")
    model: str = Field(..., description="Model name executed")
    prompt_tokens: int = Field(0, description="Input tokens counted")
    completion_tokens: int = Field(0, description="Output tokens generated")
    raw_response: dict[str, Any] = Field(default_factory=dict, description="Underlying provider metadata")


class BaseLLMProvider(ABC):
    """Abstract interface for LLM completion providers."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        """Execute chat completion.

        Args:
            messages: List of chat messages [{'role': 'user', 'content': '...'}]
            system_prompt: Grounding system prompt forcing citations and JSON output.
            temperature: Sampling temperature (0.0 for deterministic factual answers).
            max_tokens: Maximum tokens in completion.

        Returns:
            LLMResponse object.
        """
        pass
