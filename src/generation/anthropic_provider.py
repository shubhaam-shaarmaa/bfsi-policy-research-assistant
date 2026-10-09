"""Anthropic Claude LLM Provider.

Interfaces with Anthropic's Messages API (Claude 3.5 Sonnet / Claude 3.7 Sonnet).
Enforces zero-temperature deterministic completions with exact token tracking.
"""

import os
from src.core.logging import app_logger
from src.generation.llm_base import BaseLLMProvider, LLMResponse


class AnthropicClaudeProvider(BaseLLMProvider):
    """Claude LLM implementation via anthropic SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "claude-3-5-sonnet-20241022",
    ) -> None:
        super().__init__(model_name=model_name)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.client = None
        if self.api_key:
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                app_logger.warning("Could not initialize Anthropic client: %s", e)

    def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        if not self.client:
            raise RuntimeError(
                "Anthropic client not configured. Set ANTHROPIC_API_KEY in .env or switch to MockLLMProvider."
            )

        app_logger.info("Calling Anthropic Claude (%s) at temperature=%.1f", self.model_name, temperature)

        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content_text = ""
        for block in response.content:
            if getattr(block, "type", "") == "text":
                content_text += block.text

        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

        return LLMResponse(
            content=content_text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_response={"id": response.id, "stop_reason": response.stop_reason},
        )
