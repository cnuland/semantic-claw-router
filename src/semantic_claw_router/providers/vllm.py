"""vLLM / OpenAI-compatible provider.

Routes requests to any backend that speaks the OpenAI chat completion API,
including vLLM, Ollama, text-generation-inference, and OpenAI itself.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

from ..router.types import ModelBackend, RoutingResponse
from .base import LLMProvider

logger = logging.getLogger(__name__)


class VLLMProvider(LLMProvider):
    """Provider for vLLM and OpenAI-compatible backends."""

    async def chat_completion(
        self,
        model: ModelBackend,
        request_body: dict[str, Any],
    ) -> RoutingResponse:
        """Send a non-streaming chat completion request."""
        client = await self.get_client()
        url = f"{model.endpoint}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        if model.api_key:
            headers["Authorization"] = f"Bearer {model.api_key}"

        # Ensure we're not streaming
        body = dict(request_body)
        body["stream"] = False
        body["model"] = model.name

        start = time.monotonic()
        try:
            resp = await client.post(url, json=body, headers=headers)
            latency = (time.monotonic() - start) * 1000

            resp_body = resp.json() if resp.status_code == 200 else None
            tokens = {}
            if resp_body and "usage" in resp_body:
                usage = resp_body["usage"]
                tokens = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

            return RoutingResponse(
                status_code=resp.status_code,
                headers=dict(resp.headers),
                body=resp_body,
                raw_body=resp.content,
                latency_ms=latency,
                tokens_used=tokens,
            )
        except httpx.TimeoutException as e:
            logger.error("Timeout calling %s: %s", url, e)
            return RoutingResponse(
                status_code=504,
                headers={},
                body={"error": {"message": f"Backend timeout: {model.name}"}},
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            logger.error("Error calling %s: %s", url, e)
            return RoutingResponse(
                status_code=502,
                headers={},
                body={"error": {"message": f"Backend error: {e}"}},
                latency_ms=(time.monotonic() - start) * 1000,
            )

    async def chat_completion_stream(
        self,
        model: ModelBackend,
        request_body: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Stream a chat completion response."""
        client = await self.get_client()
        url = f"{model.endpoint}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        if model.api_key:
            headers["Authorization"] = f"Bearer {model.api_key}"

        body = dict(request_body)
        body["stream"] = True
        body["model"] = model.name

        async with client.stream("POST", url, json=body, headers=headers) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk
