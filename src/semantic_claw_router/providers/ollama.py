"""Ollama provider — routes to Ollama's native and OpenAI-compatible APIs.

Ollama exposes both its native API (/api/chat) and an OpenAI-compatible
API (/v1/chat/completions). This provider uses the OpenAI-compatible
endpoint for consistency, but handles Ollama-specific features like
thinking mode control via the /no_think tag.
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


class OllamaProvider(LLMProvider):
    """Provider for Ollama backends."""

    async def chat_completion(
        self,
        model: ModelBackend,
        request_body: dict[str, Any],
    ) -> RoutingResponse:
        """Send a non-streaming chat completion via Ollama's OpenAI-compat API."""
        client = await self.get_client()
        url = f"{model.endpoint}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        if model.api_key:
            headers["Authorization"] = f"Bearer {model.api_key}"

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

            # Handle Qwen3 thinking mode: if content is empty but reasoning has text
            if resp_body and "choices" in resp_body:
                for choice in resp_body["choices"]:
                    msg = choice.get("message", {})
                    if not msg.get("content") and msg.get("reasoning"):
                        msg["content"] = msg["reasoning"]

            return RoutingResponse(
                status_code=resp.status_code,
                headers=dict(resp.headers),
                body=resp_body,
                raw_body=resp.content,
                latency_ms=latency,
                tokens_used=tokens,
            )
        except httpx.TimeoutException as e:
            logger.error("Timeout calling Ollama %s: %s", url, e)
            return RoutingResponse(
                status_code=504,
                headers={},
                body={"error": {"message": f"Ollama timeout: {model.name}"}},
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            logger.error("Error calling Ollama %s: %s", url, e)
            return RoutingResponse(
                status_code=502,
                headers={},
                body={"error": {"message": f"Ollama error: {e}"}},
                latency_ms=(time.monotonic() - start) * 1000,
            )

    async def chat_completion_stream(
        self,
        model: ModelBackend,
        request_body: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Stream a chat completion response from Ollama.

        Filters out reasoning-only chunks (where content is empty but reasoning
        is present) so downstream clients don't see long pauses while the model
        thinks. Only chunks with actual content or finish_reason are forwarded.
        """
        client = await self.get_client()
        url = f"{model.endpoint}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        if model.api_key:
            headers["Authorization"] = f"Bearer {model.api_key}"

        body = dict(request_body)
        body["stream"] = True
        body["model"] = model.name

        async with client.stream("POST", url, json=body, headers=headers) as resp:
            buffer = b""
            async for raw_bytes in resp.aiter_bytes():
                buffer += raw_bytes
                # SSE chunks are separated by \n\n
                while b"\n\n" in buffer:
                    frame, buffer = buffer.split(b"\n\n", 1)
                    frame = frame.strip()
                    if not frame:
                        continue
                    # Handle "data: [DONE]"
                    if frame == b"data: [DONE]":
                        yield b"data: [DONE]\n\n"
                        return
                    if not frame.startswith(b"data: "):
                        continue
                    json_bytes = frame[6:]
                    try:
                        chunk_data = json.loads(json_bytes)
                    except json.JSONDecodeError:
                        yield frame + b"\n\n"
                        continue

                    # Check if this chunk has actual content
                    choices = chunk_data.get("choices", [])
                    has_content = False
                    for choice in choices:
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        finish = choice.get("finish_reason")
                        if content or finish:
                            has_content = True
                        # Strip the reasoning field from the delta
                        if "reasoning" in delta:
                            del delta["reasoning"]

                    if has_content:
                        yield f"data: {json.dumps(chunk_data)}\n\n".encode()

    async def health_check(self, model: ModelBackend) -> bool:
        """Check if Ollama is healthy via its native API."""
        try:
            client = await self.get_client()
            resp = await client.get(
                f"{model.endpoint}/api/version",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False
