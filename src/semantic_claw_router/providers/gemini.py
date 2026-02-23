"""Google Gemini / AI Studio provider.

Translates OpenAI-format chat completion requests to the Gemini API format
and translates responses back. Enables seamless routing between vLLM and
Gemini without client changes.
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

# Gemini API base URL
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _openai_to_gemini_messages(messages: list[dict[str, Any]]) -> tuple[list[dict], str | None]:
    """Convert OpenAI message format to Gemini format.

    Returns:
        (gemini_contents, system_instruction)
    """
    system_instruction = None
    contents = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            # Gemini uses systemInstruction, not a system message
            system_instruction = content
            continue

        # Map OpenAI roles to Gemini roles
        gemini_role = "user" if role == "user" else "model"

        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append({"text": part["text"]})

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    return contents, system_instruction


def _gemini_to_openai_response(
    gemini_resp: dict[str, Any], model_name: str
) -> dict[str, Any]:
    """Convert Gemini response format to OpenAI chat completion format."""
    candidates = gemini_resp.get("candidates", [])
    choices = []

    for i, candidate in enumerate(candidates):
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        # Filter to only text parts (skip thought/thinking parts)
        text = " ".join(
            p.get("text", "") for p in parts
            if "text" in p and not p.get("thought", False)
        )
        # If no non-thought text, include thought text as fallback
        if not text.strip():
            text = " ".join(p.get("text", "") for p in parts if "text" in p)

        finish_reason_map = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
        }
        finish_reason = finish_reason_map.get(
            candidate.get("finishReason", "STOP"), "stop"
        )

        choices.append({
            "index": i,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "finish_reason": finish_reason,
        })

    # Extract usage
    usage_meta = gemini_resp.get("usageMetadata", {})
    usage = {
        "prompt_tokens": usage_meta.get("promptTokenCount", 0),
        "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
        "total_tokens": usage_meta.get("totalTokenCount", 0),
    }

    return {
        "id": f"chatcmpl-gemini-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": choices,
        "usage": usage,
    }


class GeminiProvider(LLMProvider):
    """Provider for Google Gemini via AI Studio API."""

    async def chat_completion(
        self,
        model: ModelBackend,
        request_body: dict[str, Any],
    ) -> RoutingResponse:
        """Send a chat completion request to Gemini, translating formats."""
        client = await self.get_client()

        # Extract the Gemini model name (strip provider prefix if present)
        gemini_model = model.name
        if "/" in gemini_model:
            gemini_model = gemini_model.split("/")[-1]

        url = f"{GEMINI_API_BASE}/models/{gemini_model}:generateContent"
        params = {"key": model.api_key} if model.api_key else {}

        # Translate OpenAI format → Gemini format
        messages = request_body.get("messages", [])
        contents, system_instruction = _openai_to_gemini_messages(messages)

        gemini_body: dict[str, Any] = {"contents": contents}

        if system_instruction:
            gemini_body["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        # Map generation config
        gen_config: dict[str, Any] = {}
        if "max_tokens" in request_body:
            gen_config["maxOutputTokens"] = request_body["max_tokens"]
        if "temperature" in request_body:
            gen_config["temperature"] = request_body["temperature"]
        if "top_p" in request_body:
            gen_config["topP"] = request_body["top_p"]
        if gen_config:
            gemini_body["generationConfig"] = gen_config

        start = time.monotonic()
        try:
            resp = await client.post(url, json=gemini_body, params=params)
            latency = (time.monotonic() - start) * 1000

            if resp.status_code == 200:
                gemini_resp = resp.json()
                openai_resp = _gemini_to_openai_response(gemini_resp, model.name)
                tokens = openai_resp.get("usage", {})
                return RoutingResponse(
                    status_code=200,
                    headers={"content-type": "application/json"},
                    body=openai_resp,
                    raw_body=json.dumps(openai_resp).encode(),
                    latency_ms=latency,
                    tokens_used=tokens,
                )
            else:
                error_body = resp.json() if resp.content else {}
                logger.error(
                    "Gemini API error %d: %s", resp.status_code, error_body
                )
                return RoutingResponse(
                    status_code=resp.status_code,
                    headers=dict(resp.headers),
                    body={"error": {"message": f"Gemini error: {error_body}"}},
                    latency_ms=latency,
                )
        except httpx.TimeoutException as e:
            logger.error("Timeout calling Gemini: %s", e)
            return RoutingResponse(
                status_code=504,
                headers={},
                body={"error": {"message": f"Gemini timeout: {e}"}},
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            logger.error("Error calling Gemini: %s", e)
            return RoutingResponse(
                status_code=502,
                headers={},
                body={"error": {"message": f"Gemini error: {e}"}},
                latency_ms=(time.monotonic() - start) * 1000,
            )

    async def chat_completion_stream(
        self,
        model: ModelBackend,
        request_body: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Gemini streaming — translates SSE chunks to OpenAI format."""
        # For the POC, we do non-streaming and yield the full response
        resp = await self.chat_completion(model, request_body)
        if resp.body:
            chunk = {
                "id": resp.body.get("id", ""),
                "object": "chat.completion.chunk",
                "created": resp.body.get("created", 0),
                "model": resp.body.get("model", ""),
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": resp.body.get("choices", [{}])[0]
                            .get("message", {}).get("content", ""),
                    },
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"
