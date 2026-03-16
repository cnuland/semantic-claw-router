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

    Handles tool-calling messages:
    - assistant messages with tool_calls → model role with functionCall parts
    - tool messages (results) → user role with functionResponse parts

    Returns:
        (gemini_contents, system_instruction)
    """
    system_instruction = None
    contents = []

    # Build a lookup from tool_call_id → function name so we can resolve
    # tool result names even when the "name" field is missing.
    tool_call_id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                func_name = tc.get("function", {}).get("name", "")
                if tc_id and func_name:
                    tool_call_id_to_name[tc_id] = func_name

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            # Gemini uses systemInstruction, not a system message
            system_instruction = content
            continue

        # Handle assistant messages with tool_calls.
        # Convert to plain text instead of functionCall parts to avoid
        # Gemini's thought_signature requirement on historical function calls.
        # Gemini can still make NEW tool calls via the tools param.
        if role == "assistant" and msg.get("tool_calls"):
            text_parts = []
            if content:
                text_parts.append(content)
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                func_name = func.get("name", "unknown")
                func_args = func.get("arguments", "{}")
                text_parts.append(f"[Called tool: {func_name}({func_args})]")
            if text_parts:
                contents.append({
                    "role": "model",
                    "parts": [{"text": "\n".join(text_parts)}]
                })
            continue

        # Handle tool result messages — convert to plain text user message
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tool_name = msg.get("name", "") or tool_call_id_to_name.get(tool_call_id, "tool")
            tool_content = content if isinstance(content, str) else json.dumps(content)
            # Truncate very long tool results to avoid bloating context
            if len(tool_content) > 4000:
                tool_content = tool_content[:4000] + "\n... (truncated)"
            contents.append({
                "role": "user",
                "parts": [{"text": f"[Tool result from {tool_name}]: {tool_content}"}]
            })
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

        message: dict[str, Any] = {
            "role": "assistant",
            "content": text if text.strip() else None,
        }

        # Translate Gemini functionCall parts to OpenAI tool_calls
        func_calls = [p["functionCall"] for p in parts if "functionCall" in p]
        if func_calls:
            tool_calls = []
            for j, fc in enumerate(func_calls):
                tool_calls.append({
                    "id": f"call_gemini_{int(time.time())}_{j}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    },
                })
            message["tool_calls"] = tool_calls

        choices.append({
            "index": i,
            "message": message,
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


def _openai_to_gemini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool definitions to Gemini function declarations."""
    declarations = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        decl: dict[str, Any] = {"name": func.get("name", "")}
        if func.get("description"):
            decl["description"] = func["description"]
        if func.get("parameters"):
            decl["parameters"] = func["parameters"]
        declarations.append(decl)
    if not declarations:
        return []
    return [{"functionDeclarations": declarations}]


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

        # Map tool definitions if present
        if request_body.get("tools"):
            gemini_body["tools"] = _openai_to_gemini_tools(request_body["tools"])

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
        """Real Gemini streaming via streamGenerateContent endpoint.

        Gemini streams newline-delimited JSON arrays. Each element contains
        a candidates[] with incremental text parts. We translate each chunk
        to an OpenAI SSE streaming format on the fly.
        """
        client = await self.get_client()

        gemini_model = model.name
        if "/" in gemini_model:
            gemini_model = gemini_model.split("/")[-1]

        url = f"{GEMINI_API_BASE}/models/{gemini_model}:streamGenerateContent"
        params = {"key": model.api_key, "alt": "sse"} if model.api_key else {"alt": "sse"}

        messages = request_body.get("messages", [])
        contents, system_instruction = _openai_to_gemini_messages(messages)

        gemini_body: dict[str, Any] = {"contents": contents}
        if system_instruction:
            gemini_body["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        gen_config: dict[str, Any] = {}
        if "max_tokens" in request_body:
            gen_config["maxOutputTokens"] = request_body["max_tokens"]
        if "temperature" in request_body:
            gen_config["temperature"] = request_body["temperature"]
        if "top_p" in request_body:
            gen_config["topP"] = request_body["top_p"]
        if gen_config:
            gemini_body["generationConfig"] = gen_config

        # Map tool definitions if present
        if request_body.get("tools"):
            gemini_body["tools"] = _openai_to_gemini_tools(request_body["tools"])

        chunk_id = f"chatcmpl-gemini-{int(time.time())}"
        created = int(time.time())
        first_chunk = True

        try:
            async with client.stream(
                "POST", url, json=gemini_body, params=params
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    logger.error("Gemini stream error %d: %s", resp.status_code, error_body[:500])
                    error_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model.name,
                        "choices": [{"index": 0, "delta": {"content": f"[Gemini error {resp.status_code}]"}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return

                buffer = ""
                async for raw_chunk in resp.aiter_text():
                    buffer += raw_chunk
                    # Gemini SSE uses \r\n\r\n as delimiter
                    while "\r\n\r\n" in buffer:
                        line, buffer = buffer.split("\r\n\r\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        json_str = line[len("data:"):].strip()
                        if not json_str:
                            continue
                        try:
                            gemini_chunk = json.loads(json_str)
                        except json.JSONDecodeError:
                            continue

                        candidates = gemini_chunk.get("candidates", [])
                        for candidate in candidates:
                            parts = candidate.get("content", {}).get("parts", [])

                            # Extract text content (skip thinking parts)
                            text_parts = [
                                p.get("text", "")
                                for p in parts
                                if "text" in p and not p.get("thought", False)
                            ]
                            text = "".join(text_parts)

                            # Extract function calls
                            func_calls = [
                                p["functionCall"]
                                for p in parts
                                if "functionCall" in p
                            ]

                            if not text and not func_calls:
                                # Check for finishReason even without content
                                gemini_finish = candidate.get("finishReason")
                                if gemini_finish and gemini_finish not in ("FINISH_REASON_UNSPECIFIED",):
                                    finish_map = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter"}
                                    finish_reason = finish_map.get(gemini_finish, "stop")
                                    openai_chunk = {
                                        "id": chunk_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model.name,
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                                    }
                                    yield f"data: {json.dumps(openai_chunk)}\n\n".encode()
                                continue

                            delta: dict[str, Any] = {}
                            if first_chunk:
                                delta["role"] = "assistant"
                                first_chunk = False

                            if text:
                                delta["content"] = text
                                logger.info("Gemini stream: text chunk len=%d", len(text))

                            # Translate Gemini functionCall to OpenAI tool_calls
                            if func_calls:
                                logger.info("Gemini stream: tool_call %s", [fc.get("name") for fc in func_calls])
                                tool_calls = []
                                for i, fc in enumerate(func_calls):
                                    tool_calls.append({
                                        "index": i,
                                        "id": f"call_{chunk_id}_{i}",
                                        "type": "function",
                                        "function": {
                                            "name": fc.get("name", ""),
                                            "arguments": json.dumps(fc.get("args", {})),
                                        },
                                    })
                                delta["tool_calls"] = tool_calls

                            finish_reason = None
                            gemini_finish = candidate.get("finishReason")
                            if gemini_finish and gemini_finish not in ("FINISH_REASON_UNSPECIFIED",):
                                finish_map = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter"}
                                finish_reason = finish_map.get(gemini_finish, "stop")

                            openai_chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model.name,
                                "choices": [{
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": finish_reason,
                                }],
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n".encode()

            yield b"data: [DONE]\n\n"

        except httpx.TimeoutException as e:
            logger.error("Timeout streaming from Gemini: %s", e)
            yield b"data: [DONE]\n\n"
        except Exception as e:
            logger.error("Error streaming from Gemini: %s", e)
            yield b"data: [DONE]\n\n"
