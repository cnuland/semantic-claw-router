"""Google Gemini / AI Studio provider.

Translates OpenAI-format chat completion requests to the Gemini API format
and translates responses back. Enables seamless routing between vLLM and
Gemini without client changes.

Key design: Gemini 3.1 Pro Preview requires thinking mode, and thinking mode
requires thought_signatures on historical functionCall parts. These signatures
can't survive the OpenAI format natively (no field for them). We solve this by
smuggling the thoughtSignature and Gemini function-call ID inside the OpenAI
tool_call_id field using a structured prefix:
    "gsig:<gemini_fc_id>:<base64_thought_signature>"
On the return trip, we extract these and reconstruct native Gemini function
calling format, giving the model clean tool history it can continue from.
"""

from __future__ import annotations

import hashlib
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

# Prefix used to identify tool_call_ids that carry Gemini metadata
# Version 1 format: gsig1:<fc_id>:<crc4>:<thought_signature>
_GSIG_PREFIX = "gsig1:"


def _crc8(data: str) -> str:
    """Compute a short checksum for corruption detection."""
    return hashlib.md5(data.encode()).hexdigest()[:4]


def _pack_tool_call_id(gemini_fc_id: str, thought_signature: str) -> str:
    """Pack Gemini function call ID and thought signature into an OpenAI tool_call_id.

    Format: "gsig1:<gemini_fc_id>:<crc>:<thought_signature>"
    - gsig1 = version 1 prefix (allows future format changes)
    - crc = 4-char MD5 prefix for corruption detection
    - thought_signature = raw value from Gemini (already base64)
    """
    crc = _crc8(thought_signature) if thought_signature else "0000"
    return f"{_GSIG_PREFIX}{gemini_fc_id}:{crc}:{thought_signature}"


def _unpack_tool_call_id(tool_call_id: str) -> tuple[str | None, str | None]:
    """Extract Gemini function call ID and thought signature from a packed tool_call_id.

    Returns (gemini_fc_id, thought_signature) or (None, None) if not a packed ID
    or if the checksum doesn't match (corruption detected).
    """
    if not tool_call_id.startswith(_GSIG_PREFIX):
        return None, None
    remainder = tool_call_id[len(_GSIG_PREFIX):]
    # Format: <fc_id>:<crc>:<signature>
    parts = remainder.split(":", 2)
    if len(parts) < 3:
        logger.warning("Malformed packed tool_call_id: missing parts")
        return None, None
    gemini_fc_id, crc, thought_sig = parts

    # Verify checksum
    if thought_sig:
        expected_crc = _crc8(thought_sig)
        if crc != expected_crc:
            logger.error(
                "Thought signature corruption detected for fc_id=%s: "
                "expected crc %s, got %s. Falling back to text mode.",
                gemini_fc_id, expected_crc, crc,
            )
            return None, None

    return gemini_fc_id, thought_sig


def _openai_to_gemini_messages(messages: list[dict[str, Any]]) -> tuple[list[dict], str | None]:
    """Convert OpenAI message format to Gemini format.

    Uses native Gemini functionCall/functionResponse parts when thought
    signatures are available (packed in tool_call_id). Falls back to text
    flattening for tool calls without signatures (e.g., from non-Gemini models).

    Returns:
        (gemini_contents, system_instruction)
    """
    system_instruction = None
    contents = []

    # Build lookups from tool_call_id → function name and tool_call_id → full info
    tool_call_id_to_name: dict[str, str] = {}
    tool_call_id_to_info: dict[str, dict] = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                func_name = func.get("name", "")
                if tc_id:
                    tool_call_id_to_name[tc_id] = func_name
                    tool_call_id_to_info[tc_id] = tc

    for msg in messages:
        role = msg.get("role", "user")
        raw_content = msg.get("content", "")

        # Normalize content: OpenAI format allows both string and list of parts
        if isinstance(raw_content, list):
            content = " ".join(
                p.get("text", "") for p in raw_content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        elif isinstance(raw_content, str):
            content = raw_content
        else:
            content = str(raw_content) if raw_content else ""

        if role == "system":
            system_instruction = content
            continue

        # Handle assistant messages with tool_calls
        if role == "assistant" and msg.get("tool_calls"):
            parts = []

            # Add any text content as a text part
            if content and content.strip():
                parts.append({"text": content})

            # Check if ANY tool call in this turn has a packed Gemini signature.
            # If so, use native functionCall format for ALL calls in this turn.
            # For parallel calls, only the first functionCall gets the
            # thoughtSignature — subsequent ones in the same turn have empty sigs.
            any_has_sig = any(
                _unpack_tool_call_id(tc.get("id", ""))[0] is not None
                for tc in msg["tool_calls"]
            )

            if any_has_sig:
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id", "")
                    func = tc.get("function", {})
                    func_name = func.get("name", "")
                    args_str = func.get("arguments", "{}")

                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                    gemini_fc_id, thought_sig = _unpack_tool_call_id(tc_id)
                    part: dict[str, Any] = {
                        "functionCall": {
                            "name": func_name,
                            "args": args,
                            "id": gemini_fc_id or tc_id,
                        },
                    }
                    # Only add thoughtSignature if present (first call in parallel)
                    if thought_sig:
                        part["thoughtSignature"] = thought_sig
                    parts.append(part)

                contents.append({"role": "model", "parts": parts})
            else:
                # Fallback: generate text summary (for non-Gemini tool calls)
                summary = content if (content and content.strip()) else _summarize_tool_calls(msg["tool_calls"])
                contents.append({
                    "role": "model",
                    "parts": [{"text": summary}]
                })
            continue

        # Handle tool result messages
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tool_name = msg.get("name", "") or tool_call_id_to_name.get(tool_call_id, "tool")
            tool_content = content if isinstance(content, str) else json.dumps(content)

            # Truncate very long tool results
            if len(tool_content) > 4000:
                tool_content = tool_content[:4000] + "\n... (truncated)"

            # Check if the corresponding tool call had a thought signature
            gemini_fc_id, thought_sig = _unpack_tool_call_id(tool_call_id)
            if gemini_fc_id:
                # Native Gemini functionResponse format
                try:
                    response_data = json.loads(tool_content)
                except (json.JSONDecodeError, TypeError):
                    response_data = {"result": tool_content}

                func_response_part = {
                    "functionResponse": {
                        "name": tool_name,
                        "id": gemini_fc_id,
                        "response": response_data,
                    }
                }

                # Merge into previous user turn if it also has functionResponse parts
                if (contents and contents[-1]["role"] == "user"
                        and any("functionResponse" in p for p in contents[-1]["parts"])):
                    contents[-1]["parts"].append(func_response_part)
                else:
                    contents.append({
                        "role": "user",
                        "parts": [func_response_part]
                    })
            else:
                # Fallback: plain text (for non-Gemini tool results)
                tool_text = f"[{tool_name} result]: {tool_content}"
                if contents and contents[-1]["role"] == "user":
                    contents[-1]["parts"].append({"text": tool_text})
                else:
                    contents.append({
                        "role": "user",
                        "parts": [{"text": tool_text}]
                    })
            continue

        # Regular user/assistant messages
        gemini_role = "user" if role == "user" else "model"

        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append({"text": part["text"]})

        if parts:
            if contents and contents[-1]["role"] == gemini_role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": gemini_role, "parts": parts})

    return contents, system_instruction


def _summarize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Generate a brief text summary for tool calls without thought signatures.

    Used as fallback when native Gemini format isn't available.
    """
    parts = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "action")
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            args = {}

        if name in ("read", "Read"):
            path = args.get("path", args.get("file_path", "file"))
            parts.append(f"I read {path}")
        elif name in ("write", "Write"):
            path = args.get("path", args.get("file_path", "file"))
            parts.append(f"I wrote to {path}")
        elif name in ("edit", "Edit"):
            path = args.get("path", args.get("file_path", "file"))
            parts.append(f"I edited {path}")
        elif name in ("exec", "Bash"):
            cmd = args.get("command", "a command")
            if len(cmd) > 60:
                cmd = cmd[:57] + "..."
            parts.append(f"I ran: {cmd}")
        elif name == "message":
            parts.append("I sent a message")
        else:
            first_val = next(iter(args.values()), "") if args else ""
            if isinstance(first_val, str) and len(first_val) > 40:
                first_val = first_val[:37] + "..."
            parts.append(f"I called {name}({first_val})" if first_val else f"I called {name}")

    return "; ".join(parts) if parts else "I took an action"


def _gemini_to_openai_response(
    gemini_resp: dict[str, Any], model_name: str
) -> dict[str, Any]:
    """Convert Gemini response format to OpenAI chat completion format.

    Packs thoughtSignature into tool_call_id for round-trip preservation.
    """
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
        # Pack thoughtSignature into the tool_call_id for round-trip preservation
        func_call_parts = [p for p in parts if "functionCall" in p]
        if func_call_parts:
            tool_calls = []
            for j, part in enumerate(func_call_parts):
                fc = part["functionCall"]
                thought_sig = part.get("thoughtSignature", "")
                gemini_fc_id = fc.get("id", f"fc_{int(time.time())}_{j}")

                # Pack the signature into the tool_call_id
                if thought_sig:
                    packed_id = _pack_tool_call_id(gemini_fc_id, thought_sig)
                else:
                    packed_id = f"call_gemini_{int(time.time())}_{j}"

                tool_calls.append({
                    "id": packed_id,
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    },
                })
            message["tool_calls"] = tool_calls
            # Per OpenAI convention, content is null when making tool calls.
            message["content"] = None

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

                            # Extract function calls with thought signatures
                            func_call_parts = [
                                p for p in parts if "functionCall" in p
                            ]

                            if not text and not func_call_parts:
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
                            # Pack thoughtSignature into tool_call_id
                            if func_call_parts:
                                logger.info("Gemini stream: tool_call %s",
                                           [p["functionCall"].get("name") for p in func_call_parts])
                                tool_calls = []
                                for idx, part in enumerate(func_call_parts):
                                    fc = part["functionCall"]
                                    thought_sig = part.get("thoughtSignature", "")
                                    gemini_fc_id = fc.get("id", f"fc_{chunk_id}_{idx}")

                                    if thought_sig:
                                        packed_id = _pack_tool_call_id(gemini_fc_id, thought_sig)
                                    else:
                                        packed_id = f"call_{chunk_id}_{idx}"

                                    tool_calls.append({
                                        "index": idx,
                                        "id": packed_id,
                                        "type": "function",
                                        "function": {
                                            "name": fc.get("name", ""),
                                            "arguments": json.dumps(fc.get("args", {})),
                                        },
                                    })
                                delta["tool_calls"] = tool_calls
                                # Suppress text when making tool calls
                                delta.pop("content", None)

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
