"""Base provider interface for LLM backends."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

import httpx

from ..router.types import ModelBackend, RoutingResponse

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM provider integrations."""

    def __init__(self, timeout: float = 120.0):
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                follow_redirects=True,
                verify=False,  # For self-signed certs on OpenShift routes
            )
        return self._client

    @abstractmethod
    async def chat_completion(
        self,
        model: ModelBackend,
        request_body: dict[str, Any],
    ) -> RoutingResponse:
        """Send a chat completion request to the backend."""
        ...

    @abstractmethod
    async def chat_completion_stream(
        self,
        model: ModelBackend,
        request_body: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Stream a chat completion response from the backend."""
        ...

    async def health_check(self, model: ModelBackend) -> bool:
        """Check if the backend is healthy."""
        try:
            client = await self.get_client()
            resp = await client.get(
                f"{model.endpoint}/v1/models",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
