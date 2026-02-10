"""
Custom LangChain ChatModel wrapper for llama.cpp server.
Makes bare HTTP requests to the /v1/chat/completions endpoint.
"""
import json
import aiohttp
import requests
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult


class ChatLlamaCpp(BaseChatModel):
    """Chat model that sends bare HTTP requests to a llama.cpp server."""

    server_url: str = "http://localhost:8080/v1/chat/completions"
    temperature: float = 0.7
    top_p: float = 0.9
    min_p: float = 0.05
    max_tokens: int = 4096
    request_timeout: int = 300

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-server"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to OpenAI-compatible format."""
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted.append({"role": "assistant", "content": msg.content})
            else:
                converted.append({"role": "user", "content": msg.content})
        return converted

    def _build_payload(self, messages: List[Dict[str, str]], **kwargs) -> dict:
        return {
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "min_p": kwargs.get("min_p", self.min_p),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation via bare HTTP POST."""
        converted = self._convert_messages(messages)
        payload = self._build_payload(converted, **kwargs)
        if stop:
            payload["stop"] = stop

        response = requests.post(
            self.server_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=(10, self.request_timeout),
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation via bare HTTP POST with aiohttp."""
        converted = self._convert_messages(messages)
        payload = self._build_payload(converted, **kwargs)
        if stop:
            payload["stop"] = stop

        timeout = aiohttp.ClientTimeout(
            sock_connect=10, sock_read=self.request_timeout
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.server_url,
                headers={"Content-Type": "application/json"},
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        content = data["choices"][0]["message"]["content"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
