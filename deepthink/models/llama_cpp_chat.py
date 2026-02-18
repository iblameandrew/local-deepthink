import os
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult


def get_llamacpp_client():
    base_url = os.environ.get("LLAMACPP_BASE_URL", "http://localhost:8080/v1")
    api_key = os.environ.get("LLAMACPP_API_KEY", "no-key-required")
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
        client.models.list()
        return client
    except Exception:
        return None


class ChatLlamaCpp(BaseChatModel):
    """Chat model that uses OpenAI client to connect to llama.cpp server."""

    base_url: str = "http://localhost:8080/v1"
    api_key: str = "no-key-required"
    temperature: float = 0.7
    max_tokens: int = 4096
    model: str = "llama-3.2-1b-instruct"

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

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation via OpenAI client."""
        from openai import OpenAI

        converted = self._convert_messages(messages)
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=converted,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stop=stop,
        )

        content = response.choices[0].message.content
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation via OpenAI client."""
        from openai import AsyncOpenAI

        converted = self._convert_messages(messages)
        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

        response = await client.chat.completions.create(
            model=self.model,
            messages=converted,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stop=stop,
        )

        content = response.choices[0].message.content
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )
