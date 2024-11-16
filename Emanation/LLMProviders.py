from typing import List, Dict, Any, Optional, Literal, Protocol
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
import time


@dataclass
class MessageFormat:
    """Configuration for message formatting"""
    style: Literal["chatgpt", "llama", "mistral"] = "chatgpt"

    def format_system_message(self, role: str, content: str) -> Dict[str, str]:
        if self.style == "llama":
            return {"role": "user", "content": content}
        else:
            return {"role": role, "content": content}


@dataclass
class LLMConfig:
    """Configuration for the LLM"""
    message_format: MessageFormat = field(default_factory=MessageFormat)
    model: str = "gpt-4"
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    provider: str = "openai"
    rate_limit_delay: Optional[float] = None  # Time to sleep between requests in seconds

    def to_dict(self) -> Dict[str, Any]:
        config_dict = asdict(self)
        # Remove non-API parameters
        for key in ['message_format', 'api_key', 'api_base_url', 'provider', 'rate_limit_delay']:
            del config_dict[key]
        return {k: v for k, v in config_dict.items() if v is not None}


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    def __init__(self, config: LLMConfig):
        self.config = config
        self.rate_limit_delay = config.rate_limit_delay

    def _apply_rate_limit(self, delay: Optional[float] = None):
        """Sleep for specified delay period"""
        sleep_time = delay if delay is not None else self.rate_limit_delay
        if sleep_time:
            time.sleep(sleep_time)

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], delay: Optional[float] = None) -> str:
        pass


class OpenAIClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=config.api_key or "dummy-key",
            base_url=config.api_base_url
        )

    def generate_response(self, messages: List[Dict[str, str]], delay: Optional[float] = None) -> str:
        try:
            self._apply_rate_limit(delay)
            response = self.client.chat.completions.create(
                messages=messages,
                **self.config.to_dict()
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"


class MistralClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from mistralai import Mistral
        self.client = Mistral(api_key=config.api_key)

    def generate_response(self, messages: List[Dict[str, str]], delay: Optional[float] = None) -> str:
        try:
            self._apply_rate_limit(delay)
            response = self.client.chat.complete(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                presence_penalty=None,  # Explicitly None for Mistral
                frequency_penalty=None  # Explicitly None for Mistral
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"


class LLMFactory:
    """Factory for creating LLM clients"""
    _providers = {
        "openai": OpenAIClient,
        "mistral": MistralClient
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: type[LLMClient]):
        """Register a new LLM provider"""
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, config: LLMConfig) -> LLMClient:
        """Create an LLM client based on the configuration"""
        provider_class = cls._providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
        return provider_class(config)


def get_chat_llm(config: LLMConfig) -> LLMClient:
    """Convenience function to create an LLM client"""
    return LLMFactory.create(config)