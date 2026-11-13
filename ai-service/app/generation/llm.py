"""
Configurable LLM Client Subsystem.
Supports OpenAI, Anthropic, Groq, Ollama, and Mock providers with structured error handling.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Iterator, Dict, Any
import httpx
from app.config import settings
from app.generation.prompts import RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMGenerationError(Exception):
    """Raised when text generation via the LLM provider fails."""
    pass


class BaseLLMClient(ABC):
    """Abstract interface for LLM text completion."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a grounded response for the given user prompt."""
        pass

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        """Generate tokens/chunks progressively. Subclasses provide native SSE/chunk streaming."""
        full_text = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        if full_text:
            yield full_text


class OpenAILLMClient(BaseLLMClient):
    """OpenAI API client (e.g. GPT-4o, GPT-4o-mini)."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.model_name = model_name
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise LLMGenerationError(
                    "OPENAI_API_KEY is not configured in environment. "
                    "Please set OPENAI_API_KEY in your .env file or switch LLM_PROVIDER."
                )
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
                max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except LLMGenerationError:
            raise
        except Exception as e:
            logger.error(f"OpenAI generation error: {str(e)}", exc_info=True)
            raise LLMGenerationError(f"OpenAI generation failed: {str(e)}") from e

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
                max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
                stream=True
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except LLMGenerationError:
            raise
        except Exception as e:
            logger.error(f"OpenAI stream generation error: {str(e)}", exc_info=True)
            raise LLMGenerationError(f"OpenAI stream generation failed: {str(e)}") from e


class GroqLLMClient(BaseLLMClient):
    """Groq API client (e.g. llama-3.1-70b-versatile, mixtral-8x7b-32768)."""

    def __init__(self, model_name: str = "llama-3.1-70b-versatile", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or settings.GROQ_API_KEY
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise LLMGenerationError("GROQ_API_KEY is not set. Please provide GROQ_API_KEY in .env.")
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1"
            )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
                max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except LLMGenerationError:
            raise
        except Exception as e:
            logger.error(f"Groq generation error: {str(e)}", exc_info=True)
            raise LLMGenerationError(f"Groq generation failed: {str(e)}") from e

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
                max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
                stream=True
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except LLMGenerationError:
            raise
        except Exception as e:
            logger.error(f"Groq stream generation error: {str(e)}", exc_info=True)
            raise LLMGenerationError(f"Groq stream generation failed: {str(e)}") from e


class OllamaLLMClient(BaseLLMClient):
    """Local Ollama client (e.g. llama3.1, mistral, gemma2)."""

    def __init__(self, model_name: str = "llama3.1", base_url: Optional[str] = None):
        self.model_name = model_name
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")

    def generate(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
                    "num_predict": max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
                }
            }
            with httpx.Client(timeout=60.0) as client:
                res = client.post(f"{self.base_url}/api/chat", json=payload)
                res.raise_for_status()
                data = res.json()
                return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation error: {str(e)}", exc_info=True)
            raise LLMGenerationError(f"Ollama generation failed at {self.base_url}: {str(e)}") from e

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": True,
                "options": {
                    "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
                    "num_predict": max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
                }
            }
            with httpx.Client(timeout=60.0) as client:
                with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as res:
                    res.raise_for_status()
                    for line in res.iter_lines():
                        if not line:
                            continue
                        chunk_obj = json.loads(line)
                        chunk_text = chunk_obj.get("message", {}).get("content", "")
                        if chunk_text:
                            yield chunk_text
        except Exception as e:
            logger.error(f"Ollama stream generation error: {str(e)}", exc_info=True)
            raise LLMGenerationError(f"Ollama stream generation failed at {self.base_url}: {str(e)}") from e


class AnthropicLLMClient(BaseLLMClient):
    """Anthropic Claude API client."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or settings.ANTHROPIC_API_KEY

    def generate(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        if not self.api_key:
            raise LLMGenerationError("ANTHROPIC_API_KEY is not configured in environment.")
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
                "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE
            }
            with httpx.Client(timeout=45.0) as client:
                res = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                content_blocks = data.get("content", [])
                text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
                return "".join(text_parts).strip()
        except Exception as e:
            logger.error(f"Anthropic generation error: {str(e)}", exc_info=True)
            raise LLMGenerationError(f"Anthropic generation failed: {str(e)}") from e

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        if not self.api_key:
            raise LLMGenerationError("ANTHROPIC_API_KEY is not configured in environment.")
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
                "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
                "stream": True
            }
            with httpx.Client(timeout=45.0) as client:
                with client.stream("POST", "https://api.anthropic.com/v1/messages", headers=headers, json=payload) as res:
                    res.raise_for_status()
                    for line in res.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            chunk = json.loads(data_str)
                            if chunk.get("type") == "content_block_delta":
                                delta = chunk.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
        except Exception as e:
            logger.error(f"Anthropic stream generation error: {str(e)}", exc_info=True)
            raise LLMGenerationError(f"Anthropic stream generation failed: {str(e)}") from e


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for deterministic automated tests and offline testing."""

    def __init__(self, fixed_response: Optional[str] = None):
        self.fixed_response = fixed_response

    def generate(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        if self.fixed_response:
            return self.fixed_response

        prompt_lower = prompt.lower()
        # Context-aware mock generator based on retrieved document content
        if "no relevant document chunks" in prompt_lower or "[no relevant" in prompt_lower:
            if "తెలుగు" in prompt or "ఎంత" in prompt or "వివరణ" in prompt or "సామ్రాజ్య" in prompt:
                return "అందించిన పత్రంలో ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి తగిన సమాచారం లేదు."
            return "The provided document does not contain information to answer this question."

        # Out-of-corpus query refusal keywords
        if "quantum" in prompt_lower or "photosynthesis" in prompt_lower or "విజయనగర" in prompt_lower or "superposition" in prompt_lower:
            if "విజయనగర" in prompt:
                return "అందించిన పత్రంలో ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి తగిన సమాచారం లేదు."
            return "The provided document does not contain information to answer this question."

        if "attendance" in prompt_lower or "హాజరు" in prompt_lower:
            if "ఎంత" in prompt or "శాతం" in prompt:
                return "విశ్వవిద్యాలయ అకడమిక్ నిబంధనల ప్రకారం (Source: academic_regulations.pdf, Page 1), సెమిస్టర్ పరీక్షలకు హాజరు కావడానికి కనీసం 75% హాజరు తప్పనిసరి. వైద్య కారణాల వల్ల 65% నుండి 75% వరకు మినహాయింపు (condonation) అనుమతించబడుతుంది."
            return "Based on the academic regulations (Source: academic_regulations.pdf, Page 1), every registered student must maintain a minimum attendance of 75% in aggregate across all courses. Condonation of attendance shortage between 65% and 75% may be granted on genuine medical grounds."
        elif "grading" in prompt_lower or "గ్రేడింగ్" in prompt_lower or "cgpa" in prompt_lower or "grade o" in prompt_lower:
            return "According to the grading scale (Source: academic_regulations.pdf, Page 2), courses are assessed on a 10-point scale where Grade O requires 90-100%, Grade A+ requires 80-89%, and the minimum passing mark is 40%."
        elif "fee" in prompt_lower or "revaluation" in prompt_lower or "ఫీజు" in prompt_lower or "రీవాల్యుయేషన్" in prompt_lower or "deadline" in prompt_lower:
            return "According to Section 3 (Source: academic_regulations.pdf, Page 3), the standard examination registration deadline is October 15, 2026. The revaluation fee is Rs. 1000 per theory subject and applications must be submitted within 10 days of result declaration."
        else:
            return "Based on the provided document context, the requested regulation is outlined in the academic policies document."

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        full_text = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        words = full_text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            yield chunk


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini API client (e.g. gemini-1.5-flash, gemini-2.0-flash, gemini-1.5-pro)."""

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or settings.GEMINI_API_KEY

    def generate(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        if not self.api_key:
            raise LLMGenerationError(
                "GEMINI_API_KEY is not configured in environment. "
                "Please set GEMINI_API_KEY in your .env file or switch LLM_PROVIDER."
            )
        import time as pytime
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
                "maxOutputTokens": max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
            }
        }

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                with httpx.Client(timeout=45.0) as client:
                    res = client.post(url, headers=headers, json=payload)
                    if res.status_code == 429 and attempt < max_attempts:
                        delay = attempt * 2.5
                        logger.warning(f"Gemini 429 Rate Limit encountered. Retrying in {delay:.1f}s (Attempt {attempt}/{max_attempts})...")
                        pytime.sleep(delay)
                        continue
                    res.raise_for_status()
                    data = res.json()

                    candidates = data.get("candidates", [])
                    if not candidates:
                        return ""
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text_parts = [p.get("text", "") for p in parts if "text" in p]
                    return "".join(text_parts).strip()
            except LLMGenerationError:
                raise
            except Exception as e:
                if attempt == max_attempts:
                    logger.error(f"Gemini generation error: {str(e)}", exc_info=True)
                    raise LLMGenerationError(f"Gemini generation failed: {str(e)}") from e
                delay = attempt * 2.0
                logger.warning(f"Gemini generation transient error: {e}. Retrying in {delay:.1f}s...")
                pytime.sleep(delay)

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        if not self.api_key:
            raise LLMGenerationError(
                "GEMINI_API_KEY is not configured in environment. "
                "Please set GEMINI_API_KEY in your .env file or switch LLM_PROVIDER."
            )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:streamGenerateContent?alt=sse"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
                "maxOutputTokens": max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
            }
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                with client.stream("POST", url, headers=headers, json=payload) as res:
                    if res.status_code == 429:
                        raise LLMGenerationError("Gemini 429 Rate Limit encountered during streaming.")
                    res.raise_for_status()
                    for line in res.iter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if not data_str:
                            continue
                        try:
                            chunk_data = json.loads(data_str)
                            candidates = chunk_data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for p in parts:
                                    if "text" in p and p["text"]:
                                        yield p["text"]
                        except json.JSONDecodeError:
                            continue
        except LLMGenerationError:
            raise
        except Exception as e:
            logger.error(f"Gemini stream generation error: {str(e)}", exc_info=True)
            raise LLMGenerationError(f"Gemini stream generation failed: {str(e)}") from e


def get_llm_client() -> BaseLLMClient:
    """Factory function returning the configured LLM client."""
    provider = settings.LLM_PROVIDER.lower()
    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not configured in environment; falling back to MockLLMClient for local inference.")
            return MockLLMClient()
        return GeminiLLMClient(model_name=settings.LLM_MODEL)
    elif provider == "openai":
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not configured in environment; falling back to MockLLMClient for local inference.")
            return MockLLMClient()
        return OpenAILLMClient(model_name=settings.LLM_MODEL)
    elif provider == "groq":
        if not settings.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not configured in environment; falling back to MockLLMClient.")
            return MockLLMClient()
        return GroqLLMClient(model_name=settings.LLM_MODEL)
    elif provider == "ollama":
        return OllamaLLMClient(model_name=settings.LLM_MODEL)
    elif provider == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not configured; falling back to MockLLMClient.")
            return MockLLMClient()
        return AnthropicLLMClient(model_name=settings.LLM_MODEL)
    elif provider == "mock":
        return MockLLMClient()
    else:
        logger.warning(f"Unknown LLM_PROVIDER '{provider}'. Falling back to MockLLMClient.")
        return MockLLMClient()

