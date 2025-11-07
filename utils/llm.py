"""
Async LLM client for Ollama integration.
Replaces the Django ollama_client.py with async support.
"""
import httpx
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from app.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async client for Ollama API interactions."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API base URL (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.generate_url = f"{self.base_url}/api/generate"
        self.embeddings_url = f"{self.base_url}/api/embeddings"
        self.tags_url = f"{self.base_url}/api/tags"

    async def query(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Send a prompt to Ollama and get response.

        Args:
            prompt: Text prompt to send
            model: Model to use (defaults to configured model)
            temperature: Sampling temperature (defaults to settings)
            max_tokens: Max tokens to generate
            stream: Whether to stream response

        Returns:
            Generated text response

        Raises:
            httpx.HTTPError: If API request fails
        """
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": stream,
        }

        # Add optional parameters
        if temperature is not None:
            payload["temperature"] = temperature
        elif settings.temperature is not None:
            payload["temperature"] = settings.temperature

        if max_tokens is not None:
            payload["num_predict"] = max_tokens
        elif settings.max_tokens is not None:
            payload["num_predict"] = settings.max_tokens

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self.generate_url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error querying Ollama: {str(e)}")
            raise

    async def query_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from Ollama token by token.

        Args:
            prompt: Text prompt
            model: Model to use
            temperature: Sampling temperature

        Yields:
            Response tokens as they arrive
        """
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": True,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        elif settings.temperature is not None:
            payload["temperature"] = settings.temperature

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", self.generate_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        import json
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue

    async def get_embeddings(self, text: str, model: Optional[str] = None) -> list[float]:
        """
        Get embeddings for text.

        Args:
            text: Text to embed
            model: Model to use

        Returns:
            Embedding vector
        """
        payload = {
            "model": model or self.model,
            "prompt": text
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.embeddings_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])

    async def check_health(self) -> Dict[str, Any]:
        """
        Check if Ollama is running and model is available.

        Returns:
            Dict with status information
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check if Ollama is running
                response = await client.get(self.tags_url)
                response.raise_for_status()
                data = response.json()

                # Check if our model is available
                models = data.get("models", [])
                model_names = [m.get("name", "") for m in models]
                model_available = any(self.model in name for name in model_names)

                return {
                    "connected": True,
                    "model_available": model_available,
                    "available_models": model_names,
                    "configured_model": self.model
                }
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            return {
                "connected": False,
                "model_available": False,
                "error": str(e)
            }


# Global client instance
ollama_client = OllamaClient()


async def query_ollama(model: str, prompt: str) -> str:
    """
    Compatibility function matching the original Django ollama_client API.

    Args:
        model: Model name
        prompt: Text prompt

    Returns:
        Generated response text
    """
    return await ollama_client.query(prompt=prompt, model=model)
