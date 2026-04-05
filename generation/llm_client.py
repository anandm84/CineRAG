"""Ollama LLM client wrapper for local inference with Llama 3.1."""

import logging

import requests

import config

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120  # seconds — local models can be slow on CPU


class OllamaClient:
    """Wrapper around the Ollama REST API for text generation."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or config.LLM_MODEL
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")

    def health_check(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            available = [m.get("name", "") for m in models]
            if any(self.model in name for name in available):
                return True
            logger.warning(
                f"Model '{self.model}' not found. "
                f"Available: {available}. Run: ollama pull {self.model}"
            )
            return False
        except requests.ConnectionError:
            logger.error(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )
            return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt / full formatted prompt.
            system: Optional system message.
            temperature: Sampling temperature (lower = more focused).
            timeout: Request timeout in seconds.

        Returns:
            The generated text response.

        Raises:
            ConnectionError: If Ollama is not reachable.
            RuntimeError: If generation fails.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            response_text = result.get("response", "")

            # Log timing info if available
            total_duration = result.get("total_duration", 0)
            if total_duration:
                seconds = total_duration / 1e9
                logger.info(f"LLM response generated in {seconds:.1f}s")

            return response_text

        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Start Ollama with: ollama serve"
            )
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {timeout}s. "
                "The model may be too slow on this hardware."
            )
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}")
