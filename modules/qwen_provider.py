
import aiohttp
import time
from typing import Any, List, Optional, Tuple

class QwenVLProvider:
    def __init__(self, server_url: str = "http://localhost:8000/v1/chat/completions", model_name: str = "qwen-vl-8b", temperature: float = 0.1, max_tokens: int = 512, timeout: int = 60, seed: Optional[int] = None):
        self.server_url = server_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.seed = seed

    async def generate_detailed(self, messages: List[dict]) -> dict[str, Any]:
        """Send one request and retain API status for frame-level auditing."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = int(self.seed)

        t0_llm = time.perf_counter()
        result = {
            "response_text": "",
            "latency_sec": 0.0,
            "api_success": False,
            "http_status": None,
            "api_error": None,
            "request_seed": self.seed,
        }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(self.server_url, json=payload) as resp:
                    result["http_status"] = resp.status
                    if resp.status == 200:
                        data = await resp.json()
                        result["response_text"] = data["choices"][0]["message"]["content"] or ""
                        result["api_success"] = True
                    else:
                        result["api_error"] = await resp.text()
                        print(f" [QwenVLProvider] Error {resp.status}: {result['api_error']}")
        except Exception as exc:
            result["api_error"] = f"{type(exc).__name__}: {exc}"
            print(f" [QwenVLProvider] Request failed: {result['api_error']}")
        result["latency_sec"] = time.perf_counter() - t0_llm
        return result

    async def generate(self, messages: List[dict]) -> Tuple[str, float]:
        """Send a chat-completion request to the vLLM server and return (response_text, latency_sec).

        Args:
            messages: OpenAI-format message list, e.g.:
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "..."},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                        ]
                    }
                ]

        Returns:
            (response_text, vlm_latency): empty string and 0.0 on any error.
        """
        result = await self.generate_detailed(messages)
        return result["response_text"], result["latency_sec"]
