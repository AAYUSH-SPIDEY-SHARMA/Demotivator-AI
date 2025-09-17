# backend/model_client.py
"""
Async model client for vLLM OpenAI-compatible API.
Handles streaming SSE parsing and connection management.
"""

import os
import json
import logging
import asyncio
from typing import AsyncIterator, List, Dict, Any, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ModelClient:
    """Async client for vLLM OpenAI-compatible endpoints."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        """Initialize the model client.

        Args:
            base_url: OpenAI-compatible API base URL (may or may not include /v1)
            timeout: Request timeout in seconds
        """
        self.timeout = float(timeout or os.getenv("REQUEST_TIMEOUT", 30))
        raw_base = base_url or os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        raw_base = raw_base.rstrip("/")

        # Ensure base includes /v1 prefix for OpenAI-compatible APIs
        if not raw_base.endswith("/v1"):
            self.base_url = raw_base + "/v1"
        else:
            self.base_url = raw_base

        # Use a shared AsyncClient but send full URLs (avoid path join quirks)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    # ---------- helpers ----------
    def _chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _models_url(self) -> str:
        return f"{self.base_url}/models"

    async def chat(self, messages: List[Dict[str, str]], model: str,
                   temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Non-streaming chat completion.

        Returns generated assistant content as a string. If the response shape
        is unexpected, returns an empty string and logs a warning.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        url = self._chat_url()
        try:
            resp = await self.client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # defensive extraction
            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                logger.warning("chat(): missing choices in response")
                return ""

            msg = choices[0].get("message")
            if not msg or "content" not in msg:
                # older style may have 'text'
                text = choices[0].get("text")
                if text:
                    return text
                logger.warning("chat(): unexpected choice format")
                return ""

            content = msg.get("content") or ""
            # do not log content (sensitive). log short diagnostic only.
            logger.debug(f"chat(): received content length={len(content)}")
            return content

        except (httpx.TimeoutException, httpx.ReadTimeout) as e:
            logger.warning("Request timeout in chat(), retrying once...")
            # Retry once with slightly reduced timeout
            try:
                resp = await self.client.post(url, json=payload, timeout=self.timeout * 0.8)
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                    return choices[0]["message"]["content"]
                return choices[0].get("text", "") if choices else ""
            except Exception as e2:
                logger.error(f"Retry failed in chat(): {e2}")
                raise

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in chat(): {e.response.status_code} - {e.response.text[:400]}")
            raise
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            raise

    async def chat_stream(self, messages: List[Dict[str, str]], model: str,
                          temperature: float = 0.7, max_tokens: int = 500) -> AsyncIterator[str]:
        """Streaming chat completion with vLLM SSE parsing.

        Yields token strings (deltas) as they arrive.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        url = self._chat_url()
        try:
            async with self.client.stream("POST", url, json=payload, timeout=self.timeout) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # small debug; never log full token text
                    logger.debug(f"Raw stream line len={len(line)}")

                    # SSE style: "data: {...}"
                    if line.startswith("data: "):
                        line = line[6:]

                    # vLLM/OpenAI may send "[DONE]"
                    if line.strip() == "[DONE]":
                        break

                    # Try parse JSON chunk
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("chat_stream(): non-JSON line received; skipping")
                        continue

                    # Expected: {"choices":[{"delta":{"content":"..."},"index":0,"finish_reason":null}], ...}
                    try:
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]

                        # streaming delta
                        if "delta" in choice and isinstance(choice["delta"], dict):
                            content = choice["delta"].get("content")
                            if content:
                                # yield raw content (do not log full)
                                yield content
                                continue

                        # fallback: some servers provide "text"
                        if "text" in choice and choice["text"]:
                            yield choice["text"]
                            continue

                    except Exception as e:
                        logger.debug(f"chat_stream(): unexpected chunk format: {e}")
                        continue

        except asyncio.CancelledError:
            logger.info("Stream cancelled by caller")
            raise
        except (httpx.TimeoutException, httpx.ReadTimeout) as e:
            logger.error("Stream timeout")
            raise
        except Exception as e:
            logger.error(f"Stream error: {e}")
            raise

    async def models(self) -> Dict[str, Any]:
        """Get available models from the API."""
        url = self._models_url()
        try:
            resp = await self.client.get(url)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return {"error": str(e)}

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
