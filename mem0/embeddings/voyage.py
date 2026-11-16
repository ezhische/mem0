import os
import time
from typing import Literal, Optional

import httpx

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

DEFAULT_BASE_URL = "https://api.voyageai.com/v1"
MAX_BATCH = 128
REQUEST_TIMEOUT = 60.0
# Seconds to wait before each retry. Rate limits (HTTP 429) get a longer
# schedule: accounts without a payment method are limited to 3 requests/minute.
RETRY_DELAYS = (1, 2)
RATE_LIMIT_RETRY_DELAYS = (5, 15, 25, 25)


class VoyageEmbedding(EmbeddingBase):
    """Voyage AI embeddings over the native REST API.

    Set ``http_client_proxies`` in the embedder config to route only Voyage
    traffic through an HTTP proxy.
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "voyage-4"
        # Only send `output_dimension` when the user asked for a specific size;
        # older Voyage models reject the parameter.
        self._output_dimension = self.config.embedding_dims
        self.config.embedding_dims = self.config.embedding_dims or 1024

        self.api_key = self.config.api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Voyage API key is missing. Set VOYAGE_API_KEY or pass `api_key` in the embedder config.")

        self.base_url = (os.getenv("VOYAGE_API_BASE") or DEFAULT_BASE_URL).rstrip("/")
        # config.http_client is already built from http_client_proxies when set.
        self.client = self.config.http_client or httpx.Client()

        self._input_types = {
            "add": self.config.memory_add_embedding_type or "document",
            "update": self.config.memory_update_embedding_type or "document",
            "search": self.config.memory_search_embedding_type or "query",
        }

    def _request(self, texts, memory_action):
        payload = {"input": texts, "model": self.config.model}
        input_type = self._input_types.get(memory_action)
        if input_type:
            payload["input_type"] = input_type
        if self._output_dimension:
            payload["output_dimension"] = self._output_dimension

        errors = 0
        rate_limits = 0
        while True:
            try:
                response = self.client.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=REQUEST_TIMEOUT,
                )
            except httpx.TransportError:
                if errors >= len(RETRY_DELAYS):
                    raise
                time.sleep(RETRY_DELAYS[errors])
                errors += 1
                continue

            if response.status_code == 429 and rate_limits < len(RATE_LIMIT_RETRY_DELAYS):
                time.sleep(RATE_LIMIT_RETRY_DELAYS[rate_limits])
                rate_limits += 1
            elif response.status_code >= 500 and errors < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[errors])
                errors += 1
            else:
                break

        if response.status_code != 200:
            raise RuntimeError(
                f"Voyage embeddings request failed with HTTP {response.status_code}: {response.text[:300]}"
            )

        data = sorted(response.json()["data"], key=lambda item: item["index"])
        if len(data) != len(texts):
            raise ValueError(
                f"Voyage returned {len(data)} embeddings for {len(texts)} texts using model '{self.config.model}'"
            )
        return [item["embedding"] for item in data]

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Voyage AI.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        return self._request([text], memory_action)[0]

    def embed_batch(self, texts, memory_action="add"):
        """Embed multiple texts, chunked to stay within Voyage's per-request limits."""
        embeddings = []
        for i in range(0, len(texts), MAX_BATCH):
            embeddings.extend(self._request(texts[i : i + MAX_BATCH], memory_action))
        return embeddings
