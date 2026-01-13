from __future__ import annotations

import cv2
import httpx
import numpy as np


class HttpImageFetcher:
    """Small helper to download an image and decode it into a numpy array."""

    def __init__(self, timeout: float = 10.0, max_size: int = 8 * 1024 * 1024):
        self.timeout = timeout
        self.max_size = max_size
        self._client = httpx.AsyncClient(timeout=self.timeout, follow_redirects=True)

    async def fetch(self, url: str) -> np.ndarray:
        response = await self._client.get(url)
        response.raise_for_status()

        content = response.content
        if len(content) > self.max_size:
            raise ValueError(f"Image too large ({len(content)} bytes > {self.max_size})")

        array = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "HttpImageFetcher":
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()
