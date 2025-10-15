"""In-memory cache to serve recommendation results via HTTP."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Dict, List, Optional

from .models import RecommendationResultMessage


class JobResultCache:
    """Store the latest result per job-id with TTL and waiter support."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = ttl_seconds
        self._values: Dict[str, RecommendationResultMessage] = {}
        self._expires: Dict[str, float] = {}
        self._waiters: Dict[str, List[asyncio.Future]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def set(self, result: RecommendationResultMessage) -> None:
        expiry = time.monotonic() + self._ttl
        async with self._lock:
            self._values[result.job_id] = result
            self._expires[result.job_id] = expiry
            for waiter in self._waiters.pop(result.job_id, []):
                if not waiter.done():
                    waiter.set_result(result)

    async def get(self, job_id: str) -> Optional[RecommendationResultMessage]:
        async with self._lock:
            result = self._values.get(job_id)
            expiry = self._expires.get(job_id)
            if result is None:
                return None
            if expiry is not None and time.monotonic() > expiry:
                self._values.pop(job_id, None)
                self._expires.pop(job_id, None)
                return None
            return result

    async def wait_for(self, job_id: str, timeout: float) -> Optional[RecommendationResultMessage]:
        existing = await self.get(job_id)
        if existing is not None:
            return existing
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        async with self._lock:
            self._waiters[job_id].append(future)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            future.cancel()
            async with self._lock:
                waiters = self._waiters.get(job_id)
                if waiters and future in waiters:
                    waiters.remove(future)
                    if not waiters:
                        self._waiters.pop(job_id, None)
            return None

    async def prune(self) -> None:
        async with self._lock:
            now = time.monotonic()
            stale = [job_id for job_id, expiry in self._expires.items() if expiry < now]
            for job_id in stale:
                self._values.pop(job_id, None)
                self._expires.pop(job_id, None)

    async def size(self) -> int:
        async with self._lock:
            return len(self._values)
