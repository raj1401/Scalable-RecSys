"""HTTP service to expose job results stored in Kafka."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, Dict, Optional

from aiokafka import AIOKafkaConsumer  # type: ignore[import-untyped]
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from .config import KafkaSettings, load_kafka_settings
from .job_cache import JobResultCache
from .models import JobStatus, RecommendationResultMessage

LOGGER = logging.getLogger(__name__)


class ResultStream:
    """Background task consuming Kafka results and updating the cache."""

    def __init__(self, settings: KafkaSettings, cache: JobResultCache) -> None:
        self._settings = settings
        self._cache = cache
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._task: Optional[asyncio.Task] = None
        self._prune_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._task is not None:
            return
        loop = asyncio.get_running_loop()
        consumer = AIOKafkaConsumer(
            self._settings.result_topic,
            loop=loop,
            bootstrap_servers=self._settings.bootstrap_servers,
            client_id=f"{self._settings.client_id}-results",
            group_id=self._settings.consumer_group_results,
            enable_auto_commit=True,
            auto_offset_reset=self._settings.result_auto_offset_reset,
            **self._settings.security_params(),
        )
        await consumer.start()
        self._consumer = consumer
        self._task = asyncio.create_task(self._consume())
        self._prune_task = asyncio.create_task(self._prune_loop())
        LOGGER.info("Result consumer running (group=%s)", self._settings.consumer_group_results)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._prune_task:
            self._prune_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._prune_task
            self._prune_task = None
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
            LOGGER.info("Result consumer stopped")

    async def _consume(self) -> None:
        assert self._consumer is not None
        try:
            async for message in self._consumer:
                try:
                    payload = json.loads(message.value.decode("utf-8"))
                    result = RecommendationResultMessage.model_validate(payload)
                except Exception:  # pragma: no cover - defensive
                    LOGGER.exception("Failed to decode result message")
                    continue
                await self._cache.set(result)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("Result consumer crashed")
            raise

    async def _prune_loop(self) -> None:
        try:
            interval = max(1.0, self._settings.result_cache_ttl_seconds / 2)
            while True:
                await asyncio.sleep(interval)
                await self._cache.prune()
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise


def create_app(settings: Optional[KafkaSettings] = None) -> FastAPI:
    configured_settings = settings or load_kafka_settings()
    cache = JobResultCache(ttl_seconds=configured_settings.result_cache_ttl_seconds)
    stream = ResultStream(configured_settings, cache)
    app = FastAPI(title="ALS Recommendation Results", version="0.1.0")

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - FastAPI lifecycle
        await stream.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - FastAPI lifecycle
        await stream.stop()

    @app.get("/recommendations/{job_id}", response_class=JSONResponse)
    async def get_result(
        job_id: str,
        wait_seconds: Optional[float] = Query(default=None, ge=0, le=60),
    ) -> Dict[str, Any]:
        result = await cache.get(job_id)
        if result is None and wait_seconds:
            result = await cache.wait_for(job_id, wait_seconds)
        if result is None:
            return {
                "job_id": job_id,
                "status": JobStatus.pending.value,
            }
        body: Dict[str, Any] = {
            "job_id": result.job_id,
            "status": result.status.value,
            "updated_at": result.updated_at.isoformat(),
        }
        if result.error:
            body["error"] = result.error
        if result.items:
            body["items"] = [item.model_dump(mode="json") for item in result.items]
        if result.context:
            body["context"] = result.context
        return body

    @app.get("/healthz", response_class=JSONResponse)
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
