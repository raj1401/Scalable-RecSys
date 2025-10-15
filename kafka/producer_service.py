"""HTTP → Kafka gateway for creating recommendation jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional

from aiokafka import AIOKafkaProducer  # type: ignore[import-untyped]
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .config import KafkaSettings, load_kafka_settings
from .models import JobStatus, RecommendationJobMessage, RecommendationRequest

LOGGER = logging.getLogger(__name__)


class JobPublisher:
    """Wrapper around :class:`AIOKafkaProducer` for job submission."""

    def __init__(self, settings: KafkaSettings):
        self._settings = settings
        self._producer: Optional[AIOKafkaProducer] = None
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._start_lock:
            if self._producer is not None:
                return
            loop = asyncio.get_running_loop()
            producer = AIOKafkaProducer(
                loop=loop,
                bootstrap_servers=self._settings.bootstrap_servers,
                client_id=self._settings.client_id,
                **self._settings.security_params(),
            )
            await producer.start()
            self._producer = producer
            LOGGER.info("Kafka producer started (client_id=%s)", self._settings.client_id)

    async def stop(self) -> None:
        if self._producer is None:
            return
        await self._producer.stop()
        self._producer = None
        LOGGER.info("Kafka producer stopped")

    async def submit_job(self, payload: RecommendationRequest) -> RecommendationJobMessage:
        if self._producer is None:
            raise RuntimeError("Producer not started")
        job_id = str(uuid.uuid4())
        message = RecommendationJobMessage(job_id=job_id, request=payload)
        data = message.model_dump(mode="json")
        blob = json.dumps(data, separators=(",", ":")).encode("utf-8")
        await self._producer.send_and_wait(
            topic=self._settings.request_topic,
            key=job_id.encode("utf-8"),
            value=blob,
        )
        LOGGER.debug("Submitted job %s to %s", job_id, self._settings.request_topic)
        return message


def create_app(settings: Optional[KafkaSettings] = None) -> FastAPI:
    app = FastAPI(title="ALS Recommendation Producer", version="0.1.0")
    configured_settings = settings or load_kafka_settings()
    publisher = JobPublisher(configured_settings)

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - FastAPI lifecycle
        await publisher.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - FastAPI lifecycle
        await publisher.stop()

    @app.post("/recommendations", response_class=JSONResponse)
    async def submit_recommendation(request: RecommendationRequest) -> Dict[str, Any]:
        try:
            job_message = await publisher.submit_job(request)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to submit recommendation job")
            raise HTTPException(status_code=502, detail="Failed to submit job") from exc

        response = {
            "job_id": job_message.job_id,
            "status": JobStatus.pending.value,
            "requested_at": job_message.requested_at.isoformat(),
        }
        return response

    @app.get("/healthz", response_class=JSONResponse)
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
