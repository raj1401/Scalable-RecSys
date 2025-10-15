"""Kafka consumer that runs ALS inference for recommendation jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer  # type: ignore[import-untyped]

from .als_engine import EngineConfig, RecommendationEngine
from .config import KafkaSettings, load_kafka_settings
from .models import JobStatus, RecommendationJobMessage, RecommendationResultMessage

LOGGER = logging.getLogger(__name__)


class RecommendationWorker:
    """Consume recommendation requests, execute ALS, and push results."""

    def __init__(self, settings: KafkaSettings, engine: RecommendationEngine) -> None:
        self._settings = settings
        self._engine = engine
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._producer: Optional[AIOKafkaProducer] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        consumer = AIOKafkaConsumer(
            self._settings.request_topic,
            loop=loop,
            bootstrap_servers=self._settings.bootstrap_servers,
            client_id=f"{self._settings.client_id}-worker",
            group_id=self._settings.consumer_group_requests,
            enable_auto_commit=False,
            auto_offset_reset=self._settings.request_auto_offset_reset,
            **self._settings.security_params(),
        )
        producer = AIOKafkaProducer(
            loop=loop,
            bootstrap_servers=self._settings.bootstrap_servers,
            client_id=f"{self._settings.client_id}-worker",
            **self._settings.security_params(),
        )
        await consumer.start()
        await producer.start()
        self._consumer = consumer
        self._producer = producer
        LOGGER.info(
            "Worker consuming from %s (group=%s)",
            self._settings.request_topic,
            self._settings.consumer_group_requests,
        )

    async def stop(self) -> None:
        if self._consumer:
            consumer = self._consumer
            self._consumer = None
            await consumer.stop()
        if self._producer:
            producer = self._producer
            self._producer = None
            await producer.stop()
        self._stop_event.set()
        LOGGER.info("Worker shutdown complete")

    async def run(self) -> None:
        if self._consumer is None or self._producer is None:
            raise RuntimeError("Worker not started")
        consumer = self._consumer
        try:
            async for message in consumer:
                await self._handle_message(message.value, message.key)
                await consumer.commit()
                if self._stop_event.is_set():
                    break
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        finally:
            await self.stop()

    async def _handle_message(self, raw_value: bytes, raw_key: Optional[bytes]) -> None:
        assert self._producer is not None
        job_id = raw_key.decode("utf-8") if raw_key else None
        try:
            payload = json.loads(raw_value.decode("utf-8"))
            job_message = RecommendationJobMessage.model_validate(payload)
            if job_id is None:
                job_id = job_message.job_id
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to decode job message")
            await self._emit_failure(job_id or "unknown", f"decode-error: {exc}")
            return

        await self._emit_status(job_message.job_id, JobStatus.running)
        try:
            items = self._engine.recommend(job_message.request)
            result = RecommendationResultMessage(
                job_id=job_message.job_id,
                status=JobStatus.done,
                items=items,
                updated_at=datetime.now(timezone.utc),
                context=job_message.request.context,
            )
        except Exception as exc:  # pragma: no cover - inference failure
            LOGGER.exception("ALS inference failed for job %s", job_message.job_id)
            await self._emit_failure(job_message.job_id, str(exc))
            return

        await self._send_result(result)

    async def _emit_status(self, job_id: str, status: JobStatus) -> None:
        await self._send_result(
            RecommendationResultMessage(
                job_id=job_id,
                status=status,
                updated_at=datetime.now(timezone.utc),
            )
        )

    async def _emit_failure(self, job_id: str, error: str) -> None:
        await self._send_result(
            RecommendationResultMessage(
                job_id=job_id,
                status=JobStatus.failed,
                error=error,
                updated_at=datetime.now(timezone.utc),
            )
        )

    async def _send_result(self, result: RecommendationResultMessage) -> None:
        assert self._producer is not None
        data = result.model_dump(mode="json")
        blob = json.dumps(data, separators=(",", ":")).encode("utf-8")
        await self._producer.send_and_wait(
            topic=self._settings.result_topic,
            key=result.job_id.encode("utf-8"),
            value=blob,
        )
        LOGGER.debug("Published %s for job %s", result.status.value, result.job_id)


@asynccontextmanager
async def worker_context(settings: Optional[KafkaSettings] = None) -> AsyncIterator[RecommendationWorker]:
    configured = settings or load_kafka_settings()
    engine = RecommendationEngine(EngineConfig(allow_faiss=True))
    worker = RecommendationWorker(configured, engine)
    await worker.start()
    try:
        yield worker
    finally:
        await worker.stop()


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = load_kafka_settings()
    engine = RecommendationEngine(EngineConfig(allow_faiss=True))
    worker = RecommendationWorker(settings, engine)
    await worker.start()

    loop = asyncio.get_running_loop()

    def _handle_stop() -> None:
        LOGGER.info("Received shutdown signal")
        worker._stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_stop)

    try:
        await worker.run()
    finally:
        await worker.stop()


if __name__ == "__main__":  # pragma: no cover - script entry
    asyncio.run(main())
