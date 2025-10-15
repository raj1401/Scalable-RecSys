"""Configuration helpers for Kafka-backed services."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass(frozen=True)
class TopicNames:
    """Logical topic names used by the recommendation platform."""

    requests: str = "recs.requests"
    results: str = "recs.results"


@dataclass
class KafkaSettings:
    """Runtime configuration for Kafka producers and consumers."""

    bootstrap_servers: str = "localhost:9092"
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    client_id: str = "recsys-gateway"
    request_topic: str = field(default_factory=lambda: TopicNames().requests)
    result_topic: str = field(default_factory=lambda: TopicNames().results)
    request_partitions: int = 3
    result_partitions: int = 3
    request_replication: int = 1
    result_replication: int = 1
    consumer_group_requests: str = "recsys-workers"
    consumer_group_results: str = "recsys-result-api"
    request_auto_offset_reset: str = "earliest"
    result_auto_offset_reset: str = "latest"
    result_cache_ttl_seconds: int = 3600
    health_topic_timeout_seconds: float = 2.0

    def security_params(self) -> dict:
        """Return a dictionary of security-related keyword arguments."""

        params: dict[str, str] = {"security_protocol": self.security_protocol}
        if self.security_protocol.upper() != "PLAINTEXT":
            if self.sasl_mechanism:
                params["sasl_mechanism"] = self.sasl_mechanism
            if self.sasl_username:
                params["sasl_plain_username"] = self.sasl_username
            if self.sasl_password:
                params["sasl_plain_password"] = self.sasl_password
        return params


def _get_int(mapping: Mapping[str, str], key: str, default: int) -> int:
    raw = mapping.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid integer value for {key}: {raw}") from exc


def _get_float(mapping: Mapping[str, str], key: str, default: float) -> float:
    raw = mapping.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid float value for {key}: {raw}") from exc


def load_kafka_settings(env: Optional[Mapping[str, str]] = None) -> KafkaSettings:
    """Create :class:`KafkaSettings` from environment variables."""

    data = env or os.environ
    topics = TopicNames(
        requests=data.get("KAFKA_TOPIC_REQUESTS", TopicNames().requests),
        results=data.get("KAFKA_TOPIC_RESULTS", TopicNames().results),
    )

    return KafkaSettings(
        bootstrap_servers=data.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        security_protocol=data.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
        sasl_mechanism=data.get("KAFKA_SASL_MECHANISM"),
        sasl_username=data.get("KAFKA_SASL_USERNAME"),
        sasl_password=data.get("KAFKA_SASL_PASSWORD"),
        client_id=data.get("KAFKA_CLIENT_ID", "recsys-gateway"),
        request_topic=topics.requests,
        result_topic=topics.results,
        request_partitions=_get_int(data, "KAFKA_TOPIC_REQUESTS_PARTITIONS", 3),
        result_partitions=_get_int(data, "KAFKA_TOPIC_RESULTS_PARTITIONS", 3),
        request_replication=_get_int(data, "KAFKA_TOPIC_REQUESTS_REPLICATION", 1),
        result_replication=_get_int(data, "KAFKA_TOPIC_RESULTS_REPLICATION", 1),
        consumer_group_requests=data.get("KAFKA_REQUEST_CONSUMER_GROUP", "recsys-workers"),
        consumer_group_results=data.get("KAFKA_RESULT_CONSUMER_GROUP", "recsys-result-api"),
        request_auto_offset_reset=data.get("KAFKA_REQUEST_AUTO_OFFSET_RESET", "earliest"),
        result_auto_offset_reset=data.get("KAFKA_RESULT_AUTO_OFFSET_RESET", "latest"),
        result_cache_ttl_seconds=_get_int(data, "KAFKA_RESULT_CACHE_TTL_SECONDS", 3600),
        health_topic_timeout_seconds=_get_float(data, "KAFKA_HEALTH_TIMEOUT_SECONDS", 2.0),
    )
