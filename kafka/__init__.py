"""Kafka integration components for the recommendation system."""

from .config import KafkaSettings, TopicNames, load_kafka_settings
from .models import (
    JobStatus,
    RecommendationItem,
    RecommendationJobMessage,
    RecommendationRequest,
    RecommendationResultMessage,
)

__all__ = [
    "KafkaSettings",
    "TopicNames",
    "load_kafka_settings",
    "JobStatus",
    "RecommendationItem",
    "RecommendationJobMessage",
    "RecommendationRequest",
    "RecommendationResultMessage",
]
