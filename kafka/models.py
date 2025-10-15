"""Data models shared across Kafka services."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    """Lifecycle status for recommendation jobs."""

    pending = "PENDING"
    running = "RUNNING"
    done = "DONE"
    failed = "FAILED"


class FeedbackEvent(BaseModel):
    """Single implicit feedback entry used for fold-in requests."""

    item_id: int = Field(..., ge=0)
    rating: float = Field(...)


class RecommendationRequest(BaseModel):
    """Payload submitted by clients for recommendation jobs."""

    user_id: Optional[int] = Field(default=None, ge=0)
    feedback: List[FeedbackEvent] = Field(default_factory=list)
    k: int = Field(default=20, ge=1, le=500)
    exclude_item_ids: List[int] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_union(self) -> "RecommendationRequest":
        if self.user_id is None and not self.feedback:
            raise ValueError("Provide either user_id or feedback for fold-in jobs")
        if self.user_id is not None and self.user_id < 0:
            raise ValueError("user_id must be non-negative")
        return self


class RecommendationItem(BaseModel):
    """Single recommendation entry returned to clients."""

    item_id: int
    score: float


class RecommendationJobMessage(BaseModel):
    """Envelope pushed to the request topic for a new job."""

    job_id: str
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request: RecommendationRequest


class RecommendationResultMessage(BaseModel):
    """Envelope pushed to the result topic as workers update job status."""

    job_id: str
    status: JobStatus
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    items: List[RecommendationItem] = Field(default_factory=list)
    error: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_error_semantics(self) -> "RecommendationResultMessage":
        if self.status is JobStatus.failed and not self.error:
            raise ValueError("FAILED results must include an error message")
        if self.status in {JobStatus.pending, JobStatus.running} and self.items:
            raise ValueError("Intermediate statuses cannot include recommendation items")
        return self
