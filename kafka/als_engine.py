"""Utilities to execute ALS inference for Kafka workers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from inference.server import (  # type: ignore[attr-defined]
    ALSBundle,
    load_bundle,
    resolve_bundle_path,
)

from .models import FeedbackEvent, RecommendationItem, RecommendationRequest

LOGGER = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration required to load ALS bundles."""

    bundle_path: Path | None = None
    bundle_version: str | None = None
    bundles_root: Path = Path("models/compiled_artifacts")
    allow_faiss: bool = True


class RecommendationEngine:
    """Execute ALS scoring for incoming recommendation jobs."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        resolved = resolve_bundle_path(config.bundle_path, config.bundle_version, config.bundles_root)
        LOGGER.info("Loading ALS bundle from %s", resolved)
        self._bundle: ALSBundle = load_bundle(resolved, enable_faiss=config.allow_faiss)
        self._use_faiss = config.allow_faiss and self._bundle.faiss_index is not None

    def recommend(self, request: RecommendationRequest) -> List[RecommendationItem]:
        if request.user_id is not None:
            return self._recommend_for_user(request.user_id, request.k, request.exclude_item_ids)
        return self._fold_in_and_recommend(request.feedback, request.k, request.exclude_item_ids)

    def _fold_in_and_recommend(
        self,
        feedback: Sequence[FeedbackEvent],
        k: int,
        exclude: Sequence[int],
    ) -> List[RecommendationItem]:
        folded = self._fold_in(feedback)
        item_ids, scores = self._topk(folded, k, exclude)
        return [RecommendationItem(item_id=int(i), score=float(s)) for i, s in zip(item_ids, scores)]

    def _recommend_for_user(self, user_id: int, k: int, exclude: Sequence[int]) -> List[RecommendationItem]:
        if user_id not in self._bundle.uid2row:
            raise KeyError(f"Unknown user_id {user_id}")
        user_row = self._bundle.uid2row[user_id]
        vector = self._bundle.U[user_row]
        item_ids, scores = self._topk(vector, k, exclude)
        return [RecommendationItem(item_id=int(i), score=float(s)) for i, s in zip(item_ids, scores)]

    def _topk(self, vector: np.ndarray, k: int, exclude: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        if self._use_faiss and self._bundle.faiss_index is not None:
            return self._topk_faiss(vector, k, exclude)
        return self._topk_numpy(vector, k, exclude)

    def _topk_numpy(self, vector: np.ndarray, k: int, exclude: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        scores = self._bundle.V @ vector
        ids = self._bundle.row2vid
        scores, ids = self._exclude(scores, ids, exclude)
        if scores.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        k = max(1, min(k if k > 0 else 10, scores.shape[0]))
        idx = np.argpartition(scores, -k)[-k:]
        ordered = idx[np.argsort(scores[idx])[::-1]]
        return ids[ordered], scores[ordered]

    def _topk_faiss(self, vector: np.ndarray, k: int, exclude: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        assert self._bundle.faiss_index is not None
        k = k if k > 0 else 10
        q = vector.reshape(1, -1).astype(np.float32)
        additional = len(exclude)
        search_k = min(self._bundle.row2vid.size, k + additional if additional else k)
        search_k = max(search_k, k)
        distances, indices = self._bundle.faiss_index.search(q, search_k)  # type: ignore[attr-defined]
        valid_mask = indices[0] >= 0
        ids = self._bundle.row2vid[indices[0][valid_mask]]
        scores = distances[0][valid_mask]
        scores, ids = self._exclude(scores, ids, exclude)
        if scores.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        k = min(k, scores.shape[0])
        return ids[:k], scores[:k]

    def _fold_in(self, feedback: Sequence[FeedbackEvent], lam: float = 0.1) -> np.ndarray:
        if not feedback:
            return self._bundle.V.mean(axis=0)
        filtered = [
            (self._bundle.vid2row[event.item_id], event.rating)
            for event in feedback
            if event.item_id in self._bundle.vid2row
        ]
        if not filtered:
            return self._bundle.V.mean(axis=0)
        rows, ratings = zip(*filtered)
        Qi = self._bundle.V[np.array(rows, dtype=np.int64)]
        r = np.asarray(ratings, dtype=np.float32)
        A = Qi.T @ Qi + lam * np.eye(Qi.shape[1], dtype=np.float32)
        b = Qi.T @ r
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:  # pragma: no cover - fallback path
            x = np.linalg.pinv(A) @ b
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _exclude(
        scores: np.ndarray,
        ids: np.ndarray,
        exclude: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        exclude_set = set(int(e) for e in exclude)
        if not exclude_set:
            return scores, ids
        mask = np.array([vid not in exclude_set for vid in ids], dtype=bool)
        return scores[mask], ids[mask]
