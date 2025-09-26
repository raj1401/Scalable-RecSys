# server.py
"""gRPC inference service for ALS-based recommenders."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, cast

import grpc
import numpy as np

import recs_pb2
import recs_pb2_grpc

if TYPE_CHECKING:  # pragma: no cover
    import recs_pb2 as recs_pb2_module  # noqa: F401
    import recs_pb2_grpc as recs_pb2_grpc_module  # noqa: F401
else:  # Relax typing for generated modules without type hints
    recs_pb2 = cast(Any, recs_pb2)
    recs_pb2_grpc = cast(Any, recs_pb2_grpc)

try:  # Optional ANN acceleration
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore


LOGGER = logging.getLogger(__name__)

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
DEFAULT_BUNDLES_ROOT = PROJECT_ROOT / "models" / "compiled_artifacts"
NPZ_FILENAME = "als_model.npz"
UID_MAP_FILENAME = "uid2row.json"
VID_MAP_FILENAME = "vid2row.json"


@dataclass(slots=True)
class ALSBundle:
    """In-memory representation of a compiled ALS artifact bundle."""

    U: np.ndarray
    V: np.ndarray
    uid2row: Mapping[int, int]
    vid2row: Mapping[int, int]
    row2vid: np.ndarray
    faiss_index: Optional[Any] = None

    @property
    def rank(self) -> int:
        return int(self.V.shape[1])


def _load_json_map(path: Path, cast_key, cast_val) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {cast_key(k): cast_val(v) for k, v in data.items()}


def _build_row2vid(mapping: Mapping[int, int]) -> np.ndarray:
    size = max(mapping.values(), default=-1) + 1
    row2vid = np.full(size, -1, dtype=np.int64)
    for vid, row in mapping.items():
        if row < 0 or row >= size:
            raise ValueError(f"Invalid row index {row} for vid {vid}")
        row2vid[row] = vid
    if (row2vid < 0).any():
        missing = np.nonzero(row2vid < 0)[0]
        raise ValueError(f"Row indices without video ids detected: {missing[:10]}")
    return row2vid


def _list_bundle_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def resolve_bundle_path(
    bundle_path: Optional[Path],
    bundle_version: Optional[str],
    bundles_root: Path,
) -> Path:
    bundles_root = bundles_root.expanduser()
    if not bundles_root.is_absolute():
        bundles_root = (PROJECT_ROOT / bundles_root).resolve()

    if bundle_path is not None:
        bundle_path = bundle_path.expanduser()
        if not bundle_path.is_absolute():
            bundle_path = (PROJECT_ROOT / bundle_path).resolve()
        return bundle_path
    if bundle_version is not None:
        candidate = bundles_root / bundle_version
        if not candidate.exists():
            available = ", ".join(p.name for p in _list_bundle_dirs(bundles_root)) or "<none>"
            raise FileNotFoundError(
                f"Bundle version '{bundle_version}' not found under {bundles_root}. "
                f"Available: {available}"
            )
        return candidate
    bundles = _list_bundle_dirs(bundles_root)
    if not bundles:
        raise FileNotFoundError(
            f"No bundles found. Expected artifacts under {bundles_root}."
        )
    return bundles[-1]


def load_bundle(bundle_dir: Path, enable_faiss: bool = True) -> ALSBundle:
    """Load ALS factors and lookup tables from a compiled bundle directory."""

    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_dir}")

    npz_path = bundle_dir / NPZ_FILENAME
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {NPZ_FILENAME} in {bundle_dir}")

    LOGGER.info("Loading factor matrices from %s", npz_path)
    bundle = np.load(npz_path, allow_pickle=False)
    U = np.ascontiguousarray(bundle["U"].astype(np.float32, copy=False))
    V = np.ascontiguousarray(bundle["V"].astype(np.float32, copy=False))

    uid_map_path = bundle_dir / UID_MAP_FILENAME
    vid_map_path = bundle_dir / VID_MAP_FILENAME
    uid2row = {
        int(uid): int(row)
        for uid, row in _load_json_map(uid_map_path, int, int).items()
    }
    vid2row = {
        int(vid): int(row)
        for vid, row in _load_json_map(vid_map_path, int, int).items()
    }
    row2vid = _build_row2vid(vid2row)

    index = None
    if enable_faiss and faiss is not None:
        LOGGER.info("Building FAISS inner-product index (dim=%d)", V.shape[1])
        index = faiss.IndexFlatIP(V.shape[1])  # type: ignore[attr-defined]
        index.add(V)  # type: ignore[call-arg]
    elif enable_faiss and faiss is None:
        LOGGER.warning("FAISS requested but not available; falling back to numpy search")

    return ALSBundle(U=U, V=V, uid2row=uid2row, vid2row=vid2row, row2vid=row2vid, faiss_index=index)


class RecommenderServicer(recs_pb2_grpc.RecommenderServicer):
    """gRPC service wrapper around a loaded ALS bundle."""

    def __init__(self, bundle: ALSBundle, use_faiss: bool = True):
        self.bundle = bundle
        self.use_faiss = use_faiss and bundle.faiss_index is not None

    # ---------- Recommendation helpers ----------

    @staticmethod
    def _exclude(scores: np.ndarray, ids: np.ndarray, exclude_ids: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
        exclude_set = set(int(e) for e in exclude_ids)
        if not exclude_set:
            return scores, ids
        mask = np.array([id_ not in exclude_set for id_ in ids], dtype=bool)
        return scores[mask], ids[mask]

    def _topk_numpy(self, u_vec: np.ndarray, k: int, exclude_ids: Sequence[int]) -> Tuple[List[int], List[float]]:
        scores = self.bundle.V @ u_vec
        ids = self.bundle.row2vid
        scores, ids = self._exclude(scores, ids, exclude_ids)
        if scores.size == 0:
            return [], []
        k = max(1, min(k if k > 0 else 10, scores.shape[0]))
        idx = np.argpartition(scores, -k)[-k:]
        ordered = idx[np.argsort(scores[idx])[::-1]]
        top_ids = ids[ordered].astype(int).tolist()
        top_scores = scores[ordered].astype(float).tolist()
        return top_ids, top_scores

    def _topk_faiss(self, u_vec: np.ndarray, k: int, exclude_ids: Sequence[int]) -> Tuple[List[int], List[float]]:
        if not self.use_faiss or self.bundle.faiss_index is None:
            return self._topk_numpy(u_vec, k, exclude_ids)

        k = k if k > 0 else 10
        q = u_vec.reshape(1, -1).astype(np.float32)
        additional = len(exclude_ids)
        search_k = min(self.bundle.row2vid.size, k + additional if additional else k)
        search_k = max(search_k, k)
        distances, indices = self.bundle.faiss_index.search(q, search_k)
        valid_mask = indices[0] >= 0
        ids = self.bundle.row2vid[indices[0][valid_mask]]
        scores = distances[0][valid_mask]
        scores, ids = self._exclude(scores, ids, exclude_ids)
        if scores.size == 0:
            return [], []
        k = min(k, scores.shape[0])
        return ids[:k].astype(int).tolist(), scores[:k].astype(float).tolist()

    def fold_in(self, items: Sequence[int], ratings: Sequence[float], lam: float = 0.1) -> np.ndarray:
        filtered: List[Tuple[int, float]] = [
            (self.bundle.vid2row[item], rating)
            for item, rating in zip(items, ratings)
            if item in self.bundle.vid2row
        ]
        if not filtered:
            return self.bundle.V.mean(axis=0)

        rows, filtered_ratings = zip(*filtered)
        Qi = self.bundle.V[np.array(rows, dtype=np.int64)]
        r = np.asarray(filtered_ratings, dtype=np.float32)
        A = Qi.T @ Qi + lam * np.eye(Qi.shape[1], dtype=np.float32)
        b = Qi.T @ r
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x = np.linalg.pinv(A) @ b
        return np.asarray(x, dtype=np.float32)

    # ---------- gRPC methods ----------

    def Health(self, request, context):  # noqa: N802 - gRPC naming
        return recs_pb2.HealthResponse(status="ok")  # type: ignore[attr-defined]

    def RecommendForUser(self, request, context):  # noqa: N802
        user_id = int(request.user_id)
        if user_id not in self.bundle.uid2row:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("unknown user_id")
            return recs_pb2.RecommendResponse()  # type: ignore[attr-defined]

        rank_vector = self.bundle.U[self.bundle.uid2row[user_id]]
        k = request.k if request.k else 20
        exclude = list(request.exclude_item_ids)
        item_ids, scores = (
            self._topk_faiss(rank_vector, k, exclude)
            if self.use_faiss
            else self._topk_numpy(rank_vector, k, exclude)
        )
        return recs_pb2.RecommendResponse(item_ids=item_ids, scores=scores)  # type: ignore[attr-defined]

    def FoldInAndRecommend(self, request, context):  # noqa: N802
        k = request.k if request.k else 20
        exclude = list(request.exclude_item_ids)
        folded = self.fold_in(request.item_ids, request.ratings, lam=0.1)
        item_ids, scores = (
            self._topk_faiss(folded, k, exclude)
            if self.use_faiss
            else self._topk_numpy(folded, k, exclude)
        )
        return recs_pb2.RecommendResponse(item_ids=item_ids, scores=scores)  # type: ignore[attr-defined]


def serve(
    *,
    port: int = 50051,
    max_workers: int = 4,
    use_tls: bool = False,
    bundle_path: Optional[Path] = None,
    bundle_version: Optional[str] = None,
    bundles_root: Path = DEFAULT_BUNDLES_ROOT,
    allow_faiss: Optional[bool] = None,
    tls_cert: Optional[Path] = None,
    tls_key: Optional[Path] = None,
) -> None:
    """Start the gRPC recommender server."""

    resolved_bundle = resolve_bundle_path(bundle_path, bundle_version, bundles_root)
    LOGGER.info("Using bundle at %s", resolved_bundle)

    bundle = load_bundle(resolved_bundle, enable_faiss=allow_faiss is not False)
    enable_faiss = allow_faiss if allow_faiss is not None else True

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    recs_pb2_grpc.add_RecommenderServicer_to_server(
        RecommenderServicer(bundle, use_faiss=enable_faiss),
        server,
    )

    address = f"[::]:{port}"
    if use_tls:
        if tls_cert is None or tls_key is None:
            raise ValueError("TLS requested but certificate or key path is missing")
        private_key = tls_key.read_bytes()
        certificate_chain = tls_cert.read_bytes()
        server_credentials = grpc.ssl_server_credentials([(private_key, certificate_chain)])
        server.add_secure_port(address, server_credentials)
        LOGGER.info("Started secure gRPC server on %s", address)
    else:
        server.add_insecure_port(address)
        LOGGER.info("Started insecure gRPC server on %s", address)

    server.start()
    server.wait_for_termination()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALS recommender gRPC server")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--max-workers", type=int, default=4, help="Thread pool size")
    parser.add_argument(
        "--bundle-path",
        type=Path,
        help="Explicit path to a compiled artifact bundle directory",
    )
    parser.add_argument(
        "--bundle-version",
        type=str,
        help="Version directory name under --bundles-root to load",
    )
    parser.add_argument(
        "--bundles-root",
        type=Path,
        default=DEFAULT_BUNDLES_ROOT,
        help="Root directory containing compiled artifact bundles",
    )
    parser.add_argument(
        "--no-faiss",
        action="store_true",
        help="Disable FAISS even if available",
    )
    parser.add_argument("--use-tls", action="store_true", help="Serve over TLS")
    parser.add_argument("--tls-cert", type=Path, help="Path to server certificate (PEM)")
    parser.add_argument("--tls-key", type=Path, help="Path to server private key (PEM)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the bundle and exit without starting the server",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    resolved_bundle = resolve_bundle_path(args.bundle_path, args.bundle_version, args.bundles_root)
    LOGGER.info("Resolved bundle path: %s", resolved_bundle)

    if args.dry_run:
        load_bundle(resolved_bundle, enable_faiss=not args.no_faiss)
        LOGGER.info("Dry run complete; bundle loaded successfully.")
        return

    serve(
        port=args.port,
        max_workers=args.max_workers,
        use_tls=args.use_tls,
        bundle_path=resolved_bundle,
        bundles_root=args.bundles_root,
        allow_faiss=not args.no_faiss,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
    )


if __name__ == "__main__":
    main()
