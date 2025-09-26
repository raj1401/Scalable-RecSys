# client.py
"""
gRPC client for the ALS recommender service.
TEMPORARY
Usage: uv run client.py --host localhost --port 50051 --user-id 10 --k 10 --exclude 99 100 --fold-in-items 10 20 30 --fold-in-ratings 5 3 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import grpc

PROJECT_ROOT = Path(__file__).resolve().parent
INFERENCE_DIR = PROJECT_ROOT / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

import recs_pb2  # type: ignore  # noqa: E402
import recs_pb2_grpc  # type: ignore  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="gRPC client for the ALS recommender")
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument(
        "--user-id",
        type=int,
        default=12345,
        help="User id to request recommendations for",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of recommendations to request",
    )
    parser.add_argument(
        "--exclude",
        type=int,
        nargs="*",
        default=(),
        help="Optional item ids to exclude",
    )
    parser.add_argument(
        "--fold-in-items",
        type=int,
        nargs="*",
        default=(),
        help="Item ids for fold-in request",
    )
    parser.add_argument(
        "--fold-in-ratings",
        type=float,
        nargs="*",
        default=(),
        help="Ratings aligned with --fold-in-items",
    )
    return parser.parse_args(argv)


def _print_recommendations(prefix: str, items: Iterable[int], scores: Iterable[float]) -> None:
    pairs = list(zip(items, scores))
    formatted = ", ".join(f"{item}:{score:.3f}" for item, score in pairs)
    print(f"{prefix}: [{formatted}]" if pairs else f"{prefix}: <none>")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    target = f"{args.host}:{args.port}"
    channel = grpc.insecure_channel(target)
    stub = recs_pb2_grpc.RecommenderStub(channel)

    try:
        health = stub.Health(recs_pb2.HealthRequest())
        print("Health status:", health.status)

        response = stub.RecommendForUser(
            recs_pb2.RecommendForUserRequest(
                user_id=args.user_id,
                k=args.k,
                exclude_item_ids=args.exclude,
            )
        )
        _print_recommendations("RecommendForUser", response.item_ids, response.scores)

        if args.fold_in_items and args.fold_in_ratings:
            if len(args.fold_in_items) != len(args.fold_in_ratings):
                raise ValueError("--fold-in-items and --fold-in-ratings must have equal lengths")
            fold_in_resp = stub.FoldInAndRecommend(
                recs_pb2.FoldInRequest(
                    k=args.k,
                    item_ids=args.fold_in_items,
                    ratings=args.fold_in_ratings,
                    exclude_item_ids=args.exclude,
                )
            )
            _print_recommendations("FoldInAndRecommend", fold_in_resp.item_ids, fold_in_resp.scores)
        else:
            print("FoldInAndRecommend: skipped (provide --fold-in-items and --fold-in-ratings to enable)")
    except grpc.RpcError as exc:  # pragma: no cover - network edge cases
        print(f"RPC failed: {exc.code().name} - {exc.details()}")
        return 1
    except Exception as exc:  # pragma: no cover - CLI misuse
        print(f"Client error: {exc}")
        return 1
    finally:
        channel.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
