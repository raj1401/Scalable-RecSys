"""Utilities for compiling model factor artifacts into deployable bundles.

The raw artifacts produced by the offline training flow live under
``models/artifacts/<version>/`` and contain two parquet datasets:

* ``user_factors``: a table with the stable customer identifier and the
  learned latent factor vector for each user.
* ``item_factors``: a table with the stable content identifier and the
  learned latent factor vector for each item.

This module provides helpers to stack those factors into dense numpy arrays,
generate lookup tables, and persist the compiled assets under
``models/compiled_artifacts/<version>/``. The code is structured so it can be
reused from an Airflow DAG via a lightweight PythonOperator.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


LOGGER = logging.getLogger(__name__)


DEFAULT_USER_ID_COLUMN = "CUST_ID"
DEFAULT_ITEM_ID_COLUMN = "MOVIE_ID"
FEATURES_COLUMN = "features"


@dataclass
class CompileConfig:
    """Runtime configuration for the compilation process."""

    artifacts_root: Path = Path("models") / "artifacts"
    compiled_root: Path = Path("models") / "compiled_artifacts"
    user_id_column: str = DEFAULT_USER_ID_COLUMN
    item_id_column: str = DEFAULT_ITEM_ID_COLUMN
    force: bool = False


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - informative error path
        message = (
            "pyarrow is required to compile artifacts. Install it with `uv pip "
            "install pyarrow` or add it to your environment."
        )
        raise RuntimeError(message) from exc
    return pq


def _list_versions(path: Path) -> List[Path]:
    if not path.exists():
        LOGGER.debug("Artifacts root %s does not exist", path)
        return []
    return sorted(child for child in path.iterdir() if child.is_dir())


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _chunked_to_numpy(chunked_array) -> np.ndarray:
    """Convert a pyarrow ChunkedArray into a contiguous numpy array."""

    # pyarrow exposes ``to_numpy`` for numeric columns, but we need to fall back
    # to Python lists for nested data such as the latent factor vectors.
    try:
        return chunked_array.to_numpy(zero_copy_only=False)  # type: ignore[attr-defined]
    except (AttributeError, NotImplementedError, TypeError):
        return np.asarray(chunked_array.to_pylist())


def _load_factor_matrix(path: Path, id_column: str) -> tuple[np.ndarray, np.ndarray]:
    pq = _require_pyarrow()

    if not path.exists():
        raise FileNotFoundError(f"Expected parquet directory at {path}")

    table = pq.read_table(path.as_posix())

    if id_column not in table.column_names:
        raise ValueError(
            f"Column '{id_column}' is missing in dataset {path}. "
            f"Available columns: {table.column_names}"
        )
    if FEATURES_COLUMN not in table.column_names:
        raise ValueError(
            f"Column '{FEATURES_COLUMN}' is missing in dataset {path}. "
            f"Available columns: {table.column_names}"
        )

    ids = _chunked_to_numpy(table[id_column]).astype(np.int64, copy=False)

    feature_rows = table[FEATURES_COLUMN].to_pylist()
    if not feature_rows:
        raise ValueError(f"Dataset {path} does not contain any factor rows")

    features = np.asarray(feature_rows, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(
            f"Expected 2-D factors for column '{FEATURES_COLUMN}' in {path}, "
            f"got shape {features.shape}"
        )

    LOGGER.debug(
        "Loaded %d rows with %d dimensions from %s",
        features.shape[0],
        features.shape[1],
        path,
    )

    return ids, features


def _write_numpy_bundle(
    output_dir: Path,
    user_ids: np.ndarray,
    user_factors: np.ndarray,
    item_ids: np.ndarray,
    item_factors: np.ndarray,
) -> Path:
    compiled_path = output_dir / "als_model.npz"
    np.savez(
        compiled_path,
        U=user_factors,
        V=item_factors,
        u_ids=user_ids,
        v_ids=item_ids,
    )
    return compiled_path


def _write_lookup_json(path: Path, ids: Iterable[int]) -> None:
    mapping = {int(identifier): int(idx) for idx, identifier in enumerate(ids)}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False)


def compile_version(version_dir: Path, config: CompileConfig) -> Path:
    """Compile a single artifact version into deployable numpy bundles."""

    user_factors_dir = version_dir / "user_factors"
    item_factors_dir = version_dir / "item_factors"

    if not user_factors_dir.exists():
        raise FileNotFoundError(f"user_factors directory missing in {version_dir}")
    if not item_factors_dir.exists():
        raise FileNotFoundError(f"item_factors directory missing in {version_dir}")

    output_dir = config.compiled_root / version_dir.name
    _ensure_output_dir(output_dir)

    compiled_npz = output_dir / "als_model.npz"
    if compiled_npz.exists() and not config.force:
        LOGGER.info("Skipping %s (already compiled)", version_dir.name)
        return compiled_npz

    user_ids, user_factors = _load_factor_matrix(user_factors_dir, config.user_id_column)
    item_ids, item_factors = _load_factor_matrix(item_factors_dir, config.item_id_column)

    compiled_npz = _write_numpy_bundle(output_dir, user_ids, user_factors, item_ids, item_factors)

    _write_lookup_json(output_dir / "uid2row.json", user_ids)
    _write_lookup_json(output_dir / "vid2row.json", item_ids)

    metadata = {
        "user_count": int(user_factors.shape[0]),
        "item_count": int(item_factors.shape[0]),
        "rank": int(user_factors.shape[1]),
        "user_id_column": config.user_id_column,
        "item_id_column": config.item_id_column,
        "source_version": version_dir.name,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Compiled version %s -> %s (users=%d, items=%d, rank=%d)",
        version_dir.name,
        compiled_npz,
        metadata["user_count"],
        metadata["item_count"],
        metadata["rank"],
    )

    return compiled_npz


def compile_all(config: CompileConfig, version: str | None = None) -> List[Path]:
    versions = _list_versions(config.artifacts_root)
    if not versions:
        LOGGER.warning("No artifact versions found under %s", config.artifacts_root)
        return []

    if version is not None:
        selected = [v for v in versions if v.name == version]
        if not selected:
            raise ValueError(
                f"Requested version '{version}' does not exist. Available versions: "
                f"{[v.name for v in versions]}"
            )
        versions = selected

    compiled_paths = []
    for version_dir in versions:
        compiled_paths.append(compile_version(version_dir, config))
    return compiled_paths


def airflow_task(**context) -> List[str]:
    """Entry point compatible with Airflow's PythonOperator."""

    logging.basicConfig(level=logging.INFO)

    config = CompileConfig()
    version = context.get("params", {}).get("version") if context else None

    compiled = compile_all(config, version=version)
    # Returning file paths gives downstream tasks visibility into the artifacts.
    return [str(path) for path in compiled]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile factor artifacts into deployable bundles.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=CompileConfig.artifacts_root,
        help="Directory containing versioned factor artifacts.",
    )
    parser.add_argument(
        "--compiled-root",
        type=Path,
        default=CompileConfig.compiled_root,
        help="Destination directory for compiled bundles.",
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Only compile a specific artifact version.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompile even if the target files already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available versions and exit without compiling.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase log verbosity.",
    )
    parser.add_argument(
        "--user-id-column",
        type=str,
        default=DEFAULT_USER_ID_COLUMN,
        help="Column name for user identifiers in the user_factors parquet.",
    )
    parser.add_argument(
        "--item-id-column",
        type=str,
        default=DEFAULT_ITEM_ID_COLUMN,
        help="Column name for item identifiers in the item_factors parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    config = CompileConfig(
        artifacts_root=args.artifacts_root,
        compiled_root=args.compiled_root,
        user_id_column=args.user_id_column,
        item_id_column=args.item_id_column,
        force=args.force,
    )

    if args.list:
        versions = _list_versions(config.artifacts_root)
        if not versions:
            print("No artifact versions found.")
        else:
            print("Available versions:")
            for version_dir in versions:
                print(f"- {version_dir.name}")
        return

    compiled = compile_all(config, version=args.version)
    if not compiled:
        LOGGER.info("No artifacts were compiled.")


if __name__ == "__main__":
    main()
