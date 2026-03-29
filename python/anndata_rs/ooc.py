"""Out-of-core scatter / permute / split for AnnData Zarr stores.

obs is handled entirely in Python (pandas) where categoricals, nullable
dtypes, etc. work natively.  The heavy matrix I/O (X, layers, obsm, obsp)
plus var/uns/varm/varp copying is delegated to the Rust engine via _scatter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import zarr

from .anndata_rs import _scatter


def scatter(
    input: str | Path,
    outputs: Sequence[tuple[str | Path, np.ndarray]],
    *,
    memory_limit: int | None = None,
    chunk_size: int | None = None,
    shard_size: int | None = None,
    target_shard_bytes: int | None = None,
) -> None:
    """Scatter an AnnData Zarr store into one or more output stores.

    Parameters
    ----------
    input
        Path to the source ``.zarr`` AnnData store.
    outputs
        Each entry is ``(output_path, indices)`` where ``indices`` is a 1-D
        int64 array of source row indices selecting (and reordering) rows.
    memory_limit
        Maximum RAM (bytes) for internal Rust buffers.  Default 2 GiB.
    chunk_size
        Rows per output sub-chunk along axis 0.
    shard_size
        Rows per output shard along axis 0.
    target_shard_bytes
        Target shard size in bytes (overrides ``shard_size``).
    """
    input = str(input)

    rust_outputs = []
    for dst_path, indices in outputs:
        dst_path = str(dst_path)
        idx = np.asarray(indices, dtype=np.int64)
        rust_outputs.append((dst_path, idx))

    kwargs = {}
    if memory_limit is not None:
        kwargs["memory_limit"] = memory_limit
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    if shard_size is not None:
        kwargs["shard_size"] = shard_size
    if target_shard_bytes is not None:
        kwargs["target_shard_bytes"] = target_shard_bytes

    _scatter(input, rust_outputs, **kwargs)

    src_obs = ad.read_zarr(input).obs
    for dst_path, idx in rust_outputs:
        dst_obs = src_obs.iloc[idx].copy()
        store = zarr.open(dst_path, mode="r+")
        ad.io.write_elem(store, "obs", dst_obs)


def permute(
    input: str | Path,
    output: str | Path,
    indices: np.ndarray,
    **kwargs,
) -> None:
    """Permute (reorder / subset) rows of an AnnData Zarr store.

    Convenience wrapper around :func:`scatter` for single-output use.
    """
    scatter(input, [(output, indices)], **kwargs)


def split(
    input: str | Path,
    output_dir: str | Path,
    column: str,
    *,
    obs: pd.DataFrame | None = None,
    **kwargs,
) -> list[tuple[str, str]]:
    """Split an AnnData Zarr store by an obs column.

    Returns a list of ``(group_value, output_path)`` pairs.

    Parameters
    ----------
    input
        Source ``.zarr`` path.
    output_dir
        Directory for output stores (one per group).
    column
        Name of the obs column to group by.
    obs
        Pre-loaded obs DataFrame. If *None*, read from ``input``.
    **kwargs
        Forwarded to :func:`scatter` (memory_limit, chunk_size, etc.).
    """
    input = str(input)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if obs is None:
        obs = ad.read_zarr(input).obs

    groups = {}
    for val in obs[column].unique():
        mask = obs[column] == val
        indices = np.where(mask)[0].astype(np.int64)
        safe = str(val).replace("/", "_").replace("\\", "_").replace(" ", "_")
        groups[str(val)] = (os.path.join(output_dir, f"{safe}.zarr"), indices)

    outputs = list(groups.values())
    scatter(input, outputs, **kwargs)

    return [(val, path) for val, (path, _) in groups.items()]
