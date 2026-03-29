"""anndata_rs -- Rust-backed AnnData I/O with out-of-core scatter engine."""

from __future__ import annotations

from .anndata_rs import *  # noqa: F401,F403  -- compiled extension
from .anndata_rs import _scatter
from .ooc import scatter, permute, split

__all__ = [
    # re-exported from compiled extension
    "AnnData",
    "AnnDataSet",
    "read",
    "read_dataset",
    "read_mtx",
    "concat",
    # out-of-core engine
    "scatter",
    "permute",
    "split",
]
