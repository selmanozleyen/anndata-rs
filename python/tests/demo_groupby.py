#!/usr/bin/env python3
"""Demo: create a synthetic AnnData Zarr store and split it by an obs column.

Usage:
    python tests/demo_groupby.py
    python tests/demo_groupby.py --rows 50000 --groups 8 --mem-gb 1
    RUST_LOG=info python tests/demo_groupby.py
"""
import argparse
import os
import shutil
import tempfile
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

NCOLS = 36_000
AVG_NNZ = 1458
STD_NNZ = 720
MIN_NNZ = 278
NNZ_CHUNK = 349_310


def create_source(path: str, n_rows: int, n_groups: int, seed: int = 42):
    """Build a synthetic AnnData Zarr with a 'sample' obs column."""
    rng = np.random.default_rng(seed)

    # sparse X
    nnz_per_row = rng.normal(AVG_NNZ, STD_NNZ, size=n_rows).astype(np.int64)
    nnz_per_row = np.clip(nnz_per_row, MIN_NNZ, NCOLS)
    total_nnz = int(nnz_per_row.sum())
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(nnz_per_row, out=indptr[1:])

    indices = np.empty(total_nnz, dtype=np.int32)
    pos = 0
    for i in range(n_rows):
        nnz = int(nnz_per_row[i])
        cols = rng.choice(NCOLS, size=nnz, replace=False).astype(np.int32)
        cols.sort()
        indices[pos:pos + nnz] = cols
        pos += nnz
    data = rng.standard_normal(total_nnz).astype(np.float32)

    X = sp.csr_matrix((data, indices, indptr), shape=(n_rows, NCOLS))

    # obs with a categorical 'sample' column
    labels = [f"sample_{i}" for i in range(n_groups)]
    obs = pd.DataFrame({
        "sample": pd.Categorical(rng.choice(labels, size=n_rows)),
    }, index=[f"cell_{i}" for i in range(n_rows)])

    # var
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(NCOLS)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    print(f"Writing AnnData ({n_rows:,} x {NCOLS:,}, nnz={total_nnz:,}) ...")
    t0 = time.time()
    adata.write_zarr(path)
    print(f"  wrote {path} in {time.time() - t0:.1f}s")
    return adata


def main():
    parser = argparse.ArgumentParser(description="Demo: groupby split")
    parser.add_argument("--rows", type=int, default=20_000,
                        help="Number of rows (default 20K)")
    parser.add_argument("--groups", type=int, default=4,
                        help="Number of distinct sample groups")
    parser.add_argument("--mem-gb", type=float, default=1.0,
                        help="Rust memory budget in GB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep", action="store_true",
                        help="Keep output files for inspection")
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="demo_groupby_")
    src_path = os.path.join(tmpdir, "input.zarr")
    out_dir = os.path.join(tmpdir, "split")

    try:
        # 1) create source dataset
        print(f"\n=== Creating synthetic dataset ({args.rows:,} rows, "
              f"{args.groups} groups) ===")
        adata = create_source(src_path, args.rows, args.groups, args.seed)
        counts = adata.obs["sample"].value_counts()
        print(f"\n  Group sizes:")
        for name, cnt in counts.items():
            print(f"    {name}: {cnt:,} rows")

        # 2) split by 'sample'
        print(f"\n=== Splitting by 'sample' (mem={args.mem_gb} GB) ===")
        from anndata_rs import split

        t0 = time.time()
        result = split(
            src_path, out_dir, "sample",
            memory_limit=int(args.mem_gb * 1e9),
        )
        elapsed = time.time() - t0
        print(f"\n  Split finished in {elapsed:.1f}s")

        # 3) verify outputs
        print(f"\n=== Verifying outputs ===")
        for group_val, out_path in sorted(result):
            out_adata = ad.read_zarr(out_path)
            expected = int(counts[group_val])
            actual = out_adata.n_obs
            status = "OK" if actual == expected else "MISMATCH"
            print(f"  {group_val}: {actual:,} rows (expected {expected:,}) [{status}]")
            assert actual == expected, f"{group_val}: got {actual}, expected {expected}"

            assert (out_adata.obs["sample"] == group_val).all(), \
                f"{group_val}: obs column has wrong values"

        print(f"\n  All groups verified OK.")

        nnz = adata.X.nnz
        gb = nnz * 8 / 1e9
        if elapsed > 0:
            print(f"  Throughput: {gb / elapsed:.2f} GB/s "
                  f"({gb:.1f} GB uncompressed in {elapsed:.1f}s)")

        if args.keep:
            print(f"\n  Files kept in: {tmpdir}")

    finally:
        if not args.keep:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
