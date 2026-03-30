#!/usr/bin/env python3
"""End-to-end scatter benchmark with a synthetic Zarr AnnData store.

Creates a sparse CSR dataset mimicking the tahoe10m chunk/density stats,
then runs _scatter with various chunk/shard configurations.

Usage:
    # Default: 4 GB uncompressed, shuffle, 2 GB memory budget
    python tests/bench_e2e.py

    # Custom size & memory
    python tests/bench_e2e.py --rows 100000 --mem-gb 4

    # Sweep chunk sizes
    python tests/bench_e2e.py --sweep

    # Keep the output for inspection
    python tests/bench_e2e.py --keep

Env vars:
    RUST_LOG=info   to see Rust-side logs
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import scipy.sparse as sp
import zarr


# ---- tahoe10m-like stats ----
TAHOE_AVG_NNZ = 1458
TAHOE_STD_NNZ = 720
TAHOE_MIN_NNZ = 278
TAHOE_NCOLS = 36_000
TAHOE_SRC_CHUNK = 349_310  # NNZ chunk size for data/indices arrays (from tahoe zarr.json)
TAHOE_INDPTR_CHUNK = 349_310


def create_synthetic_zarr(
    path: str,
    n_rows: int,
    n_cols: int = TAHOE_NCOLS,
    avg_nnz: int = TAHOE_AVG_NNZ,
    std_nnz: int = TAHOE_STD_NNZ,
    min_nnz: int = TAHOE_MIN_NNZ,
    chunk_size: int = TAHOE_SRC_CHUNK,
    seed: int = 42,
) -> dict:
    """Create a Zarr v3 AnnData store with a synthetic CSR X matrix."""
    rng = np.random.default_rng(seed)

    t0 = time.time()
    nnz_per_row = rng.normal(avg_nnz, std_nnz, size=n_rows).astype(np.int64)
    nnz_per_row = np.clip(nnz_per_row, min_nnz, n_cols)
    total_nnz = int(nnz_per_row.sum())

    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(nnz_per_row, out=indptr[1:])
    t_indptr = time.time() - t0

    # Build CSR structure with random column indices and float32 data
    t1 = time.time()
    indices = np.empty(total_nnz, dtype=np.int32)
    pos = 0
    batch = 100_000
    for start in range(0, n_rows, batch):
        end = min(start + batch, n_rows)
        for i in range(start, end):
            nnz = int(nnz_per_row[i])
            cols = rng.choice(n_cols, size=nnz, replace=False).astype(np.int32)
            cols.sort()
            indices[pos:pos + nnz] = cols
            pos += nnz
    t_indices = time.time() - t1

    t2 = time.time()
    data = rng.standard_normal(total_nnz).astype(np.float32)
    t_data = time.time() - t2

    stats = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_nnz": total_nnz,
        "avg_nnz": float(nnz_per_row.mean()),
        "data_bytes": total_nnz * 4,
        "indices_bytes": total_nnz * 4,
        "uncompressed_gb": total_nnz * 8 / 1e9,
    }

    print(f"Generated sparse matrix in {time.time() - t0:.1f}s")
    print(f"  shape:      {n_rows:,} x {n_cols:,}")
    print(f"  nnz:        {total_nnz:,}")
    print(f"  avg_nnz:    {nnz_per_row.mean():.0f} (std={nnz_per_row.std():.0f})")
    print(f"  uncompressed: {stats['uncompressed_gb']:.1f} GB (data+indices)")

    # Write Zarr store
    t3 = time.time()
    store = zarr.open(path, mode="w")

    xg = store.create_group("X")
    xg.attrs["encoding-type"] = "csr_matrix"
    xg.attrs["encoding-version"] = "0.1.0"
    xg.attrs["shape"] = [n_rows, n_cols]

    cs = min(chunk_size, total_nnz)
    xg.create_array("data", data=data, chunks=(cs,),
                     compressors=zarr.codecs.ZstdCodec(level=3))
    xg.create_array("indices", data=indices, chunks=(cs,),
                     compressors=zarr.codecs.ZstdCodec(level=3))
    xg.create_array("indptr", data=indptr,
                     chunks=(min(TAHOE_INDPTR_CHUNK, n_rows + 1),),
                     compressors=zarr.codecs.ZstdCodec(level=3))

    disk_size = dir_size(path)
    t_write = time.time() - t3
    stats["disk_bytes"] = disk_size
    stats["compressed_gb"] = disk_size / 1e9

    print(f"  wrote zarr: {disk_size / 1e6:.0f} MB on disk ({t_write:.1f}s)")
    return stats


def dir_size(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def run_scatter(src: str, dst: str, perm: np.ndarray, mem_bytes: int,
                chunk_size=None, shard_size=None, target_shard_bytes=None,
                compression_level=None):
    """Run _scatter and return (elapsed, final_size)."""
    from anndata_rs.anndata_rs import _scatter

    kwargs = dict(memory_limit=mem_bytes)
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    if shard_size is not None:
        kwargs["shard_size"] = shard_size
    if target_shard_bytes is not None:
        kwargs["target_shard_bytes"] = target_shard_bytes
    if compression_level is not None:
        kwargs["compression_level"] = compression_level

    if os.path.exists(dst):
        shutil.rmtree(dst)

    error_box = [None]
    elapsed_box = [0.0]

    def worker():
        try:
            t0 = time.time()
            _scatter(src, [(dst, perm)], **kwargs)
            elapsed_box[0] = time.time() - t0
        except Exception as e:
            elapsed_box[0] = time.time() - t0
            error_box[0] = e

    thread = threading.Thread(target=worker, daemon=True)
    t_start = time.time()
    thread.start()

    # Monitor progress
    while thread.is_alive():
        sz = dir_size(dst) if os.path.isdir(dst) else 0
        elapsed = time.time() - t_start
        print(f"\r  {sz / 1e6:8.1f} MB written  {elapsed:6.1f}s", end="", flush=True)
        time.sleep(0.5)

    thread.join()
    final_sz = dir_size(dst) if os.path.isdir(dst) else 0
    elapsed = elapsed_box[0]
    print(f"\r  {final_sz / 1e6:8.1f} MB written  {elapsed:6.1f}s  DONE")

    if error_box[0]:
        raise error_box[0]

    return elapsed, final_sz


def verify_output(src: str, dst: str, perm: np.ndarray):
    """Quick correctness check on the output."""
    src_store = zarr.open(src, mode="r")
    dst_store = zarr.open(dst, mode="r")

    src_indptr = src_store["X"]["indptr"][:]
    dst_indptr = dst_store["X"]["indptr"][:]

    n = len(perm)
    assert len(dst_indptr) == n + 1, f"indptr len {len(dst_indptr)} != {n + 1}"

    expected_nnz = sum(int(src_indptr[r + 1] - src_indptr[r]) for r in perm[:100])
    actual_nnz = sum(int(dst_indptr[i + 1] - dst_indptr[i]) for i in range(100))
    assert expected_nnz == actual_nnz, f"first 100 rows: nnz {actual_nnz} != {expected_nnz}"

    total_src_nnz = sum(int(src_indptr[r + 1] - src_indptr[r]) for r in perm)
    assert int(dst_indptr[-1]) == total_src_nnz, \
        f"total nnz {dst_indptr[-1]} != {total_src_nnz}"

    print("  correctness: OK")


def main():
    parser = argparse.ArgumentParser(description="E2E scatter benchmark")
    parser.add_argument("--rows", type=int, default=343_000,
                        help="Number of rows (default: 343K -> ~4GB)")
    parser.add_argument("--mem-gb", type=float, default=2.0,
                        help="Rust memory budget in GB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep", action="store_true",
                        help="Keep output files for inspection")
    parser.add_argument("--sweep", action="store_true",
                        help="Run multiple chunk/shard configs")
    parser.add_argument("--tmpdir", type=str, default=None,
                        help="Directory for test data (default: system tmp)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix="scatter_bench_")
    src_path = os.path.join(tmpdir, "src.zarr")
    created_tmp = args.tmpdir is None

    try:
        # Create source
        if not os.path.exists(src_path):
            print(f"\n{'='*60}")
            print("Creating synthetic dataset...")
            print(f"{'='*60}")
            stats = create_synthetic_zarr(src_path, args.rows, seed=args.seed)
        else:
            print(f"Reusing existing {src_path}")
            stats = {"n_rows": args.rows}

        n_rows = args.rows
        mem_bytes = int(args.mem_gb * 1e9)

        # Permutation
        perm = np.random.default_rng(args.seed).permutation(n_rows).astype(np.int64)

        if args.sweep:
            configs = [
                {"label": "default (passthrough)", "kwargs": {}},
                {"label": "chunk=64K",  "kwargs": {"chunk_size": 64_000}},
                {"label": "chunk=256K", "kwargs": {"chunk_size": 256_000}},
                {"label": "chunk=1M",   "kwargs": {"chunk_size": 1_000_000}},
                {"label": "shard=16MB", "kwargs": {"target_shard_bytes": 16_000_000}},
                {"label": "shard=64MB", "kwargs": {"target_shard_bytes": 64_000_000}},
                {"label": "shard=256MB","kwargs": {"target_shard_bytes": 256_000_000}},
                {"label": "zstd=1",     "kwargs": {"compression_level": 1}},
                {"label": "zstd=6",     "kwargs": {"compression_level": 6}},
                {"label": "no compress", "kwargs": {"compression_level": 0}},
            ]
        else:
            configs = [
                {"label": "default", "kwargs": {}},
            ]

        results = []
        for cfg in configs:
            dst_path = os.path.join(tmpdir, f"dst_{cfg['label'].replace(' ', '_')}.zarr")
            print(f"\n{'='*60}")
            print(f"Scatter: {cfg['label']}  (mem={args.mem_gb}GB)")
            print(f"{'='*60}")

            elapsed, out_size = run_scatter(
                src_path, dst_path, perm, mem_bytes, **cfg["kwargs"]
            )
            verify_output(src_path, dst_path, perm)

            throughput = stats.get("uncompressed_gb", 4.0) / elapsed if elapsed > 0 else 0
            results.append({
                "label": cfg["label"],
                "elapsed": elapsed,
                "out_mb": out_size / 1e6,
                "throughput_gbps": throughput,
            })

            if not args.keep:
                shutil.rmtree(dst_path, ignore_errors=True)

        # Summary
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Source: {n_rows:,} rows, ~{stats.get('uncompressed_gb', 4.0):.1f} GB uncompressed")
        print(f"  Memory: {args.mem_gb} GB")
        print(f"  {'Label':<20s} {'Time':>8s} {'Output':>10s} {'Throughput':>12s}")
        print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*12}")
        for r in results:
            print(f"  {r['label']:<20s} {r['elapsed']:7.1f}s {r['out_mb']:8.0f} MB"
                  f"  {r['throughput_gbps']:8.2f} GB/s")

        if args.keep:
            print(f"\nFiles kept in: {tmpdir}")

    finally:
        if not args.keep and created_tmp:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
