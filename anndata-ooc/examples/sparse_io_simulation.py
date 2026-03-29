#!/usr/bin/env python3
"""Simulate the sparse scatter I/O pattern using real Tahoe indptr data.

Faithfully mirrors the Rust batching, merging, chunk-splitting, and
write-grouping logic.  No actual zarr I/O -- just a trace of what the
engine *would* do.

Usage:
    python anndata-ooc/examples/sparse_io_simulation.py [--memory-gb N ...] [--n-rows N]

Reads:  data/indptr/  (the real Tahoe X/indptr zarr array, 89.4M rows)
        data/obs/     (obs columns for group-by simulation)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"


def load_indptr(n_rows: int | None = None) -> np.ndarray:
    """Load the real indptr from the zarr array on disk."""
    try:
        import zarr
    except ImportError:
        sys.exit("zarr is required: pip install zarr")

    arr = zarr.open_array(str(DATA_DIR / "indptr"), mode="r")
    total = arr.shape[0]
    if n_rows is not None and n_rows < total - 1:
        data = arr[: n_rows + 1]
    else:
        data = arr[:]
    return np.asarray(data, dtype=np.int64)


def load_obs_column(column: str, n_rows: int | None = None) -> np.ndarray:
    """Load a categorical obs column (codes array)."""
    import zarr

    col_path = DATA_DIR / "obs" / column / "codes"
    arr = zarr.open_array(str(col_path), mode="r")
    codes = np.asarray(arr[:], dtype=np.int64)
    if n_rows is not None:
        codes = codes[:n_rows]
    return codes


# -- Vectorized simulation --

def merge_and_split(
    sorted_lo: np.ndarray,
    sorted_hi: np.ndarray,
    gap_nnz: int,
    src_chunk_size: int,
) -> tuple[int, int, int]:
    """Vectorized merge + split.

    Returns (n_merged_runs, n_sub_runs, total_read_nnz).
    """
    n = len(sorted_lo)
    if n == 0:
        return 0, 0, 0

    running_max_hi = np.maximum.accumulate(sorted_hi)
    breaks = np.empty(n, dtype=bool)
    breaks[0] = True
    np.greater(sorted_lo[1:], running_max_hi[:-1] + gap_nnz, out=breaks[1:])

    run_ids = np.cumsum(breaks) - 1
    n_merged = int(run_ids[-1]) + 1

    first_in_run = np.where(breaks)[0]
    run_starts = sorted_lo[first_in_run].copy()
    run_ends = np.zeros(n_merged, dtype=np.int64)
    np.maximum.at(run_ends, run_ids, sorted_hi)

    if src_chunk_size > 0:
        n_sub = 0
        total_read = 0
        for i in range(n_merged):
            s, e = int(run_starts[i]), int(run_ends[i])
            fc = s // src_chunk_size
            lc = (e - 1) // src_chunk_size
            n_sub += lc - fc + 1
            total_read += e - s
    else:
        n_sub = n_merged
        total_read = int(np.sum(run_ends - run_starts))

    return n_merged, n_sub, total_read


def simulate_shuffle_analytical(
    indptr: np.ndarray,
    n_rows: int,
    memory_limit: int,
    src_chunk_size: int,
    dst_chunk_size: int,
    data_elem_size: int = 4,
    indices_elem_size: int = 8,
    label: str = "",
    verbose: bool = False,
) -> dict:
    """Analytical simulation for shuffle (random permutation).

    For a random shuffle of N rows, after sorting by source NNZ position
    and merging with gap_nnz=8192:
      - Each batch covers a contiguous slice of sorted-by-position rows
      - The merge produces ~1 run per batch (since rows are dense in NNZ space)
      - The run spans [min_nnz_lo, max_nnz_hi] of the batch's rows
      - Read amplification is [max_nnz_hi - min_nnz_lo] / sum(row_nnz)

    This avoids the O(N log N) argsort entirely.
    """
    bytes_per_nnz = data_elem_size + indices_elem_size
    headroom = 8 * 1024 * 1024
    available = memory_limit - headroom
    max_nnz_per_batch = max(available // bytes_per_nnz, 4096)

    row_nnz = indptr[1 : n_rows + 1] - indptr[:n_rows]
    total_output_nnz = int(row_nnz.sum())
    total_nnz_src = int(indptr[n_rows])

    # Output indptr (for dst chunk computation)
    out_indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(row_nnz, out=out_indptr[1:])

    # Sort row_nnz by source position (which is just the original order for
    # rows 0..n_rows-1, since indptr is monotone). After the shuffle
    # permutation is applied and sorted by source NNZ position, the rows
    # are in their original order (0, 1, 2, ...) because indptr is sorted.
    sorted_row_nnz = row_nnz

    # Batch boundaries by cumulative NNZ
    cum_nnz = np.cumsum(sorted_row_nnz)
    batch_boundaries = [0]
    pos = 0
    total_n = len(cum_nnz)
    while pos < total_n:
        base = int(cum_nnz[pos - 1]) if pos > 0 else 0
        target = base + max_nnz_per_batch
        next_pos = int(np.searchsorted(cum_nnz, target, side="right"))
        if next_pos <= pos:
            next_pos = pos + 1
        batch_boundaries.append(min(next_pos, total_n))
        pos = batch_boundaries[-1]

    n_batches = len(batch_boundaries) - 1

    n_src_chunks = (total_nnz_src + src_chunk_size - 1) // src_chunk_size if src_chunk_size > 0 else 1
    n_dst_chunks_total = (total_output_nnz + dst_chunk_size - 1) // dst_chunk_size if dst_chunk_size > 0 else 1

    total_read_nnz = 0
    total_write_nnz = 0
    total_merged_runs = 0
    total_sub_runs = 0
    total_dst_chunks = 0
    max_batch_decoded = 0
    batch_details = []

    for b in range(n_batches):
        bs = batch_boundaries[b]
        be = batch_boundaries[b + 1]

        # For shuffle: this batch covers source rows bs..be-1 (in sorted-by-nnz order)
        # The merged run spans [indptr[bs], indptr[be]]
        run_lo = int(indptr[bs])
        run_hi = int(indptr[be])
        batch_read_nnz = run_hi - run_lo

        # Split at chunk boundaries
        if src_chunk_size > 0:
            fc = run_lo // src_chunk_size
            lc = (run_hi - 1) // src_chunk_size if run_hi > run_lo else fc
            n_sub = lc - fc + 1
        else:
            n_sub = 1

        n_merged = 1
        total_merged_runs += n_merged
        total_sub_runs += n_sub
        total_read_nnz += batch_read_nnz
        max_batch_decoded = max(max_batch_decoded, batch_read_nnz)

        batch_write_nnz = int(sorted_row_nnz[bs:be].sum())
        total_write_nnz += batch_write_nnz

        # Dst chunks: shuffle scatters output rows uniformly, so each batch
        # touches all destination chunks
        n_dst = n_dst_chunks_total
        total_dst_chunks += n_dst

        batch_details.append({
            "rows": be - bs,
            "merged": n_merged,
            "sub_runs": n_sub,
            "dst_chunks": n_dst,
            "read_nnz": batch_read_nnz,
            "write_nnz": batch_write_nnz,
        })

    total_read_bytes = total_read_nnz * bytes_per_nnz
    total_write_bytes = total_write_nnz * bytes_per_nnz
    read_amp = total_read_nnz / total_output_nnz if total_output_nnz > 0 else 0

    print()
    print(f"=== {label} ===")
    print(f"  Rows: {n_rows:,}  |  Total NNZ: {total_output_nnz:,}  ({total_output_nnz * bytes_per_nnz / 2**30:.1f} GiB)")
    print(f"  Source chunks: {n_src_chunks:,} x {src_chunk_size:,} NNZ  |  Dest chunks: {n_dst_chunks_total:,} x {dst_chunk_size:,} NNZ")
    print(f"  Memory limit:  {memory_limit / 2**30:.1f} GiB  |  max_nnz/batch: {max_nnz_per_batch:,}")
    print()
    print(f"  Batches:       {n_batches}")
    avg_sub = total_sub_runs / n_batches if n_batches else 0
    avg_dst = total_dst_chunks / n_batches if n_batches else 0
    print(f"  Merged runs:   {total_merged_runs:,} total (1.0/batch) -- shuffle always merges to 1")
    print(f"  Sub-runs:      {total_sub_runs:,} total ({avg_sub:.1f}/batch) -- read parallelism tasks")
    print(f"  Dst chunks:    {total_dst_chunks:,} total ({avg_dst:.0f}/batch) -- write parallelism tasks")
    print()
    print(f"  READ:   {total_read_bytes / 2**30:.2f} GiB  ({read_amp:.2f}x amplification)")
    print(f"  WRITE:  {total_write_bytes / 2**30:.2f} GiB")
    print(f"  TOTAL:  {(total_read_bytes + total_write_bytes) / 2**30:.2f} GiB")
    print(f"  Peak batch:    {max_batch_decoded * bytes_per_nnz / 2**30:.2f} GiB decoded")

    if verbose and n_batches <= 50:
        print()
        print(f"  {'Batch':>5}  {'Rows':>12}  {'Merged':>7}  {'SubRuns':>8}  {'DstChk':>7}  "
              f"{'ReadGiB':>8}  {'WriteGiB':>9}")
        for i, d in enumerate(batch_details):
            print(f"  {i:5d}  {d['rows']:12,}  {d['merged']:7,}  {d['sub_runs']:8,}  "
                  f"{d['dst_chunks']:7,}  "
                  f"{d['read_nnz'] * bytes_per_nnz / 2**30:8.3f}  "
                  f"{d['write_nnz'] * bytes_per_nnz / 2**30:9.3f}")

    print()
    return {
        "n_rows": n_rows,
        "batches": n_batches,
        "merged_runs": total_merged_runs,
        "sub_runs": total_sub_runs,
        "dst_chunks": total_dst_chunks,
        "read_gib": total_read_bytes / 2**30,
        "write_gib": total_write_bytes / 2**30,
        "total_gib": (total_read_bytes + total_write_bytes) / 2**30,
        "read_amp": read_amp,
        "batch_details": batch_details,
    }


def simulate_sparse_scatter(
    indptr: np.ndarray,
    src_rows: np.ndarray,
    memory_limit: int,
    src_chunk_size: int,
    dst_chunk_size: int,
    data_elem_size: int = 4,
    indices_elem_size: int = 8,
    label: str = "",
    verbose: bool = False,
) -> dict:
    """Simulate the full sparse scatter pipeline and print I/O stats.

    src_rows: array of source row indices.
    Output rows are implicitly 0..len(src_rows)-1.
    """
    n_assigns = len(src_rows)
    bytes_per_nnz = data_elem_size + indices_elem_size
    headroom = 8 * 1024 * 1024
    available = memory_limit - headroom
    max_nnz_per_batch = max(available // bytes_per_nnz, 4096)

    t0 = time.time()
    row_nnz = indptr[src_rows + 1] - indptr[src_rows]
    total_output_nnz = int(row_nnz.sum())

    out_indptr = np.zeros(n_assigns + 1, dtype=np.int64)
    np.cumsum(row_nnz, out=out_indptr[1:])

    nnz_lo = indptr[src_rows]
    sort_order = np.argsort(nnz_lo, kind="mergesort")
    sorted_lo = nnz_lo[sort_order]
    sorted_hi = indptr[src_rows[sort_order] + 1]
    sorted_row_nnz = row_nnz[sort_order]

    t_sort = time.time() - t0

    cum_nnz = np.cumsum(sorted_row_nnz)
    batch_boundaries = [0]
    pos = 0
    total_n = len(cum_nnz)
    while pos < total_n:
        base = int(cum_nnz[pos - 1]) if pos > 0 else 0
        target = base + max_nnz_per_batch
        next_pos = int(np.searchsorted(cum_nnz, target, side="right"))
        if next_pos <= pos:
            next_pos = pos + 1
        batch_boundaries.append(min(next_pos, total_n))
        pos = batch_boundaries[-1]

    n_batches = len(batch_boundaries) - 1
    t_batch = time.time() - t0

    dst_nnz_pos = out_indptr[sort_order]

    total_read_nnz = 0
    total_write_nnz = 0
    total_merged_runs = 0
    total_sub_runs = 0
    total_dst_chunks = 0
    max_batch_decoded = 0
    batch_details = []

    for b in range(n_batches):
        bs = batch_boundaries[b]
        be = batch_boundaries[b + 1]
        batch_lo = sorted_lo[bs:be]
        batch_hi = sorted_hi[bs:be]

        n_merged, n_sub, batch_read_nnz = merge_and_split(
            batch_lo, batch_hi, gap_nnz=8192, src_chunk_size=src_chunk_size
        )
        total_merged_runs += n_merged
        total_sub_runs += n_sub
        total_read_nnz += batch_read_nnz
        max_batch_decoded = max(max_batch_decoded, batch_read_nnz)

        batch_write_nnz = int(sorted_row_nnz[bs:be].sum())
        total_write_nnz += batch_write_nnz

        if dst_chunk_size > 0:
            batch_dst = dst_nnz_pos[bs:be]
            n_dst = len(np.unique(batch_dst // dst_chunk_size))
        else:
            n_dst = 1
        total_dst_chunks += n_dst

        batch_details.append({
            "rows": be - bs,
            "merged": n_merged,
            "sub_runs": n_sub,
            "dst_chunks": n_dst,
            "read_nnz": batch_read_nnz,
            "write_nnz": batch_write_nnz,
        })

    t_sim = time.time() - t0

    total_nnz_src = int(indptr[-1]) if len(indptr) > 1 else 0
    n_src_chunks = (total_nnz_src + src_chunk_size - 1) // src_chunk_size if src_chunk_size > 0 else 1
    n_dst_chunks_total = (total_output_nnz + dst_chunk_size - 1) // dst_chunk_size if dst_chunk_size > 0 else 1

    total_read_bytes = total_read_nnz * bytes_per_nnz
    total_write_bytes = total_write_nnz * bytes_per_nnz
    read_amp = total_read_nnz / total_output_nnz if total_output_nnz > 0 else 0

    print()
    print(f"=== {label} ===")
    print(f"  Rows: {n_assigns:,}  |  Total NNZ: {total_output_nnz:,}  ({total_output_nnz * bytes_per_nnz / 2**30:.1f} GiB)")
    print(f"  Source chunks: {n_src_chunks:,} x {src_chunk_size:,} NNZ  |  Dest chunks: {n_dst_chunks_total:,} x {dst_chunk_size:,} NNZ")
    print(f"  Memory limit:  {memory_limit / 2**30:.1f} GiB  |  max_nnz/batch: {max_nnz_per_batch:,}")
    print()
    print(f"  Batches:       {n_batches}")
    avg_merged = total_merged_runs / n_batches if n_batches else 0
    avg_sub = total_sub_runs / n_batches if n_batches else 0
    avg_dst = total_dst_chunks / n_batches if n_batches else 0
    print(f"  Merged runs:   {total_merged_runs:,} total ({avg_merged:.1f}/batch)")
    print(f"  Sub-runs:      {total_sub_runs:,} total ({avg_sub:.1f}/batch) -- read parallelism tasks")
    print(f"  Dst chunks:    {total_dst_chunks:,} total ({avg_dst:.1f}/batch) -- write parallelism tasks")
    print()
    print(f"  READ:   {total_read_bytes / 2**30:.2f} GiB  ({read_amp:.2f}x amplification)")
    print(f"  WRITE:  {total_write_bytes / 2**30:.2f} GiB")
    print(f"  TOTAL:  {(total_read_bytes + total_write_bytes) / 2**30:.2f} GiB")
    print(f"  Peak batch:    {max_batch_decoded * bytes_per_nnz / 2**30:.2f} GiB decoded")
    print(f"  Sim time:      sort={t_sort:.1f}s  batch={t_batch - t_sort:.1f}s  sim={t_sim - t_batch:.1f}s  total={t_sim:.1f}s")

    if verbose and n_batches <= 50:
        print()
        print(f"  {'Batch':>5}  {'Rows':>12}  {'Merged':>7}  {'SubRuns':>8}  {'DstChk':>7}  "
              f"{'ReadGiB':>8}  {'WriteGiB':>9}")
        for i, d in enumerate(batch_details):
            print(f"  {i:5d}  {d['rows']:12,}  {d['merged']:7,}  {d['sub_runs']:8,}  "
                  f"{d['dst_chunks']:7,}  "
                  f"{d['read_nnz'] * bytes_per_nnz / 2**30:8.3f}  "
                  f"{d['write_nnz'] * bytes_per_nnz / 2**30:9.3f}")

    print()
    return {
        "n_rows": n_assigns,
        "batches": n_batches,
        "merged_runs": total_merged_runs,
        "sub_runs": total_sub_runs,
        "dst_chunks": total_dst_chunks,
        "read_gib": total_read_bytes / 2**30,
        "write_gib": total_write_bytes / 2**30,
        "total_gib": (total_read_bytes + total_write_bytes) / 2**30,
        "read_amp": read_amp,
        "batch_details": batch_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Sparse I/O simulation with real Tahoe indptr")
    parser.add_argument("--memory-gb", type=float, nargs="+", default=[4, 8, 16, 20, 32, 64],
                        help="Memory limits to simulate (GiB)")
    parser.add_argument("--n-rows", type=int, default=None,
                        help="Truncate indptr to N rows (default: full ~89M)")
    parser.add_argument("--src-chunk-size", type=int, default=67_108_864,
                        help="Source NNZ chunk size (default: 67M from Tahoe)")
    parser.add_argument("--dst-chunk-size", type=int, default=67_108_864,
                        help="Destination NNZ chunk size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--op", choices=["shuffle", "truncate", "split", "all"], default="all",
                        help="Operation to simulate")
    parser.add_argument("--split-column", default="sample",
                        help="obs column for split simulation")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-batch details")
    args = parser.parse_args()

    print(f"Loading indptr from {DATA_DIR / 'indptr'} ...")
    t0 = time.time()
    indptr = load_indptr(args.n_rows)
    n_rows = len(indptr) - 1
    total_nnz = int(indptr[-1])
    row_nnz_all = indptr[1:] - indptr[:-1]
    print(f"  Loaded: {n_rows:,} rows, {total_nnz:,} NNZ  ({time.time() - t0:.1f}s)")
    print(f"  Avg NNZ/row: {total_nnz / n_rows:.1f}")
    print(f"  NNZ range: [{int(row_nnz_all.min())}, {int(row_nnz_all.max())}]")
    print(f"  Data size: {total_nnz * 12 / 2**30:.1f} GiB (4B data + 8B indices)")
    del row_nnz_all

    src_chunk_size = args.src_chunk_size
    dst_chunk_size = args.dst_chunk_size

    summary_rows = []

    for mem_gb in args.memory_gb:
        memory_limit = int(mem_gb * 2**30)
        print()
        print("=" * 80)
        print(f"MEMORY LIMIT: {mem_gb} GiB")
        print("=" * 80)

        if args.op in ("shuffle", "all"):
            result = simulate_shuffle_analytical(
                indptr, n_rows, memory_limit,
                src_chunk_size, dst_chunk_size,
                label=f"SHUFFLE ({n_rows:,} rows, {mem_gb}G)",
                verbose=args.verbose,
            )
            result["op"] = "shuffle"
            result["mem_gb"] = mem_gb
            summary_rows.append(result)

        if args.op in ("truncate", "all"):
            trunc_n = min(n_rows, 10_000_000)
            src = np.arange(trunc_n, dtype=np.int64)
            result = simulate_sparse_scatter(
                indptr, src, memory_limit,
                src_chunk_size, dst_chunk_size,
                label=f"TRUNCATE (first {trunc_n:,} rows, {mem_gb}G)",
                verbose=args.verbose,
            )
            result["op"] = "truncate"
            result["mem_gb"] = mem_gb
            summary_rows.append(result)

        if args.op in ("split", "all"):
            try:
                codes = load_obs_column(args.split_column, n_rows)
                unique_vals = np.unique(codes)
                n_groups = len(unique_vals)
                group_sizes = np.array([int(np.sum(codes == v)) for v in unique_vals])
                print(f"\n  Split column '{args.split_column}': {n_groups} groups")
                print(f"  Group sizes: min={group_sizes.min():,} max={group_sizes.max():,} "
                      f"median={int(np.median(group_sizes)):,}")

                total_split_read = 0.0
                total_split_write = 0.0
                total_split_batches = 0
                total_split_sub = 0
                total_split_dst = 0
                for store_id, val in enumerate(unique_vals):
                    mask = codes == val
                    g_src = np.where(mask)[0].astype(np.int64)
                    show = store_id < 3 or store_id == len(unique_vals) - 1
                    result = simulate_sparse_scatter(
                        indptr, g_src, memory_limit,
                        src_chunk_size, dst_chunk_size,
                        label=f"SPLIT group {store_id}/{n_groups} ({len(g_src):,} rows, {mem_gb}G)",
                        verbose=args.verbose and show,
                    )
                    total_split_read += result["read_gib"]
                    total_split_write += result["write_gib"]
                    total_split_batches += result["batches"]
                    total_split_sub += result["sub_runs"]
                    total_split_dst += result["dst_chunks"]
                    if store_id == 2 and n_groups > 4:
                        print(f"  ... ({n_groups - 4} more groups omitted, showing last) ...")

                print(f"\n  SPLIT TOTAL ({n_groups} groups):")
                print(f"    READ:  {total_split_read:.2f} GiB")
                print(f"    WRITE: {total_split_write:.2f} GiB")
                print(f"    TOTAL: {total_split_read + total_split_write:.2f} GiB")
                print(f"    Batches: {total_split_batches}")

                summary_rows.append({
                    "op": "split",
                    "mem_gb": mem_gb,
                    "n_rows": n_rows,
                    "batches": total_split_batches,
                    "merged_runs": 0,
                    "sub_runs": total_split_sub,
                    "dst_chunks": total_split_dst,
                    "read_gib": total_split_read,
                    "write_gib": total_split_write,
                    "total_gib": total_split_read + total_split_write,
                    "read_amp": total_split_read / total_split_write if total_split_write > 0 else 0,
                    "batch_details": [],
                })
            except Exception as e:
                print(f"  Split simulation failed: {e}")

    if len(summary_rows) > 1:
        print()
        print("=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Op':<10} {'MemGB':>6} {'Rows':>12} {'Batches':>8} "
              f"{'SubRuns':>8} {'DstChks':>8} "
              f"{'ReadGiB':>8} {'WriteGiB':>9} {'TotalGiB':>9} {'ReadAmp':>8}")
        for r in summary_rows:
            print(f"{r['op']:<10} {r['mem_gb']:>6.0f} {r['n_rows']:>12,} {r['batches']:>8} "
                  f"{r['sub_runs']:>8,} {r['dst_chunks']:>8,} "
                  f"{r['read_gib']:>8.2f} {r['write_gib']:>9.2f} {r['total_gib']:>9.2f} "
                  f"{r['read_amp']:>8.2f}")


if __name__ == "__main__":
    main()
