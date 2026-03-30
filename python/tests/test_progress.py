"""Quick smoke test: create a small Zarr AnnData, scatter it, watch output grow.

Usage:
    python tests/test_progress.py

Requires only numpy, scipy, zarr -- no anndata/pandas/polars.
"""
import json
import os
import shutil
import tempfile
import threading
import time

import numpy as np
import scipy.sparse as sp
import zarr


def create_test_zarr(path: str, n_rows: int, n_cols: int, density: float = 0.01):
    """Create a minimal Zarr AnnData with only a CSR X matrix."""
    store = zarr.open(path, mode="w")

    rng = np.random.default_rng(42)
    X = sp.random(n_rows, n_cols, density=density, format="csr",
                  dtype=np.float32, random_state=rng)

    g = store.create_group("X")
    g.attrs["encoding-type"] = "csr_matrix"
    g.attrs["encoding-version"] = "0.1.0"
    g.attrs["shape"] = [n_rows, n_cols]

    g.create_array("data", data=X.data,
                    chunks=(min(len(X.data), 256_000),))
    g.create_array("indices", data=X.indices,
                    chunks=(min(len(X.indices), 256_000),))
    g.create_array("indptr", data=X.indptr,
                    chunks=(min(len(X.indptr), 16_384),))

    print(f"Created {path}")
    print(f"  shape: {n_rows} x {n_cols}")
    print(f"  nnz:   {X.nnz} ({X.nnz * 4 / 1e6:.1f} MB data)")
    return X


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


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from anndata_rs.anndata_rs import _scatter

    n_rows = 100_000
    n_cols = 5_000
    density = 0.01  # ~5M NNZ, ~20 MB

    tmpdir = tempfile.mkdtemp(prefix="scatter_test_")
    src_path = os.path.join(tmpdir, "src.zarr")
    dst_path = os.path.join(tmpdir, "dst.zarr")

    try:
        X = create_test_zarr(src_path, n_rows, n_cols, density)
        expected_nnz_bytes = X.nnz * 4 * 2  # data + indices, uncompressed

        perm = np.random.default_rng(123).permutation(n_rows).astype(np.int64)
        outputs = [(dst_path, perm)]

        error_box = [None]

        def run_scatter():
            try:
                _scatter(src_path, outputs, memory_limit=512 * 1024 * 1024)
            except Exception as e:
                error_box[0] = e

        t0 = time.time()
        thread = threading.Thread(target=run_scatter, daemon=True)
        thread.start()

        print(f"\nMonitoring output growth (expected ~{expected_nnz_bytes / 1e6:.0f} MB)...")
        while thread.is_alive():
            sz = dir_size(dst_path) if os.path.isdir(dst_path) else 0
            elapsed = time.time() - t0
            pct = min(sz / max(expected_nnz_bytes, 1) * 100, 100)
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(
                f"\r  [{bar}] {pct:5.1f}%  {sz / 1e6:7.1f} MB  {elapsed:5.1f}s",
                end="", flush=True,
            )
            time.sleep(0.2)

        thread.join()
        elapsed = time.time() - t0
        final_sz = dir_size(dst_path) if os.path.isdir(dst_path) else 0
        print(f"\r  [{'#' * 40}] 100.0%  {final_sz / 1e6:7.1f} MB  {elapsed:5.1f}s")

        if error_box[0]:
            print(f"\nERROR: {error_box[0]}")
            raise error_box[0]

        print(f"\nDone in {elapsed:.2f}s")
        print(f"  output size: {final_sz / 1e6:.1f} MB")

        # Quick correctness check: read back indptr and verify row count
        dst = zarr.open(dst_path, mode="r")
        dst_indptr = dst["X"]["indptr"][:]
        assert len(dst_indptr) == n_rows + 1, f"indptr len {len(dst_indptr)} != {n_rows + 1}"
        assert dst_indptr[-1] == X.nnz, f"total nnz {dst_indptr[-1]} != {X.nnz}"
        print("  indptr check: OK")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
