"""Tests for anndata_rs out-of-core scatter engine.

Tests permutation, splitting, categorical obs columns, duplicate indices,
and subset operations via anndata_rs.permute / split / scatter.
"""

import json
import os
from pathlib import Path

import anndata as ad
import anndata_rs
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, issparse


@pytest.fixture
def zarr_pair(tmp_path):
    """Yield (src_path, dst_path) for a test, cleaned up afterwards."""
    src = tmp_path / "src.zarr"
    dst = tmp_path / "dst.zarr"
    yield str(src), str(dst)


def _make_adata(path, X, obs=None, var=None):
    n_obs = X.shape[0] if hasattr(X, 'shape') else X.toarray().shape[0]
    n_vars = X.shape[1] if hasattr(X, 'shape') else X.toarray().shape[1]
    if obs is None:
        obs = pd.DataFrame(
            {"_dummy": np.zeros(n_obs, dtype=np.int8)},
            index=[f"cell_{i}" for i in range(n_obs)],
        )
    if var is None:
        var = pd.DataFrame(
            {"_dummy": np.zeros(n_vars, dtype=np.int8)},
            index=[f"gene_{i}" for i in range(n_vars)],
        )
    adata = anndata_rs.AnnData(X=X, obs=obs, var=var, filename=path, backend="zarr")
    adata.close()
    return adata


class TestDensePermute:
    def test_identity(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(0)
        X = np.random.randn(100, 30).astype(np.float32)
        _make_adata(src, X)

        perm = np.arange(100, dtype=np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        np.testing.assert_allclose(result.X, X, atol=1e-6)

    def test_reverse(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(1)
        X = np.random.randn(80, 25).astype(np.float32)
        _make_adata(src, X)

        perm = np.arange(80, dtype=np.int64)[::-1].copy()
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        np.testing.assert_allclose(result.X, X[perm], atol=1e-6)

    def test_random_permutation(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(2)
        X = np.random.randn(200, 50).astype(np.float32)
        _make_adata(src, X)

        perm = np.random.permutation(200).astype(np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        np.testing.assert_allclose(result.X, X[perm], atol=1e-6)

    def test_float64(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(3)
        X = np.random.randn(60, 15).astype(np.float64)
        _make_adata(src, X)

        perm = np.random.permutation(60).astype(np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        np.testing.assert_allclose(result.X, X[perm], atol=1e-12)

    def test_int32(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(4)
        X = np.random.randint(0, 1000, (50, 20)).astype(np.int32)
        _make_adata(src, X)

        perm = np.random.permutation(50).astype(np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        np.testing.assert_array_equal(result.X, X[perm])


class TestSparsePermute:
    def test_csr_random(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(10)
        dense = np.random.randn(300, 80).astype(np.float32)
        dense[dense < 0.5] = 0
        X = csr_matrix(dense)
        _make_adata(src, X)

        perm = np.random.permutation(300).astype(np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        expected = dense[perm]
        actual = result.X.toarray() if issparse(result.X) else result.X
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_csr_reverse(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(11)
        dense = np.random.randn(100, 40).astype(np.float32)
        dense[dense < 1.0] = 0
        X = csr_matrix(dense)
        _make_adata(src, X)

        perm = np.arange(100, dtype=np.int64)[::-1].copy()
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        expected = dense[perm]
        actual = result.X.toarray() if issparse(result.X) else result.X
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_csr_very_sparse(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(12)
        dense = np.zeros((200, 100), dtype=np.float32)
        for i in range(200):
            nnz = np.random.randint(0, 5)
            cols = np.random.choice(100, nnz, replace=False)
            dense[i, cols] = np.random.randn(nnz).astype(np.float32)
        X = csr_matrix(dense)
        _make_adata(src, X)

        perm = np.random.permutation(200).astype(np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        expected = dense[perm]
        actual = result.X.toarray() if issparse(result.X) else result.X
        np.testing.assert_allclose(actual, expected, atol=1e-6)


class TestObsPermute:
    def test_obs_index_permuted(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(20)
        n = 100
        X = np.random.randn(n, 20).astype(np.float32)
        obs = pd.DataFrame({
            "cell_type": np.random.choice(["T", "B", "NK", "Mono"], n),
            "score": np.random.randn(n).astype(np.float32),
        }, index=[f"cell_{i}" for i in range(n)])
        _make_adata(src, X, obs=obs)

        perm = np.random.permutation(n).astype(np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        src_ad = ad.read_zarr(src)

        for i in range(n):
            assert result.obs.index[i] == src_ad.obs.index[perm[i]]

    def test_obs_columns_permuted(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(21)
        n = 80
        X = np.random.randn(n, 15).astype(np.float32)
        obs = pd.DataFrame({
            "cluster": np.random.choice(["A", "B", "C"], n),
            "total_counts": np.random.randint(100, 10000, n),
        }, index=[f"c_{i}" for i in range(n)])
        _make_adata(src, X, obs=obs)

        perm = np.random.permutation(n).astype(np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        src_ad = ad.read_zarr(src)

        for col in ["cluster"]:
            expected = [src_ad.obs[col].iloc[perm[i]] for i in range(n)]
            actual = list(result.obs[col])
            assert actual == expected, f"Column {col} mismatch"


class TestCategoricalObs:
    """Verify categorical obs columns survive permute and split."""

    def test_permute_categorical(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(100)
        n = 60
        X = np.random.randn(n, 10).astype(np.float32)
        obs = pd.DataFrame({
            "cell_type": pd.Categorical(
                np.random.choice(["T", "B", "NK", "Mono"], n)
            ),
            "score": np.random.randn(n).astype(np.float32),
        }, index=[f"cell_{i}" for i in range(n)])
        _make_adata(src, X, obs=obs)

        perm = np.random.permutation(n).astype(np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        src_ad = ad.read_zarr(src)

        np.testing.assert_allclose(result.X, X[perm], atol=1e-6)
        expected_types = src_ad.obs["cell_type"].values[perm]
        actual_types = result.obs["cell_type"].values
        assert list(actual_types) == list(expected_types)

    def test_split_categorical(self, tmp_path):
        src = str(tmp_path / "src.zarr")
        np.random.seed(101)
        n = 90
        X = np.random.randn(n, 15).astype(np.float32)
        types = pd.Categorical(["T", "B", "NK"] * 30)
        obs = pd.DataFrame({
            "cell_type": types,
            "val": np.random.randn(n).astype(np.float32),
        }, index=[f"c_{i}" for i in range(n)])
        _make_adata(src, X, obs=obs)

        out_dir = str(tmp_path / "split_cat")
        groups = anndata_rs.split(src, out_dir, "cell_type", obs=obs)

        total = 0
        for val, path in groups:
            result = ad.read_zarr(path)
            mask = obs["cell_type"] == val
            expected_X = X[mask.values]
            np.testing.assert_allclose(result.X, expected_X, atol=1e-6)
            total += result.X.shape[0]
        assert total == n


class TestVarUnchanged:
    def test_var_preserved(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(30)
        n_obs, n_vars = 60, 25
        X = np.random.randn(n_obs, n_vars).astype(np.float32)
        var = pd.DataFrame({
            "gene_name": [f"gene_{i}" for i in range(n_vars)],
            "highly_variable": np.random.choice([True, False], n_vars),
        }, index=[f"var_{i}" for i in range(n_vars)])
        _make_adata(src, X, var=var)

        perm = np.random.permutation(n_obs).astype(np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        src_ad = ad.read_zarr(src)

        assert list(result.var.index) == list(src_ad.var.index)
        assert list(result.var["gene_name"]) == list(src_ad.var["gene_name"])


class TestChunkSize:
    def test_custom_chunk_size(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(40)
        X = np.random.randn(500, 100).astype(np.float32)
        _make_adata(src, X)

        perm = np.random.permutation(500).astype(np.int64)
        anndata_rs.permute(src, dst, perm, chunk_size=256)

        with open(os.path.join(dst, "X", "zarr.json")) as f:
            d = json.load(f)
            for codec in d.get("codecs", []):
                if codec["name"] == "sharding_indexed":
                    sub_chunk = codec["configuration"]["chunk_shape"]
                    assert sub_chunk[0] == 256, f"Expected sub-chunk rows=256, got {sub_chunk[0]}"
                    break

        result = ad.read_zarr(dst)
        np.testing.assert_allclose(result.X, X[perm], atol=1e-6)

    def test_chunk_size_larger_than_array(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(41)
        X = np.random.randn(50, 20).astype(np.float32)
        _make_adata(src, X)

        perm = np.random.permutation(50).astype(np.int64)
        anndata_rs.permute(src, dst, perm, chunk_size=1024)

        result = ad.read_zarr(dst)
        np.testing.assert_allclose(result.X, X[perm], atol=1e-6)


class TestShardSize:
    @staticmethod
    def _get_zarr_shapes(dst):
        with open(os.path.join(dst, "X", "zarr.json")) as f:
            d = json.load(f)
        shard = d["chunk_grid"]["configuration"]["chunk_shape"]
        sub_chunk = None
        for c in d.get("codecs", []):
            if c["name"] == "sharding_indexed":
                sub_chunk = c["configuration"]["chunk_shape"]
        return shard, sub_chunk

    def test_explicit_shard_size(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(60)
        X = np.random.randn(500, 100).astype(np.float32)
        _make_adata(src, X)

        perm = np.random.permutation(500).astype(np.int64)
        anndata_rs.permute(src, dst, perm, chunk_size=128, shard_size=512)

        shard, sub_chunk = self._get_zarr_shapes(dst)
        assert sub_chunk[0] == 128
        assert shard[0] == 512

        result = ad.read_zarr(dst)
        np.testing.assert_allclose(result.X, X[perm], atol=1e-6)

    def test_target_shard_bytes_1mb(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(61)
        n_obs, n_vars = 1000, 200
        X = np.random.randn(n_obs, n_vars).astype(np.float32)
        _make_adata(src, X)

        perm = np.random.permutation(n_obs).astype(np.int64)
        anndata_rs.permute(src, dst, perm, chunk_size=128,
                 target_shard_bytes=1 * 1024 * 1024)

        shard, sub_chunk = self._get_zarr_shapes(dst)
        shard_bytes = shard[0] * shard[1] * 4
        assert abs(shard_bytes - 1024 * 1024) < 1024 * 1024, \
            f"Shard bytes {shard_bytes} not close to 1 MB"

        result = ad.read_zarr(dst)
        np.testing.assert_allclose(result.X, X[perm], atol=1e-6)

    def test_target_shard_bytes_4mb(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(62)
        n_obs, n_vars = 1000, 200
        X = np.random.randn(n_obs, n_vars).astype(np.float32)
        _make_adata(src, X)

        perm = np.random.permutation(n_obs).astype(np.int64)
        anndata_rs.permute(src, dst, perm, chunk_size=128,
                 target_shard_bytes=4 * 1024 * 1024)

        shard, sub_chunk = self._get_zarr_shapes(dst)
        shard_bytes = shard[0] * shard[1] * 4
        target = 4 * 1024 * 1024
        assert shard_bytes >= target, \
            f"Shard bytes {shard_bytes} < target {target}"
        assert shard_bytes < target * 1.1, \
            f"Shard bytes {shard_bytes} too far above target {target}"

        result = ad.read_zarr(dst)
        np.testing.assert_allclose(result.X, X[perm], atol=1e-6)

    def test_target_overrides_shard_size(self, zarr_pair):
        """target_shard_bytes takes priority over shard_size."""
        src, dst = zarr_pair
        np.random.seed(63)
        n_obs, n_vars = 800, 150
        X = np.random.randn(n_obs, n_vars).astype(np.float32)
        _make_adata(src, X)

        perm = np.random.permutation(n_obs).astype(np.int64)
        anndata_rs.permute(src, dst, perm, chunk_size=128,
                 shard_size=256, target_shard_bytes=4 * 1024 * 1024)

        shard, _ = self._get_zarr_shapes(dst)
        shard_bytes = shard[0] * shard[1] * 4
        assert shard_bytes >= 4 * 1024 * 1024, \
            f"target_shard_bytes should override shard_size, got {shard_bytes}"

        result = ad.read_zarr(dst)
        np.testing.assert_allclose(result.X, X[perm], atol=1e-6)


class TestAnnDataRsRoundtrip:
    """Verify anndata_rs.read() can open the permuted output."""

    def test_read_dense(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(50)
        n = 100
        X = np.random.randn(n, 30).astype(np.float32)
        obs = pd.DataFrame({
            "score": np.random.randn(n).astype(np.float32),
        }, index=[f"cell_{i}" for i in range(n)])
        _make_adata(src, X, obs=obs)

        perm = np.random.permutation(n).astype(np.int64)
        anndata_rs.permute(src, dst, perm)

        result = anndata_rs.read(dst, backed="r", backend="zarr")
        assert result.shape == (n, 30)
        R = np.array(result.X[:])
        np.testing.assert_allclose(R, X[perm], atol=1e-6)
        result.close()

    def test_read_sparse(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(51)
        n = 150
        dense = np.random.randn(n, 40).astype(np.float32)
        dense[dense < 0.5] = 0
        X = csr_matrix(dense)
        _make_adata(src, X)

        perm = np.random.permutation(n).astype(np.int64)
        anndata_rs.permute(src, dst, perm)

        result = anndata_rs.read(dst, backed="r", backend="zarr")
        assert result.shape == (n, 40)
        R = np.array(result.X[:].todense())
        np.testing.assert_allclose(R, dense[perm], atol=1e-6)
        result.close()

    def test_cross_library_agreement(self, zarr_pair):
        """anndata_rs.read() and anndata.read_zarr() should return identical X."""
        src, dst = zarr_pair
        np.random.seed(52)
        n = 120
        X = np.random.randn(n, 35).astype(np.float32)
        obs = pd.DataFrame({
            "group": np.random.choice(["X", "Y", "Z"], n),
            "val": np.random.randn(n).astype(np.float32),
        }, index=[f"c_{i}" for i in range(n)])
        _make_adata(src, X, obs=obs)

        perm = np.random.permutation(n).astype(np.int64)
        anndata_rs.permute(src, dst, perm)

        rs_result = anndata_rs.read(dst, backed="r", backend="zarr")
        py_result = ad.read_zarr(dst)

        R_rs = np.array(rs_result.X[:])
        R_py = py_result.X
        np.testing.assert_allclose(R_rs, R_py, atol=1e-12)
        rs_result.close()


class TestDuplicateIndices:
    """Permutation with duplicate source rows (copying data)."""

    def test_duplicate_rows_dense(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(70)
        X = np.random.randn(50, 20).astype(np.float32)
        _make_adata(src, X)

        perm = np.array([0, 0, 0, 1, 1, 2, 3, 3] + list(range(50)) + [49]*22,
                        dtype=np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        assert result.X.shape[0] == len(perm)
        np.testing.assert_allclose(result.X, X[perm], atol=1e-6)

    def test_duplicate_rows_sparse(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(71)
        dense = np.random.randn(40, 30).astype(np.float32)
        dense[dense < 0.5] = 0
        X = csr_matrix(dense)
        _make_adata(src, X)

        perm = np.array([5, 5, 5, 10, 10, 0] + list(range(40)),
                        dtype=np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        actual = result.X.toarray() if issparse(result.X) else result.X
        np.testing.assert_allclose(actual, dense[perm], atol=1e-6)


class TestSubsetPermute:
    """Permutation that discards rows (output smaller than input)."""

    def test_subset_dense(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(80)
        X = np.random.randn(200, 30).astype(np.float32)
        _make_adata(src, X)

        perm = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        assert result.X.shape[0] == 5
        np.testing.assert_allclose(result.X, X[perm], atol=1e-6)

    def test_subset_sparse(self, zarr_pair):
        src, dst = zarr_pair
        np.random.seed(81)
        dense = np.random.randn(100, 40).astype(np.float32)
        dense[dense < 0.5] = 0
        X = csr_matrix(dense)
        _make_adata(src, X)

        perm = np.random.choice(100, 20, replace=False).astype(np.int64)
        anndata_rs.permute(src, dst, perm)

        result = ad.read_zarr(dst)
        actual = result.X.toarray() if issparse(result.X) else result.X
        assert actual.shape[0] == 20
        np.testing.assert_allclose(actual, dense[perm], atol=1e-6)


class TestSplit:
    """Tests for scatter-based split by obs column."""

    @pytest.fixture
    def split_src(self, tmp_path):
        """Create a source AnnData with a cell_type column."""
        src = tmp_path / "src.zarr"
        np.random.seed(90)
        n = 120
        X = np.random.randn(n, 30).astype(np.float32)
        types = np.array(["T_cell", "B_cell", "NK_cell"] * 40)
        obs = pd.DataFrame({
            "cell_type": types,
            "score": np.random.randn(n).astype(np.float32),
        }, index=[f"cell_{i}" for i in range(n)])
        _make_adata(str(src), X, obs=obs)
        return str(src), tmp_path, X, obs

    def test_split_by_column(self, split_src, tmp_path):
        src, base, X, obs = split_src
        out_dir = str(tmp_path / "split_out")

        groups = anndata_rs.split(src, out_dir, "cell_type", obs=obs)

        assert len(groups) > 0
        total_rows = 0
        for value, path in groups:
            result = ad.read_zarr(path)
            n_expected = (obs["cell_type"] == value).sum()
            assert result.X.shape[0] == n_expected, \
                f"Group '{value}': expected {n_expected} rows, got {result.X.shape[0]}"
            assert result.X.shape[1] == 30

            mask = obs["cell_type"] == value
            expected_X = X[mask.values]
            np.testing.assert_allclose(result.X, expected_X, atol=1e-6)
            total_rows += result.X.shape[0]

        assert total_rows == 120, f"Total rows {total_rows} != 120"

    def test_split_preserves_var(self, split_src, tmp_path):
        src, base, X, obs = split_src
        out_dir = str(tmp_path / "split_var")

        groups = anndata_rs.split(src, out_dir, "cell_type", obs=obs)

        src_ad = ad.read_zarr(src)
        for _, path in groups:
            result = ad.read_zarr(path)
            assert list(result.var.index) == list(src_ad.var.index)

    def test_split_obs_columns(self, split_src, tmp_path):
        src, base, X, obs = split_src
        out_dir = str(tmp_path / "split_obs")

        groups = anndata_rs.split(src, out_dir, "cell_type", obs=obs)

        for value, path in groups:
            result = ad.read_zarr(path)
            mask = obs["cell_type"] == value
            expected_scores = obs.loc[mask, "score"].values
            np.testing.assert_allclose(
                result.obs["score"].values, expected_scores, atol=1e-6,
            )
