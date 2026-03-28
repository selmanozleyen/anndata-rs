use std::path::Path;

use anyhow::Result;

use crate::scatter::ScatterPlanner;
use crate::scatter_engine::{scatter_anndata, ScatterConfig, OutputStoreConfig};

/// Configuration for out-of-core permutation.
pub struct PermuteConfig {
    /// Maximum RAM (in bytes) the engine may use for buffers.
    pub memory_limit: usize,
    /// Sub-chunk size (rows per sub-chunk) for output arrays along axis 0.
    /// When None, the backend default is used (min(n, 128) for 2-D numeric).
    pub chunk_size: Option<usize>,
    /// Shard size (rows per shard) for output arrays along axis 0.
    /// Must be a multiple of chunk_size. When None, defaults to chunk_size * 8.
    /// Ignored when target_shard_bytes is set.
    pub shard_size: Option<usize>,
    /// Target shard size in bytes. When set, the engine auto-calculates the
    /// shard row count so that each shard is approximately this many bytes.
    /// This is the most Lustre-friendly option: set it to the stripe size
    /// (e.g. 4 * 1024 * 1024 for 4 MB stripes). Overrides shard_size.
    pub target_shard_bytes: Option<usize>,
}

impl Default for PermuteConfig {
    fn default() -> Self {
        Self {
            memory_limit: 2 * 1024 * 1024 * 1024, // 2 GB
            chunk_size: None,
            shard_size: None,
            target_shard_bytes: None,
        }
    }
}

/// Permute an entire AnnData Zarr store out-of-core.
///
/// Given `src_path` (an existing .zarr AnnData), `dst_path` (where the output
/// is written), and `permutation` (output_row -> source_row), this function:
///
/// 1. Copies `var` metadata unchanged (column annotations are row-independent)
/// 2. Permutes `obs` (row annotations) by the given index
/// 3. Permutes `X` (the main data matrix -- dense or CSR/CSC)
/// 4. Permutes every array in `obsm`, `obsp`, `layers` along axis 0
/// 5. Copies `varm`, `varp`, `uns` unchanged
///
/// Supports duplicate source indices (row duplication) and shorter-than-input
/// permutations (row subsetting / discarding).
///
/// Internally delegates to the unified scatter engine with a single output
/// store, so permute, split, and scatter all share the same optimized I/O path.
pub fn permute_anndata(
    src_path: &Path,
    dst_path: &Path,
    permutation: &[usize],
    config: &PermuteConfig,
) -> Result<()> {
    let n_output = permutation.len();
    let assignments = ScatterPlanner::from_permutation(permutation);

    let scatter_config = ScatterConfig {
        memory_limit: config.memory_limit,
        chunk_size: config.chunk_size,
        shard_size: config.shard_size,
        target_shard_bytes: config.target_shard_bytes,
    };

    let outputs = vec![OutputStoreConfig {
        path: dst_path.to_path_buf(),
        n_rows: n_output,
    }];

    scatter_anndata(src_path, &outputs, &assignments, &scatter_config)
}
