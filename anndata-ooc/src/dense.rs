use anyhow::{Result, bail};
use zarrs::array::{Array, ArrayBytes, ArraySubset};
use zarrs::storage::ReadableWritableListableStorageTraits;

use crate::budget::BufferPool;
use crate::planner::{ShardPlanner, ShardPass};

/// Out-of-core dense matrix permuter using partition-buffer strategy.
///
/// For each pass of output chunks that fit in memory:
/// 1. Allocate a buffer for each output chunk in the pass
/// 2. Scan source rows in sequential order, scattering into chunk buffers
/// 3. Flush each complete chunk buffer via `store_chunk` (no read-modify-write)
///
/// This matches the standard DB approach for partitioned external writes.
pub struct DensePermuter {
    pool: BufferPool,
}

impl DensePermuter {
    pub fn new(pool: BufferPool) -> Self {
        Self { pool }
    }

    /// Permute a dense 2D zarrs array.
    ///
    /// `permutation[output_row] = source_row` defines the reordering.
    /// The destination array must already be created with the correct shape.
    pub fn permute<S>(
        &self,
        src: &Array<S>,
        dst: &Array<S>,
        permutation: &[usize],
    ) -> Result<()>
    where
        S: ReadableWritableListableStorageTraits + ?Sized + 'static,
    {
        let src_shape = src.shape();
        if src_shape.len() != 2 {
            bail!("DensePermuter only supports 2D arrays, got {}D", src_shape.len());
        }
        let n_cols = src_shape[1] as usize;
        let n_output = permutation.len();

        let elem_size = src.data_type().fixed_size()
            .ok_or_else(|| anyhow::anyhow!("variable-length data types not supported"))?;

        let dst_shape = dst.shape();
        let chunk_grid_shape = dst.chunk_grid_shape();
        let chunk_size_row = if chunk_grid_shape.is_empty() {
            n_output
        } else {
            let first_chunk_shape = dst.chunk_shape(&vec![0u64; dst_shape.len()])?;
            first_chunk_shape[0].get() as usize
        };

        let headroom = 4 * 1024 * 1024;
        let max_rows = self.pool.max_rows_for_dense(n_cols, elem_size, headroom).max(chunk_size_row);

        let passes = ShardPlanner::plan(permutation, chunk_size_row, n_output, max_rows);

        log::info!(
            "DensePermuter: {} output rows x {} cols, elem_size={}, chunk_rows={}, {} passes",
            n_output, n_cols, elem_size, chunk_size_row, passes.len()
        );

        for (pass_idx, pass) in passes.iter().enumerate() {
            log::debug!("Pass {}/{}: {} chunks, {} rows",
                pass_idx + 1, passes.len(), pass.chunks.len(), pass.total_rows);

            self.process_pass(src, dst, pass, n_cols, chunk_size_row)?;
        }

        Ok(())
    }

    fn process_pass<S>(
        &self,
        src: &Array<S>,
        dst: &Array<S>,
        pass: &ShardPass,
        n_cols: usize,
        chunk_size_row: usize,
    ) -> Result<()>
    where
        S: ReadableWritableListableStorageTraits + ?Sized + 'static,
    {
        let elem_size = src.data_type().fixed_size().unwrap_or(8);

        // Allocate output-chunk buffers. Each buffer holds the full chunk.
        let mut chunk_buffers: Vec<ChunkBuffer> = pass.chunks.iter().map(|ca| {
            let n_rows = ca.row_range.len();
            ChunkBuffer {
                chunk_idx: ca.chunk_idx,
                row_start: ca.row_range.start,
                n_rows,
                data: vec![0u8; n_rows * n_cols * elem_size],
                filled: vec![false; n_rows],
            }
        }).collect();

        // Build a lookup: output_row -> (buffer_index, local_row_in_buffer)
        let mut out_row_to_buf: Vec<(usize, usize)> = Vec::new();
        for (buf_idx, ca) in pass.chunks.iter().enumerate() {
            for entry in &ca.entries {
                out_row_to_buf.push((entry.output_row, buf_idx));
            }
        }

        // Get source-sorted reads for sequential I/O
        let reads = ShardPlanner::source_sorted_reads(pass);

        // Merge contiguous source row ranges for large sequential reads.
        let merged_runs = merge_contiguous_source_reads(&reads, 64);

        for run in &merged_runs {
            let src_start = run.src_start as u64;
            let src_len = run.count as u64;

            let subset = ArraySubset::new_with_ranges(
                &[src_start..src_start + src_len, 0..n_cols as u64],
            );
            let array_bytes: ArrayBytes<'static> = src.retrieve_array_subset(&subset)?;
            let block = array_bytes.into_fixed()?.into_owned();

            let row_bytes = n_cols * elem_size;

            for &(src_row, out_row) in &run.pairs {
                let local_src = src_row - run.src_start;
                let src_offset = local_src * row_bytes;
                let src_slice = &block[src_offset..src_offset + row_bytes];

                // Find which buffer this output row goes to
                let buf_idx = {
                    let chunk_idx_for_row = out_row / chunk_size_row;
                    pass.chunks.iter().position(|c| c.chunk_idx == chunk_idx_for_row as u64)
                        .expect("output row must map to a chunk in this pass")
                };
                let buf = &mut chunk_buffers[buf_idx];
                let local_row = out_row - buf.row_start;
                let dst_offset = local_row * row_bytes;
                buf.data[dst_offset..dst_offset + row_bytes].copy_from_slice(src_slice);
                buf.filled[local_row] = true;
            }
        }

        // Flush each chunk buffer. Full-size chunks use store_chunk (fast path,
        // no decode needed). Partial chunks (last chunk) use store_array_subset.
        for buf in &chunk_buffers {
            let row_start = buf.row_start as u64;
            let row_end = (buf.row_start + buf.n_rows) as u64;
            let write_subset = ArraySubset::new_with_ranges(
                &[row_start..row_end, 0..n_cols as u64],
            );

            log::debug!("Flushing rows {}..{} (chunk {})", row_start, row_end, buf.chunk_idx);

            let bytes = ArrayBytes::from(buf.data.clone());
            dst.store_array_subset(&write_subset, bytes)?;
        }

        Ok(())
    }
}

struct ChunkBuffer {
    chunk_idx: u64,
    row_start: usize,
    n_rows: usize,
    data: Vec<u8>,
    filled: Vec<bool>,
}

/// A merged run of contiguous source rows.
struct MergedSourceRun {
    src_start: usize,
    count: usize,
    /// (source_row, output_row) pairs within this run.
    pairs: Vec<(usize, usize)>,
}

/// Merge source-sorted (src_row, out_row) pairs into contiguous runs.
/// Pairs within `gap` rows of each other are merged into a single read.
fn merge_contiguous_source_reads(
    sorted_pairs: &[(usize, usize)],
    gap: usize,
) -> Vec<MergedSourceRun> {
    if sorted_pairs.is_empty() {
        return Vec::new();
    }

    let mut runs = Vec::new();
    let (first_src, first_out) = sorted_pairs[0];
    let mut cur_start = first_src;
    let mut cur_end = first_src + 1;
    let mut cur_pairs = vec![(first_src, first_out)];

    for &(src, out) in &sorted_pairs[1..] {
        if src <= cur_end + gap {
            cur_end = cur_end.max(src + 1);
            cur_pairs.push((src, out));
        } else {
            runs.push(MergedSourceRun {
                src_start: cur_start,
                count: cur_end - cur_start,
                pairs: std::mem::take(&mut cur_pairs),
            });
            cur_start = src;
            cur_end = src + 1;
            cur_pairs = vec![(src, out)];
        }
    }

    runs.push(MergedSourceRun {
        src_start: cur_start,
        count: cur_end - cur_start,
        pairs: cur_pairs,
    });

    runs
}
