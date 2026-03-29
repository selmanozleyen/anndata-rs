use anyhow::{Result, bail};
use rayon::prelude::*;
use zarrs::array::{Array, ArrayBytes, ArraySubset};
use zarrs::storage::ReadableWritableListableStorageTraits;

use crate::budget::BufferPool;
use crate::scatter::{ScatterPlanner, ScatterPass, RowAssignment};

/// Out-of-core dense matrix scatterer supporting multiple output stores.
///
/// Generalizes DensePermuter: given a source 2D array and N destination arrays,
/// scatters source rows into the correct destination chunk buffers, then flushes.
/// Source reads are sorted for sequential I/O. Each pass processes as many
/// output chunks (across all stores) as fit in the memory budget.
pub struct DenseScatterer {
    pool: BufferPool,
}

impl DenseScatterer {
    pub fn new(pool: BufferPool) -> Self {
        Self { pool }
    }

    /// Scatter a dense 2D source array into one or more destination arrays.
    ///
    /// `dsts[i]` is the zarrs Array for store i.
    /// `assignments` maps each source row to a (store_id, output_row).
    /// `store_n_rows[i]` is the total row count for destination i.
    pub fn scatter<S>(
        &self,
        src: &Array<S>,
        dsts: &[&Array<S>],
        assignments: &[RowAssignment],
        store_n_rows: &[usize],
    ) -> Result<()>
    where
        S: ReadableWritableListableStorageTraits + ?Sized + 'static,
    {
        let src_shape = src.shape();
        if src_shape.len() != 2 {
            bail!("DenseScatterer only supports 2D arrays, got {}D", src_shape.len());
        }
        let n_cols = src_shape[1] as usize;

        let elem_size = src.data_type().fixed_size()
            .ok_or_else(|| anyhow::anyhow!("variable-length data types not supported"))?;

        // Determine chunk sizes per store from destination array metadata
        let mut store_chunk_sizes: Vec<usize> = Vec::with_capacity(dsts.len());
        for (i, dst) in dsts.iter().enumerate() {
            let dst_shape = dst.shape();
            let chunk_grid_shape = dst.chunk_grid_shape();
            let cs = if chunk_grid_shape.is_empty() {
                store_n_rows[i]
            } else {
                let first_chunk_shape = dst.chunk_shape(&vec![0u64; dst_shape.len()])?;
                first_chunk_shape[0].get() as usize
            };
            store_chunk_sizes.push(cs);
        }

        let headroom = 4 * 1024 * 1024;
        let min_chunk = store_chunk_sizes.iter().copied().min().unwrap_or(1);
        let max_rows = self.pool.max_rows_for_dense(n_cols, elem_size, headroom).max(min_chunk);

        let passes = ScatterPlanner::plan(assignments, &store_chunk_sizes, store_n_rows, max_rows);

        log::info!(
            "DenseScatterer: {} stores, {} total assignments, {} cols, elem_size={}, {} passes",
            dsts.len(), assignments.len(), n_cols, elem_size, passes.len()
        );

        for (pass_idx, pass) in passes.iter().enumerate() {
            log::debug!("Pass {}/{}: {} chunks, {} rows",
                pass_idx + 1, passes.len(), pass.chunks.len(), pass.total_rows);

            self.process_pass(src, dsts, pass, n_cols)?;
        }

        Ok(())
    }

    fn process_pass<S>(
        &self,
        src: &Array<S>,
        dsts: &[&Array<S>],
        pass: &ScatterPass,
        n_cols: usize,
    ) -> Result<()>
    where
        S: ReadableWritableListableStorageTraits + ?Sized + 'static,
    {
        let elem_size = src.data_type().fixed_size().unwrap_or(8);
        let row_bytes = n_cols * elem_size;

        // Allocate a buffer per chunk in this pass
        struct ChunkBuf {
            store_id: u16,
            row_start: usize,
            n_rows: usize,
            data: Vec<u8>,
        }

        let mut chunk_buffers: Vec<ChunkBuf> = pass.chunks.iter().map(|sc| {
            let n_rows = sc.row_range.len();
            ChunkBuf {
                store_id: sc.store_id,
                row_start: sc.row_range.start,
                n_rows,
                data: vec![0u8; n_rows * row_bytes],
            }
        }).collect();

        // Get source-sorted reads
        let reads = ScatterPlanner::source_sorted_reads(pass);

        // Merge contiguous source rows for large sequential reads
        let merged_runs = merge_contiguous_source_reads(&reads, 256);

        for run in &merged_runs {
            let src_start = run.src_start as u64;
            let src_len = run.count as u64;

            let subset = ArraySubset::new_with_ranges(
                &[src_start..src_start + src_len, 0..n_cols as u64],
            );
            let array_bytes: ArrayBytes<'static> = src.retrieve_array_subset(&subset)?;
            let block = array_bytes.into_fixed()?.into_owned();

            for &(src_row, packed) in &run.pairs {
                let pass_chunk_idx = packed >> 32;
                let local_row = packed & 0xFFFF_FFFF;

                let local_src = src_row - run.src_start;
                let src_offset = local_src * row_bytes;
                let src_slice = &block[src_offset..src_offset + row_bytes];

                let buf = &mut chunk_buffers[pass_chunk_idx];
                let dst_offset = local_row * row_bytes;
                buf.data[dst_offset..dst_offset + row_bytes].copy_from_slice(src_slice);
            }
        }

        chunk_buffers.into_par_iter().try_for_each(|buf| -> Result<()> {
            let dst = dsts[buf.store_id as usize];
            let row_start = buf.row_start as u64;
            let row_end = (buf.row_start + buf.n_rows) as u64;
            let write_subset = ArraySubset::new_with_ranges(
                &[row_start..row_end, 0..n_cols as u64],
            );
            dst.store_array_subset(&write_subset, ArrayBytes::from(buf.data))?;
            Ok(())
        })?;

        Ok(())
    }
}

struct MergedSourceRun {
    src_start: usize,
    count: usize,
    pairs: Vec<(usize, usize)>,
}

fn merge_contiguous_source_reads(
    sorted_pairs: &[(usize, usize)],
    gap: usize,
) -> Vec<MergedSourceRun> {
    if sorted_pairs.is_empty() {
        return Vec::new();
    }

    let mut runs = Vec::new();
    let (first_src, first_packed) = sorted_pairs[0];
    let mut cur_start = first_src;
    let mut cur_end = first_src + 1;
    let mut cur_pairs = vec![(first_src, first_packed)];

    for &(src, packed) in &sorted_pairs[1..] {
        if src <= cur_end + gap {
            cur_end = cur_end.max(src + 1);
            cur_pairs.push((src, packed));
        } else {
            runs.push(MergedSourceRun {
                src_start: cur_start,
                count: cur_end - cur_start,
                pairs: std::mem::take(&mut cur_pairs),
            });
            cur_start = src;
            cur_end = src + 1;
            cur_pairs = vec![(src, packed)];
        }
    }

    runs.push(MergedSourceRun {
        src_start: cur_start,
        count: cur_end - cur_start,
        pairs: cur_pairs,
    });

    runs
}
