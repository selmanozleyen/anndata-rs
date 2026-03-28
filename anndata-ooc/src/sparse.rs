use anyhow::Result;
use zarrs::array::{Array, ArrayBytes, ArraySubset};
use zarrs::storage::ReadableWritableListableStorageTraits;

use crate::budget::BufferPool;

/// Out-of-core CSR sparse data/indices permuter using partition-buffer strategy.
///
/// Handles only the `data` and `indices` 1D arrays. The caller is responsible
/// for reading the source indptr, computing the output indptr, and writing it.
///
/// The strategy:
/// 1. Group output rows into batches by cumulative NNZ that fits in memory
/// 2. For each batch, read source data/indices with source-sorted access
/// 3. Assemble the complete contiguous output interval
/// 4. Write data/indices in a single large store_array_subset per batch
pub struct SparsePermuter {
    pool: BufferPool,
}

impl SparsePermuter {
    pub fn new(pool: BufferPool) -> Self {
        Self { pool }
    }

    /// Permute CSR data/indices arrays out-of-core.
    ///
    /// - `src_indices`, `src_data`: source 1D zarrs arrays
    /// - `dst_indices`, `dst_data`: pre-allocated destination 1D zarrs arrays
    /// - `permutation[output_row] = source_row`
    /// - `src_indptr`: full source indptr (already read into memory)
    /// - `out_indptr`: full output indptr (already computed)
    pub fn permute_data_indices<S>(
        &self,
        src_indices: &Array<S>,
        src_data: &Array<S>,
        dst_indices: &Array<S>,
        dst_data: &Array<S>,
        permutation: &[usize],
        src_indptr: &[i64],
        out_indptr: &[i64],
    ) -> Result<()>
    where
        S: ReadableWritableListableStorageTraits + ?Sized + 'static,
    {
        let n_output = permutation.len();

        let data_elem_size = src_data.data_type().fixed_size().unwrap_or(8);
        let indices_elem_size = src_indices.data_type().fixed_size().unwrap_or(8);
        let bytes_per_nnz = data_elem_size + indices_elem_size;
        let headroom = 8 * 1024 * 1024;
        let available = self.pool.budget().available().saturating_sub(headroom);
        let max_nnz_per_batch = if bytes_per_nnz > 0 {
            (available / bytes_per_nnz).max(4096)
        } else {
            usize::MAX
        };

        // Group output rows into batches by cumulative NNZ
        let mut batch_ranges: Vec<(usize, usize)> = Vec::new();
        let mut batch_start = 0usize;
        let mut batch_nnz = 0usize;
        for (out_row, &src_row) in permutation.iter().enumerate() {
            let row_nnz = (src_indptr[src_row + 1] - src_indptr[src_row]) as usize;
            if batch_nnz + row_nnz > max_nnz_per_batch && batch_nnz > 0 {
                batch_ranges.push((batch_start, out_row));
                batch_start = out_row;
                batch_nnz = 0;
            }
            batch_nnz += row_nnz;
        }
        batch_ranges.push((batch_start, n_output));

        log::info!("SparsePermuter: {} batches, max_nnz_per_batch={}", batch_ranges.len(), max_nnz_per_batch);

        for (batch_idx, &(out_start, out_end)) in batch_ranges.iter().enumerate() {
            log::debug!("Batch {}/{}: output rows {}..{}", batch_idx + 1, batch_ranges.len(), out_start, out_end);

            self.process_batch(
                src_indices, src_data,
                dst_indices, dst_data,
                permutation, src_indptr, out_indptr,
                out_start, out_end,
                data_elem_size, indices_elem_size,
            )?;
        }

        Ok(())
    }

    fn process_batch<S>(
        &self,
        src_indices: &Array<S>,
        src_data: &Array<S>,
        dst_indices: &Array<S>,
        dst_data: &Array<S>,
        permutation: &[usize],
        src_indptr: &[i64],
        out_indptr: &[i64],
        out_start: usize,
        out_end: usize,
        data_elem_size: usize,
        indices_elem_size: usize,
    ) -> Result<()>
    where
        S: ReadableWritableListableStorageTraits + ?Sized + 'static,
    {
        let batch_perm = &permutation[out_start..out_end];

        // Sort by source NNZ position for sequential reads
        let mut pairs: Vec<(usize, usize)> = batch_perm.iter()
            .enumerate()
            .map(|(local, &src)| (src, out_start + local))
            .collect();
        pairs.sort_unstable_by_key(|&(src, _)| src_indptr[src] as usize);

        // Merge nearby source rows into larger reads
        let merged = merge_sparse_reads(&pairs, src_indptr, 512);

        // Read all needed source data into a lookup
        let mut row_map: std::collections::HashMap<usize, (Vec<u8>, Vec<u8>)> =
            std::collections::HashMap::with_capacity(pairs.len());

        for run in &merged {
            if run.nnz_end <= run.nnz_start {
                continue;
            }

            let subset = ArraySubset::new_with_ranges(
                &[run.nnz_start as u64..run.nnz_end as u64],
            );

            let data_bytes: ArrayBytes<'static> = src_data.retrieve_array_subset(&subset)?;
            let data_raw = data_bytes.into_fixed()?.into_owned();
            let indices_bytes: ArrayBytes<'static> = src_indices.retrieve_array_subset(&subset)?;
            let indices_raw = indices_bytes.into_fixed()?.into_owned();

            for &(src_row, _out_row) in &run.pairs {
                let lo = src_indptr[src_row] as usize;
                let hi = src_indptr[src_row + 1] as usize;
                if hi <= lo {
                    continue;
                }
                let rel_lo = lo - run.nnz_start;
                let rel_hi = hi - run.nnz_start;

                let d = data_raw[rel_lo * data_elem_size..rel_hi * data_elem_size].to_vec();
                let i = indices_raw[rel_lo * indices_elem_size..rel_hi * indices_elem_size].to_vec();
                row_map.insert(src_row, (d, i));
            }
        }

        // Assemble output in correct output order as one contiguous block
        let dst_nnz_start = out_indptr[out_start] as usize;
        let dst_nnz_end = out_indptr[out_end] as usize;
        let total_batch_nnz = dst_nnz_end - dst_nnz_start;

        if total_batch_nnz == 0 {
            return Ok(());
        }

        let mut assembled_data = Vec::with_capacity(total_batch_nnz * data_elem_size);
        let mut assembled_indices = Vec::with_capacity(total_batch_nnz * indices_elem_size);

        for out_row in out_start..out_end {
            let src_row = permutation[out_row];
            if let Some((d, i)) = row_map.get(&src_row) {
                assembled_data.extend_from_slice(d);
                assembled_indices.extend_from_slice(i);
            }
        }

        // Single large contiguous write for the entire batch
        let write_subset = ArraySubset::new_with_ranges(
            &[dst_nnz_start as u64..dst_nnz_end as u64],
        );
        dst_data.store_array_subset(&write_subset, ArrayBytes::from(assembled_data))?;
        dst_indices.store_array_subset(&write_subset, ArrayBytes::from(assembled_indices))?;

        Ok(())
    }
}

struct MergedSparseRun {
    nnz_start: usize,
    nnz_end: usize,
    pairs: Vec<(usize, usize)>,
}

fn merge_sparse_reads(
    sorted_pairs: &[(usize, usize)],
    src_indptr: &[i64],
    gap_nnz: usize,
) -> Vec<MergedSparseRun> {
    if sorted_pairs.is_empty() {
        return Vec::new();
    }

    let mut runs = Vec::new();
    let (first_src, first_out) = sorted_pairs[0];
    let mut cur_nnz_start = src_indptr[first_src] as usize;
    let mut cur_nnz_end = src_indptr[first_src + 1] as usize;
    let mut cur_pairs = vec![(first_src, first_out)];

    for &(src, out) in &sorted_pairs[1..] {
        let lo = src_indptr[src] as usize;
        let hi = src_indptr[src + 1] as usize;

        if lo <= cur_nnz_end + gap_nnz {
            cur_nnz_end = cur_nnz_end.max(hi);
            cur_pairs.push((src, out));
        } else {
            runs.push(MergedSparseRun {
                nnz_start: cur_nnz_start,
                nnz_end: cur_nnz_end,
                pairs: std::mem::take(&mut cur_pairs),
            });
            cur_nnz_start = lo;
            cur_nnz_end = hi;
            cur_pairs = vec![(src, out)];
        }
    }

    runs.push(MergedSparseRun {
        nnz_start: cur_nnz_start,
        nnz_end: cur_nnz_end,
        pairs: cur_pairs,
    });

    runs
}
