use anyhow::Result;
use rayon::prelude::*;
use zarrs::array::{Array, ArrayBytes, ArraySubset};
use zarrs::storage::ReadableWritableListableStorageTraits;

use crate::budget::BufferPool;
use crate::scatter::RowAssignment;

/// Out-of-core CSR sparse data/indices scatterer supporting multiple outputs.
///
/// Generalizes SparsePermuter: reads source data/indices once per batch and
/// writes to N destination data/indices arrays. Batching is by cumulative NNZ
/// that fits in memory.
pub struct SparseScatterer {
    pool: BufferPool,
}

/// Per-store CSR arrays and indptr for the scatter engine.
pub struct SparseStoreArrays<'a, S: ?Sized> {
    pub dst_indices: &'a Array<S>,
    pub dst_data: &'a Array<S>,
    pub out_indptr: Vec<i64>,
}

impl SparseScatterer {
    pub fn new(pool: BufferPool) -> Self {
        Self { pool }
    }

    /// Scatter CSR data/indices from one source into multiple destinations.
    ///
    /// `assignments` maps source rows to (store_id, output_row).
    /// `stores[i]` holds the destination arrays and indptr for store i.
    /// `src_indptr` is the full source indptr (already in memory).
    pub fn scatter_data_indices<S>(
        &self,
        src_indices: &Array<S>,
        src_data: &Array<S>,
        stores: &[SparseStoreArrays<'_, S>],
        assignments: &[RowAssignment],
        src_indptr: &[i64],
    ) -> Result<()>
    where
        S: ReadableWritableListableStorageTraits + ?Sized + 'static,
    {
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

        // Sort assignments by source indptr position for sequential reads
        let mut sorted_assigns: Vec<&RowAssignment> = assignments.iter().collect();
        sorted_assigns.sort_unstable_by_key(|a| src_indptr[a.source_row] as usize);

        // Group into batches by cumulative NNZ
        let mut batch_ranges: Vec<(usize, usize)> = Vec::new();
        let mut batch_start = 0usize;
        let mut batch_nnz = 0usize;

        for (idx, a) in sorted_assigns.iter().enumerate() {
            let row_nnz = (src_indptr[a.source_row + 1] - src_indptr[a.source_row]) as usize;
            if batch_nnz + row_nnz > max_nnz_per_batch && batch_nnz > 0 {
                batch_ranges.push((batch_start, idx));
                batch_start = idx;
                batch_nnz = 0;
            }
            batch_nnz += row_nnz;
        }
        batch_ranges.push((batch_start, sorted_assigns.len()));

        log::info!(
            "SparseScatterer: {} stores, {} assignments, {} batches, max_nnz={}",
            stores.len(), assignments.len(), batch_ranges.len(), max_nnz_per_batch
        );

        for (batch_idx, &(start, end)) in batch_ranges.iter().enumerate() {
            log::debug!("Batch {}/{}: {} assignments", batch_idx + 1, batch_ranges.len(), end - start);

            self.process_batch(
                src_indices, src_data, stores,
                &sorted_assigns[start..end],
                src_indptr,
                data_elem_size, indices_elem_size,
            )?;
        }

        Ok(())
    }

    fn process_batch<S>(
        &self,
        src_indices: &Array<S>,
        src_data: &Array<S>,
        stores: &[SparseStoreArrays<'_, S>],
        batch_assigns: &[&RowAssignment],
        src_indptr: &[i64],
        data_elem_size: usize,
        indices_elem_size: usize,
    ) -> Result<()>
    where
        S: ReadableWritableListableStorageTraits + ?Sized + 'static,
    {
        // Merge nearby source rows into larger reads
        let merged = merge_sparse_reads(batch_assigns, src_indptr, 512);

        // Read all needed source data into a lookup
        let mut row_map: std::collections::HashMap<usize, (Vec<u8>, Vec<u8>)> =
            std::collections::HashMap::with_capacity(batch_assigns.len());

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

            for a in &run.assignments {
                let lo = src_indptr[a.source_row] as usize;
                let hi = src_indptr[a.source_row + 1] as usize;
                if hi <= lo {
                    continue;
                }
                let rel_lo = lo - run.nnz_start;
                let rel_hi = hi - run.nnz_start;

                let d = data_raw[rel_lo * data_elem_size..rel_hi * data_elem_size].to_vec();
                let i = indices_raw[rel_lo * indices_elem_size..rel_hi * indices_elem_size].to_vec();
                row_map.insert(a.source_row, (d, i));
            }
        }

        // Group batch assignments by store_id, then write each store in parallel
        let n_stores = stores.len();
        let mut per_store: Vec<Vec<&RowAssignment>> = vec![Vec::new(); n_stores];
        for a in batch_assigns {
            per_store[a.store_id as usize].push(a);
        }

        // Assemble and flush each store's portion in parallel -- each store
        // writes to independent arrays so there are no data races.
        per_store.par_iter().enumerate().try_for_each(
            |(store_id, store_assigns)| -> Result<()> {
                if store_assigns.is_empty() {
                    return Ok(());
                }

                let mut sorted: Vec<&&RowAssignment> = store_assigns.iter().collect();
                sorted.sort_unstable_by_key(|a| a.output_row);

                let runs = find_contiguous_output_runs(
                    &sorted, &stores[store_id].out_indptr,
                );

                for run in &runs {
                    let dst_nnz_start = stores[store_id].out_indptr[run.out_start] as usize;
                    let dst_nnz_end = stores[store_id].out_indptr[run.out_end] as usize;
                    let total_nnz = dst_nnz_end - dst_nnz_start;
                    if total_nnz == 0 {
                        continue;
                    }

                    let mut assembled_data = Vec::with_capacity(total_nnz * data_elem_size);
                    let mut assembled_indices = Vec::with_capacity(total_nnz * indices_elem_size);

                    for out_row in run.out_start..run.out_end {
                        let src_row = run.assignments_by_out[&out_row];
                        if let Some((d, i)) = row_map.get(&src_row) {
                            assembled_data.extend_from_slice(d);
                            assembled_indices.extend_from_slice(i);
                        }
                    }

                    let write_subset = ArraySubset::new_with_ranges(
                        &[dst_nnz_start as u64..dst_nnz_end as u64],
                    );
                    stores[store_id].dst_data.store_array_subset(
                        &write_subset, ArrayBytes::from(assembled_data),
                    )?;
                    stores[store_id].dst_indices.store_array_subset(
                        &write_subset, ArrayBytes::from(assembled_indices),
                    )?;
                }
                Ok(())
            },
        )?;

        Ok(())
    }
}

struct MergedSparseRun<'a> {
    nnz_start: usize,
    nnz_end: usize,
    assignments: Vec<&'a RowAssignment>,
}

fn merge_sparse_reads<'a>(
    batch_assigns: &[&'a RowAssignment],
    src_indptr: &[i64],
    gap_nnz: usize,
) -> Vec<MergedSparseRun<'a>> {
    if batch_assigns.is_empty() {
        return Vec::new();
    }

    let mut runs = Vec::new();
    let first = batch_assigns[0];
    let mut cur_nnz_start = src_indptr[first.source_row] as usize;
    let mut cur_nnz_end = src_indptr[first.source_row + 1] as usize;
    let mut cur_assigns = vec![first];

    for &a in &batch_assigns[1..] {
        let lo = src_indptr[a.source_row] as usize;
        let hi = src_indptr[a.source_row + 1] as usize;

        if lo <= cur_nnz_end + gap_nnz {
            cur_nnz_end = cur_nnz_end.max(hi);
            cur_assigns.push(a);
        } else {
            runs.push(MergedSparseRun {
                nnz_start: cur_nnz_start,
                nnz_end: cur_nnz_end,
                assignments: std::mem::take(&mut cur_assigns),
            });
            cur_nnz_start = lo;
            cur_nnz_end = hi;
            cur_assigns = vec![a];
        }
    }

    runs.push(MergedSparseRun {
        nnz_start: cur_nnz_start,
        nnz_end: cur_nnz_end,
        assignments: cur_assigns,
    });

    runs
}

struct OutputRun {
    out_start: usize,
    out_end: usize,
    assignments_by_out: std::collections::HashMap<usize, usize>,
}

fn find_contiguous_output_runs(
    sorted_assigns: &[&&RowAssignment],
    _out_indptr: &[i64],
) -> Vec<OutputRun> {
    if sorted_assigns.is_empty() {
        return Vec::new();
    }

    let mut runs = Vec::new();
    let mut cur_start = sorted_assigns[0].output_row;
    let mut cur_end = cur_start + 1;
    let mut cur_map = std::collections::HashMap::new();
    cur_map.insert(sorted_assigns[0].output_row, sorted_assigns[0].source_row);

    for &a in &sorted_assigns[1..] {
        if a.output_row == cur_end {
            cur_end += 1;
            cur_map.insert(a.output_row, a.source_row);
        } else {
            runs.push(OutputRun {
                out_start: cur_start,
                out_end: cur_end,
                assignments_by_out: std::mem::take(&mut cur_map),
            });
            cur_start = a.output_row;
            cur_end = cur_start + 1;
            cur_map = std::collections::HashMap::new();
            cur_map.insert(a.output_row, a.source_row);
        }
    }

    runs.push(OutputRun {
        out_start: cur_start,
        out_end: cur_end,
        assignments_by_out: cur_map,
    });

    runs
}
