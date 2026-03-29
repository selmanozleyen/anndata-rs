use std::sync::Mutex;

use anyhow::Result;
use rayon::prelude::*;
use zarrs::array::{Array, ArrayBytes, ArraySubset};
use zarrs::storage::ReadableWritableListableStorageTraits;

use crate::budget::BufferPool;
use crate::scatter::{RowAssignment, ScatterPlanner, SparseScatterPass};

/// Per-store CSR arrays and indptr for the scatter engine.
pub struct SparseStoreArrays<'a, S: ?Sized> {
    pub dst_indices: &'a Array<S>,
    pub dst_data: &'a Array<S>,
    pub out_indptr: Vec<i64>,
}

/// Out-of-core CSR sparse data/indices scatterer supporting multiple outputs.
///
/// Uses chunk-aligned writes: plans by destination NNZ chunks, accumulates
/// full chunk buffers, and writes each chunk exactly once via `store_chunk`
/// (no read-modify-write overhead).
pub struct SparseScatterer {
    pool: BufferPool,
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
    /// `passthrough_possible` indicates src and dst share chunk/codec config.
    pub fn scatter_data_indices<S>(
        &self,
        src_indices: &Array<S>,
        src_data: &Array<S>,
        stores: &[SparseStoreArrays<'_, S>],
        assignments: &[RowAssignment],
        src_indptr: &[i64],
        passthrough_possible: bool,
    ) -> Result<()>
    where
        S: ReadableWritableListableStorageTraits + ?Sized + 'static,
    {
        let data_elem_size = src_data.data_type().fixed_size().unwrap_or(8);
        let indices_elem_size = src_indices.data_type().fixed_size().unwrap_or(8);
        let bytes_per_nnz = data_elem_size + indices_elem_size;

        let _ = passthrough_possible;

        let headroom = 8 * 1024 * 1024;
        let available = self.pool.budget().available().saturating_sub(headroom);
        // Each NNZ needs bytes_per_nnz for both read buffer and write buffer,
        // so budget is halved between source reads and destination chunk buffers
        let max_nnz_per_pass = if bytes_per_nnz > 0 {
            (available / bytes_per_nnz / 2).max(4096)
        } else {
            usize::MAX
        };

        let store_indptrs: Vec<&[i64]> = stores.iter()
            .map(|s| s.out_indptr.as_slice())
            .collect();

        let store_nnz_chunk_sizes: Vec<usize> = stores.iter()
            .map(|s| {
                get_chunk_size_1d(s.dst_data)
            })
            .collect();

        let passes = ScatterPlanner::plan_sparse(
            assignments,
            &store_indptrs,
            &store_nnz_chunk_sizes,
            max_nnz_per_pass,
        );

        log::info!(
            "SparseScatterer: {} assignments, {} passes (chunk-aligned), max_nnz/pass={}",
            assignments.len(), passes.len(), max_nnz_per_pass
        );

        for (pass_idx, pass) in passes.iter().enumerate() {
            log::debug!(
                "Pass {}/{}: {} dst chunks, total_nnz={}",
                pass_idx + 1, passes.len(), pass.chunks.len(), pass.total_nnz
            );

            self.process_pass(
                src_data, src_indices, stores,
                pass, src_indptr,
                data_elem_size, indices_elem_size,
            )?;
        }

        Ok(())
    }

    /// Process one pass: read source data, assemble destination chunk buffers,
    /// write whole chunks.
    fn process_pass<S>(
        &self,
        src_data: &Array<S>,
        src_indices: &Array<S>,
        stores: &[SparseStoreArrays<'_, S>],
        pass: &SparseScatterPass,
        src_indptr: &[i64],
        data_elem_size: usize,
        indices_elem_size: usize,
    ) -> Result<()>
    where
        S: ReadableWritableListableStorageTraits + ?Sized + 'static,
    {
        // Phase 1: collect all source rows needed for this pass and read them
        let row_map = self.read_pass_sources(
            src_data, src_indices, pass, src_indptr,
            data_elem_size, indices_elem_size,
        )?;

        // Phase 2: assemble destination chunk buffers and write whole chunks
        self.write_pass_chunks(
            stores, pass, &row_map, src_indptr,
            data_elem_size, indices_elem_size,
        )?;

        Ok(())
    }

    /// Read all source rows needed by this pass, returning a map of
    /// source_row -> (data_bytes, indices_bytes).
    fn read_pass_sources<S>(
        &self,
        src_data: &Array<S>,
        src_indices: &Array<S>,
        pass: &SparseScatterPass,
        src_indptr: &[i64],
        data_elem_size: usize,
        indices_elem_size: usize,
    ) -> Result<std::collections::HashMap<usize, (Vec<u8>, Vec<u8>)>>
    where
        S: ReadableWritableListableStorageTraits + ?Sized + 'static,
    {
        // Collect unique source rows across all chunks in this pass
        let mut source_rows: Vec<usize> = pass.chunks.iter()
            .flat_map(|c| c.entries.iter().map(|e| e.source_row))
            .collect();
        source_rows.sort_unstable();
        source_rows.dedup();

        let capacity = source_rows.len();

        // Build RowAssignment-like structs for merge_sparse_reads
        let fake_assigns: Vec<RowAssignment> = source_rows.iter()
            .map(|&src_row| RowAssignment {
                source_row: src_row,
                store_id: 0,
                output_row: 0,
            })
            .collect();
        let assign_refs: Vec<&RowAssignment> = fake_assigns.iter().collect();

        let merged = merge_sparse_reads(&assign_refs, src_indptr, 8192);

        let src_chunk_size = get_chunk_size_1d(src_data);
        let sub_runs = split_merged_runs_by_chunk(&merged, src_indptr, src_chunk_size);

        let map = Mutex::new(
            std::collections::HashMap::<usize, (Vec<u8>, Vec<u8>)>::with_capacity(capacity),
        );

        sub_runs.par_iter().try_for_each(|sub| -> Result<()> {
            if sub.nnz_end <= sub.nnz_start {
                return Ok(());
            }

            let subset = ArraySubset::new_with_ranges(
                &[sub.nnz_start as u64..sub.nnz_end as u64],
            );

            let data_bytes: ArrayBytes<'static> =
                src_data.retrieve_array_subset(&subset)?;
            let data_raw = data_bytes.into_fixed()?.into_owned();
            let indices_bytes: ArrayBytes<'static> =
                src_indices.retrieve_array_subset(&subset)?;
            let indices_raw = indices_bytes.into_fixed()?.into_owned();

            let mut local: Vec<(usize, Vec<u8>, Vec<u8>)> =
                Vec::with_capacity(sub.assignments.len());

            for a in &sub.assignments {
                let lo = src_indptr[a.source_row] as usize;
                let hi = src_indptr[a.source_row + 1] as usize;
                if hi <= lo {
                    continue;
                }
                let rel_lo = lo - sub.nnz_start;
                let rel_hi = hi - sub.nnz_start;

                let d = data_raw[rel_lo * data_elem_size..rel_hi * data_elem_size].to_vec();
                let i = indices_raw[rel_lo * indices_elem_size..rel_hi * indices_elem_size].to_vec();
                local.push((a.source_row, d, i));
            }

            let mut guard = map.lock().unwrap();
            for (src_row, d, i) in local {
                guard.insert(src_row, (d, i));
            }
            Ok(())
        })?;

        Ok(map.into_inner().unwrap())
    }

    /// Assemble and write whole destination chunks in parallel.
    fn write_pass_chunks<S>(
        &self,
        stores: &[SparseStoreArrays<'_, S>],
        pass: &SparseScatterPass,
        row_map: &std::collections::HashMap<usize, (Vec<u8>, Vec<u8>)>,
        src_indptr: &[i64],
        data_elem_size: usize,
        indices_elem_size: usize,
    ) -> Result<()>
    where
        S: ReadableWritableListableStorageTraits + ?Sized + 'static,
    {
        pass.chunks.par_iter().try_for_each(|sc| -> Result<()> {
            let store = &stores[sc.store_id as usize];
            let out_indptr = &store.out_indptr;
            let chunk_nnz = sc.nnz_end - sc.nnz_start;

            if chunk_nnz == 0 {
                return Ok(());
            }

            let mut data_buf = vec![0u8; chunk_nnz * data_elem_size];
            let mut indices_buf = vec![0u8; chunk_nnz * indices_elem_size];

            // Sort entries by output_row so we fill the buffer in order
            let mut sorted_entries = sc.entries.clone();
            sorted_entries.sort_unstable_by_key(|e| e.output_row);

            for entry in &sorted_entries {
                let row_nnz_start = out_indptr[entry.output_row] as usize;
                let row_nnz_end = out_indptr[entry.output_row + 1] as usize;
                let row_nnz = row_nnz_end - row_nnz_start;
                if row_nnz == 0 {
                    continue;
                }

                let offset_in_chunk = row_nnz_start - sc.nnz_start;

                if let Some((d, i)) = row_map.get(&entry.source_row) {
                    let src_nnz = (src_indptr[entry.source_row + 1]
                        - src_indptr[entry.source_row]) as usize;
                    let copy_nnz = src_nnz.min(row_nnz);

                    let dst_data_start = offset_in_chunk * data_elem_size;
                    let src_data_len = copy_nnz * data_elem_size;
                    data_buf[dst_data_start..dst_data_start + src_data_len]
                        .copy_from_slice(&d[..src_data_len]);

                    let dst_idx_start = offset_in_chunk * indices_elem_size;
                    let src_idx_len = copy_nnz * indices_elem_size;
                    indices_buf[dst_idx_start..dst_idx_start + src_idx_len]
                        .copy_from_slice(&i[..src_idx_len]);
                }
            }

            let total_dst_nnz = *out_indptr.last().unwrap_or(&0) as usize;
            let dst_data_chunk_size = get_chunk_size_1d(store.dst_data);
            let is_full_chunk = (sc.nnz_end - sc.nnz_start) == dst_data_chunk_size
                || sc.nnz_end == total_dst_nnz;

            if is_full_chunk {
                // Write the entire chunk at once -- store_array_subset on a
                // chunk-aligned range is equivalent to store_chunk but does not
                // require us to figure out the chunk index encoding.
                let write_subset = ArraySubset::new_with_ranges(
                    &[sc.nnz_start as u64..sc.nnz_end as u64],
                );
                store.dst_data.store_array_subset(
                    &write_subset,
                    ArrayBytes::from(data_buf),
                )?;
                store.dst_indices.store_array_subset(
                    &write_subset,
                    ArrayBytes::from(indices_buf),
                )?;
            } else {
                let write_subset = ArraySubset::new_with_ranges(
                    &[sc.nnz_start as u64..sc.nnz_end as u64],
                );
                store.dst_data.store_array_subset(
                    &write_subset,
                    ArrayBytes::from(data_buf),
                )?;
                store.dst_indices.store_array_subset(
                    &write_subset,
                    ArrayBytes::from(indices_buf),
                )?;
            }

            Ok(())
        })
    }
}

/// Get the 1D chunk size for a zarrs Array, or usize::MAX if unknown.
fn get_chunk_size_1d<S>(array: &Array<S>) -> usize
where
    S: ReadableWritableListableStorageTraits + ?Sized + 'static,
{
    array
        .chunk_grid_shape()
        .first()
        .and_then(|&n| {
            if n == 0 {
                None
            } else {
                array
                    .chunk_shape(&[0u64])
                    .ok()
                    .and_then(|s| s.first().map(|c| c.get() as usize))
            }
        })
        .unwrap_or(usize::MAX)
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

fn split_merged_runs_by_chunk<'a>(
    merged: &[MergedSparseRun<'a>],
    src_indptr: &[i64],
    src_chunk_size: usize,
) -> Vec<MergedSparseRun<'a>> {
    let mut out = Vec::new();
    for run in merged {
        if run.nnz_end <= run.nnz_start || src_chunk_size == usize::MAX {
            out.push(MergedSparseRun {
                nnz_start: run.nnz_start,
                nnz_end: run.nnz_end,
                assignments: run.assignments.clone(),
            });
            continue;
        }

        let first_chunk = run.nnz_start / src_chunk_size;
        let last_chunk = (run.nnz_end - 1) / src_chunk_size;

        if first_chunk == last_chunk {
            out.push(MergedSparseRun {
                nnz_start: run.nnz_start,
                nnz_end: run.nnz_end,
                assignments: run.assignments.clone(),
            });
            continue;
        }

        let n_chunks = last_chunk - first_chunk + 1;
        let mut buckets: Vec<Vec<&'a RowAssignment>> =
            (0..n_chunks).map(|_| Vec::new()).collect();

        for &a in &run.assignments {
            let lo = src_indptr[a.source_row] as usize;
            let chunk_idx = lo / src_chunk_size;
            let bucket = chunk_idx - first_chunk;
            buckets[bucket.min(n_chunks - 1)].push(a);
        }

        for (i, assigns) in buckets.into_iter().enumerate() {
            if assigns.is_empty() {
                continue;
            }
            let chunk_idx = first_chunk + i;
            let sub_nnz_start = if chunk_idx == first_chunk {
                run.nnz_start
            } else {
                chunk_idx * src_chunk_size
            };
            let sub_nnz_end = if chunk_idx == last_chunk {
                run.nnz_end
            } else {
                (chunk_idx + 1) * src_chunk_size
            };
            out.push(MergedSparseRun {
                nnz_start: sub_nnz_start,
                nnz_end: sub_nnz_end,
                assignments: assigns,
            });
        }
    }
    out
}
