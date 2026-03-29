use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::Result;
use rayon::prelude::*;
use zarrs::array::{Array, ArrayBytes, ArraySubset};
use zarrs::storage::ReadableWritableListableStorageTraits;

use crate::budget::BufferPool;
use crate::scatter::{RowAssignment, ScatterPlanner, SparseScatterPass, SparseScatterEntry};

/// Per-store CSR arrays and indptr for the scatter engine.
pub struct SparseStoreArrays<'a, S: ?Sized> {
    pub dst_indices: &'a Array<S>,
    pub dst_data: &'a Array<S>,
    pub out_indptr: Vec<i64>,
}

/// Out-of-core CSR sparse data/indices scatterer supporting multiple outputs.
///
/// Uses chunk-aligned writes with streaming flush: source chunks are decoded
/// in parallel and their rows are scattered directly into pre-allocated
/// destination chunk buffers. As soon as a destination chunk receives all
/// its expected rows, the decoding thread flushes it immediately -- reads
/// and writes overlap with no barrier.
pub struct SparseScatterer {
    pool: BufferPool,
}

impl SparseScatterer {
    pub fn new(pool: BufferPool) -> Self {
        Self { pool }
    }

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
        let max_nnz_per_pass = if bytes_per_nnz > 0 {
            (available / bytes_per_nnz / 2).max(4096)
        } else {
            usize::MAX
        };

        let store_indptrs: Vec<&[i64]> = stores.iter()
            .map(|s| s.out_indptr.as_slice())
            .collect();

        let store_nnz_chunk_sizes: Vec<usize> = stores.iter()
            .map(|s| get_chunk_size_1d(s.dst_data))
            .collect();

        let passes = ScatterPlanner::plan_sparse(
            assignments,
            &store_indptrs,
            &store_nnz_chunk_sizes,
            max_nnz_per_pass,
        );

        log::info!(
            "SparseScatterer: {} assignments, {} passes (streaming), max_nnz/pass={}",
            assignments.len(), passes.len(), max_nnz_per_pass
        );

        for (pass_idx, pass) in passes.iter().enumerate() {
            log::debug!(
                "Pass {}/{}: {} dst chunks, total_nnz={}",
                pass_idx + 1, passes.len(), pass.chunks.len(), pass.total_nnz
            );

            self.process_pass_streaming(
                src_data, src_indices, stores,
                pass, src_indptr,
                data_elem_size, indices_elem_size,
            )?;
        }

        Ok(())
    }

    /// Streaming pass: decode source chunks in parallel, scatter rows directly
    /// into destination chunk buffers, flush each buffer the moment it is complete.
    fn process_pass_streaming<S>(
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
        let dst_bufs: Vec<DstChunkBuf> = pass.chunks.iter().map(|sc| {
            let chunk_nnz = sc.nnz_end - sc.nnz_start;
            DstChunkBuf {
                store_id: sc.store_id,
                nnz_start: sc.nnz_start,
                nnz_end: sc.nnz_end,
                data_buf: vec![0u8; chunk_nnz * data_elem_size],
                indices_buf: vec![0u8; chunk_nnz * indices_elem_size],
                remaining: AtomicUsize::new(sc.entries.len()),
                flushed: AtomicUsize::new(0),
            }
        }).collect();

        // Build a lookup: source_row -> list of (dst_chunk_idx, entry)
        // so that when we decode a source row, we know which dst buffers to fill.
        let mut src_to_dst: std::collections::HashMap<usize, Vec<(usize, SparseScatterEntry)>> =
            std::collections::HashMap::new();
        for (chunk_idx, sc) in pass.chunks.iter().enumerate() {
            for entry in &sc.entries {
                src_to_dst
                    .entry(entry.source_row)
                    .or_default()
                    .push((chunk_idx, *entry));
            }
        }

        // Collect unique source rows, build merged read runs
        let mut source_rows: Vec<usize> = src_to_dst.keys().copied().collect();
        source_rows.sort_unstable();

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

        // Collect flush errors
        let flush_errors: Mutex<Vec<anyhow::Error>> = Mutex::new(Vec::new());

        // -- Stream: decode sub-runs in parallel, scatter + flush --
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

            for a in &sub.assignments {
                let lo = src_indptr[a.source_row] as usize;
                let hi = src_indptr[a.source_row + 1] as usize;
                if hi <= lo {
                    // Still decrement counters for zero-nnz rows
                    if let Some(targets) = src_to_dst.get(&a.source_row) {
                        for &(chunk_idx, _) in targets {
                            let buf = &dst_bufs[chunk_idx];
                            let prev = buf.remaining.fetch_sub(1, Ordering::AcqRel);
                            if prev == 1 {
                                if let Err(e) = flush_chunk(
                                    buf, stores, data_elem_size, indices_elem_size,
                                ) {
                                    flush_errors.lock().unwrap().push(e);
                                }
                            }
                        }
                    }
                    continue;
                }

                let rel_lo = lo - sub.nnz_start;
                let rel_hi = hi - sub.nnz_start;
                let src_data_slice =
                    &data_raw[rel_lo * data_elem_size..rel_hi * data_elem_size];
                let src_idx_slice =
                    &indices_raw[rel_lo * indices_elem_size..rel_hi * indices_elem_size];
                let src_nnz = hi - lo;

                if let Some(targets) = src_to_dst.get(&a.source_row) {
                    for &(chunk_idx, ref entry) in targets {
                        let buf = &dst_bufs[chunk_idx];
                        let store = &stores[buf.store_id as usize];
                        let out_indptr = &store.out_indptr;

                        let row_nnz_start = out_indptr[entry.output_row] as usize;
                        let row_nnz_end = out_indptr[entry.output_row + 1] as usize;
                        let row_nnz = row_nnz_end - row_nnz_start;

                        if row_nnz > 0 {
                            let offset_in_chunk = row_nnz_start - buf.nnz_start;
                            let copy_nnz = src_nnz.min(row_nnz);

                            let dst_data_start = offset_in_chunk * data_elem_size;
                            let src_data_len = copy_nnz * data_elem_size;

                            // Safety: each entry targets a unique output_row within
                            // this chunk, so different entries write to disjoint
                            // byte ranges of the buffer. The atomic counter ensures
                            // the buffer is not read for flushing until all writes
                            // are complete.
                            unsafe {
                                let data_ptr = buf.data_buf.as_ptr() as *mut u8;
                                std::ptr::copy_nonoverlapping(
                                    src_data_slice.as_ptr(),
                                    data_ptr.add(dst_data_start),
                                    src_data_len,
                                );

                                let idx_ptr = buf.indices_buf.as_ptr() as *mut u8;
                                let dst_idx_start = offset_in_chunk * indices_elem_size;
                                let src_idx_len = copy_nnz * indices_elem_size;
                                std::ptr::copy_nonoverlapping(
                                    src_idx_slice.as_ptr(),
                                    idx_ptr.add(dst_idx_start),
                                    src_idx_len,
                                );
                            }
                        }

                        let prev = buf.remaining.fetch_sub(1, Ordering::AcqRel);
                        if prev == 1 {
                            if let Err(e) = flush_chunk(
                                buf, stores, data_elem_size, indices_elem_size,
                            ) {
                                flush_errors.lock().unwrap().push(e);
                            }
                        }
                    }
                }
            }
            Ok(())
        })?;

        // Check for flush errors
        let errors = flush_errors.into_inner().unwrap();
        if let Some(e) = errors.into_iter().next() {
            return Err(e);
        }

        // Flush any dst chunks that were never triggered (shouldn't happen
        // if the planner is correct, but safety net)
        for buf in &dst_bufs {
            if buf.flushed.load(Ordering::Acquire) == 0
                && buf.remaining.load(Ordering::Acquire) == 0
                && (buf.nnz_end > buf.nnz_start)
            {
                flush_chunk(buf, stores, data_elem_size, indices_elem_size)?;
            }
        }

        Ok(())
    }
}

/// Flush a completed destination chunk buffer to disk.
fn flush_chunk<S>(
    buf: &DstChunkBuf,
    stores: &[SparseStoreArrays<'_, S>],
    _data_elem_size: usize,
    _indices_elem_size: usize,
) -> Result<()>
where
    S: ReadableWritableListableStorageTraits + ?Sized + 'static,
{
    buf.flushed.store(1, Ordering::Release);

    let chunk_nnz = buf.nnz_end - buf.nnz_start;
    if chunk_nnz == 0 {
        return Ok(());
    }

    let store = &stores[buf.store_id as usize];
    let write_subset = ArraySubset::new_with_ranges(
        &[buf.nnz_start as u64..buf.nnz_end as u64],
    );

    // Safety: we only reach here after all writers have finished (atomic
    // counter hit zero with AcqRel ordering), so reading the buffers is safe.
    store.dst_data.store_array_subset(
        &write_subset,
        ArrayBytes::from(buf.data_buf.as_slice()),
    )?;
    store.dst_indices.store_array_subset(
        &write_subset,
        ArrayBytes::from(buf.indices_buf.as_slice()),
    )?;

    Ok(())
}

// Use the struct at module level so flush_chunk can reference it
struct DstChunkBuf {
    store_id: u16,
    nnz_start: usize,
    nnz_end: usize,
    data_buf: Vec<u8>,
    indices_buf: Vec<u8>,
    remaining: AtomicUsize,
    flushed: AtomicUsize,
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
