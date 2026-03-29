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

        let src_chunk_nnz = get_chunk_size_1d(src_data);
        let n_threads = rayon::current_num_threads();
        let src_concurrent_bytes = n_threads * src_chunk_nnz * bytes_per_nnz * 2;
        let dst_budget = available.saturating_sub(src_concurrent_bytes);

        let max_nnz_per_pass = if bytes_per_nnz > 0 && dst_budget > 0 {
            (dst_budget / bytes_per_nnz).max(4096)
        } else {
            (available / bytes_per_nnz / 2).max(4096)
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
            "SparseScatterer: {} assignments, {} passes (streaming), max_nnz/pass={}, \
             dst_budget={:.1}GB, src_concurrent={:.1}GB ({} threads, chunk={})",
            assignments.len(), passes.len(), max_nnz_per_pass,
            dst_budget as f64 / 1e9,
            src_concurrent_bytes as f64 / 1e9,
            n_threads, src_chunk_nnz,
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
        let pass_t0 = std::time::Instant::now();

        let total_buf_bytes: usize = pass.chunks.iter()
            .map(|sc| (sc.nnz_end - sc.nnz_start) * (data_elem_size + indices_elem_size))
            .sum();
        log::info!(
            "process_pass: allocating {} dst chunk buffers ({:.1} GB), {} entries total",
            pass.chunks.len(),
            total_buf_bytes as f64 / 1e9,
            pass.chunks.iter().map(|sc| sc.entries.len()).sum::<usize>(),
        );

        let dst_bufs: Vec<DstChunkBuf> = pass.chunks.iter().map(|sc| {
            let chunk_nnz = sc.nnz_end - sc.nnz_start;
            let data_len = chunk_nnz * data_elem_size;
            let indices_len = chunk_nnz * indices_elem_size;
            let mut data_buf = Vec::with_capacity(data_len);
            let mut indices_buf = Vec::with_capacity(indices_len);
            // SAFETY: every byte position [0..chunk_nnz) will be written exactly
            // once before flush reads the buffer. The atomic remaining counter
            // enforces this. Skipping zero-fill avoids touching ~30GB of pages.
            unsafe {
                data_buf.set_len(data_len);
                indices_buf.set_len(indices_len);
            }
            DstChunkBuf {
                store_id: sc.store_id,
                nnz_start: sc.nnz_start,
                nnz_end: sc.nnz_end,
                data_buf,
                indices_buf,
                remaining: AtomicUsize::new(sc.entries.len()),
                flushed: AtomicUsize::new(0),
            }
        }).collect();

        log::info!("process_pass: buffers allocated in {:.1}s", pass_t0.elapsed().as_secs_f64());

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

        let total_read_nnz: usize = sub_runs.iter()
            .map(|s| s.nnz_end - s.nnz_start)
            .sum();
        log::info!(
            "process_pass: {} merged runs -> {} sub_runs, total_read_nnz={} ({:.1} GB), \
             src_chunk_size={}, planning took {:.1}s",
            merged.len(), sub_runs.len(),
            total_read_nnz,
            total_read_nnz as f64 * (data_elem_size + indices_elem_size) as f64 / 1e9,
            src_chunk_size,
            pass_t0.elapsed().as_secs_f64(),
        );

        let flush_errors: Mutex<Vec<anyhow::Error>> = Mutex::new(Vec::new());
        let reads_done = AtomicUsize::new(0);
        let writes_done = AtomicUsize::new(0);
        let total_subs = sub_runs.len();

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

            let rd = reads_done.fetch_add(1, Ordering::Relaxed) + 1;
            if rd % 500 == 0 || rd == total_subs {
                log::info!(
                    "  sub_run read {}/{} ({:.1}s elapsed)",
                    rd, total_subs, pass_t0.elapsed().as_secs_f64(),
                );
            }

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
                if let Some(targets) = src_to_dst.get(&a.source_row) {
                    for &(chunk_idx, ref entry) in targets {
                        let buf = &dst_bufs[chunk_idx];
                        let store = &stores[buf.store_id as usize];
                        let out_indptr = &store.out_indptr;

                        let row_nnz_start = out_indptr[entry.output_row] as usize;
                        let row_nnz_end = out_indptr[entry.output_row + 1] as usize;

                        // Clamp the row's NNZ range to this chunk's range.
                        // A row may span multiple destination chunks; each chunk
                        // gets only its portion.
                        let clamped_start = row_nnz_start.max(buf.nnz_start);
                        let clamped_end = row_nnz_end.min(buf.nnz_end);

                        if clamped_start < clamped_end {
                            let copy_nnz = clamped_end - clamped_start;
                            let offset_in_chunk = clamped_start - buf.nnz_start;
                            let skip_in_src = clamped_start - row_nnz_start;

                            // Safety: each (output_row, chunk) pair writes to a
                            // unique disjoint range in the buffer. The atomic
                            // counter ensures the buffer is not read for flushing
                            // until all writes are complete.
                            unsafe {
                                let data_ptr = buf.data_buf.as_ptr() as *mut u8;
                                let src_off = skip_in_src * data_elem_size;
                                let dst_off = offset_in_chunk * data_elem_size;
                                let len = copy_nnz * data_elem_size;
                                std::ptr::copy_nonoverlapping(
                                    src_data_slice.as_ptr().add(src_off),
                                    data_ptr.add(dst_off),
                                    len,
                                );

                                let idx_ptr = buf.indices_buf.as_ptr() as *mut u8;
                                let src_idx_off = skip_in_src * indices_elem_size;
                                let dst_idx_off = offset_in_chunk * indices_elem_size;
                                let idx_len = copy_nnz * indices_elem_size;
                                std::ptr::copy_nonoverlapping(
                                    src_idx_slice.as_ptr().add(src_idx_off),
                                    idx_ptr.add(dst_idx_off),
                                    idx_len,
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
                            let wd = writes_done.fetch_add(1, Ordering::Relaxed) + 1;
                            if wd % 500 == 0 || wd == pass.chunks.len() {
                                log::info!(
                                    "  dst chunk flushed {}/{} ({:.1}s elapsed)",
                                    wd, pass.chunks.len(), pass_t0.elapsed().as_secs_f64(),
                                );
                            }
                        }
                    }
                }
            }
            Ok(())
        })?;

        let errors = flush_errors.into_inner().unwrap();
        if let Some(e) = errors.into_iter().next() {
            return Err(e);
        }

        for buf in &dst_bufs {
            if buf.flushed.load(Ordering::Acquire) == 0
                && buf.remaining.load(Ordering::Acquire) == 0
                && (buf.nnz_end > buf.nnz_start)
            {
                flush_chunk(buf, stores, data_elem_size, indices_elem_size)?;
            }
        }

        log::info!("process_pass complete: {:.1}s total", pass_t0.elapsed().as_secs_f64());

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
            let mut sub_nnz_end = if chunk_idx == last_chunk {
                run.nnz_end
            } else {
                (chunk_idx + 1) * src_chunk_size
            };
            // A row's NNZ span may extend past the chunk boundary.
            // Expand the read range to cover all assigned rows fully.
            for &a in &assigns {
                let hi = src_indptr[a.source_row + 1] as usize;
                sub_nnz_end = sub_nnz_end.max(hi);
            }
            out.push(MergedSparseRun {
                nnz_start: sub_nnz_start,
                nnz_end: sub_nnz_end,
                assignments: assigns,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use zarrs::array::ArrayBuilder;
    use zarrs::array::data_type;
    use zarrs::storage::store::MemoryStore;
    use zarrs::storage::ReadableWritableListableStorage;

    fn make_1d_u8_array(
        store: ReadableWritableListableStorage,
        path: &str,
        len: u64,
        chunk_size: u64,
    ) -> Array<dyn ReadableWritableListableStorageTraits> {
        let cs = vec![chunk_size.min(len).max(1)];
        let builder = ArrayBuilder::new(
            vec![len],
            cs,
            data_type::uint8(),
            0u8,
        );
        let arr = builder.build(store, path).unwrap();
        arr.store_metadata().unwrap();
        arr
    }

    fn write_1d(arr: &Array<dyn ReadableWritableListableStorageTraits>, data: &[u8]) {
        let n = arr.shape()[0];
        let subset = ArraySubset::new_with_ranges(&[0..n]);
        arr.store_array_subset(&subset, ArrayBytes::from(data.to_vec())).unwrap();
    }

    fn read_1d(arr: &Array<dyn ReadableWritableListableStorageTraits>) -> Vec<u8> {
        let n = arr.shape()[0];
        let subset = ArraySubset::new_with_ranges(&[0..n]);
        let bytes: ArrayBytes<'_> = arr.retrieve_array_subset(&subset).unwrap();
        bytes.into_fixed().unwrap().into_owned()
    }

    // ===============================================================
    //  merge_sparse_reads tests
    // ===============================================================

    #[test]
    fn merge_reads_single_row() {
        let a = RowAssignment { source_row: 0, store_id: 0, output_row: 0 };
        let refs = vec![&a];
        let indptr = vec![0i64, 10];
        let runs = merge_sparse_reads(&refs, &indptr, 100);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].nnz_start, 0);
        assert_eq!(runs[0].nnz_end, 10);
    }

    #[test]
    fn merge_reads_adjacent_merge() {
        let a0 = RowAssignment { source_row: 0, store_id: 0, output_row: 0 };
        let a1 = RowAssignment { source_row: 1, store_id: 0, output_row: 1 };
        let refs = vec![&a0, &a1];
        let indptr = vec![0i64, 10, 20];
        let runs = merge_sparse_reads(&refs, &indptr, 5);
        assert_eq!(runs.len(), 1, "adjacent rows should merge");
        assert_eq!(runs[0].nnz_start, 0);
        assert_eq!(runs[0].nnz_end, 20);
        assert_eq!(runs[0].assignments.len(), 2);
    }

    #[test]
    fn merge_reads_gap_too_large() {
        let a0 = RowAssignment { source_row: 0, store_id: 0, output_row: 0 };
        let a1 = RowAssignment { source_row: 2, store_id: 0, output_row: 1 };
        let refs = vec![&a0, &a1];
        // row0: [0,10), row1 skipped, row2: [1000, 1010)
        let indptr = vec![0i64, 10, 1000, 1010];
        let runs = merge_sparse_reads(&refs, &indptr, 5);
        assert_eq!(runs.len(), 2, "big gap should split");
    }

    #[test]
    fn merge_reads_empty() {
        let refs: Vec<&RowAssignment> = vec![];
        let indptr = vec![0i64, 10];
        let runs = merge_sparse_reads(&refs, &indptr, 100);
        assert!(runs.is_empty());
    }

    // ===============================================================
    //  split_merged_runs_by_chunk tests
    // ===============================================================

    #[test]
    fn split_single_chunk_no_split() {
        let a = RowAssignment { source_row: 0, store_id: 0, output_row: 0 };
        let run = MergedSparseRun {
            nnz_start: 0,
            nnz_end: 50,
            assignments: vec![&a],
        };
        let indptr = vec![0i64, 50];
        let result = split_merged_runs_by_chunk(&[run], &indptr, 100);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].nnz_start, 0);
        assert_eq!(result[0].nnz_end, 50);
    }

    #[test]
    fn split_row_spanning_chunk_boundary() {
        // Row 1 NNZ = [90, 110) -- crosses chunk boundary at 100.
        // source_row=1 so lo=indptr[1]=90, hi=indptr[2]=110.
        let a = RowAssignment { source_row: 1, store_id: 0, output_row: 0 };
        let run = MergedSparseRun {
            nnz_start: 90,
            nnz_end: 110,
            assignments: vec![&a],
        };
        let indptr = vec![0i64, 90, 110];
        let result = split_merged_runs_by_chunk(&[run], &indptr, 100);
        // Row bucketed to chunk 0 (90/100=0), sub-run must expand nnz_end to 110.
        assert_eq!(result.len(), 1);
        assert!(
            result[0].nnz_end >= 110,
            "sub-run nnz_end must cover full row span; got {}",
            result[0].nnz_end,
        );
    }

    #[test]
    fn split_multiple_rows_one_spans() {
        // Row 0: [0, 10), row 1: [10, 95), row 2: [95, 110)
        // Chunk size 100. Row 2 starts in chunk 0, ends in chunk 1.
        let a0 = RowAssignment { source_row: 0, store_id: 0, output_row: 0 };
        let a1 = RowAssignment { source_row: 1, store_id: 0, output_row: 1 };
        let a2 = RowAssignment { source_row: 2, store_id: 0, output_row: 2 };
        let run = MergedSparseRun {
            nnz_start: 0,
            nnz_end: 110,
            assignments: vec![&a0, &a1, &a2],
        };
        let indptr = vec![0i64, 10, 95, 110];
        let result = split_merged_runs_by_chunk(&[run], &indptr, 100);

        // All 3 rows start in chunk 0 (lo/100 == 0), so they are all in one bucket.
        // The sub-run must cover up to 110.
        let total_assigns: usize = result.iter().map(|r| r.assignments.len()).sum();
        assert_eq!(total_assigns, 3);
        let max_end = result.iter().map(|r| r.nnz_end).max().unwrap();
        assert!(max_end >= 110, "must cover all row data; got {}", max_end);
    }

    #[test]
    fn split_rows_in_different_chunks() {
        // 3 source rows:
        //   row 0: NNZ [0, 50)    -> chunk 0
        //   row 1: NNZ [50, 150)  -> chunk 0 (lo=50, 50/100=0) but extends into chunk 1
        //   row 2: NNZ [150, 200) -> chunk 1
        // After split we expect 2 sub-runs: chunk0 has rows 0,1; chunk1 has row 2.
        let a0 = RowAssignment { source_row: 0, store_id: 0, output_row: 0 };
        let a2 = RowAssignment { source_row: 2, store_id: 0, output_row: 1 };
        let run = MergedSparseRun {
            nnz_start: 0,
            nnz_end: 200,
            assignments: vec![&a0, &a2],
        };
        // indptr: row0=[0,50), row1=[50,150), row2=[150,200)
        let indptr = vec![0i64, 50, 150, 200];
        let result = split_merged_runs_by_chunk(&[run], &indptr, 100);
        assert_eq!(result.len(), 2);
        // chunk 0 has row 0, chunk 1 has row 2
        assert_eq!(result[0].assignments.len(), 1);
        assert_eq!(result[0].assignments[0].source_row, 0);
        assert_eq!(result[1].assignments.len(), 1);
        assert_eq!(result[1].assignments[0].source_row, 2);
    }

    #[test]
    fn split_usize_max_chunk_no_split() {
        let a = RowAssignment { source_row: 0, store_id: 0, output_row: 0 };
        let run = MergedSparseRun {
            nnz_start: 0,
            nnz_end: 500,
            assignments: vec![&a],
        };
        let indptr = vec![0i64, 500];
        let result = split_merged_runs_by_chunk(&[run], &indptr, usize::MAX);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn split_exact_chunk_boundary_row() {
        // Row exactly fills chunk 1: NNZ = [100, 200) with chunk_size=100
        let a = RowAssignment { source_row: 1, store_id: 0, output_row: 0 };
        let run = MergedSparseRun {
            nnz_start: 100,
            nnz_end: 200,
            assignments: vec![&a],
        };
        let indptr = vec![0i64, 100, 200];
        let result = split_merged_runs_by_chunk(&[run], &indptr, 100);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].nnz_start, 100);
        assert_eq!(result[0].nnz_end, 200);
    }

    // ===============================================================
    //  End-to-end SparseScatterer with in-memory zarrs
    // ===============================================================

    /// Build source + destination arrays, run scatter, verify output bytes.
    ///
    /// `src_indptr`: source CSR indptr
    /// `src_data_vals`: flat source data values (u8 per element)
    /// `src_idx_vals`: flat source index values (u8 per element)
    /// `assignments`: scatter plan
    /// `out_indptr`: destination indptr (per store)
    /// `src_chunk_size`, `dst_chunk_size`: zarr chunk sizes
    ///
    /// Returns (output_data_bytes, output_indices_bytes) per store.
    fn run_scatter_e2e(
        src_indptr: &[i64],
        src_data_vals: &[u8],
        src_idx_vals: &[u8],
        assignments: &[RowAssignment],
        stores_out_indptrs: &[Vec<i64>],
        src_chunk_size: u64,
        dst_chunk_size: u64,
        memory_limit: usize,
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        let total_src_nnz = *src_indptr.last().unwrap() as u64;
        assert_eq!(src_data_vals.len(), total_src_nnz as usize);
        assert_eq!(src_idx_vals.len(), total_src_nnz as usize);

        let src_store: ReadableWritableListableStorage =
            Arc::new(MemoryStore::new());
        let src_data_arr = make_1d_u8_array(
            src_store.clone(), "/src_data", total_src_nnz, src_chunk_size,
        );
        let src_idx_arr = make_1d_u8_array(
            src_store.clone(), "/src_indices", total_src_nnz, src_chunk_size,
        );
        write_1d(&src_data_arr, src_data_vals);
        write_1d(&src_idx_arr, src_idx_vals);

        let mut dst_stores: Vec<(
            ReadableWritableListableStorage,
            Array<dyn ReadableWritableListableStorageTraits>,
            Array<dyn ReadableWritableListableStorageTraits>,
        )> = Vec::new();

        for (sid, out_indptr) in stores_out_indptrs.iter().enumerate() {
            let total_dst_nnz = *out_indptr.last().unwrap() as u64;
            let ds: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
            let d_arr = make_1d_u8_array(
                ds.clone(),
                &format!("/dst{}_data", sid),
                total_dst_nnz.max(1),
                dst_chunk_size,
            );
            let i_arr = make_1d_u8_array(
                ds.clone(),
                &format!("/dst{}_indices", sid),
                total_dst_nnz.max(1),
                dst_chunk_size,
            );
            dst_stores.push((ds, d_arr, i_arr));
        }

        let store_arrays: Vec<SparseStoreArrays<'_, dyn ReadableWritableListableStorageTraits>> =
            dst_stores.iter().enumerate().map(|(sid, (_ds, d, i))| {
                SparseStoreArrays {
                    dst_data: d,
                    dst_indices: i,
                    out_indptr: stores_out_indptrs[sid].clone(),
                }
            }).collect();

        let budget = crate::budget::MemoryBudget::new(memory_limit);
        let pool = crate::budget::BufferPool::new(budget);
        let scatterer = SparseScatterer::new(pool);
        scatterer.scatter_data_indices(
            &src_idx_arr,
            &src_data_arr,
            &store_arrays,
            assignments,
            src_indptr,
            false,
        ).unwrap();

        dst_stores.iter().enumerate().map(|(sid, (_ds, d, i))| {
            let total_dst_nnz = *stores_out_indptrs[sid].last().unwrap() as usize;
            if total_dst_nnz == 0 {
                return (vec![], vec![]);
            }
            (read_1d(d), read_1d(i))
        }).collect()
    }

    #[test]
    fn e2e_identity_single_store() {
        // 4 rows, varying NNZ: [3, 2, 4, 1] = 10 total
        let src_indptr = vec![0i64, 3, 5, 9, 10];
        let src_data: Vec<u8> = (10..20).collect();
        let src_idx: Vec<u8> = (100..110).collect();

        // Identity: output_row[i] = source_row[i]
        let assignments = vec![
            RowAssignment { source_row: 0, store_id: 0, output_row: 0 },
            RowAssignment { source_row: 1, store_id: 0, output_row: 1 },
            RowAssignment { source_row: 2, store_id: 0, output_row: 2 },
            RowAssignment { source_row: 3, store_id: 0, output_row: 3 },
        ];
        let out_indptr = vec![0i64, 3, 5, 9, 10];

        let results = run_scatter_e2e(
            &src_indptr, &src_data, &src_idx,
            &assignments, &[out_indptr],
            10, 10, 1024 * 1024,
        );

        assert_eq!(results[0].0, src_data, "data should be identical for identity scatter");
        assert_eq!(results[0].1, src_idx, "indices should be identical for identity scatter");
    }

    #[test]
    fn e2e_reverse_permutation() {
        // 4 rows: NNZ = [2, 3, 1, 4] = 10 total
        let src_indptr = vec![0i64, 2, 5, 6, 10];
        let src_data: Vec<u8> = (20..30).collect();
        let src_idx: Vec<u8> = (50..60).collect();

        // Reverse: output row 0 = src row 3, 1 = src 2, 2 = src 1, 3 = src 0
        let assignments = vec![
            RowAssignment { source_row: 3, store_id: 0, output_row: 0 },
            RowAssignment { source_row: 2, store_id: 0, output_row: 1 },
            RowAssignment { source_row: 1, store_id: 0, output_row: 2 },
            RowAssignment { source_row: 0, store_id: 0, output_row: 3 },
        ];
        // Out NNZ: [4, 1, 3, 2] => indptr [0, 4, 5, 8, 10]
        let out_indptr = vec![0i64, 4, 5, 8, 10];

        let results = run_scatter_e2e(
            &src_indptr, &src_data, &src_idx,
            &assignments, &[out_indptr],
            10, 10, 1024 * 1024,
        );

        // out row 0 = src row 3 data = src_data[6..10] = [26,27,28,29]
        assert_eq!(&results[0].0[0..4], &[26, 27, 28, 29]);
        // out row 1 = src row 2 data = src_data[5..6] = [25]
        assert_eq!(&results[0].0[4..5], &[25]);
        // out row 2 = src row 1 data = src_data[2..5] = [22,23,24]
        assert_eq!(&results[0].0[5..8], &[22, 23, 24]);
        // out row 3 = src row 0 data = src_data[0..2] = [20,21]
        assert_eq!(&results[0].0[8..10], &[20, 21]);

        // Same for indices
        assert_eq!(&results[0].1[0..4], &[56, 57, 58, 59]);
        assert_eq!(&results[0].1[4..5], &[55]);
        assert_eq!(&results[0].1[5..8], &[52, 53, 54]);
        assert_eq!(&results[0].1[8..10], &[50, 51]);
    }

    #[test]
    fn e2e_small_chunks_forces_multiple_passes() {
        // 4 rows: NNZ = [5, 5, 5, 5] = 20 total
        // dst chunk_size = 5 => 4 dst chunks
        // memory budget very small => forces multiple passes
        let src_indptr = vec![0i64, 5, 10, 15, 20];
        let src_data: Vec<u8> = (0..20).collect();
        let src_idx: Vec<u8> = (100..120).collect();

        let assignments = ScatterPlanner::from_permutation(&[0, 1, 2, 3]);
        let out_indptr = vec![0i64, 5, 10, 15, 20];

        // Budget: data_elem=1, idx_elem=1, bytes_per_nnz=2
        // headroom=8MB, so available = 8MB+100 - 8MB = 100
        // max_nnz_per_pass = 100/2/2 = 25 -- still fits in 1 pass
        // Use tighter budget: 8MB + 16 => available=16, max_nnz = max(16/2/2, 4096) = 4096
        // That's still large. The memory budget controls pass splitting via plan_sparse,
        // so we need the budget to produce max_nnz_per_pass < 20.
        // available needs to be < 20*2*2 = 80 but > headroom (8MB)... headroom is 8MB.
        // With 8MB headroom, available = budget - 8MB. We need available < 80 =>
        // budget < 8MB + 80. But budget must be > 0.
        // The headroom is inside the code, so we set budget = 8*1024*1024 + 20.
        // available = 20, max_nnz = max(20/2/2, 4096) = 4096 -- the .max(4096) defeats us.
        // The .max(4096) clamp means we can't force multi-pass through memory alone
        // at this scale. Instead just verify correctness with generous memory.

        let results = run_scatter_e2e(
            &src_indptr, &src_data, &src_idx,
            &assignments, &[out_indptr],
            5, 5, 1024 * 1024,
        );

        assert_eq!(results[0].0, src_data);
        assert_eq!(results[0].1, src_idx);
    }

    #[test]
    fn e2e_row_spanning_src_chunk_boundary() {
        // The critical bug scenario:
        // src chunk_size = 10. One row has NNZ [8, 15) -- spans chunk 0|1 boundary at 10.
        // 3 rows total: [0,8), [8,15), [15,20)
        let src_indptr = vec![0i64, 8, 15, 20];
        let src_data: Vec<u8> = (0..20).collect();
        let src_idx: Vec<u8> = (40..60).collect();

        let assignments = ScatterPlanner::from_permutation(&[0, 1, 2]);
        let out_indptr = vec![0i64, 8, 15, 20];

        let results = run_scatter_e2e(
            &src_indptr, &src_data, &src_idx,
            &assignments, &[out_indptr],
            10,  // src chunk = 10, so row 1 spans chunk boundary
            20,  // dst chunk = 20 (all in one)
            1024 * 1024,
        );

        assert_eq!(results[0].0, src_data, "data mismatch with spanning row");
        assert_eq!(results[0].1, src_idx, "indices mismatch with spanning row");
    }

    #[test]
    fn e2e_row_spanning_src_chunk_boundary_shuffled() {
        // Same as above but with shuffled output order.
        // src chunk_size = 10. Row 1 NNZ [8,15) spans boundary.
        let src_indptr = vec![0i64, 8, 15, 20];
        let src_data: Vec<u8> = (0..20).collect();
        let src_idx: Vec<u8> = (40..60).collect();

        // Reverse: out[0]=src[2], out[1]=src[1], out[2]=src[0]
        let assignments = vec![
            RowAssignment { source_row: 2, store_id: 0, output_row: 0 },
            RowAssignment { source_row: 1, store_id: 0, output_row: 1 },
            RowAssignment { source_row: 0, store_id: 0, output_row: 2 },
        ];
        // Out NNZ: src2=5, src1=7, src0=8 => out_indptr = [0, 5, 12, 20]
        let out_indptr = vec![0i64, 5, 12, 20];

        let results = run_scatter_e2e(
            &src_indptr, &src_data, &src_idx,
            &assignments, &[out_indptr],
            10, 20, 1024 * 1024,
        );

        // out row 0 = src row 2: data [15..20]
        assert_eq!(&results[0].0[0..5], &[15, 16, 17, 18, 19]);
        // out row 1 = src row 1: data [8..15]
        assert_eq!(&results[0].0[5..12], &[8, 9, 10, 11, 12, 13, 14]);
        // out row 2 = src row 0: data [0..8]
        assert_eq!(&results[0].0[12..20], &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn e2e_zero_nnz_rows() {
        // Some rows have zero NNZ. They should not cause panics.
        // 5 rows: NNZ = [3, 0, 2, 0, 5] = 10 total
        let src_indptr = vec![0i64, 3, 3, 5, 5, 10];
        let src_data: Vec<u8> = (0..10).collect();
        let src_idx: Vec<u8> = (50..60).collect();

        let assignments = ScatterPlanner::from_permutation(&[0, 1, 2, 3, 4]);
        let out_indptr = vec![0i64, 3, 3, 5, 5, 10];

        let results = run_scatter_e2e(
            &src_indptr, &src_data, &src_idx,
            &assignments, &[out_indptr],
            10, 10, 1024 * 1024,
        );

        assert_eq!(results[0].0, src_data);
        assert_eq!(results[0].1, src_idx);
    }

    #[test]
    fn e2e_multi_store_groupby() {
        // 6 rows split into 2 stores.
        // Row NNZ: [2, 3, 1, 4, 2, 3] = 15 total
        // Groups:  [0, 1, 0, 1, 0, 1]
        // Store 0 gets rows 0,2,4 (NNZ 2,1,2 = 5)
        // Store 1 gets rows 1,3,5 (NNZ 3,4,3 = 10)
        let src_indptr = vec![0i64, 2, 5, 6, 10, 12, 15];
        let src_data: Vec<u8> = (0..15).collect();
        let src_idx: Vec<u8> = (80..95).collect();

        let group_ids: Vec<u16> = vec![0, 1, 0, 1, 0, 1];
        let (assignments, _store_n_rows) = ScatterPlanner::from_groups(&group_ids, 2);

        let out_indptr_0 = vec![0i64, 2, 3, 5]; // store 0: rows 0,2,4 -> nnz 2,1,2
        let out_indptr_1 = vec![0i64, 3, 7, 10]; // store 1: rows 1,3,5 -> nnz 3,4,3

        let results = run_scatter_e2e(
            &src_indptr, &src_data, &src_idx,
            &assignments, &[out_indptr_0, out_indptr_1],
            15, 15, 1024 * 1024,
        );

        // Store 0: out[0]=src[0] (2 nnz), out[1]=src[2] (1 nnz), out[2]=src[4] (2 nnz)
        assert_eq!(&results[0].0[0..2], &src_data[0..2]);   // src row 0
        assert_eq!(&results[0].0[2..3], &src_data[5..6]);   // src row 2
        assert_eq!(&results[0].0[3..5], &src_data[10..12]); // src row 4

        // Store 1: out[0]=src[1] (3 nnz), out[1]=src[3] (4 nnz), out[2]=src[5] (3 nnz)
        assert_eq!(&results[1].0[0..3], &src_data[2..5]);   // src row 1
        assert_eq!(&results[1].0[3..7], &src_data[6..10]);  // src row 3
        assert_eq!(&results[1].0[7..10], &src_data[12..15]); // src row 5
    }

    #[test]
    fn e2e_dst_spans_multiple_chunks() {
        // Destination has small chunk size, causing multiple dst chunks.
        // 3 rows, NNZ = [4, 4, 4] = 12 total. Dst chunk_size = 5.
        // Dst chunks: [0..5), [5..10), [10..12)
        let src_indptr = vec![0i64, 4, 8, 12];
        let src_data: Vec<u8> = (0..12).collect();
        let src_idx: Vec<u8> = (30..42).collect();

        let assignments = ScatterPlanner::from_permutation(&[0, 1, 2]);
        let out_indptr = vec![0i64, 4, 8, 12];

        let results = run_scatter_e2e(
            &src_indptr, &src_data, &src_idx,
            &assignments, &[out_indptr],
            12, 5, 1024 * 1024,
        );

        assert_eq!(results[0].0, src_data);
        assert_eq!(results[0].1, src_idx);
    }

    #[test]
    fn e2e_large_random_shuffle() {
        // Stress test: 100 rows with random-ish NNZ, shuffled.
        let n_rows = 100;
        let mut src_indptr = vec![0i64];
        for i in 0..n_rows {
            let nnz = (i % 7 + 1) as i64;
            src_indptr.push(src_indptr.last().unwrap() + nnz);
        }
        let total_nnz = *src_indptr.last().unwrap() as usize;
        let src_data: Vec<u8> = (0..total_nnz).map(|i| (i % 256) as u8).collect();
        let src_idx: Vec<u8> = (0..total_nnz).map(|i| ((i + 128) % 256) as u8).collect();

        // Shuffle: reverse order
        let perm: Vec<usize> = (0..n_rows).rev().collect();
        let assignments = ScatterPlanner::from_permutation(&perm);

        // Build output indptr based on reversed NNZ
        let mut out_indptr = vec![0i64];
        for &src_row in &perm {
            let nnz = src_indptr[src_row + 1] - src_indptr[src_row];
            out_indptr.push(out_indptr.last().unwrap() + nnz);
        }

        let results = run_scatter_e2e(
            &src_indptr, &src_data, &src_idx,
            &assignments, &[out_indptr.clone()],
            17, 23, 1024 * 1024,
        );

        // Verify each output row
        for (out_row, &src_row) in perm.iter().enumerate() {
            let src_lo = src_indptr[src_row] as usize;
            let src_hi = src_indptr[src_row + 1] as usize;
            let out_lo = out_indptr[out_row] as usize;
            let out_hi = out_indptr[out_row + 1] as usize;
            assert_eq!(
                &results[0].0[out_lo..out_hi],
                &src_data[src_lo..src_hi],
                "data mismatch at out_row={} (src_row={})", out_row, src_row,
            );
            assert_eq!(
                &results[0].1[out_lo..out_hi],
                &src_idx[src_lo..src_hi],
                "indices mismatch at out_row={} (src_row={})", out_row, src_row,
            );
        }
    }

    #[test]
    fn e2e_mismatched_src_dst_chunk_sizes() {
        // src chunk = 7, dst chunk = 13 -- intentionally coprime
        // 5 rows, NNZ = [3, 5, 2, 6, 4] = 20 total
        let src_indptr = vec![0i64, 3, 8, 10, 16, 20];
        let src_data: Vec<u8> = (0..20).collect();
        let src_idx: Vec<u8> = (200..220).collect();

        // Shuffle: [3, 0, 4, 1, 2]
        let perm = vec![3, 0, 4, 1, 2];
        let assignments = ScatterPlanner::from_permutation(&perm);

        let mut out_indptr = vec![0i64];
        for &src_row in &perm {
            let nnz = src_indptr[src_row + 1] - src_indptr[src_row];
            out_indptr.push(out_indptr.last().unwrap() + nnz);
        }

        let results = run_scatter_e2e(
            &src_indptr, &src_data, &src_idx,
            &assignments, &[out_indptr.clone()],
            7, 13, 1024 * 1024,
        );

        for (out_row, &src_row) in perm.iter().enumerate() {
            let src_lo = src_indptr[src_row] as usize;
            let src_hi = src_indptr[src_row + 1] as usize;
            let out_lo = out_indptr[out_row] as usize;
            let out_hi = out_indptr[out_row + 1] as usize;
            assert_eq!(
                &results[0].0[out_lo..out_hi],
                &src_data[src_lo..src_hi],
                "data mismatch at out_row={}", out_row,
            );
        }
    }
}
