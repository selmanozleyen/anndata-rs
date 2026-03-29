# Proposal: Parallel Codec Pipeline for anndata-rs scatter_engine

## Motivation: Real-World Profiling Shows 99% of Cores Idle

We profiled a shuffle of the Tahoe 10M-row dataset (15.2B NNZ, 48 GB
compressed) on a 128-core node with 755 GB RAM and Lustre storage:

```
Machine: 128 cores, 755 GB RAM, Lustre (10-stripe PFL, 1 MB stripe)

$ ps -T -p 898109 -o pid,spid,pcpu,stat

    PID    SPID %CPU STAT
 898109  898109 96.8 RNl+    <-- the only running thread
 898109  898110  0.0 SNl+    <-- 35 sleeping threads (rayon pool)
 898109  898111  0.0 SNl+
 ...
 (36 threads total, 35 sleeping)

Aggregate: 106.4% CPU out of 12,800% available (128 cores)
           = 0.8% utilization
```

The process used **1 core** out of 128 for the entire 4-minute run.
htop confirmed: 98.3% CPU on a single core, `R` (running) state, 21.4 GB RSS.

### Observed timings on Tahoe (10M rows, 15.2B NNZ)

| Operation | Time | Throughput | CPU% | Method |
|-----------|------|------------|------|--------|
| Fast truncation (chunk copy, no codec) | 167s | 294 MB/s | <5% | `shutil.copy2` in Python |
| Shuffle (anndata_rs.permute, 20 GB mem) | 239s | ~21 MB/s | 98% single-core | Rust scatter engine |

The fast truncation proves Lustre can sustain ~450 MB/s sequential I/O.
The shuffle is 14x slower despite moving less data, because it is
single-core CPU-bound on blosc decompress/compress.

### I/O simulation vs reality

Our I/O simulation (which faithfully mirrors the Rust batching/merging
logic) predicted 6 batches with 1.00x read amplification for both
truncate and shuffle at 20 GB memory. The actual `/proc/<pid>/io`
reported 7.5 GB read and 4.9 GB write -- confirming that the compressed
I/O volume is modest and Lustre is not the bottleneck.

## Root Cause Analysis

### Where parallelism exists today

1. **`per_store.par_iter()`** in `sparse_scatter.rs:154` -- parallelizes
   writes across multiple output stores. With `permute` (1 output store),
   this provides zero parallelism.

2. **`chunk_buffers.par_iter()`** in `dense_scatter.rs:146` -- flushes
   dense chunk buffers in parallel. This helps for dense arrays but not
   CSR.

### Where parallelism is missing

The entire hot path for CSR scatter is sequential within each batch:

```
for batch in batches:            // sequential
    for run in merged_reads:     // sequential
        retrieve_array_subset()  // blosc DECODE (single-threaded)
        slice into row_map       // memcpy (fast)

    for run in output_runs:      // sequential
        assemble rows            // memcpy (fast)
        store_array_subset()     // blosc ENCODE (single-threaded)
```

`retrieve_array_subset` and `store_array_subset` each touch a single
chunk at a time and call blosc decode/encode internally. zarrs does not
parallelize across chunks within a single `retrieve/store` call for
1D arrays.

## Proposed Changes

### Phase 1: Parallel chunk reads within a batch

In `process_batch`, the merged reads are independent -- each reads a
disjoint NNZ range. These can be parallelized with rayon:

```rust
// Before (sequential)
for run in &merged {
    let data_bytes = src_data.retrieve_array_subset(&subset)?;
    let indices_bytes = src_indices.retrieve_array_subset(&subset)?;
    // ... slice into row_map
}

// After (parallel)
let decoded_runs: Vec<_> = merged.par_iter().map(|run| {
    let subset = ArraySubset::new_with_ranges(
        &[run.nnz_start as u64..run.nnz_end as u64],
    );
    let data_bytes = src_data.retrieve_array_subset(&subset)?;
    let indices_bytes = src_indices.retrieve_array_subset(&subset)?;
    Ok((run, data_bytes, indices_bytes))
}).collect::<Result<Vec<_>>>()?;

// Then build row_map from decoded_runs (single-threaded, cheap memcpy)
```

**Expected impact**: With 6 batches and merged reads covering ~2.5B NNZ
each, there are typically 1-2 merged read runs per batch for truncation
(contiguous) but many for shuffle (fragmented). For shuffle, this
parallelizes blosc decode across cores.

**Risk**: Memory usage increases because all decoded data for a batch is
held simultaneously. With max_nnz_per_batch already constrained by the
memory budget, this should be safe -- the batch was sized to fit in RAM.

### Phase 2: Parallel chunk writes within a batch

Similarly, the output write runs are independent per-store (already
parallelized across stores) but sequential within a store. For a single
output store with many fragmented write runs (shuffle produces ~8M
runs), the writes can be parallelized:

```rust
// Before (sequential within a store)
for run in &runs {
    // assemble rows
    stores[store_id].dst_data.store_array_subset(&write_subset, ...)?;
    stores[store_id].dst_indices.store_array_subset(&write_subset, ...)?;
}

// After: group runs into chunks that map to the same zarr chunk,
// then parallelize across zarr chunks
let chunk_groups = group_runs_by_dst_chunk(&runs, chunk_size);
chunk_groups.par_iter().try_for_each(|group| {
    for run in group {
        // assemble + store
    }
    Ok(())
})?;
```

**Expected impact**: For shuffle, write runs are small and numerous.
Grouping them by destination chunk and parallelizing across chunks
distributes blosc encode work. With 226 destination chunks (for
the data array), this gives up to 226-way parallelism.

**Risk**: zarrs `store_array_subset` on non-overlapping subsets of
different chunks should be thread-safe (each chunk is an independent
file on disk). Need to verify zarrs thread-safety guarantees for
concurrent writes to different chunks of the same Array.

### Phase 3: Pipeline reads and writes across batches

Currently batches are fully sequential: read all, then write all, then
move to next batch. A producer-consumer pipeline would overlap I/O:

```
Batch 1: [READ ][WRITE]
Batch 2:        [READ ][WRITE]

becomes:

Batch 1: [READ ][WRITE]
Batch 2:  [READ ][WRITE]
              ^-- overlap: batch 2 reads while batch 1 writes
```

This requires double-buffering: while one batch's data is being written,
the next batch's data is being read. The memory budget already accounts
for one batch; we'd need to allocate for two (or reduce batch size by
half).

**Expected impact**: Hides read latency behind write time. Most
beneficial when read and write times are balanced. For shuffle at 20 GB
mem limit, each batch processes ~2.5B NNZ -- overlapping would save
~15-30% wall time.

**Risk**: Doubles peak memory for the overlap window. At 20 GB limit
this means effectively 10 GB per batch, increasing batch count from 6
to 12. Net benefit depends on whether the I/O overlap outweighs the
extra batch overhead.

### Phase 4: Parallel codec via zarrs async or thread pool

zarrs supports `par_retrieve_array_subset` for parallel chunk decoding
within a single read operation. If the merged read covers multiple
chunks, this would parallelize the blosc decode calls:

```rust
// Instead of retrieve_array_subset (sequential chunks):
let data_bytes = src_data.par_retrieve_array_subset(&subset)?;
```

This is the simplest change and potentially the highest impact: it
parallelizes the codec at the zarrs level without restructuring the
scatter engine.

**Expected impact**: If a single merged read spans N chunks, blosc
decode is parallelized N-ways. For Tahoe's 67M-element chunks, a batch
of 2.5B NNZ spans ~37 chunks -- so up to 37x speedup on decode.

**Risk**: Need to verify `par_retrieve_array_subset` exists in the
zarrs version we depend on (it was added in zarrs 0.18+).

## Implementation Priority

| Phase | Effort | Impact | Risk |
|-------|--------|--------|------|
| Phase 4 (par_retrieve/store) | Low | High | Check zarrs API availability |
| Phase 1 (parallel reads) | Medium | High | Safe (independent ranges) |
| Phase 2 (parallel writes) | Medium | Medium | Verify zarrs thread safety |
| Phase 3 (pipeline batches) | High | Medium | Memory pressure |

**Recommended order**: 4 -> 1 -> 2 -> 3

Phase 4 alone could yield 10-30x speedup on the codec-bound path
with minimal code changes. Phases 1 and 2 provide additional gains
when zarrs internal parallelism is insufficient.

## Expected Results

### Conservative (Phase 4 only)

With par_retrieve/store using 16 threads on a 128-core node:

| Operation | Current | Expected | Speedup |
|-----------|---------|----------|---------|
| Shuffle 10M | 239s | ~30-50s | 5-8x |
| Shuffle 89M (full Tahoe) | ~35 min (est) | ~5-7 min | 5-7x |

### Optimistic (Phases 1-4)

With full pipeline parallelism saturating cores and I/O:

| Operation | Current | Expected | Speedup |
|-----------|---------|----------|---------|
| Shuffle 10M | 239s | ~15-25s | 10-16x |
| Shuffle 89M | ~35 min | ~3-5 min | 7-12x |

The theoretical floor is the Lustre I/O bandwidth. At 450 MB/s
(measured with fast_truncate) and ~48 GB compressed output, the minimum
wall time is ~107s for 10M rows. With codec parallelism, the CPU
ceases to be the bottleneck and I/O becomes the limiter.

## Interaction with Chunk Passthrough

The chunk passthrough optimization (see `plans/chunk_passthrough.md`)
eliminates codec work entirely for identity-mapped chunks. Parallelism
and passthrough are complementary:

- **Truncation**: passthrough handles 226/227 chunks with zero codec;
  parallelism helps with the 1 remaining partial chunk (negligible).
  Passthrough alone is sufficient for truncation.

- **Shuffle**: passthrough helps 0 chunks (random permutation);
  parallelism is the only way to speed up shuffle.

Both should be implemented. Passthrough is a pure win for contiguous
operations; parallelism is a pure win for random operations.

## Files to Modify

- `anndata-ooc/src/sparse_scatter.rs` -- parallel merged reads & writes
- `anndata-ooc/src/dense_scatter.rs` -- parallel chunk buffer flushes
  (already partially done)
- `anndata-ooc/Cargo.toml` -- possibly enable zarrs parallel features
- `anndata-ooc/src/budget.rs` -- double-buffer support for Phase 3
