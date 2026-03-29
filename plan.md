# Chunk Passthrough Optimization for scatter_engine

## Problem

Even for trivial operations like truncation (`arange(10M)`), the scatter engine
decompresses every source chunk, copies the raw bytes into output buffers, then
recompresses and writes new chunks. For Tahoe's 10M-row truncation this means
113 GB of NNZ data is decoded and re-encoded despite being a contiguous prefix
copy. At 27 MB/s observed throughput, that takes ~70 minutes.

## Insight

When a full source chunk maps 1:1 into a destination chunk with the same codec
pipeline and chunk shape, the compressed bytes on disk can be copied verbatim --
no decode, no re-encode, no memory allocation beyond a small transfer buffer.
zarrs 0.23 already exposes this:

```rust
// Read compressed bytes (no decode)
let encoded: Option<Vec<u8>> = src.retrieve_encoded_chunk(&chunk_idx)?;

// Write compressed bytes (no encode) -- unsafe: caller guarantees encoding
unsafe { dst.store_encoded_chunk(&chunk_idx, encoded.into())? };
```

## When passthrough is valid

A source chunk can be passed through if **all** of these hold:

1. **Full chunk coverage**: every element in the source chunk is included in the
   output, and maps to the same local position within the destination chunk
   (i.e. identity mapping for that chunk).
2. **Same chunk shape**: source and destination use the same chunk grid shape
   along all dimensions.
3. **Same codec chain**: source and destination have identical codecs (same
   compression algorithm, level, filters). If the destination is created by
   cloning the source's `ArrayMetadataV3` and only changing the array shape,
   this is automatically true.
4. **Same data type and fill value**: guaranteed when we create the destination
   from the source's metadata.

For truncation (`arange(N)`), conditions 1-4 hold for every chunk whose row
range `[chunk_start, chunk_end)` is fully within `[0, N)`. Only the last
partial chunk needs the decode/scatter/encode path.

For shuffle, very few chunks will qualify (only if a random permutation happens
to map an entire chunk identically, which is astronomically unlikely). So the
optimization is a no-op for shuffle, adding zero overhead beyond a cheap check.

## Scope of changes

### 1. Detect identity-mapped chunks (both dense and sparse)

Given the sorted assignments for a batch/pass, identify chunks where:
- All rows in the chunk are present in the assignment
- `source_row == output_row` for every row in the chunk (identity)
- The chunk is not a boundary chunk (all rows fit)

For the sparse path, this means identifying NNZ ranges that correspond to full
source chunks of both `data` and `indices` arrays.

### 2. Create destination arrays with matching codec/chunk config

Currently, `scatter_csr_group` creates destination `data` and `indices` arrays
via `new_empty_dataset_typed` which uses `get_default_write_config()` -- this
may produce different chunk sizes or codecs than the source.

**Change**: read the source array's metadata (chunk shape, codecs) and clone it
for the destination, only adjusting the array shape. zarrs `Array` exposes
`metadata()` which returns `ArrayMetadataV3` containing the full codec chain.

```rust
// Pseudocode
let src_meta = src_data.metadata();
let mut dst_meta = src_meta.clone();
dst_meta.shape = vec![total_nnz as u64]; // only change shape
let dst_data = Array::new_with_metadata(dst_store, "X/data", dst_meta)?;
```

This guarantees conditions 2-4 automatically.

### 3. Dense passthrough (`dense_scatter.rs`)

In `DenseScatterer::scatter`, before `process_pass`:

```
for each pass:
    partition chunks into:
        passthrough_chunks: full identity-mapped chunks
        scatter_chunks: everything else

    for each passthrough chunk:
        encoded = src.retrieve_encoded_chunk(&chunk_idx)?
        if let Some(bytes) = encoded:
            unsafe { dst.store_encoded_chunk(&dst_chunk_idx, bytes)? }

    process_pass(src, dsts, scatter_only_pass, n_cols)?
```

### 4. Sparse passthrough (`sparse_scatter.rs`)

More subtle because CSR `data`/`indices` are 1D arrays indexed by NNZ position,
not by row. A "chunk" in the NNZ dimension corresponds to a variable number of
rows depending on per-row sparsity.

For contiguous source rows mapping to contiguous output rows at the same
positions (i.e. identity prefix), the NNZ ranges are also identical. So:

```
for each batch:
    for each source chunk in data/indices whose NNZ range is fully
    covered by identity-mapped rows in this batch:
        encoded_data = src_data.retrieve_encoded_chunk(&[chunk_idx])?
        encoded_indices = src_indices.retrieve_encoded_chunk(&[chunk_idx])?
        unsafe {
            dst_data.store_encoded_chunk(&[chunk_idx], encoded_data)?
            dst_indices.store_encoded_chunk(&[chunk_idx], encoded_indices)?
        }

    run normal decode/scatter/encode path for remaining NNZ ranges
```

The key check: for a 1D chunk `[nnz_start, nnz_end)` in the source, find all
rows whose NNZ falls in that range. If every such row is identity-mapped
(source_row == output_row), the chunk can be copied verbatim.

### 5. No-compression passthrough

When the source store has no compression (raw/uncompressed codec), the encoded
bytes *are* the raw bytes. `retrieve_encoded_chunk` / `store_encoded_chunk`
still work and skip codec overhead entirely. This is the fastest possible path.

The same logic applies: if chunks align, copy the blob. The only difference is
that uncompressed chunks are larger on the wire, but there's no CPU overhead.

### 6. `copy_group` optimization

The existing `copy_group` (used for `var`, `uns`, `varm`, `varp`) currently does
`Data::read` + `Data::write`, which decodes and re-encodes. This should also
use chunk-level passthrough since these groups are copied identically. This is
independent of the scatter optimization but follows the same principle:

```rust
fn copy_group_raw(src_arr: &Array<S>, dst_arr: &Array<S>) -> Result<()> {
    for chunk_idx in src_arr.chunk_indices() {
        if let Some(bytes) = src_arr.retrieve_encoded_chunk(&chunk_idx)? {
            unsafe { dst_arr.store_encoded_chunk(&chunk_idx, bytes)?; }
        }
    }
    Ok(())
}
```

## Expected impact

### Truncation (first 10M rows from 89M, Tahoe)

- **Before**: 113 GB decoded + 113 GB re-encoded = 226 GB through CPU, ~70 min
- **After**: ~112.9 GB chunk-copied as compressed blobs (~242 GB on disk but no
  CPU work), only the last partial chunk decoded/encoded. Dominated by Lustre
  sequential read+write bandwidth. At ~1 GB/s aggregate (10 stripes), ~4 min.
- **Speedup**: ~15-20x

### Shuffle

- No chunks qualify for passthrough (random permutation)
- Zero overhead: the identity check is O(chunk_count) and very cheap
- No regression

### Uncompressed stores

- Same passthrough logic applies
- Even faster since chunks are already raw bytes
- Useful for intermediate/scratch stores where compression is unnecessary

## Implementation order

1. **Metadata cloning for destination arrays** -- ensure src/dst chunk shapes
   and codecs match. This is a prerequisite for all passthrough.
2. **Dense passthrough** -- simpler to implement, test with a dense obsm matrix.
3. **Sparse (CSR) passthrough** -- the high-value target for Tahoe truncation.
4. **`copy_group` raw passthrough** -- independent, low-hanging fruit.
5. **Benchmarks** -- add truncation to `bench_shuffle.py` to measure speedup.

## Files to modify

- `anndata-ooc/src/scatter_engine.rs` -- metadata cloning, `copy_group` raw path
- `anndata-ooc/src/dense_scatter.rs` -- identity-chunk detection + passthrough
- `anndata-ooc/src/sparse_scatter.rs` -- identity-NNZ-chunk detection + passthrough
- `anndata-ooc/src/lib.rs` -- re-export any new config if needed

## Risks and mitigations

- **`store_encoded_chunk` is `unsafe`**: we must guarantee the bytes are valid
  for the destination's codec chain. By cloning source metadata, this is
  inherently safe. Add a debug assertion comparing codec metadata.
- **Sharded arrays**: if source uses sharding, `retrieve_encoded_chunk` returns
  the full shard blob. Passthrough still works if the entire shard is identity-
  mapped, but partial shards cannot be passed through. The fallback path handles
  this correctly.
- **Different zarr versions (v2 vs v3)**: chunk key encoding differs. Both
  source and destination should use the same zarr version. The current codebase
  uses zarrs which handles this transparently.
e