use std::sync::Arc;
use std::time::Instant;

use anndata_ooc::{
    RowAssignment, MemoryBudget, BufferPool,
    SparseScatterer, SparseStoreArrays,
};

use zarrs::array::{Array, ArrayBuilder, ArrayBytes, ArraySubset};
use zarrs::array::data_type;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits};

fn make_1d_array(
    store: ReadableWritableListableStorage,
    path: &str,
    len: u64,
    chunk_size: u64,
) -> Array<dyn ReadableWritableListableStorageTraits> {
    let cs = vec![chunk_size.min(len).max(1)];
    let builder = ArrayBuilder::new(vec![len], cs, data_type::uint8(), 0u8);
    let arr = builder.build(store, path).unwrap();
    arr.store_metadata().unwrap();
    arr
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    // Simulate the user's dataset at 1/10 scale to keep it quick but
    // expose the same algorithmic costs.
    //
    // Real:  10M rows,  15.2B NNZ,  349310 chunk, ~43K chunks
    // Bench:  1M rows, 1.52B NNZ  -- too big for memory
    //
    // Use 1M rows, ~150 NNZ/row => 150M NNZ total.
    // That's 150M * 1 byte(u8) = 150 MB per array, fits in RAM.
    let n_rows: usize = std::env::var("BENCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);
    let avg_nnz: usize = std::env::var("BENCH_AVG_NNZ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(150);
    let src_chunk: usize = std::env::var("BENCH_SRC_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(349_310);
    let dst_chunk: usize = std::env::var("BENCH_DST_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(349_310);
    let memory_limit: usize = std::env::var("BENCH_MEM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2 * 1024 * 1024 * 1024);

    let total_nnz = n_rows * avg_nnz;

    eprintln!("=== Profile scatter benchmark ===");
    eprintln!("  rows:        {}", n_rows);
    eprintln!("  avg_nnz:     {}", avg_nnz);
    eprintln!("  total_nnz:   {} ({:.1} MB)", total_nnz, total_nnz as f64 / 1e6);
    eprintln!("  src_chunk:   {}", src_chunk);
    eprintln!("  dst_chunk:   {}", dst_chunk);
    eprintln!("  memory:      {:.1} GB", memory_limit as f64 / 1e9);

    // Build indptr (uniform NNZ per row for simplicity)
    let t0 = Instant::now();
    let mut src_indptr = vec![0i64; n_rows + 1];
    for i in 0..n_rows {
        src_indptr[i + 1] = src_indptr[i] + avg_nnz as i64;
    }

    // Random permutation
    let mut perm: Vec<usize> = (0..n_rows).collect();
    {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        for i in (1..perm.len()).rev() {
            let mut h = DefaultHasher::new();
            i.hash(&mut h);
            42u64.hash(&mut h);
            let j = (h.finish() as usize) % (i + 1);
            perm.swap(i, j);
        }
    }

    let assignments: Vec<RowAssignment> = perm.iter().enumerate().map(|(out, &src)| {
        RowAssignment { source_row: src, store_id: 0, output_row: out }
    }).collect();
    eprintln!("  setup:       {:.2}s", t0.elapsed().as_secs_f64());

    // -- Create in-memory Zarr arrays --
    let t1 = Instant::now();
    let src_store: ReadableWritableListableStorage =
        Arc::new(MemoryStore::default());
    let src_data = make_1d_array(
        src_store.clone(), "/src_data", total_nnz as u64, src_chunk as u64,
    );
    let src_indices = make_1d_array(
        src_store.clone(), "/src_idx", total_nnz as u64, src_chunk as u64,
    );

    // Fill source with sequential bytes so we can verify correctness
    let fill: Vec<u8> = (0..total_nnz).map(|i| (i % 251) as u8).collect();
    src_data.store_array_subset(
        &ArraySubset::new_with_ranges(&[0..total_nnz as u64]),
        ArrayBytes::from(fill.clone()),
    ).unwrap();
    src_indices.store_array_subset(
        &ArraySubset::new_with_ranges(&[0..total_nnz as u64]),
        ArrayBytes::from(fill),
    ).unwrap();

    let dst_store: ReadableWritableListableStorage =
        Arc::new(MemoryStore::default());
    let dst_data = make_1d_array(
        dst_store.clone(), "/dst_data", total_nnz as u64, dst_chunk as u64,
    );
    let dst_indices = make_1d_array(
        dst_store.clone(), "/dst_idx", total_nnz as u64, dst_chunk as u64,
    );

    let out_indptr = src_indptr.clone(); // same shape for 1:1 permutation
    eprintln!("  arrays:      {:.2}s", t1.elapsed().as_secs_f64());

    let store_arrays = vec![SparseStoreArrays {
        dst_data: &dst_data,
        dst_indices: &dst_indices,
        out_indptr,
    }];

    // -- Run the scatter --
    let budget = MemoryBudget::new(memory_limit);
    let pool = BufferPool::new(budget);
    let scatterer = SparseScatterer::new(pool, None, anndata_ooc::SparsePlannerMode::Auto);

    eprintln!("\n--- scatter_data_indices ---");
    let t2 = Instant::now();
    scatterer.scatter_data_indices(
        &src_indices,
        &src_data,
        &store_arrays,
        &assignments,
        &src_indptr,
        false,
    ).unwrap();
    let elapsed = t2.elapsed().as_secs_f64();
    eprintln!("  TOTAL:       {:.2}s", elapsed);
    eprintln!("  throughput:  {:.1} MB/s", total_nnz as f64 * 2.0 / 1e6 / elapsed);
}
