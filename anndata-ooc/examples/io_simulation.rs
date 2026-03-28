/// Simulates the exact I/O pattern of the shard-aware permutation engine
/// on realistic datasets, printing every read and write to stdout.
///
/// No actual Zarr I/O happens -- this is a pure trace of what *would* happen.
///
/// Run with: cargo run -p anndata-ooc --example io_simulation

use std::collections::BTreeMap;

fn main() {
    let scenarios: Vec<(&str, usize, usize, usize, usize, usize, usize)> = vec![
        //                  n_rows   n_cols  elem mem_GB chunk  gap
        ("100K x 30K, 2GB", 100_000, 30_000, 4,   2,    1024,  4),
        ("100K x 30K, 4GB", 100_000, 30_000, 4,   4,    1024,  4),
        ("100K x 30K, 8GB", 100_000, 30_000, 4,   8,    1024,  4),
        ("1M x 30K,   8GB", 1_000_000, 30_000, 4, 8,    1024,  4),
        ("1M x 30K,  32GB", 1_000_000, 30_000, 4, 32,   1024,  4),
        ("1M x 30K,  64GB", 1_000_000, 30_000, 4, 64,   1024,  4),
    ];

    for (label, n_rows, n_cols, elem_size, mem_gb, chunk_size_row, merge_gap) in &scenarios {
        let memory_limit = mem_gb * 1024 * 1024 * 1024;
        run_simulation(label, *n_rows, *n_cols, *elem_size, memory_limit, *chunk_size_row, *merge_gap);
        println!("{}", "-".repeat(80));
    }
}

fn run_simulation(
    label: &str,
    n_rows: usize,
    n_cols: usize,
    elem_size: usize,
    memory_limit: usize,
    chunk_size_row: usize,
    merge_gap: usize,
) {
    let row_bytes = n_cols * elem_size;
    let chunk_bytes = chunk_size_row * row_bytes;
    let total_data = n_rows * row_bytes;

    // Generate a random permutation (Fisher-Yates with simple LCG)
    let mut permutation: Vec<usize> = (0..n_rows).collect();
    let mut rng_state: u64 = 42;
    for i in (1..n_rows).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng_state >> 33) as usize % (i + 1);
        permutation.swap(i, j);
    }

    // Partition by output chunk
    let mut chunk_map: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for (out_row, &src_row) in permutation.iter().enumerate() {
        let chunk_idx = out_row / chunk_size_row;
        chunk_map.entry(chunk_idx).or_default().push((out_row, src_row));
    }

    let n_chunks = chunk_map.len();

    let headroom = 4 * 1024 * 1024;
    let usable_memory = memory_limit - headroom;
    let max_rows_in_memory = usable_memory / row_bytes;
    let max_chunks_per_pass = (max_rows_in_memory / chunk_size_row).max(1);

    // Pack chunks into passes
    let chunk_indices: Vec<usize> = chunk_map.keys().copied().collect();
    let mut passes: Vec<Vec<usize>> = Vec::new();
    let mut i = 0;
    while i < chunk_indices.len() {
        let end = (i + max_chunks_per_pass).min(chunk_indices.len());
        passes.push(chunk_indices[i..end].to_vec());
        i = end;
    }

    // Simulate each pass
    let mut total_src_reads = 0usize;
    let mut total_src_bytes_read = 0usize;
    let mut total_dst_writes = 0usize;
    let mut total_dst_bytes_written = 0usize;
    let mut source_rows_read_total = 0usize;
    let mut max_pass_buffer = 0usize;

    for pass_chunks in &passes {
        let pass_rows: usize = pass_chunks.iter().map(|&ci| {
            let start = ci * chunk_size_row;
            let end = ((ci + 1) * chunk_size_row).min(n_rows);
            end - start
        }).sum();

        let pass_buffer_bytes = pass_rows * row_bytes;
        max_pass_buffer = max_pass_buffer.max(pass_buffer_bytes);

        // Collect (src_row, out_row) sorted by src_row
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for &ci in pass_chunks {
            for &(out_row, src_row) in &chunk_map[&ci] {
                pairs.push((src_row, out_row));
            }
        }
        pairs.sort_unstable_by_key(|&(src, _)| src);

        let merged_runs = merge_contiguous(&pairs, merge_gap);
        let n_reads = merged_runs.len();
        let bytes_read: usize = merged_runs.iter().map(|r| r.count * row_bytes).sum();
        let rows_in_runs: usize = merged_runs.iter().map(|r| r.count).sum();

        total_src_reads += n_reads;
        total_src_bytes_read += bytes_read;
        source_rows_read_total += rows_in_runs;

        let n_writes = pass_chunks.len();
        let bytes_written = pass_rows * row_bytes;
        total_dst_writes += n_writes;
        total_dst_bytes_written += bytes_written;
    }

    let n_passes = passes.len();

    println!();
    println!("=== {} ===", label);
    println!("  Data: {:.1} GB  |  Row: {:.0} KB  |  Chunk: {} rows = {:.0} MB  |  {} chunks",
        total_data as f64 / 1e9, row_bytes as f64 / 1024.0,
        chunk_size_row, chunk_bytes as f64 / 1e6, n_chunks);
    println!("  Memory: {:.0} GB  |  Max rows buffered: {}  |  Passes: {}",
        memory_limit as f64 / 1e9, max_rows_in_memory, n_passes);
    println!();
    println!("  READS:   {} ops, {:.2} GB total ({:.1}x read amp vs ideal {:.2} GB)",
        total_src_reads, total_src_bytes_read as f64 / 1e9,
        source_rows_read_total as f64 / n_rows as f64,
        total_data as f64 / 1e9);
    println!("  WRITES:  {} ops, {:.2} GB total (== ideal, 0 dest reads, no RMW)",
        total_dst_writes, total_dst_bytes_written as f64 / 1e9);
    println!("  PEAK MEM: {:.0} MB", max_pass_buffer as f64 / 1e6);
    println!("  Source row re-reads: each row read {:.1}x on avg ({} passes over source)",
        source_rows_read_total as f64 / n_rows as f64, n_passes);

    // Comparison with old approach
    let old_rmw_io = 2.0 * n_rows as f64 * chunk_bytes as f64;
    println!("  OLD approach dest I/O would be: {:.1} TB (read-modify-write per row)",
        old_rmw_io / 1e12);
}

struct MergedRun {
    _src_start: usize,
    count: usize,
    _pairs: Vec<(usize, usize)>,
}

fn merge_contiguous(sorted_pairs: &[(usize, usize)], gap: usize) -> Vec<MergedRun> {
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
            runs.push(MergedRun {
                _src_start: cur_start,
                count: cur_end - cur_start,
                _pairs: std::mem::take(&mut cur_pairs),
            });
            cur_start = src;
            cur_end = src + 1;
            cur_pairs = vec![(src, out)];
        }
    }
    runs.push(MergedRun {
        _src_start: cur_start,
        count: cur_end - cur_start,
        _pairs: cur_pairs,
    });
    runs
}
