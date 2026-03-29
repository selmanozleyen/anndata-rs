/// Simulates the sparse scatter I/O pattern using real Tahoe indptr data.
///
/// Faithfully mirrors the Rust engine's batching, merging, chunk-splitting,
/// and write-grouping logic.  No actual Zarr I/O -- just a trace.
///
/// Run with:
///   cargo run -p anndata-ooc --release --example sparse_io_simulation -- --help
///   cargo run -p anndata-ooc --release --example sparse_io_simulation -- \
///       --memory-gb 4 8 16 20 32 64 --op shuffle -v

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args = parse_args();

    let data_dir = find_data_dir();
    eprintln!("Loading indptr from {:?} ...", data_dir.join("indptr"));
    let t0 = Instant::now();
    let indptr = load_indptr_zarr(&data_dir.join("indptr"), args.n_rows);
    let n_rows = indptr.len() - 1;
    let total_nnz = indptr[n_rows] as u64;
    let row_nnz: Vec<u64> = (0..n_rows)
        .map(|i| (indptr[i + 1] - indptr[i]) as u64)
        .collect();
    let min_nnz = *row_nnz.iter().min().unwrap_or(&0);
    let max_nnz = *row_nnz.iter().max().unwrap_or(&0);

    eprintln!(
        "  Loaded: {} rows, {} NNZ  ({:.1}s)",
        fmt_num(n_rows),
        fmt_num(total_nnz as usize),
        t0.elapsed().as_secs_f64()
    );
    eprintln!("  Avg NNZ/row: {:.1}", total_nnz as f64 / n_rows as f64);
    eprintln!("  NNZ range: [{}, {}]", min_nnz, max_nnz);
    eprintln!(
        "  Data size: {:.1} GiB (4B data + 8B indices)",
        total_nnz as f64 * 12.0 / GIB
    );

    let mut summary: Vec<SummaryRow> = Vec::new();

    for &mem_gb in &args.memory_gb {
        let memory_limit = (mem_gb * GIB) as usize;
        println!();
        println!("{}", "=".repeat(80));
        println!("MEMORY LIMIT: {} GiB", mem_gb);
        println!("{}", "=".repeat(80));

        if args.op == "shuffle" || args.op == "all" {
            let t0 = Instant::now();
            let perm = fisher_yates(n_rows, 42);
            let assigns: Vec<Assignment> = (0..n_rows)
                .map(|out_row| Assignment {
                    source_row: perm[out_row],
                    store_id: 0,
                    output_row: out_row,
                })
                .collect();

            let label = format!(
                "SHUFFLE ({} rows, {}G)",
                fmt_num(n_rows),
                mem_gb
            );
            let result = simulate_sparse_scatter(
                &indptr, &assigns, memory_limit,
                args.src_chunk_size, args.dst_chunk_size,
                4, 8, &label, args.verbose,
            );
            eprintln!("  shuffle sim: {:.1}s", t0.elapsed().as_secs_f64());
            summary.push(SummaryRow { op: "shuffle", mem_gb, ..result });
        }

        if args.op == "truncate" || args.op == "all" {
            let trunc_n = n_rows.min(10_000_000);
            let assigns: Vec<Assignment> = (0..trunc_n)
                .map(|i| Assignment {
                    source_row: i,
                    store_id: 0,
                    output_row: i,
                })
                .collect();

            let label = format!(
                "TRUNCATE (first {} rows, {}G)",
                fmt_num(trunc_n),
                mem_gb
            );
            let result = simulate_sparse_scatter(
                &indptr, &assigns, memory_limit,
                args.src_chunk_size, args.dst_chunk_size,
                4, 8, &label, args.verbose,
            );
            summary.push(SummaryRow { op: "truncate", mem_gb, ..result });
        }

        if args.op == "split" || args.op == "all" {
            let codes_path = data_dir.join("obs").join(&args.split_column).join("codes");
            match load_obs_codes_zarr(&codes_path, n_rows) {
                Ok(codes) => {
                    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
                    for (i, &c) in codes.iter().enumerate() {
                        groups.entry(c).or_default().push(i);
                    }
                    let n_groups = groups.len();
                    let group_sizes: Vec<usize> =
                        groups.values().map(|v| v.len()).collect();
                    let min_g = *group_sizes.iter().min().unwrap_or(&0);
                    let max_g = *group_sizes.iter().max().unwrap_or(&0);
                    let mut sorted_sizes = group_sizes.clone();
                    sorted_sizes.sort_unstable();
                    let median_g = sorted_sizes[sorted_sizes.len() / 2];

                    println!(
                        "\n  Split column '{}': {} groups",
                        args.split_column, n_groups
                    );
                    println!(
                        "  Group sizes: min={} max={} median={}",
                        fmt_num(min_g),
                        fmt_num(max_g),
                        fmt_num(median_g)
                    );

                    let mut total_read = 0.0f64;
                    let mut total_write = 0.0f64;
                    let mut total_batches = 0usize;
                    let mut total_sub = 0usize;
                    let mut total_dst = 0usize;

                    for (gid, (_, src_rows)) in groups.iter().enumerate() {
                        let assigns: Vec<Assignment> = src_rows
                            .iter()
                            .enumerate()
                            .map(|(out, &src)| Assignment {
                                source_row: src,
                                store_id: 0,
                                output_row: out,
                            })
                            .collect();

                        let show = gid < 3 || gid == n_groups - 1;
                        let label = format!(
                            "SPLIT group {}/{} ({} rows, {}G)",
                            gid, n_groups,
                            fmt_num(src_rows.len()),
                            mem_gb
                        );
                        let r = simulate_sparse_scatter(
                            &indptr, &assigns, memory_limit,
                            args.src_chunk_size, args.dst_chunk_size,
                            4, 8, &label, args.verbose && show,
                        );
                        total_read += r.read_gib;
                        total_write += r.write_gib;
                        total_batches += r.batches;
                        total_sub += r.sub_runs;
                        total_dst += r.dst_chunks;

                        if gid == 2 && n_groups > 4 {
                            println!(
                                "  ... ({} more groups omitted, showing last) ...",
                                n_groups - 4
                            );
                        }
                    }

                    println!("\n  SPLIT TOTAL ({} groups):", n_groups);
                    println!("    READ:  {:.2} GiB", total_read);
                    println!("    WRITE: {:.2} GiB", total_write);
                    println!("    TOTAL: {:.2} GiB", total_read + total_write);
                    println!("    Batches: {}", total_batches);

                    summary.push(SummaryRow {
                        op: "split",
                        mem_gb,
                        n_rows,
                        batches: total_batches,
                        merged_runs: 0,
                        sub_runs: total_sub,
                        dst_chunks: total_dst,
                        read_gib: total_read,
                        write_gib: total_write,
                        read_amp: if total_write > 0.0 {
                            total_read / total_write
                        } else {
                            0.0
                        },
                    });
                }
                Err(e) => {
                    eprintln!("  Split simulation failed: {}", e);
                }
            }
        }
    }

    if summary.len() > 1 {
        println!();
        println!("{}", "=".repeat(100));
        println!("SUMMARY TABLE");
        println!("{}", "=".repeat(100));
        println!(
            "{:<10} {:>6} {:>14} {:>8} {:>10} {:>10} {:>9} {:>9} {:>9} {:>8}",
            "Op", "MemGB", "Rows", "Batches", "SubRuns", "DstChks",
            "ReadGiB", "WriteGiB", "TotalGiB", "ReadAmp"
        );
        for r in &summary {
            println!(
                "{:<10} {:>6.0} {:>14} {:>8} {:>10} {:>10} {:>9.2} {:>9.2} {:>9.2} {:>8.2}",
                r.op, r.mem_gb,
                fmt_num(r.n_rows),
                r.batches,
                fmt_num(r.sub_runs),
                fmt_num(r.dst_chunks),
                r.read_gib, r.write_gib,
                r.read_gib + r.write_gib,
                r.read_amp,
            );
        }
    }
}

// ---------------------------------------------------------------------------

const GIB: f64 = (1u64 << 30) as f64;

#[derive(Clone, Copy)]
struct Assignment {
    source_row: usize,
    store_id: u16,
    output_row: usize,
}

struct SimResult {
    n_rows: usize,
    batches: usize,
    merged_runs: usize,
    sub_runs: usize,
    dst_chunks: usize,
    read_gib: f64,
    write_gib: f64,
    read_amp: f64,
}

struct SummaryRow {
    op: &'static str,
    mem_gb: f64,
    n_rows: usize,
    batches: usize,
    merged_runs: usize,
    sub_runs: usize,
    dst_chunks: usize,
    read_gib: f64,
    write_gib: f64,
    read_amp: f64,
}

fn simulate_sparse_scatter(
    indptr: &[i64],
    assignments: &[Assignment],
    memory_limit: usize,
    src_chunk_size: usize,
    dst_chunk_size: usize,
    data_elem_size: usize,
    indices_elem_size: usize,
    label: &str,
    verbose: bool,
) -> SimResult {
    let n_assigns = assignments.len();
    let bytes_per_nnz = data_elem_size + indices_elem_size;
    let headroom = 8 * 1024 * 1024usize;
    let available = memory_limit.saturating_sub(headroom);
    let max_nnz_per_batch = if bytes_per_nnz > 0 {
        (available / bytes_per_nnz).max(4096)
    } else {
        usize::MAX
    };

    // Sort assignments by source indptr position
    let mut sorted: Vec<&Assignment> = assignments.iter().collect();
    sorted.sort_unstable_by_key(|a| indptr[a.source_row] as usize);

    // Build output indptr per store
    let n_stores = assignments.iter().map(|a| a.store_id as usize + 1).max().unwrap_or(1);
    let mut store_max_out: Vec<usize> = vec![0; n_stores];
    for a in assignments {
        store_max_out[a.store_id as usize] =
            store_max_out[a.store_id as usize].max(a.output_row + 1);
    }
    let mut out_indptrs: Vec<Vec<i64>> = (0..n_stores)
        .map(|s| vec![0i64; store_max_out[s] + 1])
        .collect();
    for a in assignments {
        let sid = a.store_id as usize;
        let row_nnz = indptr[a.source_row + 1] - indptr[a.source_row];
        out_indptrs[sid][a.output_row + 1] = row_nnz;
    }
    for s in 0..n_stores {
        for i in 1..out_indptrs[s].len() {
            out_indptrs[s][i] += out_indptrs[s][i - 1];
        }
    }

    let total_output_nnz: i64 = out_indptrs.iter().map(|ip| *ip.last().unwrap_or(&0)).sum();

    // Batch by cumulative NNZ
    let mut batch_ranges: Vec<(usize, usize)> = Vec::new();
    let mut batch_start = 0usize;
    let mut batch_nnz = 0usize;
    for (idx, &a) in sorted.iter().enumerate() {
        let row_nnz = (indptr[a.source_row + 1] - indptr[a.source_row]) as usize;
        if batch_nnz + row_nnz > max_nnz_per_batch && batch_nnz > 0 {
            batch_ranges.push((batch_start, idx));
            batch_start = idx;
            batch_nnz = 0;
        }
        batch_nnz += row_nnz;
    }
    batch_ranges.push((batch_start, sorted.len()));

    let n_batches = batch_ranges.len();

    // Simulate each batch
    let mut total_read_nnz = 0u64;
    let mut total_write_nnz = 0u64;
    let mut total_merged_runs = 0usize;
    let mut total_sub_runs = 0usize;
    let mut total_dst_chunks = 0usize;
    let mut max_batch_decoded = 0u64;

    struct BatchDetail {
        rows: usize,
        merged: usize,
        sub_runs: usize,
        dst_chunks: usize,
        read_nnz: u64,
        write_nnz: u64,
    }
    let mut batch_details: Vec<BatchDetail> = Vec::with_capacity(n_batches);

    for &(bs, be) in &batch_ranges {
        let batch_assigns = &sorted[bs..be];

        // Merge reads (exact Rust engine logic)
        let merged = merge_sparse_reads(batch_assigns, indptr, 8192);
        let n_merged = merged.len();
        total_merged_runs += n_merged;

        // Split at chunk boundaries
        let sub_runs = split_merged_runs_by_chunk(&merged, indptr, src_chunk_size);
        let n_sub = sub_runs.len();
        total_sub_runs += n_sub;

        let batch_read_nnz: u64 = sub_runs.iter().map(|r| (r.nnz_end - r.nnz_start) as u64).sum();
        total_read_nnz += batch_read_nnz;
        max_batch_decoded = max_batch_decoded.max(batch_read_nnz);

        let batch_write_nnz: u64 = batch_assigns
            .iter()
            .map(|a| (indptr[a.source_row + 1] - indptr[a.source_row]) as u64)
            .sum();
        total_write_nnz += batch_write_nnz;

        // Destination chunk groups (exact engine logic)
        let n_dst = count_dst_chunk_groups(
            batch_assigns, &out_indptrs, dst_chunk_size,
        );
        total_dst_chunks += n_dst;

        batch_details.push(BatchDetail {
            rows: be - bs,
            merged: n_merged,
            sub_runs: n_sub,
            dst_chunks: n_dst,
            read_nnz: batch_read_nnz,
            write_nnz: batch_write_nnz,
        });
    }

    let total_nnz_src = *indptr.last().unwrap_or(&0) as u64;
    let n_src_chunks = if src_chunk_size > 0 {
        (total_nnz_src as usize + src_chunk_size - 1) / src_chunk_size
    } else {
        1
    };
    let n_dst_chunks_total = if dst_chunk_size > 0 {
        (total_output_nnz as usize + dst_chunk_size - 1) / dst_chunk_size
    } else {
        1
    };

    let total_read_bytes = total_read_nnz as f64 * bytes_per_nnz as f64;
    let total_write_bytes = total_write_nnz as f64 * bytes_per_nnz as f64;
    let read_amp = if total_output_nnz > 0 {
        total_read_nnz as f64 / total_output_nnz as f64
    } else {
        0.0
    };

    println!();
    println!("=== {} ===", label);
    println!(
        "  Rows: {}  |  Total NNZ: {}  ({:.1} GiB)",
        fmt_num(n_assigns),
        fmt_num(total_output_nnz as usize),
        total_output_nnz as f64 * bytes_per_nnz as f64 / GIB
    );
    println!(
        "  Source chunks: {} x {} NNZ  |  Dest chunks: {} x {} NNZ",
        fmt_num(n_src_chunks),
        fmt_num(src_chunk_size),
        fmt_num(n_dst_chunks_total),
        fmt_num(dst_chunk_size)
    );
    println!(
        "  Memory limit:  {:.1} GiB  |  max_nnz/batch: {}",
        memory_limit as f64 / GIB,
        fmt_num(max_nnz_per_batch)
    );
    println!();
    println!("  Batches:       {}", n_batches);
    let avg_merged = total_merged_runs as f64 / n_batches.max(1) as f64;
    let avg_sub = total_sub_runs as f64 / n_batches.max(1) as f64;
    let avg_dst = total_dst_chunks as f64 / n_batches.max(1) as f64;
    println!(
        "  Merged runs:   {} total ({:.1}/batch)",
        fmt_num(total_merged_runs), avg_merged
    );
    println!(
        "  Sub-runs:      {} total ({:.1}/batch) -- read parallelism tasks",
        fmt_num(total_sub_runs), avg_sub
    );
    println!(
        "  Dst chunks:    {} total ({:.0}/batch) -- write parallelism tasks",
        fmt_num(total_dst_chunks), avg_dst
    );
    println!();
    println!(
        "  READ:   {:.2} GiB  ({:.2}x amplification)",
        total_read_bytes / GIB, read_amp
    );
    println!("  WRITE:  {:.2} GiB", total_write_bytes / GIB);
    println!(
        "  TOTAL:  {:.2} GiB",
        (total_read_bytes + total_write_bytes) / GIB
    );
    println!(
        "  Peak batch:    {:.2} GiB decoded",
        max_batch_decoded as f64 * bytes_per_nnz as f64 / GIB
    );

    if verbose && n_batches <= 50 {
        println!();
        println!(
            "  {:>5}  {:>12}  {:>7}  {:>8}  {:>7}  {:>8}  {:>9}",
            "Batch", "Rows", "Merged", "SubRuns", "DstChk", "ReadGiB", "WriteGiB"
        );
        for (i, d) in batch_details.iter().enumerate() {
            println!(
                "  {:5}  {:>12}  {:7}  {:8}  {:7}  {:8.3}  {:9.3}",
                i,
                fmt_num(d.rows),
                d.merged,
                d.sub_runs,
                d.dst_chunks,
                d.read_nnz as f64 * bytes_per_nnz as f64 / GIB,
                d.write_nnz as f64 * bytes_per_nnz as f64 / GIB,
            );
        }
    }

    println!();

    SimResult {
        n_rows: n_assigns,
        batches: n_batches,
        merged_runs: total_merged_runs,
        sub_runs: total_sub_runs,
        dst_chunks: total_dst_chunks,
        read_gib: total_read_bytes / GIB,
        write_gib: total_write_bytes / GIB,
        read_amp,
    }
}

// -- Engine logic replicas (no zarr dependency, pure indptr arithmetic) --

struct MergedRun {
    nnz_start: usize,
    nnz_end: usize,
    assignments: Vec<usize>,
}

fn merge_sparse_reads(
    batch_assigns: &[&Assignment],
    indptr: &[i64],
    gap_nnz: usize,
) -> Vec<MergedRun> {
    if batch_assigns.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let first = batch_assigns[0];
    let mut cur_start = indptr[first.source_row] as usize;
    let mut cur_end = indptr[first.source_row + 1] as usize;
    let mut cur_indices = vec![0usize];

    for (idx, &a) in batch_assigns[1..].iter().enumerate() {
        let lo = indptr[a.source_row] as usize;
        let hi = indptr[a.source_row + 1] as usize;
        if lo <= cur_end + gap_nnz {
            cur_end = cur_end.max(hi);
            cur_indices.push(idx + 1);
        } else {
            runs.push(MergedRun {
                nnz_start: cur_start,
                nnz_end: cur_end,
                assignments: std::mem::take(&mut cur_indices),
            });
            cur_start = lo;
            cur_end = hi;
            cur_indices = vec![idx + 1];
        }
    }
    runs.push(MergedRun {
        nnz_start: cur_start,
        nnz_end: cur_end,
        assignments: cur_indices,
    });
    runs
}

struct SubRun {
    nnz_start: usize,
    nnz_end: usize,
}

fn split_merged_runs_by_chunk(
    merged: &[MergedRun],
    _indptr: &[i64],
    src_chunk_size: usize,
) -> Vec<SubRun> {
    let mut out = Vec::new();
    for run in merged {
        if run.nnz_end <= run.nnz_start || src_chunk_size == usize::MAX {
            out.push(SubRun {
                nnz_start: run.nnz_start,
                nnz_end: run.nnz_end,
            });
            continue;
        }
        let first_chunk = run.nnz_start / src_chunk_size;
        let last_chunk = (run.nnz_end - 1) / src_chunk_size;
        if first_chunk == last_chunk {
            out.push(SubRun {
                nnz_start: run.nnz_start,
                nnz_end: run.nnz_end,
            });
        } else {
            for c in first_chunk..=last_chunk {
                let s = run.nnz_start.max(c * src_chunk_size);
                let e = run.nnz_end.min((c + 1) * src_chunk_size);
                if e > s {
                    out.push(SubRun { nnz_start: s, nnz_end: e });
                }
            }
        }
    }
    out
}

fn count_dst_chunk_groups(
    batch_assigns: &[&Assignment],
    out_indptrs: &[Vec<i64>],
    dst_chunk_size: usize,
) -> usize {
    if dst_chunk_size == 0 {
        return 1;
    }

    // Group by store, find contiguous output runs, then group by dst chunk
    let mut per_store: HashMap<u16, Vec<&Assignment>> = HashMap::new();
    for &a in batch_assigns {
        per_store.entry(a.store_id).or_default().push(a);
    }

    let mut total = 0usize;
    for (&sid, store_assigns) in &per_store {
        let ip = &out_indptrs[sid as usize];
        let mut sorted: Vec<&&Assignment> = store_assigns.iter().collect();
        sorted.sort_unstable_by_key(|a| a.output_row);

        // Find contiguous output runs
        let runs = find_contiguous_output_runs(&sorted, ip);

        // Group by destination chunk
        let mut touched: HashSet<u64> = HashSet::new();
        for run in &runs {
            let nnz_start = ip[run.0] as u64;
            let nnz_end = ip[run.1] as u64;
            if nnz_end <= nnz_start {
                continue;
            }
            let fc = nnz_start / dst_chunk_size as u64;
            let lc = (nnz_end - 1) / dst_chunk_size as u64;
            for c in fc..=lc {
                touched.insert(c);
            }
        }
        total += touched.len();
    }
    total
}

fn find_contiguous_output_runs(
    sorted: &[&&Assignment],
    _out_indptr: &[i64],
) -> Vec<(usize, usize)> {
    if sorted.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut cur_start = sorted[0].output_row;
    let mut cur_end = cur_start + 1;
    for &a in &sorted[1..] {
        if a.output_row == cur_end {
            cur_end += 1;
        } else {
            runs.push((cur_start, cur_end));
            cur_start = a.output_row;
            cur_end = cur_start + 1;
        }
    }
    runs.push((cur_start, cur_end));
    runs
}

// -- I/O: load indptr from zarr chunks on disk --

fn load_indptr_zarr(zarr_path: &std::path::Path, max_rows: Option<usize>) -> Vec<i64> {
    let meta_path = zarr_path.join("zarr.json");
    let meta_str = std::fs::read_to_string(&meta_path)
        .unwrap_or_else(|e| panic!("Cannot read {:?}: {}", meta_path, e));
    let meta: serde_json::Value = serde_json::from_str(&meta_str).expect("Invalid zarr.json");

    let shape = meta["shape"][0].as_u64().expect("missing shape") as usize;
    let chunk_size = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
        .as_u64()
        .expect("missing chunk_shape") as usize;

    let n_chunks = (shape + chunk_size - 1) / chunk_size;
    let limit = max_rows.map_or(shape, |n| (n + 1).min(shape));

    let mut data = Vec::with_capacity(limit);
    let chunks_dir = zarr_path.join("c");

    for ci in 0..n_chunks {
        if data.len() >= limit {
            break;
        }
        let chunk_path = chunks_dir.join(ci.to_string());
        let compressed = std::fs::read(&chunk_path)
            .unwrap_or_else(|e| panic!("Cannot read chunk {:?}: {}", chunk_path, e));

        let decompressed = blosc_decompress(&compressed);
        let n_elems = decompressed.len() / 8;
        for j in 0..n_elems {
            if data.len() >= limit {
                break;
            }
            let val = i64::from_le_bytes(
                decompressed[j * 8..(j + 1) * 8].try_into().unwrap(),
            );
            data.push(val);
        }
    }
    data.truncate(limit);
    data
}

fn load_obs_codes_zarr(
    zarr_path: &std::path::Path,
    max_rows: usize,
) -> Result<Vec<i64>, String> {
    let meta_path = zarr_path.join("zarr.json");
    let meta_str = std::fs::read_to_string(&meta_path)
        .map_err(|e| format!("Cannot read {:?}: {}", meta_path, e))?;
    let meta: serde_json::Value =
        serde_json::from_str(&meta_str).map_err(|e| format!("Invalid zarr.json: {}", e))?;

    let shape = meta["shape"][0].as_u64().ok_or("missing shape")? as usize;
    let chunk_size = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
        .as_u64()
        .ok_or("missing chunk_shape")? as usize;

    let dtype = meta["data_type"].as_str().unwrap_or("int8");
    let elem_size: usize = match dtype {
        "int8" | "uint8" => 1,
        "int16" | "uint16" => 2,
        "int32" | "uint32" => 4,
        "int64" | "uint64" => 8,
        _ => return Err(format!("Unsupported dtype: {}", dtype)),
    };

    let n_chunks = (shape + chunk_size - 1) / chunk_size;
    let limit = max_rows.min(shape);
    let mut data = Vec::with_capacity(limit);
    let chunks_dir = zarr_path.join("c");

    for ci in 0..n_chunks {
        if data.len() >= limit {
            break;
        }
        let chunk_path = chunks_dir.join(ci.to_string());
        let compressed = std::fs::read(&chunk_path)
            .map_err(|e| format!("Cannot read chunk {:?}: {}", chunk_path, e))?;

        let decompressed = blosc_decompress(&compressed);
        let n_elems = decompressed.len() / elem_size;
        for j in 0..n_elems {
            if data.len() >= limit {
                break;
            }
            let val = match elem_size {
                1 => decompressed[j] as i8 as i64,
                2 => i16::from_le_bytes(
                    decompressed[j * 2..(j + 1) * 2].try_into().unwrap(),
                ) as i64,
                4 => i32::from_le_bytes(
                    decompressed[j * 4..(j + 1) * 4].try_into().unwrap(),
                ) as i64,
                8 => i64::from_le_bytes(
                    decompressed[j * 8..(j + 1) * 8].try_into().unwrap(),
                ),
                _ => unreachable!(),
            };
            data.push(val);
        }
    }
    data.truncate(limit);
    Ok(data)
}

fn blosc_decompress(input: &[u8]) -> Vec<u8> {
    if input.len() < 16 {
        return input.to_vec();
    }

    // Blosc header: bytes 4..8 = uncompressed size, 8..12 = compressed size
    let nbytes = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;

    // Try blosc2 frame format first (starts with magic bytes)
    // For blosc1 chunk format: use the C blosc library via zarrs
    // Fallback: use flate2/lz4 directly based on the compressor byte
    let version = input[0];
    let _flags = input[2];
    let _compressor = input[1]; // 0=blosclz, 1=lz4, 2=lz4hc, ...

    if version == 2 {
        // Blosc2 - the compressor byte at offset 1 in header
        // We'll use the zarrs blosc codec
    }

    // Use the blosc-src crate or call C blosc directly.
    // Since anndata-ooc already links zarrs with blosc support,
    // we can use zarrs' blosc codec. But for a standalone example,
    // let's use a raw FFI call to the blosc library that's already linked.

    // Actually, let's use zarrs' codec infrastructure since it's already a dependency.
    use zarrs::array::codec::BytesToBytesCodecTraits;

    let blosc_config = serde_json::json!({
        "cname": "lz4",
        "clevel": 3,
        "shuffle": "shuffle",
        "typesize": 8,
        "blocksize": 0
    });
    let codec = zarrs::array::codec::BloscCodec::new_with_configuration(
        &serde_json::from_value(blosc_config).unwrap(),
    ).expect("Failed to create blosc codec");

    let decoded = codec
        .decode(
            bytes::Bytes::from(input.to_vec()),
            &zarrs::array::codec::CodecOptions::default(),
        )
        .expect("blosc decompress failed");

    decoded.to_vec()
}

// -- Utilities --

fn fisher_yates(n: usize, seed: u64) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (state >> 33) as usize % (i + 1);
        perm.swap(i, j);
    }
    perm
}

fn fmt_num(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn find_data_dir() -> PathBuf {
    let mut dir = std::env::current_dir().unwrap();
    loop {
        let candidate = dir.join("data");
        if candidate.join("indptr").exists() {
            return candidate;
        }
        if !dir.pop() {
            break;
        }
    }
    // Try relative to the binary
    let exe = std::env::current_exe().unwrap();
    let repo_root = exe
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .unwrap_or(std::path::Path::new("."));
    let candidate = repo_root.join("data");
    if candidate.join("indptr").exists() {
        return candidate;
    }
    panic!("Cannot find data/indptr directory. Run from the repo root.");
}

struct Args {
    memory_gb: Vec<f64>,
    n_rows: Option<usize>,
    src_chunk_size: usize,
    dst_chunk_size: usize,
    op: String,
    split_column: String,
    verbose: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut memory_gb: Vec<f64> = Vec::new();
    let mut n_rows: Option<usize> = None;
    let mut src_chunk_size: usize = 67_108_864;
    let mut dst_chunk_size: usize = 67_108_864;
    let mut op = String::from("all");
    let mut split_column = String::from("sample");
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--memory-gb" => {
                i += 1;
                while i < args.len() && !args[i].starts_with('-') {
                    memory_gb.push(args[i].parse().expect("invalid memory-gb"));
                    i += 1;
                }
            }
            "--n-rows" => {
                i += 1;
                n_rows = Some(args[i].parse().expect("invalid n-rows"));
                i += 1;
            }
            "--src-chunk-size" => {
                i += 1;
                src_chunk_size = args[i].parse().expect("invalid src-chunk-size");
                i += 1;
            }
            "--dst-chunk-size" => {
                i += 1;
                dst_chunk_size = args[i].parse().expect("invalid dst-chunk-size");
                i += 1;
            }
            "--op" => {
                i += 1;
                op = args[i].clone();
                i += 1;
            }
            "--split-column" => {
                i += 1;
                split_column = args[i].clone();
                i += 1;
            }
            "-v" | "--verbose" => {
                verbose = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("Usage: sparse_io_simulation [OPTIONS]");
                eprintln!("  --memory-gb N [N ...]   Memory limits (GiB) [default: 4 8 16 20 32 64]");
                eprintln!("  --n-rows N              Truncate indptr to N rows");
                eprintln!("  --src-chunk-size N      Source NNZ chunk size [default: 67108864]");
                eprintln!("  --dst-chunk-size N      Dest NNZ chunk size [default: 67108864]");
                eprintln!("  --op OP                 shuffle|truncate|split|all [default: all]");
                eprintln!("  --split-column COL      obs column for split [default: sample]");
                eprintln!("  -v, --verbose           Print per-batch details");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    if memory_gb.is_empty() {
        memory_gb = vec![4.0, 8.0, 16.0, 20.0, 32.0, 64.0];
    }

    Args {
        memory_gb,
        n_rows,
        src_chunk_size,
        dst_chunk_size,
        op,
        split_column,
        verbose,
    }
}
