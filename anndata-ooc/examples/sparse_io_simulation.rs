/// Simulates the chunk-aligned sparse scatter I/O pattern using real indptr data.
///
/// Calls the actual `ScatterPlanner::plan_sparse()` and `ScatterPlanner::from_groups()`
/// from the anndata-ooc core, so the simulation is guaranteed to match the engine.
///
/// Run with:
///   cargo run -p anndata-ooc --release --example sparse_io_simulation -- --help
///   cargo run -p anndata-ooc --release --example sparse_io_simulation -- \
///       --memory-gb 4 8 16 20 32 64 --op shuffle -v
///   cargo run -p anndata-ooc --release --example sparse_io_simulation -- \
///       --memory-gb 16 32 64 --op groupby --split-column cell_line -v

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use anndata_ooc::{RowAssignment, ScatterPlanner, SparseScatterPass};

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

    let bytes_per_nnz = args.data_elem_size + args.indices_elem_size;

    eprintln!(
        "  Loaded: {} rows, {} NNZ  ({:.1}s)",
        fmt_num(n_rows),
        fmt_num(total_nnz as usize),
        t0.elapsed().as_secs_f64()
    );
    eprintln!("  Avg NNZ/row: {:.1}", total_nnz as f64 / n_rows as f64);
    eprintln!("  NNZ range: [{}, {}]", min_nnz, max_nnz);
    let uncompressed_gib = total_nnz as f64 * bytes_per_nnz as f64 / GIB;
    // Typical compression: f32 data ~2x, i32 indices ~3x
    let est_compressed_gib = total_nnz as f64
        * (args.data_elem_size as f64 / 2.0 + args.indices_elem_size as f64 / 3.0) / GIB;
    eprintln!(
        "  Data size: {:.1} GiB uncompressed, ~{:.0} GiB on disk ({}B data ~2x, {}B indices ~3x)",
        uncompressed_gib, est_compressed_gib,
        args.data_elem_size, args.indices_elem_size
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
            let assignments: Vec<RowAssignment> = (0..n_rows)
                .map(|out_row| RowAssignment {
                    source_row: perm[out_row],
                    store_id: 0,
                    output_row: out_row,
                })
                .collect();

            let out_indptr = build_output_indptr(&assignments, &indptr, n_rows);

            let label = format!(
                "SHUFFLE ({} rows, {}G)",
                fmt_num(n_rows), mem_gb
            );
            let result = simulate_sparse_scatter(
                &indptr, &assignments, &[&out_indptr],
                memory_limit, args.src_chunk_size, args.dst_chunk_size,
                bytes_per_nnz, &label, args.verbose,
            );
            eprintln!("  shuffle sim: {:.1}s", t0.elapsed().as_secs_f64());
            summary.push(result.to_summary("shuffle", mem_gb));
        }

        if args.op == "truncate" || args.op == "all" {
            let trunc_n = n_rows.min(10_000_000);
            let assignments: Vec<RowAssignment> = (0..trunc_n)
                .map(|i| RowAssignment {
                    source_row: i,
                    store_id: 0,
                    output_row: i,
                })
                .collect();
            let out_indptr = build_output_indptr(&assignments, &indptr, trunc_n);

            let label = format!(
                "TRUNCATE (first {} rows, {}G)",
                fmt_num(trunc_n), mem_gb
            );
            let result = simulate_sparse_scatter(
                &indptr, &assignments, &[&out_indptr],
                memory_limit, args.src_chunk_size, args.dst_chunk_size,
                bytes_per_nnz, &label, args.verbose,
            );
            summary.push(result.to_summary("truncate", mem_gb));
        }

        if args.op == "split" || args.op == "all" {
            run_split_sim(
                &data_dir, &args, &indptr, n_rows, memory_limit, mem_gb,
                bytes_per_nnz, &mut summary,
            );
        }

        if args.op == "groupby" {
            run_groupby_sim(
                &data_dir, &args, &indptr, n_rows, memory_limit, mem_gb,
                bytes_per_nnz, &mut summary,
            );
        }
    }

    if summary.len() > 1 {
        println!();
        println!("{}", "=".repeat(115));
        println!("SUMMARY TABLE");
        println!("{}", "=".repeat(115));
        println!(
            "{:<12} {:>6} {:>14} {:>7} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9} {:>8}",
            "Op", "MemGB", "Rows", "Passes", "DstChks", "SrcChks", "SubRuns",
            "ReadGiB", "WriteGiB", "TotalGiB", "ReadAmp"
        );
        for r in &summary {
            println!(
                "{:<12} {:>6.0} {:>14} {:>7} {:>10} {:>10} {:>10} {:>9.2} {:>9.2} {:>9.2} {:>8.2}",
                r.op, r.mem_gb,
                fmt_num(r.n_rows),
                r.passes,
                fmt_num(r.dst_chunks_total),
                fmt_num(r.src_chunks_read),
                fmt_num(r.sub_runs),
                r.read_gib, r.write_gib,
                r.read_gib + r.write_gib,
                r.read_amp,
            );
        }
    }
}

// ---------------------------------------------------------------------------

const GIB: f64 = (1u64 << 30) as f64;

struct SimResult {
    n_rows: usize,
    passes: usize,
    dst_chunks_total: usize,
    src_chunks_read: usize,
    sub_runs: usize,
    read_gib: f64,
    write_gib: f64,
    read_amp: f64,
}

impl SimResult {
    fn to_summary(&self, op: &'static str, mem_gb: f64) -> SummaryRow {
        SummaryRow {
            op, mem_gb,
            n_rows: self.n_rows,
            passes: self.passes,
            dst_chunks_total: self.dst_chunks_total,
            src_chunks_read: self.src_chunks_read,
            sub_runs: self.sub_runs,
            read_gib: self.read_gib,
            write_gib: self.write_gib,
            read_amp: self.read_amp,
        }
    }
}

struct SummaryRow {
    op: &'static str,
    mem_gb: f64,
    n_rows: usize,
    passes: usize,
    dst_chunks_total: usize,
    src_chunks_read: usize,
    sub_runs: usize,
    read_gib: f64,
    write_gib: f64,
    read_amp: f64,
}

/// Build the output indptr for a single store from assignments + source indptr.
fn build_output_indptr(
    assignments: &[RowAssignment],
    src_indptr: &[i64],
    n_out_rows: usize,
) -> Vec<i64> {
    let mut out = vec![0i64; n_out_rows + 1];
    for a in assignments {
        let row_nnz = src_indptr[a.source_row + 1] - src_indptr[a.source_row];
        out[a.output_row + 1] = row_nnz;
    }
    for i in 1..out.len() {
        out[i] += out[i - 1];
    }
    out
}

/// Build per-store output indptrs from multi-store assignments.
fn build_multi_store_indptrs(
    assignments: &[RowAssignment],
    src_indptr: &[i64],
    store_n_rows: &[usize],
) -> Vec<Vec<i64>> {
    let n_stores = store_n_rows.len();
    let mut indptrs: Vec<Vec<i64>> = (0..n_stores)
        .map(|s| vec![0i64; store_n_rows[s] + 1])
        .collect();
    for a in assignments {
        let sid = a.store_id as usize;
        let row_nnz = src_indptr[a.source_row + 1] - src_indptr[a.source_row];
        indptrs[sid][a.output_row + 1] = row_nnz;
    }
    for s in 0..n_stores {
        for i in 1..indptrs[s].len() {
            indptrs[s][i] += indptrs[s][i - 1];
        }
    }
    indptrs
}

/// Core simulation: calls `ScatterPlanner::plan_sparse()` then traces I/O.
fn simulate_sparse_scatter(
    src_indptr: &[i64],
    assignments: &[RowAssignment],
    store_indptrs: &[&[i64]],
    memory_limit: usize,
    src_chunk_size: usize,
    dst_chunk_size: usize,
    bytes_per_nnz: usize,
    label: &str,
    verbose: bool,
) -> SimResult {
    let n_assigns = assignments.len();
    let headroom = 8 * 1024 * 1024usize;
    let available = memory_limit.saturating_sub(headroom);
    let max_nnz_per_pass = if bytes_per_nnz > 0 {
        (available / bytes_per_nnz / 2).max(4096)
    } else {
        usize::MAX
    };

    let n_stores = store_indptrs.len();
    let store_nnz_chunk_sizes: Vec<usize> = vec![dst_chunk_size; n_stores];

    // ---- Use the real planner ----
    let passes = ScatterPlanner::plan_sparse(
        assignments,
        store_indptrs,
        &store_nnz_chunk_sizes,
        src_indptr,
        src_chunk_size,
        max_nnz_per_pass,
    );

    let total_output_nnz: u64 = store_indptrs.iter()
        .map(|ip| *ip.last().unwrap_or(&0) as u64)
        .sum();

    trace_passes(
        &passes, src_indptr, src_chunk_size, dst_chunk_size,
        bytes_per_nnz, max_nnz_per_pass, memory_limit,
        n_assigns, total_output_nnz,
        label, verbose,
    )
}

/// Trace I/O for a set of passes (shared by all simulation modes).
fn trace_passes(
    passes: &[SparseScatterPass],
    src_indptr: &[i64],
    src_chunk_size: usize,
    dst_chunk_size: usize,
    bytes_per_nnz: usize,
    max_nnz_per_pass: usize,
    memory_limit: usize,
    n_assigns: usize,
    total_output_nnz: u64,
    label: &str,
    verbose: bool,
) -> SimResult {
    let n_passes = passes.len();
    let n_dst_chunks_total: usize = passes.iter().map(|p| p.chunks.len()).sum();

    let mut total_src_chunks_read = 0u64;
    let mut total_write_nnz = 0u64;
    let mut total_sub_runs = 0usize;
    let mut max_pass_nnz = 0usize;

    struct PassDetail {
        dst_chunks: usize,
        src_chunks_read: usize,
        unique_src_rows: usize,
        sub_runs: usize,
        read_nnz: u64,
        write_nnz: u64,
    }
    let mut pass_details: Vec<PassDetail> = Vec::with_capacity(n_passes);

    for pass in passes {
        // Collect unique source rows (same as engine's read_pass_sources)
        let mut source_rows: Vec<usize> = pass.chunks.iter()
            .flat_map(|c| c.entries.iter().map(|e| e.source_row))
            .collect();
        source_rows.sort_unstable();
        source_rows.dedup();

        let merged = merge_source_reads(&source_rows, src_indptr, 8192);
        let sub_runs = split_merged_runs_by_chunk(&merged, src_chunk_size);
        let n_sub = sub_runs.len();
        total_sub_runs += n_sub;

        let mut touched_src: HashSet<usize> = HashSet::new();
        let mut pass_read_nnz = 0u64;
        for r in &sub_runs {
            pass_read_nnz += (r.nnz_end - r.nnz_start) as u64;
            if src_chunk_size > 0 && src_chunk_size < usize::MAX {
                let fc = r.nnz_start / src_chunk_size;
                let lc = if r.nnz_end > 0 { (r.nnz_end - 1) / src_chunk_size } else { fc };
                for c in fc..=lc {
                    touched_src.insert(c);
                }
            }
        }
        let src_chunks_read = touched_src.len();
        total_src_chunks_read += src_chunks_read as u64;

        let pass_write_nnz: u64 = pass.chunks.iter()
            .map(|c| (c.nnz_end - c.nnz_start) as u64)
            .sum();
        total_write_nnz += pass_write_nnz;

        max_pass_nnz = max_pass_nnz.max(pass.total_nnz);

        pass_details.push(PassDetail {
            dst_chunks: pass.chunks.len(),
            src_chunks_read,
            unique_src_rows: source_rows.len(),
            sub_runs: n_sub,
            read_nnz: pass_read_nnz,
            write_nnz: pass_write_nnz,
        });
    }

    let total_nnz_src = *src_indptr.last().unwrap_or(&0) as u64;
    let n_src_chunks = if src_chunk_size > 0 {
        (total_nnz_src as usize + src_chunk_size - 1) / src_chunk_size
    } else {
        1
    };
    let n_dst_chunks_ideal = if dst_chunk_size > 0 {
        (total_output_nnz as usize + dst_chunk_size - 1) / dst_chunk_size
    } else {
        1
    };

    let chunk_bytes = src_chunk_size as f64 * bytes_per_nnz as f64;
    let total_read_bytes = total_src_chunks_read as f64 * chunk_bytes;
    let total_write_bytes = total_write_nnz as f64 * bytes_per_nnz as f64;
    let read_amp = if total_write_bytes > 0.0 {
        total_read_bytes / total_write_bytes
    } else {
        0.0
    };

    println!();
    println!("=== {} ===", label);
    println!(
        "  Rows: {}  |  Total output NNZ: {}  ({:.1} GiB)",
        fmt_num(n_assigns),
        fmt_num(total_output_nnz as usize),
        total_output_nnz as f64 * bytes_per_nnz as f64 / GIB
    );
    println!(
        "  Source chunks: {} x {} NNZ ({:.0} MiB/chunk)  |  Dest chunks: {} x {} NNZ",
        fmt_num(n_src_chunks),
        fmt_num(src_chunk_size),
        chunk_bytes / (1024.0 * 1024.0),
        fmt_num(n_dst_chunks_ideal),
        fmt_num(dst_chunk_size)
    );
    println!(
        "  Memory limit:  {:.1} GiB  |  max_nnz/pass: {}",
        memory_limit as f64 / GIB,
        fmt_num(max_nnz_per_pass)
    );
    println!();
    println!("  Passes:        {}", n_passes);
    println!(
        "  Dst chunks:    {} written (each exactly once)",
        fmt_num(n_dst_chunks_total)
    );
    let avg_sub = total_sub_runs as f64 / n_passes.max(1) as f64;
    println!(
        "  Sub-runs:      {} total ({:.1}/pass) -- read parallelism tasks",
        fmt_num(total_sub_runs), avg_sub
    );
    println!();
    println!(
        "  READ:          {:.2} GiB  ({} unique src chunks, {:.1}x amp vs ideal)",
        total_read_bytes / GIB,
        fmt_num(total_src_chunks_read as usize),
        read_amp,
    );
    println!(
        "  WRITE:         {:.2} GiB  ({} dst chunks, each written once -- NO read-modify-write)",
        total_write_bytes / GIB,
        fmt_num(n_dst_chunks_total)
    );
    println!(
        "  TOTAL I/O:     {:.2} GiB",
        (total_read_bytes + total_write_bytes) / GIB
    );
    println!(
        "  Peak pass:     {:.2} GiB chunk buffers",
        max_pass_nnz as f64 * bytes_per_nnz as f64 / GIB
    );

    if verbose && n_passes <= 50 {
        println!();
        println!(
            "  {:>5}  {:>8}  {:>9}  {:>10}  {:>8}  {:>9}  {:>9}",
            "Pass", "DstChks", "SrcChks", "SrcRows", "SubRuns", "ReadGiB", "WriteGiB"
        );
        for (i, d) in pass_details.iter().enumerate() {
            println!(
                "  {:5}  {:>8}  {:>9}  {:>10}  {:>8}  {:9.3}  {:9.3}",
                i,
                d.dst_chunks,
                d.src_chunks_read,
                fmt_num(d.unique_src_rows),
                d.sub_runs,
                d.read_nnz as f64 * bytes_per_nnz as f64 / GIB,
                d.write_nnz as f64 * bytes_per_nnz as f64 / GIB,
            );
        }
    }

    println!();

    SimResult {
        n_rows: n_assigns,
        passes: n_passes,
        dst_chunks_total: n_dst_chunks_total,
        src_chunks_read: total_src_chunks_read as usize,
        sub_runs: total_sub_runs,
        read_gib: total_read_bytes / GIB,
        write_gib: total_write_bytes / GIB,
        read_amp,
    }
}

// -- Split mode: each group is an independent single-store scatter --

fn run_split_sim(
    data_dir: &PathBuf,
    args: &Args,
    indptr: &[i64],
    n_rows: usize,
    memory_limit: usize,
    mem_gb: f64,
    bytes_per_nnz: usize,
    summary: &mut Vec<SummaryRow>,
) {
    let codes_path = data_dir.join("obs").join(&args.split_column).join("codes");
    let codes = match load_obs_codes_zarr(&codes_path, n_rows) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Split simulation failed: {}", e);
            return;
        }
    };

    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, &c) in codes.iter().enumerate() {
        groups.entry(c).or_default().push(i);
    }
    let n_groups = groups.len();
    print_group_stats(&args.split_column, &groups);

    let mut total_read = 0.0f64;
    let mut total_write = 0.0f64;
    let mut total_passes = 0usize;
    let mut total_sub = 0usize;
    let mut total_dst = 0usize;
    let mut total_src = 0usize;
    let mut total_rows = 0usize;

    for (gid, (_, src_rows)) in groups.iter().enumerate() {
        let n_out = src_rows.len();
        let assigns: Vec<RowAssignment> = src_rows
            .iter()
            .enumerate()
            .map(|(out, &src)| RowAssignment {
                source_row: src,
                store_id: 0,
                output_row: out,
            })
            .collect();
        let out_indptr = build_output_indptr(&assigns, indptr, n_out);

        let show = gid < 3 || gid == n_groups - 1;
        let label = format!(
            "SPLIT group {}/{} ({} rows, {}G)",
            gid, n_groups, fmt_num(n_out), mem_gb
        );
        let r = simulate_sparse_scatter(
            indptr, &assigns, &[&out_indptr],
            memory_limit, args.src_chunk_size, args.dst_chunk_size,
            bytes_per_nnz, &label, args.verbose && show,
        );
        total_read += r.read_gib;
        total_write += r.write_gib;
        total_passes += r.passes;
        total_sub += r.sub_runs;
        total_dst += r.dst_chunks_total;
        total_src += r.src_chunks_read;
        total_rows += r.n_rows;

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
    println!("    Passes: {}", total_passes);

    summary.push(SummaryRow {
        op: "split",
        mem_gb,
        n_rows: total_rows,
        passes: total_passes,
        dst_chunks_total: total_dst,
        src_chunks_read: total_src,
        sub_runs: total_sub,
        read_gib: total_read,
        write_gib: total_write,
        read_amp: if total_write > 0.0 { total_read / total_write } else { 0.0 },
    });
}

// -- Groupby mode: all groups planned together as multi-store scatter --

fn run_groupby_sim(
    data_dir: &PathBuf,
    args: &Args,
    indptr: &[i64],
    n_rows: usize,
    memory_limit: usize,
    mem_gb: f64,
    bytes_per_nnz: usize,
    summary: &mut Vec<SummaryRow>,
) {
    let codes_path = data_dir.join("obs").join(&args.split_column).join("codes");
    let codes = match load_obs_codes_zarr(&codes_path, n_rows) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Groupby simulation failed: {}", e);
            return;
        }
    };

    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, &c) in codes.iter().enumerate() {
        groups.entry(c).or_default().push(i);
    }
    let n_groups = groups.len();
    print_group_stats(&args.split_column, &groups);

    // Map raw group codes -> contiguous store_ids 0..n_groups
    let code_to_sid: BTreeMap<i64, u16> = groups.keys()
        .enumerate()
        .map(|(i, &code)| (code, i as u16))
        .collect();

    let group_ids: Vec<u16> = codes.iter()
        .map(|&c| *code_to_sid.get(&c).unwrap())
        .collect();

    // Use the core from_groups to build assignments + per-store row counts
    let (assignments, store_n_rows) =
        ScatterPlanner::from_groups(&group_ids, n_groups);

    let store_indptrs = build_multi_store_indptrs(&assignments, indptr, &store_n_rows);
    let store_indptr_refs: Vec<&[i64]> = store_indptrs.iter()
        .map(|ip| ip.as_slice())
        .collect();

    let total_output_nnz: u64 = store_indptrs.iter()
        .map(|ip| *ip.last().unwrap_or(&0) as u64)
        .sum();

    let headroom = 8 * 1024 * 1024usize;
    let available = memory_limit.saturating_sub(headroom);
    let max_nnz_per_pass = if bytes_per_nnz > 0 {
        (available / bytes_per_nnz / 2).max(4096)
    } else {
        usize::MAX
    };

    let store_nnz_chunk_sizes: Vec<usize> = vec![args.dst_chunk_size; n_groups];

    // Use the real planner
    let passes = ScatterPlanner::plan_sparse(
        &assignments,
        &store_indptr_refs,
        &store_nnz_chunk_sizes,
        indptr,
        args.src_chunk_size,
        max_nnz_per_pass,
    );

    let label = format!(
        "GROUPBY '{}' ({} groups, {} rows, {}G)",
        args.split_column, n_groups, fmt_num(n_rows), mem_gb
    );

    let result = trace_passes(
        &passes, indptr, args.src_chunk_size, args.dst_chunk_size,
        bytes_per_nnz, max_nnz_per_pass, memory_limit,
        assignments.len(), total_output_nnz,
        &label, args.verbose,
    );

    // Per-store breakdown
    println!("  Per-store breakdown:");
    println!(
        "    {:>5}  {:>12}  {:>14}  {:>10}",
        "Store", "Rows", "NNZ", "DstChunks"
    );
    for (sid, n) in store_n_rows.iter().enumerate() {
        let store_nnz = *store_indptrs[sid].last().unwrap_or(&0) as usize;
        let store_dst_chunks = if args.dst_chunk_size > 0 {
            (store_nnz + args.dst_chunk_size - 1) / args.dst_chunk_size
        } else {
            1
        };
        if sid < 5 || sid == n_groups - 1 {
            println!(
                "    {:>5}  {:>12}  {:>14}  {:>10}",
                sid, fmt_num(*n), fmt_num(store_nnz), fmt_num(store_dst_chunks)
            );
        } else if sid == 5 {
            println!("    {:>5}  ... ({} more stores) ...", "", n_groups - 6);
        }
    }
    println!();

    summary.push(SummaryRow {
        op: "groupby",
        mem_gb,
        n_rows,
        passes: result.passes,
        dst_chunks_total: result.dst_chunks_total,
        src_chunks_read: result.src_chunks_read,
        sub_runs: result.sub_runs,
        read_gib: result.read_gib,
        write_gib: result.write_gib,
        read_amp: result.read_amp,
    });
}

fn print_group_stats(column: &str, groups: &BTreeMap<i64, Vec<usize>>) {
    let n_groups = groups.len();
    let group_sizes: Vec<usize> = groups.values().map(|v| v.len()).collect();
    let min_g = *group_sizes.iter().min().unwrap_or(&0);
    let max_g = *group_sizes.iter().max().unwrap_or(&0);
    let mut sorted_sizes = group_sizes.clone();
    sorted_sizes.sort_unstable();
    let median_g = sorted_sizes[sorted_sizes.len() / 2];

    println!(
        "\n  Column '{}': {} groups",
        column, n_groups
    );
    println!(
        "  Group sizes: min={} max={} median={}",
        fmt_num(min_g), fmt_num(max_g), fmt_num(median_g)
    );
}

// -- Read-side I/O simulation helpers (mirrors engine logic) --

struct MergedRun {
    nnz_start: usize,
    nnz_end: usize,
}

fn merge_source_reads(
    sorted_source_rows: &[usize],
    indptr: &[i64],
    gap_nnz: usize,
) -> Vec<MergedRun> {
    if sorted_source_rows.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let first = sorted_source_rows[0];
    let mut cur_start = indptr[first] as usize;
    let mut cur_end = indptr[first + 1] as usize;

    for &src_row in &sorted_source_rows[1..] {
        let lo = indptr[src_row] as usize;
        let hi = indptr[src_row + 1] as usize;
        if lo <= cur_end + gap_nnz {
            cur_end = cur_end.max(hi);
        } else {
            runs.push(MergedRun {
                nnz_start: cur_start,
                nnz_end: cur_end,
            });
            cur_start = lo;
            cur_end = hi;
        }
    }
    runs.push(MergedRun {
        nnz_start: cur_start,
        nnz_end: cur_end,
    });
    runs
}

struct SubRun {
    nnz_start: usize,
    nnz_end: usize,
}

fn split_merged_runs_by_chunk(
    merged: &[MergedRun],
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

/// Load obs codes using zarrs Array API (handles any codec chain automatically).
fn load_obs_codes_zarr(
    zarr_path: &std::path::Path,
    max_rows: usize,
) -> Result<Vec<i64>, String> {
    use std::sync::Arc;
    use zarrs::array::{Array, ArraySubset};
    use zarrs::filesystem::FilesystemStore;

    // zarr_path is e.g. /path/to/store.zarr/obs/cell_line/codes
    // The FilesystemStore needs the store root; the array path is relative.
    // Walk up to find the store root (directory containing zarr.json at top or
    // the parent of "obs"). We open the store at the array's parent and use
    // "/" as array path.
    let store = Arc::new(
        FilesystemStore::new(zarr_path)
            .map_err(|e| format!("Cannot open store at {:?}: {}", zarr_path, e))?
    );

    let array = Array::open(store, "/")
        .map_err(|e| format!("Cannot open array at {:?}: {}", zarr_path, e))?;

    let shape = array.shape();
    let n_elems = shape[0] as usize;
    let limit = max_rows.min(n_elems);

    let subset = ArraySubset::new_with_ranges(&[0..limit as u64]);
    let bytes: zarrs::array::ArrayBytes<'_> = array.retrieve_array_subset(&subset)
        .map_err(|e| format!("Cannot read array subset: {}", e))?;
    let raw = bytes.into_fixed()
        .map_err(|e| format!("Not fixed-size elements: {}", e))?;

    let elem_size = array.data_type().fixed_size().unwrap_or(1);
    let mut data = Vec::with_capacity(limit);
    for j in 0..limit {
        let val = match elem_size {
            1 => raw[j] as i8 as i64,
            2 => i16::from_le_bytes(
                raw[j * 2..(j + 1) * 2].try_into().unwrap(),
            ) as i64,
            4 => i32::from_le_bytes(
                raw[j * 4..(j + 1) * 4].try_into().unwrap(),
            ) as i64,
            8 => i64::from_le_bytes(
                raw[j * 8..(j + 1) * 8].try_into().unwrap(),
            ),
            _ => return Err(format!("Unsupported element size: {}", elem_size)),
        };
        data.push(val);
    }

    Ok(data)
}

fn blosc_decompress(input: &[u8]) -> Vec<u8> {
    if input.len() < 16 {
        return input.to_vec();
    }

    use std::borrow::Cow;
    use zarrs::array::BytesToBytesCodecTraits;
    use zarrs::array::CodecOptions;
    use zarrs::array::BytesRepresentation;

    // Blosc header bytes 4..8 = uncompressed size
    let nbytes = u32::from_le_bytes(input[4..8].try_into().unwrap()) as u64;

    let blosc_config = serde_json::json!({
        "cname": "lz4",
        "clevel": 3,
        "shuffle": "noshuffle",
        "typesize": 1,
        "blocksize": 0
    });
    let codec = zarrs::array::codec::BloscCodec::new_with_configuration(
        &serde_json::from_value(blosc_config).unwrap(),
    ).expect("Failed to create blosc codec");

    let decoded = codec
        .decode(
            Cow::Borrowed(input),
            &BytesRepresentation::FixedSize(nbytes),
            &CodecOptions::default(),
        )
        .expect("blosc decompress failed");

    decoded.into_owned()
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
    data_elem_size: usize,
    indices_elem_size: usize,
    op: String,
    split_column: String,
    verbose: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut memory_gb: Vec<f64> = Vec::new();
    let mut n_rows: Option<usize> = None;
    let mut src_chunk_size: usize = 349_310;
    let mut dst_chunk_size: usize = 349_310;
    let mut data_elem_size: usize = 4;
    let mut indices_elem_size: usize = 4;
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
            "--data-elem-size" => {
                i += 1;
                data_elem_size = args[i].parse().expect("invalid data-elem-size");
                i += 1;
            }
            "--indices-elem-size" => {
                i += 1;
                indices_elem_size = args[i].parse().expect("invalid indices-elem-size");
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
                eprintln!("  --data-elem-size N      Bytes per data element [default: 4 (f32)]");
                eprintln!("  --indices-elem-size N   Bytes per index element [default: 4 (i32)]");
                eprintln!("  --op OP                 shuffle|truncate|split|groupby|all [default: all]");
                eprintln!("  --split-column COL      obs column for split/groupby [default: sample]");
                eprintln!("  -v, --verbose           Print per-pass details");
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
        data_elem_size,
        indices_elem_size,
        op,
        split_column,
        verbose,
    }
}
