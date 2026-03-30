/// End-to-end groupby: simulate the I/O pattern, then optionally execute
/// the real scatter on an AnnData Zarr store.
///
/// Run with:
///   cargo run -p anndata-ooc --release --example e2e_groupby -- --help
///
/// Simulate groupby on tahoe10m (dry-run by default):
///   cargo run -p anndata-ooc --release --example e2e_groupby -- \
///       --input /lustre/boost_ai/users/selman.ozleyen/data/tahoe10m_fast.zarr \
///       --column cell_line --memory-gb 16 32 64 -v
///
/// Actually run the scatter:
///   cargo run -p anndata-ooc --release --example e2e_groupby -- \
///       --input /lustre/boost_ai/users/selman.ozleyen/data/tahoe10m_fast.zarr \
///       --output-dir /tmp/tahoe10m_groupby \
///       --column cell_line --memory-gb 32 --run

use std::collections::{BTreeMap, HashSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anndata_ooc::{
    RowAssignment, ScatterPlanner, SparsePlannerMode, SparseScatterPass,
    RuntimeBudget, compute_runtime_sparse_budget,
};
use rayon::ThreadPoolBuilder;

fn main() {
    let args = parse_args();
    if let Some(threads) = args.threads {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap_or_else(|e| {
                eprintln!("FATAL: failed to configure Rayon thread pool: {}", e);
                std::process::exit(1);
            });
    }
    let effective_threads = rayon::current_num_threads();
    let mut logger = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    );
    logger.format_timestamp_secs();
    logger.init();

    // ---------------------------------------------------------------
    // 1. Load source metadata
    // ---------------------------------------------------------------
    let src = &args.input;
    eprintln!("Source: {}", src.display());

    let x_dir = src.join("X");
    eprintln!("Loading X/indptr ...");
    let t0 = Instant::now();
    let indptr = load_indptr_zarr(&x_dir.join("indptr"), None);
    let n_rows = indptr.len() - 1;
    let total_nnz = indptr[n_rows] as u64;
    eprintln!(
        "  {} rows, {} NNZ  ({:.1}s)",
        fmt(n_rows), fmt(total_nnz as usize), t0.elapsed().as_secs_f64()
    );

    let avg_nnz = total_nnz as f64 / n_rows as f64;
    let row_nnz: Vec<u64> = (0..n_rows)
        .map(|i| (indptr[i + 1] - indptr[i]) as u64)
        .collect();
    let min_nnz = *row_nnz.iter().min().unwrap_or(&0);
    let max_nnz = *row_nnz.iter().max().unwrap_or(&0);
    eprintln!("  Avg NNZ/row: {:.1}  range: [{}, {}]", avg_nnz, min_nnz, max_nnz);

    let data_meta = read_zarr_json(&x_dir.join("data").join("zarr.json"));
    let indices_meta = read_zarr_json(&x_dir.join("indices").join("zarr.json"));
    let data_elem = elem_size_from_dtype(data_meta["data_type"].as_str().unwrap_or("float32"));
    let idx_elem = elem_size_from_dtype(indices_meta["data_type"].as_str().unwrap_or("int32"));
    let bytes_per_nnz = data_elem + idx_elem;

    let src_chunk_size = data_meta
        .pointer("/chunk_grid/configuration/chunk_shape/0")
        .and_then(|v| v.as_u64())
        .unwrap_or(67_108_864) as usize;

    let uncompressed_gib = total_nnz as f64 * bytes_per_nnz as f64 / GIB;
    eprintln!(
        "  Data size: {:.1} GiB uncompressed ({}B data + {}B indices, chunk={})",
        uncompressed_gib, data_elem, idx_elem, fmt(src_chunk_size)
    );

    // ---------------------------------------------------------------
    // 2. Load obs codes for the groupby column
    // ---------------------------------------------------------------
    let codes_path = src.join("obs").join(&args.column).join("codes");
    eprintln!("\nLoading obs/{}/codes ...", args.column);
    let t0 = Instant::now();
    let codes = load_obs_codes_zarr(&codes_path, n_rows)
        .unwrap_or_else(|e| {
            eprintln!("FATAL: Cannot load codes from {:?}: {}", codes_path, e);
            std::process::exit(1);
        });
    eprintln!("  Loaded {} codes in {:.1}s", fmt(codes.len()), t0.elapsed().as_secs_f64());

    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, &c) in codes.iter().enumerate() {
        groups.entry(c).or_default().push(i);
    }
    let n_groups = groups.len();
    print_group_stats(&args.column, &groups);

    let code_to_sid: BTreeMap<i64, u16> = groups.keys()
        .enumerate()
        .map(|(i, &code)| (code, i as u16))
        .collect();
    let group_ids: Vec<u16> = codes.iter()
        .map(|&c| *code_to_sid.get(&c).unwrap())
        .collect();

    let (assignments, store_n_rows) = ScatterPlanner::from_groups(&group_ids, n_groups);

    let store_indptrs = build_multi_store_indptrs(&assignments, &indptr, &store_n_rows);

    // Category names (for output directory naming)
    let cat_names = load_category_names(
        &src.join("obs").join(&args.column).join("categories")
    );

    // ---------------------------------------------------------------
    // 3. Simulate I/O at each memory budget
    // ---------------------------------------------------------------
    let dst_chunk_size = args.dst_chunk_size.unwrap_or(src_chunk_size);
    let row_split_stats = compute_row_split_stats(&assignments, &store_indptrs, dst_chunk_size);
    let src_boundary_stats = compute_source_boundary_stats(&indptr, src_chunk_size);

    let mut summary: Vec<SummaryRow> = Vec::new();

    for &mem_gb in &args.memory_gb {
        let memory_limit = (mem_gb * GIB) as usize;
        let store_indptr_refs: Vec<&[i64]> = store_indptrs.iter()
            .map(|ip| ip.as_slice())
            .collect();

        for &(planner_mode, planner_label) in &[
            (SparsePlannerMode::Greedy, "greedy"),
            (SparsePlannerMode::GroupbyAware, "groupby-aware"),
        ] {
            println!();
            println!("{}", "=".repeat(90));
            println!(
                "GROUPBY '{}': {} groups, {} rows | MEMORY {:.0} GiB | PLANNER {}",
                args.column, n_groups, fmt(n_rows), mem_gb, planner_label
            );
            println!("{}", "=".repeat(90));

            let result = simulate_groupby(
                &indptr, &assignments, &store_indptr_refs,
                planner_mode,
                effective_threads,
                memory_limit, src_chunk_size, dst_chunk_size,
                bytes_per_nnz, n_rows, &row_split_stats, &src_boundary_stats, args.verbose,
            );

            // Per-store breakdown
            println!("  Per-store breakdown:");
            println!(
                "    {:>5}  {:>12}  {:>14}  {:>10}  {}",
                "Store", "Rows", "NNZ", "DstChunks", "Name"
            );
            for (sid, n) in store_n_rows.iter().enumerate() {
                let store_nnz = *store_indptrs[sid].last().unwrap_or(&0) as usize;
                let store_dst = if dst_chunk_size > 0 {
                    (store_nnz + dst_chunk_size - 1) / dst_chunk_size
                } else {
                    1
                };
                let name = cat_names.as_ref()
                    .and_then(|names| names.get(sid))
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                if sid < 8 || sid >= n_groups - 2 {
                    println!(
                        "    {:>5}  {:>12}  {:>14}  {:>10}  {}",
                        sid, fmt(*n), fmt(store_nnz), fmt(store_dst), name
                    );
                } else if sid == 8 {
                    println!("    {:>5}  ... ({} more groups) ...", "", n_groups - 10);
                }
            }
            println!();

            summary.push(SummaryRow {
                planner: planner_label,
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
    }

    if summary.len() > 1 {
        println!();
        println!("{}", "=".repeat(105));
        println!("SUMMARY TABLE");
        println!("{}", "=".repeat(105));
        println!(
            "{:<14} {:>6} {:>14} {:>7} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9} {:>8}",
            "Planner", "MemGB", "Rows", "Passes", "DstChks", "SrcChks", "SubRuns",
            "ReadGiB", "WriteGiB", "TotalGiB", "ReadAmp"
        );
        for r in &summary {
            println!(
                "{:<14} {:>6.0} {:>14} {:>7} {:>10} {:>10} {:>10} {:>9.2} {:>9.2} {:>9.2} {:>8.2}",
                r.planner, r.mem_gb, fmt(r.n_rows),
                r.passes, fmt(r.dst_chunks_total), fmt(r.src_chunks_read), fmt(r.sub_runs),
                r.read_gib, r.write_gib,
                r.read_gib + r.write_gib,
                r.read_amp,
            );
        }
    }

    // ---------------------------------------------------------------
    // 4. Optionally run the real scatter
    // ---------------------------------------------------------------
    if !args.run {
        eprintln!();
        eprintln!("Dry run complete. Pass --run to execute the scatter.");
        return;
    }

    let mem_gb = *args.memory_gb.last().unwrap();
    let memory_limit = (mem_gb * GIB) as usize;
    let output_dir = args.output_dir.clone().unwrap_or_else(|| {
        src.parent().unwrap_or(Path::new("/tmp"))
            .join(format!("{}_groupby_{}", src.file_stem().unwrap().to_string_lossy(), args.column))
    });

    eprintln!();
    eprintln!("{}", "=".repeat(90));
    eprintln!("EXECUTING SCATTER (memory={:.0} GiB)", mem_gb);
    eprintln!("  Output directory: {}", output_dir.display());
    eprintln!("{}", "=".repeat(90));
    eprintln!("  Execution model:");
    eprintln!("    1. Create output stores and copy metadata groups.");
    eprintln!("    2. For each pass, allocate destination chunk buffers.");
    eprintln!("    3. Read source sub-runs, scatter rows into those buffers.");
    eprintln!("    4. Flush each destination chunk once it is complete.");
    eprintln!("  Rayon threads: {}", effective_threads);
    eprintln!("  Progress only moves when destination chunks flush.");
    eprintln!("  So long stretches at 0.00 GiB mean setup or read-heavy work, not a hang.");
    eprintln!("  Planner mode: {}", planner_mode_label(args.planner_mode));
    eprintln!(
        "  Row-to-chunk fanout: {} / {} nonempty rows cross a destination chunk boundary ({:.4}%), avg {:.4} chunks/nonempty-row, max {}",
        fmt(row_split_stats.rows_crossing_chunk_boundary),
        fmt(row_split_stats.nonempty_rows),
        row_split_stats.rows_crossing_chunk_boundary as f64 * 100.0 / row_split_stats.nonempty_rows.max(1) as f64,
        row_split_stats.avg_chunks_per_nonempty_row,
        row_split_stats.max_chunks_per_row,
    );
    eprintln!(
        "  Source edge rows: {} / {} nonempty rows cross a source chunk boundary ({:.4}%), max {} source chunks/row",
        fmt(src_boundary_stats.rows_crossing_source_chunk_boundary),
        fmt(src_boundary_stats.nonempty_rows),
        src_boundary_stats.rows_crossing_source_chunk_boundary as f64 * 100.0 / src_boundary_stats.nonempty_rows.max(1) as f64,
        src_boundary_stats.max_chunks_per_row,
    );

    std::fs::create_dir_all(&output_dir).expect("Cannot create output directory");

    let outputs: Vec<anndata_ooc::OutputStoreConfig> = (0..n_groups).map(|sid| {
        let name = cat_names.as_ref()
            .and_then(|names| names.get(sid))
            .cloned()
            .unwrap_or_else(|| format!("group_{}", sid));
        let safe_name = name.replace('/', "_").replace(' ', "_");
        anndata_ooc::OutputStoreConfig {
            path: output_dir.join(format!("{}.zarr", safe_name)),
            n_rows: store_n_rows[sid],
        }
    }).collect();

    eprintln!("  {} output stores:", outputs.len());
    for (i, o) in outputs.iter().enumerate() {
        if i < 5 || i >= n_groups - 1 {
            eprintln!("    [{}] {} ({} rows)", i, o.path.display(), fmt(o.n_rows));
        } else if i == 5 {
            eprintln!("    ... ({} more) ...", n_groups - 6);
        }
    }

    let progress = Arc::new(AtomicU64::new(0));
    let progress_clone = progress.clone();

    let total_write_nnz: u64 = store_indptrs.iter()
        .map(|ip| *ip.last().unwrap_or(&0) as u64)
        .sum();
    let est_total_bytes = total_write_nnz * bytes_per_nnz as u64;

    let progress_handle = std::thread::spawn(move || {
        let mut last_written = u64::MAX;
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let written = progress_clone.load(Ordering::Relaxed);
            if written == u64::MAX {
                break;
            }
            if written == last_written {
                continue;
            }
            last_written = written;
            let pct = if est_total_bytes > 0 {
                written as f64 / est_total_bytes as f64 * 100.0
            } else {
                0.0
            };
            eprint!(
                "\r  Progress: {:.2} GiB flushed ({:.1}%)",
                written as f64 / GIB, pct
            );
            let _ = io::stderr().flush();
        }
    });

    let config = anndata_ooc::ScatterConfig {
        memory_limit,
        chunk_size: None,
        shard_size: None,
        target_shard_bytes: None,
        compression_level: None,
        progress: Some(progress.clone()),
        planner_mode: args.planner_mode,
    };

    let scatter_t0 = Instant::now();
    match anndata_ooc::scatter_anndata(src, &outputs, &assignments, &config) {
        Ok(()) => {
            let elapsed = scatter_t0.elapsed().as_secs_f64();
            progress.store(u64::MAX, Ordering::Relaxed);
            let _ = progress_handle.join();

            eprintln!();
            eprintln!("Scatter complete in {:.1}s", elapsed);
            let throughput = uncompressed_gib / elapsed;
            eprintln!("  Throughput: {:.2} GiB/s (uncompressed)", throughput);
            for o in &outputs {
                eprintln!("  -> {}", o.path.display());
            }
        }
        Err(e) => {
            progress.store(u64::MAX, Ordering::Relaxed);
            let _ = progress_handle.join();
            eprintln!("SCATTER FAILED: {:#}", e);
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

const GIB: f64 = (1u64 << 30) as f64;

struct SimResult {
    passes: usize,
    dst_chunks_total: usize,
    src_chunks_read: usize,
    sub_runs: usize,
    read_gib: f64,
    write_gib: f64,
    read_amp: f64,
}

struct PlanningMetrics {
    lower_bound_src_chunk_touches: usize,
    lower_bound_read_gib: f64,
    actual_vs_lower_bound: f64,
    mean_src_chunk_multiplicity: f64,
    max_src_chunk_multiplicity: usize,
    reused_src_chunks: usize,
    reused_src_chunks_pct: f64,
    overlap_gain_chunks: usize,
    overlap_gain_pct: f64,
    avg_chunk_src_footprint: f64,
    max_chunk_src_footprint: usize,
    median_chunk_src_span_rows: usize,
    p95_chunk_src_span_rows: usize,
}

struct RowSplitStats {
    nonempty_rows: usize,
    rows_crossing_chunk_boundary: usize,
    max_chunks_per_row: usize,
    avg_chunks_per_nonempty_row: f64,
}

struct SourceBoundaryStats {
    nonempty_rows: usize,
    rows_crossing_source_chunk_boundary: usize,
    max_chunks_per_row: usize,
}

struct SummaryRow {
    planner: &'static str,
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

fn simulate_groupby(
    src_indptr: &[i64],
    assignments: &[RowAssignment],
    store_indptrs: &[&[i64]],
    planner_mode: SparsePlannerMode,
    n_threads: usize,
    memory_limit: usize,
    src_chunk_size: usize,
    dst_chunk_size: usize,
    bytes_per_nnz: usize,
    n_rows: usize,
    row_split_stats: &RowSplitStats,
    src_boundary_stats: &SourceBoundaryStats,
    verbose: bool,
) -> SimResult {
    let budget = compute_runtime_sparse_budget(
        memory_limit,
        src_chunk_size,
        bytes_per_nnz,
        n_threads,
    );
    let max_nnz_per_pass = budget.max_nnz_per_pass;

    let n_stores = store_indptrs.len();
    let store_nnz_chunk_sizes: Vec<usize> = vec![dst_chunk_size; n_stores];

    let t0 = Instant::now();
    let passes = ScatterPlanner::plan_sparse_with_mode(
        assignments,
        store_indptrs,
        &store_nnz_chunk_sizes,
        src_indptr,
        src_chunk_size,
        max_nnz_per_pass,
        planner_mode,
    );
    eprintln!("  Planning: {:.2}s", t0.elapsed().as_secs_f64());

    let total_output_nnz: u64 = store_indptrs.iter()
        .map(|ip| *ip.last().unwrap_or(&0) as u64)
        .sum();

    trace_passes(
        &passes, src_indptr, src_chunk_size, dst_chunk_size,
        bytes_per_nnz, &budget, memory_limit,
        n_rows, total_output_nnz, row_split_stats, src_boundary_stats, verbose,
    )
}

fn trace_passes(
    passes: &[SparseScatterPass],
    src_indptr: &[i64],
    src_chunk_size: usize,
    dst_chunk_size: usize,
    bytes_per_nnz: usize,
    budget: &RuntimeBudget,
    memory_limit: usize,
    n_rows: usize,
    total_output_nnz: u64,
    row_split_stats: &RowSplitStats,
    src_boundary_stats: &SourceBoundaryStats,
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
        overlap_gain_chunks: usize,
        avg_chunk_src_footprint: f64,
    }
    let mut pass_details: Vec<PassDetail> = Vec::with_capacity(n_passes);

    let total_nnz_src = *src_indptr.last().unwrap_or(&0) as u64;
    let n_src_chunks = if src_chunk_size > 0 {
        (total_nnz_src as usize + src_chunk_size - 1) / src_chunk_size
    } else {
        1
    };
    let mut src_chunk_load_nnz = vec![0u64; n_src_chunks];
    let mut src_chunk_pass_multiplicity = vec![0usize; n_src_chunks];
    let mut total_chunk_footprint_sum = 0usize;
    let mut total_overlap_gain_chunks = 0usize;
    let mut chunk_src_footprints: Vec<usize> = Vec::new();
    let mut chunk_src_span_rows: Vec<usize> = Vec::new();

    for pass in passes {
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
        for &src_chunk in &touched_src {
            if src_chunk < src_chunk_pass_multiplicity.len() {
                src_chunk_pass_multiplicity[src_chunk] += 1;
            }
        }

        let pass_write_nnz: u64 = pass.chunks.iter()
            .map(|c| (c.nnz_end - c.nnz_start) as u64)
            .sum();
        total_write_nnz += pass_write_nnz;
        max_pass_nnz = max_pass_nnz.max(pass.total_nnz);

        let mut pass_chunk_footprint_sum = 0usize;
        for chunk in &pass.chunks {
            let mut chunk_src_chunks: HashSet<usize> = HashSet::new();
            let mut min_src_row = usize::MAX;
            let mut max_src_row = 0usize;
            for entry in &chunk.entries {
                min_src_row = min_src_row.min(entry.source_row);
                max_src_row = max_src_row.max(entry.source_row);
                let lo = src_indptr[entry.source_row] as usize;
                let hi = src_indptr[entry.source_row + 1] as usize;
                if hi <= lo || src_chunk_size == 0 || src_chunk_size == usize::MAX {
                    continue;
                }
                let first_src_chunk = lo / src_chunk_size;
                let last_src_chunk = (hi - 1) / src_chunk_size;
                for src_chunk in first_src_chunk..=last_src_chunk {
                    chunk_src_chunks.insert(src_chunk);
                }
            }

            let footprint_size = chunk_src_chunks.len();
            pass_chunk_footprint_sum += footprint_size;
            total_chunk_footprint_sum += footprint_size;
            chunk_src_footprints.push(footprint_size);
            if min_src_row != usize::MAX {
                chunk_src_span_rows.push(max_src_row - min_src_row + 1);
            }

            let chunk_nnz = (chunk.nnz_end - chunk.nnz_start) as u64;
            for src_chunk in chunk_src_chunks {
                if src_chunk < src_chunk_load_nnz.len() {
                    src_chunk_load_nnz[src_chunk] += chunk_nnz;
                }
            }
        }
        let pass_overlap_gain = pass_chunk_footprint_sum.saturating_sub(src_chunks_read);
        total_overlap_gain_chunks += pass_overlap_gain;

        pass_details.push(PassDetail {
            dst_chunks: pass.chunks.len(),
            src_chunks_read,
            unique_src_rows: source_rows.len(),
            sub_runs: n_sub,
            read_nnz: pass_read_nnz,
            write_nnz: pass_write_nnz,
            overlap_gain_chunks: pass_overlap_gain,
            avg_chunk_src_footprint: if pass.chunks.is_empty() {
                0.0
            } else {
                pass_chunk_footprint_sum as f64 / pass.chunks.len() as f64
            },
        });
    }
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
    let lower_bound_src_chunk_touches: usize = src_chunk_load_nnz.iter()
        .map(|&load_nnz| {
            if load_nnz == 0 {
                0
            } else {
                load_nnz.div_ceil(budget.max_nnz_per_pass as u64) as usize
            }
        })
        .sum();
    let lower_bound_read_bytes = lower_bound_src_chunk_touches as f64 * chunk_bytes;
    let used_src_chunks = src_chunk_load_nnz.iter().filter(|&&load| load > 0).count();
    let reused_src_chunks = src_chunk_pass_multiplicity.iter().filter(|&&m| m > 1).count();
    let mean_src_chunk_multiplicity = if used_src_chunks > 0 {
        total_src_chunks_read as f64 / used_src_chunks as f64
    } else {
        0.0
    };
    chunk_src_footprints.sort_unstable();
    chunk_src_span_rows.sort_unstable();
    let planning_metrics = PlanningMetrics {
        lower_bound_src_chunk_touches,
        lower_bound_read_gib: lower_bound_read_bytes / GIB,
        actual_vs_lower_bound: if lower_bound_src_chunk_touches > 0 {
            total_src_chunks_read as f64 / lower_bound_src_chunk_touches as f64
        } else {
            0.0
        },
        mean_src_chunk_multiplicity,
        max_src_chunk_multiplicity: src_chunk_pass_multiplicity.iter().copied().max().unwrap_or(0),
        reused_src_chunks,
        reused_src_chunks_pct: if used_src_chunks > 0 {
            reused_src_chunks as f64 * 100.0 / used_src_chunks as f64
        } else {
            0.0
        },
        overlap_gain_chunks: total_overlap_gain_chunks,
        overlap_gain_pct: if total_chunk_footprint_sum > 0 {
            total_overlap_gain_chunks as f64 * 100.0 / total_chunk_footprint_sum as f64
        } else {
            0.0
        },
        avg_chunk_src_footprint: if chunk_src_footprints.is_empty() {
            0.0
        } else {
            chunk_src_footprints.iter().sum::<usize>() as f64 / chunk_src_footprints.len() as f64
        },
        max_chunk_src_footprint: chunk_src_footprints.iter().copied().max().unwrap_or(0),
        median_chunk_src_span_rows: percentile_sorted(&chunk_src_span_rows, 0.50),
        p95_chunk_src_span_rows: percentile_sorted(&chunk_src_span_rows, 0.95),
    };

    println!();
    println!(
        "  Rows: {}  |  Total output NNZ: {}  ({:.1} GiB)",
        fmt(n_rows), fmt(total_output_nnz as usize),
        total_output_nnz as f64 * bytes_per_nnz as f64 / GIB
    );
    println!(
        "  Source chunks: {} x {} NNZ ({:.0} MiB/chunk)  |  Dest chunks: {} x {} NNZ",
        fmt(n_src_chunks), fmt(src_chunk_size),
        chunk_bytes / (1024.0 * 1024.0),
        fmt(n_dst_chunks_ideal), fmt(dst_chunk_size)
    );
    println!(
        "  Memory limit:  {:.1} GiB  |  max_nnz/pass: {}",
        memory_limit as f64 / GIB, fmt(budget.max_nnz_per_pass)
    );
    println!(
        "  Runtime model: {} threads  |  src_concurrent: {:.2} GiB  |  dst_budget: {:.2} GiB",
        budget.n_threads,
        budget.src_concurrent_bytes as f64 / GIB,
        budget.dst_budget_bytes as f64 / GIB,
    );
    println!();
    println!("  Passes:        {}", n_passes);
    println!("  Dst chunks:    {} written", fmt(n_dst_chunks_total));
    println!(
        "  Row fanout:    {} / {} nonempty rows cross dst chunks ({:.4}%), avg {:.4} chunks/nonempty-row, max {}",
        fmt(row_split_stats.rows_crossing_chunk_boundary),
        fmt(row_split_stats.nonempty_rows),
        row_split_stats.rows_crossing_chunk_boundary as f64 * 100.0 / row_split_stats.nonempty_rows.max(1) as f64,
        row_split_stats.avg_chunks_per_nonempty_row,
        row_split_stats.max_chunks_per_row,
    );
    println!(
        "  Source edges:  {} / {} nonempty rows cross source chunks ({:.4}%), max {} source chunks/row",
        fmt(src_boundary_stats.rows_crossing_source_chunk_boundary),
        fmt(src_boundary_stats.nonempty_rows),
        src_boundary_stats.rows_crossing_source_chunk_boundary as f64 * 100.0 / src_boundary_stats.nonempty_rows.max(1) as f64,
        src_boundary_stats.max_chunks_per_row,
    );
    let avg_sub = total_sub_runs as f64 / n_passes.max(1) as f64;
    println!(
        "  Sub-runs:      {} total ({:.1}/pass)",
        fmt(total_sub_runs), avg_sub
    );
    println!();
    println!(
        "  READ:          {:.2} GiB  ({} src chunks, {:.1}x amp)",
        total_read_bytes / GIB, fmt(total_src_chunks_read as usize), read_amp,
    );
    println!(
        "  WRITE:         {:.2} GiB  ({} dst chunks)",
        total_write_bytes / GIB, fmt(n_dst_chunks_total)
    );
    println!(
        "  TOTAL I/O:     {:.2} GiB",
        (total_read_bytes + total_write_bytes) / GIB
    );
    println!(
        "  Peak pass:     {:.2} GiB chunk buffers",
        max_pass_nnz as f64 * bytes_per_nnz as f64 / GIB
    );
    println!();
    println!("  Planner quality:");
    println!(
        "    Lower bound: {:.2} GiB read ({} src chunk touches)",
        planning_metrics.lower_bound_read_gib,
        fmt(planning_metrics.lower_bound_src_chunk_touches)
    );
    println!(
        "    Gap to LB:   {:.2}x actual vs lower bound",
        planning_metrics.actual_vs_lower_bound
    );
    println!(
        "    Src reuse:   mean {:.2}x, max {}x, {} / {} chunks reused across passes ({:.1}%)",
        planning_metrics.mean_src_chunk_multiplicity,
        planning_metrics.max_src_chunk_multiplicity,
        fmt(planning_metrics.reused_src_chunks),
        fmt(used_src_chunks),
        planning_metrics.reused_src_chunks_pct,
    );
    println!(
        "    Overlap:     {} duplicate chunk-touches removed within passes ({:.1}% overlap gain)",
        fmt(planning_metrics.overlap_gain_chunks),
        planning_metrics.overlap_gain_pct,
    );
    println!(
        "    Chunk span:  avg {:.2} src chunks/chunk, max {}, median {} source rows, p95 {} source rows",
        planning_metrics.avg_chunk_src_footprint,
        fmt(planning_metrics.max_chunk_src_footprint),
        fmt(planning_metrics.median_chunk_src_span_rows),
        fmt(planning_metrics.p95_chunk_src_span_rows),
    );

    if verbose && n_passes <= 80 {
        println!();
        println!(
            "  {:>5}  {:>8}  {:>9}  {:>10}  {:>8}  {:>9}  {:>9}  {:>8}  {:>8}",
            "Pass", "DstChks", "SrcChks", "SrcRows", "SubRuns", "ReadGiB", "WriteGiB", "Ovlp", "AvgFp"
        );
        for (i, d) in pass_details.iter().enumerate() {
            println!(
                "  {:5}  {:>8}  {:>9}  {:>10}  {:>8}  {:9.3}  {:9.3}  {:>8}  {:>8.2}",
                i,
                d.dst_chunks,
                d.src_chunks_read,
                fmt(d.unique_src_rows),
                d.sub_runs,
                d.read_nnz as f64 * bytes_per_nnz as f64 / GIB,
                d.write_nnz as f64 * bytes_per_nnz as f64 / GIB,
                fmt(d.overlap_gain_chunks),
                d.avg_chunk_src_footprint,
            );
            println!(
                "         writes: {}",
                summarize_pass_destinations(&passes[i])
            );
        }
    }

    println!();

    SimResult {
        passes: n_passes,
        dst_chunks_total: n_dst_chunks_total,
        src_chunks_read: total_src_chunks_read as usize,
        sub_runs: total_sub_runs,
        read_gib: total_read_bytes / GIB,
        write_gib: total_write_bytes / GIB,
        read_amp,
    }
}

fn compute_row_split_stats(
    assignments: &[RowAssignment],
    store_indptrs: &[Vec<i64>],
    dst_chunk_size: usize,
) -> RowSplitStats {
    if dst_chunk_size == 0 {
        return RowSplitStats {
            nonempty_rows: 0,
            rows_crossing_chunk_boundary: 0,
            max_chunks_per_row: 0,
            avg_chunks_per_nonempty_row: 0.0,
        };
    }

    let mut nonempty_rows = 0usize;
    let mut rows_crossing = 0usize;
    let mut total_chunk_refs = 0usize;
    let mut max_chunks_per_row = 0usize;

    for a in assignments {
        let indptr = &store_indptrs[a.store_id as usize];
        let row_nnz_start = indptr[a.output_row] as usize;
        let row_nnz_end = indptr[a.output_row + 1] as usize;
        if row_nnz_end <= row_nnz_start {
            continue;
        }
        nonempty_rows += 1;
        let first_chunk = row_nnz_start / dst_chunk_size;
        let last_chunk = (row_nnz_end - 1) / dst_chunk_size;
        let chunk_refs = last_chunk - first_chunk + 1;
        total_chunk_refs += chunk_refs;
        max_chunks_per_row = max_chunks_per_row.max(chunk_refs);
        if chunk_refs > 1 {
            rows_crossing += 1;
        }
    }

    RowSplitStats {
        nonempty_rows,
        rows_crossing_chunk_boundary: rows_crossing,
        max_chunks_per_row,
        avg_chunks_per_nonempty_row: if nonempty_rows > 0 {
            total_chunk_refs as f64 / nonempty_rows as f64
        } else {
            0.0
        },
    }
}

fn compute_source_boundary_stats(
    src_indptr: &[i64],
    src_chunk_size: usize,
) -> SourceBoundaryStats {
    if src_chunk_size == 0 || src_chunk_size == usize::MAX || src_indptr.len() < 2 {
        return SourceBoundaryStats {
            nonempty_rows: 0,
            rows_crossing_source_chunk_boundary: 0,
            max_chunks_per_row: 0,
        };
    }

    let mut nonempty_rows = 0usize;
    let mut rows_crossing = 0usize;
    let mut max_chunks_per_row = 0usize;
    for row in 0..src_indptr.len() - 1 {
        let lo = src_indptr[row] as usize;
        let hi = src_indptr[row + 1] as usize;
        if hi <= lo {
            continue;
        }
        nonempty_rows += 1;
        let first_chunk = lo / src_chunk_size;
        let last_chunk = (hi - 1) / src_chunk_size;
        let chunks = last_chunk - first_chunk + 1;
        max_chunks_per_row = max_chunks_per_row.max(chunks);
        if chunks > 1 {
            rows_crossing += 1;
        }
    }

    SourceBoundaryStats {
        nonempty_rows,
        rows_crossing_source_chunk_boundary: rows_crossing,
        max_chunks_per_row,
    }
}

fn summarize_pass_destinations(pass: &SparseScatterPass) -> String {
    let mut per_store: BTreeMap<u16, (u64, u64, usize)> = BTreeMap::new();
    for chunk in &pass.chunks {
        per_store
            .entry(chunk.store_id)
            .and_modify(|entry| {
                entry.0 = entry.0.min(chunk.chunk_idx);
                entry.1 = entry.1.max(chunk.chunk_idx);
                entry.2 += 1;
            })
            .or_insert((chunk.chunk_idx, chunk.chunk_idx, 1));
    }

    let mut parts = Vec::new();
    for (idx, (store_id, (first, last, count))) in per_store.iter().enumerate() {
        if idx >= 6 {
            parts.push(format!("... {} more stores", per_store.len() - idx));
            break;
        }
        parts.push(format!("s{}:c{}-{}({})", store_id, first, last, count));
    }
    parts.join("  ")
}

fn percentile_sorted(values: &[usize], q: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    let idx = ((values.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    values[idx]
}

// ---------------------------------------------------------------------------
// I/O helpers -- merged runs, sub-runs (mirrors engine logic)
// ---------------------------------------------------------------------------

struct MergedRun { nnz_start: usize, nnz_end: usize }
struct SubRun { nnz_start: usize, nnz_end: usize }

fn merge_source_reads(
    sorted_source_rows: &[usize], indptr: &[i64], gap_nnz: usize,
) -> Vec<MergedRun> {
    if sorted_source_rows.is_empty() { return Vec::new(); }
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
            runs.push(MergedRun { nnz_start: cur_start, nnz_end: cur_end });
            cur_start = lo;
            cur_end = hi;
        }
    }
    runs.push(MergedRun { nnz_start: cur_start, nnz_end: cur_end });
    runs
}

fn split_merged_runs_by_chunk(merged: &[MergedRun], chunk_size: usize) -> Vec<SubRun> {
    let mut out = Vec::new();
    for run in merged {
        if run.nnz_end <= run.nnz_start || chunk_size == usize::MAX {
            out.push(SubRun { nnz_start: run.nnz_start, nnz_end: run.nnz_end });
            continue;
        }
        let first_chunk = run.nnz_start / chunk_size;
        let last_chunk = (run.nnz_end - 1) / chunk_size;
        if first_chunk == last_chunk {
            out.push(SubRun { nnz_start: run.nnz_start, nnz_end: run.nnz_end });
        } else {
            for c in first_chunk..=last_chunk {
                let s = run.nnz_start.max(c * chunk_size);
                let e = run.nnz_end.min((c + 1) * chunk_size);
                if e > s {
                    out.push(SubRun { nnz_start: s, nnz_end: e });
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Zarr I/O
// ---------------------------------------------------------------------------

fn read_zarr_json(path: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("Invalid JSON in {}: {}", path.display(), e))
}

fn elem_size_from_dtype(dtype: &str) -> usize {
    match dtype {
        "float16" | "int16" | "uint16" => 2,
        "float32" | "int32" | "uint32" => 4,
        "float64" | "int64" | "uint64" => 8,
        "int8" | "uint8" | "bool" => 1,
        _ => 4,
    }
}

fn load_indptr_zarr(zarr_path: &Path, max_rows: Option<usize>) -> Vec<i64> {
    let meta_str = std::fs::read_to_string(zarr_path.join("zarr.json"))
        .unwrap_or_else(|e| panic!("Cannot read zarr.json in {:?}: {}", zarr_path, e));
    let meta: serde_json::Value = serde_json::from_str(&meta_str).expect("Invalid zarr.json");

    let shape = meta["shape"][0].as_u64().expect("missing shape") as usize;
    let chunk_size = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
        .as_u64().expect("missing chunk_shape") as usize;

    let n_chunks = (shape + chunk_size - 1) / chunk_size;
    let limit = max_rows.map_or(shape, |n| (n + 1).min(shape));

    let mut data = Vec::with_capacity(limit);
    let chunks_dir = zarr_path.join("c");

    for ci in 0..n_chunks {
        if data.len() >= limit { break; }
        let chunk_path = chunks_dir.join(ci.to_string());
        let compressed = std::fs::read(&chunk_path)
            .unwrap_or_else(|e| panic!("Cannot read chunk {:?}: {}", chunk_path, e));
        let decompressed = blosc_decompress(&compressed);
        let n_elems = decompressed.len() / 8;
        for j in 0..n_elems {
            if data.len() >= limit { break; }
            let val = i64::from_le_bytes(
                decompressed[j * 8..(j + 1) * 8].try_into().unwrap(),
            );
            data.push(val);
        }
    }
    data.truncate(limit);
    data
}

fn load_obs_codes_zarr(zarr_path: &Path, max_rows: usize) -> Result<Vec<i64>, String> {
    use zarrs::array::{Array, ArraySubset};
    use zarrs::filesystem::FilesystemStore;

    let store = Arc::new(
        FilesystemStore::new(zarr_path)
            .map_err(|e| format!("Cannot open store at {:?}: {}", zarr_path, e))?
    );
    let array = Array::open(store, "/")
        .map_err(|e| format!("Cannot open array at {:?}: {}", zarr_path, e))?;

    let n_elems = array.shape()[0] as usize;
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
            2 => i16::from_le_bytes(raw[j * 2..(j + 1) * 2].try_into().unwrap()) as i64,
            4 => i32::from_le_bytes(raw[j * 4..(j + 1) * 4].try_into().unwrap()) as i64,
            8 => i64::from_le_bytes(raw[j * 8..(j + 1) * 8].try_into().unwrap()),
            _ => return Err(format!("Unsupported element size: {}", elem_size)),
        };
        data.push(val);
    }
    Ok(data)
}

fn load_category_names(zarr_path: &Path) -> Option<Vec<String>> {
    use zarrs::array::Array;
    use zarrs::filesystem::FilesystemStore;

    let store = Arc::new(FilesystemStore::new(zarr_path).ok()?);
    let array = Array::open(store, "/").ok()?;
    let n = array.shape()[0] as usize;
    let subset = zarrs::array::ArraySubset::new_with_ranges(&[0..n as u64]);
    let elements = array.retrieve_array_subset::<Vec<String>>(&subset).ok()?;
    Some(elements)
}

fn blosc_decompress(input: &[u8]) -> Vec<u8> {
    if input.len() < 16 { return input.to_vec(); }

    use std::borrow::Cow;
    use zarrs::array::BytesToBytesCodecTraits;
    use zarrs::array::CodecOptions;
    use zarrs::array::BytesRepresentation;

    let nbytes = u32::from_le_bytes(input[4..8].try_into().unwrap()) as u64;
    let blosc_config = serde_json::json!({
        "cname": "lz4", "clevel": 3, "shuffle": "noshuffle", "typesize": 1, "blocksize": 0
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

fn build_multi_store_indptrs(
    assignments: &[RowAssignment], src_indptr: &[i64], store_n_rows: &[usize],
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

fn print_group_stats(column: &str, groups: &BTreeMap<i64, Vec<usize>>) {
    let n_groups = groups.len();
    let group_sizes: Vec<usize> = groups.values().map(|v| v.len()).collect();
    let min_g = *group_sizes.iter().min().unwrap_or(&0);
    let max_g = *group_sizes.iter().max().unwrap_or(&0);
    let total: usize = group_sizes.iter().sum();
    let mut sorted = group_sizes.clone();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];

    eprintln!();
    eprintln!(
        "  Column '{}': {} groups, {} total rows",
        column, n_groups, fmt(total)
    );
    eprintln!(
        "  Group sizes: min={} max={} median={}",
        fmt(min_g), fmt(max_g), fmt(median)
    );
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn fmt(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { result.push(','); }
        result.push(c);
    }
    result.chars().rev().collect()
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Args {
    input: PathBuf,
    output_dir: Option<PathBuf>,
    column: String,
    memory_gb: Vec<f64>,
    dst_chunk_size: Option<usize>,
    threads: Option<usize>,
    planner_mode: SparsePlannerMode,
    verbose: bool,
    run: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut column = String::from("cell_line");
    let mut memory_gb: Vec<f64> = Vec::new();
    let mut dst_chunk_size: Option<usize> = None;
    let mut threads: Option<usize> = None;
    let mut planner_mode = SparsePlannerMode::GroupbyAware;
    let mut verbose = false;
    let mut run = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                input = Some(PathBuf::from(&args[i]));
                i += 1;
            }
            "--output-dir" | "-o" => {
                i += 1;
                output_dir = Some(PathBuf::from(&args[i]));
                i += 1;
            }
            "--column" | "-c" => {
                i += 1;
                column = args[i].clone();
                i += 1;
            }
            "--memory-gb" | "-m" => {
                i += 1;
                while i < args.len() && !args[i].starts_with('-') {
                    memory_gb.push(args[i].parse().expect("invalid memory-gb value"));
                    i += 1;
                }
            }
            "--dst-chunk-size" => {
                i += 1;
                dst_chunk_size = Some(args[i].parse().expect("invalid dst-chunk-size"));
                i += 1;
            }
            "--threads" => {
                i += 1;
                threads = Some(args[i].parse().expect("invalid threads"));
                i += 1;
            }
            "--planner" => {
                i += 1;
                planner_mode = parse_planner_mode(&args[i]);
                i += 1;
            }
            "-v" | "--verbose" => { verbose = true; i += 1; }
            "--run" => { run = true; i += 1; }
            "--help" | "-h" => {
                eprintln!("Usage: e2e_groupby [OPTIONS]");
                eprintln!();
                eprintln!("  --input, -i PATH       Path to source AnnData .zarr store (required)");
                eprintln!("  --output-dir, -o PATH  Output directory for group stores [default: auto]");
                eprintln!("  --column, -c COL       obs column for groupby [default: cell_line]");
                eprintln!("  --memory-gb, -m N ...  Memory budgets to simulate [default: 8 16 32 64]");
                eprintln!("  --dst-chunk-size N     Dest NNZ chunk size [default: match source]");
                eprintln!("  --threads N            Override Rayon thread count for sim/run [default: runtime default]");
                eprintln!("  --planner MODE         auto|greedy|groupby-aware [default: groupby-aware]");
                eprintln!("  -v, --verbose          Print per-pass details");
                eprintln!("  --run                  Actually execute the scatter (default: simulate only)");
                eprintln!();
                eprintln!("Examples:");
                eprintln!("  # Simulate on small tahoe:");
                eprintln!("  cargo run -p anndata-ooc --release --example e2e_groupby -- \\");
                eprintln!("      -i /lustre/boost_ai/users/selman.ozleyen/data/tahoe10m_fast.zarr \\");
                eprintln!("      -c cell_line -m 16 32 64 -v");
                eprintln!();
                eprintln!("  # Simulate on full tahoe:");
                eprintln!("  cargo run -p anndata-ooc --release --example e2e_groupby -- \\");
                eprintln!("      -i /lustre/boost_ai/users/selman.ozleyen/data/tahoe.zarr \\");
                eprintln!("      -c cell_line -m 32 64 128 -v");
                eprintln!();
                eprintln!("  # Run for real:");
                eprintln!("  cargo run -p anndata-ooc --release --example e2e_groupby -- \\");
                eprintln!("      -i /lustre/boost_ai/users/selman.ozleyen/data/tahoe10m_fast.zarr \\");
                eprintln!("      -o /tmp/tahoe10m_groupby -c cell_line -m 32 --run");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                i += 1;
            }
        }
    }

    if memory_gb.is_empty() {
        memory_gb = vec![8.0, 16.0, 32.0, 64.0];
    }

    let input = input.unwrap_or_else(|| {
        eprintln!("ERROR: --input is required. Use --help for usage.");
        std::process::exit(1);
    });

    Args { input, output_dir, column, memory_gb, dst_chunk_size, threads, planner_mode, verbose, run }
}

fn parse_planner_mode(value: &str) -> SparsePlannerMode {
    match value {
        "auto" => SparsePlannerMode::Auto,
        "greedy" => SparsePlannerMode::Greedy,
        "groupby-aware" | "groupby_aware" | "groupby" => SparsePlannerMode::GroupbyAware,
        other => {
            eprintln!("ERROR: invalid planner mode '{}'. Use auto|greedy|groupby-aware.", other);
            std::process::exit(1);
        }
    }
}

fn planner_mode_label(mode: SparsePlannerMode) -> &'static str {
    match mode {
        SparsePlannerMode::Auto => "auto",
        SparsePlannerMode::Greedy => "greedy",
        SparsePlannerMode::GroupbyAware => "groupby-aware",
    }
}
