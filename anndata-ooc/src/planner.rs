use std::collections::BTreeMap;
use std::ops::Range;

/// Describes one output chunk (or shard). All rows destined for the same
/// chunk are collected here so the chunk can be written in a single
/// `store_chunk` call with zero read-modify-write overhead.
#[derive(Debug, Clone)]
pub struct ChunkAssignment {
    /// Output chunk index along axis 0.
    pub chunk_idx: u64,
    /// Row range this chunk covers in the output array: [row_start, row_end).
    pub row_range: Range<usize>,
    /// For each row in row_range, the source row it comes from.
    /// entries[i] corresponds to output row row_range.start + i.
    pub entries: Vec<SourceEntry>,
}

/// One row assignment: output_row comes from source_row.
#[derive(Debug, Clone, Copy)]
pub struct SourceEntry {
    pub output_row: usize,
    pub source_row: usize,
}

/// A pass is a set of output chunks that can be filled simultaneously
/// within the memory budget. Within a pass the source reads are sorted
/// for sequential I/O.
#[derive(Debug)]
pub struct ShardPass {
    /// Output chunks to fill in this pass, keyed by chunk index.
    pub chunks: Vec<ChunkAssignment>,
    /// Total number of output rows across all chunks in this pass.
    pub total_rows: usize,
}

/// Plans out-of-core permutation using output-shard-aware partitioning.
///
/// The strategy mirrors what database systems do for hash-partitioned
/// external sorts and shuffle writes:
///
/// 1. Partition the permutation by output chunk (shard) boundaries
/// 2. Group output chunks into passes that fit within the memory budget
/// 3. Within each pass, sort source row accesses for sequential I/O
/// 4. Each chunk is written via `store_chunk` (full-chunk write, no RMW)
pub struct ShardPlanner;

impl ShardPlanner {
    /// Partition a permutation into passes of output chunks.
    ///
    /// - `permutation[output_row] = source_row`
    /// - `chunk_size`: number of rows per output chunk along axis 0
    /// - `n_output_rows`: total output rows (= permutation.len())
    /// - `max_rows_in_memory`: max rows that fit in the memory budget at once
    ///
    /// Returns a list of passes. Each pass contains a set of output chunks
    /// whose combined row count fits within `max_rows_in_memory`.
    pub fn plan(
        permutation: &[usize],
        chunk_size: usize,
        n_output_rows: usize,
        max_rows_in_memory: usize,
    ) -> Vec<ShardPass> {
        if permutation.is_empty() || chunk_size == 0 {
            return Vec::new();
        }

        // Step 1: Partition output rows by which chunk they belong to.
        let mut chunk_map: BTreeMap<u64, Vec<SourceEntry>> = BTreeMap::new();

        for (out_row, &src_row) in permutation.iter().enumerate() {
            let chunk_idx = (out_row / chunk_size) as u64;
            chunk_map.entry(chunk_idx).or_default().push(SourceEntry {
                output_row: out_row,
                source_row: src_row,
            });
        }

        // Step 2: Build ChunkAssignments with row ranges.
        let n_chunks = (n_output_rows + chunk_size - 1) / chunk_size;
        let mut assignments: Vec<ChunkAssignment> = Vec::with_capacity(n_chunks);

        for (chunk_idx, entries) in chunk_map {
            let row_start = chunk_idx as usize * chunk_size;
            let row_end = (row_start + chunk_size).min(n_output_rows);
            assignments.push(ChunkAssignment {
                chunk_idx,
                row_range: row_start..row_end,
                entries,
            });
        }

        // Step 3: Pack chunks into passes that fit within memory.
        let mut passes = Vec::new();
        let mut current_chunks = Vec::new();
        let mut current_rows = 0usize;

        for assignment in assignments {
            let chunk_rows = assignment.row_range.len();

            if chunk_rows > max_rows_in_memory {
                // Edge case: single chunk exceeds budget. Must process alone.
                if !current_chunks.is_empty() {
                    passes.push(ShardPass {
                        total_rows: current_rows,
                        chunks: std::mem::take(&mut current_chunks),
                    });
                    current_rows = 0;
                }
                passes.push(ShardPass {
                    total_rows: chunk_rows,
                    chunks: vec![assignment],
                });
                continue;
            }

            if current_rows + chunk_rows > max_rows_in_memory {
                passes.push(ShardPass {
                    total_rows: current_rows,
                    chunks: std::mem::take(&mut current_chunks),
                });
                current_rows = 0;
            }

            current_rows += chunk_rows;
            current_chunks.push(assignment);
        }

        if !current_chunks.is_empty() {
            passes.push(ShardPass {
                total_rows: current_rows,
                chunks: current_chunks,
            });
        }

        passes
    }

    /// Within a pass, produce a source-sorted read plan.
    /// Returns (source_row, output_row) pairs sorted by source_row
    /// for sequential I/O on the source store.
    pub fn source_sorted_reads(pass: &ShardPass) -> Vec<(usize, usize)> {
        let total: usize = pass.chunks.iter().map(|c| c.entries.len()).sum();
        let mut pairs = Vec::with_capacity(total);
        for chunk in &pass.chunks {
            for entry in &chunk.entries {
                pairs.push((entry.source_row, entry.output_row));
            }
        }
        pairs.sort_unstable_by_key(|&(src, _)| src);
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_identity_permutation() {
        let perm: Vec<usize> = (0..100).collect();
        let passes = ShardPlanner::plan(&perm, 32, 100, 100);
        assert_eq!(passes.len(), 1);
        assert_eq!(passes[0].total_rows, 100);
        let total_entries: usize = passes[0].chunks.iter()
            .map(|c| c.entries.len()).sum();
        assert_eq!(total_entries, 100);
    }

    #[test]
    fn plan_splits_into_passes() {
        let perm: Vec<usize> = (0..100).collect();
        let passes = ShardPlanner::plan(&perm, 32, 100, 40);
        // 4 chunks of 32,32,32,4 rows -> passes of [32], [32], [32+4=36]
        // or [32], [32], [32], [4] depending on packing
        assert!(passes.len() >= 2);
        let total: usize = passes.iter()
            .flat_map(|p| &p.chunks)
            .map(|c| c.entries.len())
            .sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn plan_reverse_permutation() {
        let perm: Vec<usize> = (0..20).rev().collect();
        let passes = ShardPlanner::plan(&perm, 8, 20, 100);
        // All 20 rows fit in one pass
        assert_eq!(passes.len(), 1);
        // 3 chunks: [0..8], [8..16], [16..20]
        assert_eq!(passes[0].chunks.len(), 3);
    }

    #[test]
    fn source_sorted_reads_are_sequential() {
        let perm = vec![5, 2, 8, 1, 9, 0, 3, 7];
        let passes = ShardPlanner::plan(&perm, 4, 8, 100);
        assert_eq!(passes.len(), 1);
        let reads = ShardPlanner::source_sorted_reads(&passes[0]);
        for window in reads.windows(2) {
            assert!(window[0].0 <= window[1].0, "source reads must be sorted");
        }
    }

    #[test]
    fn chunk_boundaries_correct() {
        let perm: Vec<usize> = (0..10).collect();
        let passes = ShardPlanner::plan(&perm, 4, 10, 100);
        let chunks: Vec<_> = passes.iter()
            .flat_map(|p| &p.chunks)
            .collect();
        assert_eq!(chunks[0].row_range, 0..4);
        assert_eq!(chunks[1].row_range, 4..8);
        assert_eq!(chunks[2].row_range, 8..10);
    }
}
