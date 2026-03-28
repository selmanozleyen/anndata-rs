use std::collections::BTreeMap;
use std::ops::Range;

/// A row assignment mapping one source row to a destination (store, output_row).
#[derive(Debug, Clone, Copy)]
pub struct RowAssignment {
    pub source_row: usize,
    pub store_id: u16,
    pub output_row: usize,
}

/// Describes one output chunk across all stores. Extends ChunkAssignment
/// with a store_id so the flush step knows which destination to write to.
#[derive(Debug, Clone)]
pub struct ScatterChunk {
    pub store_id: u16,
    pub chunk_idx: u64,
    pub row_range: Range<usize>,
    /// (source_row, local_output_row_within_this_chunk)
    pub entries: Vec<ScatterEntry>,
}

#[derive(Debug, Clone, Copy)]
pub struct ScatterEntry {
    pub source_row: usize,
    /// Output row relative to the start of this chunk's row_range in its store
    pub local_row: usize,
}

/// A pass is a set of output chunks (potentially from different stores) that
/// can be filled simultaneously within the memory budget.
#[derive(Debug)]
pub struct ScatterPass {
    pub chunks: Vec<ScatterChunk>,
    pub total_rows: usize,
}

/// Plans multi-output scatter operations.
///
/// Given a list of (source_row, store_id, output_row) assignments and per-store
/// chunk sizes, partitions work into memory-budget-constrained passes where
/// source reads are sequential.
pub struct ScatterPlanner;

impl ScatterPlanner {
    /// Build a scatter plan from row assignments.
    ///
    /// `assignments`: unsorted list of RowAssignment
    /// `store_chunk_sizes`: chunk_size (rows per chunk) for each store
    /// `store_n_rows`: total output rows for each store
    /// `max_rows_in_memory`: memory budget in rows
    pub fn plan(
        assignments: &[RowAssignment],
        store_chunk_sizes: &[usize],
        store_n_rows: &[usize],
        max_rows_in_memory: usize,
    ) -> Vec<ScatterPass> {
        if assignments.is_empty() {
            return Vec::new();
        }

        // Step 1: Group assignments by (store_id, chunk_idx)
        // Key: (store_id, chunk_idx)
        let mut chunk_map: BTreeMap<(u16, u64), Vec<ScatterEntry>> = BTreeMap::new();

        for a in assignments {
            let cs = store_chunk_sizes[a.store_id as usize];
            let chunk_idx = (a.output_row / cs) as u64;
            let local_row = a.output_row % cs;
            chunk_map
                .entry((a.store_id, chunk_idx))
                .or_default()
                .push(ScatterEntry {
                    source_row: a.source_row,
                    local_row,
                });
        }

        // Step 2: Build ScatterChunks with row ranges
        let mut all_chunks: Vec<ScatterChunk> = Vec::with_capacity(chunk_map.len());

        for ((store_id, chunk_idx), entries) in chunk_map {
            let cs = store_chunk_sizes[store_id as usize];
            let n_rows = store_n_rows[store_id as usize];
            let row_start = chunk_idx as usize * cs;
            let row_end = (row_start + cs).min(n_rows);
            all_chunks.push(ScatterChunk {
                store_id,
                chunk_idx,
                row_range: row_start..row_end,
                entries,
            });
        }

        // Step 3: Pack chunks into passes that fit within memory
        let mut passes = Vec::new();
        let mut current_chunks = Vec::new();
        let mut current_rows = 0usize;

        for chunk in all_chunks {
            let chunk_rows = chunk.row_range.len();

            if chunk_rows > max_rows_in_memory {
                if !current_chunks.is_empty() {
                    passes.push(ScatterPass {
                        total_rows: current_rows,
                        chunks: std::mem::take(&mut current_chunks),
                    });
                    current_rows = 0;
                }
                passes.push(ScatterPass {
                    total_rows: chunk_rows,
                    chunks: vec![chunk],
                });
                continue;
            }

            if current_rows + chunk_rows > max_rows_in_memory {
                passes.push(ScatterPass {
                    total_rows: current_rows,
                    chunks: std::mem::take(&mut current_chunks),
                });
                current_rows = 0;
            }

            current_rows += chunk_rows;
            current_chunks.push(chunk);
        }

        if !current_chunks.is_empty() {
            passes.push(ScatterPass {
                total_rows: current_rows,
                chunks: current_chunks,
            });
        }

        passes
    }

    /// Within a pass, produce source-sorted read plan.
    /// Returns (source_row, store_id, local_row, chunk_index_in_pass) tuples
    /// sorted by source_row for sequential I/O.
    pub fn source_sorted_reads(pass: &ScatterPass) -> Vec<(usize, usize)> {
        let total: usize = pass.chunks.iter().map(|c| c.entries.len()).sum();
        let mut pairs = Vec::with_capacity(total);
        for (pass_chunk_idx, chunk) in pass.chunks.iter().enumerate() {
            for entry in &chunk.entries {
                // Encode pass_chunk_idx into the "output_row" slot so the caller
                // can find the right buffer. We use (source_row, packed_info).
                // packed_info = pass_chunk_idx << 32 | local_row
                let packed = (pass_chunk_idx << 32) | entry.local_row;
                pairs.push((entry.source_row, packed));
            }
        }
        pairs.sort_unstable_by_key(|&(src, _)| src);
        pairs
    }

    /// Build a single-store scatter plan from a permutation array (backwards compat).
    /// permutation[output_row] = source_row, supports duplicates and gaps.
    pub fn from_permutation(permutation: &[usize]) -> Vec<RowAssignment> {
        permutation
            .iter()
            .enumerate()
            .map(|(out_row, &src_row)| RowAssignment {
                source_row: src_row,
                store_id: 0,
                output_row: out_row,
            })
            .collect()
    }

    /// Build a multi-store scatter plan from a group-by split.
    /// `group_ids[i]` is the store_id for source row i.
    /// Returns (assignments, per_store_n_rows).
    pub fn from_groups(group_ids: &[u16], n_stores: usize) -> (Vec<RowAssignment>, Vec<usize>) {
        let mut store_counters = vec![0usize; n_stores];
        let mut assignments = Vec::with_capacity(group_ids.len());

        for (src_row, &store_id) in group_ids.iter().enumerate() {
            let output_row = store_counters[store_id as usize];
            store_counters[store_id as usize] += 1;
            assignments.push(RowAssignment {
                source_row: src_row,
                store_id,
                output_row,
            });
        }

        (assignments, store_counters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_store_identity() {
        let assignments = ScatterPlanner::from_permutation(&[0, 1, 2, 3, 4]);
        let passes = ScatterPlanner::plan(&assignments, &[3], &[5], 100);
        assert_eq!(passes.len(), 1);
        let total_entries: usize = passes[0]
            .chunks
            .iter()
            .map(|c| c.entries.len())
            .sum();
        assert_eq!(total_entries, 5);
    }

    #[test]
    fn single_store_with_duplicates() {
        // output rows 0,1,2 all come from source row 5
        let assignments = vec![
            RowAssignment {
                source_row: 5,
                store_id: 0,
                output_row: 0,
            },
            RowAssignment {
                source_row: 5,
                store_id: 0,
                output_row: 1,
            },
            RowAssignment {
                source_row: 5,
                store_id: 0,
                output_row: 2,
            },
        ];
        let passes = ScatterPlanner::plan(&assignments, &[4], &[3], 100);
        assert_eq!(passes.len(), 1);
        let total_entries: usize = passes[0]
            .chunks
            .iter()
            .map(|c| c.entries.len())
            .sum();
        assert_eq!(total_entries, 3);
    }

    #[test]
    fn multi_store_split() {
        let group_ids: Vec<u16> = vec![0, 1, 0, 1, 0, 2, 2];
        let (assignments, store_n_rows) = ScatterPlanner::from_groups(&group_ids, 3);
        assert_eq!(store_n_rows, vec![3, 2, 2]);
        assert_eq!(assignments.len(), 7);

        let passes = ScatterPlanner::plan(&assignments, &[4, 4, 4], &store_n_rows, 100);
        let total_entries: usize = passes
            .iter()
            .flat_map(|p| &p.chunks)
            .map(|c| c.entries.len())
            .sum();
        assert_eq!(total_entries, 7);
    }

    #[test]
    fn passes_respect_memory() {
        let assignments = ScatterPlanner::from_permutation(&(0..100).collect::<Vec<_>>());
        let passes = ScatterPlanner::plan(&assignments, &[32], &[100], 40);
        assert!(passes.len() >= 2);
        for pass in &passes {
            // Each pass should not exceed max_rows_in_memory (40)
            // except for single oversized chunks
            assert!(pass.total_rows <= 40 || pass.chunks.len() == 1);
        }
    }

    #[test]
    fn source_sorted_reads_are_sequential() {
        let assignments = ScatterPlanner::from_permutation(&[5, 2, 8, 1, 9, 0, 3, 7]);
        let passes = ScatterPlanner::plan(&assignments, &[4], &[8], 100);
        assert_eq!(passes.len(), 1);
        let reads = ScatterPlanner::source_sorted_reads(&passes[0]);
        for window in reads.windows(2) {
            assert!(window[0].0 <= window[1].0, "source reads must be sorted");
        }
    }
}
