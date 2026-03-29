mod budget;
mod scatter;
mod dense_scatter;
mod sparse_scatter;
mod scatter_engine;

pub use budget::{MemoryBudget, BufferPool};
pub use scatter::{RowAssignment, ScatterPlanner, SparseScatterPass, SparseScatterChunk, SparseScatterEntry};
pub use scatter_engine::{scatter_anndata, ScatterConfig, OutputStoreConfig, ProgressCounter};
pub use sparse_scatter::{SparseScatterer, SparseStoreArrays};
