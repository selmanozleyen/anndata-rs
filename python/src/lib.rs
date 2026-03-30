use pyo3::{prelude::*, pymodule, types::PyModule, PyResult};
use numpy::PyReadonlyArray1;
use std::path::PathBuf;
use anyhow::Result;

#[pyfunction]
#[pyo3(
    name = "_scatter",
    signature = (input, outputs, *, memory_limit=None, chunk_size=None, shard_size=None, target_shard_bytes=None, compression_level=None),
    text_signature = "(input, outputs, *, memory_limit=None, chunk_size=None, shard_size=None, target_shard_bytes=None, compression_level=None)",
)]
pub fn scatter(
    py: Python<'_>,
    input: PathBuf,
    outputs: Vec<(PathBuf, PyReadonlyArray1<i64>)>,
    memory_limit: Option<usize>,
    chunk_size: Option<usize>,
    shard_size: Option<usize>,
    target_shard_bytes: Option<usize>,
    compression_level: Option<u8>,
) -> Result<()> {
    let config = anndata_ooc::ScatterConfig {
        memory_limit: memory_limit.unwrap_or(2 * 1024 * 1024 * 1024),
        chunk_size,
        shard_size,
        target_shard_bytes,
        compression_level,
        progress: None,
        planner_mode: anndata_ooc::SparsePlannerMode::Auto,
    };

    let mut assignments: Vec<anndata_ooc::RowAssignment> = Vec::new();
    let mut output_configs: Vec<anndata_ooc::OutputStoreConfig> = Vec::new();

    for (store_id, (path, indices)) in outputs.iter().enumerate() {
        let idx: Vec<usize> = indices.as_array().iter().map(|&v| v as usize).collect();
        output_configs.push(anndata_ooc::OutputStoreConfig {
            path: path.clone(),
            n_rows: idx.len(),
        });
        for (output_row, source_row) in idx.into_iter().enumerate() {
            assignments.push(anndata_ooc::RowAssignment {
                source_row,
                store_id: store_id as u16,
                output_row,
            });
        }
    }

    py.allow_threads(|| {
        anndata_ooc::scatter_anndata(&input, &output_configs, &assignments, &config)
    })
}

#[pymodule]
fn anndata_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(scatter, m)?)?;

    #[cfg(feature = "full")]
    {
        use pyanndata::*;

        m.add_class::<AnnData>()?;
        m.add_class::<AnnDataSet>()?;
        m.add_class::<PyCompression>()?;

        m.add_function(wrap_pyfunction!(read, m)?)?;
        m.add_function(wrap_pyfunction!(read_dataset, m)?)?;
        m.add_function(wrap_pyfunction!(read_mtx, m)?)?;
        m.add_function(wrap_pyfunction!(concat, m)?)?;
        m.add_function(wrap_pyfunction!(py_get_default_write_config, m)?)?;
        m.add_function(wrap_pyfunction!(py_set_default_write_config, m)?)?;
    }

    Ok(())
}
