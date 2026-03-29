use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, Context};
use ndarray::{Array1, ArrayD, Ix1};

use anndata::backend::{Backend, DataType, GroupOp, ScalarType, DatasetOp, AttributeOp};
use anndata::data::slice::SelectInfoElem;
use anndata_zarr::Zarr;
use zarrs::array::ArrayBuilder;
use zarrs::storage::ReadableWritableListableStorageTraits;

use crate::budget::{BufferPool, MemoryBudget};
use crate::dense_scatter::DenseScatterer;
use crate::sparse_scatter::{SparseScatterer, SparseStoreArrays};
use crate::scatter::RowAssignment;

/// Configuration for the scatter engine (shared across all output stores).
pub struct ScatterConfig {
    pub memory_limit: usize,
    pub chunk_size: Option<usize>,
    pub shard_size: Option<usize>,
    pub target_shard_bytes: Option<usize>,
    /// Zstd compression level for output arrays. ``None`` means match the
    /// source file's compression (falls back to 3 if undetectable).
    /// 0 = no compression.
    pub compression_level: Option<u8>,
}

impl Default for ScatterConfig {
    fn default() -> Self {
        Self {
            memory_limit: 2 * 1024 * 1024 * 1024,
            chunk_size: None,
            shard_size: None,
            target_shard_bytes: None,
            compression_level: None,
        }
    }
}

/// Per-output-store configuration and metadata.
pub struct OutputStoreConfig {
    pub path: std::path::PathBuf,
    pub n_rows: usize,
}

/// Scatter the matrix data of an AnnData Zarr store into one or more outputs.
///
/// Handles X, obsm, obsp, layers (the large row-indexed arrays) and copies
/// var, uns, varm, varp unchanged. obs is NOT handled here -- the caller
/// (Python) writes obs DataFrames directly, since pandas handles categoricals,
/// nullable dtypes, etc. natively.
///
pub fn scatter_anndata(
    src_path: &Path,
    outputs: &[OutputStoreConfig],
    assignments: &[RowAssignment],
    config: &ScatterConfig,
) -> Result<()> {
    use anndata::backend::StoreOp;

    let resolved_level = config.compression_level
        .unwrap_or_else(|| detect_source_zstd_level(src_path).unwrap_or(3));
    let resolved_config = ResolvedScatterConfig {
        base: config,
        compression_level: resolved_level,
    };

    let passthrough_possible = config.chunk_size.is_none()
        && config.shard_size.is_none()
        && config.target_shard_bytes.is_none();

    log::info!(
        "Scatter: compression_level={} (requested={:?}), passthrough_possible={}",
        resolved_level, config.compression_level, passthrough_possible
    );

    let src_store: <Zarr as Backend>::Store = Zarr::open(src_path)
        .with_context(|| format!("failed to open source: {}", src_path.display()))?;

    let dst_stores: Vec<<Zarr as Backend>::Store> = outputs.iter().map(|o| {
        Zarr::new(&o.path)
            .with_context(|| format!("failed to create destination: {}", o.path.display()))
    }).collect::<Result<Vec<_>>>()?;

    let budget = MemoryBudget::new(config.memory_limit);
    let pool = BufferPool::new(budget);

    let src_items = src_store.list()?;
    let store_n_rows: Vec<usize> = outputs.iter().map(|o| o.n_rows).collect();

    for name in &["var", "uns", "varm", "varp"] {
        if src_items.contains(&name.to_string()) {
            for dst_store in &dst_stores {
                copy_group(&src_store, dst_store, name)?;
            }
        }
    }

    if src_items.contains(&"X".to_string()) {
        scatter_matrix_element(
            &src_store, &dst_stores, "X", assignments, &store_n_rows, &pool,
            &resolved_config, passthrough_possible,
        )?;
    }

    for group_name in &["obsm", "obsp", "layers"] {
        if src_items.contains(&group_name.to_string()) {
            let src_group = src_store.open_group(group_name)?;
            let dst_groups: Vec<<Zarr as Backend>::Group> = dst_stores.iter()
                .map(|d| d.new_group(group_name))
                .collect::<Result<Vec<_>>>()?;

            let children = src_group.list()?;
            for child in &children {
                scatter_matrix_element_in_group(
                    &src_group, &dst_groups, child, assignments, &store_n_rows, &pool,
                    &resolved_config, passthrough_possible,
                )?;
            }
        }
    }

    for dst_store in dst_stores {
        dst_store.close()?;
    }
    src_store.close()?;

    log::info!(
        "Scatter complete: {} -> {} output stores",
        src_path.display(), outputs.len()
    );
    Ok(())
}

/// Resolved config with the compression level pinned to a concrete value.
struct ResolvedScatterConfig<'a> {
    base: &'a ScatterConfig,
    compression_level: u8,
}

/// Read the source X/zarr.json and extract the Zstd level from the codec chain.
fn detect_source_zstd_level(src_path: &Path) -> Option<u8> {
    let zarr_json = src_path.join("X").join("zarr.json");
    let data = std::fs::read_to_string(&zarr_json).ok()?;
    let v: serde_json::Value = serde_json::from_str(&data).ok()?;
    extract_zstd_level(&v)
}

fn extract_zstd_level(v: &serde_json::Value) -> Option<u8> {
    // Top-level codecs array
    if let Some(codecs) = v.get("codecs").and_then(|c| c.as_array()) {
        for codec in codecs {
            if let Some(level) = zstd_level_from_codec(codec) {
                return Some(level);
            }
            // Sharding codec embeds sub-codecs
            if codec.get("name").and_then(|n| n.as_str()) == Some("sharding_indexed") {
                if let Some(sub) = codec.pointer("/configuration/codecs").and_then(|c| c.as_array()) {
                    for sc in sub {
                        if let Some(level) = zstd_level_from_codec(sc) {
                            return Some(level);
                        }
                    }
                }
            }
        }
    }
    None
}

fn zstd_level_from_codec(codec: &serde_json::Value) -> Option<u8> {
    let name = codec.get("name").and_then(|n| n.as_str())?;
    if name == "zstd" {
        let level = codec.pointer("/configuration/level")
            .and_then(|l| l.as_i64())
            .unwrap_or(3);
        Some(level.clamp(0, 22) as u8)
    } else {
        None
    }
}

// --- Internal functions ---

fn scatter_matrix_element<G: GroupOp<Zarr>>(
    src_store: &G,
    dst_stores: &[<Zarr as Backend>::Store],
    name: &str,
    assignments: &[RowAssignment],
    store_n_rows: &[usize],
    pool: &BufferPool,
    config: &ResolvedScatterConfig,
    passthrough_possible: bool,
) -> Result<()> {
    scatter_matrix_element_in_group(
        src_store, dst_stores, name, assignments, store_n_rows, pool, config, passthrough_possible,
    )
}

fn scatter_matrix_element_in_group<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_groups: &[impl GroupOp<Zarr>],
    name: &str,
    assignments: &[RowAssignment],
    store_n_rows: &[usize],
    pool: &BufferPool,
    config: &ResolvedScatterConfig,
    passthrough_possible: bool,
) -> Result<()> {
    use anndata::backend::DataContainer;

    let container = DataContainer::<Zarr>::open(src_group, name)?;
    let encoding = container.encoding_type()?;

    match encoding {
        DataType::Array(scalar_type) | DataType::Scalar(scalar_type) => {
            let src_ds = src_group.open_dataset(name)?;
            let shape = src_ds.shape();
            if shape.ndim() >= 2 {
                scatter_dense_dataset(
                    src_group, dst_groups, name, assignments, store_n_rows,
                    scalar_type, pool, config, passthrough_possible,
                )?;
            } else {
                log::warn!("Skipping 1-D dataset '{}' (obs handled in Python)", name);
            }
        }
        DataType::CsrMatrix(scalar_type) => {
            scatter_csr_group(
                src_group, dst_groups, name, assignments, store_n_rows,
                scalar_type, pool, config, passthrough_possible,
            )?;
        }
        DataType::CscMatrix(_) => {
            log::warn!("CSC matrix '{}': falling back to per-store in-memory scatter", name);
            scatter_via_anndata_select(src_group, dst_groups, name, assignments, store_n_rows)?;
        }
        other => {
            log::warn!("Unsupported encoding {:?} for '{}', copying to all stores", other, name);
            for dst_group in dst_groups {
                copy_group_child(src_group, dst_group, name)?;
            }
        }
    }

    Ok(())
}

fn scatter_dense_dataset<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_groups: &[impl GroupOp<Zarr>],
    name: &str,
    assignments: &[RowAssignment],
    store_n_rows: &[usize],
    scalar_type: ScalarType,
    pool: &BufferPool,
    config: &ResolvedScatterConfig,
    passthrough_possible: bool,
) -> Result<()> {
    let src_ds = src_group.open_dataset(name)?;
    let src_shape = src_ds.shape();

    let use_cloned_metadata = passthrough_possible
        && config.base.compression_level.is_none();

    let mut dst_datasets: Vec<<Zarr as Backend>::Dataset> = Vec::with_capacity(dst_groups.len());
    for (i, dst_group) in dst_groups.iter().enumerate() {
        let mut out_shape = src_shape.as_ref().to_vec();
        out_shape[0] = store_n_rows[i];

        if use_cloned_metadata {
            let ds = clone_dataset_with_shape(
                src_ds.inner(), dst_group, name, &out_shape,
            )?;
            dst_datasets.push(ds);
        } else {
            let mut ds = dst_group.new_empty_dataset_typed_configured(
                name, scalar_type, &out_shape.into(),
                config.base.chunk_size, config.base.shard_size, config.base.target_shard_bytes,
                Some(config.compression_level),
            )?;
            copy_encoding_attrs(&src_ds, &mut ds)?;
            dst_datasets.push(ds);
        }
    }

    let dst_inners: Vec<_> = dst_datasets.iter().map(|d| d.inner()).collect();
    let dst_refs: Vec<&zarrs::array::Array<_>> = dst_inners.iter().map(|a| *a).collect();

    let can_passthrough = passthrough_possible && use_cloned_metadata;

    let scatterer = DenseScatterer::new(pool.clone_with_same_budget());
    scatterer.scatter(src_ds.inner(), &dst_refs, assignments, store_n_rows, can_passthrough)
}

fn scatter_csr_group<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_groups: &[impl GroupOp<Zarr>],
    name: &str,
    assignments: &[RowAssignment],
    store_n_rows: &[usize],
    scalar_type: ScalarType,
    pool: &BufferPool,
    config: &ResolvedScatterConfig,
    passthrough_possible: bool,
) -> Result<()> {
    let src_g = src_group.open_group(name)?;
    let shape_attr: Vec<u64> = src_g.get_attr("shape")?;
    let n_rows = shape_attr[0] as usize;
    let n_cols = shape_attr[1] as usize;

    let src_indptr_ds = src_g.open_dataset("indptr")?;
    let indptr_sel = [SelectInfoElem::from(0..n_rows + 1)];
    let src_indptr: Vec<i64> = src_indptr_ds
        .read_array_slice_cast::<i64, Ix1, _>(&indptr_sel)?
        .to_vec();

    let mut store_indptrs: Vec<Vec<i64>> = store_n_rows.iter()
        .map(|&n| vec![0i64; n + 1])
        .collect();

    for a in assignments {
        let row_nnz = src_indptr[a.source_row + 1] - src_indptr[a.source_row];
        let indptr = &mut store_indptrs[a.store_id as usize];
        indptr[a.output_row + 1] = row_nnz;
    }

    for indptr in &mut store_indptrs {
        for i in 1..indptr.len() {
            indptr[i] += indptr[i - 1];
        }
    }

    let src_data_ds = src_g.open_dataset("data")?;
    let src_indices_ds = src_g.open_dataset("indices")?;
    let indices_dtype = src_indices_ds.dtype()?;

    let use_cloned_metadata = passthrough_possible
        && config.base.compression_level.is_none();

    let mut store_arrays: Vec<SparseStoreArrays<'_, _>> = Vec::new();

    struct CsrStoreState {
        _group: <Zarr as Backend>::Group,
        data_ds: <Zarr as Backend>::Dataset,
        indices_ds: <Zarr as Backend>::Dataset,
    }

    let mut store_states: Vec<CsrStoreState> = Vec::new();

    for (store_id, dst_group) in dst_groups.iter().enumerate() {
        let n_out = store_n_rows[store_id];
        let total_nnz = *store_indptrs[store_id].last().unwrap() as usize;

        let mut dst_g = dst_group.new_group(name)?;
        dst_g.new_attr("encoding-type", "csr_matrix")?;
        dst_g.new_attr("encoding-version", "0.1.0")?;
        dst_g.new_attr("h5sparse_format", "csr")?;
        dst_g.new_attr("shape", [n_out as u64, n_cols as u64].as_slice())?;

        let indptr_arr: ArrayD<i64> = Array1::from_vec(store_indptrs[store_id].clone()).into_dyn();
        dst_g.new_array_dataset(
            "indptr",
            indptr_arr.view().into(),
            anndata::backend::get_default_write_config(),
        )?;

        let (dst_data_ds, dst_indices_ds) = if use_cloned_metadata && total_nnz > 0 {
            let dd = clone_dataset_with_shape(
                src_data_ds.inner(), &dst_g, "data", &[total_nnz],
            )?;
            let di = clone_dataset_with_shape(
                src_indices_ds.inner(), &dst_g, "indices", &[total_nnz],
            )?;
            (dd, di)
        } else {
            let dd = dst_g.new_empty_dataset_typed(
                "data", scalar_type, &vec![total_nnz].into(),
            )?;
            let di = dst_g.new_empty_dataset_typed(
                "indices", indices_dtype, &vec![total_nnz].into(),
            )?;
            (dd, di)
        };

        store_states.push(CsrStoreState {
            _group: dst_g,
            data_ds: dst_data_ds,
            indices_ds: dst_indices_ds,
        });
    }

    for (store_id, state) in store_states.iter().enumerate() {
        store_arrays.push(SparseStoreArrays {
            dst_indices: state.indices_ds.inner(),
            dst_data: state.data_ds.inner(),
            out_indptr: store_indptrs[store_id].clone(),
        });
    }

    let can_passthrough = passthrough_possible && use_cloned_metadata;

    let scatterer = SparseScatterer::new(pool.clone_with_same_budget());
    scatterer.scatter_data_indices(
        src_indices_ds.inner(),
        src_data_ds.inner(),
        &store_arrays,
        assignments,
        &src_indptr,
        can_passthrough,
    )?;

    Ok(())
}

fn scatter_via_anndata_select<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_groups: &[impl GroupOp<Zarr>],
    name: &str,
    assignments: &[RowAssignment],
    _store_n_rows: &[usize],
) -> Result<()> {
    use anndata::data::{ArrayData, data_traits::{ReadableArray, Writable}};
    use anndata::backend::DataContainer;

    let container = DataContainer::<Zarr>::open(src_group, name)?;

    for (store_id, dst_group) in dst_groups.iter().enumerate() {
        let mut store_perm: Vec<(usize, usize)> = assignments.iter()
            .filter(|a| a.store_id == store_id as u16)
            .map(|a| (a.output_row, a.source_row))
            .collect();
        store_perm.sort_unstable_by_key(|&(out, _)| out);

        let perm: Vec<usize> = store_perm.iter().map(|&(_, src)| src).collect();
        let sel = vec![
            SelectInfoElem::Index(perm),
            SelectInfoElem::full(),
        ];
        let data = ArrayData::read_select(&container, &sel)?;
        data.write(dst_group, name)?;
    }

    Ok(())
}

// --- Helpers ---

fn copy_encoding_attrs(
    src: &<Zarr as Backend>::Dataset,
    dst: &mut <Zarr as Backend>::Dataset,
) -> Result<()> {
    if let Ok(enc) = src.get_json_attr("encoding-type") {
        dst.new_json_attr("encoding-type", &enc)?;
    }
    if let Ok(ver) = src.get_json_attr("encoding-version") {
        dst.new_json_attr("encoding-version", &ver)?;
    }
    Ok(())
}

/// Clone a source zarrs Array with a new shape, preserving all codec/chunk
/// configuration. Returns a ZarrDataset backed by the destination group's store.
fn clone_dataset_with_shape<G: GroupOp<Zarr>>(
    src_arr: &zarrs::array::Array<dyn ReadableWritableListableStorageTraits>,
    dst_group: &G,
    name: &str,
    out_shape: &[usize],
) -> Result<<Zarr as Backend>::Dataset> {
    let shape_u64: Vec<u64> = out_shape.iter().map(|&s| s as u64).collect();
    let mut builder = ArrayBuilder::from_array(src_arr);
    builder.shape(shape_u64);

    let dst_ds = dst_group.open_dataset("__probe_for_store_path__");
    drop(dst_ds);

    let src_ds_for_store = src_arr;
    let _ = src_ds_for_store;

    let tmp_ds = dst_group.new_empty_dataset::<u8>(
        "__tmp_probe__",
        &vec![1usize].into(),
        anndata::backend::get_default_write_config(),
    )?;
    let probe_store: anndata_zarr::ZarrStore = tmp_ds.store()?;
    let probe_path = tmp_ds.path();
    let parent_path = probe_path.parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "/".to_string());
    dst_group.delete("__tmp_probe__")?;

    let dst_path = if parent_path == "/" || parent_path.is_empty() {
        format!("/{}", name)
    } else {
        format!("{}/{}", parent_path, name)
    };

    let store_arc: Arc<dyn ReadableWritableListableStorageTraits> = (*probe_store).clone();
    let dst_arr = builder.build(store_arc.clone(), &dst_path)?;
    dst_arr.store_metadata()?;

    let src_ds_tmp = dst_group.open_dataset(name)?;

    if let Some(enc) = src_arr.attributes().get("encoding-type") {
        let mut ds = src_ds_tmp;
        ds.new_json_attr("encoding-type", enc)?;
        if let Some(ver) = src_arr.attributes().get("encoding-version") {
            ds.new_json_attr("encoding-version", ver)?;
        }
        return Ok(ds);
    }

    Ok(src_ds_tmp)
}

fn copy_group<G: GroupOp<Zarr>>(
    src_store: &G,
    dst_store: &G,
    name: &str,
) -> Result<()> {
    use anndata::data::{Data, data_traits::{Readable, Writable}};
    use anndata::backend::DataContainer;

    let container = DataContainer::<Zarr>::open(src_store, name)?;
    let data = Data::read(&container)?;
    data.write(dst_store, name)?;
    Ok(())
}

fn copy_group_child<G1: GroupOp<Zarr>, G2: GroupOp<Zarr>>(
    src_group: &G1,
    dst_group: &G2,
    name: &str,
) -> Result<()> {
    use anndata::data::{Data, data_traits::{Readable, Writable}};
    use anndata::backend::DataContainer;

    let container = DataContainer::<Zarr>::open(src_group, name)?;
    let data = Data::read(&container)?;
    data.write(dst_group, name)?;
    Ok(())
}

/// Extension trait for creating typed empty datasets with optional chunk/shard config.
trait GroupOpExt<B: Backend>: GroupOp<B> {
    fn new_empty_dataset_typed(
        &self,
        name: &str,
        dtype: ScalarType,
        shape: &anndata::data::slice::Shape,
    ) -> Result<B::Dataset> {
        self.new_empty_dataset_typed_configured(name, dtype, shape, None, None, None, None)
    }

    fn new_empty_dataset_typed_configured(
        &self,
        name: &str,
        dtype: ScalarType,
        shape: &anndata::data::slice::Shape,
        chunk_size: Option<usize>,
        shard_size: Option<usize>,
        target_shard_bytes: Option<usize>,
        compression_level: Option<u8>,
    ) -> Result<B::Dataset> {
        let mut config = anndata::backend::get_default_write_config();
        if let Some(level) = compression_level {
            config.compression = if level == 0 {
                None
            } else {
                Some(anndata::backend::Compression::Zst(level))
            };
        }
        let ndim = shape.ndim();
        let shape_ref = shape.as_ref();

        // Sub-chunk: use full column width (only partition along axis 0).
        // For scatter workloads, splitting columns creates many tiny Zstd
        // frames which kills throughput.
        let default_chunk_rows = if ndim == 1 {
            shape_ref[0].min(16384).max(1)
        } else {
            chunk_size.unwrap_or(1024).min(shape_ref[0]).max(1)
        };

        let mut block: Vec<usize> = shape_ref.to_vec();
        block[0] = if let Some(cs) = chunk_size {
            cs.min(shape_ref[0]).max(1)
        } else {
            default_chunk_rows
        };
        config.block_size = Some(block.clone().into());

        // Shard: default to 8x sub-chunk rows, or use explicit config.
        let chunk_rows = block[0];
        let elem_size = scalar_type_elem_size(dtype);
        let full_cols: usize = block.iter().skip(1).product::<usize>().max(1);

        let n_rows = shape_ref[0];
        let shard_rows = if let Some(target_bytes) = target_shard_bytes {
            let row_bytes = full_cols * elem_size;
            let raw = if row_bytes > 0 { target_bytes / row_bytes } else { chunk_rows };
            round_up_to(raw.max(chunk_rows), chunk_rows)
        } else if let Some(ss) = shard_size {
            round_up_to(ss.max(chunk_rows), chunk_rows)
        } else {
            let ideal = chunk_rows * 8;
            if ideal >= n_rows {
                round_up_to(n_rows, chunk_rows)
            } else {
                ideal
            }
        };

        let mut shard = block.clone();
        shard[0] = shard_rows;
        config.shard_size = Some(shard.into());

        macro_rules! dispatch {
            ($($variant:ident => $ty:ty),+ $(,)?) => {
                match dtype {
                    $(ScalarType::$variant => self.new_empty_dataset::<$ty>(name, shape, config),)+
                }
            }
        }
        dispatch!(
            U8 => u8, U16 => u16, U32 => u32, U64 => u64,
            I8 => i8, I16 => i16, I32 => i32, I64 => i64,
            F32 => f32, F64 => f64, Bool => bool, String => String,
        )
    }
}

impl<T: GroupOp<Zarr>> GroupOpExt<Zarr> for T {}

fn scalar_type_elem_size(dtype: ScalarType) -> usize {
    match dtype {
        ScalarType::U8 | ScalarType::I8 | ScalarType::Bool => 1,
        ScalarType::U16 | ScalarType::I16 => 2,
        ScalarType::U32 | ScalarType::I32 | ScalarType::F32 => 4,
        ScalarType::U64 | ScalarType::I64 | ScalarType::F64 => 8,
        ScalarType::String => 8,
    }
}

fn round_up_to(value: usize, multiple: usize) -> usize {
    if multiple == 0 { return value; }
    ((value + multiple - 1) / multiple) * multiple
}
