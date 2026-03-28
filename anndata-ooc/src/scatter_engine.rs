use std::path::{Path, PathBuf};
use std::collections::HashMap;

use anyhow::{Result, Context};
use ndarray::{Array1, ArrayD, Ix1};

use anndata::backend::{Backend, DataType, GroupOp, ScalarType, DatasetOp, AttributeOp};
use anndata::data::slice::SelectInfoElem;
use anndata_zarr::Zarr;

use crate::budget::{BufferPool, MemoryBudget};
use crate::dense_scatter::DenseScatterer;
use crate::sparse_scatter::{SparseScatterer, SparseStoreArrays};
use crate::scatter::{RowAssignment, ScatterPlanner};

/// Configuration for the scatter engine (shared across all output stores).
pub struct ScatterConfig {
    pub memory_limit: usize,
    pub chunk_size: Option<usize>,
    pub shard_size: Option<usize>,
    pub target_shard_bytes: Option<usize>,
}

impl Default for ScatterConfig {
    fn default() -> Self {
        Self {
            memory_limit: 2 * 1024 * 1024 * 1024,
            chunk_size: None,
            shard_size: None,
            target_shard_bytes: None,
        }
    }
}

impl From<&crate::permute::PermuteConfig> for ScatterConfig {
    fn from(pc: &crate::permute::PermuteConfig) -> Self {
        Self {
            memory_limit: pc.memory_limit,
            chunk_size: pc.chunk_size,
            shard_size: pc.shard_size,
            target_shard_bytes: pc.target_shard_bytes,
        }
    }
}

/// Per-output-store configuration and metadata.
pub struct OutputStoreConfig {
    pub path: PathBuf,
    pub n_rows: usize,
}

/// Scatter an AnnData Zarr store into one or more output stores.
///
/// This is the unified I/O engine. `assignments` maps each source row to a
/// (store_id, output_row) pair. Multiple output stores share a single source
/// read pass for maximum I/O efficiency.
pub fn scatter_anndata(
    src_path: &Path,
    outputs: &[OutputStoreConfig],
    assignments: &[RowAssignment],
    config: &ScatterConfig,
) -> Result<()> {
    use anndata::backend::StoreOp;

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

    // Copy var, uns, varm, varp to all output stores unchanged
    for name in &["var", "uns", "varm", "varp"] {
        if src_items.contains(&name.to_string()) {
            for dst_store in &dst_stores {
                copy_group(&src_store, dst_store, name)?;
            }
        }
    }

    // Scatter obs (row annotations)
    if src_items.contains(&"obs".to_string()) {
        scatter_dataframe_group(&src_store, &dst_stores, "obs", assignments, &store_n_rows)?;
    }

    // Scatter X
    if src_items.contains(&"X".to_string()) {
        scatter_matrix_element(
            &src_store, &dst_stores, "X", assignments, &store_n_rows, &pool, config,
        )?;
    }

    // Scatter obsm, obsp, layers
    for group_name in &["obsm", "obsp", "layers"] {
        if src_items.contains(&group_name.to_string()) {
            let src_group = src_store.open_group(group_name)?;
            let dst_groups: Vec<<Zarr as Backend>::Group> = dst_stores.iter()
                .map(|d| d.new_group(group_name))
                .collect::<Result<Vec<_>>>()?;

            let children = src_group.list()?;
            for child in &children {
                scatter_matrix_element_in_group(
                    &src_group, &dst_groups, child, assignments, &store_n_rows, &pool, config,
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

/// Split an AnnData Zarr store by the values of an obs column.
///
/// Each unique value in the column becomes a separate output store.
/// Returns the list of (value, output_path) pairs.
pub fn split_anndata(
    src_path: &Path,
    output_dir: &Path,
    column: &str,
    config: &ScatterConfig,
) -> Result<Vec<(String, PathBuf)>> {
    let src_store: <Zarr as Backend>::Store = Zarr::open(src_path)
        .with_context(|| format!("failed to open source: {}", src_path.display()))?;

    // Read the obs column to determine group membership
    let obs_group = src_store.open_group("obs")?;
    let col_ds = obs_group.open_dataset(column)
        .with_context(|| format!("obs column '{}' not found", column))?;
    let n_rows = col_ds.shape()[0];

    let sel = [SelectInfoElem::from(0..n_rows)];
    let col_data = col_ds.read_dyn_array_slice(&sel)?;

    // Convert values to strings for grouping
    let string_values = dyn_array_to_strings(&col_data)?;

    {
        use anndata::backend::StoreOp;
        src_store.close()?;
    }

    // Build group mapping: value -> store_id
    let mut value_to_id: HashMap<String, u16> = HashMap::new();
    let mut group_names: Vec<String> = Vec::new();
    let mut group_ids: Vec<u16> = Vec::with_capacity(n_rows);

    for val in &string_values {
        let id = if let Some(&id) = value_to_id.get(val) {
            id
        } else {
            let id = group_names.len() as u16;
            value_to_id.insert(val.clone(), id);
            group_names.push(val.clone());
            id
        };
        group_ids.push(id);
    }

    let n_stores = group_names.len();
    let (assignments, store_n_rows) = ScatterPlanner::from_groups(&group_ids, n_stores);

    // Build output store configs
    let outputs: Vec<OutputStoreConfig> = group_names.iter().enumerate().map(|(i, name)| {
        let safe_name = name.replace(['/', '\\', ' '], "_");
        OutputStoreConfig {
            path: output_dir.join(format!("{}.zarr", safe_name)),
            n_rows: store_n_rows[i],
        }
    }).collect();

    let result: Vec<(String, PathBuf)> = group_names.iter()
        .zip(outputs.iter())
        .map(|(name, o)| (name.clone(), o.path.clone()))
        .collect();

    scatter_anndata(src_path, &outputs, &assignments, config)?;

    Ok(result)
}

// --- Internal functions ---

fn scatter_matrix_element<G: GroupOp<Zarr>>(
    src_store: &G,
    dst_stores: &[<Zarr as Backend>::Store],
    name: &str,
    assignments: &[RowAssignment],
    store_n_rows: &[usize],
    pool: &BufferPool,
    config: &ScatterConfig,
) -> Result<()> {
    scatter_matrix_element_in_group(src_store, dst_stores, name, assignments, store_n_rows, pool, config)
}

fn scatter_matrix_element_in_group<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_groups: &[impl GroupOp<Zarr>],
    name: &str,
    assignments: &[RowAssignment],
    store_n_rows: &[usize],
    pool: &BufferPool,
    config: &ScatterConfig,
) -> Result<()> {
    use anndata::backend::DataContainer;

    let container = DataContainer::<Zarr>::open(src_group, name)?;
    let encoding = container.encoding_type()?;

    match encoding {
        DataType::Array(scalar_type) | DataType::Scalar(scalar_type) => {
            let src_ds = src_group.open_dataset(name)?;
            let shape = src_ds.shape();
            if shape.ndim() < 2 {
                scatter_1d_dataset(src_group, dst_groups, name, assignments, store_n_rows)?;
            } else {
                scatter_dense_dataset(
                    src_group, dst_groups, name, assignments, store_n_rows,
                    scalar_type, pool, config,
                )?;
            }
        }
        DataType::CsrMatrix(scalar_type) => {
            scatter_csr_group(
                src_group, dst_groups, name, assignments, store_n_rows,
                scalar_type, pool, config,
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
    config: &ScatterConfig,
) -> Result<()> {
    let src_ds = src_group.open_dataset(name)?;
    let src_shape = src_ds.shape();

    // Create destination datasets
    let mut dst_datasets: Vec<<Zarr as Backend>::Dataset> = Vec::with_capacity(dst_groups.len());
    for (i, dst_group) in dst_groups.iter().enumerate() {
        let mut out_shape = src_shape.as_ref().to_vec();
        out_shape[0] = store_n_rows[i];

        let mut ds = dst_group.new_empty_dataset_typed_configured(
            name, scalar_type, &out_shape.into(),
            config.chunk_size, config.shard_size, config.target_shard_bytes,
        )?;
        copy_encoding_attrs(&src_ds, &mut ds)?;
        dst_datasets.push(ds);
    }

    // Get raw zarrs Array references
    let dst_inners: Vec<_> = dst_datasets.iter().map(|d| d.inner()).collect();
    let dst_refs: Vec<&zarrs::array::Array<_>> = dst_inners.iter().map(|a| *a).collect();

    let scatterer = DenseScatterer::new(pool.clone_with_same_budget());
    scatterer.scatter(src_ds.inner(), &dst_refs, assignments, store_n_rows)
}

fn scatter_1d_dataset<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_groups: &[impl GroupOp<Zarr>],
    name: &str,
    assignments: &[RowAssignment],
    store_n_rows: &[usize],
) -> Result<()> {
    let src_ds = src_group.open_dataset(name)?;
    let dtype = src_ds.dtype()?;
    let n = src_ds.shape()[0];

    let sel = [SelectInfoElem::from(0..n)];
    let full = src_ds.read_dyn_array_slice(&sel)?;

    // Build per-store permutation arrays from assignments
    for (store_id, dst_group) in dst_groups.iter().enumerate() {
        let n_out = store_n_rows[store_id];
        let mut out_shape = src_ds.shape().as_ref().to_vec();
        out_shape[0] = n_out;

        let mut dst_ds = dst_group.new_empty_dataset_typed(
            name, dtype, &out_shape.into(),
        )?;
        copy_encoding_attrs(&src_ds, &mut dst_ds)?;

        // Collect the source rows for this store in output order
        let mut store_perm: Vec<(usize, usize)> = assignments.iter()
            .filter(|a| a.store_id == store_id as u16)
            .map(|a| (a.output_row, a.source_row))
            .collect();
        store_perm.sort_unstable_by_key(|&(out, _)| out);

        let perm: Vec<usize> = store_perm.iter().map(|&(_, src)| src).collect();
        let permuted = permute_dyn_array_1d(&full, &perm);
        let write_sel = vec![SelectInfoElem::from(0..n_out)];
        write_dyn_slice(&dst_ds, &permuted, &write_sel, dtype)?;
    }

    Ok(())
}

fn scatter_csr_group<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_groups: &[impl GroupOp<Zarr>],
    name: &str,
    assignments: &[RowAssignment],
    store_n_rows: &[usize],
    scalar_type: ScalarType,
    pool: &BufferPool,
    _config: &ScatterConfig,
) -> Result<()> {
    let src_g = src_group.open_group(name)?;
    let shape_attr: Vec<u64> = src_g.get_attr("shape")?;
    let n_rows = shape_attr[0] as usize;
    let n_cols = shape_attr[1] as usize;

    // Read source indptr
    let src_indptr_ds = src_g.open_dataset("indptr")?;
    let indptr_sel = [SelectInfoElem::from(0..n_rows + 1)];
    let src_indptr: Vec<i64> = src_indptr_ds
        .read_array_slice_cast::<i64, Ix1, _>(&indptr_sel)?
        .to_vec();

    // Build per-store output indptr
    let mut store_indptrs: Vec<Vec<i64>> = store_n_rows.iter()
        .map(|&n| vec![0i64; n + 1])
        .collect();

    // For each assignment, accumulate NNZ into the right store's indptr
    for a in assignments {
        let row_nnz = src_indptr[a.source_row + 1] - src_indptr[a.source_row];
        let indptr = &mut store_indptrs[a.store_id as usize];
        indptr[a.output_row + 1] = row_nnz;
    }

    // Prefix sum to get actual indptr values
    for indptr in &mut store_indptrs {
        for i in 1..indptr.len() {
            indptr[i] += indptr[i - 1];
        }
    }

    // Create output groups and datasets
    let src_data_ds = src_g.open_dataset("data")?;
    let src_indices_ds = src_g.open_dataset("indices")?;
    let indices_dtype = src_indices_ds.dtype()?;

    let mut store_arrays: Vec<SparseStoreArrays<'_, _>> = Vec::new();

    // We need to keep the groups alive, so collect them
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

        let dst_data_ds = dst_g.new_empty_dataset_typed(
            "data", scalar_type, &vec![total_nnz].into(),
        )?;
        let dst_indices_ds = dst_g.new_empty_dataset_typed(
            "indices", indices_dtype, &vec![total_nnz].into(),
        )?;

        store_states.push(CsrStoreState {
            _group: dst_g,
            data_ds: dst_data_ds,
            indices_ds: dst_indices_ds,
        });
    }

    // Build SparseStoreArrays references
    for (store_id, state) in store_states.iter().enumerate() {
        store_arrays.push(SparseStoreArrays {
            dst_indices: state.indices_ds.inner(),
            dst_data: state.data_ds.inner(),
            out_indptr: store_indptrs[store_id].clone(),
        });
    }

    let scatterer = SparseScatterer::new(pool.clone_with_same_budget());
    scatterer.scatter_data_indices(
        src_indices_ds.inner(),
        src_data_ds.inner(),
        &store_arrays,
        assignments,
        &src_indptr,
    )?;

    Ok(())
}

fn scatter_dataframe_group(
    src_store: &<Zarr as Backend>::Store,
    dst_stores: &[<Zarr as Backend>::Store],
    name: &str,
    assignments: &[RowAssignment],
    store_n_rows: &[usize],
) -> Result<()> {
    let src_g = src_store.open_group(name)?;

    let mut dst_groups: Vec<<Zarr as Backend>::Group> = Vec::new();
    for dst_store in dst_stores {
        let mut dst_g = dst_store.new_group(name)?;

        if let Ok(enc) = src_g.get_json_attr("encoding-type") {
            dst_g.new_json_attr("encoding-type", &enc)?;
        }
        if let Ok(enc) = src_g.get_json_attr("encoding-version") {
            dst_g.new_json_attr("encoding-version", &enc)?;
        }
        if let Ok(idx) = src_g.get_json_attr("_index") {
            dst_g.new_json_attr("_index", &idx)?;
        }
        if let Ok(ord) = src_g.get_json_attr("column-order") {
            dst_g.new_json_attr("column-order", &ord)?;
        }

        dst_groups.push(dst_g);
    }

    let children = src_g.list()?;
    for col_name in &children {
        scatter_1d_dataset(&src_g, &dst_groups, col_name, assignments, store_n_rows)?;
    }

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

use anndata::data::array::DynArray;

fn permute_dyn_array_1d(arr: &DynArray, permutation: &[usize]) -> DynArray {
    macro_rules! permute_1d {
        ($arr:expr, $perm:expr, $( $variant:ident ),+ $(,)?) => {
            match $arr {
                $(
                    DynArray::$variant(a) => {
                        let src = a.as_slice().expect("1D array must be contiguous");
                        let permuted: Vec<_> = $perm.iter().map(|&i| src[i].clone()).collect();
                        DynArray::$variant(Array1::from_vec(permuted).into_dyn())
                    }
                )+
            }
        }
    }
    permute_1d!(arr, permutation, U8, U16, U32, U64, I8, I16, I32, I64, F32, F64, Bool, String)
}

fn write_dyn_slice(
    ds: &<Zarr as Backend>::Dataset,
    arr: &DynArray,
    sel: &[SelectInfoElem],
    _dtype: ScalarType,
) -> Result<()> {
    macro_rules! write_typed {
        ($ds:expr, $arr:expr, $sel:expr, $( $variant:ident => $ty:ty ),+ $(,)?) => {
            match $arr {
                $(
                    DynArray::$variant(a) => $ds.write_array_slice(a.view().into(), $sel),
                )+
            }
        }
    }

    write_typed!(ds, arr, sel,
        U8 => u8,
        U16 => u16,
        U32 => u32,
        U64 => u64,
        I8 => i8,
        I16 => i16,
        I32 => i32,
        I64 => i64,
        F32 => f32,
        F64 => f64,
        Bool => bool,
        String => String,
    )
}

fn dyn_array_to_strings(arr: &DynArray) -> Result<Vec<String>> {
    macro_rules! to_strings {
        ($arr:expr, $( $variant:ident ),+ $(,)?) => {
            match $arr {
                $(
                    DynArray::$variant(a) => {
                        let s = a.as_slice().expect("1D array must be contiguous");
                        Ok(s.iter().map(|v| format!("{}", v)).collect())
                    }
                )+
                DynArray::String(a) => {
                    let s = a.as_slice().expect("1D array must be contiguous");
                    Ok(s.to_vec())
                }
                DynArray::Bool(a) => {
                    let s = a.as_slice().expect("1D array must be contiguous");
                    Ok(s.iter().map(|v| format!("{}", v)).collect())
                }
            }
        }
    }
    to_strings!(arr, U8, U16, U32, U64, I8, I16, I32, I64, F32, F64)
}

/// Extension trait for creating typed empty datasets (mirrors the one in permute.rs).
trait GroupOpExt<B: Backend>: GroupOp<B> {
    fn new_empty_dataset_typed(
        &self,
        name: &str,
        dtype: ScalarType,
        shape: &anndata::data::slice::Shape,
    ) -> Result<B::Dataset> {
        self.new_empty_dataset_typed_configured(name, dtype, shape, None, None, None)
    }

    fn new_empty_dataset_typed_configured(
        &self,
        name: &str,
        dtype: ScalarType,
        shape: &anndata::data::slice::Shape,
        chunk_size: Option<usize>,
        shard_size: Option<usize>,
        target_shard_bytes: Option<usize>,
    ) -> Result<B::Dataset> {
        let mut config = anndata::backend::get_default_write_config();
        let ndim = shape.ndim();

        let block = if let Some(cs) = chunk_size {
            let mut b: Vec<usize> = shape.as_ref().to_vec();
            b[0] = cs.min(b[0]);
            for dim in b.iter_mut().skip(1) {
                *dim = (*dim).min(if ndim == 1 { 16384 } else { 128 }).max(1);
            }
            Some(b)
        } else {
            None
        };

        if let Some(ref b) = block {
            config.block_size = Some(b.clone().into());
        }

        if target_shard_bytes.is_some() || shard_size.is_some() {
            let chunk_rows = block.as_ref()
                .map(|b| b[0])
                .unwrap_or_else(|| shape.as_ref()[0].min(if ndim == 1 { 16384 } else { 128 }).max(1));

            let shard_rows = if let Some(target_bytes) = target_shard_bytes {
                let elem_size = scalar_type_elem_size(dtype);
                let shard_cols: usize = block.as_ref()
                    .map(|b| b.iter().skip(1).product::<usize>().max(1))
                    .unwrap_or_else(|| {
                        shape.as_ref().iter().skip(1)
                            .map(|&x| x.min(if ndim == 1 { 16384 } else { 128 }).max(1))
                            .product::<usize>().max(1)
                    });
                let row_bytes = shard_cols * elem_size;
                let raw_rows = if row_bytes > 0 { target_bytes / row_bytes } else { chunk_rows };
                let raw_rows = raw_rows.max(chunk_rows);
                round_up_to(raw_rows, chunk_rows)
            } else {
                shard_size.unwrap()
            };

            let mut shard: Vec<usize> = block.as_ref()
                .cloned()
                .unwrap_or_else(|| {
                    if ndim == 1 {
                        vec![shape.as_ref()[0].min(16384).max(1)]
                    } else {
                        shape.as_ref().iter().map(|&x| x.min(128).max(1)).collect()
                    }
                });
            shard[0] = shard_rows;
            config.shard_size = Some(shard.into());
        }

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
