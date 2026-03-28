use std::path::Path;

use anyhow::{Result, Context};
use ndarray::{Array1, ArrayD, Ix1};

use anndata::backend::{Backend, DataType, GroupOp, ScalarType, StoreOp, DatasetOp, AttributeOp};
use anndata::data::slice::SelectInfoElem;
use anndata_zarr::Zarr;

use crate::budget::{BufferPool, MemoryBudget};
use crate::dense::DensePermuter;
use crate::sparse::SparsePermuter;

/// Configuration for out-of-core permutation.
pub struct PermuteConfig {
    /// Maximum RAM (in bytes) the engine may use for buffers.
    pub memory_limit: usize,
    /// Sub-chunk size (rows per sub-chunk) for output arrays along axis 0.
    /// When None, the backend default is used (min(n, 128) for 2-D numeric).
    pub chunk_size: Option<usize>,
    /// Shard size (rows per shard) for output arrays along axis 0.
    /// Must be a multiple of chunk_size. When None, defaults to chunk_size * 8.
    /// Ignored when target_shard_bytes is set.
    pub shard_size: Option<usize>,
    /// Target shard size in bytes. When set, the engine auto-calculates the
    /// shard row count so that each shard is approximately this many bytes.
    /// This is the most Lustre-friendly option: set it to the stripe size
    /// (e.g. 4 * 1024 * 1024 for 4 MB stripes). Overrides shard_size.
    pub target_shard_bytes: Option<usize>,
}

impl Default for PermuteConfig {
    fn default() -> Self {
        Self {
            memory_limit: 2 * 1024 * 1024 * 1024, // 2 GB
            chunk_size: None,
            shard_size: None,
            target_shard_bytes: None,
        }
    }
}

/// Permute an entire AnnData Zarr store out-of-core.
///
/// Given `src_path` (an existing .zarr AnnData), `dst_path` (where the output
/// is written), and `permutation` (output_row -> source_row), this function:
///
/// 1. Copies `var` metadata unchanged (column annotations are row-independent)
/// 2. Permutes `obs` (row annotations) by the given index
/// 3. Permutes `X` (the main data matrix -- dense or CSR/CSC)
/// 4. Permutes every array in `obsm`, `obsp`, `layers` along axis 0
/// 5. Copies `varm`, `varp`, `uns` unchanged
pub fn permute_anndata(
    src_path: &Path,
    dst_path: &Path,
    permutation: &[usize],
    config: &PermuteConfig,
) -> Result<()> {
    let src_store = Zarr::open(src_path)
        .with_context(|| format!("failed to open source: {}", src_path.display()))?;
    let dst_store = Zarr::new(dst_path)
        .with_context(|| format!("failed to create destination: {}", dst_path.display()))?;

    let budget = MemoryBudget::new(config.memory_limit);
    let pool = BufferPool::new(budget);

    let src_items = src_store.list()?;

    // Copy var (columns) unchanged
    if src_items.contains(&"var".to_string()) {
        copy_group(&src_store, &dst_store, "var")?;
    }

    // Copy uns unchanged
    if src_items.contains(&"uns".to_string()) {
        copy_group(&src_store, &dst_store, "uns")?;
    }

    // Copy varm, varp unchanged
    for name in &["varm", "varp"] {
        if src_items.contains(&name.to_string()) {
            copy_group(&src_store, &dst_store, name)?;
        }
    }

    // Permute obs
    if src_items.contains(&"obs".to_string()) {
        permute_dataframe_group(&src_store, &dst_store, "obs", permutation)?;
    }

    // Permute X
    if src_items.contains(&"X".to_string()) {
        permute_matrix_element(&src_store, &dst_store, "X", permutation, &pool, config)?;
    }

    // Permute obsm, obsp, layers (each is a group of arrays)
    for group_name in &["obsm", "obsp", "layers"] {
        if src_items.contains(&group_name.to_string()) {
            let src_group = src_store.open_group(group_name)?;
            let dst_group = dst_store.new_group(group_name)?;

            let children = src_group.list()?;
            for child in &children {
                permute_matrix_element_in_group(
                    &src_group, &dst_group, child, permutation, &pool, config,
                )?;
            }
        }
    }

    src_store.close()?;
    dst_store.close()?;

    log::info!("Permutation complete: {} -> {}", src_path.display(), dst_path.display());
    Ok(())
}

fn permute_matrix_element<G: GroupOp<Zarr>>(
    src_store: &G,
    dst_store: &G,
    name: &str,
    permutation: &[usize],
    pool: &BufferPool,
    config: &PermuteConfig,
) -> Result<()> {
    permute_matrix_element_in_group(src_store, dst_store, name, permutation, pool, config)
}

fn permute_matrix_element_in_group<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_group: &G,
    name: &str,
    permutation: &[usize],
    pool: &BufferPool,
    config: &PermuteConfig,
) -> Result<()> {
    use anndata::backend::DataContainer;

    let container = DataContainer::<Zarr>::open(src_group, name)?;
    let encoding = container.encoding_type()?;

    match encoding {
        DataType::Array(scalar_type) | DataType::Scalar(scalar_type) => {
            let src_ds = src_group.open_dataset(name)?;
            let shape = src_ds.shape();
            if shape.ndim() < 2 {
                permute_1d_dataset(src_group, dst_group, name, permutation)?;
            } else {
                permute_dense_dataset(
                    src_group, dst_group, name, permutation, scalar_type, pool, config,
                )?;
            }
        }
        DataType::CsrMatrix(scalar_type) => {
            permute_csr_group(src_group, dst_group, name, permutation, scalar_type, pool, config)?;
        }
        DataType::CscMatrix(_) => {
            log::warn!("CSC matrix '{}': falling back to in-memory permutation", name);
            permute_via_anndata_select(src_group, dst_group, name, permutation)?;
        }
        other => {
            log::warn!("Unsupported encoding {:?} for '{}', copying unchanged", other, name);
            copy_group_child(src_group, dst_group, name)?;
        }
    }

    Ok(())
}

fn permute_dense_dataset<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_group: &G,
    name: &str,
    permutation: &[usize],
    scalar_type: ScalarType,
    pool: &BufferPool,
    config: &PermuteConfig,
) -> Result<()> {
    let src_ds = src_group.open_dataset(name)?;
    let src_shape = src_ds.shape();
    let n_output = permutation.len();
    let _n_cols = if src_shape.ndim() >= 2 { src_shape[1] } else { 1 };

    let mut out_shape = src_shape.as_ref().to_vec();
    out_shape[0] = n_output;

    let mut dst_ds = dst_group.new_empty_dataset_typed_configured(
        name, scalar_type, &out_shape.into(),
        config.chunk_size, config.shard_size, config.target_shard_bytes,
    )?;
    copy_encoding_attrs(&src_ds, &mut dst_ds)?;

    let permuter = DensePermuter::new(pool.clone_with_same_budget());
    permuter.permute(src_ds.inner(), dst_ds.inner(), permutation)
}

fn permute_1d_dataset<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_group: &G,
    name: &str,
    permutation: &[usize],
) -> Result<()> {
    let src_ds = src_group.open_dataset(name)?;
    let dtype = src_ds.dtype()?;
    let src_shape = src_ds.shape();
    let n = src_shape[0];

    let mut out_shape = src_shape.as_ref().to_vec();
    out_shape[0] = permutation.len();
    let mut dst_ds = dst_group.new_empty_dataset_typed(name, dtype, &out_shape.into())?;
    copy_encoding_attrs(&src_ds, &mut dst_ds)?;

    let full = read_dyn_slice(&src_ds, &[SelectInfoElem::from(0..n)], dtype)?;
    let permuted = permute_dyn_array_1d(&full, permutation);
    let write_sel = vec![SelectInfoElem::from(0..permutation.len())];
    write_dyn_slice(&dst_ds, &permuted, &write_sel, dtype)?;

    Ok(())
}

fn permute_csr_group<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_group: &G,
    name: &str,
    permutation: &[usize],
    scalar_type: ScalarType,
    pool: &BufferPool,
    _config: &PermuteConfig,
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

    let n_output = permutation.len();

    // Build output indptr
    let mut out_indptr = vec![0i64; n_output + 1];
    for (out_row, &src_row) in permutation.iter().enumerate() {
        let row_nnz = src_indptr[src_row + 1] - src_indptr[src_row];
        out_indptr[out_row + 1] = out_indptr[out_row] + row_nnz;
    }
    let total_nnz = *out_indptr.last().unwrap() as usize;

    log::info!("CSR '{}': {}x{}, {} output rows, nnz={}", name, n_rows, n_cols, n_output, total_nnz);

    // Create output group
    let mut dst_g = dst_group.new_group(name)?;
    dst_g.new_attr("encoding-type", "csr_matrix")?;
    dst_g.new_attr("encoding-version", "0.1.0")?;
    dst_g.new_attr("h5sparse_format", "csr")?;
    dst_g.new_attr("shape", [n_output as u64, n_cols as u64].as_slice())?;

    // Write output indptr
    let indptr_arr: ArrayD<i64> = Array1::from_vec(out_indptr.clone()).into_dyn();
    dst_g.new_array_dataset(
        "indptr",
        indptr_arr.view().into(),
        anndata::backend::get_default_write_config(),
    )?;

    // Create output data/indices datasets
    let src_data_ds = src_g.open_dataset("data")?;
    let src_indices_ds = src_g.open_dataset("indices")?;
    let indices_dtype = src_indices_ds.dtype()?;

    let dst_data_ds = dst_g.new_empty_dataset_typed(
        "data", scalar_type, &vec![total_nnz].into(),
    )?;
    let dst_indices_ds = dst_g.new_empty_dataset_typed(
        "indices", indices_dtype, &vec![total_nnz].into(),
    )?;

    // Use SparsePermuter with direct zarrs array access for efficient I/O
    let permuter = SparsePermuter::new(pool.clone_with_same_budget());
    permuter.permute_data_indices(
        src_indices_ds.inner(),
        src_data_ds.inner(),
        dst_indices_ds.inner(),
        dst_data_ds.inner(),
        permutation,
        &src_indptr,
        &out_indptr,
    )?;

    Ok(())
}

fn permute_dataframe_group<G: GroupOp<Zarr>>(
    src_store: &G,
    dst_store: &G,
    name: &str,
    permutation: &[usize],
) -> Result<()> {
    let src_g = src_store.open_group(name)?;
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

    let children = src_g.list()?;
    for col_name in &children {
        permute_1d_column(&src_g, &dst_g, col_name, permutation)?;
    }

    Ok(())
}

fn permute_1d_column<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_group: &G,
    name: &str,
    permutation: &[usize],
) -> Result<()> {
    let src_ds = src_group.open_dataset(name)?;
    let dtype = src_ds.dtype()?;
    let n = src_ds.shape()[0];
    let n_output = permutation.len();

    let mut dst_ds = dst_group.new_empty_dataset_typed(
        name, dtype, &vec![n_output].into(),
    )?;

    if let Ok(enc) = src_ds.get_json_attr("encoding-type") {
        dst_ds.new_json_attr("encoding-type", &enc)?;
    }
    if let Ok(enc) = src_ds.get_json_attr("encoding-version") {
        dst_ds.new_json_attr("encoding-version", &enc)?;
    }

    // Read, permute in memory, write in one go
    let full = read_dyn_slice(&src_ds, &[SelectInfoElem::from(0..n)], dtype)?;
    let permuted = permute_dyn_array_1d(&full, permutation);
    let write_sel = vec![SelectInfoElem::from(0..n_output)];
    write_dyn_slice(&dst_ds, &permuted, &write_sel, dtype)?;

    Ok(())
}

// --- helpers ---

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

use anndata::data::array::DynArray;

fn read_dyn_slice(
    ds: &<Zarr as Backend>::Dataset,
    sel: &[SelectInfoElem],
    _dtype: ScalarType,
) -> Result<DynArray> {
    ds.read_dyn_array_slice(sel)
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

/// Permute a 1D DynArray in memory. Much faster than row-by-row writes.
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

fn copy_group_child<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_group: &G,
    name: &str,
) -> Result<()> {
    copy_group(src_group, dst_group, name)
}

fn permute_via_anndata_select<G: GroupOp<Zarr>>(
    src_group: &G,
    dst_group: &G,
    name: &str,
    permutation: &[usize],
) -> Result<()> {
    use anndata::data::{ArrayData, data_traits::{ReadableArray, Writable}};
    use anndata::backend::DataContainer;

    let container = DataContainer::<Zarr>::open(src_group, name)?;
    let sel = vec![
        SelectInfoElem::Index(permutation.to_vec()),
        SelectInfoElem::full(),
    ];
    let data = ArrayData::read_select(&container, &sel)?;
    data.write(dst_group, name)?;
    Ok(())
}

/// Extension trait to create typed empty datasets.
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

        // Resolve shard shape: target_shard_bytes takes priority over shard_size.
        if target_shard_bytes.is_some() || shard_size.is_some() {
            let chunk_rows = block.as_ref()
                .map(|b| b[0])
                .unwrap_or_else(|| shape.as_ref()[0].min(if ndim == 1 { 16384 } else { 128 }).max(1));

            let shard_rows = if let Some(target_bytes) = target_shard_bytes {
                let elem_size = scalar_type_elem_size(dtype);
                // Use the shard column count (= sub-chunk columns), not full array width,
                // since each shard only spans one sub-chunk in non-row dimensions.
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
        ScalarType::String => 8, // conservative estimate for variable-length
    }
}

fn round_up_to(value: usize, multiple: usize) -> usize {
    if multiple == 0 { return value; }
    ((value + multiple - 1) / multiple) * multiple
}
