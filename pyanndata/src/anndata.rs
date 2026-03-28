mod backed;
mod dataset;
pub mod memory;

pub use backed::AnnData;
pub use dataset::AnnDataSet;
pub use memory::PyAnnData;

use anndata;
use anndata::concat::JoinType;
use anndata::Backend;
use anndata_hdf5::H5;
use anndata_zarr::Zarr;
use anyhow::Result;
use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use std::{
    collections::HashMap,
    ops::Deref,
    path::{Path, PathBuf},
};

pub(crate) fn get_backend<P: AsRef<Path>>(filename: P, backend: Option<&str>) -> &str {
    if let Some(backend) = backend {
        backend
    } else {
        if let Some(ext) = filename.as_ref().extension() {
            match ext.to_str().unwrap() {
                "h5ad" | "h5" | "h5ads" => H5::NAME,
                "zarr" | "zrad" => Zarr::NAME,
                _ => H5::NAME,
            }
        } else {
            H5::NAME
        }
    }
}

/// Read `.h5ad`-formatted hdf5 file.
///
/// Parameters
/// ----------
///
/// filename: Path
///     File name of data file.
/// backed: Literal['r', 'r+'] | None
///     Default is `r+`.
///     If `'r'`, the file is opened in read-only mode.
///     If `'r+'`, the file is opened in read/write mode.
///     If `None`, the AnnData object is read into memory.
/// backend: Literal['hdf5', 'zarr']
#[pyfunction]
#[pyo3(
    signature = (filename, backed="r+", backend=None),
    text_signature = "(filename, backed='r+', backend=None)",
)]
pub fn read<'py>(
    py: Python<'py>,
    filename: PathBuf,
    backed: Option<&str>,
    backend: Option<&str>,
) -> Result<Bound<'py, PyAny>> {
    let adata = match backed {
        Some(m) => {
            let backend = get_backend(&filename, backend);
            AnnData::new_from(filename, m, backend)
                .unwrap()
                .into_pyobject(py)?
                .into_any()
        }
        None => PyModule::import(py, "anndata")?
            .getattr("read_h5ad")?
            .call1((filename,))?
            .into_pyobject(py)?
            .into_any(),
    };
    Ok(adata)
}

/// Concatenates AnnData objects.
/// 
/// When `file` is provided, this function saves the merged AnnData object on disk
/// in a streaming fashion. This is memory efficient and allows merging large datasets
/// that do not fit into memory.
///
/// Parameters
/// ----------
///
/// adatas: list[AnnData]
///     List of AnnData objects to concatenate.
/// join: Literal['inner', 'outer']
///     How to handle observations and variables that are not shared between all AnnData objects.
/// label: str | None
///     Column in axis annotation (i.e. .obs or .var) to place batch information in. If it’s None, no column is added.
/// keys
///     Names for each object being added. These values are used for column values for label.
/// file: Path | None
///     If provided, the concatenated AnnData will be saved to this file.
/// backend: Literal['hdf5', 'zarr']
///     Backend to use for writing the output file.
/// 
/// Returns
/// -------
/// 
/// AnnData
///     The concatenated AnnData object.
/// 
/// See Also
/// --------
/// AnnDataSet
#[pyfunction]
#[pyo3(
    signature = (adatas, *, join="inner", label=None, keys=None, file=None, backend=None),
    text_signature = "(adatas, *, join='inner', label=None, keys=None, file=None, backend=None)",
)]
pub fn concat<'py>(
    py: Python<'py>,
    adatas: Vec<Py<PyAny>>,
    join: &str,
    label: Option<&str>,
    keys: Option<Vec<String>>,
    file: Option<PathBuf>,
    backend: Option<&str>,
) -> Result<Bound<'py, PyAny>> {
    let join = match join {
        "inner" => JoinType::Inner,
        "outer" => JoinType::Outer,
        _ => panic!("Unknown join type"),
    };

    enum T<'a> {
        H5(anndata::AnnData<H5>),
        Zarr(anndata::AnnData<Zarr>),
        Py(PyAnnData<'a>),
    }

    let keys = keys.as_ref().map(|x| x.as_slice());
    let out = if let Some(file) = file {
        let backend = get_backend(&file, backend);
        match backend {
            H5::NAME => {
                let adata = anndata::AnnData::<H5>::new(file)?;
                T::H5(adata)
            }
            Zarr::NAME => {
                let adata = anndata::AnnData::<Zarr>::new(file)?;
                T::Zarr(adata)
            }
            backend => todo!("Backend {} is not supported", backend),
        }
    } else {
        T::Py(PyAnnData::new(py)?)
    };

    if !adatas.is_empty() {
        if let Ok(adata) = adatas[0].extract::<AnnData>(py) {
            match adata.backend().as_str() {
                H5::NAME => {
                    let adatas = adatas
                        .into_iter()
                        .map(|x| x.extract::<AnnData>(py).unwrap())
                        .collect::<Vec<_>>();
                    let adatas: Vec<_> = adatas.iter().map(|x| x.inner_ref::<H5>()).collect();
                    let adatas: Vec<_> = adatas.iter().map(|x| x.deref()).collect();
                    match &out {
                        T::H5(out) => anndata::concat::concat(&adatas, join, label, keys, out)?,
                        T::Zarr(out) => anndata::concat::concat(&adatas, join, label, keys, out)?,
                        T::Py(out) => anndata::concat::concat(&adatas, join, label, keys, out)?,
                    }
                }
                Zarr::NAME => {
                    let adatas = adatas
                        .into_iter()
                        .map(|x| x.extract::<AnnData>(py).unwrap())
                        .collect::<Vec<_>>();
                    let adatas: Vec<_> = adatas.iter().map(|x| x.inner_ref::<Zarr>()).collect();
                    let adatas: Vec<_> = adatas.iter().map(|x| x.deref()).collect();
                    match &out {
                        T::H5(out) => anndata::concat::concat(&adatas, join, label, keys, out)?,
                        T::Zarr(out) => anndata::concat::concat(&adatas, join, label, keys, out)?,
                        T::Py(out) => anndata::concat::concat(&adatas, join, label, keys, out)?,
                    }
                }
                _ => todo!(),
            }
        } else {
            let adatas = adatas
                .into_iter()
                .map(|x| x.extract::<PyAnnData>(py).unwrap())
                .collect::<Vec<_>>();
            match &out {
                T::H5(out) => anndata::concat::concat(&adatas, join, label, keys, out)?,
                T::Zarr(out) => anndata::concat::concat(&adatas, join, label, keys, out)?,
                T::Py(out) => anndata::concat::concat(&adatas, join, label, keys, out)?,
            }
        }
    }

    match out {
        T::H5(adata) => Ok(AnnData::from(adata).into_pyobject(py)?.into_any()),
        T::Zarr(adata) => Ok(AnnData::from(adata).into_pyobject(py)?.into_any()),
        T::Py(adata) => Ok(adata.into_pyobject(py)?.into_any()),
    }
}

/// Read Matrix Market file.
///
/// Parameters
/// ----------
///
/// mtx_file
///     File name of the input matrix market file.
/// obs_names
///     File that stores the observation names.
/// var_names
///     File that stores the variable names.
/// file
///     File name of the output ".h5ad" file.
/// backend: Literal['hdf5', 'zarr']
///     Backend to use for writing the output file.
/// sorted
///     If true, the input matrix is assumed to be sorted by rows.
///     Sorted input matrix can be read faster.
#[pyfunction]
#[pyo3(
    signature = (mtx_file, *, obs_names=None, var_names=None, file=None, backend=None, sorted=false),
    text_signature = "(mtx_file, *, obs_names=None, var_names=None, file=None, backend=None, sorted=False)",
)]
pub fn read_mtx<'py>(
    py: Python<'py>,
    mtx_file: PathBuf,
    obs_names: Option<PathBuf>,
    var_names: Option<PathBuf>,
    file: Option<PathBuf>,
    backend: Option<&str>,
    sorted: bool,
) -> Result<Bound<'py, PyAny>> {
    let mut reader = anndata::reader::MMReader::from_path(mtx_file)?;
    if let Some(obs_names) = obs_names {
        reader = reader.obs_names(obs_names)?;
    }
    if let Some(var_names) = var_names {
        reader = reader.var_names(var_names)?;
    }
    if sorted {
        reader = reader.is_sorted();
    }
    if let Some(file) = file {
        let backend = get_backend(&file, backend);
        match backend {
            H5::NAME => {
                let adata = anndata::AnnData::<H5>::new(file)?;
                reader.finish(&adata)?;
                Ok(AnnData::from(adata).into_pyobject(py)?.into_any())
            }
            Zarr::NAME => {
                let adata = anndata::AnnData::<Zarr>::new(file)?;
                reader.finish(&adata)?;
                Ok(AnnData::from(adata).into_pyobject(py)?.into_any())
            }
            backend => todo!("Backend {} is not supported", backend),
        }
    } else {
        let adata = PyAnnData::new(py)?;
        reader.finish(&adata)?;
        Ok(adata.into_pyobject(py)?)
    }
}

/// Read AnnDataSet object.
///
/// Read AnnDataSet from .h5ads file. If the file paths stored in AnnDataSet
/// object are relative paths, it will look for component .h5ad files in .h5ads file's parent directory.
///
/// Parameters
/// ----------
/// filename: Path
///     File name.
/// adata_files_update: Mapping[str, Path] | Path | None
///     AnnDataSet internally stores links to component anndata files.
///     You can find this information in `.uns['AnnDataSet']`.
///     These links may be invalid if the anndata files are moved to a different location.
///     This parameter provides a way to update the locations of component anndata files.
///     The value of this parameter can be either a mapping from component anndata file names to their new locations,
///     or a directory containing component anndata files.
/// mode: str
///     "r": Read-only mode; "r+": can modify annotation file but not component anndata files.
/// backend: Literal['hdf5', 'zarr']
///     Backend to use for reading the annotation file.
///
/// Returns
/// -------
/// AnnDataSet
#[pyfunction]
#[pyo3(
    signature = (filename, *, adata_files_update=None, mode="r+", backend=None),
    text_signature = "(filename, *, adata_files_update=None, mode='r+', backend=None)",
)]
pub fn read_dataset(
    filename: PathBuf,
    adata_files_update: Option<LocationUpdate>,
    mode: &str,
    backend: Option<&str>,
) -> Result<AnnDataSet> {
    let adata_files_update = match adata_files_update {
        Some(LocationUpdate::Map(map)) => Some(Ok(map)),
        Some(LocationUpdate::Dir(dir)) => Some(Err(dir)),
        None => None,
    };
    let backend = get_backend(&filename, backend);
    match backend {
        H5::NAME => {
            let file = match mode {
                "r" => H5::open(filename)?,
                "r+" => H5::open_rw(filename)?,
                _ => panic!("Unknown mode"),
            };
            Ok(anndata::AnnDataSet::<H5>::open(file, adata_files_update)?.into())
        }
        Zarr::NAME => {
            let file = match mode {
                "r" => Zarr::open(filename)?,
                "r+" => Zarr::open_rw(filename)?,
                _ => panic!("Unknown mode"),
            };
            Ok(anndata::AnnDataSet::<Zarr>::open(file, adata_files_update)?.into())
        }
        _ => todo!(),
    }
}

#[derive(FromPyObject)]
pub enum LocationUpdate {
    Map(HashMap<String, PathBuf>),
    Dir(PathBuf),
}

/// Permute an AnnData Zarr store out-of-core.
///
/// Reorders observations (rows) of a .zarr AnnData according to a permutation
/// index, writing the result to a new .zarr store. Runs entirely out-of-core
/// with bounded memory usage -- the full dataset is never loaded into RAM.
///
/// Parameters
/// ----------
/// input : str | Path
///     Path to the source .zarr AnnData store.
/// output : str | Path
///     Path for the destination .zarr store (will be created).
/// permutation : numpy.ndarray[int64]
///     1-D array where ``permutation[i]`` is the source row index for output
///     row ``i``. Length determines the number of output rows (can be a subset).
/// memory_limit : int, optional
///     Maximum RAM in bytes for internal buffers. Default is 2 GB.
/// chunk_size : int, optional
///     Number of rows per output sub-chunk along axis 0. When None (default),
///     the backend picks ``min(n_rows, 128)`` for 2-D arrays.
/// shard_size : int, optional
///     Number of rows per output shard (outer chunk) along axis 0. Must be a
///     multiple of ``chunk_size``. When None, defaults to ``chunk_size * 8``.
///     Ignored when ``target_shard_bytes`` is set.
/// target_shard_bytes : int, optional
///     Target shard size in bytes. The engine auto-calculates the shard row
///     count so each shard is approximately this many bytes. Set to the Lustre
///     stripe size (e.g. ``4 * 1024 * 1024`` for 4 MB) for optimal I/O
///     alignment. Overrides ``shard_size`` when set.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> import anndata_rs
/// >>> perm = np.random.permutation(adata.n_obs).astype(np.int64)
/// >>> anndata_rs.permute("input.zarr", "output.zarr", perm)
/// >>> anndata_rs.permute("in.zarr", "out.zarr", perm, chunk_size=256)
/// >>> anndata_rs.permute("in.zarr", "out.zarr", perm, shard_size=2048)
/// >>> anndata_rs.permute("in.zarr", "out.zarr", perm, target_shard_bytes=4*1024*1024)
#[pyfunction]
#[pyo3(
    signature = (input, output, permutation, *, memory_limit=None, chunk_size=None, shard_size=None, target_shard_bytes=None),
    text_signature = "(input, output, permutation, *, memory_limit=None, chunk_size=None, shard_size=None, target_shard_bytes=None)",
)]
pub fn permute(
    input: PathBuf,
    output: PathBuf,
    permutation: PyReadonlyArray1<i64>,
    memory_limit: Option<usize>,
    chunk_size: Option<usize>,
    shard_size: Option<usize>,
    target_shard_bytes: Option<usize>,
) -> Result<()> {
    let perm: Vec<usize> = permutation
        .as_array()
        .iter()
        .map(|&v| v as usize)
        .collect();

    let config = anndata_ooc::PermuteConfig {
        memory_limit: memory_limit.unwrap_or(2 * 1024 * 1024 * 1024),
        chunk_size,
        shard_size,
        target_shard_bytes,
    };

    anndata_ooc::permute_anndata(&input, &output, &perm, &config)
}

/// Split an AnnData Zarr store by the values of an obs column.
///
/// Each unique value in the specified observation column becomes a separate
/// output .zarr store under ``output_dir``. The split is performed out-of-core:
/// the source data matrix is read once and scattered to all outputs in a single
/// pass (per memory-budget batch).
///
/// Parameters
/// ----------
/// input : str | Path
///     Path to the source .zarr AnnData store.
/// output_dir : str | Path
///     Directory where output stores will be created.
///     Each store is named ``{value}.zarr``.
/// column : str
///     Name of the obs column to split by.
/// memory_limit : int, optional
///     Maximum RAM in bytes for internal buffers. Default is 2 GB.
/// chunk_size : int, optional
///     Number of rows per output sub-chunk along axis 0.
/// shard_size : int, optional
///     Number of rows per output shard along axis 0.
/// target_shard_bytes : int, optional
///     Target shard size in bytes. Overrides ``shard_size`` when set.
///
/// Returns
/// -------
/// list of (str, str)
///     List of (column_value, output_path) pairs for each group.
///
/// Examples
/// --------
/// >>> import anndata_rs
/// >>> groups = anndata_rs.split("input.zarr", "splits/", "cell_type")
/// >>> for value, path in groups:
/// ...     print(f"{value} -> {path}")
#[pyfunction]
#[pyo3(
    signature = (input, output_dir, column, *, memory_limit=None, chunk_size=None, shard_size=None, target_shard_bytes=None),
    text_signature = "(input, output_dir, column, *, memory_limit=None, chunk_size=None, shard_size=None, target_shard_bytes=None)",
)]
pub fn split(
    input: PathBuf,
    output_dir: PathBuf,
    column: String,
    memory_limit: Option<usize>,
    chunk_size: Option<usize>,
    shard_size: Option<usize>,
    target_shard_bytes: Option<usize>,
) -> Result<Vec<(String, String)>> {
    let config = anndata_ooc::ScatterConfig {
        memory_limit: memory_limit.unwrap_or(2 * 1024 * 1024 * 1024),
        chunk_size,
        shard_size,
        target_shard_bytes,
    };

    let results = anndata_ooc::split_anndata(&input, &output_dir, &column, &config)?;

    Ok(results.into_iter().map(|(val, path)| {
        (val, path.display().to_string())
    }).collect())
}
