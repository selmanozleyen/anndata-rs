use anndata::{
    backend::*,
    data::{DynArray, DynCowArray, SelectInfoBounds, SelectInfoElem, SelectInfoElemBounds, Shape},
};

use anyhow::{bail, Context, Result};
use ndarray::{Array, ArrayD, ArrayView, CowArray, Dimension, IxDyn, SliceInfoElem};
use std::{
    borrow::Cow,
    ops::{Deref, Index},
    path::{Path, PathBuf},
};
use std::{sync::Arc, vec};
use zarrs::array::codec::bytes_to_bytes::zstd::ZstdCodec;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;
use zarrs::storage::ReadableWritableListableStorageTraits;
use zarrs::{
    array::{
        data_type::{self, *},
        ArrayShardedReadableExt, CodecOptions, Element, ElementOwned,
    },
    storage::StorePrefix,
};

pub struct Zarr;

#[derive(Clone)]
pub struct ZarrStore {
    inner: Arc<dyn ReadableWritableListableStorageTraits>,
    path: PathBuf,
}

impl Deref for ZarrStore {
    type Target = Arc<dyn ReadableWritableListableStorageTraits>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct ZarrGroup {
    group: Group<dyn ReadableWritableListableStorageTraits>,
    store: ZarrStore,
}

pub struct ZarrDataset {
    dataset: zarrs::array::Array<dyn ReadableWritableListableStorageTraits>,
    cache: zarrs::array::ArrayShardedReadableExtCache,
    store: ZarrStore,
}

impl ZarrDataset {
    pub fn inner(&self) -> &zarrs::array::Array<dyn ReadableWritableListableStorageTraits> {
        &self.dataset
    }
}

impl Backend for Zarr {
    const NAME: &'static str = "zarr";

    type Store = ZarrStore;

    type Group = ZarrGroup;

    type Dataset = ZarrDataset;

    fn new<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        if path.as_ref().try_exists()? {
            let metadata = std::fs::metadata(&path)?;
            if metadata.is_file() {
                std::fs::remove_file(&path)?;
            } else {
                std::fs::remove_dir_all(&path)?;
            }
        }

        let inner = Arc::new(FilesystemStore::new(path.as_ref())?);
        zarrs::group::GroupBuilder::new()
            .build(inner.clone(), "/")?
            .store_metadata()?;
        Ok(ZarrStore {
            path: path.as_ref().to_path_buf(),
            inner,
        })
    }

    fn open<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        Ok(ZarrStore {
            path: path.as_ref().to_path_buf(),
            inner: Arc::new(FilesystemStore::new(path)?),
        })
    }

    fn open_rw<P: AsRef<Path>>(path: P) -> Result<Self::Store> {
        Ok(ZarrStore {
            path: path.as_ref().to_path_buf(),
            inner: Arc::new(FilesystemStore::new(path)?),
        })
    }
}

impl StoreOp<Zarr> for ZarrStore {
    fn filename(&self) -> PathBuf {
        self.path.clone()
    }

    fn close(self) -> Result<()> {
        drop(self);
        Ok(())
    }
}

impl GroupOp<Zarr> for ZarrStore {
    fn list(&self) -> Result<Vec<String>> {
        let result = self.list_dir(&StorePrefix::root())?;
        Ok(result
            .prefixes()
            .into_iter()
            .map(|x| x.as_str().trim_end_matches("/").to_string())
            .collect())
    }

    fn new_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let path = canonicalize_path(name);
        let group = zarrs::group::GroupBuilder::new().build(self.inner.clone(), &path)?;
        group.store_metadata()?;
        Ok(ZarrGroup {
            group,
            store: self.clone(),
        })
    }

    fn open_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let group = zarrs::group::Group::open(self.inner.clone(), &canonicalize_path(name))?;
        Ok(ZarrGroup {
            group,
            store: self.clone(),
        })
    }

    fn new_empty_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<Zarr as Backend>::Dataset> {
        let path = canonicalize_path(name);
        let array = new_empty_dataset_helper::<T, _>(self.inner.clone(), &path, shape, config)?;
        array.store_metadata()?;
        let cache = zarrs::array::ArrayShardedReadableExtCache::new(&array);
        Ok(ZarrDataset {
            dataset: array,
            cache,
            store: self.clone(),
        })
    }

    fn open_dataset(&self, name: &str) -> Result<<Zarr as Backend>::Dataset> {
        let array = zarrs::array::Array::open(self.inner.clone(), &canonicalize_path(name))?;
        let cache = zarrs::array::ArrayShardedReadableExtCache::new(&array);
        Ok(ZarrDataset {
            dataset: array,
            cache,
            store: self.clone(),
        })
    }

    fn delete(&self, name: &str) -> Result<()> {
        self.inner.erase_prefix(&str_to_prefix(name))?;
        Ok(())
    }

    fn exists(&self, name: &str) -> Result<bool> {
        let path = format!("/{}", name);
        Ok(zarrs::node::node_exists(
            &self.inner,
            &path.as_str().try_into()?,
        )?)
    }
}

impl GroupOp<Zarr> for ZarrGroup {
    fn list(&self) -> Result<Vec<String>> {
        let current_path = str_to_prefix(self.group.path().as_str());
        let result = self
            .store
            .list_dir(&current_path.as_str().try_into()?)?
            .prefixes()
            .into_iter()
            .map(|x| {
                x.as_str()
                    .strip_prefix(current_path.as_str())
                    .unwrap()
                    .strip_suffix("/")
                    .unwrap()
                    .to_owned()
            })
            .collect();
        Ok(result)
    }

    fn new_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let path = self.group.path().as_path().join(name);
        let group = zarrs::group::GroupBuilder::new()
            .build(self.store.inner.clone(), path.to_str().unwrap())?;
        group.store_metadata()?;
        Ok(ZarrGroup {
            group,
            store: self.store.clone(),
        })
    }

    fn open_group(&self, name: &str) -> Result<<Zarr as Backend>::Group> {
        let path = self.group.path().as_path().join(name);
        let group = zarrs::group::Group::open(self.store.inner.clone(), path.to_str().unwrap())?;
        Ok(ZarrGroup {
            group,
            store: self.store.clone(),
        })
    }

    fn new_empty_dataset<T: BackendData>(
        &self,
        name: &str,
        shape: &Shape,
        config: WriteConfig,
    ) -> Result<<Zarr as Backend>::Dataset> {
        let path = self.group.path().as_path().join(name);
        let array = new_empty_dataset_helper::<T, _>(
            self.store.inner.clone(),
            path.to_str().unwrap(),
            shape,
            config,
        )?;
        array.store_metadata()?;
        let cache = zarrs::array::ArrayShardedReadableExtCache::new(&array);
        Ok(ZarrDataset {
            dataset: array,
            cache,
            store: self.store.clone(),
        })
    }

    fn open_dataset(&self, name: &str) -> Result<<Zarr as Backend>::Dataset> {
        let path = self.group.path().as_path().join(name);
        let array = zarrs::array::Array::open(self.store.inner.clone(), path.to_str().unwrap())?;
        let cache = zarrs::array::ArrayShardedReadableExtCache::new(&array);
        Ok(ZarrDataset {
            dataset: array,
            cache,
            store: self.store.clone(),
        })
    }

    fn delete(&self, name: &str) -> Result<()> {
        let path = format!("{}/{}", self.group.path().as_str(), name);
        self.store.erase_prefix(&str_to_prefix(&path))?;
        Ok(())
    }

    fn exists(&self, name: &str) -> Result<bool> {
        let path = self
            .group
            .path()
            .as_path()
            .join(name)
            .as_os_str()
            .to_str()
            .unwrap()
            .try_into()?;
        Ok(zarrs::node::node_exists(&self.store.inner, &path)?)
    }
}

impl AttributeOp<Zarr> for ZarrGroup {
    fn store(&self) -> Result<<Zarr as Backend>::Store> {
        Ok(self.store.clone())
    }

    fn path(&self) -> PathBuf {
        self.group.path().as_path().to_path_buf()
    }

    fn new_json_attr(&mut self, name: &str, value: &Value) -> Result<()> {
        self.group
            .attributes_mut()
            .insert(name.to_string(), value.clone());
        self.group.store_metadata()?;
        Ok(())
    }

    fn get_json_attr(&self, name: &str) -> Result<Value> {
        Ok(self
            .group
            .attributes()
            .get(name)
            .with_context(|| format!("Attribute {} not found", name))?
            .clone())
    }
}

impl AttributeOp<Zarr> for ZarrDataset {
    fn store(&self) -> Result<<Zarr as Backend>::Store> {
        Ok(self.store.clone())
    }

    fn path(&self) -> PathBuf {
        self.dataset.path().as_path().to_path_buf()
    }

    fn new_json_attr(&mut self, name: &str, value: &Value) -> Result<()> {
        self.dataset
            .attributes_mut()
            .insert(name.to_string(), value.clone());
        self.dataset.store_metadata()?;
        Ok(())
    }

    fn get_json_attr(&self, name: &str) -> Result<Value> {
        Ok(self
            .dataset
            .attributes()
            .get(name)
            .with_context(|| format!("Attribute {} not found", name))?
            .clone())
    }
}

impl DatasetOp<Zarr> for ZarrDataset {
    fn dtype(&self) -> Result<ScalarType> {
        let dt = self.dataset.data_type();
        if dt.is::<UInt8DataType>() {
            Ok(ScalarType::U8)
        } else if dt.is::<UInt16DataType>() {
            Ok(ScalarType::U16)
        } else if dt.is::<UInt32DataType>() {
            Ok(ScalarType::U32)
        } else if dt.is::<UInt64DataType>() {
            Ok(ScalarType::U64)
        } else if dt.is::<Int8DataType>() {
            Ok(ScalarType::I8)
        } else if dt.is::<Int16DataType>() {
            Ok(ScalarType::I16)
        } else if dt.is::<Int32DataType>() {
            Ok(ScalarType::I32)
        } else if dt.is::<Int64DataType>() {
            Ok(ScalarType::I64)
        } else if dt.is::<Float32DataType>() {
            Ok(ScalarType::F32)
        } else if dt.is::<Float64DataType>() {
            Ok(ScalarType::F64)
        } else if dt.is::<BoolDataType>() {
            Ok(ScalarType::Bool)
        } else if dt.is::<StringDataType>() {
            Ok(ScalarType::String)
        } else {
            bail!("Unsupported data type: {:?}", dt)
        }
    }

    fn shape(&self) -> Shape {
        self.dataset
            .shape()
            .into_iter()
            .map(|x| *x as usize)
            .collect()
    }

    fn reshape(&mut self, shape: &Shape) -> Result<()> {
        self.dataset
            .set_shape(shape.as_ref().iter().map(|x| *x as u64).collect())?;
        self.dataset.store_metadata()?;
        Ok(())
    }

    fn read_array_slice<T: BackendData, S, D>(&self, selection: &[S]) -> Result<Array<T, D>>
    where
        S: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        fn read_arr<T, S, D>(dataset: &ZarrDataset, selection: &[S]) -> Result<Array<T, D>>
        where
            T: ElementOwned + BackendData,
            S: AsRef<SelectInfoElem>,
            D: Dimension,
        {
            let sel = SelectInfoBounds::new(&selection, &dataset.shape());
            if let Some(subset) = to_array_subset(sel) {
                let arr: ndarray::ArrayD<T> = dataset
                    .dataset
                    .retrieve_array_subset_sharded_opt(
                        &dataset.cache,
                        &subset,
                        &CodecOptions::default(),
                    )?;
                Ok(arr.into_dimensionality::<D>()?)
            } else {
                let arr: ndarray::ArrayD<T> = dataset
                    .dataset
                    .retrieve_array_subset(&dataset.dataset.subset_all())?;
                let selected = select(arr.view(), selection);
                Ok(selected.into_dimensionality::<D>()?)
            }
        }

        let array: DynArray = match T::DTYPE {
            ScalarType::U8 => read_arr::<u8, _, D>(self, selection)?.into(),
            ScalarType::U16 => read_arr::<u16, _, D>(self, selection)?.into(),
            ScalarType::U32 => read_arr::<u32, _, D>(self, selection)?.into(),
            ScalarType::U64 => read_arr::<u64, _, D>(self, selection)?.into(),
            ScalarType::I8 => read_arr::<i8, _, D>(self, selection)?.into(),
            ScalarType::I16 => read_arr::<i16, _, D>(self, selection)?.into(),
            ScalarType::I32 => read_arr::<i32, _, D>(self, selection)?.into(),
            ScalarType::I64 => read_arr::<i64, _, D>(self, selection)?.into(),
            ScalarType::F32 => read_arr::<f32, _, D>(self, selection)?.into(),
            ScalarType::F64 => read_arr::<f64, _, D>(self, selection)?.into(),
            ScalarType::Bool => read_arr::<bool, _, D>(self, selection)?.into(),
            ScalarType::String => read_arr::<String, _, D>(self, selection)?.into(),
        };
        Ok(BackendData::from_dyn_arr(array)?.into_dimensionality::<D>()?)
    }

    fn write_array_slice<S, T, D>(&self, arr: CowArray<'_, T, D>, selection: &[S]) -> Result<()>
    where
        T: BackendData,
        S: AsRef<SelectInfoElem>,
        D: Dimension,
    {
        fn write_array_impl<T, S>(
            container: &ZarrDataset,
            arr: CowArray<'_, T, IxDyn>,
            selection: &[S],
        ) -> Result<()>
        where
            T: Element + 'static,
            S: AsRef<SelectInfoElem>,
        {
            let selection = SelectInfoBounds::new(&selection, &container.shape());
            let starts: Vec<_> = selection
                .iter()
                .flat_map(|x| {
                    if let SelectInfoElemBounds::Slice(slice) = x {
                        if slice.step == 1 {
                            Some(slice.start as u64)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
            if starts.len() == selection.ndim() {
                let owned = arr.into_owned();
                let shape: Vec<u64> = owned.shape().iter().map(|&s| s as u64).collect();
                let ranges: Vec<_> = starts.iter().zip(&shape)
                    .map(|(&start, &len)| start..start + len)
                    .collect();
                let subset = zarrs::array::ArraySubset::new_with_ranges(&ranges);
                container
                    .dataset
                    .store_array_subset(&subset, owned)?;
                container.cache.clear();
            } else {
                panic!("Not implemented");
            }
            Ok(())
        }

        match BackendData::into_dyn_arr(arr.into_dyn()) {
            DynCowArray::U8(x) => write_array_impl(self, x, selection),
            DynCowArray::U16(x) => write_array_impl(self, x, selection),
            DynCowArray::U32(x) => write_array_impl(self, x, selection),
            DynCowArray::U64(x) => write_array_impl(self, x, selection),
            DynCowArray::I8(x) => write_array_impl(self, x, selection),
            DynCowArray::I16(x) => write_array_impl(self, x, selection),
            DynCowArray::I32(x) => write_array_impl(self, x, selection),
            DynCowArray::I64(x) => write_array_impl(self, x, selection),
            DynCowArray::F32(x) => write_array_impl(self, x, selection),
            DynCowArray::F64(x) => write_array_impl(self, x, selection),
            DynCowArray::Bool(x) => write_array_impl(self, x, selection),
            DynCowArray::String(x) => write_array_impl(self, x, selection),
        }
    }
}

fn select<S, T>(arr: ArrayView<'_, T, IxDyn>, info: &[S]) -> ArrayD<T>
where
    S: AsRef<SelectInfoElem>,
    T: Clone,
{
    let arr = arr.into_dyn();
    let slices = info
        .as_ref()
        .into_iter()
        .map(|x| match x.as_ref() {
            SelectInfoElem::Slice(slice) => Some(SliceInfoElem::from(slice.clone())),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    if let Some(slices) = slices {
        arr.slice(slices.as_slice()).into_owned()
    } else {
        let shape = arr.shape();
        let select: Vec<_> = info
            .as_ref()
            .into_iter()
            .zip(shape)
            .map(|(x, n)| SelectInfoElemBounds::new(x.as_ref(), *n))
            .collect();
        let new_shape = select.iter().map(|x| x.len()).collect::<Vec<_>>();
        ArrayD::from_shape_fn(new_shape, |idx| {
            let new_idx: Vec<_> = (0..idx.ndim())
                .into_iter()
                .map(|i| select[i].index(idx[i]))
                .collect();
            arr.index(new_idx.as_slice()).clone()
        })
    }
}

fn str_to_prefix(s: &str) -> StorePrefix {
    if s.is_empty() {
        StorePrefix::root()
    } else {
        let s = s.trim_matches('/').to_string();
        StorePrefix::new((s + "/").as_str()).unwrap()
    }
}

fn canonicalize_path<'a>(path: &'a str) -> Cow<'a, str> {
    if path.starts_with("/") {
        path.into()
    } else {
        format!("/{}", path).into()
    }
}

fn to_array_subset(info: SelectInfoBounds) -> Option<zarrs::array::ArraySubset> {
    let ranges = info
        .iter()
        .map(|x| {
            if let SelectInfoElemBounds::Slice(slice) = x {
                if slice.step == 1 {
                    Some(slice.start as u64..slice.end as u64)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect::<Option<Vec<_>>>()?;
    Some(zarrs::array::ArraySubset::new_with_ranges(&ranges))
}

fn scalar_type_to_zarr(
    dtype: ScalarType,
) -> (zarrs::array::DataType, zarrs::array::FillValue) {
    match dtype {
        ScalarType::U8 => (data_type::uint8(), 0u8.into()),
        ScalarType::U16 => (data_type::uint16(), 0u16.into()),
        ScalarType::U32 => (data_type::uint32(), 0u32.into()),
        ScalarType::U64 => (data_type::uint64(), 0u64.into()),
        ScalarType::I8 => (data_type::int8(), 0i8.into()),
        ScalarType::I16 => (data_type::int16(), 0i16.into()),
        ScalarType::I32 => (data_type::int32(), 0i32.into()),
        ScalarType::I64 => (data_type::int64(), 0i64.into()),
        ScalarType::F32 => (data_type::float32(), f32::NAN.into()),
        ScalarType::F64 => (data_type::float64(), f64::NAN.into()),
        ScalarType::Bool => (data_type::bool(), false.into()),
        ScalarType::String => (data_type::string(), "".into()),
    }
}

fn new_empty_dataset_helper<T: BackendData, S: ?Sized>(
    store: Arc<S>,
    path: &str,
    shape: &Shape,
    config: WriteConfig,
) -> Result<zarrs::array::Array<S>> {
    let (datatype, fill) = scalar_type_to_zarr(T::DTYPE);

    let shape_ref = shape.as_ref();
    let chunk_size: Vec<u64> = match config.block_size {
        Some(s) => s.as_ref().into_iter().map(|x| (*x).max(1) as u64).collect(),
        _ => {
            if shape_ref.len() == 1 {
                vec![shape_ref[0].min(16384).max(1) as u64]
            } else {
                shape_ref.iter().map(|&x| x.min(128).max(1) as u64).collect()
            }
        }
    };

    let use_sharding = !datatype.is::<StringDataType>();

    let array_shape: Vec<u64> = shape_ref.iter().map(|x| *x as u64).collect();

    let array = if use_sharding {
        let shard_shape: Vec<u64> = match config.shard_size {
            Some(s) => s.as_ref().iter().map(|x| (*x).max(1) as u64).collect(),
            None => chunk_size.iter().map(|&x| x * 8).collect(),
        };
        zarrs::array::ArrayBuilder::new(
            array_shape,
            shard_shape,
            datatype,
            fill,
        )
        .subchunk_shape(chunk_size)
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(7, false))])
        .build(store, path)?
    } else {
        zarrs::array::ArrayBuilder::new(
            array_shape,
            chunk_size,
            datatype,
            fill,
        )
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(7, false))])
        .build(store, path)?
    };

    Ok(array)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndata::s;
    use ndarray::{array, concatenate, Array2, Axis, Ix2};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use std::path::PathBuf;
    use tempfile::tempdir;

    pub fn with_tmp_dir<T, F: FnMut(PathBuf) -> T>(mut func: F) -> T {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();
        func(path)
    }

    fn with_tmp_path<T, F: Fn(PathBuf) -> T>(func: F) -> T {
        with_tmp_dir(|dir| func(dir.join("temp")))
    }

    #[test]
    fn test_basic() -> Result<()> {
        with_tmp_path(|path| {
            let store = Zarr::new(&path)?;
            store.open_group("/")?;

            store.new_scalar_dataset("data", &4)?;
            store.open_dataset("data")?;

            let group = store.new_group("group")?;
            assert!(store.exists("group")?);

            let subgroup = group.new_group("group")?;
            assert!(group.exists("group")?);

            let subsubgroup = subgroup.new_group("group")?;
            assert!(subgroup.exists("group")?);

            let data = subsubgroup.new_scalar_dataset("group", &4)?;
            assert!(subsubgroup.exists("group")?);
            subsubgroup.open_dataset("group")?;

            {
                let store = Zarr::open(&path)?;
                DataContainer::open(&store, "group")?;
            }

            assert_eq!(group.path(), PathBuf::from("/group"));
            assert_eq!(subgroup.path(), PathBuf::from("/group/group"));
            assert_eq!(subsubgroup.path(), PathBuf::from("/group/group/group"));
            assert_eq!(data.path(), PathBuf::from("/group/group/group/group"));
            Ok(())
        })
    }

    #[test]
    fn test_write_empty() -> Result<()> {
        with_tmp_path(|path| {
            let store = Zarr::new(&path)?;
            let group = store.new_group("group")?;
            let config = WriteConfig {
                ..Default::default()
            };

            let empty: Array2<i64> = array![[]];
            let dataset = group.new_array_dataset("test", empty.view().into(), config)?;
            assert_eq!(empty, dataset.read_array::<i64, Ix2>()?);
            Ok(())
        })
    }

    #[test]
    fn test_write_slice() -> Result<()> {
        with_tmp_path(|path| {
            let store = Zarr::new(&path)?;
            let config = WriteConfig {
                block_size: Some(vec![2, 2].as_slice().into()),
                ..Default::default()
            };

            let group = store.new_group("group")?;
            let mut dataset =
                group.new_empty_dataset::<i32>("test", &[20, 50].as_slice().into(), config)?;

            let arr = Array::random((10, 10), Uniform::new(0, 100).unwrap());
            dataset.write_array_slice(arr.view().into(), s![5..15, 10..20].as_ref())?;
            assert_eq!(
                arr,
                dataset.read_array_slice::<i32, _, _>(s![5..15, 10..20].as_ref())?
            );

            let arr = Array::random((20, 50), Uniform::new(0, 100).unwrap());
            dataset.write_array_slice(arr.view().into(), s![.., ..].as_ref())?;
            dataset.write_array_slice(arr.view().into(), s![.., ..].as_ref())?;

            dataset.reshape(&[40, 50].as_slice().into())?;
            dataset.write_array_slice(arr.view().into(), s![20..40, ..].as_ref())?;

            let merged = concatenate(Axis(0), &[arr.view(), arr.view()])?;
            assert_eq!(merged, dataset.read_array::<i32, _>()?);

            dataset.reshape(&[20, 50].as_slice().into())?;
            assert_eq!(arr, dataset.read_array::<i32, _>()?);

            dataset.reshape(&[50, 50].as_slice().into())?;
            assert_eq!(
                [50, 50],
                store
                    .open_group("group")?
                    .open_dataset("test")?
                    .shape()
                    .as_ref(),
            );

            assert_eq!(vec!["group"], store.list()?);
            assert_eq!(vec!["test"], group.list()?);

            assert!(store.exists("group")?);
            assert!(group.exists("test")?);

            store.delete("group")?;
            assert!(!store.exists("group")?);
            assert!(!group.exists("test")?);

            Ok(())
        })
    }
}
