use crate::backend::{AttributeOp, Backend, DataContainer, DatasetOp, GroupOp};
use crate::data::array::slice::SelectInfoElem;
use crate::data::data_traits::*;
use crate::data::index::{Index, Interval};

use anyhow::{Result, bail};
use log::warn;
use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct DataFrameIndex {
    pub index_name: String,
    index: Index,
}

impl std::cmp::PartialEq for DataFrameIndex {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl DataFrameIndex {
    pub fn empty() -> Self {
        Self {
            index_name: "index".to_string(),
            index: Index::empty(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn get_index(&self, k: &str) -> Option<usize> {
        self.index.get_index(k)
    }

    pub fn into_vec(self) -> Vec<String> {
        self.index.into_vec()
    }

    pub fn select(&self, select: &SelectInfoElem) -> Self {
        let index = self.index.select(select);
        Self {
            index_name: self.index_name.clone(),
            index,
        }
    }

    pub(crate) fn read<B: Backend>(container: &DataContainer<B>) -> Result<Self> {
        let index_name: String = container.get_attr("_index")?;
        let dataset = container.as_group()?.open_dataset(&index_name)?;
        match dataset
            .get_attr::<String>("index_type")
            .as_ref()
            .map_or("list", |x| x.as_str())
        {
            "list" => {
                let data = dataset.read_array()?;
                let mut index: DataFrameIndex = data.to_vec().into();
                index.index_name = index_name;
                Ok(index)
            }
            "intervals" => {
                let keys: Vec<String> = dataset.get_attr("names")?;
                let values: Vec<Vec<u64>> = dataset.get_attr("intervals")?;
                Ok(keys
                    .into_iter()
                    .zip(values.into_iter().map(|row| Interval {
                        start: row[0] as usize,
                        end: row[1] as usize,
                        size: row[2] as usize,
                        step: row[3] as usize,
                    }))
                    .collect())
            }
            "range" => {
                let start: u64 = dataset.get_attr("start")?;
                let end: u64 = dataset.get_attr("end")?;
                Ok((start as usize..end as usize).into())
            }
            x => bail!("Unknown index type: {}", x),
        }
    }

    pub(crate) fn overwrite<B: Backend>(&self, container: &mut DataContainer<B>) -> Result<()> {
        if let Ok(index_name) = container.get_attr::<String>("_index") {
            container.as_group()?.delete(&index_name)?;
        }
        container.new_attr("_index", self.index_name.clone())?;
        let group = container.as_group()?;
        let arr: Array1<String> = self.clone().into_iter().collect();
        let mut data = arr.write(group, &self.index_name)?;
        match &self.index {
            Index::List(_) => {
                data.new_attr("index_type", "list")?;
            }
            Index::Intervals(intervals) => {
                let keys: Vec<String> = intervals.keys().cloned().collect();
                let values: Vec<Vec<u64>> = intervals
                    .values()
                    .map(|x| vec![x.start as u64, x.end as u64, x.size as u64, x.step as u64])
                    .collect();
                if data.new_attr("names", keys).is_err()
                    || data.new_attr("intervals", values).is_err()
                {
                    data.new_attr("index_type", "list")?;
                    warn!("Failed to save interval index as attributes, fallback to list index");
                } else {
                    data.new_attr("index_type", "intervals")?;
                }
            }
            Index::Range(range) => {
                data.new_attr("index_type", "range")?;
                data.new_attr("start", range.start as u64)?;
                data.new_attr("end", range.end as u64)?;
            }
        }
        Ok(())
    }
}

impl IntoIterator for DataFrameIndex {
    type Item = String;
    type IntoIter = Box<dyn Iterator<Item = String>>;

    fn into_iter(self) -> Self::IntoIter {
        self.index.into_iter()
    }
}

impl<D> From<D> for DataFrameIndex
where
    Index: From<D>,
{
    fn from(data: D) -> Self {
        Self {
            index_name: "index".to_owned(),
            index: data.into(),
        }
    }
}

impl<D> FromIterator<D> for DataFrameIndex
where
    Index: FromIterator<D>,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = D>,
    {
        Self {
            index_name: "index".to_owned(),
            index: iter.into_iter().collect(),
        }
    }
}
