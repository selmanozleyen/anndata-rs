#[cfg(feature = "polars")]
mod anndata;
#[cfg(feature = "polars")]
pub mod concat;
#[cfg(feature = "polars")]
pub mod traits;
pub mod backend;
pub mod data;
#[cfg(feature = "polars")]
pub mod container;
#[cfg(feature = "polars")]
pub mod reader;
mod macros;

#[cfg(feature = "polars")]
pub use traits::{AnnDataOp, AxisArraysOp, ElemCollectionOp, ArrayElemOp};
#[cfg(feature = "polars")]
pub use crate::anndata::{AnnData, AnnDataSet, StackedAnnData};
pub use backend::Backend;
pub use data::{HasShape, Data, Readable, Writable, ArrayData, WritableArray, ReadableArray, Selectable};
#[cfg(feature = "polars")]
pub use container::{
    AxisArrays, DataFrameElem, Elem, ElemCollection, ArrayElem, 
    StackedAxisArrays, StackedDataFrame, StackedArrayElem,
};