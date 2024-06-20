use crate::DimensionalStorage;
use num::Num;
use std::ops::{Index, IndexMut};

/// An enum representing the memory layout of a linear array.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LinearArrayLayout {
    /// Row-major layout (default).
    RowMajor,
    // TODO: figure out if we want to support column-major layout
    #[allow(dead_code)]
    ColumnMajor,
}

/// A linear array storage backend for multidimensional arrays.
///
/// This struct stores the array data in a contiguous block of memory,
/// using either row-major or column-major layout.
///
/// # Type Parameters
///
/// * `T`: The element type of the array. Must implement `Num` and `Copy`.
/// * `N`: The number of dimensions of the array.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearArrayStorage<T: Num + Copy, const N: usize> {
    data: Vec<T>,
    layout: LinearArrayLayout,
    strides: [usize; N],
}

impl<T: Num + Copy, const N: usize> Index<[usize; N]> for LinearArrayStorage<T, N> {
    type Output = T;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        let linear_index = self.layout_index(index);
        &self.data[linear_index]
    }
}

impl<T: Num + Copy, const N: usize> IndexMut<[usize; N]> for LinearArrayStorage<T, N> {
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        let linear_index = self.layout_index(index);
        &mut self.data[linear_index]
    }
}

impl<T: Num + Copy, const N: usize> DimensionalStorage<T, N> for LinearArrayStorage<T, N> {
    fn zeros(shape: [usize; N]) -> Self {
        let data = vec![T::zero(); shape.iter().product::<usize>()];
        LinearArrayStorage::new(shape, data, LinearArrayLayout::RowMajor, 1)
    }

    fn ones(shape: [usize; N]) -> Self {
        let data = vec![T::one(); shape.iter().product::<usize>()];
        LinearArrayStorage::new(shape, data, LinearArrayLayout::RowMajor, 1)
    }

    fn from_vec(shape: [usize; N], data: Vec<T>) -> Self {
        LinearArrayStorage::new(shape, data, LinearArrayLayout::RowMajor, 1)
    }

    /// Returns a mutable slice of the underlying data.
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: Num + Copy, const N: usize> LinearArrayStorage<T, N> {
    /// Computes the strides for a given shape and layout.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
    /// * `stride`: The base stride (usually 1).
    /// * `layout`: The memory layout of the array.
    fn compute_strides(
        shape: &[usize; N],
        stride: &usize,
        layout: &LinearArrayLayout,
    ) -> [usize; N] {
        let mut strides = [0; N];
        match layout {
            LinearArrayLayout::RowMajor => {
                strides[N - 1] = *stride;
                for i in (0..N - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
            }
            LinearArrayLayout::ColumnMajor => {
                strides[0] = *stride;
                for i in 1..N {
                    strides[i] = strides[i - 1] * shape[i - 1];
                }
            }
        }
        strides
    }

    /// Computes the linear index for a given multidimensional index.
    ///
    /// # Arguments
    ///
    /// * `index`: The multidimensional index.
    fn layout_index(&self, index: [usize; N]) -> usize {
        match self.layout {
            LinearArrayLayout::RowMajor => index
                .iter()
                .zip(self.strides.iter())
                .map(|(i, &stride)| i * stride)
                .sum(),
            LinearArrayLayout::ColumnMajor => index
                .iter()
                .rev()
                .zip(self.strides.iter().rev())
                .map(|(i, &stride)| i * stride)
                .sum(),
        }
    }

    /// Creates a new `LinearArrayStorage` with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
    /// * `data`: The data to initialize the array with.
    /// * `layout`: The memory layout of the array.
    /// * `stride`: The base stride (usually 1).
    pub fn new(shape: [usize; N], data: Vec<T>, layout: LinearArrayLayout, stride: usize) -> Self {
        let strides = Self::compute_strides(&shape, &stride, &layout);
        Self {
            data,
            layout,
            strides,
        }
    }
}
