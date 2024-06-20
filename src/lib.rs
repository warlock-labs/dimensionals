//! The Dimensionals library provides a multidimensional array implementation
//! with a generic storage backend over a generic number type.
//!
//! The main types are:
//!
//! - [`DimensionalStorage`]: A trait defining methods for storage backends.
//! - [`LinearArrayStorage`]: A specific storage backend using a linear memory layout.
//! - [`Dimensional`]: The main multidimensional array type, generic over the storage backend.
//!
//! The library also provides some convenience macros for creating arrays:
//!
//! - [`vector!`]: Creates a 1-dimensional array.
//! - [`matrix!`]: Creates a 2-dimensional array.
//!
//! # Example
//!
//! ```
//! use dimensionals::{matrix, Dimensional, LinearArrayStorage};
//!
//! let m: Dimensional<f64, LinearArrayStorage<f64, 2>, 2> = matrix![
//!     [1.0, 2.0, 3.0],
//!     [4.0, 5.0, 6.0]
//! ];
//! assert_eq!(m[[0, 0]], 1.0);
//! assert_eq!(m[[1, 1]], 5.0);
//! ```
//!
//! # Performance
//!
//! The `LinearArrayStorage` backend stores elements in a contiguous `Vec<T>`
//! and computes element indices on the fly. This provides good cache locality
//! for traversals, but may not be optimal for sparse or very high dimensional arrays.
//!  
//! Alternative storage backends can be implemented by defining a type that
//! implements the `DimensionalStorage` trait.

mod iterators;
mod linear_storage;
mod operators;

pub use linear_storage::{LinearArrayLayout, LinearArrayStorage};

use num::Num;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// A trait for storage backends for multidimensional arrays.
///
/// This trait defines methods for creating arrays filled with zeros or ones,
/// and for creating an array from a vector of data.
///
/// # Type Parameters
///
/// * `T`: The element type of the array. Must implement `Num` and `Copy`.
/// * `N`: The number of dimensions of the array.
pub trait DimensionalStorage<T: Num + Copy, const N: usize>:
    Index<[usize; N], Output = T> + IndexMut<[usize; N], Output = T>
{
    /// Creates an array filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
    fn zeros(shape: [usize; N]) -> Self;

    /// Creates an array filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
    fn ones(shape: [usize; N]) -> Self;

    /// Creates an array from a vector of data.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
    /// * `data`: The data to initialize the array with.
    fn from_vec(shape: [usize; N], data: Vec<T>) -> Self;

    /// Returns a mutable slice of the underlying data from storage
    fn as_mut_slice(&mut self) -> &mut [T];
}

/// A multidimensional array type.
///
/// This struct represents a multidimensional array with a generic storage backend.
///
/// # Type Parameters
///
/// * `T`: The element type of the array. Must implement `Num` and `Copy`.
/// * `S`: The storage backend for the array. Must implement `DimensionalStorage`.
/// * `N`: The number of dimensions of the array.
#[derive(Debug, Copy, Clone)]
pub struct Dimensional<T: Num + Copy, S, const N: usize>
where
    S: DimensionalStorage<T, N>,
{
    shape: [usize; N],
    storage: S,
    _marker: PhantomData<T>,
}

impl<T: Num + Copy, S, const N: usize> Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    /// Creates a new array filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
    pub fn zeros(shape: [usize; N]) -> Self
    where
        S: DimensionalStorage<T, N>,
    {
        let storage = S::zeros(shape);
        Self {
            shape,
            storage,
            _marker: PhantomData,
        }
    }

    /// Creates a new array filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
    pub fn ones(shape: [usize; N]) -> Self
    where
        S: DimensionalStorage<T, N>,
    {
        let storage = S::ones(shape);
        Self {
            shape,
            storage,
            _marker: PhantomData,
        }
    }

    /// Creates a new multidimensional array.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
    /// * `storage`: The storage backend for the array.
    pub fn new(shape: [usize; N], storage: S) -> Self {
        Self {
            shape,
            storage,
            _marker: PhantomData,
        }
    }

    /// Creates a new array using a function to initialize each element.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
    /// * `f`: A function that takes an index and returns the value for that index.
    pub fn from_fn<F>(shape: [usize; N], f: F) -> Self
    where
        F: Fn([usize; N]) -> T,
        S: DimensionalStorage<T, N>,
    {
        let data = (0..shape.iter().product::<usize>())
            .map(|i| {
                let index = Self::unravel_index(i, &shape);
                f(index)
            })
            .collect();

        let storage = S::from_vec(shape, data);
        Self {
            shape,
            storage,
            _marker: PhantomData,
        }
    }

    /// Converts a linear index to a multidimensional index.
    ///
    /// # Arguments
    ///
    /// * `index`: The linear index.
    /// * `shape`: The shape of the array.
    fn unravel_index(index: usize, shape: &[usize; N]) -> [usize; N] {
        let mut index = index;
        let mut unraveled = [0; N];

        for i in (0..N).rev() {
            unraveled[i] = index % shape[i];
            index /= shape[i];
        }

        unraveled
    }

    /// Converts a multidimensional index to a linear index.
    ///
    /// # Arguments
    ///
    /// * `indices`: The multidimensional index.
    /// * `shape`: The shape of the array.
    fn ravel_index(indices: &[usize; N], shape: &[usize; N]) -> usize {
        indices
            .iter()
            .zip(shape.iter())
            .fold(0, |acc, (&i, &s)| acc * s + i)
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    /// Returns the number of dimensions of the array.
    pub fn ndim(&self) -> usize {
        N
    }

    /// Returns the total number of elements in the array.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns `true` if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the length of the array along a given axis.
    ///
    /// # Arguments
    ///
    /// * `axis`: The axis to get the length of.
    pub fn len_axis(&self, axis: usize) -> usize {
        self.shape[axis]
    }

    /// Returns a mutable slice of the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }
}

// Macros

#[macro_export]
macro_rules! vector {
    ($($value:expr),+) => {
        {
            let data = vec![$($value),+];
            let shape = [data.len()];
            Dimensional::<_, LinearArrayStorage<_, 1>, 1>::from_fn(shape, |[i]| data[i])
        }
    };
}

#[macro_export]
macro_rules! matrix {
    ($([$($value:expr),* $(,)?]),+ $(,)?) => {
        {
            let data: Vec<Vec<_>> = vec![$(vec![$($value),*]),+];
            let rows = data.len();
            let cols = data[0].len();
            let shape = [rows, cols];
            Dimensional::<_, LinearArrayStorage<_, 2>, 2>::from_fn(shape, |[i, j]| data[i][j])
        }
    };
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::linear_storage::LinearArrayStorage;
    use crate::{matrix, vector};

    #[test]
    fn test_shape() {
        let v = vector![1, 2, 3, 4, 5];
        assert_eq!(v.shape(), [5]);

        let m = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert_eq!(m.shape(), [3, 3]);
    }

    #[test]
    fn test_ndim() {
        let v = vector![1, 2, 3, 4, 5];
        assert_eq!(v.ndim(), 1);

        let m = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert_eq!(m.ndim(), 2);
    }

    #[test]
    fn test_len() {
        let v = vector![1, 2, 3, 4, 5];
        assert_eq!(v.len(), 5);

        let m = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert_eq!(m.len(), 9);
    }

    #[test]
    fn test_len_axis() {
        let m = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert_eq!(m.len_axis(0), 3);
        assert_eq!(m.len_axis(1), 3);
    }
}
