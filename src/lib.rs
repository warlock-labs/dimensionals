//! The Dimensionals library provides a multidimensional array implementation
//! with a generic storage backend over a generic number type.
//!
//! # Core Concepts
//!
//! - Element type `T`: The type of data stored in the array.
//! - Storage backend `S`: The underlying storage mechanism for the array.
//! - Number of dimensions `N`: The dimensionality of the array.
//!
//! # Dimensional Types
//!
//! - Scalar: A 0-dimensional object, or just the element of type `T` itself.
//! - Vector: A 1-dimensional array of elements with the type `T`.
//! - Matrix: A 2-dimensional array of elements with the type `T`.
//! - Tensor: An `N`-dimensional array of elements with the type `T`, where `N` > 2.
//!
//! # Goals
//!
//! The primary goal of this library is to provide a flexible and efficient way to work with
//! multidimensional arrays of numeric types in Rust.
//!
//! Using a generic storage backend, `S`, allows for different memory layouts and optimizations.
//!
//! # Convenience Macros
//!
//! The library provides convenience macros for creating arrays:
//!
//! - [`vector!`]: Creates a 1-dimensional array.
//! - [`matrix!`]: Creates a 2-dimensional array.
//!
//! # Example
//!
//! ```
//! use dimensionals::{matrix, vector, Dimensional, LinearArrayStorage};
//!
//! // Create a vector
//! let v: Dimensional<i32, LinearArrayStorage<i32, 1>, 1> = vector![1, 2, 3, 4, 5];
//! assert_eq!(v[[0]], 1);
//!
//! // Create a matrix
//! let m: Dimensional<f64, LinearArrayStorage<f64, 2>, 2> = matrix![
//!     [1.0, 2.0, 3.0],
//!     [4.0, 5.0, 6.0]
//! ];
//! assert_eq!(m[[0, 0]], 1.0);
//! assert_eq!(m[[1, 1]], 5.0);
//! ```
mod core;
mod display;
mod iterators;
mod operators;
mod storage;
mod brownian;

// Public API
pub use crate::core::Dimensional;
pub use iterators::*;
pub use storage::DimensionalStorage;
pub use storage::LinearArrayStorage;

/// Creates a 1-dimensional array (vector).
///
/// # Examples
///
/// ```
/// use dimensionals::{vector, Dimensional, LinearArrayStorage};
///
/// let v: Dimensional<i32, LinearArrayStorage<i32, 1>, 1> = vector![1, 2, 3, 4, 5];
/// assert_eq!(v[[0]], 1);
/// assert_eq!(v[[4]], 5);
/// ```
#[macro_export]
macro_rules! vector {
    ($($value:expr),+ $(,)?) => {
        {
            let data = vec![$($value),+];
            let shape = [data.len()];
            Dimensional::<_, LinearArrayStorage<_, 1>, 1>::from_fn(shape, |[i]| data[i])
        }
    };
}

/// Creates a 2-dimensional array (matrix).
///
/// # Examples
///
/// ```
/// use dimensionals::{matrix, Dimensional, LinearArrayStorage};
///
/// let m: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = matrix![
///     [1, 2, 3],
///     [4, 5, 6]
/// ];
/// assert_eq!(m[[0, 0]], 1);
/// assert_eq!(m[[1, 2]], 6);
/// ```
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

// TODO: Implement a generic tensor macro
// The tensor macro should create an N-dimensional array (N > 2) with the following features:
// - Infer the number of dimensions and shape from the input
// - Work with any number of dimensions (3 or more)
// - Be as user-friendly as the vector! and matrix! macros
// - Handle type inference correctly
// - Integrate seamlessly with the Dimensional struct and LinearArrayStorage

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{matrix, vector};

    #[test]
    fn test_vector_creation() {
        let v: Dimensional<i32, LinearArrayStorage<i32, 1>, 1> = vector![1, 2, 3, 4, 5];
        assert_eq!(v.shape(), [5]);
        assert_eq!(v[[0]], 1);
        assert_eq!(v[[4]], 5);
    }

    #[test]
    fn test_vector_indexing() {
        let v = vector![10, 20, 30, 40, 50];
        assert_eq!(v[[0]], 10);
        assert_eq!(v[[2]], 30);
        assert_eq!(v[[4]], 50);
    }

    #[test]
    fn test_vector_iteration() {
        let v = vector![1, 2, 3, 4, 5];
        let sum: i32 = v.iter().sum();
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_matrix_creation() {
        let m: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = matrix![[1, 2, 3], [4, 5, 6]];
        assert_eq!(m.shape(), [2, 3]);
        assert_eq!(m[[0, 0]], 1);
        assert_eq!(m[[1, 2]], 6);
    }

    #[test]
    fn test_matrix_indexing() {
        let m = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert_eq!(m[[0, 0]], 1);
        assert_eq!(m[[1, 1]], 5);
        assert_eq!(m[[2, 2]], 9);
    }

    #[test]
    fn test_matrix_iteration() {
        let m = matrix![[1, 2], [3, 4]];
        let sum: i32 = m.iter().sum();
        assert_eq!(sum, 10);
    }

    #[test]
    fn test_dimensional_properties() {
        let v = vector![1, 2, 3, 4, 5];
        assert_eq!(v.ndim(), 1);
        assert_eq!(v.len(), 5);
        assert_eq!(v.len_axis(0), 5);

        let m = matrix![[1, 2, 3], [4, 5, 6]];
        assert_eq!(m.ndim(), 2);
        assert_eq!(m.len(), 6);
        assert_eq!(m.len_axis(0), 2);
        assert_eq!(m.len_axis(1), 3);
    }

    #[test]
    fn test_dimensional_from_fn() {
        let v = Dimensional::<_, LinearArrayStorage<_, 1>, 1>::from_fn([5], |[i]| i * 2);
        assert_eq!(v[[0]], 0);
        assert_eq!(v[[2]], 4);
        assert_eq!(v[[4]], 8);

        let m = Dimensional::<_, LinearArrayStorage<_, 2>, 2>::from_fn([3, 3], |[i, j]| i + j);
        assert_eq!(m[[0, 0]], 0);
        assert_eq!(m[[1, 1]], 2);
        assert_eq!(m[[2, 2]], 4);
    }

    #[test]
    fn test_dimensional_zeros_and_ones() {
        let v_zeros = Dimensional::<i32, LinearArrayStorage<i32, 1>, 1>::zeros([5]);
        assert_eq!(v_zeros.iter().sum::<i32>(), 0);

        let v_ones = Dimensional::<i32, LinearArrayStorage<i32, 1>, 1>::ones([5]);
        assert_eq!(v_ones.iter().sum::<i32>(), 5);

        let m_zeros = Dimensional::<i32, LinearArrayStorage<i32, 2>, 2>::zeros([3, 3]);
        assert_eq!(m_zeros.iter().sum::<i32>(), 0);

        let m_ones = Dimensional::<i32, LinearArrayStorage<i32, 2>, 2>::ones([3, 3]);
        assert_eq!(m_ones.iter().sum::<i32>(), 9);
    }
}
