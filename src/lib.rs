//! The Dimensionals library provides a multidimensional array implementation
//! with a generic storage backend over a generic number type.
//!
//! In other words, it's got and element type `T`, a storage backend `S`
//! and a number of dimensions `N`.
//!
//! A scalar is a 0-dimensional object, or just the element of type `T` itself
//! A vector is a 1-dimensional array of elements of type `T`
//! A matrix is a 2-dimensional array of elements of type `T`
//! A tensor is an `N`-dimensional array of elements of type `T`
//!
//! The goal of this library is to provide a flexible and efficient way to work with
//! multidimensional arrays of numerics in Rust. Storage is generic over `S` to allow
//! for different memory layouts and optimizations.
//!
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

mod core;
mod iterators;
mod operators;
mod storage;

// Public API
pub use core::Dimensional;
pub use iterators::*;
pub use storage::LinearArrayStorage;

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
    use crate::storage::LinearArrayStorage;
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
