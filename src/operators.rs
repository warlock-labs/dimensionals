/// Syntactic sugar for idiomatic usage of Dimensionals.
use crate::{storage::DimensionalStorage, Dimensional};
use num::Num;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// Indexes into the array using a multi-dimensional index à la `array[[i, j; N]]`.
impl<T: Num + Copy, S, const N: usize> Index<[usize; N]> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = T;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.storage[index]
    }
}

/// Mutable indexing into the array using a multi-dimensional index à la `array[[i, j; N]]`.
impl<T: Num + Copy, S, const N: usize> IndexMut<[usize; N]> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        &mut self.storage[index]
    }
}

// TODO(The arithmetic operations really require good iterators to be efficient).

/// Equality comparison for arrays.
impl<T: Num + Copy + PartialEq, S, const N: usize> PartialEq for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }

        for i in 0..self.shape.iter().product::<usize>() {
            let index = Self::unravel_index(i, &self.shape);
            if self[index] != other[index] {
                return false;
            }
        }

        true
    }
}
impl<T: Num + Copy + PartialEq, S, const N: usize> Eq for Dimensional<T, S, N> where
    S: DimensionalStorage<T, N>
{
}

// Scalar addition
impl<T: Num + Copy, S, const N: usize> Add<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Self;

    /// Adds a scalar `rhs` to each element of the array.
    fn add(self, _rhs: T) -> Self::Output {
        todo!("Implement scalar addition")
    }
}

// Scalar subtraction
impl<T: Num + Copy, S, const N: usize> Sub<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Self;

    /// Subtracts a scalar `rhs` from each element of the array.
    fn sub(self, _rhs: T) -> Self::Output {
        todo!("Implement scalar subtraction")
    }
}

// Scalar multiplication
impl<T: Num + Copy, S, const N: usize> Mul<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Self;

    /// Multiplies a scalar `rhs` for each element of the array.
    fn mul(self, _rhs: T) -> Self::Output {
        todo!("Implement scalar multiplication")
    }
}

// Scalar division
impl<T: Num + Copy, S, const N: usize> Div<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Self;

    /// Divides each element of the array by a scalar `rhs`.
    fn div(self, _rhs: T) -> Self::Output {
        todo!("Implement scalar division")
    }
}

// Element-wise operations

// Tensor Addition
impl<T: Num + Copy, S, const N: usize> Add<Dimensional<T, S, N>> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Self;

    /// Adds two arrays element-wise.
    fn add(self, _rhs: Dimensional<T, S, N>) -> Self::Output {
        todo!("Implement tensor division")
    }
}

// This should support all other possible operator overloads to perform linear operations

// tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LinearArrayStorage;
    use crate::{matrix, vector};

    #[test]
    fn test_indexing() {
        let v = vector![1, 2, 3, 4, 5];
        assert_eq!(v[[0]], 1);
        assert_eq!(v[[2]], 3);
        assert_eq!(v[[4]], 5);

        let m = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert_eq!(m[[0, 0]], 1);
        assert_eq!(m[[1, 1]], 5);
        assert_eq!(m[[2, 2]], 9);
    }

    #[test]
    fn test_mutable_indexing() {
        let mut v = vector![1, 2, 3, 4, 5];
        v[[0]] = 10;
        v[[2]] = 30;
        v[[4]] = 50;
        assert_eq!(v[[0]], 10);
        assert_eq!(v[[2]], 30);
        assert_eq!(v[[4]], 50);

        let mut m = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        m[[0, 0]] = 10;
        m[[1, 1]] = 50;
        m[[2, 2]] = 90;
        assert_eq!(m[[0, 0]], 10);
        assert_eq!(m[[1, 1]], 50);
        assert_eq!(m[[2, 2]], 90);
    }
}
