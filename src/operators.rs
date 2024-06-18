use crate::{Dimensional, DimensionalStorage};
use num::Num;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// Indexes into the array using a multi-dimensional index à la `array[[i, j]]`.
impl<T: Num + Copy, S, const N: usize> Index<[usize; N]> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = T;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.storage[index]
    }
}

/// Mutable indexing into the array using a multi-dimensional index à la `array[[i, j]]`.
impl<T: Num + Copy, S, const N: usize> IndexMut<[usize; N]> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        &mut self.storage[index]
    }
}

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

// Scalar addition
impl<T: Num + Copy, S, const N: usize> Add<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        // Create a new array filled with zeros
        let mut result = Dimensional::zeros(self.shape);
        // Fill each element with the sum of the `rhs` and the corresponding element in `self`
        for i in 0..self.shape.iter().product::<usize>() {
            let index = Self::unravel_index(i, &self.shape);
            result[index] = self[index] + rhs;
        }
        result
    }
}

// Scalar subtraction
impl<T: Num + Copy, S, const N: usize> Sub<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        // Create a new array filled with zeros
        let mut result = Dimensional::zeros(self.shape);
        // Fill each element with the difference of the `rhs` and the corresponding element in `self`
        for i in 0..self.shape.iter().product::<usize>() {
            let index = Self::unravel_index(i, &self.shape);
            result[index] = self[index] - rhs;
        }
        result
    }
}

// Scalar multiplication
impl<T: Num + Copy, S, const N: usize> Mul<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        // Create a new array filled with zeros
        let mut result = Dimensional::zeros(self.shape);
        // Fill each element with the product of the `rhs` and the corresponding element in `self`
        for i in 0..self.shape.iter().product::<usize>() {
            let index = Self::unravel_index(i, &self.shape);
            result[index] = self[index] * rhs;
        }
        result
    }
}

// Scalar division
impl<T: Num + Copy, S, const N: usize> Div<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        // Create a new array filled with zeros
        let mut result = Dimensional::zeros(self.shape);
        // Fill each element with the quotient of the `rhs` and the corresponding element in `self`
        for i in 0..self.shape.iter().product::<usize>() {
            let index = Self::unravel_index(i, &self.shape);
            result[index] = self[index] / rhs;
        }
        result
    }
}
