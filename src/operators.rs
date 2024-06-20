use crate::{storage::DimensionalStorage, Dimensional};
use num::Num;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Implements indexing operations for Dimensional arrays.
impl<T: Num + Copy, S, const N: usize> Index<[usize; N]> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = T;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.storage[index]
    }
}

/// Implements mutable indexing operations for Dimensional arrays.
impl<T: Num + Copy, S, const N: usize> IndexMut<[usize; N]> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        &mut self.storage[index]
    }
}

/// Implements equality comparison for Dimensional arrays.
impl<T: Num + Copy + PartialEq, S, const N: usize> PartialEq for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }

        self.as_slice() == other.as_slice()
    }
}

impl<T: Num + Copy + Eq, S, const N: usize> Eq for Dimensional<T, S, N> where
    S: DimensionalStorage<T, N>
{
}

// Scalar arithmetic operations

/// Implements scalar addition for Dimensional arrays.
impl<T: Num + Copy, S, const N: usize> Add<T> for &Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Dimensional<T, S, N>;

    fn add(self, rhs: T) -> Self::Output {
        self.map(|x| x + rhs)
    }
}

/// Implements scalar subtraction for Dimensional arrays.
impl<T: Num + Copy, S, const N: usize> Sub<T> for &Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Dimensional<T, S, N>;

    fn sub(self, rhs: T) -> Self::Output {
        self.map(|x| x - rhs)
    }
}

/// Implements scalar multiplication for Dimensional arrays.
impl<T: Num + Copy, S, const N: usize> Mul<T> for &Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Dimensional<T, S, N>;

    fn mul(self, rhs: T) -> Self::Output {
        self.map(|x| x * rhs)
    }
}

/// Implements scalar division for Dimensional arrays.
impl<T: Num + Copy, S, const N: usize> Div<T> for &Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Dimensional<T, S, N>;

    fn div(self, rhs: T) -> Self::Output {
        self.map(|x| x / rhs)
    }
}

// Element-wise operations

/// Implements element-wise addition for Dimensional arrays.
impl<T: Num + Copy, S, const N: usize> Add for &Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Dimensional<T, S, N>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Shapes must match for element-wise addition"
        );
        self.zip_map(rhs, |a, b| a + b)
    }
}

/// Implements element-wise subtraction for Dimensional arrays.
impl<T: Num + Copy, S, const N: usize> Sub for &Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Dimensional<T, S, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Shapes must match for element-wise subtraction"
        );
        self.zip_map(rhs, |a, b| a - b)
    }
}

/// Implements element-wise multiplication for Dimensional arrays.
impl<T: Num + Copy, S, const N: usize> Mul for &Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Dimensional<T, S, N>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Shapes must match for element-wise multiplication"
        );
        self.zip_map(rhs, |a, b| a * b)
    }
}

/// Implements element-wise division for Dimensional arrays.
impl<T: Num + Copy, S, const N: usize> Div for &Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Dimensional<T, S, N>;

    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Shapes must match for element-wise division"
        );
        self.zip_map(rhs, |a, b| a / b)
    }
}

// Assignment operations

/// Implements scalar addition assignment for Dimensional arrays.
impl<T: Num + Copy + AddAssign, S, const N: usize> AddAssign<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn add_assign(&mut self, rhs: T) {
        self.map_inplace(|x| *x += rhs);
    }
}

/// Implements scalar subtraction assignment for Dimensional arrays.
impl<T: Num + Copy + SubAssign, S, const N: usize> SubAssign<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn sub_assign(&mut self, rhs: T) {
        self.map_inplace(|x| *x -= rhs);
    }
}

/// Implements scalar multiplication assignment for Dimensional arrays.
impl<T: Num + Copy + MulAssign, S, const N: usize> MulAssign<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.map_inplace(|x| *x *= rhs);
    }
}

/// Implements scalar division assignment for Dimensional arrays.
impl<T: Num + Copy + DivAssign, S, const N: usize> DivAssign<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn div_assign(&mut self, rhs: T) {
        self.map_inplace(|x| *x /= rhs);
    }
}

/// Implements element-wise addition assignment for Dimensional arrays.
impl<T: Num + Copy + AddAssign, S, const N: usize> AddAssign<&Dimensional<T, S, N>>
    for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn add_assign(&mut self, rhs: &Dimensional<T, S, N>) {
        assert_eq!(
            self.shape, rhs.shape,
            "Shapes must match for element-wise addition assignment"
        );
        self.zip_map_inplace(&rhs, |a, b| *a += b);
    }
}

/// Implements element-wise subtraction assignment for Dimensional arrays.
impl<T: Num + Copy + SubAssign, S, const N: usize> SubAssign<&Dimensional<T, S, N>>
    for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn sub_assign(&mut self, rhs: &Dimensional<T, S, N>) {
        assert_eq!(
            self.shape, rhs.shape,
            "Shapes must match for element-wise subtraction assignment"
        );
        self.zip_map_inplace(&rhs, |a, b| *a -= b);
    }
}

/// Implements element-wise multiplication assignment for Dimensional arrays.
impl<T: Num + Copy + MulAssign, S, const N: usize> MulAssign<&Dimensional<T, S, N>>
    for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn mul_assign(&mut self, rhs: &Dimensional<T, S, N>) {
        assert_eq!(
            self.shape, rhs.shape,
            "Shapes must match for element-wise multiplication assignment"
        );
        self.zip_map_inplace(&rhs, |a, b| *a *= b);
    }
}

/// Implements element-wise division assignment for Dimensional arrays.
impl<T: Num + Copy + DivAssign, S, const N: usize> DivAssign<&Dimensional<T, S, N>>
    for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn div_assign(&mut self, rhs: &Dimensional<T, S, N>) {
        assert_eq!(
            self.shape, rhs.shape,
            "Shapes must match for element-wise division assignment"
        );
        self.zip_map_inplace(&rhs, |a, b| *a /= b);
    }
}

// Implement unary negation for references
impl<T: Num + Copy + Neg<Output = T>, S, const N: usize> Neg for &Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Dimensional<T, S, N>;

    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

// TODO How much are these helper abstractions really helping?
// Seems like .zip .map etc should do it without these.
// We don't want bloat, we want a razor sharp and performant tool.

impl<T, S, const N: usize> Dimensional<T, S, N>
where
    T: Num + Copy,
    S: DimensionalStorage<T, N>,
{
    /// Applies a function to each element of the array, creating a new array.
    fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        Self::from_fn(self.shape, |idx| f(self[idx]))
    }

    /// Applies a function to each element of the array in-place.
    fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&mut T),
    {
        for x in self.as_mut_slice() {
            f(x);
        }
    }

    /// Applies a function to pairs of elements from two arrays, creating a new array.
    fn zip_map<F>(&self, other: &Self, f: F) -> Self
    where
        F: Fn(T, T) -> T,
    {
        assert_eq!(
            self.shape, other.shape,
            "Shapes must match for zip_map operation"
        );
        Self::from_fn(self.shape, |idx| f(self[idx], other[idx]))
    }

    /// Applies a function to pairs of elements from two arrays in-place.
    fn zip_map_inplace<F>(&mut self, other: &Self, f: F)
    where
        F: Fn(&mut T, T),
    {
        assert_eq!(
            self.shape, other.shape,
            "Shapes must match for zip_map_inplace operation"
        );
        for (a, &b) in self.as_mut_slice().iter_mut().zip(other.as_slice().iter()) {
            f(a, b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{matrix, vector, LinearArrayStorage};

    #[test]
    fn test_scalar_operations() {
        let v = vector![1, 2, 3, 4, 5];

        assert_eq!(&v + 1, vector![2, 3, 4, 5, 6]);
        assert_eq!(&v - 1, vector![0, 1, 2, 3, 4]);
        assert_eq!(&v * 2, vector![2, 4, 6, 8, 10]);
        assert_eq!(&v / 2, vector![0, 1, 1, 2, 2]); // Integer division
    }

    #[test]
    fn test_element_wise_operations() {
        let v1 = vector![1, 2, 3, 4, 5];
        let v2 = vector![5, 4, 3, 2, 1];

        assert_eq!(&v1 + &v2, vector![6, 6, 6, 6, 6]);
        assert_eq!(&v1 - &v2, vector![-4, -2, 0, 2, 4]);
        assert_eq!(&v1 * &v2, vector![5, 8, 9, 8, 5]);
        assert_eq!(&v1 / &v2, vector![0, 0, 1, 2, 5]); // Integer division
    }

    #[test]
    fn test_assignment_operations() {
        let mut v = vector![1, 2, 3, 4, 5];

        v += 1;
        assert_eq!(v, vector![2, 3, 4, 5, 6]);

        v -= 1;
        assert_eq!(v, vector![1, 2, 3, 4, 5]);

        v *= 2;
        assert_eq!(v, vector![2, 4, 6, 8, 10]);

        v /= 2;
        assert_eq!(v, vector![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_element_wise_assignment_operations() {
        let mut v1 = vector![1, 2, 3, 4, 5];
        let v2 = vector![5, 4, 3, 2, 1];

        v1 += &v2;
        assert_eq!(v1, vector![6, 6, 6, 6, 6]);

        v1 -= &v2;
        assert_eq!(v1, vector![1, 2, 3, 4, 5]);

        v1 *= &v2;
        assert_eq!(v1, vector![5, 8, 9, 8, 5]);

        v1 /= &v2;
        assert_eq!(v1, vector![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_negation() {
        let v = vector![1, -2, 3, -4, 5];
        assert_eq!(-&v, vector![-1, 2, -3, 4, -5]);
    }

    #[test]
    fn test_matrix_operations() {
        let m1 = matrix![[1, 2], [3, 4]];
        let m2 = matrix![[5, 6], [7, 8]];

        assert_eq!(&m1 + &m2, matrix![[6, 8], [10, 12]]);
        assert_eq!(&m1 - &m2, matrix![[-4, -4], [-4, -4]]);
        assert_eq!(&m1 * &m2, matrix![[5, 12], [21, 32]]);
        assert_eq!(&m1 / &m2, matrix![[0, 0], [0, 0]]); // Integer division

        let mut m3 = m1.clone();
        m3 += 1;
        assert_eq!(m3, matrix![[2, 3], [4, 5]]);

        m3 -= 1;
        assert_eq!(m3, m1);

        m3 *= 2;
        assert_eq!(m3, matrix![[2, 4], [6, 8]]);

        m3 /= 2;
        assert_eq!(m3, m1);

        m3 += &m2;
        assert_eq!(m3, matrix![[6, 8], [10, 12]]);

        m3 -= &m2;
        assert_eq!(m3, m1);

        m3 *= &m2;
        assert_eq!(m3, matrix![[5, 12], [21, 32]]);

        // Note: We don't test m3 /= m2 here because it would result in a matrix of zeros due to integer division
    }

    #[test]
    fn test_mixed_dimensional_operations() {
        let v = vector![1, 2, 3];
        let m = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];

        assert_eq!(&v + 1, vector![2, 3, 4]);
        assert_eq!(&m + 1, matrix![[2, 3, 4], [5, 6, 7], [8, 9, 10]]);

        assert_eq!(&v * 2, vector![2, 4, 6]);
        assert_eq!(&m * 2, matrix![[2, 4, 6], [8, 10, 12], [14, 16, 18]]);
    }

    #[test]
    #[should_panic(expected = "Shapes must match for element-wise addition")]
    fn test_mismatched_shapes_addition() {
        let v1 = vector![1, 2, 3];
        let v2 = vector![1, 2, 3, 4];
        let _ = &v1 + &v2;
    }

    #[test]
    #[should_panic(expected = "Shapes must match for element-wise multiplication")]
    fn test_mismatched_shapes_multiplication() {
        let m1 = matrix![[1, 2], [3, 4]];
        let m2 = matrix![[1, 2, 3], [4, 5, 6]];
        let _ = &m1 * &m2;
    }

    #[test]
    fn test_scalar_operations_with_floats() {
        let v: Dimensional<f64, LinearArrayStorage<f64, 1>, 1> = vector![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(&v + 1.5, vector![2.5, 3.5, 4.5, 5.5, 6.5]);
        assert_eq!(&v - 0.5, vector![0.5, 1.5, 2.5, 3.5, 4.5]);
        assert_eq!(&v * 2.0, vector![2.0, 4.0, 6.0, 8.0, 10.0]);
        assert_eq!(&v / 2.0, vector![0.5, 1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn test_element_wise_operations_with_floats() {
        let v1: Dimensional<f64, LinearArrayStorage<f64, 1>, 1> = vector![1.0, 2.0, 3.0, 4.0, 5.0];
        let v2: Dimensional<f64, LinearArrayStorage<f64, 1>, 1> = vector![0.5, 1.0, 1.5, 2.0, 2.5];

        assert_eq!(&v1 + &v2, vector![1.5, 3.0, 4.5, 6.0, 7.5]);
        assert_eq!(&v1 - &v2, vector![0.5, 1.0, 1.5, 2.0, 2.5]);
        assert_eq!(&v1 * &v2, vector![0.5, 2.0, 4.5, 8.0, 12.5]);
        assert_eq!(&v1 / &v2, vector![2.0, 2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_negation_with_floats() {
        let v: Dimensional<f64, LinearArrayStorage<f64, 1>, 1> = vector![1.5, -2.5, 3.5, -4.5, 5.5];
        assert_eq!(-&v, vector![-1.5, 2.5, -3.5, 4.5, -5.5]);
    }

    #[test]
    fn test_equality() {
        let v1 = vector![1, 2, 3, 4, 5];
        let v2 = vector![1, 2, 3, 4, 5];
        let v3 = vector![1, 2, 3, 4, 6];

        assert_eq!(v1, v2);
        assert_ne!(v1, v3);

        let m1 = matrix![[1, 2], [3, 4]];
        let m2 = matrix![[1, 2], [3, 4]];
        let m3 = matrix![[1, 2], [3, 5]];

        assert_eq!(m1, m2);
        assert_ne!(m1, m3);
    }

    #[test]
    fn test_higher_dimensional_arrays() {
        let a1: Dimensional<i32, LinearArrayStorage<i32, 3>, 3> =
            Dimensional::from_fn([2, 2, 2], |[i, j, k]| (i * 4 + j * 2 + k + 1) as i32);
        let a2: Dimensional<i32, LinearArrayStorage<i32, 3>, 3> =
            Dimensional::from_fn([2, 2, 2], |[i, j, k]| (8 - i * 4 - j * 2 - k) as i32);

        let sum = &a1 + &a2;
        assert_eq!(sum.as_slice(), &[9; 8]);

        let product = &a1 * &a2;
        assert_eq!(product.as_slice(), &[8, 14, 18, 20, 20, 18, 14, 8]);
    }
}
