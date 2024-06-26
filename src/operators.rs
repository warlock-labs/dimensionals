use crate::{storage::DimensionalStorage, Dimensional, LinearArrayStorage};
use num_traits::Num;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Implements indexing operations for Dimensional arrays.

impl<T: Num + Copy, S, const N: usize> Index<[usize; N]> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = T;

    /// Returns an index into the array using a multidimensional index à la [i, j, k].
    fn index(&self, index: [usize; N]) -> &Self::Output {
        // TODO(This is too tightly coupled to the storage layout)
        &self.storage[index]
    }
}

/// Implements mutable indexing operations for Dimensional arrays.
impl<T: Num + Copy, S, const N: usize> IndexMut<[usize; N]> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    /// Returns a mutable index into the array using a multidimensional index à la [i, j, k].
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        // TODO(This is too tightly coupled to the storage layout)
        &mut self.storage[index]
    }
}

/// Implements partial equality comparison for Dimensional arrays.
impl<T: Num + Copy + PartialEq, S, const N: usize> PartialEq for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    /// Compares two `Dimensional` arrays for partial equality.
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }

        // TODO(Benchmark copying these to slice vs not)
        self.as_slice() == other.as_slice()
    }
}

// TODO(These operators in general need an audit and correction)
// for varying operations between scalars, vectors, matrices and tensors.
//
// Some will fit neatly into the rust operators
// Some will need to be implemented as methods
//
// Scalars:
// Addition: a + b
// Subtraction: a - b
// Multiplication: a * b
// Division: a / b
// Vectors:
// Addition: (a + b)_i = a_i + b_i
// Subtraction: (a - b)_i = a_i - b_i
// Scalar Multiplication: (ca)_i = c * a_i
// Dot Product: a · b = Σ a_i * b_i
// Cross Product (3D): (a × b)_1 = a_2b_3 - a_3b_2, (a × b)_2 = a_3b_1 - a_1b_3, (a × b)_3 = a_1b_2 - a_2b_1
// Tensor Product: (a ⊗ b)_ij = a_i * b_j
// Matrices:
// Addition: (A + B)_ij = A_ij + B_ij
// Subtraction: (A - B)_ij = A_ij - B_ij
// Scalar Multiplication: (cA)_ij = c * A_ij
// Matrix Multiplication: (AB)_ij = Σ A_ik * B_kj
// Transpose: (A^T)_ij = A_ji
// Inverse: AA^(-1) = A^(-1)A = I
// Trace: tr(A) = Σ A_ii
// Determinant: det(A)
// Tensors:
// Addition: (A + B)_i1i2...in = A_i1i2...in + B_i1i2...in
// Subtraction: (A - B)_i1i2...in = A_i1i2...in - B_i1i2...in
// Scalar Multiplication: (cA)_i1i2...in = c * A_i1i2...in
// Hadamard Product: (A ⊙ B)_i1i2...in = A_i1i2...in * B_i1i2...in
// Tensor Product: (A ⊗ B)_i1...in,j1...jm = A_i1...in * B_j1...jm
// Transpose: (A^T)_i1...ik...in = A_i1...in...ik
// Contraction: Σ A_i1...ik...in
// There is also broadcasting between different shapes to consider.

// Scalar arithmetic operations

/// Implements scalar addition for Dimensional arrays.
impl<T: Num + Copy, S, const N: usize> Add<T> for &Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Dimensional<T, S, N>;

    /// Adds a scalar to a `Dimensional` array element-wise.
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

    /// Subtracts a scalar from a `Dimensional` array element-wise.
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

    /// Multiplies a `Dimensional` array by a scalar element-wise.
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

    // Divides a `Dimensional` array by a scalar element-wise.
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

    /// Adds two `Dimensional` arrays element-wise.
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

    /// Subtracts one `Dimensional` array from another element-wise.
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

    /// Multiplies two `Dimensional` arrays element-wise.
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

    /// Divides one `Dimensional` array by another element-wise.
    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Shapes must match for element-wise division"
        );
        self.zip_map(rhs, |a, b| a / b)
    }
}

impl<T: Num + Copy + std::iter::Sum, S, const N: usize> Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    pub fn transpose(&self) -> Dimensional<T, S, N> {
        let r: Vec<T> = self
            .iter_transpose()
            .enumerate()
            .map(|(_, val)| *val)
            .collect();
        let new_shape: [usize; N] = self
            .shape()
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Dimensional::from_fn(new_shape, |idxs: [usize; N]| {
            r[Dimensional::<T, LinearArrayStorage<T, N>, N>::ravel_index(&idxs, &new_shape)]
        })
    }
}

///Implements matrix multiplication for 2-Dimensional arrays
impl<T: Num + Copy + std::iter::Sum, S> Dimensional<T, S, 2>
where
    S: DimensionalStorage<T, 2>,
{
    pub fn matmul(&self, rhs: &Self) -> Dimensional<T, S, 2> {
        assert_eq!(
            self.shape()[1],
            rhs.shape()[0],
            "Requires matrices be of the shapes (MxN) % (NxK). Interior dimensions do not match."
        );
        let m = self.shape()[0];
        let n = self.shape()[1];
        let k = rhs.shape()[1];

        // given combination of dimensions, and the fact that current built in
        // iterators and mappings only iterate pairwise with identical indices,
        // something more custom is needed. naive algorithm with for looping
        // given below
        let shape = [m, k];
        let r: Vec<T> = (0..m)
            .flat_map(|i| {
                (0..k).map(move |j| {
                    (0..n)
                        .map(|x| {
                            let raveled =
                                Dimensional::<T, LinearArrayStorage<T, 2>, 2>::ravel_index(
                                    &[i, x],
                                    &self.shape(),
                                );
                            let raveled_rhs =
                                Dimensional::<T, LinearArrayStorage<T, 2>, 2>::ravel_index(
                                    &[x, j],
                                    &rhs.shape(),
                                );
                            self.as_slice()[raveled] * rhs.as_slice()[raveled_rhs]
                        })
                        .sum()
                })
            })
            .collect();
        Dimensional::from_fn(shape, |[i, j]| r[k * i + j])
    }
}

// Assignment operations

/// Implements scalar addition assignment for Dimensional arrays.
impl<T: Num + Copy + AddAssign, S, const N: usize> AddAssign<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    /// Adds a scalar to a `Dimensional` array element-wise in-place.
    fn add_assign(&mut self, rhs: T) {
        self.map_inplace(|x| *x += rhs);
    }
}

/// Implements scalar subtraction assignment for Dimensional arrays.
impl<T: Num + Copy + SubAssign, S, const N: usize> SubAssign<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    /// Subtracts a scalar from a `Dimensional` array element-wise in-place.
    fn sub_assign(&mut self, rhs: T) {
        self.map_inplace(|x| *x -= rhs);
    }
}

/// Implements scalar multiplication assignment for Dimensional arrays.
impl<T: Num + Copy + MulAssign, S, const N: usize> MulAssign<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    /// Multiplies a `Dimensional` array by a scalar element-wise in-place.
    fn mul_assign(&mut self, rhs: T) {
        self.map_inplace(|x| *x *= rhs);
    }
}

/// Implements scalar division assignment for Dimensional arrays.
impl<T: Num + Copy + DivAssign, S, const N: usize> DivAssign<T> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    /// Divides a `Dimensional` array by a scalar element-wise in-place.
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
    /// Adds two `Dimensional` arrays element-wise in-place.
    fn add_assign(&mut self, rhs: &Dimensional<T, S, N>) {
        assert_eq!(
            self.shape, rhs.shape,
            "Shapes must match for element-wise addition assignment"
        );
        self.zip_map_inplace(rhs, |a, b| *a += b);
    }
}

/// Implements element-wise subtraction assignment for Dimensional arrays.
impl<T: Num + Copy + SubAssign, S, const N: usize> SubAssign<&Dimensional<T, S, N>>
    for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    /// Subtracts one `Dimensional` array from another element-wise in-place.
    fn sub_assign(&mut self, rhs: &Dimensional<T, S, N>) {
        assert_eq!(
            self.shape, rhs.shape,
            "Shapes must match for element-wise subtraction assignment"
        );
        self.zip_map_inplace(rhs, |a, b| *a -= b);
    }
}

/// Implements element-wise multiplication assignment for Dimensional arrays.
impl<T: Num + Copy + MulAssign, S, const N: usize> MulAssign<&Dimensional<T, S, N>>
    for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    /// Multiplies two `Dimensional` arrays element-wise in-place.
    fn mul_assign(&mut self, rhs: &Dimensional<T, S, N>) {
        assert_eq!(
            self.shape, rhs.shape,
            "Shapes must match for element-wise multiplication assignment"
        );
        self.zip_map_inplace(rhs, |a, b| *a *= b);
    }
}

/// Implements element-wise division assignment for Dimensional arrays.
impl<T: Num + Copy + DivAssign, S, const N: usize> DivAssign<&Dimensional<T, S, N>>
    for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    /// Divides one `Dimensional` array by another element-wise in-place.
    fn div_assign(&mut self, rhs: &Dimensional<T, S, N>) {
        assert_eq!(
            self.shape, rhs.shape,
            "Shapes must match for element-wise division assignment"
        );
        self.zip_map_inplace(rhs, |a, b| *a /= b);
    }
}

// Implement unary negation for references
impl<T: Num + Copy + Neg<Output = T>, S, const N: usize> Neg for &Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = Dimensional<T, S, N>;

    /// Negates a `Dimensional` array element-wise.
    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

// TODO How much are these helper abstractions really helping?
// Seems like .zip .map etc should do it without these.
// We don't want bloat, we want a razor sharp and performant tool.
// We can likely create a map/zip/collect implementation or override
// to make this better.

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

        assert_eq!(m1.matmul(&m2), matrix![[19, 22], [43, 50]]);

        assert_eq!(m1.transpose(), matrix![[1, 3], [2, 4]])

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
