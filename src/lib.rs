use num::Num;
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut};

pub trait DimensionalStorage<T: Num + Copy, const N: usize>:
    Index<[usize; N], Output = T> + IndexMut<[usize; N], Output = T>
{
    fn zeros(shape: [usize; N]) -> Self;
    fn ones(shape: [usize; N]) -> Self;
    fn from_vec(shape: [usize; N], data: Vec<T>) -> Self;
}

enum LinearArrayLayout {
    RowMajor,
    ColumnMajor,
}

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
}

impl<T: Num + Copy, const N: usize> LinearArrayStorage<T, N> {
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

    fn new(shape: [usize; N], data: Vec<T>, layout: LinearArrayLayout, stride: usize) -> Self {
        let strides = Self::compute_strides(&shape, &stride, &layout);
        Self {
            data,
            layout,
            strides,
        }
    }
}

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

    fn unravel_index(index: usize, shape: &[usize; N]) -> [usize; N] {
        let mut index = index;
        let mut unraveled = [0; N];

        for i in (0..N).rev() {
            unraveled[i] = index % shape[i];
            index /= shape[i];
        }

        unraveled
    }

    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    pub fn ndim(&self) -> usize {
        N
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len_axis(&self, axis: usize) -> usize {
        self.shape[axis]
    }
}

impl<T: Num + Copy, S, const N: usize> Index<[usize; N]> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    type Output = T;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.storage[index]
    }
}

impl<T: Num + Copy, S, const N: usize> IndexMut<[usize; N]> for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        &mut self.storage[index]
    }
}

impl<T: Num + Copy, S: DimensionalStorage<T, 1>> Add for Dimensional<T, S, 1> {
    type Output = Dimensional<T, S, 1>;

    fn add(self, rhs: Self) -> Self::Output {
        let shape = self.shape;
        let mut result = Dimensional::zeros(shape);

        for i in 0..shape[0] {
            result[[i]] = self[[i]] + rhs[[i]];
        }

        result
    }
}

impl<T: Num + Copy, S: DimensionalStorage<T, 2>> Add for Dimensional<T, S, 2> {
    type Output = Dimensional<T, S, 2>;

    // Vector addition
    fn add(self, rhs: Self) -> Self::Output {
        let shape = self.shape;
        let mut result = Dimensional::zeros(shape);

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result[[i, j]] = self[[i, j]] + rhs[[i, j]];
            }
        }

        result
    }
}

#[macro_export]
macro_rules! scalar {
    ($value:expr) => {{
        let data = vec![$value];
        let shape = [1];
        Dimensional::<_, LinearArrayStorage<_, 1>, 1>::from_fn(shape, |[i]| data[i])
    }};
}

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
    use std::ops::Add;
    use super::*;
    use crate::{matrix, scalar, vector};

    #[test]
    fn test_dimensional_array_column_major_layout() {
        let shape = [2, 3];
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let array = Dimensional::new(
            shape,
            LinearArrayStorage::new(shape, data, LinearArrayLayout::ColumnMajor, 1),
        );

        assert_eq!(array[[0, 0]], 1.0);
        assert_eq!(array[[1, 0]], 4.0);
        assert_eq!(array[[0, 1]], 2.0);
        assert_eq!(array[[1, 1]], 5.0);
        assert_eq!(array[[0, 2]], 3.0);
        assert_eq!(array[[1, 2]], 6.0);
    }

    #[test]
    fn test_scalar() {
        let s = scalar!(42);
        assert_eq!(s.shape(), [1]);
        assert_eq!(s[[0]], 42);
    }

    #[test]
    fn test_vector() {
        let v = vector![1, 2, 3, 4, 5];
        assert_eq!(v.shape(), [5]);
        assert_eq!(v[[0]], 1);
        assert_eq!(v[[2]], 3);
        assert_eq!(v[[4]], 5);
    }

    #[test]
    fn test_matrix() {
        let m = matrix![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert_eq!(m.shape(), [3, 3]);
        assert_eq!(m[[0, 0]], 1);
        assert_eq!(m[[1, 1]], 5);
        assert_eq!(m[[2, 2]], 9);
    }

    #[test]
    fn test_zeros() {
        let z = Dimensional::<f64, LinearArrayStorage<f64, 2>, 2>::zeros([3, 4]);
        assert_eq!(z.shape(), [3, 4]);
        assert_eq!(z[[0, 0]], 0.0);
        assert_eq!(z[[1, 2]], 0.0);
        assert_eq!(z[[2, 3]], 0.0);
    }

    #[test]
    fn test_ones() {
        let o = Dimensional::<i32, LinearArrayStorage<i32, 3>, 3>::ones([2, 3, 4]);
        assert_eq!(o.shape(), [2, 3, 4]);
        assert_eq!(o[[0, 0, 0]], 1);
        assert_eq!(o[[1, 1, 1]], 1);
        assert_eq!(o[[1, 2, 3]], 1);
    }

    #[test]
    fn test_from_fn() {
        let f = Dimensional::<f64, LinearArrayStorage<f64, 2>, 2>::from_fn([3, 3], |[i, j]| {
            (i + j) as f64
        });
        assert_eq!(f.shape(), [3, 3]);
        assert_eq!(f[[0, 0]], 0.0);
        assert_eq!(f[[1, 1]], 2.0);
        assert_eq!(f[[2, 2]], 4.0);
    }

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

    #[test]
    fn test_addition() {
        let v1 = vector![1, 2, 3, 4, 5];
        let v2 = vector![6, 7, 8, 9, 10];
        let v3 = v1 + v2;
        assert_eq!(v3[[0]], 7);
        assert_eq!(v3[[2]], 11);
        assert_eq!(v3[[4]], 15);
    }

    #[test]
    fn test_column_major_addition() {
        let shape = [2, 3];
        let data1 = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let data2 = vec![7.0, 10.0, 8.0, 11.0, 9.0, 12.0];
        let array1 = Dimensional::new(
            shape,
            LinearArrayStorage::new(shape, data1, LinearArrayLayout::ColumnMajor, 1),
        );
        let array2 = Dimensional::new(
            shape,
            LinearArrayStorage::new(shape, data2, LinearArrayLayout::ColumnMajor, 1),
        );
        let array3 = array1 + array2;
        assert_eq!(array3[[0, 0]], 8.0);
        assert_eq!(array3[[1, 0]], 14.0);
        assert_eq!(array3[[0, 1]], 10.0);
        assert_eq!(array3[[1, 1]], 16.0);
        assert_eq!(array3[[0, 2]], 12.0);
        assert_eq!(array3[[1, 2]], 18.0);
    }
}
