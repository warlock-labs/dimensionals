use num_traits::Num;
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

    /// Returns the total number of elements in the storage.
    fn len(&self) -> usize;

    /// Returns a mutable slice of the underlying data from storage.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Returns an immutable slice of the underlying data from storage.
    fn as_slice(&self) -> &[T];
}

/// An enum representing the memory layout of a linear array.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LinearArrayLayout {
    /// Row-major layout (default).
    RowMajor,
    /// Column-major layout.
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
    shape: [usize; N],
    layout: LinearArrayLayout,
    strides: [usize; N],
    len: usize,
}

impl<T: Num + Copy, const N: usize> LinearArrayStorage<T, N> {
    /// Computes the strides for a given shape and layout.
    ///
    /// In this implementation, strides represent the number of elements (not bytes) to skip
    /// in each dimension when traversing the array. This approach simplifies indexing calculations
    /// while still providing efficient access to elements.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
    /// * `layout`: The memory layout of the array.
    ///
    /// # Returns
    ///
    /// An array of strides, where each stride represents the number of elements to skip
    /// in the corresponding dimension.
    fn compute_strides(shape: &[usize; N], layout: &LinearArrayLayout) -> [usize; N] {
        let mut strides = [0; N];
        match layout {
            LinearArrayLayout::RowMajor => {
                strides[N - 1] = 1;
                for i in (0..N - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
            }
            LinearArrayLayout::ColumnMajor => {
                strides[0] = 1;
                for i in 1..N {
                    strides[i] = strides[i - 1] * shape[i - 1];
                }
            }
        }
        strides
    }

    /// Computes the linear index for a given multidimensional index.
    ///
    /// This method calculates the position of an element in the underlying 1D vector
    /// based on its multidimensional index and the array's strides.
    ///
    /// # Arguments
    ///
    /// * `index`: The multidimensional index.
    ///
    /// # Returns
    ///
    /// The linear index in the underlying data vector.
    fn layout_index(&self, index: [usize; N]) -> usize {
        index
            .iter()
            .zip(self.strides.iter())
            .map(|(&i, &stride)| i * stride)
            .sum()
    }

    /// Creates a new `LinearArrayStorage` with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
    /// * `data`: The data to initialize the array with.
    /// * `layout`: The memory layout of the array.
    ///
    /// # Panics
    ///
    /// Panic if the length of `data` doesn't match the product of dimensions in `shape`.
    pub fn new(shape: [usize; N], data: Vec<T>, layout: LinearArrayLayout) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "Data length must match the product of shape dimensions"
        );
        let strides = Self::compute_strides(&shape, &layout);
        let len = data.len();
        Self {
            data,
            shape,
            layout,
            strides,
            len,
        }
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize; N] {
        &self.shape
    }

    /// Returns the layout of the array.
    pub fn layout(&self) -> LinearArrayLayout {
        self.layout
    }

    /// Returns the strides of the array.
    pub fn strides(&self) -> &[usize; N] {
        &self.strides
    }
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
        LinearArrayStorage::new(shape, data, LinearArrayLayout::RowMajor)
    }

    fn ones(shape: [usize; N]) -> Self {
        let data = vec![T::one(); shape.iter().product::<usize>()];
        LinearArrayStorage::new(shape, data, LinearArrayLayout::RowMajor)
    }

    fn from_vec(shape: [usize; N], data: Vec<T>) -> Self {
        LinearArrayStorage::new(shape, data, LinearArrayLayout::RowMajor)
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_and_ones() {
        let zeros = LinearArrayStorage::<i32, 2>::zeros([2, 3]);
        assert_eq!(zeros.as_slice(), &[0, 0, 0, 0, 0, 0]);

        let ones = LinearArrayStorage::<i32, 2>::ones([2, 3]);
        assert_eq!(ones.as_slice(), &[1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let array = LinearArrayStorage::<i32, 2>::from_vec([2, 3], data.clone());
        assert_eq!(array.as_slice(), &data);
    }

    #[test]
    #[should_panic(expected = "Data length must match the product of shape dimensions")]
    fn test_from_vec_wrong_size() {
        let data = vec![1, 2, 3, 4, 5];
        LinearArrayStorage::<i32, 2>::from_vec([2, 3], data);
    }

    #[test]
    fn test_indexing() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let array = LinearArrayStorage::<i32, 2>::from_vec([2, 3], data);
        assert_eq!(array[[0, 0]], 1);
        assert_eq!(array[[0, 2]], 3);
        assert_eq!(array[[1, 1]], 5);
    }

    #[test]
    fn test_mutable_indexing() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let mut array = LinearArrayStorage::<i32, 2>::from_vec([2, 3], data);
        array[[0, 0]] = 10;
        array[[1, 2]] = 20;
        assert_eq!(array[[0, 0]], 10);
        assert_eq!(array[[1, 2]], 20);
    }

    #[test]
    fn test_strides_calculation() {
        let row_major =
            LinearArrayStorage::<i32, 3>::new([2, 3, 4], vec![0; 24], LinearArrayLayout::RowMajor);
        assert_eq!(row_major.strides(), &[12, 4, 1]);

        let col_major = LinearArrayStorage::<i32, 3>::new(
            [2, 3, 4],
            vec![0; 24],
            LinearArrayLayout::ColumnMajor,
        );
        assert_eq!(col_major.strides(), &[1, 2, 6]);
    }

    #[test]
    fn test_layout_index() {
        let row_major = LinearArrayStorage::<i32, 3>::new(
            [2, 3, 4],
            (0..24).collect(),
            LinearArrayLayout::RowMajor,
        );
        assert_eq!(row_major[[0, 0, 0]], 0);
        assert_eq!(row_major[[1, 2, 3]], 23);
        assert_eq!(row_major[[0, 1, 2]], 6);

        let col_major = LinearArrayStorage::<i32, 3>::new(
            [2, 3, 4],
            (0..24).collect(),
            LinearArrayLayout::ColumnMajor,
        );
        assert_eq!(col_major[[0, 0, 0]], 0);
        assert_eq!(col_major[[1, 2, 3]], 23);
        assert_eq!(col_major[[0, 1, 2]], 14);
    }

    #[test]
    fn test_different_layouts() {
        let data: Vec<i32> = (0..6).collect();

        let row_major = LinearArrayStorage::new([2, 3], data.clone(), LinearArrayLayout::RowMajor);
        assert_eq!(row_major[[0, 0]], 0);
        assert_eq!(row_major[[0, 2]], 2);
        assert_eq!(row_major[[1, 0]], 3);

        let col_major = LinearArrayStorage::new([2, 3], data, LinearArrayLayout::ColumnMajor);
        assert_eq!(col_major[[0, 0]], 0);
        assert_eq!(col_major[[0, 2]], 4);
        assert_eq!(col_major[[1, 0]], 1);
    }

    #[test]
    fn test_as_slice_and_as_mut_slice() {
        let mut array = LinearArrayStorage::<i32, 2>::from_vec([2, 3], vec![1, 2, 3, 4, 5, 6]);

        assert_eq!(array.as_slice(), &[1, 2, 3, 4, 5, 6]);

        {
            let slice = array.as_mut_slice();
            slice[0] = 10;
            slice[5] = 60;
        }

        assert_eq!(array.as_slice(), &[10, 2, 3, 4, 5, 60]);
    }
}
