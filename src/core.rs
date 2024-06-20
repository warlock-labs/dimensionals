use crate::storage::DimensionalStorage;
use num_traits::Num;
use std::marker::PhantomData;

/// A multidimensional array type.
///
/// This struct represents a multidimensional array with a generic storage backend.
///
/// # Type Parameters
///
/// * `T`: The element type of the array. Must implement `Num` and `Copy`.
/// * `S`: The storage backend for the array. Must implement `DimensionalStorage`.
/// * `N`: The dimensionality of the array a `usize`.
#[derive(Debug, Clone)]
pub struct Dimensional<T: Num + Copy, S, const N: usize>
where
    S: DimensionalStorage<T, N>,
{
    pub(crate) shape: [usize; N],
    pub(crate) storage: S,
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
    ///
    /// # Examples
    ///
    /// ```
    /// use dimensionals::{Dimensional, LinearArrayStorage};
    ///
    /// let zeros: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = Dimensional::zeros([2, 3]);
    /// assert_eq!(zeros.shape(), [2, 3]);
    /// assert!(zeros.as_slice().iter().all(|&x| x == 0));
    /// ```
    pub fn zeros(shape: [usize; N]) -> Self {
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
    ///
    /// # Examples
    ///
    /// ```
    /// use dimensionals::{Dimensional, LinearArrayStorage};
    ///
    /// let ones: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = Dimensional::ones([2, 3]);
    /// assert_eq!(ones.shape(), [2, 3]);
    /// assert!(ones.as_slice().iter().all(|&x| x == 1));
    /// ```
    pub fn ones(shape: [usize; N]) -> Self {
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
    ///
    /// # Examples
    ///
    /// ```
    /// use dimensionals::{Dimensional, LinearArrayStorage, DimensionalStorage};
    ///
    /// let storage = LinearArrayStorage::from_vec([2, 3], vec![1, 2, 3, 4, 5, 6]);
    /// let array = Dimensional::new([2, 3], storage);
    /// assert_eq!(array.shape(), [2, 3]);
    /// assert_eq!(array.as_slice(), &[1, 2, 3, 4, 5, 6]);
    /// ```
    pub fn new(shape: [usize; N], storage: S) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            storage.as_slice().len(),
            "Storage size must match the product of shape dimensions"
        );
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
    ///
    /// # Examples
    ///
    /// ```
    /// use dimensionals::{Dimensional, LinearArrayStorage};
    ///
    /// let array: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> =
    ///     Dimensional::from_fn([2, 3], |[i, j]| (i * 3 + j) as i32);
    /// assert_eq!(array.shape(), [2, 3]);
    /// assert_eq!(array.as_slice(), &[0, 1, 2, 3, 4, 5]);
    /// ```
    pub fn from_fn<F>(shape: [usize; N], f: F) -> Self
    where
        F: Fn([usize; N]) -> T,
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
    ///
    /// # Returns
    ///
    /// A multidimensional index as an array of `usize`.
    pub fn unravel_index(index: usize, shape: &[usize; N]) -> [usize; N] {
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
    ///
    /// # Returns
    ///
    /// A linear index as `usize`.
    pub fn ravel_index(indices: &[usize; N], shape: &[usize; N]) -> usize {
        indices
            .iter()
            .zip(shape.iter())
            .fold(0, |acc, (&i, &s)| acc * s + i)
    }

    /// Returns the shape of the array.
    ///
    /// # Returns
    ///
    /// An array of `usize` representing the shape of the array.
    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    /// Returns the number of dimensions of the array.
    ///
    /// # Returns
    ///
    /// The number of dimensions as `usize`.
    pub fn ndim(&self) -> usize {
        N
    }

    /// Returns the total number of elements in the array.
    ///
    /// # Returns
    ///
    /// The total number of elements as `usize`.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns `true` if the array is empty.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the length of the array along a given axis.
    ///
    /// # Arguments
    ///
    /// * `axis`: The axis to get the length of.
    ///
    /// # Returns
    ///
    /// The length of the specified axis as `usize`.
    ///
    /// # Panics
    ///
    /// Panics if the axis is out of bounds.
    pub fn len_axis(&self, axis: usize) -> usize {
        assert!(axis < N, "Axis out of bounds");
        self.shape[axis]
    }

    /// Returns a mutable slice of the underlying data.
    ///
    /// # Returns
    ///
    /// A mutable slice of the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }

    /// Returns an immutable slice of the underlying data.
    ///
    /// # Returns
    ///
    /// An immutable slice of the underlying data.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearArrayStorage;

    #[test]
    fn test_zeros_and_ones() {
        let zeros: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = Dimensional::zeros([2, 3]);
        assert_eq!(zeros.shape(), [2, 3]);
        assert!(zeros.as_slice().iter().all(|&x| x == 0));

        let ones: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = Dimensional::ones([2, 3]);
        assert_eq!(ones.shape(), [2, 3]);
        assert!(ones.as_slice().iter().all(|&x| x == 1));
    }

    #[test]
    fn test_new() {
        let storage = LinearArrayStorage::from_vec([2, 3], vec![1, 2, 3, 4, 5, 6]);
        let array = Dimensional::new([2, 3], storage);
        assert_eq!(array.shape(), [2, 3]);
        assert_eq!(array.as_slice(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    #[should_panic(expected = "Storage size must match the product of shape dimensions")]
    fn test_new_mismatched_shape() {
        let storage = LinearArrayStorage::from_vec([2, 2], vec![1, 2, 3, 4]);
        Dimensional::new([2, 3], storage);
    }

    #[test]
    fn test_from_fn() {
        let array: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> =
            Dimensional::from_fn([2, 3], |[i, j]| (i * 3 + j) as i32);
        assert_eq!(array.shape(), [2, 3]);
        assert_eq!(array.as_slice(), &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_unravel_and_ravel_index() {
        let shape = [2, 3, 4];
        for i in 0..24 {
            let unraveled =
                Dimensional::<i32, LinearArrayStorage<i32, 3>, 3>::unravel_index(i, &shape);
            let raveled =
                Dimensional::<i32, LinearArrayStorage<i32, 3>, 3>::ravel_index(&unraveled, &shape);
            assert_eq!(i, raveled);
        }
    }

    #[test]
    fn test_shape_and_dimensions() {
        let array: Dimensional<i32, LinearArrayStorage<i32, 3>, 3> = Dimensional::zeros([2, 3, 4]);
        assert_eq!(array.shape(), [2, 3, 4]);
        assert_eq!(array.ndim(), 3);
        assert_eq!(array.len(), 24);
        assert!(!array.is_empty());
        assert_eq!(array.len_axis(0), 2);
        assert_eq!(array.len_axis(1), 3);
        assert_eq!(array.len_axis(2), 4);
    }

    #[test]
    #[should_panic(expected = "Axis out of bounds")]
    fn test_len_axis_out_of_bounds() {
        let array: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = Dimensional::zeros([2, 3]);
        array.len_axis(2);
    }

    #[test]
    fn test_as_slice_and_as_mut_slice() {
        let mut array: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> =
            Dimensional::from_fn([2, 3], |[i, j]| (i * 3 + j) as i32);

        assert_eq!(array.as_slice(), &[0, 1, 2, 3, 4, 5]);

        {
            let slice = array.as_mut_slice();
            slice[0] = 10;
            slice[5] = 50;
        }

        assert_eq!(array.as_slice(), &[10, 1, 2, 3, 4, 50]);
    }
}
