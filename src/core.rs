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
#[derive(Debug, Clone, Eq, Copy)]
pub struct Dimensional<T: Num + Copy, S, const N: usize>
where
    S: DimensionalStorage<T, N>,
{
    pub(crate) shape: [usize; N],
    pub(crate) storage: S,
    pub(crate) len: usize,
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
        let len = shape.iter().product();

        Self {
            shape,
            storage,
            len,
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
        let len = shape.iter().product();

        Self {
            shape,
            storage,
            len,
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
        let len = shape.iter().product();
        Self {
            shape,
            storage,
            len,
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
        let len = shape.iter().product();

        Self {
            shape,
            storage,
            len,
            _marker: PhantomData,
        }
    }

    // TODO Seems like both of these could just use the shape already on the object

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

    // TODO what if any is the use case for jagged arrays?

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
        self.len
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

    // TODO Seems like there may need to be an abstraction layer here

    /// Returns a mutable slice of the underlying data.
    ///
    /// # Returns
    ///
    /// A mutable slice of the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }

    // TODO same story here, this seems pretty tightly coupled to the storage

    /// Returns an immutable slice of the underlying data.
    ///
    /// # Returns
    ///
    /// An immutable slice of the underlying data.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }
}

// Specific implementations for 2D arrays
impl<T: Num + Copy, S> Dimensional<T, S, 2>
where
    S: DimensionalStorage<T, 2>,
{
    /// Creates a new identity array (square matrix with ones on the diagonal and zeros elsewhere).
    ///
    /// # Arguments
    ///
    /// * `n`: The size of the square matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use dimensionals::{Dimensional, LinearArrayStorage};
    ///
    /// let eye: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = Dimensional::eye(3);
    /// assert_eq!(eye.shape(), [3, 3]);
    /// assert_eq!(eye[[0, 0]], 1);
    /// assert_eq!(eye[[1, 1]], 1);
    /// assert_eq!(eye[[2, 2]], 1);
    /// assert_eq!(eye[[0, 1]], 0);
    /// ```
    pub fn eye(n: usize) -> Self {
        Self::from_fn([n, n], |[i, j]| if i == j { T::one() } else { T::zero() })
    }

    /// Creates a new identity-like array with a specified value on the diagonal.
    ///
    /// # Arguments
    ///
    /// * `n`: The size of the square matrix.
    /// * `value`: The value to place on the diagonal.
    ///
    /// # Examples
    ///
    /// ```
    /// use dimensionals::{Dimensional, LinearArrayStorage};
    ///
    /// let eye: Dimensional<f64, LinearArrayStorage<f64, 2>, 2> = Dimensional::eye_value(3, 2.5);
    /// assert_eq!(eye.shape(), [3, 3]);
    /// assert_eq!(eye[[0, 0]], 2.5);
    /// assert_eq!(eye[[1, 1]], 2.5);
    /// assert_eq!(eye[[2, 2]], 2.5);
    /// assert_eq!(eye[[0, 1]], 0.0);
    /// ```
    pub fn eye_value(n: usize, value: T) -> Self {
        Self::from_fn([n, n], |[i, j]| if i == j { value } else { T::zero() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearArrayStorage;
    use num_traits::FloatConst;

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

    #[test]
    fn test_eye() {
        let eye: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = Dimensional::eye(3);

        // Check shape
        assert_eq!(eye.shape(), [3, 3]);

        // Check diagonal elements
        assert_eq!(eye[[0, 0]], 1);
        assert_eq!(eye[[1, 1]], 1);
        assert_eq!(eye[[2, 2]], 1);

        // Check off-diagonal elements
        assert_eq!(eye[[0, 1]], 0);
        assert_eq!(eye[[0, 2]], 0);
        assert_eq!(eye[[1, 0]], 0);
        assert_eq!(eye[[1, 2]], 0);
        assert_eq!(eye[[2, 0]], 0);
        assert_eq!(eye[[2, 1]], 0);

        // Check sum of all elements (should equal to the size of the matrix)
        let sum: i32 = eye.as_slice().iter().sum();
        assert_eq!(sum, 3);

        // Test with a different size
        let eye_4x4: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = Dimensional::eye(4);
        assert_eq!(eye_4x4.shape(), [4, 4]);
        assert_eq!(eye_4x4[[3, 3]], 1);
        assert_eq!(eye_4x4[[0, 3]], 0);

        // Test with a different type
        let eye_float: Dimensional<f64, LinearArrayStorage<f64, 2>, 2> = Dimensional::eye(2);
        assert_eq!(eye_float[[0, 0]], 1.0);
        assert_eq!(eye_float[[0, 1]], 0.0);
        assert_eq!(eye_float[[1, 0]], 0.0);
        assert_eq!(eye_float[[1, 1]], 1.0);
    }

    #[test]
    fn test_eye_value() {
        // Test with integer type
        let eye_int: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = Dimensional::eye_value(3, 5);
        assert_eq!(eye_int.shape(), [3, 3]);
        assert_eq!(eye_int[[0, 0]], 5);
        assert_eq!(eye_int[[1, 1]], 5);
        assert_eq!(eye_int[[2, 2]], 5);
        assert_eq!(eye_int[[0, 1]], 0);
        assert_eq!(eye_int[[1, 2]], 0);

        // Test with floating-point type
        let eye_float: Dimensional<f64, LinearArrayStorage<f64, 2>, 2> =
            Dimensional::eye_value(2, f64::PI());
        assert_eq!(eye_float.shape(), [2, 2]);
        assert_eq!(eye_float[[0, 0]], f64::PI());
        assert_eq!(eye_float[[1, 1]], f64::PI());
        assert_eq!(eye_float[[0, 1]], 0.0);
        assert_eq!(eye_float[[1, 0]], 0.0);

        // Test with a negative value
        let eye_neg: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> =
            Dimensional::eye_value(2, -1);
        assert_eq!(eye_neg.shape(), [2, 2]);
        assert_eq!(eye_neg[[0, 0]], -1);
        assert_eq!(eye_neg[[1, 1]], -1);
        assert_eq!(eye_neg[[0, 1]], 0);
        assert_eq!(eye_neg[[1, 0]], 0);
    }

    #[test]
    fn test_len() {
        let array_2d: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = Dimensional::zeros([2, 3]);
        assert_eq!(array_2d.len(), 6);

        let array_3d: Dimensional<i32, LinearArrayStorage<i32, 3>, 3> =
            Dimensional::zeros([2, 3, 4]);
        assert_eq!(array_3d.len(), 24);

        let array_1d: Dimensional<i32, LinearArrayStorage<i32, 1>, 1> = Dimensional::zeros([5]);
        assert_eq!(array_1d.len(), 5);

        let array_empty: Dimensional<i32, LinearArrayStorage<i32, 1>, 1> = Dimensional::zeros([0]);
        assert_eq!(array_empty.len(), 0);
    }
}
