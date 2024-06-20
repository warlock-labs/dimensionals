use crate::storage::DimensionalStorage;
use num::Num;
use std::marker::PhantomData;

/// A multidimensional array type.
///
/// This struct represents a multidimensional array with a generic storage backend.
///
/// # Type Parameters
///
/// * `T`: The element type of the array. Must implement `Num` and `Copy`.
/// * `S`: The storage backend for the array. Must implement `DimensionalStorage`.
/// * `N`: The number of dimensions of the array.
#[derive(Debug, Copy, Clone)]
pub struct Dimensional<T: Num + Copy, S, const N: usize>
where
    S: DimensionalStorage<T, N>,
{
    pub(crate) shape: [usize; N],
    pub(crate) storage: S,
    _marker: PhantomData<T>,
}

impl<T: Num + Copy, S, const N: usize> crate::Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    /// Creates a new array filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
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

    /// Creates a new array filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
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

    /// Creates a new array using a function to initialize each element.
    ///
    /// # Arguments
    ///
    /// * `shape`: The shape of the array.
    /// * `f`: A function that takes an index and returns the value for that index.
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

    /// Converts a linear index to a multidimensional index.
    ///
    /// # Arguments
    ///
    /// * `index`: The linear index.
    /// * `shape`: The shape of the array.
    pub(crate) fn unravel_index(index: usize, shape: &[usize; N]) -> [usize; N] {
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
    pub(crate) fn ravel_index(indices: &[usize; N], shape: &[usize; N]) -> usize {
        indices
            .iter()
            .zip(shape.iter())
            .fold(0, |acc, (&i, &s)| acc * s + i)
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    /// Returns the number of dimensions of the array.
    pub fn ndim(&self) -> usize {
        N
    }

    /// Returns the total number of elements in the array.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns `true` if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the length of the array along a given axis.
    ///
    /// # Arguments
    ///
    /// * `axis`: The axis to get the length of.
    pub fn len_axis(&self, axis: usize) -> usize {
        self.shape[axis]
    }

    /// Returns a mutable slice of the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }
}
