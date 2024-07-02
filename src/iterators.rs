//! This module provides iterator implementations for the Dimensional struct.
//! It includes both immutable and mutable iterators, allowing for efficient
//! traversal and modification of Dimensional arrays.

use crate::{storage::DimensionalStorage, Dimensional};
use num_traits::Num;

// TODO: Parallel iterators

/// An iterator over the elements of a Dimensional array.
///
/// This struct is created by the `iter` method on Dimensional. It provides
/// a way to iterate over the elements of the array in row-major order.
pub struct DimensionalIter<'a, T, S, const N: usize>
where
    T: Num + Copy,
    S: DimensionalStorage<T, N>,
{
    dimensional: &'a Dimensional<T, S, N>,
    current_index: [usize; N],
    remaining: usize,
}

impl<'a, T, S, const N: usize> Iterator for DimensionalIter<'a, T, S, N>
where
    T: Num + Copy,
    S: DimensionalStorage<T, N>,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let result = &self.dimensional[self.current_index];

        // TODO: Actually iterate correctly here over an `N`-dimensional array
        // with `N` axes each with a possibly different length.
        // and determine iteration pattern

        // Update the index for the next iteration
        for i in (0..N).rev() {
            self.current_index[i] += 1;
            if self.current_index[i] < self.dimensional.shape()[i] {
                break;
            }
            self.current_index[i] = 0;
        }

        self.remaining -= 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, T, S, const N: usize> ExactSizeIterator for DimensionalIter<'a, T, S, N>
where
    T: Num + Copy,
    S: DimensionalStorage<T, N>,
{
}

/// A mutable iterator over the elements of a Dimensional array.
///
/// This struct is created by the `iter_mut` method on Dimensional. It provides
/// a way to iterate over and modify the elements of the array in row-major order.
pub struct DimensionalIterMut<'a, T, S, const N: usize>
where
    T: Num + Copy,
    S: DimensionalStorage<T, N>,
{
    dimensional: &'a mut Dimensional<T, S, N>,
    current_index: [usize; N],
    remaining: usize,
}

impl<'a, T, S, const N: usize> Iterator for DimensionalIterMut<'a, T, S, N>
where
    T: Num + Copy,
    S: DimensionalStorage<T, N>,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let index = self.current_index;

        // TODO: Actually iterate correctly here over an `N`-dimensional array
        // with `N` axes each with a possibly different length.
        // and determine iteration pattern

        // Update the index for the next iteration
        for i in (0..N).rev() {
            self.current_index[i] += 1;
            if self.current_index[i] < self.dimensional.shape()[i] {
                break;
            }
            self.current_index[i] = 0;
        }

        self.remaining -= 1;

        let linear_index = self.dimensional.ravel_index(&index);
        // TODO: We really don't want to use unsafe rust here
        // SAFETY: This is safe because we're returning a unique reference to each element,
        // and we're iterating over each element only once.
        // But what if we modify the array while iterating?
        // What if the array is deleted while iterating?
        // What if we want to use parallel iterators?
        unsafe { Some(&mut *(&mut self.dimensional.as_mut_slice()[linear_index] as *mut T)) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, S, const N: usize> Dimensional<T, S, N>
where
    T: Num + Copy,
    S: DimensionalStorage<T, N>,
{
    /// Returns an iterator over the elements of the array.
    ///
    /// The iterator yields all items from the array in row-major order.
    ///
    /// # Examples
    ///
    /// ```
    /// use dimensionals::{Dimensional, LinearArrayStorage, vector, matrix};
    ///
    /// let v = vector![1, 2, 3, 4, 5];
    /// let mut iter = v.iter();
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// // ...
    ///
    /// let m = matrix![[1, 2], [3, 4]];
    /// let mut iter = m.iter();
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), Some(&4));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> DimensionalIter<T, S, N> {
        DimensionalIter {
            dimensional: self,
            current_index: [0; N],
            remaining: self.len(),
        }
    }

    /// Returns a mutable iterator over the elements of the array.
    ///
    /// The iterator yields all items from the array in row-major order,
    /// and allows modifying each value.
    ///
    /// # Examples
    ///
    /// ```
    /// use dimensionals::{Dimensional, LinearArrayStorage, vector, matrix};
    ///
    /// let mut v = vector![1, 2, 3, 4, 5];
    /// for elem in v.iter_mut() {
    ///     *elem *= 2;
    /// }
    /// assert_eq!(v, vector![2, 4, 6, 8, 10]);
    ///
    /// let mut m = matrix![[1, 2], [3, 4]];
    /// for elem in m.iter_mut() {
    ///     *elem += 1;
    /// }
    /// assert_eq!(m, matrix![[2, 3], [4, 5]]);
    /// ```
    pub fn iter_mut(&mut self) -> DimensionalIterMut<T, S, N> {
        let len = self.len();
        DimensionalIterMut {
            dimensional: self,
            current_index: [0; N],
            remaining: len,
        }
    }
}

// TODO: Since these are consuming, do they really need a lifetime?

impl<'a, T, S, const N: usize> IntoIterator for &'a Dimensional<T, S, N>
where
    T: Num + Copy,
    S: DimensionalStorage<T, N>,
{
    type Item = &'a T;
    type IntoIter = DimensionalIter<'a, T, S, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, S, const N: usize> IntoIterator for &'a mut Dimensional<T, S, N>
where
    T: Num + Copy,
    S: DimensionalStorage<T, N>,
{
    type Item = &'a mut T;
    type IntoIter = DimensionalIterMut<'a, T, S, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use crate::{matrix, storage::LinearArrayStorage, Dimensional};

    // ... (previous tests remain unchanged)

    #[test]
    fn test_iter_mut_borrow() {
        let mut m = matrix![[1, 2], [3, 4]];
        let mut iter = m.iter_mut();
        assert_eq!(iter.next(), Some(&mut 1));
        assert_eq!(iter.next(), Some(&mut 2));
        assert_eq!(iter.next(), Some(&mut 3));
        assert_eq!(iter.next(), Some(&mut 4));
        assert_eq!(iter.next(), None);
    }
}
