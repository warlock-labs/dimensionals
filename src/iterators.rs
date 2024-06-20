//! This module provides iterator implementations for the Dimensional struct.
//! It includes both immutable and mutable iterators, allowing for efficient
//! traversal and modification of Dimensional arrays.

use crate::{Dimensional, DimensionalStorage};
use num::Num;
use std::marker::PhantomData;

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
    dimensional: *mut Dimensional<T, S, N>,
    current_index: [usize; N],
    remaining: usize,
    _phantom: PhantomData<&'a mut Dimensional<T, S, N>>,
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

        // Update the index for the next iteration
        for i in (0..N).rev() {
            self.current_index[i] += 1;
            if self.current_index[i] < unsafe { (*self.dimensional).shape()[i] } {
                break;
            }
            self.current_index[i] = 0;
        }

        self.remaining -= 1;

        // SAFETY: We know that `dimensional` is valid for the lifetime of the iterator,
        // and we're the only mutable reference to it.
        unsafe {
            let dimensional = &mut *self.dimensional;
            let linear_index = Dimensional::<T, S, N>::ravel_index(&index, &dimensional.shape());
            Some(&mut dimensional.as_mut_slice()[linear_index])
        }
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
    /// use dimensionals::{vector, matrix};
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
    /// use dimensionals::{vector, matrix};
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
        DimensionalIterMut {
            dimensional: self as *mut Self,
            current_index: [0; N],
            remaining: self.len(),
            _phantom: PhantomData,
        }
    }
}

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
    use crate::{linear_storage::LinearArrayStorage, Dimensional, matrix, vector};

    #[test]
    fn test_iter() {
        let v = vector![1, 2, 3, 4, 5];
        let mut iter = v.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), Some(&5));
        assert_eq!(iter.next(), None);

        let m = matrix![[1, 2], [3, 4]];
        let mut iter = m.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_mut() {
        let mut v = vector![1, 2, 3, 4, 5];
        for elem in v.iter_mut() {
            *elem *= 2;
        }
        assert_eq!(v, vector![2, 4, 6, 8, 10]);

        let mut m = matrix![[1, 2], [3, 4]];
        for elem in m.iter_mut() {
            *elem += 1;
        }
        assert_eq!(m, matrix![[2, 3], [4, 5]]);
    }

    #[test]
    fn test_into_iter() {
        let v = vector![1, 2, 3, 4, 5];
        let sum: i32 = v.into_iter().sum();
        assert_eq!(sum, 15);

        let m = matrix![[1, 2], [3, 4]];
        let product: i32 = m.into_iter().product();
        assert_eq!(product, 24);
    }
}
