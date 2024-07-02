//! This module implements the Display trait for the Dimensional struct,
//! allowing for pretty-printing of multidimensional arrays.

use crate::{Dimensional, DimensionalStorage};
use num_traits::Num;
use std::fmt;

impl<T, S, const N: usize> fmt::Display for Dimensional<T, S, N>
where
    T: Num + Copy + fmt::Display,
    S: DimensionalStorage<T, N>,
{
    /// Formats the Dimensional array for display.
    ///
    /// The format differs based on the number of dimensions:
    /// - 1D arrays are displayed as a single row
    /// - 2D arrays are displayed as a matrix
    /// - Higher dimensional arrays are displayed in a compact format
    ///
    /// # Arguments
    ///
    /// * `f` - A mutable reference to the Formatter
    ///
    /// # Returns
    ///
    /// A fmt::Result indicating whether the operation was successful
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match N {
            1 => self.fmt_1d(f),
            2 => self.fmt_2d(f),
            _ => self.fmt_nd(f),
        }
    }
}

impl<T, S, const N: usize> Dimensional<T, S, N>
where
    T: Num + Copy + fmt::Display,
    S: DimensionalStorage<T, N>,
{
    /// Formats a 1D array for display.
    ///
    /// # Arguments
    ///
    /// * `f` - A mutable reference to the Formatter
    ///
    /// # Returns
    ///
    /// A fmt::Result indicating whether the operation was successful
    fn fmt_1d(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        let mut iter = self.as_slice().iter().peekable();
        while let Some(val) = iter.next() {
            // Check if a precision is specified in the formatter
            if let Some(precision) = f.precision() {
                write!(f, "{:.1$}", val, precision)?;
            } else {
                write!(f, "{}", val)?;
            }
            if iter.peek().is_some() {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }

    /// Formats a 2D array for display as a matrix.
    ///
    /// # Arguments
    ///
    /// * `f` - A mutable reference to the Formatter
    ///
    /// # Returns
    ///
    /// A fmt::Result indicating whether the operation was successful
    fn fmt_2d(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        assert_eq!(N, 2, "fmt_2d should only be called for 2D arrays");
        let shape = self.shape();
        writeln!(f, "[")?;
        for i in 0..shape[0] {
            write!(f, " [")?;
            for j in 0..shape[1] {
                let mut index_array = [0; N];
                index_array[0] = i;
                index_array[1] = j;
                let index = self.ravel_index(&index_array);
                // Check if a precision is specified in the formatter
                if let Some(precision) = f.precision() {
                    write!(f, "{:.1$}", self.as_slice()[index], precision)?;
                } else {
                    write!(f, "{}", self.as_slice()[index])?;
                }
                if j < shape[1] - 1 {
                    write!(f, ", ")?;
                }
            }
            if i < shape[0] - 1 {
                writeln!(f, "],")?;
            } else {
                writeln!(f, "]")?;
            }
        }
        write!(f, "]")
    }

    /// Formats a higher dimensional array for display in a compact format.
    ///
    /// # Arguments
    ///
    /// * `f` - A mutable reference to the Formatter
    ///
    /// # Returns
    ///
    /// A fmt::Result indicating whether the operation was successful
    fn fmt_nd(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}D array: ", N)?;
        write!(f, "shape {:?}, ", self.shape())?;
        write!(f, "data [")?;

        let slice = self.as_slice();
        let len = slice.len();
        let display_count = if len > 6 { 3 } else { len };

        for (i, val) in slice.iter().take(display_count).enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if let Some(precision) = f.precision() {
                write!(f, "{:.1$}", val, precision)?;
            } else {
                write!(f, "{}", val)?;
            }
        }

        if len > 6 {
            write!(f, ", ..., ")?;
            if let Some(precision) = f.precision() {
                write!(f, "{:.1$}", slice.last().ok_or(fmt::Error)?, precision)?;
            } else {
                write!(f, "{}", slice.last().ok_or(fmt::Error)?)?;
            }
        }

        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearArrayStorage;

    #[test]
    fn test_display_1d() {
        let array: Dimensional<i32, LinearArrayStorage<i32, 1>, 1> =
            Dimensional::from_fn([5], |[i]| i as i32);
        assert_eq!(format!("{}", array), "[0, 1, 2, 3, 4]");

        // Test empty 1D array
        let empty: Dimensional<i32, LinearArrayStorage<i32, 1>, 1> = Dimensional::zeros([0]);
        assert_eq!(format!("{}", empty), "[]");

        // Test 1D array with single element
        let single: Dimensional<i32, LinearArrayStorage<i32, 1>, 1> = Dimensional::ones([1]);
        assert_eq!(format!("{}", single), "[1]");

        // Test large 1D array
        let large: Dimensional<i32, LinearArrayStorage<i32, 1>, 1> =
            Dimensional::from_fn([10], |[i]| i as i32);
        assert_eq!(format!("{}", large), "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
    }

    #[test]
    fn test_display_2d() {
        let array: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> =
            Dimensional::from_fn([2, 3], |[i, j]| (i * 3 + j) as i32);
        assert_eq!(format!("{}", array), "[\n [0, 1, 2],\n [3, 4, 5]\n]");

        // Test 2D array with single row
        let single_row: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> =
            Dimensional::from_fn([1, 3], |[_, j]| j as i32);
        assert_eq!(format!("{}", single_row), "[\n [0, 1, 2]\n]");

        // Test 2D array with single column
        let single_column: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> =
            Dimensional::from_fn([3, 1], |[i, _]| i as i32);
        assert_eq!(format!("{}", single_column), "[\n [0],\n [1],\n [2]\n]");

        // Test empty 2D array
        let empty: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = Dimensional::zeros([0, 0]);
        assert_eq!(format!("{}", empty), "[\n]");
    }

    #[test]
    fn test_display_3d() {
        let array: Dimensional<i32, LinearArrayStorage<i32, 3>, 3> =
            Dimensional::from_fn([2, 2, 2], |[i, j, k]| (i * 4 + j * 2 + k) as i32);
        assert_eq!(
            format!("{}", array),
            "3D array: shape [2, 2, 2], data [0, 1, 2, ..., 7]"
        );

        // Test 3D array with small size
        let small: Dimensional<i32, LinearArrayStorage<i32, 3>, 3> =
            Dimensional::from_fn([1, 2, 3], |[i, j, k]| (i * 6 + j * 3 + k) as i32);
        assert_eq!(
            format!("{}", small),
            "3D array: shape [1, 2, 3], data [0, 1, 2, 3, 4, 5]"
        );

        // Test empty 3D array
        let empty: Dimensional<i32, LinearArrayStorage<i32, 3>, 3> = Dimensional::zeros([0, 0, 0]);
        assert_eq!(format!("{}", empty), "3D array: shape [0, 0, 0], data []");
    }

    #[test]
    fn test_display_float() {
        let array: Dimensional<f64, LinearArrayStorage<f64, 2>, 2> =
            Dimensional::from_fn([2, 2], |[i, j]| (i * 2 + j) as f64 + 0.5);
        assert_eq!(format!("{}", array), "[\n [0.5, 1.5],\n [2.5, 3.5]\n]");

        // Test float precision for 1D array
        let precise_1d: Dimensional<f64, LinearArrayStorage<f64, 1>, 1> =
            Dimensional::from_fn([3], |[i]| i as f64 / 3.0);
        assert_eq!(format!("{:.2}", precise_1d), "[0.00, 0.33, 0.67]");

        // Test float precision for 2D array
        let precise_2d: Dimensional<f64, LinearArrayStorage<f64, 2>, 2> =
            Dimensional::from_fn([2, 2], |[i, j]| (i + j) as f64 / 3.0);
        assert_eq!(
            format!("{:.2}", precise_2d),
            "[\n [0.00, 0.33],\n [0.33, 0.67]\n]"
        );

        // TODO(Fix this test case, maybe issue with ravel_index)
        // Test float precision for 3D array
        let precise_3d: Dimensional<f64, LinearArrayStorage<f64, 3>, 3> =
            Dimensional::from_fn([3, 3, 3], |[i, j, k]| (i + j + k) as f64 / 3.0);
        assert_eq!(
            format!("{:.2}", precise_3d),
            "3D array: shape [3, 3, 3], data [0.00, 0.33, 0.67, ..., 2.00]"
        );
    }

    #[test]
    fn test_display_large_dimensions() {
        let array: Dimensional<i32, LinearArrayStorage<i32, 4>, 4> =
            Dimensional::from_fn([2, 2, 2, 2], |[i, j, k, l]| {
                (i * 8 + j * 4 + k * 2 + l) as i32
            });
        assert_eq!(
            format!("{}", array),
            "4D array: shape [2, 2, 2, 2], data [0, 1, 2, ..., 15]"
        );

        // Test 5D array
        let array_5d: Dimensional<i32, LinearArrayStorage<i32, 5>, 5> =
            Dimensional::from_fn([2, 2, 2, 2, 2], |[i, j, k, l, m]| {
                (i * 16 + j * 8 + k * 4 + l * 2 + m) as i32
            });
        assert_eq!(
            format!("{}", array_5d),
            "5D array: shape [2, 2, 2, 2, 2], data [0, 1, 2, ..., 31]"
        );
    }
}
