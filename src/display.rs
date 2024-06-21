use std::fmt;
use num_traits::Num;
use crate::{Dimensional, DimensionalStorage};

impl<T: Num + Copy + fmt::Display, S, const N: usize> fmt::Display for Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match N {
            1 => self.fmt_1d(f),
            2 => self.fmt_2d(f),
            _ => self.fmt_nd(f),
        }
    }
}

impl<T: Num + Copy + fmt::Display, S, const N: usize> Dimensional<T, S, N>
where
    S: DimensionalStorage<T, N>,
{
    fn fmt_1d(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, val) in self.as_slice().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", val)?;
        }
        write!(f, "]")
    }

    fn fmt_2d(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        for i in 0..self.shape()[0] {
            write!(f, " [")?;
            for j in 0..self.shape()[1] {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self[[i, j]])?;
            }
            writeln!(f, "],")?;
        }
        write!(f, "]")
    }

    fn fmt_nd(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dimensional {}D array: ", N)?;
        write!(f, "shape {:?}, ", self.shape())?;
        write!(f, "data [")?;
        for (i, val) in self.as_slice().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if i >= 3 && i < self.len() - 3 {
                write!(f, "...")?;
                break;
            }
            write!(f, "{}", val)?;
        }
        if self.len() > 6 {
            write!(f, ", ..., {}", self.as_slice().last().unwrap())?;
        }
        write!(f, "]")
    }
}