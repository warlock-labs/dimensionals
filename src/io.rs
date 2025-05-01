//! IO functionality for importing and exporting dimensional arrays
//!
//! This module provides functions for reading arrays from files and writing
//! arrays to files in various formats.

use crate::{Dimensional, DimensionalStorage};
use num_traits::Num;
use std::error::Error;
use std::fmt::Debug;
use std::path::Path;
use std::str::FromStr;

#[cfg(feature = "csv")]
pub mod csv {
    use super::*;
    // To refer to the csv crate instead of current mod
    use ::csv::{ReaderBuilder, WriterBuilder};
    use std::{fs::File, sync::OnceLock};

    /// Options for CSV reading operations
    #[derive(Debug, Clone)]
    pub struct CsvReadOptions {
        /// Whether the CSV file has headers
        pub has_headers: bool,
        /// The delimiter used in the CSV file
        pub delimiter: u8,
    }

    impl Default for CsvReadOptions {
        fn default() -> Self {
            Self {
                has_headers: false,
                delimiter: b',',
            }
        }
    }

    /// Options for CSV writing operations
    #[derive(Debug, Clone)]
    pub struct CsvWriteOptions {
        /// Whether to write headers to the CSV file
        pub write_headers: bool,
        /// The delimiter to use in the CSV file
        pub delimiter: u8,
    }

    impl Default for CsvWriteOptions {
        fn default() -> Self {
            Self {
                write_headers: false,
                delimiter: b',',
            }
        }
    }

    /// Reads a CSV file into a 2D Dimensional array.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the CSV file
    /// * `options` - Optional CSV reading options
    ///
    /// # Returns
    ///
    /// A 2D Dimensional array containing the data from the CSV file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the data cannot be parsed
    /// /// # Examples
    ///
    /// ```rust
    /// use dimensionals::{matrix, Dimensional, LinearArrayStorage};
    /// use dimensionals::io::csv::{from_csv, CsvReadOptions};
    /// use std::fs::File;
    /// use std::io::Write;
    /// use tempfile::tempdir;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let dir = tempdir()?;
    /// let file_path = dir.path().join("data.csv");
    ///
    /// // Create a sample CSV file
    /// let mut file = File::create(&file_path)?;
    /// writeln!(file, "1,2,3")?;
    /// writeln!(file, "4,5,6")?;
    /// file.flush()?;
    ///
    /// // Read the CSV file into a 2D Dimensional array
    /// let options = CsvReadOptions {
    ///     has_headers: false,
    ///     delimiter: b',',
    /// };
    /// let array: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> =
    ///     from_csv(&file_path, Some(options))?;
    ///
    /// let expected = matrix![[1, 2, 3], [4, 5, 6]];
    /// assert_eq!(array, expected);
    /// # Ok(())
    /// # }
    pub fn from_csv<T, S>(
        path: impl AsRef<Path>,
        options: Option<CsvReadOptions>,
    ) -> Result<Dimensional<T, S, 2>, Box<dyn Error>>
    where
        T: Num + FromStr + Copy + Debug,
        T::Err: Error + 'static,
        S: DimensionalStorage<T, 2>,
    {
        let file = File::open(path)?;
        let options = options.unwrap_or_default();

        let mut reader = ReaderBuilder::new()
            .has_headers(options.has_headers)
            .delimiter(options.delimiter)
            .from_reader(file);

        let mut rows = Vec::new();

        let col_count: OnceLock<usize> = OnceLock::new();
        for (i, result) in reader.records().enumerate() {
            let record = result?;
            let row: Vec<T> = record
                .iter()
                .map(|field| {
                    field
                        .parse::<T>()
                        .map_err(|e| Box::new(e) as Box<dyn Error>)
                })
                .collect::<Result<Vec<T>, _>>()?;

            let current_col_count = row.len();
            let expected_col_count = *col_count.get_or_init(|| current_col_count);

            if expected_col_count != current_col_count {
                return Err(format!("Row {} has inconsistent number of columns", i).into());
            }

            rows.push(row);
        }

        if rows.is_empty() {
            return Err("CSV file is empty or contains no data rows".into());
        }

        let cols = rows[0].len();

        let shape = [rows.len(), cols];
        let data: Vec<T> = rows.into_iter().flatten().collect();

        Ok(Dimensional::new(shape, S::from_vec(shape, data)))
    }

    /// Writes a 2D Dimensional array to a CSV file.
    ///
    /// # Arguments
    ///
    /// * `array` - The 2D Dimensional array to write
    /// * `path` - The path to the CSV file
    /// * `options` - Optional CSV writing options
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written
    ///     /// Writes a 2D Dimensional array to a CSV file.
    ///
    /// # Arguments
    ///
    /// * `array` - The 2D Dimensional array to write
    /// * `path` - The path to the CSV file
    /// * `options` - Optional CSV writing options
    ///
    /// # Examples
    ///
    /// ```rust
    /// use dimensionals::{matrix, Dimensional, LinearArrayStorage};
    /// use dimensionals::io::csv::{to_csv, CsvWriteOptions};
    /// use std::fs;
    /// use tempfile::tempdir;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let dir = tempdir()?;
    /// let file_path = dir.path().join("output.csv");
    ///
    /// let array: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> =
    ///     matrix![[10, 20], [30, 40]];
    ///
    /// let options = CsvWriteOptions {
    ///     write_headers: false,
    ///     delimiter: b',',
    /// };
    ///
    /// to_csv(&array, &file_path, Some(options))?;
    ///
    /// // Verify the content of the created CSV file
    /// let content = fs::read_to_string(&file_path)?;
    /// assert_eq!(content, "10,20\n30,40\n");
    /// # Ok(())
    /// # }
    pub fn to_csv<T, S>(
        array: &Dimensional<T, S, 2>,
        path: impl AsRef<Path>,
        options: Option<CsvWriteOptions>,
    ) -> Result<(), Box<dyn Error>>
    where
        T: Num + Copy + Debug + ToString,
        S: DimensionalStorage<T, 2>,
    {
        let file = File::create(path)?;
        let options = options.unwrap_or_default();

        let mut writer = WriterBuilder::new()
            .has_headers(options.write_headers)
            .delimiter(options.delimiter)
            .from_writer(file);

        let shape = array.shape();
        let rows = shape[0];
        let cols = shape[1];

        for row_idx in 0..rows {
            let mut record = Vec::with_capacity(cols);
            for col_idx in 0..cols {
                record.push(array[[row_idx, col_idx]].to_string());
            }
            writer.write_record(&record)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Reads a CSV file into a 1D Dimensional array.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the CSV file
    /// * `options` - Optional CSV reading options
    ///
    /// # Returns
    ///
    /// A 1D Dimensional array containing the data from the CSV file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the data cannot be parsed
    /// /// # Examples
    ///
    /// ```rust
    /// use dimensionals::{vector, Dimensional, LinearArrayStorage};
    /// use dimensionals::io::csv::{from_csv_1d, CsvReadOptions};
    /// use std::fs::File;
    /// use std::io::Write;
    /// use tempfile::tempdir;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let dir = tempdir()?;
    /// let file_path = dir.path().join("data_1d.csv");
    ///
    /// // Create a sample CSV file (single row)
    /// let mut file = File::create(&file_path)?;
    /// writeln!(file, "1.1,2.2,3.3")?;
    /// file.flush()?;
    ///
    /// // Read the CSV file into a 1D Dimensional array
    /// let options = CsvReadOptions {
    ///     has_headers: false,
    ///     delimiter: b',',
    /// };
    /// let array: Dimensional<f64, LinearArrayStorage<f64, 1>, 1> =
    ///     from_csv_1d(&file_path, Some(options))?;
    ///
    /// let expected = vector![1.1, 2.2, 3.3];
    /// assert_eq!(array, expected);
    /// # Ok(())
    /// # }
    pub fn from_csv_1d<T, S>(
        path: impl AsRef<Path>,
        options: Option<CsvReadOptions>,
    ) -> Result<Dimensional<T, S, 1>, Box<dyn Error>>
    where
        T: Num + FromStr + Copy + Debug,
        T::Err: Error + 'static,
        S: DimensionalStorage<T, 1>,
    {
        let file = File::open(path)?;
        let options = options.unwrap_or_default();

        let mut reader = ReaderBuilder::new()
            .has_headers(options.has_headers)
            .delimiter(options.delimiter)
            .from_reader(file);

        let mut data = Vec::new();
        for result in reader.records() {
            let record = result?;
            for field in record.iter() {
                let value = field
                    .parse::<T>()
                    .map_err(|e| Box::new(e) as Box<dyn Error>)?;
                data.push(value);
            }
        }

        let shape = [data.len()];
        Ok(Dimensional::new(shape, S::from_vec(shape, data)))
    }

    /// Writes a 1D Dimensional array to a CSV file.
    ///
    /// # Arguments
    ///
    /// * `array` - The 1D Dimensional array to write
    /// * `path` - The path to the CSV file
    /// * `options` - Optional CSV writing options
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written
    ///
    /// # Examples
    ///
    /// ```rust
    /// use dimensionals::{vector, Dimensional, LinearArrayStorage};
    /// use dimensionals::io::csv::{to_csv_1d, CsvWriteOptions};
    /// use std::fs;
    /// use tempfile::tempdir;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let dir = tempdir()?;
    /// let file_path = dir.path().join("output_1d.csv");
    ///
    /// let array: Dimensional<f64, LinearArrayStorage<f64, 1>, 1> =
    ///     vector![10.5, 20.1, 30.9];
    ///
    /// let options = CsvWriteOptions {
    ///     write_headers: false,
    ///     delimiter: b',',
    /// };
    ///
    /// to_csv_1d(&array, &file_path, Some(options))?;
    ///
    /// // Verify the content of the created CSV file
    /// let content = fs::read_to_string(&file_path)?;
    /// assert_eq!(content, "10.5,20.1,30.9\n");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    pub fn to_csv_1d<T, S>(
        array: &Dimensional<T, S, 1>,
        path: impl AsRef<Path>,
        options: Option<CsvWriteOptions>,
    ) -> Result<(), Box<dyn Error>>
    where
        T: Num + Copy + Debug + ToString,
        S: DimensionalStorage<T, 1>,
    {
        let file = File::create(path)?;
        let options = options.unwrap_or_default();

        let mut writer = WriterBuilder::new()
            .has_headers(options.write_headers)
            .delimiter(options.delimiter)
            .from_writer(file);

        let data = array.as_slice();
        let record: Vec<String> = data.iter().map(|val| val.to_string()).collect();

        writer.write_record(&record)?;
        writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{matrix, vector, Dimensional, LinearArrayStorage};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    #[cfg(feature = "csv")]
    fn test_csv_2d_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        use crate::io::csv;

        let dir = tempdir()?;
        let file_path = dir.path().join("test_2d.csv");

        let m: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = matrix![[1, 2, 3], [4, 5, 6]];

        csv::to_csv(&m, &file_path, None)?;

        let m2 = csv::from_csv::<i32, LinearArrayStorage<i32, 2>>(&file_path, None)?;

        assert_eq!(m, m2);

        Ok(())
    }

    #[test]
    #[cfg(feature = "csv")]
    fn test_csv_1d_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        use crate::io::csv;

        let dir = tempdir()?;
        let file_path = dir.path().join("test_1d.csv");

        let v: Dimensional<f64, LinearArrayStorage<f64, 1>, 1> = vector![1.1, 2.2, 3.3, 4.4, 5.5];

        csv::to_csv_1d(&v, &file_path, None)?;

        let v2 = csv::from_csv_1d::<f64, LinearArrayStorage<f64, 1>>(&file_path, None)?;

        assert_eq!(v, v2);

        Ok(())
    }

    #[test]
    #[cfg(feature = "csv")]
    fn test_csv_with_options() -> Result<(), Box<dyn std::error::Error>> {
        use crate::io::csv::{self, CsvReadOptions, CsvWriteOptions};

        let dir = tempdir()?;
        let file_path = dir.path().join("test_options.csv");

        let m: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = matrix![[1, 2, 3], [4, 5, 6]];

        let write_options = CsvWriteOptions {
            delimiter: b';',
            write_headers: false,
        };

        csv::to_csv(&m, &file_path, Some(write_options))?;

        let content = fs::read_to_string(&file_path)?;
        assert!(content.contains(';'));

        let read_options = CsvReadOptions {
            delimiter: b';',
            has_headers: false,
        };

        let m2 = csv::from_csv::<i32, LinearArrayStorage<i32, 2>>(&file_path, Some(read_options))?;

        assert_eq!(m, m2);

        Ok(())
    }

    #[test]
    #[cfg(feature = "csv")]
    fn test_csv_inconsistent_columns() -> Result<(), Box<dyn std::error::Error>> {
        use crate::io::csv;
        use std::io::Write;

        let dir = tempdir()?;
        let file_path = dir.path().join("inconsistent.csv");

        let mut file = fs::File::create(&file_path)?;
        writeln!(file, "1,2,3")?;
        writeln!(file, "4,5")?; // Inconsistent row
        writeln!(file, "6,7,8")?;
        file.flush()?;

        let result = csv::from_csv::<i32, LinearArrayStorage<i32, 2>>(&file_path, None);

        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(
            error_message.contains("inconsistent number of columns") ||
             error_message.contains("length") || // Broader check for CSV crate errors
             error_message.contains("record") // Another common term in CSV errors
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "csv")]
    fn test_csv_empty_file() -> Result<(), Box<dyn std::error::Error>> {
        use crate::io::csv;

        let dir = tempdir()?;
        let file_path = dir.path().join("empty.csv");

        fs::File::create(&file_path)?;

        let result_2d = csv::from_csv::<i32, LinearArrayStorage<i32, 2>>(&file_path, None);
        assert!(result_2d.is_err());
        assert!(result_2d
            .unwrap_err()
            .to_string()
            .contains("CSV file is empty or contains no data rows"));

        let result_1d = csv::from_csv_1d::<i32, LinearArrayStorage<i32, 1>>(&file_path, None)?;
        assert!(result_1d.is_empty());

        Ok(())
    }

    #[test]
    #[cfg(feature = "csv")]
    fn test_csv_only_headers() -> Result<(), Box<dyn std::error::Error>> {
        use crate::io::csv::{self, CsvReadOptions};
        use std::io::Write;

        let dir = tempdir()?;
        let file_path = dir.path().join("only_headers.csv");

        let mut file = fs::File::create(&file_path)?;
        writeln!(file, "col1,col2,col3")?;
        file.flush()?;

        let options = CsvReadOptions {
            has_headers: true,
            ..Default::default()
        };

        let result = csv::from_csv::<i32, LinearArrayStorage<i32, 2>>(&file_path, Some(options));

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("CSV file is empty or contains no data rows"));

        Ok(())
    }

    #[test]
    #[cfg(feature = "csv")]
    fn test_csv_write_empty_2d() -> Result<(), Box<dyn std::error::Error>> {
        use crate::io::csv;

        let dir = tempdir()?;
        let file_path = dir.path().join("empty_write_2d.csv");

        let m: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = Dimensional::zeros([0, 3]);

        csv::to_csv(&m, &file_path, None)?;

        let content = fs::read_to_string(&file_path)?;

        assert!(content.is_empty());

        Ok(())
    }

    #[test]
    #[cfg(feature = "csv")]
    fn test_csv_write_empty_1d() -> Result<(), Box<dyn std::error::Error>> {
        use crate::io::csv;

        let dir = tempdir()?;
        let file_path = dir.path().join("empty_write_1d.csv");

        let v: Dimensional<f64, LinearArrayStorage<f64, 1>, 1> = Dimensional::zeros([0]);

        csv::to_csv_1d(&v, &file_path, None)?;

        let content = fs::read_to_string(&file_path)?;

        assert_eq!(content, "\"\"\n");

        Ok(())
    }
}
