```markdown
[![License](https://img.shields.io/crates/l/dimensionals)](https://choosealicense.com/licenses/mit/)
[![Crates.io](https://img.shields.io/crates/v/dimensionals)](https://crates.io/crates/dimensionals)
[![Docs](https://img.shields.io/crates/v/dimensionals?color=blue&label=docs)](https://docs.rs/dimensionals/)
![CI](https://github.com/warlock-labs/dimensionals/actions/workflows/CI.yml/badge.svg)

# dimensionals

Dimensionals is a Rust library for working with n-dimensional data. It provides a flexible and efficient multidimensional array implementation with a generic storage backend.

## Motivations

The key motivations behind Dimensionals are:

- A concise, idiomatic Rust API that leverages Rust's type system and ownership model.
- High performance through efficient memory layout and cache-friendly traversals.
- Extensibility via a generic storage backend, allowing for custom storage strategies.
- A foundation for a generic compute pipeline that can target GPUs and utilize SIMD instructions.

## Features

- Generic over element type and number of dimensions
- Efficient storage using a linear memory layout
- Index and mutable index operations
- Arithmetic operations for 1D and 2D arrays
- Convenient macros for array creation
- Extensible with custom storage backends

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
dimensionals = "0.1.0"
```

Then, use the crate in your Rust code:

```rust
use dimensionals::{matrix, Dimensional, LinearArrayStorage};

fn main() {
    let m: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = matrix![
        [1, 2, 3],
        [4, 5, 6]
    ];
    assert_eq!(m[[0, 0]], 1);
    assert_eq!(m[[1, 1]], 5);
}
```

For more examples and usage details, see the [API documentation](https://docs.rs/dimensionals).

## Roadmap

The following features and improvements are planned for future releases:

- SIMD support for improved performance on CPU.
- GPU support for offloading computations to compatible GPUs.
- Comprehensive scalar, vector, matrix, and tensor algebra operations.
- Reshaping and appending operations for easy data manipulation.
- Additional storage backends for optimized memory usage in various scenarios.
- Integration with popular Rust scientific computing libraries.

## Performance

The `LinearArrayStorage` backend stores elements in a contiguous `Vec<T>` and computes element indices on the fly. This provides good cache locality for traversals, but may not be optimal for sparse or very high dimensional arrays.

Alternative storage backends can be implemented by defining a type that implements the `DimensionalStorage` trait.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests on the [GitHub repository](https://github.com/warlock-labs/dimensionals).

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

## Acknowledgements

This project is inspired by and builds upon ideas from several existing multidimensional array libraries in Rust and other languages.

## Contact

Warlock Labs - [https://github.com/warlock-labs](https://github.com/warlock-labs)

Project Link: [https://github.com/warlock-labs/dimensionals](https://github.com/warlock-labs/dimensionals)
```