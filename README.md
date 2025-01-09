# Dimensionals

[![License](https://img.shields.io/crates/l/dimensionals)](https://choosealicense.com/licenses/mit/)
[![Crates.io](https://img.shields.io/crates/v/dimensionals)](https://crates.io/crates/dimensionals)
[![Docs](https://img.shields.io/crates/v/dimensionals?color=blue&label=docs)](https://docs.rs/dimensionals/)
![CI](https://github.com/warlock-labs/dimensionals/actions/workflows/CI.yml/badge.svg)

Dimensionals is a Rust library for working with n-dimensional data. It provides a flexible and efficient multidimensional array implementation with a generic storage backend over generic number types.

## Features

- Generic over element type `T` (implementing `Num` and `Copy`), number of dimensions `N`, and storage backend `S`
- Support for Scalar (0D), Vector (1D), Matrix (2D), and Tensor (N>2 D) types
- Efficient `LinearArrayStorage` backend with support for row-major and column-major layouts
- Iterators (immutable and mutable) for efficient traversal
- Indexing and slicing operations
- Arithmetic operations (element-wise and scalar) with operator overloading
- Convenient macros for vector and matrix creation (`vector!` and `matrix!`)

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
dimensionals = "0.1.0"
```

Here's a basic example of creating and using a matrix:

```rust
use dimensionals::{matrix, Dimensional, LinearArrayStorage};

fn main() {
    let m: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = matrix![
        [1, 2, 3],
        [4, 5, 6]
    ];
    assert_eq!(m[[0, 0]], 1);
    assert_eq!(m[[1, 1]], 5);

    // Element-wise addition
    let m2 = &m + &m;
    assert_eq!(m2[[0, 0]], 2);
    assert_eq!(m2[[1, 1]], 10);

    // Scalar multiplication
    let m3 = &m * 2;
    assert_eq!(m3[[0, 0]], 2);
    assert_eq!(m3[[1, 1]], 10);

    // Iteration
    for &value in m.iter() {
        println!("{}", value);
    }

    // Matrix multiplication
    let m4: Dimensional<i32, LinearArrayStorage<i32, 2>, 2> = matrix![
        [7, 8],
        [9, 10],
        [11, 12]
    ];
    let product = m.dot(&m4);
    assert_eq!(product[[0, 0]], 58);
}
```

For more examples and usage details, see the [API documentation](https://docs.rs/dimensionals).

## Core Concepts

- **Element type `T`**: The type of data stored in the array (must implement `Num` and `Copy`).
- **Storage backend `S`**: The underlying storage mechanism for the array (must implement `DimensionalStorage`).
- **Number of dimensions `N`**: The dimensionality of the array (const generic parameter).

## Performance

The `LinearArrayStorage` backend stores elements in a contiguous `Vec<T>` and supports both row-major and column-major layouts. This provides good cache locality for traversals. The storage computes strides for efficient indexing.

## Roadmap

The following features and improvements are planned for future releases:

- [x] Basic N-dimensional array
- [x] Basic indexing
- [x] Basic iterators
- [x] Basic arithmetic operations
- [x] Basic slicing
- [x] Use safe rust in indexing
- [x] Support common arithmetic operations
- [x] Use safe rust in arithmetic operations
- [ ] Move shape data to type-system for compile-time known dimensions
- [ ] Matrix multiplication
- [ ] Use safe Rust in iterators (currently uses unsafe code)
- [ ] Add tensor macro for creating higher-dimensional arrays
- [ ] Remove the need for phantom data markers
- [ ] Support reshaping, appending, and removing operations
- [ ] Implement comprehensive linear algebra functions
- [ ] Add support for common statistical functions
- [ ] Implement geometric functions like Brownian motion
- [ ] Add support for GPU offloading
- [ ] Implement SIMD optimizations
- [ ] Support Apache Arrow or safetensors storage backend
- [ ] Integrate with Polars, plotly-rs, and argmin-rs
- [ ] Add parallel processing support with Rayon
- [ ] Implement feature flags for optional functionality
- [ ] Support no_std environments
- [ ] Add WebAssembly and WebGPU support
- [ ] Implement support for SVM targets

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests on the [GitHub repository](https://github.com/warlock-labs/dimensionals).

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

## Contact

Warlock Labs - [https://github.com/warlock-labs](https://github.com/warlock-labs)

Project Link: [https://github.com/warlock-labs/dimensionals](https://github.com/warlock-labs/dimensionals)
