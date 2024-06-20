- [ ] Use safe rust in iterators
- [ * ] Use safe rust in indexing
- [ ] Add tensor macro for creating tensors
- [ ] Remove the need for phantom data markers
- [ ] Move shape data to type-system such that it is known at compile time
- [ * ] Support common arithmetic operations
- [ * ] Use safe rust in arithmetic operations
- [ ] Support reshaping
- [ ] Support appending
- [ ] Support removing
- [ ] Linear algebra functions
- [ ] Support for common statistical functions
- [ ] Support for geometric functions like Brownian motion
- [ ] Support for GPU offloading
- [ ] Support for SIMD
- [ ] Matrix multiplication
- [ ] Support Apache Arrow or safetensors storage backend?
- [ ] Support Pola.rs integration
- [ ] Support plotly-rs integration
- [ ] Support argmin-rs integration


The goal is for this library to have an intuitive interface for working with
n dimensional/tensor data, while interfacing with libraries like Polars, and
by proxy plotly-rs. It should have linear algebra functionality, and integration
with argmin-rs for optimization. It should have common statistical functions
such that it is able to work in a statistical computing environment.

The signature for creating a Dimensional is ugly in that N must be speficied twice, a generalized 
builder pattern, or obviation of the need to store twice would be ideal.

In lib.rs:

The TODO for the tensor macro is still present. This is not a bug, but a reminder for future implementation.


In core.rs:

The PhantomData<T> in the Dimensional struct is still present but unused. You might consider removing it if it's not needed for type invariance.


In iterators.rs:

The mutable iterator still uses unsafe code. While this is not necessarily a bug, it's worth noting that it introduces potential safety risks if not handled carefully.


In storage.rs:

No significant issues found. The implementation looks correct and well-tested.



Overall, the code appears to be functioning as intended. The main points to consider are:

Removing unused PhantomData if it's not needed.
Potentially finding a safe alternative to the unsafe code in the mutable iterator, if possible.
Implementing the tensor macro in the future, as noted in the TODO.

These are not critical issues, but rather areas for potential future improvement. The library as it stands should work correctly for its intended purpose.