# Release Notes

## Version 0.1.4

- Added numerically stable `exprel` function with comprehensive test coverage
- Updated lax array creation to use `jax.numpy` for zero initialization
- Updated CI JAX version for improved compatibility
- Improved code quality and removed redundant tests

## Version 0.1.3

- Compatible with `jax>=0.8.2`


## Version 0.1.2

- Renamed ``CustomArray.value`` to ``CustomArray.data`` for API consistency
- Streamlined math unwrapping for improved performance
- Refactored math module for better Quantity/CustomArray support
- Added dtype aliases for convenience
- Fixed matplotlib convert for zero-sized inputs
- Registered Array class as a PyTree node for JAX compatibility
- Added support for Python 3.14
- Updated CI configuration and dependencies

## Version 0.1.1

- Fixed dimension and unit checks in convert method to handle empty input
- Bug fixes and stability improvements

## Version 0.1.0

- Introduced ``CustomArray`` class and integrated it across saiunit modules
- Added ``Array`` class inheriting from ``CustomArray``
- Added ``maybe_custom_array`` and ``maybe_custom_array_tree`` utilities for type checking
- Added comprehensive unit tests for CustomArray integration
- Improved support for einops, activation functions, FFT, and linear algebra operations
- Enhanced Celsius conversion functions with CustomArray compatibility
- Added tutorial and documentation for CustomArray

## Version 0.0.19

- Added ``CustomArray`` class as the foundation for custom array types
- Refactored activation functions to support CustomArray
- Added ``gather`` function
- Refined ``math``, ``linalg``, ``autograd``, and constants modules
- Enabled Quantity hashing with ``__hash__`` method
- Fixed interp function and adjusted unit handling in output
- Moved metadata from setup.py to pyproject.toml

## Version 0.0.2

The new version of ``saiunit``, which separates the ``Quantity`` into the ``mantissa`` and the ``unit``.

This design is more flexible and allows for more complex operations, enabling to represent the very
large or very small values.

## Version 0.0.1

The first release of the project.



