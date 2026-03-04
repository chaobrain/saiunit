# Release Notes

## Version 0.2.0

### Highlights

This release introduces unit-aware type annotations, string-based unit parsing,
enhanced Matplotlib integration, and a comprehensive overhaul of error handling
and unit display semantics.

### New Features

- **Unit-aware type annotations** (`saiunit.typing`): Added `QuantityLike`,
  `UnitLike`, and related type aliases using `typing.Annotated` (PEP 593) for
  expressing physical-unit constraints in Python type hints.
- **String-based unit parsing**: `Quantity` now accepts string unit
  specifications during initialization (e.g., `Quantity(1.0, "meter")`).
- **Matplotlib `QuantityConverter`**: Full integration with Matplotlib's unit
  conversion framework, enabling direct plotting of `Quantity` objects with
  automatic axis labeling and unit display.
- **`unit_to_scale` parameter for activation functions**: Activation functions
  now accept an optional `unit_to_scale` parameter for explicit unit conversion.
- **`symmetrize_input` parameter**: Added to `cholesky`, `eigvalsh`, and `svd`
  for optional input symmetrization before decomposition.
- **`amu` alias**: Added `amu` as an alias for `atomic_mass` / `u` / `um_u`.
- **`concrete_or_error` shim**: Compatibility shim for `jax.core.concrete_or_error`
  to maintain support across JAX versions.
- **FFT `shape` parameter**: `_calculate_fftn_dimension` now supports a `shape`
  parameter for explicit output shape specification.

### Improvements

- **Unified unit display format**: Refactored `display_in_unit` and unit
  representation methods for consistent, human-readable output. Normalized
  exponent representation and improved formatting for dimensionless units.
  Removed the `python_code` parameter in favor of unified display.
- **Error handling overhaul**: Replaced `assert` statements with proper
  `TypeError` and `ValueError` exceptions across the codebase (`Quantity`,
  einops, lax, FFT, and unit-related functions) for clearer, more informative
  error messages.
- **Updated physical constants** (CODATA 2018): `atomic_mass`, `electron_volt`,
  `light_year`, `atmosphere`, `acre`, `fluid_ounce_imp`, `Btu_th`,
  `speed_of_sound`, and `IMF` now use more accurate conversion factors.
- **Improved display names**: `survey_foot` / `survey_mile` now show
  `"US survey ft"` / `"US survey mi"`; `gallon_imp` shows `"imp gal"`;
  `fluid_ounce_imp` shows `"imp fl oz"`; `month` shows `"mon"`;
  `Btu_IT` shows `"Btu"`.
- **Unit preference scoring**: Standard unit retrieval now uses preference
  scoring for aliases, ensuring compound unit representations maintain grouping.
- **Removed `iscompound` attribute** from the `Unit` class, simplifying the
  internal representation.
- **Cumulative product functions**: Enhanced handling for unit-aware quantities.
- **Jacobian and vector gradient**: Refactored to use `_argnums_partial` for
  improved multi-argument handling; added tests for list-style `argnums`.
- **Sparse matrix improvements**: Refactored unit handling in `BlockCSR` and
  `BlockELL`; added `transpose` method.
- **Module organization**: Restructured imports across `__init__.py` and
  submodules for consistency; defined `__all__` for decorators and constants.
- **Type annotations modernized**: Updated `Union[A, B]` to `A | B` syntax
  throughout the codebase.
- **Documentation**: Updated docstrings with examples for unit-aware functions;
  added installation instructions for CUDA and TPU; refreshed all Jupyter
  notebook examples.
- **Copyright updated** to BrainX Ecosystem Limited.

### Bug Fixes

- Fixed issue #17 (unit display edge case).
- Fixed cumulative product operations for unit-aware quantities.
- Fixed error handling in tests: corrected expected exception types from
  `AssertionError` to `TypeError` for invalid input cases.

### Internal / CI

- Refactored version handling and updated main entry point structure.
- Removed deprecated JAX version testing from CI configuration.
- Added `brainstate` to optional testing dependencies.
- Updated CI configurations to include BrainUnit installation steps.
- Removed `sys.version_info` checks for Python version compatibility.

---

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



