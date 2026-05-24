# Release Notes

## Version 0.3.0

### Highlights

``Quantity.at`` now works on every supported backend. Previously, indexed
functional updates (``x.at[idx].set(...)``, ``.add(...)``, etc.) required a
JAX-backed mantissa and raised ``BackendError`` for NumPy and the other
backends, forcing users to call ``.to_jax()`` first. As of this release the
same expression works directly on ``numpy``, ``jax``, ``cupy``, ``torch``, and
``dask`` arrays, with documented scope limits for ``dask`` and a clean
``BackendError`` for ``ndonnx``.

### New features

- **Multi-backend ``Quantity.at``.** All nine ``.at[idx].<op>(...)``
  operations — ``get``, ``set``, ``add``, ``multiply``/``mul``,
  ``divide``/``div``, ``power``, ``min``, ``max``, ``apply`` — work across
  every supported backend. The unit-tracking semantics are unchanged: the
  same dimension/magnitude checks fire regardless of backend, and the result
  preserves the original mantissa backend (a torch-backed Quantity stays
  torch, a dask-backed Quantity stays dask).
- **Unified scatter dispatch (``saiunit._scatter``).** ``Quantity.at`` and
  ``Quantity.__setitem__`` / ``scatter_add`` / ``scatter_mul`` /
  ``scatter_div`` / ``scatter_max`` / ``scatter_min`` now all route through
  one backend-aware module. This fixes a latent bug where the old in-line
  ``_scatter`` helper silently called ``jnp.asarray`` on cupy, torch, dask,
  and ndonnx mantissas.
- **Emulated ``mode`` and ``fill_value`` on non-JAX backends.** JAX's
  ``mode`` (``'promise_in_bounds'``/``'clip'``/``'drop'``/``'fill'``) and
  ``fill_value`` arguments are emulated on ``numpy``/``cupy``/``torch`` for
  scalar-int and 1D-integer-array indices, so the same code that uses
  ``x.at[20].get(mode='fill', fill_value=-1 * u.mV)`` on JAX now runs
  unchanged on NumPy. For more complex index expressions (slices, boolean
  masks, multi-axis tuples) the native backend semantics apply — these
  indices don't have an out-of-bounds notion to honor.

### Backend-specific notes

- **NumPy / CuPy:** repeated-index updates use ``np.<op>.at`` /
  ``cupy.<op>.at`` so the JAX "all-updates-applied" semantics carry over.
- **PyTorch:** ``add`` uses ``index_put_(accumulate=True)`` and matches JAX
  for repeated indices. ``multiply``/``divide``/``min``/``max``/``apply`` use
  gather + op + scatter and follow last-write-wins semantics for repeated
  indices (one of the rare places torch's native semantics differ from JAX).
- **Dask:** updates are expressed via ``da.where`` over a positional boolean
  mask, so the task graph stays lazy and chunked. Supported index types:
  scalar int, 1D integer array, slice, ellipsis, and same-shape boolean
  mask. Multi-dim fancy integer indexing raises ``NotImplementedError`` —
  call ``.to_numpy()`` first for those cases.
- **ndonnx:** raises ``BackendError`` on every ``.at[...].set/get/...``
  call. ndonnx builds a symbolic ONNX graph; it can't represent a functional
  in-place update cleanly. Call ``.to_numpy()`` to materialize first.

### Known limitations

- ``mode`` emulation is best-effort on non-JAX backends for scalar-int and
  1D-integer-array indices only. Slice/boolean/ellipsis indices ignore
  ``mode`` because they cannot go out of bounds against a same-shape source.
- ``dask`` does not yet support multi-dim fancy integer indexing under
  ``.at``; use ``.to_numpy()`` for those patterns.

### Backend-compatibility fixes

A focused sweep through the rest of the dispatcher closed several silent
JAX-coercion paths and replaced raw ``AttributeError`` failure modes with
``BackendError``:

- ``CustomArray.__setitem__`` no longer assumes JAX-style ``.at[idx].set(...)``;
  it routes through the unified ``saiunit._scatter`` dispatcher so subclasses
  with numpy / cupy / torch / dask mantissas work without a JAX install.
- ``Quantity(mantissa, dtype=...)`` now honors ``dtype`` for cupy / torch /
  dask / ndonnx mantissas. Previously the dtype kwarg was silently ignored
  for those backends.
- ``Quantity.mantissa`` setter coerces the new value to the *existing*
  backend of the Quantity rather than silently lifting torch / cupy / dask /
  ndonnx arrays to JAX.
- ``_check_units_and_collect_values`` (the helper behind
  ``Quantity([q1, q2, ...])``) honors ``set_default_backend()`` /
  ``using_backend()`` instead of always landing on JAX when JAX is installed.
- ``Quantity.strides`` / ``.flat`` / ``.T`` / ``.mT`` raise ``BackendError``
  on lazy mantissas (dask, ndonnx) instead of silently calling
  ``.compute()`` / ``.unwrap_numpy()`` on them.
- ``saiunit.fft.fftn`` / ``ifftn`` / ``rfftn`` / ``irfftn`` no longer force
  Python-scalar / list inputs through ``jnp.asarray`` for the input-ndim
  probe.
- ``saiunit.math`` activation functions (``relu``, ``sigmoid``, ``softplus``,
  ``gelu``, …) now guard with ``require_jax_backend`` and raise a clean
  ``BackendError`` on non-JAX inputs, instead of crashing with
  ``AttributeError: 'NoneType' object has no attribute 'relu'`` when JAX
  isn't installed or when called on a torch / cupy / dask mantissa.
  ``leaky_relu`` is implemented via ``where`` and remains backend-agnostic.

## Version 0.2.2

### Highlights

This release turns saiunit into a multi-backend library. ``Quantity`` can now
wrap NumPy, JAX, CuPy, PyTorch, Dask, and ndonnx arrays, with operations
dispatched via the array API standard (``array_api_compat``). JAX is now an
**optional** dependency — the core package installs and runs on NumPy alone.
This release also lands a 22-issue audit of ``Unit`` naming, display, hashing,
and parsing, and tightens correctness in ``Quantity``'s hash/equality and
tracer interactions.

### Breaking Changes

- **JAX is no longer a mandatory dependency.** Install with ``pip install
  saiunit`` for the NumPy-only build, or ``pip install "saiunit[jax]"`` (or
  ``[cpu]``/``[cuda12]``/``[cuda13]``/``[tpu]``) to enable JAX. Without JAX,
  the default backend auto-selects ``"numpy"`` and JAX-only modules
  (``saiunit.autograd``, ``saiunit.lax``, ``saiunit.sparse``) raise
  ``BackendError`` on import with an install hint.
- ``Quantity(np.ndarray(...))`` now preserves the mantissa as ``np.ndarray``
  instead of implicitly converting to ``jax.Array``. Call ``.to_jax()`` or
  use ``with using_backend("jax"):`` to restore the previous behaviour.
- ``Quantity`` is now **unhashable** (``__hash__ = None``). ``Quantity.__eq__``
  returns an array, so any hash implementation would violate the hash/eq
  invariant. This matches NumPy/JAX array semantics across all backends.
- ``Unit(dim, base=B)`` with ``B != 10`` now raises ``ValueError``. Previously
  the non-decimal base was silently folded into ``factor`` and then forgotten.
  Encode non-decimal scales directly in ``factor``.
- ``Unit("symbol", scale=..., factor=..., name=..., ...)`` now raises
  ``TypeError`` when extra construction kwargs are combined with a string
  ``dim``. Previously the extras were silently dropped.
- ``Unit(dim, factor=...)`` now raises ``ValueError`` for NaN or infinite
  factors instead of constructing a poisoned unit that propagated NaN through
  all subsequent arithmetic.
- ``Quantity + Unit`` and ``Unit + Quantity`` now both raise ``TypeError``
  symmetrically. Previously the former silently promoted the bare Unit to
  ``Quantity(1, unit)`` while the latter raised — making ``q + metre`` and
  ``metre + q`` produce different results.

### New Features

#### Multi-backend support

- **NumPy backend.** ``Quantity`` can wrap ``np.ndarray`` directly; math,
  linalg, and fft operations dispatch through ``array_api_compat``.
- **CuPy backend** (``saiunit[cupy]``). GPU arrays via the CuPy array API,
  with ``Quantity.to_cupy(device=None)`` for conversion.
- **PyTorch backend** (``saiunit[torch]``). Torch tensors with
  ``Quantity.to_torch(device=None, dtype=None)``.
- **Dask backend** (``saiunit[dask]``). Lazy chunked arrays via
  ``Quantity.to_dask(chunks='auto')``; ``__repr__`` and other materializing
  methods are lazy-safe and guarded to avoid implicit computation.
- **ndonnx backend** (``saiunit[ndonnx]``). Symbolic ONNX-graph arrays via
  ``Quantity.to_ndonnx()`` for export-oriented workflows.
- ``saiunit[all]`` meta-extra installs every optional backend.

#### Backend control API

- ``Quantity.backend`` property reporting the active backend name
  (``'numpy'``, ``'jax'``, ``'cupy'``, ``'torch'``, ``'dask'``, ``'ndonnx'``).
- ``Quantity.to_numpy()`` / ``.to_jax()`` / ``.to_cupy()`` / ``.to_torch()``
  / ``.to_dask()`` / ``.to_ndonnx()`` conversion methods.
- ``saiunit.set_default_backend()``, ``saiunit.get_default_backend()``, and
  ``saiunit.using_backend()`` context manager for controlling the default
  backend when input backend is ambiguous (Python scalars, list inputs).
- ``saiunit.is_numpy_array()``, ``is_jax_array()``, ``is_cupy_array()``,
  ``is_torch_array()``, ``is_dask_array()``, and ``is_ndonnx_array()``
  detector helpers.
- ``Quantity.__array_ufunc__`` so calls like ``np.sin(quantity)`` and
  ``np.add(q1, q2)`` preserve units instead of stripping them.
- ``saiunit.BackendError`` exception type (subclass of ``TypeError``) raised
  by JAX-only modules (``saiunit.lax``, ``saiunit.sparse``,
  ``saiunit.autograd``, and the custom ``exprel`` primitive) when given a
  non-JAX backend, with a clear ``"call .to_jax() first"`` (or install)
  hint.

#### Unit additions

- SI-prefixed kelvin variants (``ykelvin`` through ``Ykelvin``, including
  ``mK``, ``uK``, ``nK``, ``kK``, ``MK``), bringing kelvin in line with
  every other base unit. ``parse_unit("mK")`` now succeeds.

### Improvements

#### Unit display, hashing, and parsing (22-issue audit)

- **Compound-exponent normalization.** ``metre * metre2`` now displays as
  ``m^3`` instead of ``m * m^2``; display parts are merged after compound
  arithmetic.
- **Eager standard-name resolution.** Compound results from
  ``__mul__``/``__div__``/``__pow__`` now write the resolved standard-unit
  name into ``self._name``/``self._dispname`` at construction time, keeping
  ``unit.name``, ``unit.dispname``, and ``str(unit)`` in sync. Display
  parts survive ``copy``, ``deepcopy``, and pickle round-trips.
- **Hash/eq invariant restored.** ``Unit.__hash__`` no longer folds in
  spelling fields (``name``/``dispname``), so aliases like ``metre`` and
  ``meter`` hash and compare consistently — sets and dicts no longer hold
  silent duplicates. ``Unit.__eq__`` now compares
  ``(dim, scale, base, factor, _canonical_str())``, with a new
  ``is_unit_equal_math()`` helper for name-agnostic math equivalence;
  ``has_same_unit()`` delegates to it.
- **Built-in units win over user aliases.** ``add_standard_unit`` now
  stamps a monotonic registration index; built-ins (registered during
  import) outrank user-added aliases, so registering
  ``Unit(metre.dim, name="aaaa_meter")`` no longer hijacks the canonical
  metre display. Alphabetical order remains the tie-breaker for
  same-batch registrations.
- **Named-dimensionless identity preserved.** ``radian * UNITLESS``,
  ``radian ** 1``, and ``UNITLESS * radian`` now render as ``rad`` instead
  of collapsing to bare ``Unit("1")``; ``radian * radian`` → ``rad^2``;
  ``radian / radian`` → ``1`` (genuine cancellation).
- **Parser improvements.** ``parse_unit`` now accepts parenthesised
  sub-expressions in numerators (``(m * s) / A``), and numeric-base tokens
  like ``"2^3"`` are encoded as ``Unit(DIMENSIONLESS, factor=8.0)`` rather
  than raising. Anonymous ``Unit`` instances (constructed without a
  ``name``) now render with parser-compatible grammar, so
  ``parse_unit(repr(u))`` round-trips for any anonymous unit.

#### Core correctness

- **Tracer safety.** ``Quantity.update_mantissa()`` and ``__setitem__`` now
  raise a clear ``RuntimeError`` when called on a traced mantissa, instead
  of producing silently wrong state.
- **JIT-safe reductions.** New ``_reduction_count_from_shape`` and
  ``_is_concrete_zero`` helpers ensure reductions stay JIT-safe, and the
  "0 is dimensionless" convention only applies to concrete zeros — never
  tracers.
- **Foreign-tensor rejection.** ``require_jax_backend`` now names the
  offending backend and rejects foreign tensors (CuPy, Torch, Dask,
  ndonnx) at JAX-only entry points.

### Documentation

- New multi-backend user-guide sections covering NumPy, CuPy, PyTorch,
  Dask (lazy semantics), and ndonnx (symbolic execution).
- Per-backend Jupyter notebooks demonstrating end-to-end workflows.
- Updated installation docs covering the new extras layout (``jax``,
  ``cpu``, ``cuda12``, ``cuda13``, ``tpu``, ``cupy``, ``torch``, ``dask``,
  ``ndonnx``, ``all``).
- Sphinx ``conf.py`` gates brainx header injection on ``TARGET=brainunit``;
  docs build/deploy split (deploy on release, build-only on main push).

### Dependencies

- New mandatory runtime dependencies: ``array_api_compat>=1.9`` and
  ``opt_einsum``.
- ``jax`` moved to the ``[jax]`` optional extra. Install with
  ``pip install "saiunit[jax]"`` (or ``[cpu]`` / ``[cuda12]`` / ``[cuda13]``
  / ``[tpu]``) to enable JAX-backed features.
- New optional extras: ``[cupy]``, ``[torch]``, ``[dask]``, ``[ndonnx]``,
  and the ``[all]`` meta-extra.

### Internal / CI

- New ``_jax_compat`` module centralizes safely-degradable JAX symbols
  (sentinel classes, no-op decorators, NumPy fallbacks for
  ``dtypes``/``result_type``/tree-ops) when JAX is missing.
- New ``test_no_jax`` CI job installs without JAX and verifies imports
  plus ``BackendError`` gates; ``_no_jax_test.py`` smoke suite added.
- CI now installs CPU-only PyTorch wheels (``--index-url
  https://download.pytorch.org/whl/cpu``) to avoid multi-GB CUDA pulls on
  GPU-less runners.
- CI matrix extended to install ``cupy``/``torch``/``dask``/``ndonnx`` on
  all platforms so the new backends are exercised in regression tests.
- ``brainunit`` legacy package now re-exports the new backend API
  (``BackendError``, ``is_unit_equal_math``, ``set_default_backend``,
  ``using_backend``, backend detectors).
- Dependency bumps: ``actions/checkout`` 4→6, ``actions/setup-python``
  5→6, ``actions/download-artifact`` 5→8, ``appleboy/ssh-action``
  1.2.0→1.2.5, ``appleboy/scp-action`` 0.1.7→1.0.0, ``sphinx`` ≥8.1.3,
  ``sphinx-book-theme`` ≥1.1, ``sphinx-copybutton`` ≥0.5.2,
  ``jupyter-sphinx`` ≥0.5.3.

## Version 0.2.1

### Breaking Changes

- **Removed `BlockCSR` and `BlockELL` sparse classes**: These experimental
  block-sparse matrix implementations (along with their benchmarks and tests)
  have been removed. Users should use `COO`, `CSR`, or `CSC` instead.
- **`SparseMatrix` no longer inherits from `JAXSparse`**: The base class is now
  standalone, with its own `shape`, `size`, `ndim`, `T`, `block_until_ready`,
  `tree_flatten`, `tree_unflatten`, `transpose`, and `todense` interface.

### Improvements

- **Forward-compatible `**kwargs` across all wrapped functions**: All unit-aware
  wrapper functions in `math`, `lax`, `linalg`, and `fft` modules now accept
  and forward `**kwargs` to the underlying JAX functions. This ensures
  compatibility with new keyword arguments added in future JAX releases without
  requiring saiunit updates.
  - `saiunit.math`: `concatenate`, `stack`, `vstack`, `hstack`, `dstack`,
    `column_stack`, `block`, `append`, `split`, `array_split`, `dsplit`,
    `hsplit`, `vsplit`, `tile`, `repeat`, `sort`, `argsort`, `unique`,
    `searchsorted`, `where`, `clip`, `interp`, and many more
  - `saiunit.lax`: `cond`, `switch`, `scan`, `while_loop`, `fori_loop`,
    `sort`, `top_k`, `broadcasted_iota`, `concatenate`, `conv`, `pad`,
    `slice`, `dynamic_slice`, `gather`, `scatter`, and many more
  - `saiunit.linalg`: `svd`, `cholesky`, `eig`, `eigh`, `eigvalsh`, `qr`,
    `lu`, `solve`, `det`, `norm`, `matrix_power`, `cross`, `tensordot`, etc.
  - `saiunit.fft`: `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`,
    `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`, `fftshift`, `ifftshift`
- **Standalone `SparseMatrix` base class**: Decoupled from
  `jax.experimental.sparse.JAXSparse` to reduce external coupling and provide a
  self-contained sparse matrix interface with properties (`size`, `ndim`, `nse`,
  `dtype`) and methods (`__repr__`, `__len__`, `block_until_ready`).
- **Improved validation in sparse classes**: Replaced `assert` statements with
  descriptive `ValueError` exceptions in `COO.with_data`, `CSR.with_data`, and
  `CSC.with_data` for shape, dtype, and unit mismatches.
- **Broader sparse type checking**: `isinstance` checks in binary and matmul
  operations now accept both `JAXSparse` and `SparseMatrix`, ensuring correct
  behavior after the inheritance change.
- **Fixed `CSC.tree_unflatten` error message**: Corrected the error message from
  `"CSR.tree_unflatten"` to `"CSC.tree_unflatten"`.
- **Explicit attribute assignment in `tree_unflatten`**: `CSR` and `CSC` now set
  `shape`, `indices`, and `indptr` explicitly instead of using
  `__dict__.update`, improving clarity and avoiding potential issues.

---

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



