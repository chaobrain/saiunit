# Spec: Consolidate type aliases under `saiunit/_typing.py`

**Date:** 2026-05-24
**Branch:** multi-backend
**Scope:** Internal refactor; no public API breakage.

## Goal

Make `saiunit/_typing.py` the single internal source of truth for low-level
type aliases (`Array`, `ArrayLike`, `ScalarOrArrayLike`, `DTypeLike`, `Shape`,
`Axis`, `Axes`, `PyTree`). Replace every direct reference to `jax.Array`,
`jax.typing.DTypeLike`, etc. across saiunit's production code with imports
from `saiunit._typing` (or the re-exporting public `saiunit.typing`).

## Motivation

Today these aliases live in `saiunit/_jax_compat.py` (a runtime-compat
module). Direct references to `jax.Array` / `jax.typing.DTypeLike` are
scattered across ~30 files. This makes the no-JAX code path fragile —
`jax.typing.DTypeLike` blows up the moment JAX is absent — and ties typing
to the runtime-shim module. Pulling type aliases into a dedicated module
gives saiunit one place to evolve them and lets the runtime shim shrink to
its actual job (JAX vs. NumPy backend).

## Architecture

```
saiunit/_typing.py        ← NEW: internal source for basic type aliases
saiunit/_jax_compat.py    ← drops type aliases (Array/ArrayLike/etc.) from __all__
saiunit/typing.py         ← re-exports aliases from _typing + keeps Quantity-aware types
```

**Import rules:**
- `_typing.py` depends only on `numpy` and the JAX-presence probe; no Quantity dependency.
- `_jax_compat.py` no longer publishes type aliases. It may import from `_typing` for its own internal use.
- `typing.py` re-exports every alias from `_typing` so `from saiunit.typing import ArrayLike` continues to work for external users.
- All saiunit production code imports basic aliases from `saiunit._typing` (private path is appropriate for internal callers).

## `saiunit/_typing.py` contents

```python
"""Internal type aliases used across saiunit.

Centralized here so the no-JAX path stays import-safe and so saiunit core
modules don't depend on the runtime shim (`_jax_compat`) just to grab a
type alias. External users should import these from `saiunit.typing`, which
re-exports them.
"""
from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np

try:
    import jax as _jax            # noqa: F401  (presence probe only)
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


if _HAS_JAX:
    import jax as _jax
    Array = _jax.Array
    ArrayLike = _jax.Array | np.ndarray | np.number | np.bool_
    ScalarOrArrayLike = ArrayLike | bool | int | float | complex
    DTypeLike = _jax.typing.DTypeLike
else:
    class _JaxSentinel:
        __slots__ = ()
        def __init__(self, *args, **kwargs):
            raise RuntimeError("JAX is not installed")
    Array = type("Array", (_JaxSentinel,), {})
    ArrayLike = np.ndarray | np.number | np.bool_                       # type: ignore[misc]
    ScalarOrArrayLike = ArrayLike | bool | int | float | complex        # type: ignore[misc]
    DTypeLike = Any                                                     # type: ignore[misc]


Shape = Sequence[int]
Axis = int
Axes = Union[int, Sequence[int]]
PyTree = Any  # See https://docs.jax.dev/en/latest/pytrees.html — an opaque tree of leaves.


__all__ = [
    "Array",
    "ArrayLike",
    "ScalarOrArrayLike",
    "DTypeLike",
    "Shape",
    "Axis",
    "Axes",
    "PyTree",
]
```

The sentinel class for `Array` mirrors the existing pattern in
`_jax_compat.py` so `isinstance(x, Array)` keeps returning `False` (the
right answer) when JAX is absent.

## `saiunit/_jax_compat.py` changes

- Remove `Array`, `ArrayLike`, `ScalarOrArrayLike`, `DTypeLike` definitions.
- Drop them from `__all__`.
- Keep `Tracer`, `ShapedArray`, `ShapeDtypeStruct`, `DynamicJaxprTracer`, `TracerArrayConversionError`, `register_pytree_node_class`, `ensure_compile_time_eval`, `result_type`, `canonicalize_dtype`, `tree_map`, `tree_structure`, `tree_flatten`, `tree`, `device_put`, `devices`, `require_jax`, `HAS_JAX`, `jax`, `jnp`.
- If anything inside `_jax_compat.py` itself needs `Array`/`ArrayLike`/etc., import them from `saiunit._typing` at the top.

## `saiunit/typing.py` changes

- Drop the line `from ._jax_compat import Array as _JaxArray` and `from ._jax_compat import ArrayLike, ScalarOrArrayLike`.
- Replace with `from ._typing import Array, ArrayLike, ScalarOrArrayLike, DTypeLike, Shape, Axis, Axes, PyTree`.
- Add the new aliases to `__all__` so they're exported from `saiunit.typing`.
- `QuantityLike` continues to use the now-local `Array` alias.

## Production-code migration

Replace every direct reference across the following files:

**`ArrayLike` / `ScalarOrArrayLike` imports (from `_jax_compat` → from `_typing`):**
- `saiunit/custom_array.py`
- `saiunit/fft/_fft_change_unit.py`
- `saiunit/fft/_fft_keep_unit.py`
- `saiunit/lax/_lax_accept_unitless.py`
- `saiunit/lax/_lax_array_creation.py`
- `saiunit/lax/_lax_change_unit.py`
- `saiunit/lax/_lax_keep_unit.py`
- `saiunit/lax/_lax_linalg.py`
- `saiunit/lax/_lax_remove_unit.py`
- `saiunit/lax/_misc.py`
- `saiunit/linalg/_linalg_change_unit.py`
- `saiunit/linalg/_linalg_keep_unit.py`
- `saiunit/linalg/_linalg_remove_unit.py`
- `saiunit/math/_activation.py`
- `saiunit/math/_einops.py`
- `saiunit/math/_exprel.py`
- `saiunit/math/_fun_accept_unitless.py`
- `saiunit/math/_fun_array_creation.py`
- `saiunit/math/_fun_change_unit.py`
- `saiunit/math/_fun_keep_unit.py`
- `saiunit/math/_fun_remove_unit.py`
- `saiunit/math/_misc.py`
- `saiunit/sparse/_coo.py`
- `saiunit/sparse/_csr.py`
- `saiunit/_base_getters.py`
- `saiunit/_base_quantity.py`
- `saiunit/_sparse_base.py`

**`jax.typing.DTypeLike` → `DTypeLike` (imported from `saiunit._typing`):**
- `saiunit/fft/_fft_change_unit.py`
- `saiunit/lax/_lax_array_creation.py`
- `saiunit/lax/_lax_change_unit.py`
- `saiunit/lax/_lax_keep_unit.py`
- `saiunit/math/_einops.py`
- `saiunit/math/_fun_accept_unitless.py`
- `saiunit/math/_fun_array_creation.py`

**`jax.Array` → `Array` (imported from `saiunit._typing`):**
- `saiunit/lax/_lax_accept_unitless.py`
- `saiunit/lax/_lax_array_creation.py`
- `saiunit/lax/_lax_change_unit.py`
- `saiunit/lax/_lax_keep_unit.py`
- `saiunit/lax/_lax_remove_unit.py`
- `saiunit/math/_einops.py`
- (plus any other file that uses `jax.Array` in annotations or docstrings)

**Local `Shape = Sequence[int]` definitions:**
- `saiunit/fft/_fft_change_unit.py` (line 61) — remove the local definition; import `Shape` from `saiunit._typing` instead.

**Docstring replacements:**
Replace docstring references that currently say `jax.Array` or `jax.typing.DTypeLike` with `Array` / `DTypeLike` to keep the docs consistent with the new aliases. Docstring text in `Returns` / `Parameters` sections counts.

## Test scope

- Test files (`_test.py`) are **not** migrated. They may continue to reference `jax.Array` / `jax.typing.DTypeLike` directly.
- Existing tests must continue to pass: `saiunit/typing_test.py`, `saiunit/_no_jax_test.py`, `saiunit/_jax_guard_test.py`, `saiunit/_backend_parametrize_test.py`, plus the broader suite.
- Add a small block of new tests in `saiunit/typing_test.py` covering the newly-public aliases (`Array`, `DTypeLike`, `Shape`, `Axis`, `Axes`, `PyTree`) — at minimum: importable from `saiunit.typing`, are the expected types, and `isinstance` for `Array` still behaves correctly with/without JAX.

## Acceptance criteria

1. `from saiunit.typing import Array, ArrayLike, ScalarOrArrayLike, DTypeLike, Shape, Axis, Axes, PyTree` works.
2. `grep -rn "jax\.typing\|jax\.Array" saiunit/ --include="*.py"` returns no matches in production code (test files are out of scope).
3. `grep -rn "from saiunit\._jax_compat import .*ArrayLike\|from saiunit\._jax_compat import .*DTypeLike\|from saiunit\._jax_compat import .*ScalarOrArrayLike\|from \._jax_compat import .*ArrayLike\|from \._jax_compat import .*DTypeLike\|from \._jax_compat import .*ScalarOrArrayLike" saiunit/ --include="*.py" | grep -v "_test.py"` returns no matches.
4. `saiunit/_jax_compat.py`'s `__all__` no longer contains `Array`, `ArrayLike`, `ScalarOrArrayLike`, `DTypeLike`.
5. Full test suite passes (under both JAX-installed and no-JAX conditions, if separate jobs exist).
6. `mypy` type-check job (the `type_check` CI job from commit `b32ee04`) remains clean.

## Non-goals

- Top-level re-exports (`saiunit.ArrayLike`) are **not** added — aliases stay under `saiunit.typing`.
- No changes to `QuantityLike`, `UnitLike`, `DimensionLike`, `PhysicalType`, `validate_units`, or the pre-built physical-type aliases (`LENGTH`, `MASS`, …).
- No changes to test files.
- No version bump.
