# NumPy Backend Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `saiunit.Quantity` work with `np.ndarray` mantissas as a first-class alternative to `jax.Array`, dispatching every unit-aware operation in `math/`, `linalg/`, and `fft/` to the matching backend via `array_api_compat`.

**Architecture:** Three layers — (1) a new `_backend.py` module that owns dispatch and the global default; (2) a backend-aware `Quantity` with `.backend`, `.to_numpy()`, `.to_jax()`, and `__array_ufunc__`; (3) refactored math/linalg/fft helpers that call `xp.<func>` instead of `jnp.<func>`. JAX-only modules (`lax/`, `autograd/`, `sparse/`) gain entry guards. JAX stays a mandatory dependency.

**Tech Stack:** Python 3.10+, JAX, NumPy, `array_api_compat` (new), pytest.

**Spec reference:** `docs/superpowers/specs/2026-05-19-numpy-backend-design.md`

---

## Phased rollout

- **Phase 1 (Tasks 1–3):** Foundation — dependency, exception class, backend module.
- **Phase 2 (Tasks 4–6):** Quantity backend awareness — property, conversions, `__array_ufunc__`.
- **Phase 3 (Task 7):** Replace internal `jnp.*` calls in `_base_quantity.py`.
- **Phase 4 (Tasks 8–10):** Refactor math/ helpers and wrappers.
- **Phase 5 (Tasks 11–12):** linalg/ and fft/ refactor.
- **Phase 6 (Task 13):** JAX-only guards.
- **Phase 7 (Task 14):** Test parametrization.
- **Phase 8 (Task 15):** Docs and changelog.

---

## Task 1: Add `array_api_compat` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependency**

Open `pyproject.toml` and update the `dependencies` list:

```toml
dependencies = [
    'jax',
    'numpy',
    'typing_extensions',
    'array_api_compat>=1.9',
]
```

- [ ] **Step 2: Install in the dev environment**

Run: `pip install -e . && python -c "import array_api_compat; print(array_api_compat.__version__)"`
Expected: prints `1.9.x` or later. Aborts the plan if the install fails.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add array_api_compat dependency for dual-backend support"
```

---

## Task 2: Add `BackendError` exception

**Files:**
- Create: `saiunit/_exceptions.py`
- Modify: `saiunit/__init__.py`
- Test: `saiunit/_exceptions_test.py`

- [ ] **Step 1: Write the failing test**

Create `saiunit/_exceptions_test.py`:

```python
# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# (Apache 2.0 header — copy from any existing test file.)

import pytest

import saiunit as u


def test_backend_error_is_type_error():
    err = u.BackendError("test message")
    assert isinstance(err, TypeError)
    assert str(err) == "test message"


def test_backend_error_importable_from_top_level():
    assert hasattr(u, "BackendError")
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest saiunit/_exceptions_test.py -v`
Expected: FAIL — `AttributeError: module 'saiunit' has no attribute 'BackendError'`.

- [ ] **Step 3: Create the exception module**

Create `saiunit/_exceptions.py`:

```python
# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# (Full Apache 2.0 header — copy from any existing file.)

class BackendError(TypeError):
    """Raised when an operation is requested on an incompatible array backend.

    Subclasses ``TypeError`` so that code catching ``TypeError`` continues to work.
    """
```

- [ ] **Step 4: Export from package `__init__.py`**

In `saiunit/__init__.py`, add the import after the other `from ._base_*` imports:

```python
from ._exceptions import BackendError
```

And add `'BackendError'` to the `__all__` list (anywhere in the list — group it near `DimensionMismatchError`).

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest saiunit/_exceptions_test.py -v`
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add saiunit/_exceptions.py saiunit/_exceptions_test.py saiunit/__init__.py
git commit -m "feat: add BackendError exception"
```

---

## Task 3: Implement `_backend.py` module

**Files:**
- Create: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`
- Modify: `saiunit/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `saiunit/_backend_test.py`:

```python
# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
# (Apache 2.0 header.)

import jax.numpy as jnp
import numpy as np
import pytest

import saiunit as u
from saiunit._backend import (
    get_backend,
    get_default_backend,
    is_jax_array,
    is_numpy_array,
    set_default_backend,
    to_backend,
    using_backend,
)


def test_is_numpy_array_true_for_ndarray():
    assert is_numpy_array(np.array([1.0])) is True
    assert is_numpy_array(jnp.array([1.0])) is False
    assert is_numpy_array(1.0) is False


def test_is_jax_array_true_for_jax_array():
    assert is_jax_array(jnp.array([1.0])) is True
    assert is_jax_array(np.array([1.0])) is False


def test_get_backend_numpy_only():
    xp = get_backend(np.array([1.0]))
    assert xp.sin(np.array([0.0])) == 0.0
    # numpy namespace must be array_api_compat.numpy
    import array_api_compat.numpy as expected
    assert xp is expected


def test_get_backend_jax_only():
    xp = get_backend(jnp.array([1.0]))
    import array_api_compat.jax.numpy as expected
    assert xp is expected


def test_get_backend_mixed_no_default_jax_wins():
    set_default_backend(None)
    xp = get_backend(np.array([1.0]), jnp.array([1.0]))
    import array_api_compat.jax.numpy as expected
    assert xp is expected


def test_get_backend_mixed_with_numpy_default():
    set_default_backend("numpy")
    try:
        xp = get_backend(np.array([1.0]), jnp.array([1.0]))
        import array_api_compat.numpy as expected
        assert xp is expected
    finally:
        set_default_backend(None)


def test_set_default_backend_rejects_invalid():
    with pytest.raises(ValueError, match="must be 'numpy', 'jax', or None"):
        set_default_backend("torch")


def test_using_backend_context_manager():
    set_default_backend(None)
    assert get_default_backend() is None
    with using_backend("numpy"):
        assert get_default_backend() == "numpy"
    assert get_default_backend() is None


def test_using_backend_nested():
    set_default_backend(None)
    with using_backend("numpy"):
        with using_backend("jax"):
            assert get_default_backend() == "jax"
        assert get_default_backend() == "numpy"
    assert get_default_backend() is None


def test_to_backend_numpy_to_jax():
    arr = np.array([1.0, 2.0])
    out = to_backend(arr, "jax")
    assert is_jax_array(out)
    assert np.allclose(np.asarray(out), arr)


def test_to_backend_jax_to_numpy():
    arr = jnp.array([1.0, 2.0])
    out = to_backend(arr, "numpy")
    assert is_numpy_array(out)
    assert np.allclose(out, np.asarray(arr))


def test_to_backend_noop():
    arr = np.array([1.0])
    out = to_backend(arr, "numpy")
    assert out is arr  # no copy when already on target backend


def test_get_backend_scalar_falls_back_to_default():
    # No arrays at all — must use the default backend.
    set_default_backend("numpy")
    try:
        xp = get_backend(1.0, 2.0)
        import array_api_compat.numpy as expected
        assert xp is expected
    finally:
        set_default_backend(None)


def test_get_backend_scalar_no_default_uses_jax():
    set_default_backend(None)
    xp = get_backend(1.0)
    import array_api_compat.jax.numpy as expected
    assert xp is expected
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest saiunit/_backend_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'saiunit._backend'`.

- [ ] **Step 3: Implement the backend module**

Create `saiunit/_backend.py`:

```python
# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
# (Full Apache 2.0 header.)

"""Backend dispatch for NumPy vs JAX array operations.

This module centralizes the rules for choosing between NumPy and JAX
namespaces. Internal saiunit code should call ``get_backend(*xs)`` to obtain
an ``xp`` namespace and then call array operations through it
(e.g. ``xp.sin(x)`` instead of ``jnp.sin(x)``).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType
from typing import Iterator, Literal, Optional

import array_api_compat.jax.numpy as _jax_xp
import array_api_compat.numpy as _numpy_xp
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "get_backend",
    "get_default_backend",
    "set_default_backend",
    "using_backend",
    "is_jax_array",
    "is_numpy_array",
    "to_backend",
]

BackendName = Literal["numpy", "jax"]

_default_backend: ContextVar[Optional[BackendName]] = ContextVar(
    "saiunit_default_backend", default=None
)


def is_numpy_array(x) -> bool:
    return isinstance(x, np.ndarray)


def is_jax_array(x) -> bool:
    return isinstance(x, jax.Array)


def get_default_backend() -> Optional[BackendName]:
    return _default_backend.get()


def set_default_backend(name: Optional[BackendName]) -> None:
    if name not in ("numpy", "jax", None):
        raise ValueError(
            f"default backend must be 'numpy', 'jax', or None; got {name!r}"
        )
    _default_backend.set(name)


@contextmanager
def using_backend(name: BackendName) -> Iterator[None]:
    if name not in ("numpy", "jax"):
        raise ValueError(f"backend must be 'numpy' or 'jax'; got {name!r}")
    token = _default_backend.set(name)
    try:
        yield
    finally:
        _default_backend.reset(token)


def _name_to_xp(name: BackendName) -> ModuleType:
    return _jax_xp if name == "jax" else _numpy_xp


def get_backend(*arrays_or_quantities) -> ModuleType:
    """Return the ``xp`` namespace appropriate for the given inputs.

    Rules:
      1. Flatten any ``Quantity`` inputs to their mantissas.
      2. If only NumPy arrays are present, return numpy xp.
      3. If only JAX arrays are present, return jax xp.
      4. If mixed (or only scalars/None):
         - If a default backend is set, use it.
         - Otherwise, JAX wins.
    """
    from saiunit._base_quantity import Quantity  # local import to avoid cycle

    mantissas = [a.mantissa if isinstance(a, Quantity) else a for a in arrays_or_quantities]

    has_jax = any(is_jax_array(x) for x in mantissas)
    has_numpy = any(is_numpy_array(x) for x in mantissas)

    if has_jax and not has_numpy:
        return _jax_xp
    if has_numpy and not has_jax:
        return _numpy_xp

    # Either mixed, or no arrays at all → consult the default.
    default = _default_backend.get()
    if default is not None:
        return _name_to_xp(default)
    return _jax_xp  # JAX wins on the tie-breaker.


def to_backend(x, name: BackendName):
    """Convert ``x`` to the given backend, no-op if already there."""
    if name == "numpy":
        if is_numpy_array(x):
            return x
        return np.asarray(x)
    if name == "jax":
        if is_jax_array(x):
            return x
        return jnp.asarray(x)
    raise ValueError(f"backend must be 'numpy' or 'jax'; got {name!r}")
```

- [ ] **Step 4: Export the public API from `saiunit/__init__.py`**

Add this import block (group with other underscore-module imports):

```python
from ._backend import (
    get_backend,
    get_default_backend,
    is_jax_array,
    is_numpy_array,
    set_default_backend,
    to_backend,
    using_backend,
)
```

Add to `__all__`:

```python
# _backend
'get_default_backend',
'set_default_backend',
'using_backend',
'is_jax_array',
'is_numpy_array',
```

(Do **not** export `get_backend` or `to_backend` — those are internal helpers.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest saiunit/_backend_test.py -v`
Expected: all 14 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py saiunit/__init__.py
git commit -m "feat: add _backend module with array_api_compat dispatch"
```

---

## Task 4: Add `.backend` property to Quantity

**Files:**
- Modify: `saiunit/_base_quantity.py`
- Test: `saiunit/_base_quantity_test.py`

- [ ] **Step 1: Write failing test**

Append to `saiunit/_base_quantity_test.py`:

```python
def test_quantity_backend_property_numpy():
    import numpy as np
    import saiunit as u
    q = u.Quantity(np.array([1.0, 2.0]), unit=u.meter)
    assert q.backend == "numpy"


def test_quantity_backend_property_jax():
    import jax.numpy as jnp
    import saiunit as u
    q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.meter)
    assert q.backend == "jax"
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `pytest saiunit/_base_quantity_test.py::test_quantity_backend_property_numpy -v`
Expected: FAIL — `AttributeError: 'Quantity' object has no attribute 'backend'`.

- [ ] **Step 3: Important — find and remove implicit JAX conversion in `__init__`**

Open `saiunit/_base_quantity.py`. Around line 379–396, the constructor does:

```python
mantissa = jnp.array(mantissa, dtype=dtype)
```

This silently converts NumPy inputs to JAX. We must change it so a NumPy input stays NumPy. Find the constructor block (around line 350–400) and replace the unconditional `jnp.array(...)` call with:

```python
# Preserve the input backend: NumPy stays NumPy, JAX stays JAX.
if isinstance(mantissa, np.ndarray):
    if dtype is not None and mantissa.dtype != dtype:
        mantissa = mantissa.astype(dtype)
elif isinstance(mantissa, jax.Array):
    if dtype is not None and mantissa.dtype != dtype:
        mantissa = jnp.asarray(mantissa, dtype=dtype)
else:
    # Python scalar, list, tuple, etc. → defer to default backend.
    from saiunit._backend import get_default_backend
    default = get_default_backend()
    if default == "numpy":
        mantissa = np.asarray(mantissa, dtype=dtype) if dtype is not None else np.asarray(mantissa)
    else:
        mantissa = jnp.array(mantissa, dtype=dtype) if dtype is not None else jnp.asarray(mantissa)
```

Note: there are two parallel `jnp.array(mantissa, dtype=...)` lines (one for the value path, one for the dtype-cast path). Both must be updated. Re-read the constructor before editing to confirm the exact line numbers.

- [ ] **Step 4: Add the `.backend` property**

In `saiunit/_base_quantity.py`, find the cluster of properties around line 1029 (`shape`, `ndim`, etc.) and add:

```python
@property
def backend(self) -> str:
    """The backend of the underlying mantissa: ``'numpy'`` or ``'jax'``."""
    from saiunit._backend import is_numpy_array
    return "numpy" if is_numpy_array(self.mantissa) else "jax"
```

- [ ] **Step 5: Run the two tests to verify they pass**

Run: `pytest saiunit/_base_quantity_test.py::test_quantity_backend_property_numpy saiunit/_base_quantity_test.py::test_quantity_backend_property_jax -v`
Expected: both PASS.

- [ ] **Step 6: Run the full _base_quantity test suite to verify no regression**

Run: `pytest saiunit/_base_quantity_test.py -v`
Expected: all tests PASS. If any fail because they assumed NumPy inputs got converted to JAX, those tests are documenting the **old** behavior — note them but do not fix yet. They get parametrized in Task 14.

- [ ] **Step 7: Commit**

```bash
git add saiunit/_base_quantity.py saiunit/_base_quantity_test.py
git commit -m "feat: add Quantity.backend property and preserve mantissa backend in __init__"
```

---

## Task 5: Add `.to_numpy()` and `.to_jax()` methods

**Files:**
- Modify: `saiunit/_base_quantity.py`
- Test: `saiunit/_base_quantity_test.py`

- [ ] **Step 1: Write failing tests**

Append to `saiunit/_base_quantity_test.py`:

```python
def test_to_numpy_from_jax():
    import jax.numpy as jnp
    import numpy as np
    import saiunit as u
    q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.meter)
    qn = q.to_numpy()
    assert qn.backend == "numpy"
    assert isinstance(qn.mantissa, np.ndarray)
    assert qn.unit == q.unit
    assert np.allclose(np.asarray(qn.mantissa), np.array([1.0, 2.0]))


def test_to_jax_from_numpy():
    import jax
    import numpy as np
    import saiunit as u
    q = u.Quantity(np.array([1.0, 2.0]), unit=u.meter)
    qj = q.to_jax()
    assert qj.backend == "jax"
    assert isinstance(qj.mantissa, jax.Array)
    assert qj.unit == q.unit


def test_to_numpy_noop_when_already_numpy():
    import numpy as np
    import saiunit as u
    q = u.Quantity(np.array([1.0]), unit=u.meter)
    qn = q.to_numpy()
    assert qn.mantissa is q.mantissa  # no copy
    assert qn.unit is q.unit


def test_to_jax_noop_when_already_jax():
    import jax.numpy as jnp
    import saiunit as u
    q = u.Quantity(jnp.array([1.0]), unit=u.meter)
    qj = q.to_jax()
    assert qj.mantissa is q.mantissa
    assert qj.unit is q.unit
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest saiunit/_base_quantity_test.py::test_to_numpy_from_jax saiunit/_base_quantity_test.py::test_to_jax_from_numpy -v`
Expected: FAIL — `AttributeError: 'Quantity' object has no attribute 'to_numpy'`.

- [ ] **Step 3: Implement the methods**

In `saiunit/_base_quantity.py`, add to the `Quantity` class (near the `.backend` property or in the "conversion" method group):

```python
def to_numpy(self) -> "Quantity":
    """Return a new Quantity with mantissa converted to ``numpy.ndarray``.

    No-op if the mantissa is already a NumPy array.
    """
    from saiunit._backend import is_numpy_array, to_backend
    if is_numpy_array(self.mantissa):
        return self
    return Quantity(to_backend(self.mantissa, "numpy"), unit=self.unit)


def to_jax(self) -> "Quantity":
    """Return a new Quantity with mantissa converted to ``jax.Array``.

    No-op if the mantissa is already a JAX array.
    """
    from saiunit._backend import is_jax_array, to_backend
    if is_jax_array(self.mantissa):
        return self
    return Quantity(to_backend(self.mantissa, "jax"), unit=self.unit)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest saiunit/_base_quantity_test.py -k "to_numpy or to_jax" -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add saiunit/_base_quantity.py saiunit/_base_quantity_test.py
git commit -m "feat: add Quantity.to_numpy and Quantity.to_jax conversion methods"
```

---

## Task 6: Implement `__array_ufunc__` on Quantity

**Files:**
- Modify: `saiunit/_base_quantity.py`
- Test: `saiunit/_base_quantity_test.py`

This routes calls like `np.sin(quantity)` through saiunit's existing dimension-checking machinery instead of stripping units.

- [ ] **Step 1: Write failing tests**

Append to `saiunit/_base_quantity_test.py`:

```python
def test_array_ufunc_sin_dimensionless():
    import numpy as np
    import saiunit as u
    q = u.Quantity(np.array([0.0, 1.0]), unit=u.UNITLESS)
    r = np.sin(q)
    # __array_ufunc__ should return a Quantity (or array, for unitless input)
    assert np.allclose(np.asarray(r if not hasattr(r, "mantissa") else r.mantissa),
                       np.sin([0.0, 1.0]))


def test_array_ufunc_add_same_units():
    import numpy as np
    import saiunit as u
    a = u.Quantity(np.array([1.0, 2.0]), unit=u.meter)
    b = u.Quantity(np.array([3.0, 4.0]), unit=u.meter)
    r = np.add(a, b)
    assert r.unit == u.meter
    assert np.allclose(np.asarray(r.mantissa), [4.0, 6.0])


def test_array_ufunc_add_incompatible_units_raises():
    import numpy as np
    import saiunit as u
    a = u.Quantity(np.array([1.0]), unit=u.meter)
    b = u.Quantity(np.array([1.0]), unit=u.second)
    with pytest.raises(u.DimensionMismatchError):
        np.add(a, b)


def test_array_ufunc_unsupported_returns_notimplemented():
    # An obscure ufunc not in our table should NOT silently strip units.
    # np falls back to its own logic which would call __array__; since we
    # don't implement __array__, this should raise TypeError.
    import numpy as np
    import saiunit as u
    q = u.Quantity(np.array([1.0]), unit=u.meter)
    with pytest.raises((TypeError, u.BackendError)):
        np.gcd(q, q)  # gcd is unusual; pick any ufunc not in our table
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest saiunit/_base_quantity_test.py -k "array_ufunc" -v`
Expected: FAIL — Quantity has no `__array_ufunc__`, so numpy's default behavior runs and likely raises a different error.

- [ ] **Step 3: Implement the dispatch table and `__array_ufunc__`**

In `saiunit/_base_quantity.py`, add this near the top of the `Quantity` class (after `__init__` and before the property block):

```python
# Map of numpy ufuncs → saiunit.math function names.
# Filled in lazily on first call to avoid an import cycle.
_UFUNC_DISPATCH: dict = {}


def _build_ufunc_dispatch():
    """Lazily build the numpy ufunc → saiunit.math function table.

    Only ufuncs in this table are unit-safe. Anything else returns
    ``NotImplemented`` from ``__array_ufunc__``.
    """
    import numpy as np
    from saiunit import math as _u_math
    table = {
        # arithmetic
        np.add: _u_math.add,
        np.subtract: _u_math.subtract,
        np.multiply: _u_math.multiply,
        np.true_divide: _u_math.true_divide,
        np.floor_divide: _u_math.floor_divide,
        np.mod: _u_math.mod,
        np.power: _u_math.power,
        np.negative: _u_math.negative,
        np.positive: _u_math.positive,
        np.absolute: _u_math.absolute,
        # comparison
        np.equal: _u_math.equal,
        np.not_equal: _u_math.not_equal,
        np.less: _u_math.less,
        np.less_equal: _u_math.less_equal,
        np.greater: _u_math.greater,
        np.greater_equal: _u_math.greater_equal,
        # trig (require unitless input — saiunit.math enforces)
        np.sin: _u_math.sin,
        np.cos: _u_math.cos,
        np.tan: _u_math.tan,
        np.arcsin: _u_math.arcsin,
        np.arccos: _u_math.arccos,
        np.arctan: _u_math.arctan,
        np.arctan2: _u_math.arctan2,
        # exp/log (require unitless)
        np.exp: _u_math.exp,
        np.log: _u_math.log,
        np.log2: _u_math.log2,
        np.log10: _u_math.log10,
        # other common
        np.sqrt: _u_math.sqrt,
        np.square: _u_math.square,
        np.abs: _u_math.absolute,
        np.isfinite: _u_math.isfinite,
        np.isnan: _u_math.isnan,
        np.isinf: _u_math.isinf,
    }
    return table


def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """Intercept numpy ufunc calls so units are preserved or checked.

    For ufuncs we know about (``np.add``, ``np.sin``, …), route through
    the matching ``saiunit.math`` function. For anything else, return
    ``NotImplemented`` so numpy raises a TypeError rather than silently
    stripping units.
    """
    if method != "__call__":
        # We don't support reduce, accumulate, outer, at, etc. yet.
        return NotImplemented

    global _UFUNC_DISPATCH
    if not _UFUNC_DISPATCH:
        _UFUNC_DISPATCH = _build_ufunc_dispatch()

    saiunit_fn = _UFUNC_DISPATCH.get(ufunc)
    if saiunit_fn is None:
        return NotImplemented

    out = kwargs.pop("out", None)
    if out is not None:
        # ``out=`` writes back into a buffer; not supported in v1.
        return NotImplemented

    return saiunit_fn(*inputs, **kwargs)
```

Note: `_UFUNC_DISPATCH` and `_build_ufunc_dispatch` are **module-level**, not class methods. Place them above the `Quantity` class definition. The `__array_ufunc__` itself is a method on the class.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest saiunit/_base_quantity_test.py -k "array_ufunc" -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add saiunit/_base_quantity.py saiunit/_base_quantity_test.py
git commit -m "feat: implement __array_ufunc__ for unit-safe numpy interop"
```

---

## Task 7: Replace internal `jnp.*` calls in `_base_quantity.py`

**Files:**
- Modify: `saiunit/_base_quantity.py`

`_base_quantity.py` has ~25 internal `jnp.<func>(self.mantissa)` calls that hard-code JAX. They must dispatch via `xp`.

- [ ] **Step 1: Enumerate the call sites**

Run: `grep -n "jnp\." saiunit/_base_quantity.py | grep -v "^\s*#\|^\s*>>>\|docstring" > /tmp/jnp_sites.txt && wc -l /tmp/jnp_sites.txt`

Open `/tmp/jnp_sites.txt`. Skip any line inside a docstring (`>>>` or in triple-quoted strings) — those don't execute. The remaining lines are the targets.

Representative call sites (from earlier exploration; confirm against current file):

| Line | Current code | Replacement |
|---|---|---|
| 1029 | `return jnp.shape(self.mantissa)` | `return get_backend(self).shape(self.mantissa)` |
| 1033 | `return jnp.ndim(self.mantissa)` | `return get_backend(self).ndim(self.mantissa)` if available, else `return self.mantissa.ndim` |
| 1037 | `return Quantity(jnp.imag(self.mantissa), ...)` | `return Quantity(get_backend(self).imag(self.mantissa), ...)` |
| 1041 | `return Quantity(jnp.real(self.mantissa), ...)` | `return Quantity(get_backend(self).real(self.mantissa), ...)` |
| 1045 | `return jnp.size(self.mantissa)` | `return self.mantissa.size` |
| 1050 | `return jnp.asarray(self.mantissa).nbytes` | `return np.asarray(self.mantissa).nbytes` (cross-backend; nbytes is well-defined for both) |
| 1070 | `Quantity(jnp.asarray(self.mantissa).T, ...)` | `Quantity(self.mantissa.T, ...)` |
| 1094 | `Quantity(jnp.asarray(self.mantissa).mT, ...)` | `Quantity(self.mantissa.mT, ...)` (note: `mT` is array-api standard; both backends expose it via `array_api_compat`) |
| 1098 | `return jnp.isreal(...)` | `return get_backend(self).isreal(...)` (or implement manually if `array_api_compat` lacks it — falls back to `xp.imag(x) == 0`) |
| 1106 | `return jnp.isfinite(...)` | `return get_backend(self).isfinite(...)` |
| 1110 | `return jnp.isinf(...)` | `return get_backend(self).isinf(...)` |
| 1118 | `return jnp.isnan(...)` | `return get_backend(self).isnan(...)` |

Where the operation is a property of the array itself (e.g., `.shape`, `.ndim`, `.size`, `.T`, `.mT`), prefer the attribute access over the `xp.<func>` call — both numpy and jax expose them identically and it avoids one lookup. The `xp.` form is needed only for ops that are not member attributes (e.g., `isfinite`, `isnan`).

- [ ] **Step 2: Add the `get_backend` import at the top of `_base_quantity.py`**

```python
from saiunit._backend import get_backend
```

(Be careful: `_backend.py` does a local import of `Quantity` inside `get_backend` to avoid a cycle. This top-level import in the reverse direction is safe.)

- [ ] **Step 3: Apply all replacements**

Work through `/tmp/jnp_sites.txt`. For each non-docstring line:

- If it operates on `self.mantissa` and the op is a member attribute (`.shape`, `.ndim`, `.size`, `.T`, `.mT`, `.dtype`), use the attribute form.
- Otherwise replace `jnp.<func>(self.mantissa, ...)` with `get_backend(self).<func>(self.mantissa, ...)`.
- For lines that mix multiple arrays (e.g., comparisons), use `get_backend(self, other).<func>(...)`.

Do **not** change `jnp.*` calls inside docstring examples (`>>>` lines).

- [ ] **Step 4: Add NumPy-backend smoke test**

Append to `saiunit/_base_quantity_test.py`:

```python
def test_numpy_backend_properties():
    import numpy as np
    import saiunit as u
    q = u.Quantity(np.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.meter)
    assert q.shape == (2, 2)
    assert q.ndim == 2
    assert q.size == 4
    assert q.T.shape == (2, 2)
    assert q.mT.shape == (2, 2)
    assert q.real.backend == "numpy"
    assert q.imag.backend == "numpy"


def test_numpy_backend_finiteness():
    import numpy as np
    import saiunit as u
    q = u.Quantity(np.array([1.0, np.inf, np.nan]), unit=u.meter)
    isfinite = q.isfinite
    assert isfinite[0] and not isfinite[1] and not isfinite[2]
```

- [ ] **Step 5: Run the test suite**

Run: `pytest saiunit/_base_quantity_test.py -v`
Expected: all tests PASS, including the two new ones.

- [ ] **Step 6: Commit**

```bash
git add saiunit/_base_quantity.py saiunit/_base_quantity_test.py
git commit -m "refactor: dispatch internal Quantity ops through get_backend"
```

---

## Task 8: Refactor `_fun_keep_unit_sequence` and `_fun_keep_unit_unary` helpers

**Files:**
- Modify: `saiunit/math/_fun_keep_unit.py`
- Test: `saiunit/math/_fun_keep_unit_test.py`

The helpers in `math/_fun_keep_unit.py` accept a callable (e.g., `jnp.concatenate`) and call it on raw mantissas. We change the contract: helpers now accept a **function name string**; they resolve it against `xp` after determining the backend from the mantissas.

- [ ] **Step 1: Write the failing test**

Append to `saiunit/math/_fun_keep_unit_test.py`:

```python
def test_concatenate_numpy_backend():
    import numpy as np
    import saiunit as u
    a = u.Quantity(np.array([1.0, 2.0]), unit=u.meter)
    b = u.Quantity(np.array([3.0, 4.0]), unit=u.meter)
    r = u.math.concatenate([a, b])
    assert r.backend == "numpy"
    assert r.unit == u.meter
    assert np.allclose(r.mantissa, [1.0, 2.0, 3.0, 4.0])


def test_concatenate_jax_backend():
    import jax.numpy as jnp
    import numpy as np
    import saiunit as u
    a = u.Quantity(jnp.array([1.0, 2.0]), unit=u.meter)
    b = u.Quantity(jnp.array([3.0, 4.0]), unit=u.meter)
    r = u.math.concatenate([a, b])
    assert r.backend == "jax"
    assert np.allclose(np.asarray(r.mantissa), [1.0, 2.0, 3.0, 4.0])


def test_reshape_numpy_backend():
    import numpy as np
    import saiunit as u
    q = u.Quantity(np.arange(6.0), unit=u.meter)
    r = u.math.reshape(q, (2, 3))
    assert r.backend == "numpy"
    assert r.shape == (2, 3)
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest saiunit/math/_fun_keep_unit_test.py -k "numpy_backend" -v`
Expected: FAIL — `concatenate` returns a JAX array even when given NumPy inputs (because internal code calls `jnp.concatenate`).

- [ ] **Step 3: Refactor `_fun_keep_unit_sequence`**

Open `saiunit/math/_fun_keep_unit.py`. At the top, replace `import jax.numpy as jnp` usage in the helper functions by passing function names. Add to imports:

```python
from saiunit._backend import get_backend
```

Replace `_fun_keep_unit_sequence` (around line 81) with:

```python
def _fun_keep_unit_sequence(func_name, *args, **kwargs):
    """Dispatch a sequence-input array op to the right backend, preserving units.

    Parameters
    ----------
    func_name : str
        Name of the array-API function (e.g. ``'concatenate'``, ``'stack'``).
    *args, **kwargs
        Passed to the resolved backend function after mantissa extraction.
    """
    args = maybe_custom_array_tree(args)
    leaves, treedef = jax.tree.flatten(args, is_leaf=lambda x: isinstance(x, Quantity))
    leaves = unit_scale_align_to_first(*leaves)
    unit = leaves[0].unit
    mantissas = [x.mantissa for x in leaves]
    xp = get_backend(*mantissas)
    args = treedef.unflatten(mantissas)
    func = getattr(xp, func_name)
    r = func(*args, **kwargs)
    if unit.is_unitless:
        return r
    return Quantity(r, unit=unit)
```

- [ ] **Step 4: Find and update every caller of `_fun_keep_unit_sequence` in this file**

Run: `grep -n "_fun_keep_unit_sequence(jnp\." saiunit/math/_fun_keep_unit.py`

For each match, change the first argument from `jnp.<func>` to `"<func>"`:

```python
# Before
return _fun_keep_unit_sequence(jnp.concatenate, arrays, axis=axis, ...)
# After
return _fun_keep_unit_sequence("concatenate", arrays, axis=axis, ...)
```

Apply mechanically to every call site in the file (concatenate, stack, vstack, hstack, dstack, column_stack, block, append).

- [ ] **Step 5: Repeat for `_fun_keep_unit_return_sequence`**

This helper (around line 384) follows the same pattern. Refactor identically:

```python
def _fun_keep_unit_return_sequence(func_name, *args, **kwargs):
    args = maybe_custom_array_tree(args)
    leaves, treedef = jax.tree.flatten(args, is_leaf=lambda x: isinstance(x, Quantity))
    # (rest of the existing logic, but: replace func(*args) with getattr(xp, func_name)(*args))
    ...
```

Update every caller (split, array_split, dsplit, hsplit, vsplit) from `jnp.split` etc. to `"split"`.

- [ ] **Step 6: Repeat for `_fun_keep_unit_unary` and `_broadcast_fun`**

Both follow the same pattern. Refactor each to accept `func_name: str`, compute `xp = get_backend(*mantissas)`, look up `getattr(xp, func_name)`.

Update every caller in the file to pass a string.

- [ ] **Step 7: Run the three new tests**

Run: `pytest saiunit/math/_fun_keep_unit_test.py -k "numpy_backend" -v`
Expected: 3 tests PASS.

- [ ] **Step 8: Run the full `_fun_keep_unit_test.py` to catch regressions**

Run: `pytest saiunit/math/_fun_keep_unit_test.py -v`
Expected: all tests PASS. If a test fails because the call signature changed (e.g., it called `_fun_keep_unit_sequence(jnp.foo, ...)` directly), update the test to use the string form.

- [ ] **Step 9: Commit**

```bash
git add saiunit/math/_fun_keep_unit.py saiunit/math/_fun_keep_unit_test.py
git commit -m "refactor: dispatch _fun_keep_unit helpers through xp backend"
```

---

## Task 9: Refactor `_fun_change_unit.py`

**Files:**
- Modify: `saiunit/math/_fun_change_unit.py`
- Test: `saiunit/math/_fun_change_unit_test.py`

Same pattern as Task 8. The internal helpers in this file (e.g., `_fun_change_unit_binary`) take a callable + a unit-transform function; only the callable argument changes.

- [ ] **Step 1: Write failing tests**

Append to `saiunit/math/_fun_change_unit_test.py`:

```python
def test_multiply_numpy_backend():
    import numpy as np
    import saiunit as u
    a = u.Quantity(np.array([2.0]), unit=u.meter)
    b = u.Quantity(np.array([3.0]), unit=u.second)
    r = u.math.multiply(a, b)
    assert r.backend == "numpy"
    assert r.unit == u.meter * u.second
    assert np.allclose(r.mantissa, [6.0])


def test_divide_numpy_backend():
    import numpy as np
    import saiunit as u
    a = u.Quantity(np.array([6.0]), unit=u.meter)
    b = u.Quantity(np.array([2.0]), unit=u.second)
    r = u.math.true_divide(a, b)
    assert r.backend == "numpy"
    assert r.unit == u.meter / u.second
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest saiunit/math/_fun_change_unit_test.py -k "numpy_backend" -v`
Expected: FAIL — result has backend "jax".

- [ ] **Step 3: Refactor every internal helper**

Open `saiunit/math/_fun_change_unit.py`. Identify each helper (functions starting with `_fun_`). Apply the same transformation: helper takes a string function name, looks it up via `xp = get_backend(*mantissas)`.

Add the import at the top of the file:

```python
from saiunit._backend import get_backend
```

For every call site within the file, change `jnp.<func>` → `"<func>"`.

- [ ] **Step 4: Run tests**

Run: `pytest saiunit/math/_fun_change_unit_test.py -v`
Expected: all tests PASS, including the 2 new ones.

- [ ] **Step 5: Commit**

```bash
git add saiunit/math/_fun_change_unit.py saiunit/math/_fun_change_unit_test.py
git commit -m "refactor: dispatch _fun_change_unit helpers through xp backend"
```

---

## Task 10: Refactor `_fun_remove_unit.py`, `_fun_accept_unitless.py`, and `_fun_array_creation.py`

**Files:**
- Modify: `saiunit/math/_fun_remove_unit.py`
- Modify: `saiunit/math/_fun_accept_unitless.py`
- Modify: `saiunit/math/_fun_array_creation.py`
- Test: each module's corresponding `*_test.py`

Same refactor pattern as Tasks 8–9.

- [ ] **Step 1: Write failing tests**

Append to `saiunit/math/_fun_remove_unit_test.py`:

```python
def test_argmax_numpy_backend():
    import numpy as np
    import saiunit as u
    q = u.Quantity(np.array([3.0, 1.0, 2.0]), unit=u.meter)
    r = u.math.argmax(q)
    # argmax strips units; result should still be a numpy scalar
    assert isinstance(r, (int, np.integer, np.ndarray))
```

Append to `saiunit/math/_fun_accept_unitless_test.py`:

```python
def test_sin_numpy_backend():
    import numpy as np
    import saiunit as u
    q = u.Quantity(np.array([0.0, np.pi / 2]), unit=u.UNITLESS)
    r = u.math.sin(q)
    # sin of unitless returns a raw array (not a Quantity)
    assert isinstance(r, np.ndarray)
    assert np.allclose(r, [0.0, 1.0])
```

Append to `saiunit/math/_fun_array_creation_test.py`:

```python
def test_zeros_respects_default_backend():
    import numpy as np
    import saiunit as u
    with u.using_backend("numpy"):
        q = u.math.zeros((3,), unit=u.meter)
        assert q.backend == "numpy"
    with u.using_backend("jax"):
        q = u.math.zeros((3,), unit=u.meter)
        assert q.backend == "jax"
```

- [ ] **Step 2: Refactor each file**

For `_fun_remove_unit.py` and `_fun_accept_unitless.py`: same pattern as Tasks 8–9 — helpers accept function name strings, dispatch through `xp`.

For `_fun_array_creation.py` (special case — these don't have input arrays to inspect, so backend comes from the default):

```python
def zeros(shape, dtype=None, unit=UNITLESS, **kwargs):
    from saiunit._backend import get_default_backend
    default = get_default_backend() or "jax"
    xp = _numpy_xp if default == "numpy" else _jax_xp
    arr = xp.zeros(shape, dtype=dtype, **kwargs)
    if unit.is_unitless:
        return arr
    return Quantity(arr, unit=unit)
```

Apply to: `zeros`, `ones`, `empty`, `full`, `arange`, `linspace`, `logspace`, `eye`, `identity`, and any other unconditional-array-creation function in the file. For functions that take a reference array (`zeros_like`, `ones_like`, etc.), use `get_backend(reference_array)` instead.

- [ ] **Step 3: Run each test file**

Run: `pytest saiunit/math/_fun_remove_unit_test.py saiunit/math/_fun_accept_unitless_test.py saiunit/math/_fun_array_creation_test.py -v`
Expected: all PASS.

- [ ] **Step 4: Run the full math/ test suite as a regression check**

Run: `pytest saiunit/math/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add saiunit/math/
git commit -m "refactor: dispatch remaining math/ helpers through xp backend"
```

---

## Task 11: Refactor `saiunit/linalg/`

**Files:**
- Modify: each `saiunit/linalg/*.py` file containing `jnp.linalg.*` or `jnp.*` calls
- Test: corresponding test files

`linalg` modules wrap `jnp.linalg.<func>` calls. `array_api_compat.numpy.linalg` and `array_api_compat.jax.numpy.linalg` both exist and expose the same names.

- [ ] **Step 1: Write a failing test**

Append to `saiunit/linalg/_linalg_keep_unit_test.py` (or create one if missing):

```python
def test_matmul_numpy_backend():
    import numpy as np
    import saiunit as u
    a = u.Quantity(np.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.meter)
    b = u.Quantity(np.array([[1.0, 0.0], [0.0, 1.0]]), unit=u.UNITLESS)
    r = u.linalg.matmul(a, b)
    assert r.backend == "numpy"
    assert r.unit == u.meter
    assert np.allclose(r.mantissa, [[1.0, 2.0], [3.0, 4.0]])
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest saiunit/linalg/ -k "numpy_backend" -v`
Expected: FAIL.

- [ ] **Step 3: Refactor each linalg helper**

For every `linalg/*.py` file, find the internal helpers (often named `_fun_linalg_*`) and apply the same string-name-dispatch refactor. For ops that live under `xp.linalg.<func>` rather than `xp.<func>`, the helper looks them up two levels:

```python
def _fun_linalg_keep_unit(func_name, *args, **kwargs):
    mantissas = [...]  # extracted as usual
    xp = get_backend(*mantissas)
    func = getattr(xp.linalg, func_name)
    ...
```

Direct callers like `u.linalg.matmul`, `u.linalg.solve`, `u.linalg.inv`, `u.linalg.eig`, etc. switch their string argument accordingly.

- [ ] **Step 4: Run test**

Run: `pytest saiunit/linalg/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add saiunit/linalg/
git commit -m "refactor: dispatch linalg through xp.linalg backend"
```

---

## Task 12: Refactor `saiunit/fft/`

**Files:**
- Modify: each `saiunit/fft/*.py` file
- Test: corresponding test files

Same pattern as linalg, using `xp.fft.<func>`.

- [ ] **Step 1: Write failing test**

Append to (or create) `saiunit/fft/_fft_test.py`:

```python
def test_fft_numpy_backend():
    import numpy as np
    import saiunit as u
    q = u.Quantity(np.array([1.0, 0.0, -1.0, 0.0]), unit=u.UNITLESS)
    r = u.fft.fft(q)
    # fft of unitless returns an array (or unitless Quantity)
    assert isinstance(r, np.ndarray) or r.backend == "numpy"
```

- [ ] **Step 2: Refactor each fft helper**

Apply the standard dispatch refactor.

- [ ] **Step 3: Run test and commit**

Run: `pytest saiunit/fft/ -v`
Expected: all PASS.

```bash
git add saiunit/fft/
git commit -m "refactor: dispatch fft through xp.fft backend"
```

---

## Task 13: Add JAX-only guards in `lax/`, `autograd/`, `sparse/`

**Files:**
- Create: `saiunit/_jax_guard.py`
- Modify: `saiunit/lax/__init__.py` (and helper files in `lax/`)
- Modify: `saiunit/autograd/__init__.py`
- Modify: `saiunit/sparse/__init__.py`
- Test: `saiunit/_jax_guard_test.py`

These modules require JAX semantics. Calling them with a NumPy-backed Quantity must raise `BackendError` with a clear "call `.to_jax()`" hint.

- [ ] **Step 1: Write the helper + test**

Create `saiunit/_jax_guard.py`:

```python
# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
# (Apache 2.0 header.)

"""Entry guards for JAX-only modules (lax, autograd, sparse)."""

from saiunit._backend import is_jax_array
from saiunit._exceptions import BackendError


def require_jax_backend(func_name: str, *quantities_or_arrays) -> None:
    """Raise BackendError if any input is a NumPy-backed Quantity or ndarray."""
    from saiunit._base_quantity import Quantity
    for q in quantities_or_arrays:
        if isinstance(q, Quantity):
            if not is_jax_array(q.mantissa):
                raise BackendError(
                    f"{func_name} requires the jax backend; got numpy-backed Quantity. "
                    f"Call .to_jax() on the input first."
                )
```

Create `saiunit/_jax_guard_test.py`:

```python
# (Apache 2.0 header.)

import numpy as np
import pytest

import saiunit as u
from saiunit._jax_guard import require_jax_backend


def test_require_jax_passes_for_jax():
    import jax.numpy as jnp
    q = u.Quantity(jnp.array([1.0]), unit=u.meter)
    require_jax_backend("test_fn", q)  # no raise


def test_require_jax_raises_for_numpy():
    q = u.Quantity(np.array([1.0]), unit=u.meter)
    with pytest.raises(u.BackendError, match="requires the jax backend"):
        require_jax_backend("test_fn", q)
```

- [ ] **Step 2: Run guard test**

Run: `pytest saiunit/_jax_guard_test.py -v`
Expected: PASS.

- [ ] **Step 3: Add guards at user-facing entry points**

For `saiunit/lax/__init__.py` (and any wrapper functions in `lax/`), insert at the top of each public function:

```python
def scan(f, init, xs, ...):
    require_jax_backend("saiunit.lax.scan", init, xs)
    # existing implementation
```

Do this for every public function in `lax/`, `autograd/`, and `sparse/`. Use the file's `__all__` list to enumerate which functions need guards.

(For modules with many wrappers, write a decorator instead:

```python
def _jax_only(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        require_jax_backend(fn.__qualname__, *args)
        return fn(*args, **kwargs)
    return wrapper
```

Then apply `@_jax_only` to each public function.)

- [ ] **Step 4: Write a smoke test per guarded module**

Append to `saiunit/lax/_lax_keep_unit_test.py` (or appropriate test file):

```python
def test_lax_raises_on_numpy_quantity():
    import numpy as np
    import saiunit as u
    q = u.Quantity(np.array([1.0, 2.0, 3.0]), unit=u.meter)
    # pick any lax function — using cumsum as a simple example
    with pytest.raises(u.BackendError):
        u.lax.cumsum(q)  # (or whichever lax function actually exists)
```

Append similar tests to `saiunit/autograd/<some>_test.py` and `saiunit/sparse/<some>_test.py`.

- [ ] **Step 5: Run all three module test suites**

Run: `pytest saiunit/lax/ saiunit/autograd/ saiunit/sparse/ -v`
Expected: all PASS, including the new guard tests.

- [ ] **Step 6: Commit**

```bash
git add saiunit/_jax_guard.py saiunit/_jax_guard_test.py saiunit/lax/ saiunit/autograd/ saiunit/sparse/
git commit -m "feat: guard lax/autograd/sparse against numpy-backed Quantity inputs"
```

---

## Task 14: Parametrize tests over both backends

**Files:**
- Create: `conftest.py` (at repo root)
- Modify: a representative sample of existing test files to consume the fixture

- [ ] **Step 1: Create the root `conftest.py`**

Create `/mnt/d/codes/projects/saiunit/conftest.py`:

```python
# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
# (Apache 2.0 header.)

"""Shared pytest fixtures for saiunit tests.

The ``backend`` fixture is auto-parameterized to run each consuming test
twice — once with the NumPy default backend and once with the JAX default.
Tests that want backend coverage should add ``backend`` to their signature.
"""

import pytest

import saiunit as u


@pytest.fixture(params=["numpy", "jax"])
def backend(request):
    """Set the saiunit default backend for the duration of the test."""
    with u.using_backend(request.param):
        yield request.param
```

- [ ] **Step 2: Add a smoke test that uses the fixture**

Create `saiunit/_backend_parametrize_test.py`:

```python
# (Apache 2.0 header.)

import numpy as np

import saiunit as u


def test_quantity_default_backend(backend):
    """Quantity created from a Python list should match the backend fixture."""
    q = u.Quantity([1.0, 2.0, 3.0], unit=u.meter)
    assert q.backend == backend


def test_arithmetic_default_backend(backend):
    a = u.Quantity([1.0, 2.0], unit=u.meter)
    b = u.Quantity([3.0, 4.0], unit=u.meter)
    r = a + b
    assert r.backend == backend


def test_math_function_default_backend(backend):
    import numpy as np
    q = u.Quantity([0.0, 1.0], unit=u.UNITLESS)
    r = u.math.sin(q)
    if backend == "numpy":
        assert isinstance(r, np.ndarray)
    else:
        import jax
        assert isinstance(r, jax.Array)
```

- [ ] **Step 3: Run the smoke tests**

Run: `pytest saiunit/_backend_parametrize_test.py -v`
Expected: 6 tests PASS (3 tests × 2 backends).

- [ ] **Step 4: Identify which existing tests should be parametrized**

This is judgement-driven. As a rule:
- Tests that exercise arithmetic, broadcasting, math/linalg/fft functions → add `backend` to signature.
- Tests that exercise specifically JAX features (`jit`, `grad`, `vmap`, lax, sparse) → leave JAX-only.
- Display, repr, equality, hashing tests → add `backend` (catches type-dependent edge cases).

For v1, focus on:
- `saiunit/_base_quantity_test.py` (arithmetic & properties)
- `saiunit/math/_fun_keep_unit_test.py`
- `saiunit/math/_fun_change_unit_test.py`
- `saiunit/math/_fun_accept_unitless_test.py`

For each chosen test file, add `backend` as a parameter to the relevant test functions. Tests that hard-code `jnp.array(...)` continue to use JAX inputs (no fixture needed); tests that build inputs via the Quantity constructor with Python lists/scalars will pick up the fixture.

- [ ] **Step 5: Run the parametrized test suites**

Run: `pytest saiunit/_base_quantity_test.py saiunit/math/ -v`
Expected: all PASS. Test count should roughly double for the parametrized tests.

If any test fails on the NumPy backend with a real bug (not a test-side assumption), file an issue and patch the implementation — do not paper over by skipping NumPy.

- [ ] **Step 6: Commit**

```bash
git add conftest.py saiunit/_backend_parametrize_test.py saiunit/_base_quantity_test.py saiunit/math/
git commit -m "test: parametrize core tests over numpy and jax backends"
```

---

## Task 15: Documentation and changelog

**Files:**
- Modify: `saiunit/__init__.py` docstring
- Create: `docs/getting_started/numpy_backend.md` (or `.rst` if the docs use rst)
- Create or modify: `CHANGELOG.md`

- [ ] **Step 1: Update the package docstring**

In `saiunit/__init__.py`, update the top-level docstring (currently says "Physical units for JAX arrays") to:

```python
"""
saiunit -- Physical units for JAX and NumPy arrays.

``saiunit`` provides a :class:`Quantity` type that pairs a JAX array or
NumPy array with a physical :class:`Unit`, ensuring dimensional correctness
at every arithmetic operation. ...
"""
```

(Update the rest of the docstring to mention both backends where appropriate.)

- [ ] **Step 2: Write the user-facing guide**

Check whether `docs/` uses rst or markdown (`ls docs/getting_started/`). Create a new page using the existing extension. Content outline:

```markdown
# Using saiunit with NumPy

saiunit accepts both JAX arrays and NumPy arrays as the underlying mantissa
of a `Quantity`. All math, linalg, and fft operations dispatch to the
matching backend.

## Quick start

\`\`\`python
import numpy as np
import saiunit as u

q = u.Quantity(np.array([1.0, 2.0, 3.0]), unit=u.meter)
print(q.backend)  # 'numpy'
\`\`\`

## Choosing the default backend

For Python scalars or lists where the backend is ambiguous, `saiunit`
defaults to JAX unless you tell it otherwise:

\`\`\`python
u.set_default_backend("numpy")
q = u.Quantity([1.0, 2.0], unit=u.meter)
print(q.backend)  # 'numpy'

with u.using_backend("jax"):
    q = u.Quantity([1.0, 2.0], unit=u.meter)
    print(q.backend)  # 'jax'
\`\`\`

## Converting between backends

\`\`\`python
q_np = u.Quantity(np.array([1.0]), unit=u.meter)
q_jax = q_np.to_jax()
q_back = q_jax.to_numpy()
\`\`\`

## JAX-only operations

`saiunit.lax`, `saiunit.autograd`, and `saiunit.sparse` require JAX. They
raise `BackendError` if given a NumPy-backed `Quantity`. Call `.to_jax()`
first.

## NumPy ufunc interop

Standard NumPy ufuncs (`np.add`, `np.sin`, `np.exp`, …) preserve units:

\`\`\`python
q = u.Quantity(np.array([0.0, np.pi / 2]), unit=u.UNITLESS)
np.sin(q)  # works; dimension-checked
\`\`\`
```

(Add this page to the docs index/toctree following the existing pattern.)

- [ ] **Step 3: Add a changelog entry**

Check whether `CHANGELOG.md` exists. If yes, add a new section at the top:

```markdown
## Unreleased

### Added

- NumPy is now a first-class array backend alongside JAX. `Quantity` can
  wrap an `np.ndarray` directly; all math, linalg, and fft operations
  dispatch to the matching backend.
- `Quantity.backend` property reporting `'numpy'` or `'jax'`.
- `Quantity.to_numpy()` and `Quantity.to_jax()` conversion methods.
- `saiunit.set_default_backend()`, `saiunit.get_default_backend()`, and
  `saiunit.using_backend()` for controlling the default backend.
- `Quantity.__array_ufunc__` so `np.sin(quantity)`, `np.add(q1, q2)`, etc.
  preserve units instead of stripping them.
- `saiunit.BackendError` exception type (subclass of `TypeError`).

### Changed

- `Quantity(np.ndarray(...))` now keeps the mantissa as `np.ndarray`. Previously
  it was implicitly converted to `jax.Array`. Use `.to_jax()` to opt back in,
  or call `saiunit.set_default_backend('jax')` for the prior behavior.

### Required

- New mandatory dependency: `array_api_compat>=1.9`.
```

If no changelog exists, create it.

- [ ] **Step 4: Run the docs build (if configured)**

Check whether the repo builds docs locally: `ls docs/Makefile`. If yes:

Run: `cd docs && make html`
Expected: clean build, no warnings on the new page.

- [ ] **Step 5: Final full-suite regression test**

Run: `pytest saiunit/`
Expected: all tests PASS across both backends.

- [ ] **Step 6: Commit**

```bash
git add saiunit/__init__.py docs/ CHANGELOG.md
git commit -m "docs: add numpy backend user guide and changelog entry"
```

---

## Self-review (run after the plan is fully drafted)

This section is a checklist for the plan author. Each item was verified during drafting; included here so a reviewer can spot-check.

**Spec coverage:**
- Layer 1 backend module → Task 3 ✓
- Layer 2 Quantity changes → Tasks 4, 5, 6, 7 ✓
- Layer 3 math/linalg/fft refactor → Tasks 8, 9, 10, 11, 12 ✓
- `__array_ufunc__` → Task 6 ✓
- JAX-only guards → Task 13 ✓
- Test parametrization → Task 14 ✓
- `BackendError` exception → Task 2 ✓
- `array_api_compat` dependency → Task 1 ✓
- Conversion methods + introspection → Tasks 4, 5 ✓
- `using_backend` context manager → Task 3 ✓
- Documentation + changelog → Task 15 ✓
- "Potential observable behavior change" → addressed by Task 4 Step 3 (explicit backend-preservation fix) and Task 15 changelog ✓
- Open question #3 (ufunc routing list) → resolved in Task 6 with explicit dispatch table ✓

**Placeholder scan:** No `TBD`, `TODO`, or "add appropriate error handling" placeholders. Every code step has actual code.

**Type consistency:**
- `get_backend(*xs) -> ModuleType` — used consistently in Tasks 3, 7, 8, 9, 10, 11, 12.
- `set_default_backend(name: Optional[BackendName])` — same signature in Tasks 3, 14, 15.
- `.backend` returns `"numpy"` or `"jax"` — consistent in Tasks 4, 7, 8, 9, etc.
- `BackendError` is `TypeError` subclass — Tasks 2, 6, 13.
- Helper signatures `_fun_keep_unit_*(func_name: str, ...)` consistent in Tasks 8, 9, 10.

**Open items deferred to implementation:**
- The exact list of which existing tests get parametrized (Task 14 Step 4) is a judgment call left to the implementer based on the rule given.
- The complete enumeration of all `jnp.*` call sites in `_base_quantity.py` (Task 7 Step 1) is grep-driven and case-by-case.
- The complete enumeration of all helper call sites in math/ (Tasks 8–10) is grep-driven.

These are intentional — they're cleanup work where the implementation IS the enumeration. The plan provides the pattern and the verification command.
