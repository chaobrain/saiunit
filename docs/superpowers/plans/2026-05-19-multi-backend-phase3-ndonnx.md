# Multi-Backend Phase 3: ndonnx Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Phase 1 and Phase 2 must be merged before starting Phase 3.

**Goal:** Add ndonnx as a saiunit backend so users can build unit-aware computations and export them as ONNX graphs.

**Architecture:** Same closed-enum extension pattern as Phases 1 and 2. ndonnx is itself array-API-compatible (no `array_api_compat` wrapper needed); we use `ndonnx` directly as the xp namespace. Symbolic execution means saiunit's promise is narrow: dispatch routes correctly; if ndonnx doesn't implement an op, the ndonnx error surfaces unwrapped. The Phase 2 lazy-safe `__repr__` already covers ndonnx because ndonnx arrays have their own non-materializing `__repr__`.

**Tech Stack:** Python 3.10+, jax, numpy, `array_api_compat>=1.9`, optionally `ndonnx>=0.9`. Tests via pytest with `importorskip`.

**Spec:** `docs/superpowers/specs/2026-05-19-multi-backend-design.md` (Section 4.4 and 8.3).

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `saiunit/_backend.py` | Modify | Add `is_ndonnx_array`, extend `BackendName`, `get_backend`, `to_backend`, validation |
| `saiunit/_base_quantity.py` | Modify | Add `to_ndonnx`, extend `backend` property, extend lazy-safe repr |
| `saiunit/_jax_guard.py` | Modify | Reject ndonnx arrays |
| `saiunit/__init__.py` | Modify | Export `is_ndonnx_array` |
| `saiunit/_backend_test.py` | Modify | Tests for detector, dispatch, conversion |
| `saiunit/_jax_guard_test.py` | Modify | Rejection tests for ndonnx |
| `saiunit/_base_quantity_test.py` | Modify | Tests for `to_ndonnx`, `backend == 'ndonnx'` |
| `saiunit/_ndonnx_test.py` | Create | Symbolic-composition smoke tests |
| `conftest.py` | Modify | Add `ndonnx` to fixture params |
| `pyproject.toml` | Modify | Add `ndonnx` extra; update `all` to include it |
| `docs/getting_started/numpy_backend.md` | Modify | Add "ndonnx: symbolic / ONNX export" subsection |

---

## Task 1: Add `is_ndonnx_array` detector

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_backend_test.py`:

```python
def test_is_ndonnx_array_false_for_non_ndonnx():
    from saiunit._backend import is_ndonnx_array
    assert is_ndonnx_array(np.array([1.0])) is False
    assert is_ndonnx_array(jnp.array([1.0])) is False
    assert is_ndonnx_array(1.0) is False


def test_is_ndonnx_array_true_when_available():
    ndonnx = pytest.importorskip("ndonnx")
    from saiunit._backend import is_ndonnx_array
    arr = ndonnx.asarray(np.array([1.0, 2.0]))
    assert is_ndonnx_array(arr) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py -k "is_ndonnx_array" -v
```

Expected: FAIL with `ImportError: cannot import name 'is_ndonnx_array'`.

- [ ] **Step 3: Add the detector**

In `saiunit/_backend.py`, after `is_dask_array`, add:

```python
def is_ndonnx_array(x) -> bool:
    """Return True if ``x`` is an ndonnx Array. False if ndonnx is not installed."""
    ndonnx = _try_import("ndonnx")
    if ndonnx is None:
        return False
    return isinstance(x, ndonnx.Array)
```

Update `__all__` to include `"is_ndonnx_array"`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_backend_test.py -k "is_ndonnx_array" -v
```

Expected: PASS (or SKIP if ndonnx not installed).

- [ ] **Step 5: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py
git commit -m "feat(backend): add is_ndonnx_array detector"
```

---

## Task 2: Extend `BackendName` Literal and `get_backend` dispatch

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_backend_test.py`:

```python
def test_backend_name_includes_ndonnx():
    from saiunit._backend import BackendName
    import typing
    assert "ndonnx" in typing.get_args(BackendName)


def test_get_backend_ndonnx_only():
    ndonnx = pytest.importorskip("ndonnx")
    from saiunit._backend import get_backend
    xp = get_backend(ndonnx.asarray(np.array([1.0])))
    assert xp is ndonnx
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py -k "backend_name_includes_ndonnx or get_backend_ndonnx" -v
```

Expected: FAIL.

- [ ] **Step 3: Extend `BackendName`, `_xp_for`, and `get_backend`**

In `saiunit/_backend.py`:

Change:
```python
BackendName = Literal["numpy", "jax", "cupy", "torch", "dask"]
```
to:
```python
BackendName = Literal["numpy", "jax", "cupy", "torch", "dask", "ndonnx"]
```

In `_xp_for`, add a branch:
```python
    elif name == "ndonnx":
        ndonnx = _try_import("ndonnx")
        if ndonnx is None:
            raise BackendError(
                "ndonnx backend requested but ndonnx is not installed. "
                "Install with: pip install saiunit[ndonnx]"
            )
        mod = ndonnx  # ndonnx is itself array-API-compatible
```

In `get_backend`, extend the detection chain:
```python
    has_ndonnx = any(is_ndonnx_array(x) for x in mantissas)

    kinds = [name for name, has in
             [("numpy", has_numpy), ("jax", has_jax),
              ("cupy", has_cupy), ("torch", has_torch),
              ("dask", has_dask), ("ndonnx", has_ndonnx)] if has]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_backend_test.py -k "backend_name or get_backend" -v
```

Expected: PASS for all backends.

- [ ] **Step 5: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py
git commit -m "feat(backend): dispatch get_backend() to ndonnx xp namespace"
```

---

## Task 3: Extend `to_backend` with ndonnx branch

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_backend_test.py`:

```python
def test_to_backend_numpy_to_ndonnx():
    ndonnx = pytest.importorskip("ndonnx")
    from saiunit._backend import to_backend, is_ndonnx_array
    arr = np.array([1.0, 2.0])
    out = to_backend(arr, "ndonnx")
    assert is_ndonnx_array(out)


def test_to_backend_ndonnx_noop():
    ndonnx = pytest.importorskip("ndonnx")
    from saiunit._backend import to_backend
    arr = ndonnx.asarray(np.array([1.0]))
    out = to_backend(arr, "ndonnx")
    assert out is arr


def test_to_backend_ndonnx_rejects_kwargs():
    pytest.importorskip("ndonnx")
    from saiunit._backend import to_backend
    with pytest.raises(TypeError, match="does not accept"):
        to_backend(np.array([1.0]), "ndonnx", device="cuda")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py -k "to_backend.*ndonnx" -v
```

Expected: FAIL with `ValueError: backend must be one of ...; got 'ndonnx'`.

- [ ] **Step 3: Add the ndonnx branch**

In `saiunit/_backend.py`, add to `to_backend` before the final `raise ValueError`:

```python
    if name == "ndonnx":
        ndonnx = _try_import("ndonnx")
        if ndonnx is None:
            raise BackendError(
                "ndonnx backend requested but ndonnx is not installed. "
                "Install with: pip install saiunit[ndonnx]"
            )
        if kwargs:
            raise TypeError(f"to_backend(name='ndonnx') does not accept kwargs; got {sorted(kwargs)}")
        if is_ndonnx_array(x):
            return x
        return ndonnx.asarray(x)
```

And update the final error to include ndonnx:

```python
    raise ValueError(
        f"backend must be one of 'numpy', 'jax', 'cupy', 'torch', 'dask', 'ndonnx'; "
        f"got {name!r}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_backend_test.py -k "to_backend.*ndonnx" -v
```

Expected: PASS (or SKIP if ndonnx not installed).

- [ ] **Step 5: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py
git commit -m "feat(backend): add ndonnx branch to to_backend()"
```

---

## Task 4: Extend `using_backend` / `set_default_backend` validation

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_backend_test.py`:

```python
def test_set_default_backend_accepts_ndonnx():
    from saiunit._backend import set_default_backend, get_default_backend
    set_default_backend("ndonnx")
    try:
        assert get_default_backend() == "ndonnx"
    finally:
        set_default_backend(None)


def test_using_backend_accepts_ndonnx():
    from saiunit._backend import using_backend, get_default_backend
    with using_backend("ndonnx"):
        assert get_default_backend() == "ndonnx"
```

Update the existing `test_set_default_backend_rejects_invalid` to reflect the new error message:

```python
def test_set_default_backend_rejects_invalid():
    with pytest.raises(
        ValueError,
        match="must be 'numpy', 'jax', 'cupy', 'torch', 'dask', 'ndonnx', or None",
    ):
        set_default_backend("notabackend")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py -k "set_default_backend or using_backend_accepts_ndonnx" -v
```

Expected: FAIL.

- [ ] **Step 3: Broaden validation**

In `saiunit/_backend.py`:

```python
def set_default_backend(name: Optional[BackendName]) -> None:
    if name not in ("numpy", "jax", "cupy", "torch", "dask", "ndonnx", None):
        raise ValueError(
            f"default backend must be 'numpy', 'jax', 'cupy', 'torch', 'dask', "
            f"'ndonnx', or None; got {name!r}"
        )
    _default_backend.set(name)


@contextmanager
def using_backend(name: BackendName) -> Iterator[None]:
    if name not in ("numpy", "jax", "cupy", "torch", "dask", "ndonnx"):
        raise ValueError(
            f"backend must be 'numpy', 'jax', 'cupy', 'torch', 'dask', or 'ndonnx'; "
            f"got {name!r}"
        )
    token = _default_backend.set(name)
    try:
        yield
    finally:
        _default_backend.reset(token)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_backend_test.py -k "set_default_backend or using_backend" -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py
git commit -m "feat(backend): accept ndonnx in using_backend() and set_default_backend()"
```

---

## Task 5: Add `Quantity.to_ndonnx()` method

**Files:**
- Modify: `saiunit/_base_quantity.py`
- Test: `saiunit/_base_quantity_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_base_quantity_test.py`:

```python
def test_quantity_to_ndonnx_basic():
    ndonnx = pytest.importorskip("ndonnx")
    import saiunit as u
    from saiunit._backend import is_ndonnx_array
    q = u.Quantity(np.array([1.0, 2.0]), unit=u.meter)
    q2 = q.to_ndonnx()
    assert is_ndonnx_array(q2.mantissa)
    assert q2.unit == u.meter


def test_quantity_to_ndonnx_noop():
    ndonnx = pytest.importorskip("ndonnx")
    import saiunit as u
    q = u.Quantity(ndonnx.asarray(np.array([1.0])), unit=u.meter)
    q2 = q.to_ndonnx()
    assert q2 is q
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_base_quantity_test.py -k "to_ndonnx" -v
```

Expected: FAIL with `AttributeError: 'Quantity' object has no attribute 'to_ndonnx'`.

- [ ] **Step 3: Add the method**

In `saiunit/_base_quantity.py`, after `to_dask`, add:

```python
def to_ndonnx(self) -> 'Quantity':
    """Return a new Quantity with mantissa converted to an ``ndonnx.Array``.

    No-op (returns ``self``) if the mantissa is already an ndonnx array.
    ndonnx arrays are symbolic — operations build an ONNX graph rather than
    eagerly computing.
    """
    from saiunit._backend import is_ndonnx_array, to_backend
    if is_ndonnx_array(self._mantissa):
        return self
    return Quantity(to_backend(self._mantissa, "ndonnx"), unit=self.unit)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_base_quantity_test.py -k "to_ndonnx" -v
```

Expected: PASS (or SKIP if ndonnx not installed).

- [ ] **Step 5: Commit**

```bash
git add saiunit/_base_quantity.py saiunit/_base_quantity_test.py
git commit -m "feat(quantity): add Quantity.to_ndonnx()"
```

---

## Task 6: Extend `Quantity.backend` property to recognize ndonnx

**Files:**
- Modify: `saiunit/_base_quantity.py`
- Test: `saiunit/_base_quantity_test.py`

- [ ] **Step 1: Write the failing test**

Append to `saiunit/_base_quantity_test.py`:

```python
def test_quantity_backend_ndonnx():
    ndonnx = pytest.importorskip("ndonnx")
    import saiunit as u
    q = u.Quantity(ndonnx.asarray(np.array([1.0])), unit=u.meter)
    assert q.backend == "ndonnx"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest saiunit/_base_quantity_test.py::test_quantity_backend_ndonnx -v
```

Expected: FAIL — returns `'jax'`.

- [ ] **Step 3: Extend the chain**

In `saiunit/_base_quantity.py`, update the `backend` property:

```python
@property
def backend(self) -> str:
    """The backend of the underlying mantissa: one of
    ``'numpy'``, ``'jax'``, ``'cupy'``, ``'torch'``, ``'dask'``, ``'ndonnx'``."""
    from saiunit._backend import (
        is_numpy_array, is_cupy_array, is_torch_array,
        is_dask_array, is_ndonnx_array,
    )
    m = self._mantissa
    if is_numpy_array(m):
        return "numpy"
    if is_cupy_array(m):
        return "cupy"
    if is_torch_array(m):
        return "torch"
    if is_dask_array(m):
        return "dask"
    if is_ndonnx_array(m):
        return "ndonnx"
    return "jax"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest saiunit/_base_quantity_test.py -k "quantity_backend" -v
```

Expected: PASS for all six backends.

- [ ] **Step 5: Commit**

```bash
git add saiunit/_base_quantity.py saiunit/_base_quantity_test.py
git commit -m "feat(quantity): extend Quantity.backend property with ndonnx"
```

---

## Task 7: Reject ndonnx arrays in `require_jax_backend`

**Files:**
- Modify: `saiunit/_jax_guard.py`
- Test: `saiunit/_jax_guard_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_jax_guard_test.py`:

```python
def test_require_jax_raises_for_ndonnx_quantity():
    ndonnx = pytest.importorskip("ndonnx")
    q = u.Quantity(ndonnx.asarray(np.array([1.0])), unit=u.meter)
    with pytest.raises(u.BackendError, match="ndonnx-backed Quantity"):
        require_jax_backend("test_fn", q)


def test_require_jax_rejects_bare_ndonnx_array():
    ndonnx = pytest.importorskip("ndonnx")
    arr = ndonnx.asarray(np.array([1.0]))
    with pytest.raises(u.BackendError, match="ndonnx"):
        require_jax_backend("test_fn", arr)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_jax_guard_test.py -k "ndonnx" -v
```

Expected: FAIL — bare-ndonnx test doesn't raise.

- [ ] **Step 3: Extend the guard**

In `saiunit/_jax_guard.py`, update `require_jax_backend`:

```python
def require_jax_backend(func_name: str, *quantities_or_arrays) -> None:
    from saiunit._base_quantity import Quantity
    from saiunit._backend import (
        is_numpy_array, is_jax_array, is_cupy_array, is_torch_array,
        is_dask_array, is_ndonnx_array,
    )

    for q in quantities_or_arrays:
        if isinstance(q, Quantity):
            backend = q.backend
            if backend != "jax":
                raise BackendError(
                    f"{func_name} requires the jax backend; got "
                    f"{backend}-backed Quantity. Call .to_jax() on the input first."
                )
            continue
        if is_cupy_array(q):
            raise BackendError(
                f"{func_name} requires the jax backend; got cupy array. "
                f"Convert to a JAX array first."
            )
        if is_torch_array(q):
            raise BackendError(
                f"{func_name} requires the jax backend; got torch tensor. "
                f"Convert to a JAX array first."
            )
        if is_dask_array(q):
            raise BackendError(
                f"{func_name} requires the jax backend; got dask array. "
                f"Convert to a JAX array first."
            )
        if is_ndonnx_array(q):
            raise BackendError(
                f"{func_name} requires the jax backend; got ndonnx array. "
                f"Convert to a JAX array first."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_jax_guard_test.py -v
```

Expected: PASS for all guard tests.

- [ ] **Step 5: Commit**

```bash
git add saiunit/_jax_guard.py saiunit/_jax_guard_test.py
git commit -m "feat(guard): reject ndonnx arrays in require_jax_backend"
```

---

## Task 8: Export `is_ndonnx_array` in `saiunit/__init__.py`

**Files:**
- Modify: `saiunit/__init__.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing test**

Append to `saiunit/_backend_test.py`:

```python
def test_top_level_exports_is_ndonnx_array():
    import saiunit as u
    assert hasattr(u, "is_ndonnx_array")
    assert "is_ndonnx_array" in u.__all__
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest saiunit/_backend_test.py::test_top_level_exports_is_ndonnx_array -v
```

Expected: FAIL.

- [ ] **Step 3: Add the export**

In `saiunit/__init__.py`, update the `._backend` import block:

```python
from ._backend import (
    get_default_backend,
    is_cupy_array,
    is_dask_array,
    is_jax_array,
    is_ndonnx_array,
    is_numpy_array,
    is_torch_array,
    set_default_backend,
    using_backend,
)
```

And add `'is_ndonnx_array'` to the `# _backend` section of `__all__`.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest saiunit/_backend_test.py::test_top_level_exports_is_ndonnx_array -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add saiunit/__init__.py saiunit/_backend_test.py
git commit -m "feat(saiunit): export is_ndonnx_array at top level"
```

---

## Task 9: Add `ndonnx` extra to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the extra and update `all`**

In `pyproject.toml` `[project.optional-dependencies]`:

Change:
```toml
all = ["saiunit[cupy,torch,dask]"]
```
to:
```toml
ndonnx = ["ndonnx>=0.9"]
all = ["saiunit[cupy,torch,dask,ndonnx]"]
```

Full block:

```toml
[project.optional-dependencies]
testing = ['pytest', 'brainstate']
cpu = ["jax[cpu]"]
cuda12 = ["jax[cuda12]"]
cuda13 = ["jax[cuda13]"]
tpu = ["jax[tpu]"]
cupy = ["cupy-cuda12x>=13.0"]
torch = ["torch>=2.0"]
dask = ["dask[array]>=2024.1"]
ndonnx = ["ndonnx>=0.9"]
all = ["saiunit[cupy,torch,dask,ndonnx]"]
```

- [ ] **Step 2: Verify TOML parses**

```bash
python -c "import tomllib; tomllib.loads(open('pyproject.toml').read())"
```

Expected: no output.

- [ ] **Step 3: Dry-run install verification**

```bash
pip install --dry-run -e '.[ndonnx]' 2>&1 | head -20
```

Expected: ndonnx resolved.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build(deps): add ndonnx optional extra"
```

---

## Task 10: Extend `conftest.py` `backend` fixture

**Files:**
- Modify: `conftest.py`
- Test: `saiunit/_backend_parametrize_test.py`

- [ ] **Step 1: Update the assertion in the existing parametrize test**

In `saiunit/_backend_parametrize_test.py`, update `test_backend_fixture_includes_all_phase1_and_2` to:

```python
def test_backend_fixture_includes_all_backends(backend):
    assert backend in {"numpy", "jax", "cupy", "torch", "dask", "ndonnx"}
```

- [ ] **Step 2: Verify the fixture currently has 5 params**

```bash
pytest saiunit/_backend_parametrize_test.py::test_backend_fixture_includes_all_backends --collect-only -q
```

Expected: 5 items.

- [ ] **Step 3: Add ndonnx to the fixture**

In `conftest.py`:

```python
@pytest.fixture(params=["numpy", "jax", "cupy", "torch", "dask", "ndonnx"])
def backend(request):
    if request.param == "cupy":
        pytest.importorskip("cupy")
    elif request.param == "torch":
        pytest.importorskip("torch")
    elif request.param == "dask":
        pytest.importorskip("dask.array")
    elif request.param == "ndonnx":
        pytest.importorskip("ndonnx")
    with u.using_backend(request.param):
        yield request.param
```

- [ ] **Step 4: Run parametrized tests and confirm 6 items**

```bash
pytest saiunit/_backend_parametrize_test.py --collect-only -q
pytest saiunit/_backend_parametrize_test.py -v
```

Expected: 6 collected items per test; ndonnx runs (or skips).

- [ ] **Step 5: Commit**

```bash
git add conftest.py saiunit/_backend_parametrize_test.py
git commit -m "test(backend): include ndonnx in parametrized backend fixture"
```

---

## Task 11: Add ndonnx symbolic-composition smoke tests

**Files:**
- Create: `saiunit/_ndonnx_test.py`
- Modify: `saiunit/_backend_parametrize_test.py`

- [ ] **Step 1: Create the dedicated symbolic-composition test file**

Create `saiunit/_ndonnx_test.py`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Symbolic-composition smoke tests for the ndonnx backend.

ndonnx arrays are symbolic — operations build an ONNX graph rather than
eagerly computing. These tests verify that saiunit dispatch routes
correctly through ndonnx, not that every saiunit.math function works
(ndonnx may not implement every op).
"""

import numpy as np
import pytest

ndonnx = pytest.importorskip("ndonnx")

import saiunit as u
from saiunit._backend import is_ndonnx_array


def test_ndonnx_quantity_preserves_symbolic_type():
    q = u.Quantity(ndonnx.asarray(np.array([1.0, 2.0, 3.0])), unit=u.meter)
    assert is_ndonnx_array(q.mantissa)
    assert q.backend == "ndonnx"


def test_ndonnx_arithmetic_stays_symbolic():
    q = u.Quantity(ndonnx.asarray(np.array([1.0, 2.0])), unit=u.meter)
    r = q + q
    assert is_ndonnx_array(r.mantissa)
    assert r.backend == "ndonnx"
    assert r.unit == u.meter


def test_ndonnx_math_sin_dispatches():
    q = u.Quantity(ndonnx.asarray(np.array([0.0, 1.0])), unit=u.UNITLESS)
    r = u.math.sin(q)
    assert is_ndonnx_array(r)


def test_ndonnx_unit_check_still_fires():
    """Dimensional analysis is independent of backend — the check fires
    on the units, not the mantissa type."""
    a = u.Quantity(ndonnx.asarray(np.array([1.0])), unit=u.meter)
    b = u.Quantity(ndonnx.asarray(np.array([1.0])), unit=u.second)
    with pytest.raises(u.UnitMismatchError):
        a + b
```

- [ ] **Step 2: Extend the cross-backend smoke tests in `_backend_parametrize_test.py`**

Update `test_math_sin_on_each_backend` to handle ndonnx:

```python
def test_math_sin_on_each_backend(backend):
    """saiunit.math.sin returns a mantissa native to the active backend."""
    q = u.Quantity([0.0, 1.0], unit=UNITLESS)
    r = u.math.sin(q)
    if backend == "numpy":
        assert isinstance(r, np.ndarray)
    elif backend == "jax":
        import jax
        assert isinstance(r, jax.Array)
    elif backend == "cupy":
        import cupy
        assert isinstance(r, cupy.ndarray)
    elif backend == "torch":
        import torch
        assert isinstance(r, torch.Tensor)
    elif backend == "dask":
        import dask.array as da
        assert isinstance(r, da.Array)
    elif backend == "ndonnx":
        import ndonnx
        assert isinstance(r, ndonnx.Array)
```

Skip ndonnx-incompatible smoke tests cleanly. For `test_linalg_norm_on_each_backend`, the `float(...)` call materializes — ndonnx may not support that. Skip ndonnx for this test:

```python
def test_linalg_norm_on_each_backend(backend):
    if backend == "ndonnx":
        pytest.skip("ndonnx scalar materialization is out of scope for this smoke test")
    q = u.Quantity([3.0, 4.0], unit=meter)
    n = u.linalg.norm(q)
    assert n.unit == meter
    if backend == "dask":
        assert float(n.mantissa.compute()) == 5.0
    else:
        assert float(n.mantissa) == 5.0
```

- [ ] **Step 3: Run the new tests**

```bash
pytest saiunit/_ndonnx_test.py -v
pytest saiunit/_backend_parametrize_test.py -v
```

Expected: ndonnx-specific tests PASS (or SKIP if ndonnx is missing). Cross-backend smoke tests PASS for all six backends.

- [ ] **Step 4: Commit**

```bash
git add saiunit/_ndonnx_test.py saiunit/_backend_parametrize_test.py
git commit -m "test(backend): add ndonnx symbolic-composition smoke tests"
```

---

## Task 12: Update the user guide

**Files:**
- Modify: `docs/getting_started/numpy_backend.md`

- [ ] **Step 1: Append the ndonnx section**

At the end of `docs/getting_started/numpy_backend.md`, append:

````markdown
## ndonnx: symbolic / ONNX export

`saiunit` accepts `ndonnx.Array` mantissas via the optional `ndonnx` extra:

```bash
pip install saiunit[ndonnx]
```

```python
import numpy as np
import ndonnx
import saiunit as u

q = u.Quantity(ndonnx.asarray(np.array([1.0, 2.0, 3.0])), unit=u.meter)
print(q.backend)        # 'ndonnx'
print((q + q).backend)  # 'ndonnx' — still symbolic
```

Convert with `Quantity.to_ndonnx()`:

```python
q_np = u.Quantity(np.array([1.0, 2.0]), unit=u.meter)
q_nd = q_np.to_ndonnx()
```

### What works

Dispatch routes correctly: `q + q`, `q * 2`, `u.math.sin(q / u.meter)`, and
other array-API-standard operations build the ONNX graph as expected.
Dimensional analysis is independent of backend — `meter + second` still
raises `UnitMismatchError`.

### What may not work

ndonnx is still maturing; some `saiunit.math` / `saiunit.linalg` operations
may not have ndonnx implementations. When that happens, the ndonnx error
surfaces unwrapped — saiunit does not catch it. Consult the ndonnx
documentation for the supported op set.

### Exporting

Use ndonnx's own export workflow on the mantissa once your computation is
built. saiunit does not provide saiunit-level export helpers; the unit
information lives on the Python `Quantity` object and is not encoded in
the ONNX graph.
````

- [ ] **Step 2: Verify the file**

```bash
python -c "import pathlib; assert 'ndonnx: symbolic' in pathlib.Path('docs/getting_started/numpy_backend.md').read_text()"
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add docs/getting_started/numpy_backend.md
git commit -m "docs: add ndonnx symbolic-execution section to user guide"
```

---

## Task 13: Install ndonnx in CI workflows

**Files:**
- Modify: `.github/workflows/CI.yml`

- [ ] **Step 1: Add an ndonnx-install step to each platform job**

In `.github/workflows/CI.yml`, find each platform's "Install dependencies"
step. After the `pip install 'dask[array]' --no-cache-dir` line
(added in Phase 2), add:

```yaml
          pip install ndonnx --no-cache-dir
```

The full step (for each platform) becomes:

```yaml
      - name: Install dependencies
        run: |
          python -m pip cache purge
          python -m pip install --upgrade pip setuptools  --no-cache-dir
          python -m pip install -r requirements-dev.txt  --no-cache-dir
          pip install . --no-cache-dir
          pip install torch --no-cache-dir
          pip install 'dask[array]' --no-cache-dir
          pip install ndonnx --no-cache-dir
```

- [ ] **Step 2: Verify the YAML parses**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/CI.yml'))"
```

Expected: no output.

- [ ] **Step 3: If ndonnx install fails on Windows or macOS, gate it**

ndonnx is still maturing; binary wheels may not be available everywhere.
If CI fails because pip cannot install ndonnx on a given platform, change
the line to use `|| true` for that platform only:

```yaml
          pip install ndonnx --no-cache-dir || true
```

This lets the install step succeed even if ndonnx is unavailable; the
tests then skip via `pytest.importorskip`. Document the platform gap in
the user guide section added in Task 12 if applicable.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/CI.yml
git commit -m "ci: install ndonnx on all platforms to exercise the ndonnx backend"
```

---

## Task 14: Final integration verification

**Files:** none

- [ ] **Step 1: Run the full saiunit test suite**

```bash
pytest saiunit/ -x -q
```

Expected: all tests pass; ndonnx tests skip if ndonnx not installed.

- [ ] **Step 2: With ndonnx installed, run again**

```bash
pip install ndonnx
pytest saiunit/ -k "ndonnx or backend" -v
```

Expected: every ndonnx test PASSes (or skips known-unsupported ops with explicit `pytest.skip`).

- [ ] **Step 3: Quick interactive smoke test**

```bash
python -c "
import numpy as np, ndonnx
import saiunit as u
q = u.Quantity(ndonnx.asarray(np.array([1.0, 2.0])), unit=u.meter)
print('backend:', q.backend)
r = q + q
print('arithmetic backend:', r.backend)
print('unit preserved:', r.unit == u.meter)
try:
    u.lax.slice(q, (0,), (1,))
except u.BackendError as e:
    print('expected BackendError:', e)
"
```

Expected output (something like):
```
backend: ndonnx
arithmetic backend: ndonnx
unit preserved: True
expected BackendError: ...slice requires the jax backend; got ndonnx-backed Quantity...
```

- [ ] **Step 4: Push and PR**

```bash
git push -u origin <your-branch-name>
```

PR title:

> `feat(backend): add ndonnx backend (Phase 3 of multi-backend support — completes the design)`

Reference the spec and the Phase 3 plan in the body. After CI passes and merge, the full multi-backend spec is implemented.

---

## Done Criteria

Phase 3 is complete when:
1. All 14 tasks above are checked off.
2. `pytest saiunit/` passes on a machine with no ndonnx installed (ndonnx tests skip).
3. `pytest saiunit/` passes on a machine with ndonnx installed.
4. `saiunit.is_ndonnx_array`, `Quantity.to_ndonnx`, `Quantity.backend == "ndonnx"`, and `BackendName` literal all work end-to-end.
5. `saiunit.lax.slice(q_ndonnx, ...)` raises `BackendError` with a message naming ndonnx.

## Full Multi-Backend Done Criteria

The full multi-backend spec is delivered when Phases 1, 2, and 3 are all merged and:

- `pytest saiunit/` passes with `saiunit[all]` installed.
- `Quantity.backend` returns one of `{numpy, jax, cupy, torch, dask, ndonnx}` for the matching mantissa.
- Each backend has a working `Quantity.to_<backend>(...)` method.
- `require_jax_backend` rejects every non-JAX backend with a message that names the actual backend.
- The user guide at `docs/getting_started/numpy_backend.md` documents all five additional backends (numpy intro + cupy + torch + dask + ndonnx).
