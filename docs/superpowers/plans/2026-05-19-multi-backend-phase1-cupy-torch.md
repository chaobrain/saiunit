# Multi-Backend Phase 1: CuPy + PyTorch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend saiunit's backend system from `{numpy, jax}` to `{numpy, jax, cupy, torch}` — dense-eager additions that validate the dispatch pattern.

**Architecture:** Closed-enum extension: add `is_cupy_array`/`is_torch_array` detectors, extend the `BackendName` Literal, extend `get_backend()` dispatch chain, add `to_cupy`/`to_torch` conversion paths with backend-specific kwargs (`device`, `dtype`). Both backends use `array_api_compat.cupy` / `array_api_compat.torch` for their xp namespaces. Lazy imports throughout; missing libraries cause `BackendError` (with install hint), never `ImportError`.

**Tech Stack:** Python 3.10+, jax, numpy, `array_api_compat>=1.9`, optionally `cupy-cuda12x>=13.0` and `torch>=2.0`. Tests via pytest with `importorskip`.

**Spec:** `docs/superpowers/specs/2026-05-19-multi-backend-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `saiunit/_backend.py` | Modify | Add lazy-import helper, two new detectors, extend `BackendName`, `get_backend` dispatch, `to_backend` branches, validation |
| `saiunit/_base_quantity.py` | Modify | Add `to_cupy`, `to_torch` methods; extend `backend` property chain |
| `saiunit/_jax_guard.py` | Modify | Broaden `require_jax_backend` to name the offending backend and reject foreign tensors |
| `saiunit/__init__.py` | Modify | Export new detectors |
| `saiunit/_backend_test.py` | Modify | Add tests for detectors, dispatch, conversion |
| `saiunit/_jax_guard_test.py` | Modify | Add rejection tests for cupy/torch-backed Quantity |
| `conftest.py` | Modify | Extend `backend` fixture with cupy/torch params (`importorskip`) |
| `pyproject.toml` | Modify | Add `cupy`, `torch`, `all` extras |
| `docs/getting_started/numpy_backend.md` | Modify | Add CuPy and PyTorch subsections |

No new files in Phase 1 — every addition extends an existing module.

---

## Task 1: Lazy-import helper

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing test**

Append to `saiunit/_backend_test.py`:

```python
def test_try_import_returns_module_when_present():
    from saiunit._backend import _try_import
    np_mod = _try_import("numpy")
    assert np_mod is not None
    assert hasattr(np_mod, "asarray")


def test_try_import_returns_none_when_missing():
    from saiunit._backend import _try_import
    assert _try_import("definitely_not_a_real_package_xyz") is None


def test_try_import_is_cached():
    from saiunit._backend import _try_import
    a = _try_import("numpy")
    b = _try_import("numpy")
    assert a is b
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py::test_try_import_returns_module_when_present -v
```

Expected: FAIL with `ImportError: cannot import name '_try_import' from 'saiunit._backend'`.

- [ ] **Step 3: Add the helper**

In `saiunit/_backend.py`, after the existing imports and before `BackendName`, add:

```python
import functools
import importlib


@functools.lru_cache(maxsize=None)
def _try_import(module_name: str):
    """Import ``module_name`` and return it, or ``None`` on ImportError.

    Results are cached so failed imports aren't retried on every call.
    Never raises.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_backend_test.py::test_try_import_returns_module_when_present saiunit/_backend_test.py::test_try_import_returns_none_when_missing saiunit/_backend_test.py::test_try_import_is_cached -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py
git commit -m "feat(backend): add lazy-import helper for optional backends"
```

---

## Task 2: Add `is_cupy_array` and `is_torch_array` detectors

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_backend_test.py`:

```python
def test_is_cupy_array_false_when_cupy_missing_or_not_cupy():
    from saiunit._backend import is_cupy_array
    # Non-cupy inputs always return False (works whether or not cupy is installed).
    assert is_cupy_array(np.array([1.0])) is False
    assert is_cupy_array(jnp.array([1.0])) is False
    assert is_cupy_array(1.0) is False


def test_is_cupy_array_true_when_available():
    cupy = pytest.importorskip("cupy")
    from saiunit._backend import is_cupy_array
    arr = cupy.array([1.0, 2.0])
    assert is_cupy_array(arr) is True


def test_is_torch_array_false_for_non_torch():
    from saiunit._backend import is_torch_array
    assert is_torch_array(np.array([1.0])) is False
    assert is_torch_array(jnp.array([1.0])) is False
    assert is_torch_array(1.0) is False


def test_is_torch_array_true_when_available():
    torch = pytest.importorskip("torch")
    from saiunit._backend import is_torch_array
    t = torch.tensor([1.0, 2.0])
    assert is_torch_array(t) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py -k "is_cupy_array or is_torch_array" -v
```

Expected: FAIL with `ImportError: cannot import name 'is_cupy_array'`.

- [ ] **Step 3: Add the detectors**

In `saiunit/_backend.py`, after the existing `is_jax_array` function, add:

```python
def is_cupy_array(x) -> bool:
    """Return True if ``x`` is a CuPy ndarray. False if CuPy is not installed."""
    cupy = _try_import("cupy")
    if cupy is None:
        return False
    return isinstance(x, cupy.ndarray)


def is_torch_array(x) -> bool:
    """Return True if ``x`` is a PyTorch tensor. False if PyTorch is not installed."""
    torch = _try_import("torch")
    if torch is None:
        return False
    return isinstance(x, torch.Tensor)
```

Update `__all__` (top of file) to add `"is_cupy_array"`, `"is_torch_array"`:

```python
__all__ = [
    "get_backend",
    "get_default_backend",
    "set_default_backend",
    "using_backend",
    "is_jax_array",
    "is_numpy_array",
    "is_cupy_array",
    "is_torch_array",
    "to_backend",
]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_backend_test.py -k "is_cupy_array or is_torch_array" -v
```

Expected: PASS. If `cupy` or `torch` is not installed, those specific tests will be SKIPPED (via `importorskip`) — that's correct.

- [ ] **Step 5: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py
git commit -m "feat(backend): add is_cupy_array and is_torch_array detectors"
```

---

## Task 3: Extend `BackendName` Literal

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing test**

Append to `saiunit/_backend_test.py`:

```python
def test_backend_name_includes_cupy_and_torch():
    from saiunit._backend import BackendName
    import typing
    args = typing.get_args(BackendName)
    assert "cupy" in args
    assert "torch" in args
    assert "numpy" in args
    assert "jax" in args
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest saiunit/_backend_test.py::test_backend_name_includes_cupy_and_torch -v
```

Expected: FAIL with `AssertionError: assert 'cupy' in ('numpy', 'jax')`.

- [ ] **Step 3: Extend the literal**

In `saiunit/_backend.py`, change:

```python
BackendName = Literal["numpy", "jax"]
```

to:

```python
BackendName = Literal["numpy", "jax", "cupy", "torch"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest saiunit/_backend_test.py::test_backend_name_includes_cupy_and_torch -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py
git commit -m "feat(backend): extend BackendName literal with cupy and torch"
```

---

## Task 4: Extend `get_backend` dispatch

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_backend_test.py`:

```python
def test_get_backend_cupy_only():
    cupy = pytest.importorskip("cupy")
    from saiunit._backend import get_backend
    import array_api_compat.cupy as expected
    xp = get_backend(cupy.array([1.0]))
    assert xp is expected


def test_get_backend_torch_only():
    torch = pytest.importorskip("torch")
    from saiunit._backend import get_backend
    import array_api_compat.torch as expected
    xp = get_backend(torch.tensor([1.0]))
    assert xp is expected


def test_get_backend_mixed_torch_jax_default_jax_wins():
    torch = pytest.importorskip("torch")
    from saiunit._backend import get_backend, set_default_backend
    set_default_backend(None)
    import jax.numpy as expected
    xp = get_backend(torch.tensor([1.0]), jnp.array([1.0]))
    assert xp is expected


def test_get_backend_mixed_with_torch_default():
    torch = pytest.importorskip("torch")
    from saiunit._backend import get_backend, set_default_backend
    set_default_backend("torch")
    try:
        import array_api_compat.torch as expected
        xp = get_backend(torch.tensor([1.0]), jnp.array([1.0]))
        assert xp is expected
    finally:
        set_default_backend(None)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py -k "get_backend_cupy or get_backend_torch or get_backend_mixed_torch" -v
```

Expected: FAIL (either `ValueError` from `set_default_backend("torch")`, or the dispatch returns jnp for cupy/torch inputs because the chain doesn't know about them).

- [ ] **Step 3: Extend `get_backend` and add xp cache**

In `saiunit/_backend.py`, replace the existing `_name_to_xp` function and `get_backend` function with:

```python
_XP_CACHE: dict[str, ModuleType] = {}


def _xp_for(name: BackendName) -> ModuleType:
    """Return (and cache) the xp namespace for ``name``."""
    cached = _XP_CACHE.get(name)
    if cached is not None:
        return cached
    if name == "numpy":
        mod = _numpy_xp
    elif name == "jax":
        mod = _jax_xp
    elif name == "cupy":
        if _try_import("cupy") is None:
            raise BackendError(
                "cupy backend requested but cupy is not installed. "
                "Install with: pip install saiunit[cupy]"
            )
        import array_api_compat.cupy as mod  # noqa: F811
    elif name == "torch":
        if _try_import("torch") is None:
            raise BackendError(
                "torch backend requested but torch is not installed. "
                "Install with: pip install saiunit[torch]"
            )
        import array_api_compat.torch as mod  # noqa: F811
    else:
        raise ValueError(f"unknown backend: {name!r}")
    _XP_CACHE[name] = mod
    return mod


def get_backend(*arrays_or_quantities) -> ModuleType:
    """Return the ``xp`` namespace appropriate for the given inputs.

    Detection order: numpy, jax, cupy, torch. On mixed inputs or no arrays,
    consults ``get_default_backend()``; falls back to jax.
    """
    from saiunit._base_quantity import Quantity  # local import to avoid cycle

    mantissas = [a.mantissa if isinstance(a, Quantity) else a for a in arrays_or_quantities]

    has_numpy = any(is_numpy_array(x) for x in mantissas)
    has_jax = any(is_jax_array(x) for x in mantissas)
    has_cupy = any(is_cupy_array(x) for x in mantissas)
    has_torch = any(is_torch_array(x) for x in mantissas)

    kinds = [name for name, has in
             [("numpy", has_numpy), ("jax", has_jax),
              ("cupy", has_cupy), ("torch", has_torch)] if has]

    if len(kinds) == 1:
        return _xp_for(kinds[0])

    default = _default_backend.get()
    if default is not None:
        return _xp_for(default)
    return _xp_for("jax")
```

You also need to add `BackendError` to the imports at the top of `_backend.py`:

```python
from saiunit._exceptions import BackendError
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_backend_test.py -k "get_backend" -v
```

Expected: PASS (all `get_backend_*` tests, including existing numpy/jax ones).

- [ ] **Step 5: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py
git commit -m "feat(backend): dispatch get_backend() to cupy and torch xp namespaces"
```

---

## Task 5: Extend `to_backend` with cupy branch

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_backend_test.py`:

```python
def test_to_backend_numpy_to_cupy():
    cupy = pytest.importorskip("cupy")
    from saiunit._backend import to_backend, is_cupy_array
    arr = np.array([1.0, 2.0])
    out = to_backend(arr, "cupy")
    assert is_cupy_array(out)
    assert cupy.allclose(out, cupy.asarray(arr))


def test_to_backend_cupy_noop():
    cupy = pytest.importorskip("cupy")
    from saiunit._backend import to_backend
    arr = cupy.array([1.0])
    out = to_backend(arr, "cupy")
    assert out is arr


def test_to_backend_cupy_with_device_kwarg():
    cupy = pytest.importorskip("cupy")
    from saiunit._backend import to_backend, is_cupy_array
    arr = np.array([1.0])
    out = to_backend(arr, "cupy", device=0)
    assert is_cupy_array(out)


def test_to_backend_numpy_rejects_unknown_kwargs():
    from saiunit._backend import to_backend
    with pytest.raises(TypeError, match="does not accept"):
        to_backend(np.array([1.0]), "numpy", device="cuda")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py -k "to_backend_numpy_to_cupy or to_backend_cupy or to_backend_numpy_rejects" -v
```

Expected: FAIL (either ValueError from `to_backend(x, "cupy")` not recognizing the name, or `TypeError: to_backend() got an unexpected keyword argument 'device'`).

- [ ] **Step 3: Extend `to_backend`**

In `saiunit/_backend.py`, replace the existing `to_backend` function with:

```python
def to_backend(x, name: BackendName, **kwargs):
    """Convert ``x`` to the given backend; no-op if already there.

    Backend-specific kwargs:
      - cupy: device
      - torch: device, dtype
    Other backends raise TypeError on any kwargs.
    """
    if name == "numpy":
        if kwargs:
            raise TypeError(f"to_backend(name='numpy') does not accept kwargs; got {sorted(kwargs)}")
        if is_numpy_array(x):
            return x
        return np.asarray(x)
    if name == "jax":
        if kwargs:
            raise TypeError(f"to_backend(name='jax') does not accept kwargs; got {sorted(kwargs)}")
        if is_jax_array(x):
            return x
        return jnp.asarray(x)
    if name == "cupy":
        cupy = _try_import("cupy")
        if cupy is None:
            raise BackendError(
                "cupy backend requested but cupy is not installed. "
                "Install with: pip install saiunit[cupy]"
            )
        unknown = set(kwargs) - {"device"}
        if unknown:
            raise TypeError(f"to_backend(name='cupy') does not accept {sorted(unknown)}")
        if is_cupy_array(x) and "device" not in kwargs:
            return x
        device = kwargs.get("device")
        if device is not None:
            with cupy.cuda.Device(device):
                return cupy.asarray(x)
        return cupy.asarray(x)
    if name == "torch":
        # Implemented in Task 6.
        raise NotImplementedError("torch branch added in Task 6")
    raise ValueError(f"backend must be one of 'numpy', 'jax', 'cupy', 'torch'; got {name!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_backend_test.py -k "to_backend" -v
```

Expected: PASS for all `to_backend_*` tests (existing numpy/jax tests still pass; new cupy tests pass or skip; the torch tests added in Task 6 don't exist yet).

- [ ] **Step 5: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py
git commit -m "feat(backend): add cupy branch to to_backend() with device kwarg"
```

---

## Task 6: Extend `to_backend` with torch branch and dtype mapping

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_backend_test.py`:

```python
def test_to_backend_numpy_to_torch():
    torch = pytest.importorskip("torch")
    from saiunit._backend import to_backend, is_torch_array
    arr = np.array([1.0, 2.0])
    out = to_backend(arr, "torch")
    assert is_torch_array(out)
    assert torch.allclose(out, torch.tensor([1.0, 2.0]))


def test_to_backend_torch_noop():
    torch = pytest.importorskip("torch")
    from saiunit._backend import to_backend
    t = torch.tensor([1.0])
    out = to_backend(t, "torch")
    assert out is t


def test_to_backend_torch_with_dtype_torch_native():
    torch = pytest.importorskip("torch")
    from saiunit._backend import to_backend
    out = to_backend(np.array([1.0, 2.0]), "torch", dtype=torch.float64)
    assert out.dtype == torch.float64


def test_to_backend_torch_with_dtype_numpy_mapped():
    torch = pytest.importorskip("torch")
    from saiunit._backend import to_backend
    out = to_backend(np.array([1.0, 2.0]), "torch", dtype=np.float64)
    assert out.dtype == torch.float64


def test_to_backend_torch_rejects_unknown_kwarg():
    pytest.importorskip("torch")
    from saiunit._backend import to_backend
    with pytest.raises(TypeError, match="does not accept"):
        to_backend(np.array([1.0]), "torch", chunks="auto")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py -k "to_backend_torch or to_backend_numpy_to_torch" -v
```

Expected: FAIL with `NotImplementedError: torch branch added in Task 6`.

- [ ] **Step 3: Replace the torch placeholder with a real implementation**

In `saiunit/_backend.py`, replace the `if name == "torch":` block (currently `raise NotImplementedError`) with:

```python
    if name == "torch":
        torch = _try_import("torch")
        if torch is None:
            raise BackendError(
                "torch backend requested but torch is not installed. "
                "Install with: pip install saiunit[torch]"
            )
        unknown = set(kwargs) - {"device", "dtype"}
        if unknown:
            raise TypeError(f"to_backend(name='torch') does not accept {sorted(unknown)}")
        # Translate numpy dtype to torch dtype if needed.
        dtype = kwargs.get("dtype")
        if dtype is not None and not isinstance(dtype, torch.dtype):
            dtype = _numpy_to_torch_dtype(dtype, torch)
        device = kwargs.get("device")
        if is_torch_array(x) and not kwargs:
            return x
        # torch.as_tensor shares memory where possible; we accept that.
        return torch.as_tensor(x, device=device, dtype=dtype)
```

Also add the dtype-mapping helper just above `to_backend`:

```python
_NUMPY_TO_TORCH_DTYPE = {
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "bool": "bool",
    "complex64": "complex64",
    "complex128": "complex128",
}


def _numpy_to_torch_dtype(np_dtype, torch_mod):
    """Translate a numpy dtype (or np.dtype-like) to a torch dtype."""
    name = np.dtype(np_dtype).name
    torch_name = _NUMPY_TO_TORCH_DTYPE.get(name)
    if torch_name is None:
        raise TypeError(f"no torch dtype mapping for numpy dtype {name!r}")
    return getattr(torch_mod, torch_name)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_backend_test.py -k "to_backend_torch or to_backend_numpy_to_torch" -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py
git commit -m "feat(backend): add torch branch to to_backend() with device/dtype kwargs"
```

---

## Task 7: Extend `using_backend` and `set_default_backend` validation

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing test (and update existing test)**

The existing test `test_set_default_backend_rejects_invalid` asserts that `"torch"` is rejected — that assumption is now wrong. Edit it to assert rejection of a truly unknown name:

```python
def test_set_default_backend_rejects_invalid():
    with pytest.raises(ValueError, match="must be 'numpy', 'jax', 'cupy', 'torch', or None"):
        set_default_backend("notabackend")
```

Then append new positive tests:

```python
def test_set_default_backend_accepts_cupy():
    from saiunit._backend import set_default_backend, get_default_backend
    set_default_backend("cupy")
    try:
        assert get_default_backend() == "cupy"
    finally:
        set_default_backend(None)


def test_set_default_backend_accepts_torch():
    from saiunit._backend import set_default_backend, get_default_backend
    set_default_backend("torch")
    try:
        assert get_default_backend() == "torch"
    finally:
        set_default_backend(None)


def test_using_backend_accepts_cupy():
    from saiunit._backend import using_backend, get_default_backend
    with using_backend("cupy"):
        assert get_default_backend() == "cupy"


def test_using_backend_accepts_torch():
    from saiunit._backend import using_backend, get_default_backend
    with using_backend("torch"):
        assert get_default_backend() == "torch"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py -k "set_default_backend or using_backend" -v
```

Expected: the new positive tests FAIL with `ValueError: default backend must be 'numpy', 'jax', or None; got 'cupy'`.

- [ ] **Step 3: Broaden validation**

In `saiunit/_backend.py`, replace `set_default_backend` body:

```python
def set_default_backend(name: Optional[BackendName]) -> None:
    if name not in ("numpy", "jax", "cupy", "torch", None):
        raise ValueError(
            f"default backend must be 'numpy', 'jax', 'cupy', 'torch', or None; got {name!r}"
        )
    _default_backend.set(name)
```

And `using_backend` body:

```python
@contextmanager
def using_backend(name: BackendName) -> Iterator[None]:
    if name not in ("numpy", "jax", "cupy", "torch"):
        raise ValueError(f"backend must be 'numpy', 'jax', 'cupy', or 'torch'; got {name!r}")
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
git commit -m "feat(backend): accept cupy/torch in using_backend() and set_default_backend()"
```

---

## Task 8: Add `Quantity.to_cupy(*, device=None)` method

**Files:**
- Modify: `saiunit/_base_quantity.py`
- Test: `saiunit/_base_quantity_test.py` (append; verify path with `grep -n "def test_" saiunit/_base_quantity_test.py | tail`)

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_base_quantity_test.py`:

```python
def test_quantity_to_cupy_basic():
    cupy = pytest.importorskip("cupy")
    import saiunit as u
    from saiunit._backend import is_cupy_array
    q = u.Quantity(np.array([1.0, 2.0]), unit=u.meter)
    q2 = q.to_cupy()
    assert is_cupy_array(q2.mantissa)
    assert q2.unit == u.meter


def test_quantity_to_cupy_noop_when_already_cupy():
    cupy = pytest.importorskip("cupy")
    import saiunit as u
    q = u.Quantity(cupy.array([1.0]), unit=u.meter)
    q2 = q.to_cupy()
    assert q2 is q


def test_quantity_to_cupy_with_device():
    cupy = pytest.importorskip("cupy")
    import saiunit as u
    from saiunit._backend import is_cupy_array
    q = u.Quantity(np.array([1.0]), unit=u.meter)
    q2 = q.to_cupy(device=0)
    assert is_cupy_array(q2.mantissa)
```

You may need to add `import numpy as np` and `import pytest` near the top of the test file if not already present.

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_base_quantity_test.py -k "to_cupy" -v
```

Expected: FAIL with `AttributeError: 'Quantity' object has no attribute 'to_cupy'`.

- [ ] **Step 3: Add the method**

In `saiunit/_base_quantity.py`, immediately after the existing `to_jax` method (around line 1164), add:

```python
def to_cupy(self, *, device=None) -> 'Quantity':
    """Return a new Quantity with mantissa converted to a ``cupy.ndarray``.

    No-op (returns ``self``) if the mantissa is already a CuPy array and no
    ``device`` was specified.
    """
    from saiunit._backend import is_cupy_array, to_backend
    if is_cupy_array(self._mantissa) and device is None:
        return self
    kwargs = {} if device is None else {"device": device}
    return Quantity(to_backend(self._mantissa, "cupy", **kwargs), unit=self.unit)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_base_quantity_test.py -k "to_cupy" -v
```

Expected: PASS (or SKIP if cupy not installed).

- [ ] **Step 5: Commit**

```bash
git add saiunit/_base_quantity.py saiunit/_base_quantity_test.py
git commit -m "feat(quantity): add Quantity.to_cupy(device=None)"
```

---

## Task 9: Add `Quantity.to_torch(*, device=None, dtype=None)` method

**Files:**
- Modify: `saiunit/_base_quantity.py`
- Test: `saiunit/_base_quantity_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_base_quantity_test.py`:

```python
def test_quantity_to_torch_basic():
    torch = pytest.importorskip("torch")
    import saiunit as u
    from saiunit._backend import is_torch_array
    q = u.Quantity(np.array([1.0, 2.0]), unit=u.meter)
    q2 = q.to_torch()
    assert is_torch_array(q2.mantissa)
    assert q2.unit == u.meter


def test_quantity_to_torch_noop_when_already_torch():
    torch = pytest.importorskip("torch")
    import saiunit as u
    q = u.Quantity(torch.tensor([1.0]), unit=u.meter)
    q2 = q.to_torch()
    assert q2 is q


def test_quantity_to_torch_with_dtype_numpy_mapped():
    torch = pytest.importorskip("torch")
    import saiunit as u
    q = u.Quantity(np.array([1.0]), unit=u.meter)
    q2 = q.to_torch(dtype=np.float64)
    assert q2.mantissa.dtype == torch.float64


def test_quantity_to_torch_preserves_requires_grad_chain():
    torch = pytest.importorskip("torch")
    import saiunit as u
    t = torch.tensor([1.0, 2.0], requires_grad=True)
    q = u.Quantity(t, unit=u.meter)
    q2 = q.to_torch()  # noop — still the same tensor
    assert q2.mantissa.requires_grad is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_base_quantity_test.py -k "to_torch" -v
```

Expected: FAIL with `AttributeError: 'Quantity' object has no attribute 'to_torch'`.

- [ ] **Step 3: Add the method**

In `saiunit/_base_quantity.py`, immediately after the new `to_cupy` method, add:

```python
def to_torch(self, *, device=None, dtype=None) -> 'Quantity':
    """Return a new Quantity with mantissa converted to a ``torch.Tensor``.

    No-op (returns ``self``) if the mantissa is already a torch tensor and
    no ``device``/``dtype`` was specified. ``dtype`` accepts either a torch
    dtype (e.g. ``torch.float32``) or a numpy dtype (e.g. ``np.float32``).
    """
    from saiunit._backend import is_torch_array, to_backend
    if is_torch_array(self._mantissa) and device is None and dtype is None:
        return self
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    return Quantity(to_backend(self._mantissa, "torch", **kwargs), unit=self.unit)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_base_quantity_test.py -k "to_torch" -v
```

Expected: PASS (or SKIP if torch not installed).

- [ ] **Step 5: Commit**

```bash
git add saiunit/_base_quantity.py saiunit/_base_quantity_test.py
git commit -m "feat(quantity): add Quantity.to_torch(device=None, dtype=None)"
```

---

## Task 10: Extend `Quantity.backend` property chain

**Files:**
- Modify: `saiunit/_base_quantity.py`
- Test: `saiunit/_base_quantity_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_base_quantity_test.py`:

```python
def test_quantity_backend_cupy():
    cupy = pytest.importorskip("cupy")
    import saiunit as u
    q = u.Quantity(cupy.array([1.0]), unit=u.meter)
    assert q.backend == "cupy"


def test_quantity_backend_torch():
    torch = pytest.importorskip("torch")
    import saiunit as u
    q = u.Quantity(torch.tensor([1.0]), unit=u.meter)
    assert q.backend == "torch"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_base_quantity_test.py -k "quantity_backend_cupy or quantity_backend_torch" -v
```

Expected: FAIL — `assert 'jax' == 'cupy'` because the current property falls through to `"jax"`.

- [ ] **Step 3: Extend the property**

In `saiunit/_base_quantity.py`, replace the existing `backend` property (around line 1140) with:

```python
@property
def backend(self) -> str:
    """The backend of the underlying mantissa: one of
    ``'numpy'``, ``'jax'``, ``'cupy'``, ``'torch'``."""
    from saiunit._backend import (
        is_numpy_array, is_cupy_array, is_torch_array,
    )
    m = self._mantissa
    if is_numpy_array(m):
        return "numpy"
    if is_cupy_array(m):
        return "cupy"
    if is_torch_array(m):
        return "torch"
    return "jax"  # jax is the fallthrough (preserves existing behavior)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_base_quantity_test.py -k "quantity_backend" -v
```

Expected: PASS for all four (numpy, jax, cupy, torch). The existing numpy/jax tests must still pass.

- [ ] **Step 5: Commit**

```bash
git add saiunit/_base_quantity.py saiunit/_base_quantity_test.py
git commit -m "feat(quantity): extend Quantity.backend property with cupy and torch"
```

---

## Task 11: Broaden `require_jax_backend` error message and reject foreign tensors

**Files:**
- Modify: `saiunit/_jax_guard.py`
- Test: `saiunit/_jax_guard_test.py`

- [ ] **Step 1: Write the failing tests**

The existing test asserts a numpy-backed Quantity is rejected. Append:

```python
def test_require_jax_raises_for_torch_quantity():
    torch = pytest.importorskip("torch")
    q = u.Quantity(torch.tensor([1.0]), unit=u.meter)
    with pytest.raises(u.BackendError, match="torch-backed Quantity"):
        require_jax_backend("test_fn", q)


def test_require_jax_raises_for_cupy_quantity():
    cupy = pytest.importorskip("cupy")
    q = u.Quantity(cupy.array([1.0]), unit=u.meter)
    with pytest.raises(u.BackendError, match="cupy-backed Quantity"):
        require_jax_backend("test_fn", q)


def test_require_jax_message_for_numpy_quantity_names_backend():
    q = u.Quantity(np.array([1.0]), unit=u.meter)
    with pytest.raises(u.BackendError, match="numpy-backed Quantity"):
        require_jax_backend("test_fn", q)


def test_require_jax_rejects_bare_torch_tensor():
    torch = pytest.importorskip("torch")
    t = torch.tensor([1.0])
    with pytest.raises(u.BackendError, match="torch"):
        require_jax_backend("test_fn", t)


def test_require_jax_rejects_bare_cupy_array():
    cupy = pytest.importorskip("cupy")
    arr = cupy.array([1.0])
    with pytest.raises(u.BackendError, match="cupy"):
        require_jax_backend("test_fn", arr)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_jax_guard_test.py -v
```

Expected: the new tests FAIL because the current guard only checks numpy-backed Quantity and tolerates bare non-numpy arrays.

- [ ] **Step 3: Broaden the guard**

In `saiunit/_jax_guard.py`, replace `require_jax_backend` with:

```python
def require_jax_backend(func_name: str, *quantities_or_arrays) -> None:
    """Raise :class:`BackendError` if any input is not jax-compatible.

    - Quantity wrapping non-jax mantissa → reject, naming the backend.
    - Bare cupy / torch arrays → reject (jax cannot lift them).
    - Bare numpy arrays, python scalars → tolerated (jax auto-lifts).
    """
    from saiunit._base_quantity import Quantity
    from saiunit._backend import (
        is_numpy_array, is_jax_array, is_cupy_array, is_torch_array,
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
        # Bare arrays: reject cupy/torch; tolerate numpy + jax + scalars.
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
        # Anything else (numpy ndarray, jax array, python scalar) is fine.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_jax_guard_test.py -v
```

Expected: PASS for all tests, including the existing ones.

- [ ] **Step 5: Commit**

```bash
git add saiunit/_jax_guard.py saiunit/_jax_guard_test.py
git commit -m "feat(guard): broaden require_jax_backend to name backend and reject foreign tensors"
```

---

## Task 12: Export new detectors in `saiunit/__init__.py`

**Files:**
- Modify: `saiunit/__init__.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing test**

Append to `saiunit/_backend_test.py`:

```python
def test_top_level_exports_new_detectors():
    import saiunit as u
    assert hasattr(u, "is_cupy_array")
    assert hasattr(u, "is_torch_array")
    assert "is_cupy_array" in u.__all__
    assert "is_torch_array" in u.__all__
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest saiunit/_backend_test.py::test_top_level_exports_new_detectors -v
```

Expected: FAIL — `AttributeError: module 'saiunit' has no attribute 'is_cupy_array'`.

- [ ] **Step 3: Add exports**

In `saiunit/__init__.py`, update the import block from `._backend` (around line 75):

```python
from ._backend import (
    get_default_backend,
    is_cupy_array,
    is_jax_array,
    is_numpy_array,
    is_torch_array,
    set_default_backend,
    using_backend,
)
```

And in the `__all__` list, find the `# _backend` section and add:

```python
              # _backend
              'get_default_backend',
              'set_default_backend',
              'using_backend',
              'is_jax_array',
              'is_numpy_array',
              'is_cupy_array',
              'is_torch_array',
              'get_or_create_dimension',
              'get_dim_for_display',
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest saiunit/_backend_test.py::test_top_level_exports_new_detectors -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add saiunit/__init__.py saiunit/_backend_test.py
git commit -m "feat(saiunit): export is_cupy_array and is_torch_array at top level"
```

---

## Task 13: Add `cupy`, `torch`, `all` extras to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`
- Test: manual / installation check

- [ ] **Step 1: Read the existing `[project.optional-dependencies]` block**

Confirm current contents (lines 70-75):

```toml
[project.optional-dependencies]
testing = ['pytest', 'brainstate']
cpu = ["jax[cpu]"]
cuda12 = ["jax[cuda12]"]
cuda13 = ["jax[cuda13]"]
tpu = ["jax[tpu]"]
```

- [ ] **Step 2: Add the new extras**

Append to the `[project.optional-dependencies]` block:

```toml
cupy = ["cupy-cuda12x>=13.0"]
torch = ["torch>=2.0"]
all = ["saiunit[cupy,torch]"]
```

The full block becomes:

```toml
[project.optional-dependencies]
testing = ['pytest', 'brainstate']
cpu = ["jax[cpu]"]
cuda12 = ["jax[cuda12]"]
cuda13 = ["jax[cuda13]"]
tpu = ["jax[tpu]"]
cupy = ["cupy-cuda12x>=13.0"]
torch = ["torch>=2.0"]
all = ["saiunit[cupy,torch]"]
```

- [ ] **Step 3: Verify the file parses**

```bash
python -c "import tomllib; tomllib.loads(open('pyproject.toml').read())"
```

Expected: no output (exit 0). If your Python is <3.11, use `pip install tomli && python -c "import tomli; tomli.loads(open('pyproject.toml').read())"`.

- [ ] **Step 4: Verify a dry-run install resolves the new extras**

```bash
pip install --dry-run -e '.[torch]' 2>&1 | head -20
```

Expected: no errors; torch is resolved as a dependency. If torch is not available for your Python/platform, skip this step and rely on CI.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build(deps): add cupy, torch, and all optional extras"
```

---

## Task 14: Extend `conftest.py` `backend` fixture

**Files:**
- Modify: `conftest.py`
- Test: discovery via existing parametrized tests

- [ ] **Step 1: Write the failing test**

Append to `saiunit/_backend_parametrize_test.py`:

```python
def test_backend_fixture_includes_cupy_and_torch(backend):
    """If torch is installed, the fixture must include 'torch'.
    If cupy is installed, it must include 'cupy'.

    This test just asserts the fixture parameter is one of the known names —
    pytest's parametrize machinery is what actually exercises each.
    """
    assert backend in {"numpy", "jax", "cupy", "torch"}
```

Then verify that the fixture actually parametrizes over the new backends by running with `--collect-only`:

```bash
pytest saiunit/_backend_parametrize_test.py::test_backend_fixture_includes_cupy_and_torch --collect-only -q
```

Before the fixture change, you'll see two collected items (`[numpy]`, `[jax]`).

- [ ] **Step 2: Confirm the assertion in collect-only output**

Currently you should see 2 items, not 4. That's the gap.

- [ ] **Step 3: Extend the fixture**

Replace the body of `conftest.py` `backend` fixture with:

```python
@pytest.fixture(params=["numpy", "jax", "cupy", "torch"])
def backend(request):
    """Set the saiunit default backend for the duration of the test.

    Skips automatically when the requested backend's library isn't installed.
    """
    if request.param == "cupy":
        pytest.importorskip("cupy")
    elif request.param == "torch":
        pytest.importorskip("torch")
    with u.using_backend(request.param):
        yield request.param
```

- [ ] **Step 4: Verify collection and existing tests still pass**

```bash
pytest saiunit/_backend_parametrize_test.py --collect-only -q
pytest saiunit/_backend_parametrize_test.py -v
```

Expected: 4 items per test now (numpy, jax, cupy, torch). cupy/torch tests are SKIPPED if libraries are missing; otherwise PASS.

- [ ] **Step 5: Commit**

```bash
git add conftest.py saiunit/_backend_parametrize_test.py
git commit -m "test(backend): extend backend fixture to parametrize over cupy and torch"
```

---

## Task 15: Add cross-backend smoke tests

**Files:**
- Modify: `saiunit/_backend_parametrize_test.py`

- [ ] **Step 1: Write the new smoke tests**

Append to `saiunit/_backend_parametrize_test.py`:

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


def test_linalg_norm_on_each_backend(backend):
    """saiunit.linalg.norm returns a scalar of the active backend."""
    q = u.Quantity([3.0, 4.0], unit=meter)
    n = u.linalg.norm(q)
    # Should be 5 meters regardless of backend.
    assert n.unit == meter
    assert float(n.mantissa) == 5.0


def test_to_method_round_trip_on_each_backend(backend):
    """Convert to numpy and back; mantissa values preserved."""
    q = u.Quantity([1.0, 2.0, 3.0], unit=meter)
    q_np = q.to_numpy()
    assert np.allclose(q_np.mantissa, [1.0, 2.0, 3.0])
    assert q_np.unit == meter
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest saiunit/_backend_parametrize_test.py -v
```

Expected: each test runs 4 times (numpy/jax/cupy/torch). cupy/torch SKIP if libs missing. All available backends PASS.

If a test fails on cupy or torch, the failure points to a real bug in earlier tasks — diagnose and fix, do not skip.

- [ ] **Step 3: Commit**

```bash
git add saiunit/_backend_parametrize_test.py
git commit -m "test(backend): add math/linalg/conversion smoke tests across backends"
```

---

## Task 16: Update the user guide

**Files:**
- Modify: `docs/getting_started/numpy_backend.md`

- [ ] **Step 1: Add CuPy and PyTorch subsections**

Open `docs/getting_started/numpy_backend.md`. After the "NumPy ufunc interop" section (the last section), append:

````markdown
## CuPy (GPU)

`saiunit` accepts `cupy.ndarray` mantissas via the optional `cupy` extra:

```bash
pip install saiunit[cupy]
```

```python
import cupy
import saiunit as u

q = u.Quantity(cupy.array([1.0, 2.0, 3.0]), unit=u.meter)
print(q.backend)            # 'cupy'
print((q + q).backend)      # 'cupy'
print(u.math.sin(q / u.meter))  # cupy.ndarray
```

Convert from another backend with `Quantity.to_cupy(device=...)`:

```python
q_cpu = u.Quantity([1.0, 2.0], unit=u.meter)
q_gpu = q_cpu.to_cupy(device=0)
```

CuPy is GPU-only; the import will fail without a CUDA installation. saiunit
raises `BackendError` (not `ImportError`) when CuPy is missing.

## PyTorch

`saiunit` accepts `torch.Tensor` mantissas via the optional `torch` extra:

```bash
pip install saiunit[torch]
```

```python
import torch
import saiunit as u

q = u.Quantity(torch.tensor([1.0, 2.0, 3.0]), unit=u.meter)
print(q.backend)        # 'torch'
print((q + q).backend)  # 'torch'
```

Convert with `Quantity.to_torch(device=..., dtype=...)`. `dtype` accepts a
torch dtype (e.g. `torch.float32`) or a numpy dtype (e.g. `np.float32`):

```python
q_cpu  = u.Quantity([1.0, 2.0], unit=u.meter)
q_cuda = q_cpu.to_torch(device='cuda')
q_f64  = q_cpu.to_torch(dtype=torch.float64)
```

### Gradients

Wrapping a tensor with `requires_grad=True` preserves the autograd graph
through saiunit operations, but `saiunit.autograd.grad` itself remains
JAX-only. Use `torch.autograd.grad` on the mantissa for backward passes:

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
q = u.Quantity(x, unit=u.meter) * 2.0
loss = q.mantissa.sum()
grads = torch.autograd.grad(loss, x)
```
````

- [ ] **Step 2: Verify the file renders**

```bash
python -c "import pathlib; assert 'CuPy (GPU)' in pathlib.Path('docs/getting_started/numpy_backend.md').read_text()"
```

Expected: no output (exit 0).

- [ ] **Step 3: Commit**

```bash
git add docs/getting_started/numpy_backend.md
git commit -m "docs: add CuPy and PyTorch backend sections to user guide"
```

---

## Task 17: Install torch in CI workflows

**Files:**
- Modify: `.github/workflows/CI.yml`

The torch backend's tests are currently `importorskip`-ped in CI because no
installer step adds torch. To actually exercise the torch backend on CI, add
torch to the install step of each platform job.

- [ ] **Step 1: Add a torch-install step to the Linux job**

In `.github/workflows/CI.yml`, find the `test_linux` job's "Install dependencies"
step (around line 43). Immediately after the `pip install . --no-cache-dir` line,
add a new line:

```yaml
          pip install torch --no-cache-dir
```

The full step becomes:

```yaml
      - name: Install dependencies
        run: |
          python -m pip cache purge
          python -m pip install --upgrade pip setuptools  --no-cache-dir
          python -m pip install -r requirements-dev.txt  --no-cache-dir
          pip install . --no-cache-dir
          pip install torch --no-cache-dir
```

- [ ] **Step 2: Repeat for macOS and Windows jobs**

Repeat the same edit in the `test_macos` and `test_windows` job blocks
(around lines 77-82 and 109-114 respectively). Each platform's
"Install dependencies" step gains the same final `pip install torch --no-cache-dir`
line.

- [ ] **Step 3: Note cupy is intentionally skipped**

Do NOT add cupy to any CI job. GitHub Actions runners do not have CUDA;
cupy install will fail. Cupy tests `importorskip`-pe gracefully — that's
the intended CI behavior. A self-hosted CUDA runner, if added later, is
the right place for cupy CI.

- [ ] **Step 4: Verify the YAML parses**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/CI.yml'))"
```

Expected: no output (exit 0). If your system lacks `pyyaml`, install it
first: `pip install pyyaml`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/CI.yml
git commit -m "ci: install torch on all platforms to exercise the torch backend"
```

---

## Task 18: Final integration verification

**Files:** none

- [ ] **Step 1: Run the full saiunit test suite**

```bash
pytest saiunit/ -x -q
```

Expected: all tests pass. cupy/torch-specific tests SKIP if those libraries are not installed.

- [ ] **Step 2: With torch installed, run again**

```bash
pip install torch
pytest saiunit/ -k "torch or backend" -v
```

Expected: every previously-skipped torch test now PASSes.

- [ ] **Step 3: Quick interactive smoke test**

```bash
python -c "
import torch, numpy as np
import saiunit as u
q = u.Quantity(torch.tensor([3.0, 4.0]), unit=u.meter)
print('backend:', q.backend)
print('norm:', u.linalg.norm(q))
print('to_jax:', q.to_jax().backend)
try:
    u.lax.slice(q, (0,), (1,))
except u.BackendError as e:
    print('expected BackendError:', e)
"
```

Expected output (something like):
```
backend: torch
norm: 5.0 * meter
to_jax: jax
expected BackendError: ...slice requires the jax backend; got torch-backed Quantity...
```

- [ ] **Step 4: Push the branch**

```bash
git push -u origin <your-branch-name>
```

(Replace `<your-branch-name>` with whatever branch you've been working on.)

- [ ] **Step 5: Open the pull request**

Use the project's standard PR template. Title:

> `feat(backend): add CuPy and PyTorch backends (Phase 1 of multi-backend support)`

Body should reference the spec at `docs/superpowers/specs/2026-05-19-multi-backend-design.md` and the plan at `docs/superpowers/plans/2026-05-19-multi-backend-phase1-cupy-torch.md`. After CI passes and review, merge before starting Phase 2.

---

## Done Criteria

Phase 1 is complete when:
1. All 18 tasks above are checked off.
2. `pytest saiunit/` passes on a machine with no cupy/torch installed (all tests skip gracefully).
3. `pytest saiunit/` passes on a machine with torch installed (torch tests run and pass).
4. The interactive smoke test in Task 17 Step 3 produces the expected output.
5. The user guide additions render correctly in the Sphinx build.
