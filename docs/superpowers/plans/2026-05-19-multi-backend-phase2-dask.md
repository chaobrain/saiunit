# Multi-Backend Phase 2: Dask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Phase 1 (`docs/superpowers/plans/2026-05-19-multi-backend-phase1-cupy-torch.md`) must be merged before starting Phase 2.

**Goal:** Add Dask as a saiunit backend, with laziness preserved end-to-end. No saiunit operation may materialize a dask array unless the user explicitly requests it (e.g. `q.to_numpy()`).

**Architecture:** Same closed-enum extension pattern as Phase 1 (`is_dask_array`, dispatch chain, `to_dask(chunks=...)`). Plus a **laziness audit** of `_base_quantity.py` to find every place a mantissa is implicitly materialized (`bool()`, `int()`, `float()`, `len()`, `np.asarray()`, `__repr__`'s array formatter). Each finding is either guarded with a clear `BackendError` (telling users to call `.compute()` first) or restructured to use `xp.*` operations that dask supports.

**Tech Stack:** Python 3.10+, jax, numpy, `array_api_compat>=1.9`, optionally `dask[array]>=2024.1`. Tests via pytest with `importorskip`.

**Spec:** `docs/superpowers/specs/2026-05-19-multi-backend-design.md` (Section 4.3 and 8.2).

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `saiunit/_backend.py` | Modify | Add `is_dask_array`, extend `BackendName`, `get_backend`, `to_backend`, validation |
| `saiunit/_base_quantity.py` | Modify | Add `to_dask`, extend `backend` property, add `_repr_mantissa_lazy_safe`, fix audit findings |
| `saiunit/_jax_guard.py` | Modify | Reject dask arrays in `require_jax_backend` |
| `saiunit/__init__.py` | Modify | Export `is_dask_array` |
| `saiunit/_backend_test.py` | Modify | Tests for detector, dispatch, conversion |
| `saiunit/_jax_guard_test.py` | Modify | Rejection tests for dask |
| `saiunit/_base_quantity_test.py` | Modify | Tests for `to_dask`, `backend == 'dask'`, lazy-safe repr |
| `saiunit/_dask_laziness_test.py` | Create | Dedicated tests that the dask backend never triggers `.compute()` |
| `conftest.py` | Modify | Add `dask` to fixture params |
| `pyproject.toml` | Modify | Add `dask` extra; update `all` to include it |
| `docs/getting_started/numpy_backend.md` | Modify | Add "Dask: lazy semantics" subsection |

---

## Task 1: Add `is_dask_array` detector

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_backend_test.py`:

```python
def test_is_dask_array_false_for_non_dask():
    from saiunit._backend import is_dask_array
    assert is_dask_array(np.array([1.0])) is False
    assert is_dask_array(jnp.array([1.0])) is False
    assert is_dask_array(1.0) is False


def test_is_dask_array_true_when_available():
    da = pytest.importorskip("dask.array")
    from saiunit._backend import is_dask_array
    arr = da.from_array(np.array([1.0, 2.0]), chunks=1)
    assert is_dask_array(arr) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py -k "is_dask_array" -v
```

Expected: FAIL with `ImportError: cannot import name 'is_dask_array'`.

- [ ] **Step 3: Add the detector**

In `saiunit/_backend.py`, after `is_torch_array`, add:

```python
def is_dask_array(x) -> bool:
    """Return True if ``x`` is a dask Array. False if dask is not installed."""
    da = _try_import("dask.array")
    if da is None:
        return False
    return isinstance(x, da.Array)
```

Update `__all__` to include `"is_dask_array"`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_backend_test.py -k "is_dask_array" -v
```

Expected: PASS (or SKIP if dask not installed).

- [ ] **Step 5: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py
git commit -m "feat(backend): add is_dask_array detector"
```

---

## Task 2: Extend `BackendName` Literal and dispatch

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_backend_test.py`:

```python
def test_backend_name_includes_dask():
    from saiunit._backend import BackendName
    import typing
    assert "dask" in typing.get_args(BackendName)


def test_get_backend_dask_only():
    da = pytest.importorskip("dask.array")
    from saiunit._backend import get_backend
    import array_api_compat.dask.array as expected
    xp = get_backend(da.from_array(np.array([1.0]), chunks=1))
    assert xp is expected


def test_get_backend_dask_default_for_mixed():
    da = pytest.importorskip("dask.array")
    from saiunit._backend import get_backend, set_default_backend
    set_default_backend("dask")
    try:
        import array_api_compat.dask.array as expected
        xp = get_backend(da.from_array(np.array([1.0]), chunks=1), jnp.array([1.0]))
        assert xp is expected
    finally:
        set_default_backend(None)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py -k "backend_name_includes_dask or get_backend_dask" -v
```

Expected: FAIL — `assert 'dask' in (...)` and / or `ValueError: default backend must be ...`.

- [ ] **Step 3: Extend `BackendName`, `_xp_for`, and `get_backend`**

In `saiunit/_backend.py`:

Change:
```python
BackendName = Literal["numpy", "jax", "cupy", "torch"]
```
to:
```python
BackendName = Literal["numpy", "jax", "cupy", "torch", "dask"]
```

In `_xp_for`, add a branch:
```python
    elif name == "dask":
        if _try_import("dask.array") is None:
            raise BackendError(
                "dask backend requested but dask is not installed. "
                "Install with: pip install saiunit[dask]"
            )
        import array_api_compat.dask.array as mod  # noqa: F811
```

In `get_backend`, extend the detection chain:
```python
    has_dask = any(is_dask_array(x) for x in mantissas)

    kinds = [name for name, has in
             [("numpy", has_numpy), ("jax", has_jax),
              ("cupy", has_cupy), ("torch", has_torch),
              ("dask", has_dask)] if has]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_backend_test.py -k "backend_name or get_backend" -v
```

Expected: PASS for all (numpy/jax/cupy/torch/dask).

- [ ] **Step 5: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py
git commit -m "feat(backend): dispatch get_backend() to dask xp namespace"
```

---

## Task 3: Extend `to_backend` with dask branch

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_backend_test.py`:

```python
def test_to_backend_numpy_to_dask_default_chunks():
    da = pytest.importorskip("dask.array")
    from saiunit._backend import to_backend, is_dask_array
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    out = to_backend(arr, "dask")
    assert is_dask_array(out)
    assert tuple(out.compute()) == (1.0, 2.0, 3.0, 4.0)


def test_to_backend_numpy_to_dask_custom_chunks():
    da = pytest.importorskip("dask.array")
    from saiunit._backend import to_backend
    arr = np.arange(8, dtype=np.float64)
    out = to_backend(arr, "dask", chunks=2)
    # 8 elements, chunks of 2 → 4 chunks
    assert out.numblocks == (4,)


def test_to_backend_dask_noop():
    da = pytest.importorskip("dask.array")
    from saiunit._backend import to_backend
    arr = da.from_array(np.array([1.0]), chunks=1)
    out = to_backend(arr, "dask")
    assert out is arr


def test_to_backend_dask_rejects_unknown_kwarg():
    pytest.importorskip("dask.array")
    from saiunit._backend import to_backend
    with pytest.raises(TypeError, match="does not accept"):
        to_backend(np.array([1.0]), "dask", device="cuda")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py -k "to_backend.*dask" -v
```

Expected: FAIL with `ValueError: backend must be one of ...; got 'dask'`.

- [ ] **Step 3: Add the dask branch**

In `saiunit/_backend.py`, in the `to_backend` function, add a branch before the final `raise ValueError`:

```python
    if name == "dask":
        da = _try_import("dask.array")
        if da is None:
            raise BackendError(
                "dask backend requested but dask is not installed. "
                "Install with: pip install saiunit[dask]"
            )
        unknown = set(kwargs) - {"chunks"}
        if unknown:
            raise TypeError(f"to_backend(name='dask') does not accept {sorted(unknown)}")
        if is_dask_array(x) and "chunks" not in kwargs:
            return x
        chunks = kwargs.get("chunks", "auto")
        return da.from_array(x, chunks=chunks)
```

And update the final error message to include 'dask':

```python
    raise ValueError(f"backend must be one of 'numpy', 'jax', 'cupy', 'torch', 'dask'; got {name!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_backend_test.py -k "to_backend" -v
```

Expected: PASS (or SKIP if dask not installed).

- [ ] **Step 5: Commit**

```bash
git add saiunit/_backend.py saiunit/_backend_test.py
git commit -m "feat(backend): add dask branch to to_backend() with chunks kwarg"
```

---

## Task 4: Extend `using_backend` / `set_default_backend` validation

**Files:**
- Modify: `saiunit/_backend.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing test**

Append to `saiunit/_backend_test.py`:

```python
def test_set_default_backend_accepts_dask():
    from saiunit._backend import set_default_backend, get_default_backend
    set_default_backend("dask")
    try:
        assert get_default_backend() == "dask"
    finally:
        set_default_backend(None)


def test_using_backend_accepts_dask():
    from saiunit._backend import using_backend, get_default_backend
    with using_backend("dask"):
        assert get_default_backend() == "dask"
```

Also update the existing `test_set_default_backend_rejects_invalid` to match the new error message that mentions dask:

```python
def test_set_default_backend_rejects_invalid():
    with pytest.raises(ValueError, match="must be 'numpy', 'jax', 'cupy', 'torch', 'dask', or None"):
        set_default_backend("notabackend")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_backend_test.py -k "set_default_backend or using_backend_accepts_dask" -v
```

Expected: FAIL — `ValueError: default backend must be 'numpy', 'jax', 'cupy', 'torch', or None; got 'dask'`.

- [ ] **Step 3: Broaden validation**

In `saiunit/_backend.py`, update `set_default_backend`:

```python
def set_default_backend(name: Optional[BackendName]) -> None:
    if name not in ("numpy", "jax", "cupy", "torch", "dask", None):
        raise ValueError(
            f"default backend must be 'numpy', 'jax', 'cupy', 'torch', 'dask', or None; got {name!r}"
        )
    _default_backend.set(name)
```

And `using_backend`:

```python
@contextmanager
def using_backend(name: BackendName) -> Iterator[None]:
    if name not in ("numpy", "jax", "cupy", "torch", "dask"):
        raise ValueError(
            f"backend must be 'numpy', 'jax', 'cupy', 'torch', or 'dask'; got {name!r}"
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
git commit -m "feat(backend): accept dask in using_backend() and set_default_backend()"
```

---

## Task 5: Add `Quantity.to_dask(*, chunks='auto')` method

**Files:**
- Modify: `saiunit/_base_quantity.py`
- Test: `saiunit/_base_quantity_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_base_quantity_test.py`:

```python
def test_quantity_to_dask_basic():
    da = pytest.importorskip("dask.array")
    import saiunit as u
    from saiunit._backend import is_dask_array
    q = u.Quantity(np.array([1.0, 2.0, 3.0]), unit=u.meter)
    q2 = q.to_dask()
    assert is_dask_array(q2.mantissa)
    assert q2.unit == u.meter


def test_quantity_to_dask_noop_when_already_dask():
    da = pytest.importorskip("dask.array")
    import saiunit as u
    arr = da.from_array(np.array([1.0]), chunks=1)
    q = u.Quantity(arr, unit=u.meter)
    q2 = q.to_dask()
    assert q2 is q


def test_quantity_to_dask_custom_chunks():
    da = pytest.importorskip("dask.array")
    import saiunit as u
    q = u.Quantity(np.arange(8, dtype=np.float64), unit=u.meter)
    q2 = q.to_dask(chunks=2)
    assert q2.mantissa.numblocks == (4,)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_base_quantity_test.py -k "to_dask" -v
```

Expected: FAIL with `AttributeError: 'Quantity' object has no attribute 'to_dask'`.

- [ ] **Step 3: Add the method**

In `saiunit/_base_quantity.py`, after the existing `to_torch` method (added in Phase 1), add:

```python
def to_dask(self, *, chunks='auto') -> 'Quantity':
    """Return a new Quantity with mantissa converted to a ``dask.array.Array``.

    No-op (returns ``self``) if the mantissa is already a dask array and no
    ``chunks`` was specified.
    """
    from saiunit._backend import is_dask_array, to_backend
    if is_dask_array(self._mantissa) and chunks == 'auto':
        # Only no-op when default chunks; explicit chunks always rebuilds.
        return self
    return Quantity(to_backend(self._mantissa, "dask", chunks=chunks), unit=self.unit)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_base_quantity_test.py -k "to_dask" -v
```

Expected: PASS (or SKIP if dask not installed).

- [ ] **Step 5: Commit**

```bash
git add saiunit/_base_quantity.py saiunit/_base_quantity_test.py
git commit -m "feat(quantity): add Quantity.to_dask(chunks='auto')"
```

---

## Task 6: Extend `Quantity.backend` property chain to include dask

**Files:**
- Modify: `saiunit/_base_quantity.py`
- Test: `saiunit/_base_quantity_test.py`

- [ ] **Step 1: Write the failing test**

Append to `saiunit/_base_quantity_test.py`:

```python
def test_quantity_backend_dask():
    da = pytest.importorskip("dask.array")
    import saiunit as u
    arr = da.from_array(np.array([1.0]), chunks=1)
    q = u.Quantity(arr, unit=u.meter)
    assert q.backend == "dask"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest saiunit/_base_quantity_test.py::test_quantity_backend_dask -v
```

Expected: FAIL — returns `'jax'` because the dask check isn't in the chain yet.

- [ ] **Step 3: Extend the chain**

In `saiunit/_base_quantity.py`, find the `backend` property (modified in Phase 1) and update its imports and chain:

```python
@property
def backend(self) -> str:
    """The backend of the underlying mantissa: one of
    ``'numpy'``, ``'jax'``, ``'cupy'``, ``'torch'``, ``'dask'``."""
    from saiunit._backend import (
        is_numpy_array, is_cupy_array, is_torch_array, is_dask_array,
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
    return "jax"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest saiunit/_base_quantity_test.py -k "quantity_backend" -v
```

Expected: PASS for all five (numpy, jax, cupy, torch, dask).

- [ ] **Step 5: Commit**

```bash
git add saiunit/_base_quantity.py saiunit/_base_quantity_test.py
git commit -m "feat(quantity): extend Quantity.backend property with dask"
```

---

## Task 7: Broaden `require_jax_backend` to reject dask arrays

**Files:**
- Modify: `saiunit/_jax_guard.py`
- Test: `saiunit/_jax_guard_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `saiunit/_jax_guard_test.py`:

```python
def test_require_jax_raises_for_dask_quantity():
    da = pytest.importorskip("dask.array")
    q = u.Quantity(da.from_array(np.array([1.0]), chunks=1), unit=u.meter)
    with pytest.raises(u.BackendError, match="dask-backed Quantity"):
        require_jax_backend("test_fn", q)


def test_require_jax_rejects_bare_dask_array():
    da = pytest.importorskip("dask.array")
    arr = da.from_array(np.array([1.0]), chunks=1)
    with pytest.raises(u.BackendError, match="dask"):
        require_jax_backend("test_fn", arr)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest saiunit/_jax_guard_test.py -k "dask" -v
```

Expected: FAIL — the bare-dask test doesn't raise because `require_jax_backend` doesn't know about dask.

- [ ] **Step 3: Extend the guard**

In `saiunit/_jax_guard.py`, update the `require_jax_backend` function. The Quantity-branch (which calls `q.backend`) is already correct because Task 6 made the property return `"dask"`. Add a dask check to the bare-array branch:

```python
def require_jax_backend(func_name: str, *quantities_or_arrays) -> None:
    from saiunit._base_quantity import Quantity
    from saiunit._backend import (
        is_numpy_array, is_jax_array, is_cupy_array, is_torch_array, is_dask_array,
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest saiunit/_jax_guard_test.py -v
```

Expected: PASS for all guard tests.

- [ ] **Step 5: Commit**

```bash
git add saiunit/_jax_guard.py saiunit/_jax_guard_test.py
git commit -m "feat(guard): reject dask arrays in require_jax_backend"
```

---

## Task 8: Export `is_dask_array` in `saiunit/__init__.py`

**Files:**
- Modify: `saiunit/__init__.py`
- Test: `saiunit/_backend_test.py`

- [ ] **Step 1: Write the failing test**

Append to `saiunit/_backend_test.py`:

```python
def test_top_level_exports_is_dask_array():
    import saiunit as u
    assert hasattr(u, "is_dask_array")
    assert "is_dask_array" in u.__all__
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest saiunit/_backend_test.py::test_top_level_exports_is_dask_array -v
```

Expected: FAIL — `AttributeError: module 'saiunit' has no attribute 'is_dask_array'`.

- [ ] **Step 3: Add the export**

In `saiunit/__init__.py`, update the `._backend` import block to include `is_dask_array`:

```python
from ._backend import (
    get_default_backend,
    is_cupy_array,
    is_dask_array,
    is_jax_array,
    is_numpy_array,
    is_torch_array,
    set_default_backend,
    using_backend,
)
```

And add `'is_dask_array'` to the `# _backend` section of `__all__`.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest saiunit/_backend_test.py::test_top_level_exports_is_dask_array -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add saiunit/__init__.py saiunit/_backend_test.py
git commit -m "feat(saiunit): export is_dask_array at top level"
```

---

## Task 9: Add `dask` extra to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the extra and update `all`**

In `pyproject.toml`, in `[project.optional-dependencies]`:

Change:
```toml
all = ["saiunit[cupy,torch]"]
```
to:
```toml
dask = ["dask[array]>=2024.1"]
all = ["saiunit[cupy,torch,dask]"]
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
dask = ["dask[array]>=2024.1"]
all = ["saiunit[cupy,torch,dask]"]
```

- [ ] **Step 2: Verify TOML parses**

```bash
python -c "import tomllib; tomllib.loads(open('pyproject.toml').read())"
```

Expected: no output (exit 0).

- [ ] **Step 3: Dry-run install verification**

```bash
pip install --dry-run -e '.[dask]' 2>&1 | head -20
```

Expected: dask and its array deps resolved.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build(deps): add dask optional extra"
```

---

## Task 10: Audit `_base_quantity.py` for materialization triggers

**Files:**
- Read: `saiunit/_base_quantity.py`
- Create: `docs/superpowers/plans/phase2-laziness-audit.md`

This task produces a findings document. Subsequent tasks fix each finding.

- [ ] **Step 1: Grep for known materialization triggers**

Run these searches against `saiunit/_base_quantity.py` and capture each hit:

```bash
grep -n "bool(self" saiunit/_base_quantity.py
grep -n "int(self" saiunit/_base_quantity.py
grep -n "float(self" saiunit/_base_quantity.py
grep -n "len(self" saiunit/_base_quantity.py
grep -n "np.asarray" saiunit/_base_quantity.py
grep -n "np\.array(" saiunit/_base_quantity.py
grep -n "__array__" saiunit/_base_quantity.py
grep -n "__bool__" saiunit/_base_quantity.py
grep -n "__int__" saiunit/_base_quantity.py
grep -n "__float__" saiunit/_base_quantity.py
grep -n "__index__" saiunit/_base_quantity.py
grep -n "__iter__" saiunit/_base_quantity.py
grep -n "__repr__" saiunit/_base_quantity.py
grep -n "tolist" saiunit/_base_quantity.py
grep -n "item()" saiunit/_base_quantity.py
```

- [ ] **Step 2: Classify each hit**

For each location, classify as one of:
- **SAFE**: operates via `xp.*` namespace; dask supports it without compute.
- **CONDITIONALLY SAFE**: works on dask if input is dask-typed, but the implementation uses `np.*` directly — should be re-routed through `xp = get_backend(self)`.
- **MATERIALIZING**: calls `bool()`, `float()`, `tolist()`, etc. on the mantissa. Cannot be made lazy.

Create `docs/superpowers/plans/phase2-laziness-audit.md` with one row per finding:

```markdown
# Phase 2 Laziness Audit — Findings

| Line | Method | Trigger | Classification | Action |
|------|--------|---------|----------------|--------|
| 1234 | __bool__ | bool(self._mantissa) | MATERIALIZING | Guard with BackendError when dask |
| 5678 | __repr__ | format(self._mantissa) | MATERIALIZING | Use _repr_mantissa_lazy_safe |
| ...  | ...    | ...                  | ...            | ... |
```

- [ ] **Step 3: Verify the audit captures `__repr__`**

`Quantity.__repr__` is the most likely offender because it formats the mantissa via NumPy's array printer. Confirm by grepping; if not present in your audit, look harder.

- [ ] **Step 4: Commit the findings document**

```bash
git add docs/superpowers/plans/phase2-laziness-audit.md
git commit -m "docs: add phase 2 laziness-audit findings"
```

- [ ] **Step 5: Use the findings to drive Tasks 11 and 12**

The audit doc is the input to the next two tasks. Each MATERIALIZING finding gets a guard or restructure.

---

## Task 11: Add `_repr_mantissa_lazy_safe` helper and patch `Quantity.__repr__`

**Files:**
- Modify: `saiunit/_base_quantity.py`
- Test: `saiunit/_dask_laziness_test.py` (create)

- [ ] **Step 1: Create the laziness test file**

Create `saiunit/_dask_laziness_test.py`:

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

"""Lazy-safety tests for the dask backend.

These tests construct a dask Quantity from a source array that *counts*
materializations. Any saiunit operation that triggers compute() will bump
the counter; the assertions catch unintended materialization.
"""

import numpy as np
import pytest


class _ComputeCounter:
    """Wraps a numpy array; counts every time it's read."""

    def __init__(self, arr):
        self._arr = arr
        self.reads = 0

    def __getitem__(self, idx):
        self.reads += 1
        return self._arr[idx]

    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def ndim(self):
        return self._arr.ndim


@pytest.fixture
def dask_quantity():
    da = pytest.importorskip("dask.array")
    import saiunit as u
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    counter = _ComputeCounter(arr)
    darr = da.from_array(counter, chunks=2)
    q = u.Quantity(darr, unit=u.meter)
    return q, counter


def test_dask_quantity_shape_does_not_compute(dask_quantity):
    q, counter = dask_quantity
    _ = q.shape
    assert counter.reads == 0, f"q.shape triggered {counter.reads} reads"


def test_dask_quantity_repr_does_not_compute(dask_quantity):
    q, counter = dask_quantity
    s = repr(q)
    assert counter.reads == 0, f"repr(q) triggered {counter.reads} reads"
    assert "dask" in s.lower() or "Quantity" in s


def test_dask_quantity_addition_does_not_compute(dask_quantity):
    q, counter = dask_quantity
    _ = q + q
    assert counter.reads == 0, f"q + q triggered {counter.reads} reads"


def test_dask_quantity_compute_does_materialize(dask_quantity):
    q, counter = dask_quantity
    _ = q.mantissa.compute()
    assert counter.reads > 0, "explicit .compute() did not actually materialize"
```

- [ ] **Step 2: Run the laziness tests to see which fail**

```bash
pytest saiunit/_dask_laziness_test.py -v
```

Expected: at least `test_dask_quantity_repr_does_not_compute` FAILS (the current `__repr__` calls into numpy's array printer, which triggers compute). Possibly others fail too — these are real findings.

- [ ] **Step 3: Add `_repr_mantissa_lazy_safe` and patch `__repr__`**

In `saiunit/_base_quantity.py`, locate the existing `__repr__` method. Above it, add:

```python
def _repr_mantissa_lazy_safe(mantissa) -> str:
    """Return a string representation of a mantissa without triggering
    materialization for lazy backends (dask).

    For lazy mantissas, defers to the backend's own ``__repr__`` (e.g.
    ``dask.array<…, shape=…, dtype=…, chunks=…>``). For eager mantissas,
    uses ``repr(mantissa)`` directly.
    """
    from saiunit._backend import is_dask_array
    if is_dask_array(mantissa):
        return repr(mantissa)  # dask's repr is already lazy-safe
    return repr(mantissa)
```

Now update `Quantity.__repr__` to use this helper for the mantissa portion. Locate the method and replace its mantissa-formatting call with `_repr_mantissa_lazy_safe(self._mantissa)`. The exact original form depends on the current implementation; the substitution principle is:

```python
# Before (illustrative):
# return f"Quantity({self._mantissa}, unit={self.unit!s})"

# After:
return f"Quantity({_repr_mantissa_lazy_safe(self._mantissa)}, unit={self.unit!s})"
```

Read the current `__repr__` body, identify the place the mantissa is interpolated/formatted, and route through `_repr_mantissa_lazy_safe`. If `__repr__` calls a helper method, modify the helper.

- [ ] **Step 4: Run laziness tests again**

```bash
pytest saiunit/_dask_laziness_test.py::test_dask_quantity_repr_does_not_compute -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add saiunit/_base_quantity.py saiunit/_dask_laziness_test.py
git commit -m "feat(quantity): make __repr__ lazy-safe for dask backend"
```

---

## Task 12: Guard or fix remaining audit findings

**Files:**
- Modify: `saiunit/_base_quantity.py`
- Test: `saiunit/_dask_laziness_test.py`

This task is iterative — one finding at a time. The audit document from Task 10 lists each. For each MATERIALIZING finding that is NOT already covered by Task 11:

- [ ] **Step 1: Pick the next finding from the audit doc**

Read `docs/superpowers/plans/phase2-laziness-audit.md` and choose the next un-fixed finding. Examples of expected findings:

- `Quantity.__bool__` / `Quantity.__int__` / `Quantity.__float__` (only sensible on a single-element array; the dask version requires compute).
- `Quantity.tolist()`, `Quantity.item()`.
- Any internal helper that calls `np.asarray(self._mantissa)` to coerce.
- `Quantity.__iter__` if implemented (dask supports iteration but it's expensive).

- [ ] **Step 2: Write a failing test in `_dask_laziness_test.py`**

Add a test that asserts the right behavior. For materializing finders, the right behavior is "raises BackendError telling the user to call .compute()". Example:

```python
def test_dask_quantity_bool_raises_clear_error():
    da = pytest.importorskip("dask.array")
    import saiunit as u
    q = u.Quantity(da.from_array(np.array([True]), chunks=1), unit=u.UNITLESS)
    with pytest.raises(u.BackendError, match="compute"):
        bool(q)
```

- [ ] **Step 3: Add the guard at the finding's location**

In `saiunit/_base_quantity.py`, at the audit-listed line, add a guard. Pattern:

```python
def __bool__(self):
    from saiunit._backend import is_dask_array
    if is_dask_array(self._mantissa):
        raise BackendError(
            "Cannot convert a dask-backed Quantity to bool without "
            "materialization. Call `q.mantissa.compute()` first."
        )
    return bool(self._mantissa)
```

Use this pattern for `__int__`, `__float__`, `__index__`, `tolist`, `item`, and any other materializing method.

- [ ] **Step 4: Run the laziness test for this finding**

```bash
pytest saiunit/_dask_laziness_test.py -k "<the new test>" -v
```

Expected: PASS.

- [ ] **Step 5: Mark the finding fixed in the audit doc and commit**

Update `docs/superpowers/plans/phase2-laziness-audit.md` to mark the row as FIXED. Then:

```bash
git add saiunit/_base_quantity.py saiunit/_dask_laziness_test.py docs/superpowers/plans/phase2-laziness-audit.md
git commit -m "feat(quantity): guard <method> against dask materialization"
```

**Repeat Task 12 once per remaining MATERIALIZING finding.** If the audit found zero further findings beyond `__repr__`, this entire task is a no-op and can be skipped.

---

## Task 13: Extend `conftest.py` `backend` fixture

**Files:**
- Modify: `conftest.py`
- Test: existing parametrized tests

- [ ] **Step 1: Write the failing test**

Update the assertion in `saiunit/_backend_parametrize_test.py::test_backend_fixture_includes_cupy_and_torch` to include dask, and rename it:

```python
def test_backend_fixture_includes_all_phase1_and_2(backend):
    assert backend in {"numpy", "jax", "cupy", "torch", "dask"}
```

- [ ] **Step 2: Verify the fixture currently has 4 params, not 5**

```bash
pytest saiunit/_backend_parametrize_test.py::test_backend_fixture_includes_all_phase1_and_2 --collect-only -q
```

Expected: 4 items. Adding dask brings it to 5.

- [ ] **Step 3: Add dask to the fixture**

In `conftest.py`:

```python
@pytest.fixture(params=["numpy", "jax", "cupy", "torch", "dask"])
def backend(request):
    if request.param == "cupy":
        pytest.importorskip("cupy")
    elif request.param == "torch":
        pytest.importorskip("torch")
    elif request.param == "dask":
        pytest.importorskip("dask.array")
    with u.using_backend(request.param):
        yield request.param
```

- [ ] **Step 4: Run parametrized tests and confirm 5 items**

```bash
pytest saiunit/_backend_parametrize_test.py --collect-only -q
pytest saiunit/_backend_parametrize_test.py -v
```

Expected: 5 collected items per test; dask runs (or skips if dask missing); other backends still pass.

- [ ] **Step 5: Commit**

```bash
git add conftest.py saiunit/_backend_parametrize_test.py
git commit -m "test(backend): include dask in parametrized backend fixture"
```

---

## Task 14: Add dask-aware cross-backend smoke tests

**Files:**
- Modify: `saiunit/_backend_parametrize_test.py`

- [ ] **Step 1: Extend existing smoke tests to handle dask**

Update `test_math_sin_on_each_backend` in `saiunit/_backend_parametrize_test.py` to add a dask branch:

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
```

`test_linalg_norm_on_each_backend` needs to handle dask laziness — dask returns a lazy scalar:

```python
def test_linalg_norm_on_each_backend(backend):
    q = u.Quantity([3.0, 4.0], unit=meter)
    n = u.linalg.norm(q)
    assert n.unit == meter
    if backend == "dask":
        # For dask, .mantissa is a 0-d lazy array; compute to read the value.
        assert float(n.mantissa.compute()) == 5.0
    else:
        assert float(n.mantissa) == 5.0
```

- [ ] **Step 2: Run all smoke tests**

```bash
pytest saiunit/_backend_parametrize_test.py -v
```

Expected: all 5 backends run (or skip cleanly).

- [ ] **Step 3: Commit**

```bash
git add saiunit/_backend_parametrize_test.py
git commit -m "test(backend): make cross-backend smoke tests dask-aware"
```

---

## Task 15: Update the user guide

**Files:**
- Modify: `docs/getting_started/numpy_backend.md`

- [ ] **Step 1: Append the Dask section**

At the end of `docs/getting_started/numpy_backend.md`, add:

````markdown
## Dask: lazy semantics

`saiunit` accepts `dask.array.Array` mantissas via the optional `dask` extra:

```bash
pip install saiunit[dask]
```

```python
import numpy as np
import dask.array as da
import saiunit as u

big = da.from_array(np.arange(1_000_000.0), chunks=100_000)
q = u.Quantity(big, unit=u.meter)
print(q.backend)        # 'dask'
print(q.shape)          # (1000000,)  — no compute
print((q + q).backend)  # 'dask'      — still lazy
```

Convert with `Quantity.to_dask(chunks='auto')`:

```python
q_np = u.Quantity(np.arange(1_000_000.0), unit=u.meter)
q_da = q_np.to_dask(chunks=100_000)
```

### What stays lazy

Building a dask-backed `Quantity`, calling `q.shape` / `q.ndim` / `q.dtype`,
arithmetic, and most `saiunit.math` / `saiunit.linalg` operations all stay
lazy — no `.compute()` until you explicitly trigger it.

`repr(q)` is also lazy-safe; it shows dask's task-graph summary rather than
materializing the array.

### What requires compute

Operations that produce a Python scalar — `bool(q)`, `int(q)`, `float(q)`,
`q.tolist()`, `q.item()` — raise `BackendError` on dask-backed quantities.
Call `q.mantissa.compute()` first:

```python
single = u.Quantity(da.from_array(np.array([42.0]), chunks=1), unit=u.meter)
bool(single)             # raises BackendError
single.mantissa.compute()  # numpy array; now eager
```

### Mixed-backend arithmetic

Mixing a dask-backed and a non-dask-backed `Quantity` in arithmetic falls
through the default-backend tiebreaker (see "Choosing the default backend"
above). If the result lands on dask, the non-dask operand is auto-lifted to
a dask array.
````

- [ ] **Step 2: Verify the file**

```bash
python -c "import pathlib; assert 'Dask: lazy semantics' in pathlib.Path('docs/getting_started/numpy_backend.md').read_text()"
```

Expected: no output (exit 0).

- [ ] **Step 3: Commit**

```bash
git add docs/getting_started/numpy_backend.md
git commit -m "docs: add Dask lazy-semantics section to user guide"
```

---

## Task 16: Install dask in CI workflows

**Files:**
- Modify: `.github/workflows/CI.yml`

- [ ] **Step 1: Add a dask-install step to each platform job**

In `.github/workflows/CI.yml`, find each platform's "Install dependencies"
step (Linux, macOS, Windows). After the `pip install torch --no-cache-dir`
line (added in Phase 1 Task 17), add:

```yaml
          pip install 'dask[array]' --no-cache-dir
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
```

- [ ] **Step 2: Verify the YAML parses**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/CI.yml'))"
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/CI.yml
git commit -m "ci: install dask on all platforms to exercise the dask backend"
```

---

## Task 17: Final integration verification

**Files:** none

- [ ] **Step 1: Run the full saiunit test suite**

```bash
pytest saiunit/ -x -q
```

Expected: all tests pass; dask tests skip if dask not installed.

- [ ] **Step 2: With dask installed, run again**

```bash
pip install 'dask[array]'
pytest saiunit/ -k "dask or laziness or backend" -v
```

Expected: every dask test PASSes.

- [ ] **Step 3: Run the laziness suite explicitly**

```bash
pytest saiunit/_dask_laziness_test.py -v
```

Expected: all tests pass. `test_dask_quantity_compute_does_materialize` is the
positive control (it should report `reads > 0`); all others should report 0.

- [ ] **Step 4: Quick interactive smoke test**

```bash
python -c "
import numpy as np, dask.array as da
import saiunit as u
big = da.from_array(np.arange(1000.0), chunks=100)
q = u.Quantity(big, unit=u.meter)
print('backend:', q.backend)
print('shape:', q.shape)
print('repr:', repr(q)[:80])
r = u.math.sin(q / u.meter)
print('sin backend:', type(r).__name__)
print('summed:', float((q.mantissa.sum()).compute()))
try:
    u.lax.slice(q, (0,), (1,))
except u.BackendError as e:
    print('expected BackendError:', e)
"
```

Expected output (something like):
```
backend: dask
shape: (1000,)
repr: Quantity(dask.array<from-array, shape=(1000,), dtype=float64, chunksize=(100,)...
sin backend: Array
summed: 499500.0
expected BackendError: ...slice requires the jax backend; got dask-backed Quantity...
```

- [ ] **Step 5: Push and PR**

```bash
git push -u origin <your-branch-name>
```

PR title:

> `feat(backend): add Dask backend with lazy semantics (Phase 2 of multi-backend support)`

Reference the spec and the Phase 2 plan in the body. After CI passes and merge, start Phase 3.

---

## Done Criteria

Phase 2 is complete when:
1. All 17 tasks above are checked off.
2. `pytest saiunit/` passes on a machine with no dask installed (dask tests skip).
3. `pytest saiunit/` passes on a machine with dask installed (dask tests run and pass).
4. `pytest saiunit/_dask_laziness_test.py -v` confirms no unintended `.compute()` calls in `q.shape`, `repr(q)`, `q + q`.
5. The audit document `docs/superpowers/plans/phase2-laziness-audit.md` has every MATERIALIZING finding marked FIXED.
