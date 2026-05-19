# Multi-Backend Support: CuPy, PyTorch, Dask, ndonnx

**Date:** 2026-05-19
**Status:** Draft, pending user review
**Scope:** Extend saiunit's backend system from `{numpy, jax}` to `{numpy, jax, cupy, torch, dask, ndonnx}`.

## 1. Overview and Scope

### 1.1 Goal

Extend saiunit's existing closed-enum backend system (currently NumPy + JAX, built on `array_api_compat`) to four additional array libraries — **CuPy, PyTorch, Dask, ndonnx** — so users can wrap unit-aware `Quantity` objects around tensors from any of them and have `saiunit.math`, `saiunit.linalg`, and `saiunit.fft` dispatch correctly.

### 1.2 In scope

- Detection helpers (`is_cupy_array`, `is_torch_array`, `is_dask_array`, `is_ndonnx_array`).
- Backend dispatch via the existing `get_backend(*arrays)` → `xp` namespace pattern.
- `Quantity` conversion methods with backend-specific kwargs:
  `to_cupy(*, device=None)`, `to_torch(*, device=None, dtype=None)`,
  `to_dask(*, chunks='auto')`, `to_ndonnx()`.
- Extension of `BackendName` literal, `using_backend(...)`, `set_default_backend(...)`,
  `to_backend(x, name, **kwargs)`.
- Lazy imports + optional install extras:
  `saiunit[cupy]`, `saiunit[torch]`, `saiunit[dask]`, `saiunit[ndonnx]`, `saiunit[all]`.
- CI-aware test parametrization across all available backends; missing libraries are
  skipped (`pytest.importorskip`), not failed.
- Updated `require_jax_backend` error message that names the offending backend.

### 1.3 Out of scope (non-goals)

- **No cross-framework autograd.** `saiunit.autograd` stays JAX-only. We do not route
  to `torch.autograd.grad` or similar.
- **No PyData `sparse` library backend.** Deferred — overlaps with the existing
  `saiunit.sparse` (JAX-sparse) module and would require a namespace decision.
- **No plugin/registry system.** Backend list is a closed `Literal` enum; third-party
  backends are not extensible at runtime.
- **No auto-coercion across backends in arithmetic.** Mixing a torch-backed `Quantity`
  with a dask-backed `Quantity` in `q1 + q2` is not silently bridged.
- **No device/dtype promotion rules** beyond what each backend natively does. We pass
  through user kwargs; we do not invent a saiunit-level promotion lattice.
- **No `__array_function__` / `__array_namespace__` protocol implementation on
  `Quantity`** itself. We rely on `array_api_compat` for the common interop cases.

### 1.4 Success criteria

1. `q = u.Quantity(torch_tensor, unit=u.meter)` round-trips:
   `q.backend == 'torch'`, `q.to_jax().backend == 'jax'`,
   `q.to_torch(device='cuda').mantissa.device.type == 'cuda'`.
2. `saiunit.math.sin(q)` dispatches via `xp = get_backend(q)` to the correct namespace
   for each of the four new backends (verified by a smoke test per backend).
3. `saiunit.autograd.grad(f)(q)` with a non-JAX-backed `q` raises `BackendError` whose
   message names the actual backend.
4. `pytest saiunit/` passes on a machine with only `jax` + `numpy` installed
   (other backends `importorskip`-ped, not failed).
5. Each phase (CuPy+PyTorch → Dask → ndonnx) merges independently without
   regressing the prior phase.

---

## 2. Public API

### 2.1 `BackendName` extension

```python
BackendName = Literal["numpy", "jax", "cupy", "torch", "dask", "ndonnx"]
```

Order is documentation-only; runtime uses set membership.

### 2.2 New detectors in `saiunit/_backend.py`

```python
def is_cupy_array(x) -> bool: ...
def is_torch_array(x) -> bool: ...
def is_dask_array(x) -> bool: ...
def is_ndonnx_array(x) -> bool: ...
```

All four return `False` when the underlying library is not installed; they
never raise `ImportError`. Implementation prefers
`array_api_compat.is_<backend>_array` where it exists; falls back to a guarded
`isinstance` check otherwise.

All four are added to `saiunit/__init__.py`'s `__all__` alongside the existing
`is_numpy_array` / `is_jax_array`.

### 2.3 New `Quantity` conversion methods

In `saiunit/_base_quantity.py`, alongside existing `to_numpy()` / `to_jax()`:

```python
def to_cupy(self, *, device=None) -> 'Quantity': ...
def to_torch(self, *, device=None, dtype=None) -> 'Quantity': ...
def to_dask(self, *, chunks='auto') -> 'Quantity': ...
def to_ndonnx(self) -> 'Quantity': ...
```

Each method:
- Returns `self` if the mantissa is already on the target backend **and** the
  options match (device, dtype, chunks). For simplicity, "match" means: if no
  kwargs were passed, identity-return when same backend; if kwargs were passed,
  always rebuild via `to_backend`.
- Otherwise calls `to_backend(self._mantissa, "<name>", **kwargs)` and returns a
  new `Quantity` with the same `unit`.
- Conversion logic lives in `to_backend`; the methods are thin wrappers so the
  per-backend rules live in one file.

### 2.4 `to_backend(x, name, **kwargs)` extension

Signature gains a `**kwargs` tail. Per-backend rules:

| `name`     | Implementation                                       | Recognised kwargs   |
|------------|------------------------------------------------------|---------------------|
| `numpy`    | `np.asarray(x)`                                      | (none)              |
| `jax`      | `jnp.asarray(x)`                                     | (none)              |
| `cupy`     | `cupy.asarray(x)` inside `cupy.cuda.Device(device)`  | `device`            |
| `torch`    | `torch.as_tensor(x, device=device, dtype=dtype)`     | `device`, `dtype`   |
| `dask`     | `dask.array.from_array(x, chunks=chunks)`            | `chunks`            |
| `ndonnx`   | `ndonnx.asarray(x)`                                  | (none)              |

Passing kwargs to a backend that doesn't accept any (e.g. `to_backend(x, "numpy", device='cuda')`)
raises `TypeError`. This catches typos.

### 2.5 `Quantity.backend` property

Extends to return any of the six names. Implementation is a chain of
`is_<backend>_array` checks in the order
`numpy → cupy → torch → dask → ndonnx → jax` (jax last as the tie-breaker, matching
the existing fallthrough). Returns the first match.

### 2.6 `using_backend` and `set_default_backend`

Validation broadens to accept all six names. No semantic change otherwise — the
default-backend tiebreaker only fires when `get_backend(*arrays)` cannot decide
from the inputs.

### 2.7 No new top-level surface beyond detectors

The new detectors (`is_cupy_array`, `is_torch_array`, `is_dask_array`,
`is_ndonnx_array`) are added to `saiunit/__init__.py`'s `__all__`. Everything
else is reachable through existing entry points (`saiunit.to_backend`,
`q.to_torch(...)`, etc.). No new top-level functions are introduced.

---

## 3. Internal Architecture

### 3.1 `get_backend(*arrays_or_quantities)` extension

Stays a single function in `_backend.py`. After flattening `Quantity` inputs to
mantissas, dispatch proceeds in this order:

1. Build a set `kinds` of detected backends across all mantissas, using each
   `is_<backend>_array`.
2. If `kinds` has exactly one element, return that backend's xp namespace.
3. If `kinds` is empty (only Python scalars / unknown) **or** has more than one
   element (mixed), consult `get_default_backend()`. If unset, fall back to JAX
   (preserves current behavior).

Mixed-backend inputs are **not** rejected at this layer — they fall through to
the default tiebreaker. The underlying array op will fail in the chosen `xp`
with whatever error that library produces. Documented as user responsibility.

### 3.2 xp namespace sources

| Backend  | xp source                                |
|----------|------------------------------------------|
| numpy    | `array_api_compat.numpy` (existing)      |
| jax      | `jax.numpy` (existing)                   |
| cupy     | `array_api_compat.cupy`                  |
| torch    | `array_api_compat.torch`                 |
| dask     | `array_api_compat.dask.array`            |
| ndonnx   | `ndonnx` (already array-API-compatible)  |

A module-level cache (`_xp_cache: dict[BackendName, ModuleType]`) ensures we
import each xp source at most once per process. Detection helpers
(`is_cupy_array`, etc.) also memoize a `bool | None` for "library installed?"
so failed imports aren't retried on every call.

### 3.3 Lazy imports

No top-level imports of cupy / torch / dask / ndonnx anywhere in saiunit.
Each detector uses a tiny helper:

```python
def _try_import(name: str) -> ModuleType | None:
    # cached; returns None if ImportError, never raises
```

`to_backend` raises `BackendError` (not `ImportError`) when a requested backend
is missing, with an install hint: `"cupy not installed; pip install saiunit[cupy]"`.

### 3.4 `BackendError` extension

Existing `BackendError` is reused; no new exception types. Messages from
`require_jax_backend` and missing-library paths are the only call sites that
need updating.

---

## 4. Per-Backend Specifics

### 4.1 CuPy

- **xp:** `array_api_compat.cupy`.
- **Detection:** `array_api_compat.is_cupy_array(x)`.
- **Conversion:** `cupy.asarray(x)`. When `device` is given, wrap in
  `with cupy.cuda.Device(device):` for the duration of the call.
- **Requires:** a CUDA-capable machine; install is `pip install saiunit[cupy]`
  which pulls `cupy-cuda12x` (most common). Other variants documented in the
  user guide; we don't try to auto-detect CUDA version.
- **No special quantity behavior** — CuPy arrays are eager, dense, and
  array-API-compatible. Effectively a GPU-backed NumPy from the dispatch
  layer's point of view.

### 4.2 PyTorch

- **xp:** `array_api_compat.torch`.
- **Detection:** `array_api_compat.is_torch_array(x)`.
- **Conversion:** `torch.as_tensor(x, device=device, dtype=dtype)`.
  `dtype` accepts either a torch dtype (`torch.float32`) or a numpy dtype
  (`np.float32`); we translate numpy → torch via a small mapping table when
  needed.
- **Gradients are out of scope** — wrapping a `requires_grad=True` tensor in a
  `Quantity` works (the autograd graph is preserved through `xp` ops), but
  `saiunit.autograd.grad` still rejects torch-backed inputs. Users who want
  gradients call `torch.autograd.grad` themselves on the mantissa.
- **Mutability:** torch tensors are mutable; saiunit's `Quantity` does not
  rely on immutability, so this is a non-issue for arithmetic. Users who
  in-place mutate the mantissa accept the consequences.

### 4.3 Dask

- **xp:** `array_api_compat.dask.array`.
- **Detection:** `array_api_compat.is_dask_array(x)`.
- **Conversion:** `dask.array.from_array(x, chunks=chunks)`.
- **Laziness audit (Phase 2 work):** the following `Quantity` paths must NOT
  trigger eager evaluation when the mantissa is a dask array. The audit
  produces a concrete inventory and either guards or restructures each:
  - `Quantity.shape`, `Quantity.ndim`, `Quantity.dtype` — these query via
    `xp` (dask exposes them without compute), so they are expected-safe; the
    audit verifies that no code path bypasses `xp` and calls `np.shape(...)`
    or similar directly.
  - `Quantity.__repr__` — currently formats the mantissa. For dask, fall
    back to showing the dask repr (`dask.array<…, shape=…, dtype=…, chunks=…>`)
    without triggering `.compute()`. New helper:
    `_repr_mantissa_lazy_safe(mantissa)`.
  - Any `bool(mantissa)`, `int(mantissa)`, `float(mantissa)`, `len(mantissa)`,
    or `np.asarray(mantissa)` call outside an explicit conversion is treated
    as a finding; each finding is either guarded with a clear `BackendError`
    ("operation requires materialization; call `.compute()` first") or
    restructured.
- **Mixed dask + non-dask in one Quantity arithmetic op** falls through the
  default tiebreaker; if dask wins, the dask `xp` will lift the other operand
  via `dask.array.from_array` automatically.

### 4.4 ndonnx

- **xp:** `ndonnx` (the library is itself array-API-compatible; no
  `array_api_compat` wrapper needed).
- **Detection:** `isinstance(x, ndonnx.Array)` inside a guarded import.
- **Conversion:** `ndonnx.asarray(x)`.
- **Symbolic semantics:** ndonnx arrays represent ONNX graph nodes, not
  concrete data. Many ops may be unsupported or behave differently. saiunit's
  promise: dispatch routes correctly; if ndonnx doesn't implement an op, the
  ndonnx error surfaces unwrapped.
- **No tracing helpers in this spec.** Users compose ndonnx-backed `Quantity`
  computations and export via ndonnx's own export path. Documenting the
  workflow is part of Phase 3 docs but no new saiunit API.

---

## 5. JAX-Only Modules: Guard Policy

`saiunit.lax`, `saiunit.autograd`, and the existing `saiunit.sparse`
(JAX-sparse wrappers) remain strictly JAX-only.

### 5.1 Updated `require_jax_backend`

Current implementation rejects only numpy-backed `Quantity`. Generalize:

```python
def require_jax_backend(func_name: str, *quantities_or_arrays) -> None:
    for q in quantities_or_arrays:
        if isinstance(q, Quantity):
            backend = q.backend
            if backend != "jax":
                raise BackendError(
                    f"{func_name} requires the jax backend; got "
                    f"{backend}-backed Quantity. Call .to_jax() on the input first."
                )
        # plain non-Quantity arrays: tolerate only numpy (jax auto-lifts);
        # reject cupy/torch/dask/ndonnx tensors with a similar message.
```

Plain (non-`Quantity`) arrays of types JAX cannot lift — cupy tensors, torch
tensors, dask arrays, ndonnx arrays — are also rejected. Numpy arrays are
still tolerated (JAX lifts them).

### 5.2 No silent conversion

We do not call `.to_jax()` automatically. Errors are loud and actionable. The
user paid for a GPU tensor; we don't silently move it to host.

---

## 6. Dependencies and Packaging

### 6.1 `pyproject.toml` extras

```toml
[project.optional-dependencies]
cupy   = ["cupy-cuda12x>=13.0"]
torch  = ["torch>=2.0"]
dask   = ["dask[array]>=2024.1"]
ndonnx = ["ndonnx>=0.9"]
all    = ["saiunit[cupy,torch,dask,ndonnx]"]
```

Exact version pins to be finalized at implementation time against the
`array_api_compat` minimum that supports each library cleanly.

### 6.2 Core install unchanged

`pip install saiunit` still pulls only `jax`, `numpy`, `typing_extensions`,
`array_api_compat`. None of the four new backends are required.

### 6.3 `array_api_compat` minimum bump

Bump to whatever version supports the cupy + torch + dask wrappers cleanly
(likely `>=1.6`). Verified during Phase 1.

---

## 7. Testing Strategy

### 7.1 Backend fixture

Extend the existing `_backend_parametrize_test.py` (or its underlying
fixture, depending on layout discovered during Phase 1) to add the four
new backends. Each parametrize entry uses `pytest.importorskip`:

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
    return request.param
```

### 7.2 Test additions per phase

Each phase adds, at minimum:

- **`_backend_test.py` extensions:** detection round-trips, `is_<backend>_array`
  returns False for arrays from other backends, `to_backend` with each kwarg
  combination, `Quantity.backend` property correctness.
- **`_jax_guard_test.py` extensions:** `require_jax_backend` raises
  `BackendError` for `Quantity` wrapped over each new backend, with a message
  that names the actual backend.
- **Cross-backend smoke tests** (analogous to the existing ones for numpy/jax):
  basic arithmetic (`q1 + q2`, `q * 2.0`, `q.to(<unit>)`), a representative
  `saiunit.math.sin(q)` call, `saiunit.linalg.norm(q)` for non-symbolic
  backends.

### 7.3 CI matrix

- **Linux Python 3.13:** install `saiunit[torch,dask,ndonnx]`. Cupy skipped
  unless a CUDA runner is provisioned (out of scope here).
- **macOS, Windows:** install `saiunit[torch,dask,ndonnx]`. Cupy skipped.
- **Default daily CI:** `saiunit[all]` on Linux + GPU runner if available.

Tests must pass with only `jax` + `numpy` installed too — the
`pytest.importorskip` guards make this a non-failure.

---

## 8. Rollout Phases

Each phase is an independently mergeable PR. Phase N must keep prior
phases green.

### 8.1 Phase 1 — CuPy + PyTorch (dense, eager)

- Extend `BackendName` literal to include `cupy`, `torch`.
- Add `is_cupy_array`, `is_torch_array` to `_backend.py` and `__all__`.
- Extend `get_backend` dispatch chain.
- Extend `to_backend` with cupy and torch branches (with kwargs).
- Add `Quantity.to_cupy(device=None)` and `Quantity.to_torch(device=None, dtype=None)`.
- Extend `Quantity.backend` chain.
- Extend `require_jax_backend` to reject these two with named-backend message.
- Extras in `pyproject.toml`: `cupy`, `torch`.
- Tests: backend fixture parametrized over `{numpy, jax, cupy, torch}`;
  detection, conversion, dispatch, and JAX-guard rejection tests.
- Docs: a short subsection in the existing numpy-backend user guide.

### 8.2 Phase 2 — Dask (lazy)

- Same skeleton additions (`is_dask_array`, `Quantity.to_dask(chunks='auto')`,
  enum extension, `get_backend` chain, `to_backend` branch, `require_jax_backend`).
- **Audit `_base_quantity.py`** for paths that materialize lazy arrays:
  produce a list of every call site that triggers eager evaluation
  (`bool(...)`, `int(...)`, `float(...)`, `np.asarray(...)`, list iteration)
  on the mantissa. Either:
    1. Guard with `if is_dask_array(...)` and raise a clear error, or
    2. Restructure to avoid materialization.
- Add `_repr_mantissa_lazy_safe(mantissa)` for `Quantity.__repr__`.
- Dask-specific tests: a smoke test asserting that constructing a
  dask-backed `Quantity` and calling `.shape`, `repr(q)`, `q + q` does
  NOT call `.compute()` (assertion via dask's task graph or a counter
  on a synthetic delayed source).
- Extras: `dask`.
- Docs: extend the multi-backend user guide with a "Dask: lazy semantics"
  subsection noting which operations are guarded against materialization
  and how `.compute()` integrates with the unit system.

### 8.3 Phase 3 — ndonnx (symbolic)

- Same skeleton additions for ndonnx.
- No special audit beyond Phase 2's (ndonnx, like dask, is non-eager;
  the lazy-safe `__repr__` path applies).
- Smoke test: build a tiny ndonnx-backed Quantity computation
  (`q = u.Quantity(ndonnx.asarray([1., 2., 3.]), unit=u.meter); r = u.math.sin(q)`)
  and confirm `r.mantissa` is still an `ndonnx.Array`.
- Document known-unsupported ops via a short table in the multi-backend
  user guide (best-effort, links to ndonnx's own docs as the source of truth).
- Extras: `ndonnx`. CI installs it on Linux.

---

## 9. Open Risks

1. **Dask laziness leaks (Phase 2).** The current `_base_quantity.py` is
   not lazy-aware. The audit may surface non-trivial restructuring.
   *Mitigation:* time-box the audit in Phase 2; if it grows beyond a small
   PR, split into a "lazy-safe Quantity refactor" sub-phase.
2. **PyTorch dtype translation.** `torch.float32` ≠ `numpy.float32`. The
   small mapping table is straightforward but adds maintenance.
   *Mitigation:* table lives in one place (`to_backend`'s torch branch);
   tested directly.
3. **CuPy in CI.** GitHub Actions has no CUDA. We cannot fully test
   the cupy backend in stock CI.
   *Mitigation:* mock-based unit tests for the dispatch logic; integration
   tests run on a self-hosted GPU runner if/when one exists. Acceptance of
   reduced coverage documented in the user guide.
4. **ndonnx maturity.** ndonnx is alpha; ops may not exist or behave
   differently. Our promise is that dispatch routes correctly, not that
   every saiunit.math function works.
   *Mitigation:* document the contract clearly; ndonnx errors surface
   unwrapped so users see exactly what ndonnx itself reports.
5. **Mixed-backend arithmetic.** `q_torch + q_dask` falls through the
   default tiebreaker and the underlying xp will fail.
   *Mitigation:* documented as user responsibility; clear `BackendError`
   when we can detect the mix upfront (out of scope for this spec —
   may be a follow-up).
6. **`array_api_compat` version churn.** The wrapper library is still
   maturing; minor releases may shift behavior.
   *Mitigation:* pin a minimum version in `pyproject.toml`; CI catches
   regressions on the daily matrix.
