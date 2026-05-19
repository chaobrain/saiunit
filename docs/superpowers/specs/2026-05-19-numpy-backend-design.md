# NumPy Backend Support for saiunit — Design

**Status:** Draft
**Date:** 2026-05-19
**Author:** Chaoming Wang (with Claude)

## Goal

Add NumPy as a first-class array backend alongside JAX, so saiunit's `Quantity` can wrap either `np.ndarray` or `jax.Array` and dispatch every unit-aware operation to the matching backend. JAX remains a mandatory install dependency; the change is about dual-backend support, not JAX removal.

### Motivations

1. **Faster eager execution.** NumPy dispatch is cheaper than JAX for small arrays and scalar ops.
2. **NumPy-native interop.** Code paths that talk to scipy / pandas / sklearn stay in NumPy end-to-end without converting through JAX.
3. **API consistency.** A user importing `saiunit` and passing in `np.array(...)` should not silently get a JAX result back.

### Non-goals

- Removing or hiding JAX. JAX is still imported eagerly at package init.
- Supporting backends other than NumPy and JAX in v1 (CuPy / Torch are out of scope but should not be designed against).
- Porting `saiunit.lax`, `saiunit.autograd`, or `saiunit.sparse` to NumPy. These remain JAX-only.

## Decisions (from brainstorming)

| Topic | Decision |
|---|---|
| JAX dependency | Mandatory. JAX stays in `install_requires`. |
| Backend selection | Auto-detect from mantissa type; `set_default_backend()` controls preference. |
| Mixed-backend ops | Promote to global default if set; otherwise JAX wins. |
| Dispatch mechanism | `array_api_compat.array_namespace(x)` to obtain a unified `xp` module. |
| Array protocol | `__array_ufunc__` only. No `__array__` (would silently strip units). |
| Scope v1 | `_base_quantity`, `math/`, `linalg/`, `fft/`. |
| JAX-only modules | `lax/`, `autograd/`, `sparse/` raise on NumPy `Quantity`. |
| Conversions | `.to_numpy()`, `.to_jax()`, `.backend` property. |
| Module API | `set_default_backend()`, `get_default_backend()`, `using_backend(name)` context manager. |
| Testing | Parametrize all existing tests over both backends. |

## Architecture

Three layers, from innermost to outermost.

### Layer 1: Backend abstraction (`saiunit/_backend.py`, new)

A thin module that owns all backend routing. Holds the global default backend (process-local, thread-safe via `contextvars.ContextVar`).

**Public API:**

```python
def get_backend(*arrays_or_quantities) -> ModuleType
def set_default_backend(name: Literal["numpy", "jax", None]) -> None
def get_default_backend() -> Literal["numpy", "jax", None]
def using_backend(name: Literal["numpy", "jax"]) -> ContextManager
def is_jax_array(x) -> bool
def is_numpy_array(x) -> bool
def to_backend(x, name: Literal["numpy", "jax"])  # for internal use
```

**Routing rules for `get_backend(*xs)`:**

1. Flatten any `Quantity` inputs to their mantissas.
2. Collect the set of backend types present.
3. If only one backend is present → return that backend's `xp`.
4. If both are present:
   - If `get_default_backend()` is set → return that backend's `xp`.
   - Otherwise → return JAX's `xp` (JAX wins).
5. Returns either `array_api_compat.numpy` or `array_api_compat.jax.numpy`.

**Why a wrapper instead of `array_namespace` inline:** saiunit needs the global-default and mixed-backend promotion rules baked in; consumers should not have to repeat that logic at each call site.

### Layer 2: `Quantity` (modified `_base_quantity.py`)

- **Mantissa storage:** unchanged (`jax.Array | np.ndarray`).
- **`.backend` property:** returns `"numpy"` or `"jax"` based on `type(self.mantissa)`.
- **`.to_numpy()` / `.to_jax()`:** return a new `Quantity` with converted mantissa, unit preserved. No-op when already on the requested backend.
- **`__array_ufunc__` (new):** intercepts `np.<ufunc>(quantity, ...)` calls, validates dimensions, dispatches to the appropriate saiunit math wrapper (`saiunit.math.add`, `saiunit.math.sin`, etc.). Returns `NotImplemented` for ufuncs we don't support, so NumPy falls back.
- **Internal `jnp.*` calls:** all ~25 direct references (shape, ndim, size, T, mT, imag, real, etc.) replaced with `xp.<op>(self.mantissa)` where `xp = get_backend(self)`.
- **JAX integrations kept as-is:**
  - `@register_pytree_node_class` — pytree flatten/unflatten remain. For NumPy-backed `Quantity`, `tree_flatten` still works (leaf is the np.ndarray); `jax.jit` would trace it as an opaque PyTree leaf with no JAX-specific behavior.
  - `_is_tracer` checks in `_base_dimension.py` — inert for NumPy values (always returns False), unchanged for JAX.
  - `jax.ensure_compile_time_eval` in `__init__` — left in place for both backends; it's a no-op outside a JAX trace, so NumPy mantissas pass through unchanged. (Implementation phase should confirm with a focused test.)

### Layer 3: Module wrappers (modified `math/`, `linalg/`, `fft/`)

The decorator system in `_base_decorators.py` (`keep_unit`, `change_unit`, `remove_unit`, `accept_unitless`) becomes backend-aware:

```python
# Before
def _impl(*args, **kwargs):
    mantissas = [a.mantissa if isinstance(a, Quantity) else a for a in args]
    result = jnp.func(*mantissas, **kwargs)
    return Quantity(result, unit=unit_out)

# After
def _impl(*args, **kwargs):
    mantissas = [a.mantissa if isinstance(a, Quantity) else a for a in args]
    xp = get_backend(*mantissas)
    func = getattr(xp, func_name)  # or passed in
    result = func(*mantissas, **kwargs)
    return Quantity(result, unit=unit_out)
```

For functions where `array_api_compat` doesn't expose a direct equivalent (saiunit-specific extensions, JAX-only primitives), the wrapper falls back to a per-function dispatch table.

**JAX-only modules (`lax/`, `autograd/`, `sparse/`):** add an entry guard:

```python
def _require_jax(*qs):
    for q in qs:
        if isinstance(q, Quantity) and q.backend != "jax":
            raise BackendError(
                f"{func_name} requires jax backend; got {q.backend}. "
                f"Call .to_jax() first."
            )
```

## Data flow examples

**Pure NumPy:**
```python
import saiunit as u
import numpy as np

q = u.Quantity(np.array([1.0, 2.0]), unit=u.meter)  # backend="numpy"
r = u.math.sin(q / u.meter)
# r.mantissa is np.ndarray; no JAX trace, no compilation.
```

**Pure JAX:**
```python
import jax.numpy as jnp
q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.meter)  # backend="jax"
r = u.math.sin(q / u.meter)  # r.mantissa is jax.Array.
```

**Mixed, no default set:**
```python
a = u.Quantity(np.array([1.0]), unit=u.meter)
b = u.Quantity(jnp.array([2.0]), unit=u.meter)
c = a + b  # default unset → JAX wins; c.mantissa is jax.Array.
```

**Mixed, default = numpy:**
```python
u.set_default_backend("numpy")
c = a + b  # c.mantissa is np.ndarray.
```

**External NumPy ufunc:**
```python
q = u.Quantity(np.array([1.0]), unit=u.meter)
r = np.sin(q / u.meter)  # routed through __array_ufunc__ → unit-safe.
```

## Error handling

| Situation | Behavior |
|---|---|
| `jnp.add(numpy_quantity, ...)` | `__jax_array__` is not implemented; user must call `.to_jax()` first. Raise `BackendError` with that hint. |
| `lax.scan(numpy_quantity, ...)` | Guard raises `BackendError("lax requires jax backend; call .to_jax()")`. |
| `np.add(meters_q, seconds_q)` | `__array_ufunc__` runs dimension check, raises `DimensionMismatchError` as today. |
| `set_default_backend("torch")` | `ValueError("backend must be 'numpy', 'jax', or None")`. |
| `array_api_compat` missing | Raised at first import of `_backend.py` with install hint. |

New exception class: `saiunit.BackendError` (subclass of `TypeError` for compatibility with code that catches `TypeError`).

## Testing strategy

**Parametrize all existing tests** via a `conftest.py` fixture:

```python
@pytest.fixture(params=["numpy", "jax"])
def backend(request):
    with saiunit.using_backend(request.param):
        yield request.param
```

Tests that exercise array creation pick up the backend from the fixture:

```python
def test_addition(backend):
    a = u.Quantity([1, 2, 3], unit=u.meter)  # uses default → request.param
    b = u.Quantity([4, 5, 6], unit=u.meter)
    assert (a + b).backend == backend
```

For functions in `lax/`, `autograd/`, `sparse/`, fixture stays JAX-only (no parametrization).

**New test files:**
- `saiunit/_backend_test.py` — backend module behavior (dispatch, defaults, context manager, promotion).
- `saiunit/_base_quantity_backend_test.py` — `__array_ufunc__`, `.to_numpy()`, `.to_jax()`, `.backend`.
- `saiunit/math/_fun_keep_unit_numpy_test.py` (and equivalents for change/remove/accept) — numpy-specific edge cases not covered by parametrized tests.

**CI:** existing matrix unchanged. Test runtime roughly doubles; if that becomes painful, parametrize a representative subset and run the full matrix only on `CI-daily.yml`.

## Dependencies

- **New:** `array_api_compat >= 1.9` (supports both NumPy and JAX namespaces).
- **Unchanged:** `jax`, `numpy`, `typing_extensions`.

`array_api_compat` is small (~50 KB), pure Python, well-maintained by the Quansight team behind the array API standard.

## Migration & rollout

This is an additive change. Existing user code that does `Quantity(jnp.array(...))` continues to work identically — same backend (JAX), same return types.

**Potential observable behavior change:** today, passing `np.array(...)` to `Quantity(...)` may result in implicit promotion to a `jax.Array` on the first arithmetic op (because internal calls go through `jnp.*`). After this change, the mantissa stays `np.ndarray` and operations stay in NumPy. The implementation phase must enumerate which user-visible operations changed return type and document them in the changelog. `saiunit.set_default_backend("jax")` is the one-line opt-in for users who want the old behavior back.

**Suggested phased rollout** (refined in the implementation plan):

1. Phase 1: Layer 1 backend module + Quantity changes (mantissa-type detection, `.backend`, `.to_numpy()`, `.to_jax()`, `__array_ufunc__`). No math wrappers updated yet — math still routes through JAX.
2. Phase 2: Update `_base_decorators.py` so `keep_unit` / `change_unit` / `remove_unit` / `accept_unitless` dispatch via `xp`. Math, linalg, fft pick this up automatically.
3. Phase 3: Add JAX-only guards in `lax/`, `autograd/`, `sparse/`.
4. Phase 4: Parametrize tests; fix divergences uncovered by NumPy-backend runs.
5. Phase 5: Documentation + changelog.

## Open questions for implementation phase

- Does `array_api_compat.jax.numpy` cover every `jnp.*` call site saiunit currently makes? A spike at the start of Phase 2 will enumerate gaps and decide per-function fallbacks.
- Should `using_backend(None)` reset to "no default" or be invalid? (Currently planned: valid, resets to None.)
- For `__array_ufunc__`, which ufuncs do we route and which do we `NotImplemented`? Initial list: arithmetic, comparison, trig, exp/log, abs, sqrt. To be enumerated in the implementation plan.

## Out of scope (explicitly)

- Removing JAX as a dependency.
- CuPy / PyTorch backends.
- Numpy support for `lax/`, `autograd/`, `sparse/`.
- Performance benchmarks (a follow-up project — add a `benchmarks/` once dual-backend works).
- Changing the public API of math functions. Only their internal dispatch changes.
