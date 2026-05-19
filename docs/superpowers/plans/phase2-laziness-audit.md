# Phase 2 Laziness Audit — Findings

Audit of `saiunit/_base_quantity.py` for code paths that implicitly materialize
the mantissa. For dask-backed quantities, any such path triggers `.compute()`,
which we want to avoid unless the user asked for it.

Classification:
- **SAFE**: operates via `xp.*` namespace or works lazily on dask.
- **MATERIALIZING**: calls `bool()`, `float()`, `tolist()`, `np.asarray()`, etc.
  Cannot be made lazy. Must be guarded with a clear `BackendError` for dask.

| Line | Method | Trigger | Classification | Action | Status |
|------|--------|---------|----------------|--------|--------|
| ~1012 | `_format_value` | `jnp.asarray(m)` + `np.array_str(value)` | MATERIALIZING | Use lazy-safe path for dask via `_repr_mantissa_lazy_safe` | FIXED (Task 11) |
| ~1354 | `__hash__` | `np.asarray(self.mantissa).tobytes()` | MATERIALIZING | Guard with BackendError for dask | FIXED (Task 12) |
| ~1358 | `__repr__` | calls `_format_value` | MATERIALIZING | Routes through fixed `_format_value` | FIXED (Task 11) |
| ~1373 | `__str__` | calls `repr_in_unit` → `_format_value` | MATERIALIZING | Inherits fix from Task 11 | FIXED (Task 11) |
| ~1732 | `__len__` | `len(self.mantissa)` | SAFE | dask supports `len()` lazily (returns first-dim size from shape) | no-op |
| ~2962 | `tolist` | `self.mantissa.tolist()` | MATERIALIZING | Guard with BackendError for dask | FIXED (Task 12) |
| ~3218 | `__array__` | `np.asarray(self.to_decimal(), dtype=dtype)` | MATERIALIZING | Guard with BackendError for dask | FIXED (Task 12) |
| ~3228 | `__float__` | `float(self.to_decimal())` | MATERIALIZING | Guard with BackendError for dask | FIXED (Task 12) |
| ~3237 | `__int__` | `int(self.to_decimal())` | MATERIALIZING | Guard with BackendError for dask | FIXED (Task 12) |
| ~3246 | `__index__` | `operator.index(self.to_decimal())` | MATERIALIZING | Guard with BackendError for dask | FIXED (Task 12) |

## Not present

- `__bool__` — not implemented on `Quantity`; falls back to default object truthiness, which is harmless. No action.
- `item()` — not implemented as a `Quantity` method. No action.
- `__iter__` — `iter(self.mantissa)` is supported by dask (iterates the outermost axis lazily). SAFE; no action.

## Notes

`__array__` is conventionally invoked when a user explicitly calls
`np.asarray(q)` — that's the user opting in to materialization. We still
raise a clear `BackendError` for dask-backed `Quantity` because dask
prefers `q.mantissa.compute()` to be the explicit gate.
