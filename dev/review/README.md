# Code review: `Quantity`, `Unit`, `Dimension` (2026-07-23)

Inspection of `saiunit/_base_quantity.py`, `saiunit/_base_unit.py`,
`saiunit/_base_dimension.py` (plus the load-bearing helpers in
`saiunit/_base_getters.py`) for bugs, inconsistencies, and edge cases.
Every issue below was **reproduced against saiunit 0.5.2** (Python 3.13,
JAX CPU) unless marked *by design / informational*.

Details and proposed fixes live in the per-module files:

- [`quantity.md`](./quantity.md) — issues in `Quantity`
- [`unit.md`](./unit.md) — issues in `Unit` and the parser/registry
- [`dimension.md`](./dimension.md) — issues in `Dimension` (mostly clean)

## Summary

| # | Severity | Issue | File |
|---|----------|-------|------|
| Q1 | **High** | `q -= unit` silently subtracts `1*unit` — `__isub__` is missing the bare-`Unit` rejection that `+`, `-`, `+=` all have | quantity.md |
| Q2 | **High** | ~20 common NumPy ufuncs (`np.maximum`, `np.minimum`, `np.floor`, `np.hypot`, `np.tanh`, …) raise `TypeError` on Quantities even though `saiunit.math` implements all of them — `_UFUNC_DISPATCH` table is incomplete | quantity.md |
| Q3 | **High** | Arithmetic dunders never return `NotImplemented`, breaking Python's reflected-operator protocol for third-party types; `q * custom_obj` calls `custom_obj.__rmul__(mantissa)` (wrong receiver) and then crashes with a misleading message | quantity.md |
| Q4 | Medium | Zero-convention (`0` compatible with any dimension) applies to `+`, `-`, comparisons — but not to `q[i] = 0`, `q.at[i].set(0)`, `q.at[i].add(0)`, `q.clip(min=0)`, `q.fill(0)` | quantity.md |
| Q5 | Medium | `==` / `!=` raise `UnitMismatchError` on dimension mismatch instead of returning `False` — breaks `q in list`, `list.index`, generic container code | quantity.md |
| Q6 | Medium | Three different exception types for the same mistake (assigning a plain number into a unit-bearing quantity): `TypeError` (`__setitem__`), `UnitMismatchError` (`at[].set`), `DimensionMismatchError` (`fill`) | quantity.md |
| Q7 | Medium | `**` rejects dimensionless-but-scaled exponents (e.g. a percent-like `Quantity(200, Unit(scale=-2))` ≡ 2.0) that every other operation converts fine | quantity.md |
| Q8 | Low | `divmod(a, b)` with mixed dimensions computes floordiv, then raises from mod — partial evaluation, inconsistent policy | quantity.md |
| Q9 | Low | Eager-vs-JIT divergence of the concrete-zero convention: `(x - x) + 3*mV` works eagerly, raises under `jax.jit` | quantity.md |
| Q10 | Low | `Quantity.dtype` returns the Python type `bool` (not a dtype) for `bool` mantissas | quantity.md |
| Q11 | Low | `resize()` mutates in place without the tracer guard other mutators have | quantity.md |
| Q12 | Low | `take(mode='clip')` on an empty NumPy-backed array raises `IndexError` | quantity.md |
| U1 | Medium | `parse_unit` cannot parse parenthesised powers such as `"(m / s)^2"` | unit.md |
| U2 | Low | `Quantity(1, mV) == mV` is `True` while `q + mV` rejects bare units — comparison and arithmetic disagree about bare-`Unit` operands | unit.md |
| U3 | Low | `_normalise_display_parts` merges parts by *dispname*, so two distinct units sharing a display symbol merge in rendered output | unit.md |
| U4 | Info | Unbounded growth of module registries (`_unit_registration_index`, dimension cache under fractional powers) — bounded in practice, documented here | unit.md / dimension.md |
| G1 | Low | `is_unit_equal_math` docstring contradicts `Unit.__eq__` (claims `__eq__` compares names; it does not — the two functions are the same comparison) | unit.md |
| G2 | Low | `maybe_decimal(val, unit=...)` silently ignores `unit` when `val` is dimensionless | quantity.md |
| D1 | Info | `Dimension.__getstate__` / `__setstate__` are dead code (`__reduce__` takes precedence) | dimension.md |

## What was checked and found solid

- `Dimension` singleton cache: thread-safe (double-checked lock), hash/eq
  contract holds across int/float/`-0.0` exponent dtypes, pickle and
  deepcopy preserve identity, non-finite exponents rejected.
- `Unit` equality/hash contract (component-wise, name-independent),
  factor-unit conversion (`1*cal + 1*J` → correct), display round-trips
  `parse_unit(str(u)) == u` for compound, anonymous, and fractional-power
  units, registry ambiguity handling (hertz/becquerel), user aliases not
  hijacking built-in names.
- `Quantity`: list-constructor unit reconciliation (incl. nested and
  mixed-scale), `in_unit`/`to_decimal` magnitude math, `prod`/`nanprod`
  unit-exponent handling (incl. `where` masks, empty arrays),
  floordiv/mod scale alignment, pickle/deepcopy, pytree round-trip,
  0-d iteration/len guards, `__bool__`, `__format__` fallbacks,
  dask/ndonnx materialization guards.
