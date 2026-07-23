# `Quantity` issues (`saiunit/_base_quantity.py`)

All reproductions run against saiunit 0.5.2. Line numbers refer to the
current `main` checkout.

---

## Q1 (High) — `q -= unit` silently subtracts `1*unit`

`__add__`, `__radd__`, `__iadd__`, `__sub__`, `__rsub__` all call
`_reject_bare_unit`, but `__isub__` (`_base_quantity.py:2208`) does not:

```python
q = Quantity(3.0, unit=u.mV)
q - u.mV    # TypeError: Cannot subtract a Quantity with a bare Unit ...
q += u.mV   # TypeError: Cannot add a Quantity with a bare Unit ...
q -= u.mV   # silently returns Quantity(2., "mV")  ← BUG
```

`_to_quantity(u.mV)` promotes the bare unit to `Quantity(1, mV)`, so the
operation quietly subtracts 1 mV.

**Fix** — one line, mirroring `__iadd__`:

```python
def __isub__(self, oc):
    # a -= b
    self._reject_bare_unit(oc, "subtract")
    return self._binary_operation(oc, operator.sub, fail_for_mismatch=True,
                                  operator_str="-=", inplace=True)
```

---

## Q2 (High) — `_UFUNC_DISPATCH` misses ~20 common ufuncs

`__array_ufunc__` returns `NotImplemented` for any ufunc not in the
table built by `_build_ufunc_dispatch` (`_base_quantity.py:431`), so
NumPy raises `TypeError`. Confirmed unsupported despite `saiunit.math`
having a unit-aware implementation for every one of them:

```
maximum, minimum, fmax, fmin, floor, ceil, trunc, rint, sign, hypot,
expm1, log1p, sinh, cosh, tanh, arcsinh, arccosh, arctanh, fabs, copysign
```

```python
q = jnp.array([1., 2.]) * u.mV
np.maximum(q, q)   # TypeError: ... returned NotImplemented from __array_ufunc__
u.math.maximum(q, q)  # works fine
```

`np.maximum` on two same-unit quantities is entirely legitimate and is
what NumPy's own `np.clip`, `np.fmax`-based code paths, and much user
code produce.

**Fix** — extend the table in `_build_ufunc_dispatch`:

```python
# elementwise extrema / rounding / sign  (keep unit)
np.maximum: _u_math.maximum,   np.minimum: _u_math.minimum,
np.fmax: _u_math.fmax,         np.fmin: _u_math.fmin,
np.floor: _u_math.floor,       np.ceil: _u_math.ceil,
np.trunc: _u_math.trunc,       np.rint: _u_math.rint,
np.fabs: _u_math.fabs,         np.copysign: _u_math.copysign,
np.remainder: _u_math.remainder,
np.sign: _u_math.sign,         np.hypot: _u_math.hypot,
# accept-unitless transcendentals
np.expm1: _u_math.expm1,       np.log1p: _u_math.log1p,
np.sinh: _u_math.sinh,  np.cosh: _u_math.cosh,  np.tanh: _u_math.tanh,
np.arcsinh: _u_math.arcsinh, np.arccosh: _u_math.arccosh,
np.arctanh: _u_math.arctanh,
```

Also add the binary ones (`maximum`, `minimum`, `fmax`, `fmin`,
`copysign`, `hypot`) to `_BINARY_UFUNC_OPNAMES` **only if** a matching
dunder exists — otherwise leave them dispatching to `saiunit.math`
directly (they take the `saiunit_fn(*inputs)` path, which already
handles the Quantity+scalar mix).

Verify with a table-driven test that loops over the dispatch dict and
asserts `np.<uf>(q, ...)` equals `u.math.<uf>(q, ...)`.

---

## Q3 (High) — arithmetic dunders never return `NotImplemented`

`_binary_operation` unconditionally does `_to_quantity(other)`, which
wraps *any* object as an opaque pytree leaf. Two consequences:

1. Python's reflected-operator protocol is broken: for a third-party
   type implementing `__rmul__(self, quantity)`, `Quantity.__mul__`
   should return `NotImplemented` so Python calls it. It never does.
2. Worse, the operand's reflected op gets invoked **against the raw
   mantissa** (because `mantissa * obj` inside `value_operation`
   delegates at the mantissa level), and the result is then fed back
   into the `Quantity` constructor:

```python
class MyThing:
    def __rmul__(self, other): return "custom-rmul"

(3*u.mV) * MyThing()
# TypeError: Cannot create a Quantity from a str mantissa: 'custom-rmul'.
#            Did you mean Quantity(value, unit='custom-rmul')?
```

The custom `__rmul__` ran (with the *float* `3.0`, not the Quantity),
its result was discarded into a confusing constructor error, and the
class never got a chance to handle the Quantity itself. The comparison
dunders already solve this correctly with `_is_comparable_operand` →
`NotImplemented`.

**Fix** — reuse the same gate in `_binary_operation` (and therefore in
every arithmetic dunder):

```python
def _binary_operation(self, other, ...):
    if not isinstance(other, Quantity):
        if not _is_comparable_operand(other):
            return NotImplemented
        other = _to_quantity(other)
    ...
```

Then each dunder must propagate it (`maybe_decimal(NotImplemented)`
must not run — check `r is NotImplemented` before wrapping):

```python
def __mul__(self, oc):
    if isinstance(oc, SparseMatrix):
        return oc.__rmul__(self)
    r = self._binary_operation(oc, operator.mul, operator.mul)
    return r if r is NotImplemented else maybe_decimal(r)
```

Note `_is_comparable_operand` accepts `list`/`tuple`, numbers, numpy /
jax / torch / cupy / dask / ndonnx arrays and custom-array wrappers, so
all currently-working operands keep working; only truly foreign objects
start deferring. `q * None` changes from
`TypeError: unsupported operand ... 'float' and 'NoneType'` to the
standard `TypeError: unsupported operand type(s) for *: 'Quantity' and
'NoneType'`, which is an improvement.

---

## Q4 (Medium) — zero-convention not applied to assignment-like APIs

The "concrete 0 is compatible with any dimension" convention
(`_is_concrete_zero`) is honoured by `+`, `-`, and comparisons, but not
by any of the assignment-flavoured entry points:

```python
q = np.array([1., 2., 3.]) * u.mV
q + 0                 # OK
q > 0                 # OK
q[0] = 0              # TypeError
q.at[0].set(0)        # UnitMismatchError
q.at[0].add(0)        # UnitMismatchError
q.clip(min=0)         # UnitMismatchError
q.fill(0)             # DimensionMismatchError
```

`q.clip(min=0)` is particularly jarring since `q > 0` works and
clipping at zero is the single most common clip.

**Fix** — centralise value coercion. Add a helper on `Quantity`:

```python
def _coerce_assign_value(self, value, api_name: str) -> 'Quantity':
    """Coerce `value` for an assignment-like op into self's unit.

    Applies the concrete-zero convention, then converts via in_unit.
    """
    value = _to_quantity(value)
    if (value.unit.is_unitless and not self.unit.is_unitless
            and _is_concrete_zero(value.mantissa)):
        return Quantity(value.mantissa, unit=self.unit)
    return value.in_unit(self.unit)   # raises UnitMismatchError
```

and use it in `__setitem__`, `scatter_*`, `_IndexUpdateRef.set/add/
min/max`, `fill`, and for the bounds in `clip` (align each given bound
via this helper instead of `unit_scale_align_to_first`). This also
resolves Q6 below.

---

## Q5 (Medium) — `==` / `!=` raise on dimension mismatch

```python
(3*u.mV) == (3*u.ms)        # UnitMismatchError
(3*u.mV) in [3*u.ms, 3*u.mV]  # UnitMismatchError  ← container protocol broken
```

Raising from `__eq__` breaks `in`, `list.index`, `remove`, and any
generic code that probes equality. NumPy, pint, and astropy all return
elementwise `False` (or a broadcast `False` array) for incompatible
dimensions; ordering comparisons (`<`, `<=`, …) are the ones that
should raise.

**Fix** — in `_comparison`, special-case `operator.eq` / `operator.ne`:
on `UnitMismatchError` from `in_unit`, return
`xp.zeros_like(broadcast(...), dtype=bool)` for `==` (ones for `!=`)
instead of re-raising; keep the raise for `<//<=/>/>=`. Practical
implementation: pass a flag from `__eq__`/`__ne__`:

```python
def _comparison(self, other, operator_str, operation, *, mismatch_raises=True):
    ...
    try:
        other_value = other.in_unit(self.unit).mantissa
    except UnitMismatchError as e:
        if not mismatch_raises:
            eq = operation is operator.eq
            shape = np.broadcast_shapes(self.shape, other.shape)
            xp = get_backend(self)
            return xp.full(shape, not eq, dtype=bool) if not eq else xp.zeros(shape, dtype=bool)
        raise UnitMismatchError(...) from e
```

If the current raising behaviour is intentional (Brian2 heritage), keep
it — but then document it prominently and consider at least making
`__eq__` non-raising, since the container-protocol breakage is the real
cost. Decide one way; today the docs don't mention it.

---

## Q6 (Medium) — inconsistent exception types for the same mistake

Assigning a plain number into a unit-bearing quantity raises a
different exception type per entry point:

| API | Exception |
|-----|-----------|
| `q[0] = 5` | `TypeError: Only Quantity can be assigned to Quantity` |
| `q.at[0].set(5)` | `UnitMismatchError` |
| `q.fill(5)` | `DimensionMismatchError` |

**Fix** — falls out of the Q4 helper: all three paths funnel through
`_coerce_assign_value`, which raises `UnitMismatchError` with a uniform
message. Keep `TypeError` only for genuinely non-numeric values.

---

## Q7 (Medium) — `**` rejects convertible dimensionless exponents

`__pow__`/`__rpow__` require the exponent to be strictly *unitless*
(`scale == 0 and factor == 1`), but a dimensionless quantity with a
scale or factor (percent-style ratios, `Quantity(x, Unit(scale=3))`
from arithmetic like `mV / uV`) is physically a plain number and every
other code path converts it:

```python
exp = Quantity(200.0, unit=Unit(dim=None, scale=-2))  # == 2.0
exp.to_decimal()       # 2.0
exp == 2.0             # True
(2*u.mV) ** exp        # ValueError: the exponent has to be dimensionless ← BUG
2 ** exp               # ValueError ← same
```

**Fix** — convert instead of testing `is_unitless`:

```python
def __pow__(self, oc):
    self = self.factorless()
    if isinstance(oc, Quantity):
        if not oc.dim.is_dimensionless:
            raise ValueError(...)
        oc = oc.to_decimal()          # folds scale/factor into the value
    ...

def __rpow__(self, oc):
    if not self.dim.is_dimensionless:
        raise ValueError(...)
    return oc ** self.to_decimal()
```

Same change applies to the exponent handling in `__lshift__`/`__rshift__`
guards if strictness there is not deliberate (shifts arguably should
stay strict since they are integer ops).

---

## Q8 (Low) — `divmod` with mixed dimensions evaluates partially

`__floordiv__` supports mixed dimensions (returns a quotient carrying
`unitA/unitB`), but `__mod__` requires matching dimensions. So:

```python
divmod(7*u.ms, 2*u.mV)
# computes the floordiv, then raises UnitMismatchError from the mod
```

`__divmod__` should be atomic. **Fix** — validate first:

```python
def __divmod__(self, oc):
    other = _to_quantity(oc)
    if not other.unit.has_same_dim(self.unit):
        raise UnitMismatchError("divmod requires operands with the same "
                                "dimension", self.unit, other.unit)
    return self.__floordiv__(oc), self.__mod__(oc)
```

(Alternatively decide that floordiv across dimensions is itself
questionable and restrict both — but that is a behaviour change beyond
a bug fix.)

---

## Q9 (Low) — concrete-zero convention diverges between eager and JIT

```python
def f(x):
    return (x - x) + 3*u.mV
f(jnp.ones(3))            # OK: concrete zero, convention applies
jax.jit(f)(jnp.ones(3))   # UnitMismatchError: mV != 1
```

`_is_concrete_zero` deliberately returns `False` for tracers, so code
that happens to feed a zero works eagerly and fails once jitted. This
is inherent to the design (a value-dependent unit rule cannot be traced)
and probably should not change, but it is a landmine.

**Fix (docs + error message)** — when the mismatch involves a unitless
*traced* operand, append a hint to the `UnitMismatchError`:
"note: a concrete 0 would be accepted (zero-compatibility convention),
but this value is a JAX tracer; attach the expected unit explicitly."

---

## Q10 (Low) — `Quantity.dtype` returns `bool` (the Python type)

`dtype` (`_base_quantity.py:1339`) canonicalizes `int`/`float`/`complex`
mantissas through `_canonicalize_dtype` but returns the raw class
`bool` for a Python bool mantissa:

```python
Quantity(True).dtype   # <class 'bool'> — not a dtype
```

Downstream `np.dtype(bool)` happens to work, but the API is
inconsistent. **Fix**: `return np.dtype(bool)` (or route through
`_canonicalize_dtype(bool)` like the others).

---

## Q11 (Low) — `resize()` bypasses the mutation guards

`resize` (`_base_quantity.py:3187`) rebinds `self._mantissa` directly.
Unlike `update_mantissa`/`__setitem__` it has no tracer guard, so under
`jit` it silently rebinds a traced array onto the object (a side effect
the trace does not capture). **Fix** — add the same guard:

```python
if _is_tracer(self.mantissa):
    raise RuntimeError("resize() cannot mutate a traced Quantity; ...")
```

---

## Q12 (Low) — `take(mode='clip')` on an empty NumPy array

The NumPy emulation path clamps indices with `np.clip(idx, 0, max(n-1, 0))`
and then gathers; for `n == 0` the gather raises
`IndexError: cannot do a non-empty take from an empty axes`, whereas
JAX returns fill values / clips to an empty result. Minor; guard
`n == 0` by routing to the `fill` path (all indices are out of bounds).

---

## G2 (Low) — `maybe_decimal(val, unit=...)` ignores `unit` when dimensionless

`maybe_decimal` (`_base_getters.py:649`) checks
`valq.dim.is_dimensionless` *before* looking at `unit`, so an explicit
target unit is silently ignored for dimensionless input. Reorder:

```python
if unit is not None:
    return valq.to_decimal(unit)
if valq.dim.is_dimensionless:
    return valq.to_decimal()
return val
```

---

## Noted, no action proposed

- `tolist()` returns nested lists of scalar `Quantity` objects (NumPy
  returns plain scalars). Deliberate, but worth a docs callout since it
  surprises NumPy users.
- `Quantity(q, unit=incompatible)` raises `ValueError` where the rest of
  the library uses `UnitMismatchError`; trivial to align if touched.
- `_UFUNC_DISPATCH` lazy build has a benign race (two threads may build
  the table twice); last write wins, contents identical.
