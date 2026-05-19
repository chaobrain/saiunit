# Using saiunit with NumPy

`saiunit` accepts both `jax.Array` and `numpy.ndarray` as the underlying
mantissa of a `Quantity`. Every operation in `saiunit.math`,
`saiunit.linalg`, and `saiunit.fft` dispatches to the matching backend, so
you can stay in NumPy end-to-end (interop with scipy, pandas, sklearn) or
mix-and-match.

## Quick start

```python
import numpy as np
import saiunit as u

q = u.Quantity(np.array([1.0, 2.0, 3.0]), unit=u.meter)
print(q.backend)   # 'numpy'
print((q + q).backend)  # 'numpy'
```

```python
import jax.numpy as jnp

q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.meter)
print(q.backend)   # 'jax'
```

## Choosing the default backend

When a `Quantity` is built from a Python scalar or list (no array yet),
saiunit consults the default backend. The default is `'jax'` unless you
change it:

```python
u.set_default_backend("numpy")
q = u.Quantity([1.0, 2.0], unit=u.meter)
print(q.backend)   # 'numpy'

with u.using_backend("jax"):
    q = u.Quantity([1.0, 2.0], unit=u.meter)
    print(q.backend)   # 'jax'

u.set_default_backend(None)  # back to default (jax on tie-breaker)
```

The `using_backend` context manager is the recommended way to run a
particular block of code on a specific backend.

## Mixed-backend operations

When you mix a NumPy-backed `Quantity` with a JAX-backed one:

- If a default backend is set, the result lands on that backend.
- Otherwise, JAX wins (the historical default).

```python
a = u.Quantity(np.array([1.0]), unit=u.meter)
b = u.Quantity(jnp.array([2.0]), unit=u.meter)

(a + b).backend  # 'jax' by default

with u.using_backend("numpy"):
    (a + b).backend  # 'numpy'
```

## Converting between backends

```python
q_np  = u.Quantity(np.array([1.0]), unit=u.meter)
q_jax = q_np.to_jax()    # JAX-backed copy; q_np untouched
q_back = q_jax.to_numpy()
```

Both methods are no-ops (return `self`) if the mantissa is already on the
target backend.

## JAX-only operations

`saiunit.lax`, `saiunit.autograd`, and `saiunit.sparse` use JAX primitives
that have no NumPy equivalent (XLA primitives, autodiff, JAX sparse
matrices). Calling them with a NumPy-backed `Quantity` raises
`saiunit.BackendError`:

```python
q = u.Quantity(np.array([1.0]), unit=u.meter)
u.lax.slice(q, (0,), (1,))
# saiunit.BackendError: saiunit.lax.slice requires the jax backend;
# got numpy-backed Quantity. Call .to_jax() on the input first.

u.lax.slice(q.to_jax(), (0,), (1,))   # works
```

## NumPy ufunc interop

Standard NumPy ufuncs (`np.add`, `np.sin`, `np.exp`, …) preserve units:

```python
q = u.Quantity(np.array([0.0, np.pi / 2]), unit=u.UNITLESS)
np.sin(q)                              # works; dimension-checked

a = u.Quantity(np.array([1.0]), unit=u.meter)
b = u.Quantity(np.array([2.0]), unit=u.meter)
np.add(a, b)                           # Quantity([3.], "m")

c = u.Quantity(np.array([1.0]), unit=u.second)
np.add(a, c)                           # raises UnitMismatchError
```

Mixing a `Quantity` with a plain scalar via NumPy goes through the same
`Quantity.__add__`/`__radd__` path as regular Python arithmetic, so unit
checks fire identically:

```python
np.float64(5) + u.Quantity(np.array([1.0]), unit=u.meter)
# raises UnitMismatchError (the scalar is dimensionless)
```
