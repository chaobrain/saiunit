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

Operations that produce a Python scalar — `float(q)`, `int(q)`,
`operator.index(q)`, `q.tolist()`, `np.asarray(q)`, `hash(q)` — raise
`BackendError` on dask-backed quantities. Call `q.mantissa.compute()` first:

```python
single = u.Quantity(da.from_array(np.array([42.0]), chunks=1), unit=u.meter)
float(single)              # raises BackendError
single.mantissa.compute()  # numpy array; now eager
```

### Mixed-backend arithmetic

Mixing a dask-backed and a non-dask-backed `Quantity` in arithmetic falls
through the default-backend tiebreaker. If the result lands on dask, the
non-dask operand is auto-lifted to a dask array.

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
