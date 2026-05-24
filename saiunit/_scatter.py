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

"""Multi-backend functional scatter dispatch.

This module powers ``Quantity.at[...].set(...)`` (and the other index-update
ops) on every supported backend: numpy, jax, cupy, torch, dask, and ndonnx.

The public surface is a single function, :func:`scatter`. It takes a backend
mantissa array, an index expression, a value (or callable for ``"apply"``),
and an operation name. It returns a new array with the update applied,
matching JAX's functional-update semantics as closely as the backend permits.

Backend support matrix
----------------------

================  =======  =====  =====  =====  ============  =====
Op                jax      numpy  cupy   torch  dask          ndonnx
================  =======  =====  =====  =====  ============  =====
get               native   ok     ok     ok     ok            err
set               native   ok     ok     ok     mask/slice    err
add               native   ok     ok     ok     mask/slice    err
multiply          native   ok     ok     ok*    mask/slice    err
divide            native   ok     ok     ok*    mask/slice    err
power             native   ok     ok     ok*    mask/slice    err
min               native   ok     ok     ok*    mask/slice    err
max               native   ok     ok     ok*    mask/slice    err
apply             native   ok     ok     ok*    mask/slice    err
================  =======  =====  =====  =====  ============  =====

*torch does not natively scatter-multiply/divide/min/max with repeated-index
accumulation; this dispatch uses gather + op + scatter so repeated indices on
torch follow last-write-wins semantics rather than JAX's all-updates-applied.
For ``add`` torch uses ``index_put_(accumulate=True)`` and matches JAX.

The four JAX-only kwargs are handled as follows on non-JAX backends:

* ``mode='clip'``: emulated for scalar-int and 1D integer-array indices.
* ``mode='drop'``/``'fill'`` / default ``'promise_in_bounds'``: emulated for
  the same index shapes; out-of-bounds updates are dropped (or, for ``get``,
  filled with ``fill_value``).
* ``indices_are_sorted`` / ``unique_indices``: silently ignored.

For other index types (slice, boolean mask, ellipsis, tuples mixing these),
``mode`` falls back to the backend's native behavior — usually a no-op since
slices and boolean masks of the source shape can't go out of bounds.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from saiunit._backend import (
    is_cupy_array,
    is_dask_array,
    is_jax_array,
    is_ndonnx_array,
    is_numpy_array,
    is_torch_array,
)
from saiunit._exceptions import BackendError
from saiunit._jax_compat import HAS_JAX

__all__ = ["scatter"]

_VALID_OPS = frozenset(
    {"get", "set", "add", "multiply", "divide", "power", "min", "max", "apply"}
)
# Ops on which 'value' is the function to apply, not data to scatter.
_FUNCTIONAL_OPS = frozenset({"apply"})


def scatter(
    mantissa,
    index,
    value,
    op: str,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | None = None,
    fill_value: Any = None,
):
    """Return a new array with ``op`` applied at ``index``.

    Parameters
    ----------
    mantissa : array
        Backend-native source array (numpy, jax, cupy, torch, dask, or ndonnx).
    index : int, slice, ellipsis, tuple, array
        Index expression to scatter at.
    value : array_like or callable
        For ``op='apply'``, a callable taking ``mantissa[index]`` and returning
        a new value. For ``op='power'``, an integer exponent. Otherwise the
        value(s) to scatter. ``op='get'`` ignores this argument.
    op : str
        One of ``'get'``, ``'set'``, ``'add'``, ``'multiply'``, ``'divide'``,
        ``'power'``, ``'min'``, ``'max'``, ``'apply'``.
    indices_are_sorted, unique_indices : bool
        Hints. Honored on JAX; ignored elsewhere.
    mode : {'promise_in_bounds', 'clip', 'drop', 'fill'} or None
        Out-of-bounds behavior. Emulated on non-JAX backends for scalar-int and
        1D-integer-array indices; falls back to native semantics for other
        index types.
    fill_value : scalar, optional
        For ``op='get'`` with ``mode='fill'``, the value returned at
        out-of-bounds positions.
    """
    if op not in _VALID_OPS:
        raise ValueError(
            f"unknown scatter op: {op!r}; must be one of {sorted(_VALID_OPS)}"
        )

    if is_jax_array(mantissa):
        return _scatter_jax(
            mantissa, index, value, op,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
            fill_value=fill_value,
        )
    if is_numpy_array(mantissa):
        return _scatter_numpy(mantissa, index, value, op, mode=mode, fill_value=fill_value)
    if is_cupy_array(mantissa):
        return _scatter_cupy(mantissa, index, value, op, mode=mode, fill_value=fill_value)
    if is_torch_array(mantissa):
        return _scatter_torch(mantissa, index, value, op, mode=mode, fill_value=fill_value)
    if is_dask_array(mantissa):
        return _scatter_dask(mantissa, index, value, op, mode=mode, fill_value=fill_value)
    if is_ndonnx_array(mantissa):
        raise BackendError(
            "Quantity.at indexed-update is not supported on the ndonnx backend. "
            "Call .to_numpy() (or another concrete backend) on the input first."
        )

    # Bare Python scalar / list / tuple: promote to the default backend and retry.
    from saiunit._backend import _xp_for, get_default_backend
    default = get_default_backend() or ("jax" if HAS_JAX else "numpy")
    xp = _xp_for(default)
    promoted = xp.asarray(mantissa)
    return scatter(
        promoted, index, value, op,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode,
        fill_value=fill_value,
    )


# ---------------------------------------------------------------------------
# JAX — pass through to .at[index].<op>(...)
# ---------------------------------------------------------------------------

def _scatter_jax(
    mantissa,
    index,
    value,
    op: str,
    *,
    indices_are_sorted: bool,
    unique_indices: bool,
    mode: str | None,
    fill_value: Any,
):
    at = mantissa.at[index]
    common_kw = dict(
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode,
    )
    if op == "get":
        return at.get(fill_value=fill_value, **common_kw)
    if op == "apply":
        return at.apply(value, **common_kw)
    if op == "power":
        return at.power(value, **common_kw)
    # set / add / multiply / divide / min / max
    return getattr(at, op)(value, **common_kw)


# ---------------------------------------------------------------------------
# Simple-index normalization (shared by numpy/cupy/torch emulation paths)
# ---------------------------------------------------------------------------

def _is_simple_int_index(index) -> bool:
    """Return True iff ``index`` is a scalar int or a 1D integer ndarray.

    These are the cases for which we can cheaply emulate JAX's ``mode``
    semantics. Anything else (slices, boolean masks, tuples, ellipsis) falls
    through to the backend's native behavior.
    """
    if isinstance(index, (int, np.integer)):
        return True
    if isinstance(index, np.ndarray) and index.ndim == 1 and index.dtype.kind in ("i", "u"):
        return True
    return False


def _normalize_simple_int(index, n: int, mode: str | None):
    """Map a scalar-int or 1D-int index to ``(safe_index, valid_mask)``.

    ``safe_index`` is always in ``[0, n)`` so the caller can index without
    raising. ``valid_mask`` mirrors ``index`` and tells which positions were
    originally in bounds; ``None`` means "all valid" (e.g., ``mode='clip'``
    has no notion of validity — it just clips).
    """
    if isinstance(index, (int, np.integer)):
        idx = int(index)
        in_bounds = -n <= idx < n
        norm = idx % n if in_bounds else 0
        if mode == "clip":
            # Clip — clamp negative-OOB to 0 and positive-OOB to n-1.
            norm = max(0, min(n - 1, idx if idx >= 0 else idx + n))
            return norm, None
        return norm, np.bool_(in_bounds)
    # 1D int array
    arr = np.asarray(index)
    if mode == "clip":
        safe = np.where(arr < 0, np.maximum(0, arr + n), np.minimum(n - 1, arr))
        return safe, None
    valid = (arr >= -n) & (arr < n)
    safe = np.where(valid, arr % n, 0)
    return safe, valid


# ---------------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------------

_NUMPY_UFUNC_AT = {
    "add": "add",
    "multiply": "multiply",
    "divide": "true_divide",
    "min": "minimum",
    "max": "maximum",
}


def _scatter_numpy(mantissa, index, value, op: str, *, mode, fill_value):
    return _scatter_npy_like(np, mantissa, index, value, op, mode=mode, fill_value=fill_value)


def _scatter_cupy(mantissa, index, value, op: str, *, mode, fill_value):
    import cupy as cp  # type: ignore
    return _scatter_npy_like(cp, mantissa, index, value, op, mode=mode, fill_value=fill_value)


def _scatter_npy_like(xp, mantissa, index, value, op: str, *, mode, fill_value):
    """Shared numpy/cupy scatter. ``xp`` is the numpy-API module."""
    if op == "get":
        return _scatter_npy_get(xp, mantissa, index, mode=mode, fill_value=fill_value)

    # apply takes a callable in `value`
    if op == "apply":
        out = mantissa.copy()
        if not _is_simple_int_index(index) or mode == "clip":
            # native indexing; mode='clip' falls through to natural semantics
            out[index] = value(out[index])
            return out
        safe, valid = _normalize_simple_int(index, mantissa.shape[0], mode)
        if valid is None or (isinstance(valid, np.bool_) and bool(valid)):
            out[safe] = value(out[safe])
        elif isinstance(valid, np.bool_):
            # scalar idx, OOB: drop
            pass
        else:
            # 1D mask: only apply where valid
            sub_safe = safe[valid]
            out[sub_safe] = value(out[sub_safe])
        return out

    # power: gather + pow + scatter (np.power.at doesn't exist)
    if op == "power":
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"power exponent must be an integer, but got {type(value).__name__}")
        out = mantissa.copy()
        if not _is_simple_int_index(index) or mode == "clip":
            out[index] = out[index] ** value
            return out
        safe, valid = _normalize_simple_int(index, mantissa.shape[0], mode)
        if valid is None or (isinstance(valid, np.bool_) and bool(valid)):
            out[safe] = out[safe] ** value
        elif isinstance(valid, np.bool_):
            pass
        else:
            sub_safe = safe[valid]
            out[sub_safe] = out[sub_safe] ** value
        return out

    out = mantissa.copy()

    # `set` — for simple int idx with mode emulation, drop OOB; for everything
    # else, native assignment.
    if op == "set":
        if not _is_simple_int_index(index) or mode == "clip":
            if mode == "clip" and _is_simple_int_index(index):
                safe, _ = _normalize_simple_int(index, mantissa.shape[0], "clip")
                out[safe] = value
            else:
                out[index] = value
            return out
        safe, valid = _normalize_simple_int(index, mantissa.shape[0], mode)
        if valid is None or (isinstance(valid, np.bool_) and bool(valid)):
            out[safe] = value
        elif isinstance(valid, np.bool_):
            pass  # scalar OOB → drop
        else:
            # 1D mask: write only where valid
            sub_safe = safe[valid]
            sub_value = _take_masked(value, valid, xp)
            out[sub_safe] = sub_value
        return out

    # add / multiply / divide / min / max — use ufunc.at on the backend
    ufunc_name = _NUMPY_UFUNC_AT[op]
    ufunc = getattr(xp, ufunc_name)
    if not _is_simple_int_index(index) or mode == "clip":
        if mode == "clip" and _is_simple_int_index(index):
            safe, _ = _normalize_simple_int(index, mantissa.shape[0], "clip")
            ufunc.at(out, safe, value)
        else:
            ufunc.at(out, index, value)
        return out
    safe, valid = _normalize_simple_int(index, mantissa.shape[0], mode)
    if valid is None or (isinstance(valid, np.bool_) and bool(valid)):
        ufunc.at(out, safe, value)
    elif isinstance(valid, np.bool_):
        pass
    else:
        sub_safe = safe[valid]
        sub_value = _take_masked(value, valid, xp)
        ufunc.at(out, sub_safe, sub_value)
    return out


def _take_masked(value, valid, xp):
    """Subset ``value`` by ``valid`` if it's a 1D array of matching length."""
    if isinstance(value, (int, float, complex, np.number)):
        return value
    arr = xp.asarray(value)
    if arr.ndim == 0:
        return arr
    if arr.ndim == 1 and arr.shape[0] == valid.shape[0]:
        return arr[valid]
    # broadcast cases (e.g. value is scalar-like 0-D); return as-is
    return arr


def _scatter_npy_get(xp, mantissa, index, *, mode, fill_value):
    """Get with optional mode emulation. Mirrors JAX's ``.at[idx].get()``."""
    n = mantissa.shape[0] if mantissa.ndim else 0
    if not _is_simple_int_index(index):
        # Native indexing; mode is best-effort no-op for complex index types
        return mantissa[index]
    if mode in (None, "promise_in_bounds", "clip"):
        # JAX clips OOB on get for both default and 'clip'
        safe, _ = _normalize_simple_int(index, n, "clip")
        return mantissa[safe]
    if mode in ("drop", "fill"):
        if isinstance(index, (int, np.integer)):
            idx = int(index)
            in_bounds = -n <= idx < n
            if in_bounds:
                return mantissa[idx % n]
            fill = _resolve_fill_value(fill_value, mantissa.dtype, xp)
            return xp.asarray(fill, dtype=mantissa.dtype) if hasattr(xp, "asarray") else fill
        # 1D int array
        arr = np.asarray(index)
        valid = (arr >= -n) & (arr < n)
        safe = np.where(valid, arr % n, 0)
        result = mantissa[safe]
        fill = _resolve_fill_value(fill_value, mantissa.dtype, xp)
        return xp.where(xp.asarray(valid), result, xp.asarray(fill, dtype=mantissa.dtype))
    # Unknown mode — fall through
    return mantissa[index]


def _resolve_fill_value(fill_value, dtype, xp):
    """Resolve JAX's documented default fill value when one is not supplied."""
    if fill_value is not None:
        return fill_value
    dt = np.dtype(dtype)
    if dt.kind == "f" or dt.kind == "c":
        return np.nan
    if dt.kind == "i":
        return np.iinfo(dt).min
    if dt.kind == "u":
        return np.iinfo(dt).max
    if dt.kind == "b":
        return True
    return 0


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------

def _scatter_torch(mantissa, index, value, op: str, *, mode, fill_value):
    import torch  # type: ignore

    if op == "get":
        return _scatter_torch_get(mantissa, index, mode=mode, fill_value=fill_value)

    # Normalize index: convert numpy/jax arrays of integers to torch tensors
    t_index = _torch_normalize_index(index, mantissa, mode, op)

    if op == "apply":
        out = mantissa.clone()
        out[t_index] = value(out[t_index])
        return out

    if op == "power":
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"power exponent must be an integer, but got {type(value).__name__}")
        out = mantissa.clone()
        out[t_index] = out[t_index] ** value
        return out

    out = mantissa.clone()

    # `add` with repeated-index accumulation: use index_put_ + accumulate=True
    # when the index is expressible as a tuple of integer tensors.
    if op == "add":
        idx_tuple = _torch_index_to_tuple(t_index, out.shape)
        if idx_tuple is not None:
            v = _torch_value_as_tensor(value, out, expected_shape=None)
            out.index_put_(idx_tuple, v, accumulate=True)
            return out
        # Fallback: native assignment (no accumulation for repeated idx)
        cur = out[t_index]
        v = _torch_value_as_tensor(value, out, expected_shape=cur.shape)
        out[t_index] = cur + v
        return out

    if op == "set":
        v = _torch_value_as_tensor(value, out, expected_shape=None)
        out[t_index] = v
        return out

    if op == "multiply":
        cur = out[t_index]
        v = _torch_value_as_tensor(value, out, expected_shape=cur.shape)
        out[t_index] = cur * v
        return out

    if op == "divide":
        cur = out[t_index]
        v = _torch_value_as_tensor(value, out, expected_shape=cur.shape)
        out[t_index] = cur / v
        return out

    if op == "min":
        cur = out[t_index]
        v = _torch_value_as_tensor(value, out, expected_shape=cur.shape)
        out[t_index] = torch.minimum(cur, v)
        return out

    if op == "max":
        cur = out[t_index]
        v = _torch_value_as_tensor(value, out, expected_shape=cur.shape)
        out[t_index] = torch.maximum(cur, v)
        return out

    raise AssertionError(f"unreachable: {op!r}")


def _torch_normalize_index(index, mantissa, mode, op):
    """Convert ``index`` into something torch's ``__getitem__`` accepts.

    Also applies mode emulation for scalar-int and 1D-int-array indices when
    the backend can't naturally honor JAX's drop/clip semantics.
    """
    import torch  # type: ignore

    # Scalar int — torch happily accepts Python ints
    if isinstance(index, (int, np.integer)):
        n = mantissa.shape[0]
        if mode == "clip":
            safe = max(0, min(n - 1, index if index >= 0 else index + n))
            return safe
        # default / drop / fill / promise_in_bounds — if OOB, the caller will
        # handle dropping via _torch_value_as_tensor masking; clip silently here
        if mode in (None, "promise_in_bounds", "drop", "fill") and op != "get":
            in_bounds = -n <= index < n
            if not in_bounds:
                # Signal "drop entirely" via a no-op index using `slice(0, 0)`
                # — assigning to an empty slice is a no-op for set/add/etc.
                return slice(0, 0)
            return int(index) % n
        return int(index)

    # 1D int array (numpy, jax, etc.) — convert
    if isinstance(index, np.ndarray) and index.ndim == 1 and index.dtype.kind in ("i", "u"):
        n = mantissa.shape[0]
        if mode == "clip":
            arr = np.where(index < 0, np.maximum(0, index + n), np.minimum(n - 1, index))
            return torch.as_tensor(arr, dtype=torch.long, device=mantissa.device)
        valid = (index >= -n) & (index < n)
        safe = np.where(valid, index % n, 0)
        if not valid.all():
            # Subset to in-bounds only — caller paths use this for set/add etc.
            sub = safe[valid]
            return torch.as_tensor(sub, dtype=torch.long, device=mantissa.device)
        return torch.as_tensor(safe, dtype=torch.long, device=mantissa.device)

    # jax/torch tensor index — convert to torch.long
    if hasattr(index, "__array__") and not isinstance(index, torch.Tensor):
        try:
            arr = np.asarray(index)
        except Exception:
            return index
        if arr.dtype.kind in ("i", "u"):
            return torch.as_tensor(arr, dtype=torch.long, device=mantissa.device)
        if arr.dtype.kind == "b":
            return torch.as_tensor(arr, dtype=torch.bool, device=mantissa.device)
        return arr

    return index


def _torch_index_to_tuple(t_index, shape):
    """Return a tuple of torch.long tensors suitable for ``index_put_`` if
    possible, else ``None`` (caller should fall back).
    """
    import torch  # type: ignore

    if isinstance(t_index, slice):
        return None
    if isinstance(t_index, int):
        return (torch.tensor([t_index], dtype=torch.long),)
    if isinstance(t_index, torch.Tensor):
        if t_index.dtype == torch.bool:
            return None
        if t_index.ndim == 1:
            return (t_index,)
        return None
    return None


def _torch_value_as_tensor(value, mantissa, expected_shape):
    """Coerce ``value`` to a torch tensor matching mantissa's dtype/device."""
    import torch  # type: ignore

    if isinstance(value, torch.Tensor):
        v = value.to(dtype=mantissa.dtype, device=mantissa.device)
    elif isinstance(value, (int, float, complex, bool, np.number)):
        v = torch.tensor(value, dtype=mantissa.dtype, device=mantissa.device)
    else:
        v = torch.as_tensor(
            np.asarray(value), dtype=mantissa.dtype, device=mantissa.device
        )
    if expected_shape is not None and v.shape != tuple(expected_shape):
        # Allow broadcast — torch handles scalar broadcast naturally
        if v.ndim == 0 or v.numel() == 1:
            return v
    return v


def _scatter_torch_get(mantissa, index, *, mode, fill_value):
    import torch  # type: ignore

    if isinstance(index, (int, np.integer)):
        n = mantissa.shape[0]
        if mode in (None, "promise_in_bounds", "clip"):
            safe = max(0, min(n - 1, index if index >= 0 else index + n))
            return mantissa[safe]
        if mode in ("drop", "fill"):
            if -n <= index < n:
                return mantissa[int(index) % n]
            fill = _resolve_fill_value(
                fill_value, np.dtype(_torch_dtype_to_numpy(mantissa.dtype)), np
            )
            return torch.tensor(fill, dtype=mantissa.dtype, device=mantissa.device)

    if isinstance(index, np.ndarray) and index.ndim == 1 and index.dtype.kind in ("i", "u"):
        n = mantissa.shape[0]
        valid = (index >= -n) & (index < n)
        safe = np.where(valid, index % n, 0)
        result = mantissa[torch.as_tensor(safe, dtype=torch.long, device=mantissa.device)]
        if mode in ("drop", "fill") and not valid.all():
            fill = _resolve_fill_value(
                fill_value, np.dtype(_torch_dtype_to_numpy(mantissa.dtype)), np
            )
            valid_t = torch.as_tensor(valid, dtype=torch.bool, device=mantissa.device)
            fill_t = torch.full_like(result, fill)
            return torch.where(valid_t, result, fill_t)
        return result

    # Other index types — pass through
    return mantissa[index]


def _torch_dtype_to_numpy(dt):
    """Map a torch dtype to a numpy dtype string for fill-value resolution."""
    s = str(dt).replace("torch.", "")
    return s


# ---------------------------------------------------------------------------
# Dask — keep the graph lazy
# ---------------------------------------------------------------------------

_DASK_HINT = (
    "Use a boolean mask of the source shape, a slice, an int, or a 1D int "
    "array; or call .to_numpy() to materialize the array first."
)


def _scatter_dask(mantissa, index, value, op: str, *, mode, fill_value):
    import dask.array as da  # type: ignore

    if op == "get":
        return _scatter_dask_get(mantissa, index, mode=mode, fill_value=fill_value)

    # Boolean mask of same shape → da.where
    if isinstance(index, np.ndarray) and index.dtype == bool and index.shape == mantissa.shape:
        mask = da.from_array(index, chunks=mantissa.chunks)
        return _dask_apply_with_mask(mantissa, mask, value, op)
    if isinstance(index, da.Array) and index.dtype == bool and index.shape == mantissa.shape:
        return _dask_apply_with_mask(mantissa, index, value, op)

    # Slice / ellipsis / int / 1D int array → build a positional mask
    mask, idx_meta = _dask_index_to_mask(mantissa, index, mode)
    if mask is None:
        raise NotImplementedError(
            f"Quantity.at on dask backend does not support index type "
            f"{type(index).__name__!r} for op {op!r}. {_DASK_HINT}"
        )
    return _dask_apply_with_mask(mantissa, mask, value, op)


def _dask_apply_with_mask(mantissa, mask, value, op: str):
    """Apply an op at positions selected by a same-shape boolean mask."""
    import dask.array as da  # type: ignore

    if op == "set":
        return da.where(mask, value, mantissa)
    if op == "add":
        return da.where(mask, mantissa + value, mantissa)
    if op == "multiply":
        return da.where(mask, mantissa * value, mantissa)
    if op == "divide":
        return da.where(mask, mantissa / value, mantissa)
    if op == "min":
        return da.where(mask, da.minimum(mantissa, value), mantissa)
    if op == "max":
        return da.where(mask, da.maximum(mantissa, value), mantissa)
    if op == "power":
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"power exponent must be an integer, but got {type(value).__name__}")
        return da.where(mask, mantissa ** value, mantissa)
    if op == "apply":
        return da.where(mask, value(mantissa), mantissa)
    raise AssertionError(f"unreachable: {op!r}")


def _dask_index_to_mask(mantissa, index, mode):
    """Convert simple-shape index expressions into a same-shape bool mask.

    Returns ``(mask, meta)`` or ``(None, None)`` if the index can't be
    represented this way in this release.
    """
    import dask.array as da  # type: ignore

    shape = mantissa.shape

    # Ellipsis or full slice → all-True mask
    if index is Ellipsis:
        return da.ones(shape, chunks=mantissa.chunks, dtype=bool), None

    if isinstance(index, slice):
        if mantissa.ndim != 1:
            return None, None
        idx = np.zeros(shape[0], dtype=bool)
        idx[index] = True
        return da.from_array(idx, chunks=mantissa.chunks), None

    if isinstance(index, (int, np.integer)):
        if mantissa.ndim != 1:
            return None, None
        n = shape[0]
        if not (-n <= int(index) < n):
            if mode in (None, "promise_in_bounds", "drop", "fill"):
                return da.zeros(shape, chunks=mantissa.chunks, dtype=bool), None
            if mode == "clip":
                safe = max(0, min(n - 1, int(index) if int(index) >= 0 else int(index) + n))
                mask = np.zeros(n, dtype=bool)
                mask[safe] = True
                return da.from_array(mask, chunks=mantissa.chunks), None
            return None, None
        pos = int(index) % n
        mask = np.zeros(n, dtype=bool)
        mask[pos] = True
        return da.from_array(mask, chunks=mantissa.chunks), None

    if isinstance(index, np.ndarray) and index.ndim == 1 and index.dtype.kind in ("i", "u"):
        if mantissa.ndim != 1:
            return None, None
        n = shape[0]
        valid = (index >= -n) & (index < n)
        if mode == "clip":
            safe_idx = np.where(index < 0, np.maximum(0, index + n), np.minimum(n - 1, index))
        else:
            safe_idx = index[valid] % n
        mask = np.zeros(n, dtype=bool)
        mask[safe_idx] = True
        return da.from_array(mask, chunks=mantissa.chunks), None

    return None, None


def _scatter_dask_get(mantissa, index, *, mode, fill_value):
    import dask.array as da  # type: ignore

    # Native dask indexing handles slice, bool mask, int, 1D int array.
    if isinstance(index, (slice, type(...), int, np.integer)):
        return mantissa[index]
    if isinstance(index, np.ndarray):
        if index.dtype == bool:
            return mantissa[da.from_array(index, chunks=mantissa.chunks)]
        if index.ndim == 1 and index.dtype.kind in ("i", "u"):
            n = mantissa.shape[0]
            if mode in ("drop", "fill"):
                valid = (index >= -n) & (index < n)
                if not valid.all():
                    safe = np.where(valid, index % n, 0)
                    result = mantissa[da.from_array(safe, chunks=min(len(safe), 1024))]
                    fill = _resolve_fill_value(fill_value, mantissa.dtype, np)
                    return da.where(
                        da.from_array(valid, chunks=result.chunks),
                        result,
                        da.from_array(np.full(len(safe), fill, dtype=mantissa.dtype),
                                      chunks=result.chunks),
                    )
            return mantissa[da.from_array(index, chunks=min(len(index), 1024))]
    if isinstance(index, da.Array):
        return mantissa[index]
    raise NotImplementedError(
        f"Quantity.at[..].get on dask backend does not support index type "
        f"{type(index).__name__!r}. {_DASK_HINT}"
    )
