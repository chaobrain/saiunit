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

"""Backend dispatch for NumPy vs JAX array operations.

This module centralizes the rules for choosing between NumPy and JAX
namespaces. Internal saiunit code should call ``get_backend(*xs)`` to obtain
an ``xp`` namespace and then call array operations through it
(e.g. ``xp.sin(x)`` instead of ``jnp.sin(x)``).

The NumPy namespace is provided by ``array_api_compat.numpy`` (a thin wrapper
that exposes the array-API standard surface on top of plain NumPy). The JAX
namespace is plain ``jax.numpy`` — JAX 0.9+ is already array-API-compatible
and ``array_api_compat`` returns it unmodified.
"""

from __future__ import annotations

import functools
import importlib
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType
from typing import Iterator, Literal, Optional

import array_api_compat.numpy as _numpy_xp
import jax
import jax.numpy as _jax_xp
import jax.numpy as jnp
import numpy as np

from saiunit._exceptions import BackendError


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

__all__ = [
    "get_backend",
    "get_default_backend",
    "set_default_backend",
    "using_backend",
    "is_jax_array",
    "is_numpy_array",
    "is_cupy_array",
    "is_torch_array",
    "is_dask_array",
    "is_ndonnx_array",
    "to_backend",
]

BackendName = Literal["numpy", "jax", "cupy", "torch", "dask", "ndonnx"]

_default_backend: ContextVar[Optional[BackendName]] = ContextVar(
    "saiunit_default_backend", default=None
)


def is_numpy_array(x) -> bool:
    """Return True if ``x`` is a NumPy array or scalar (and not a JAX array).

    Includes ``numpy.ndarray`` as well as numpy scalar types (``numpy.float64``,
    ``numpy.int32``, …) — reductions like ``np.linalg.norm([3., 4.])`` return
    a numpy scalar, and for backend-routing purposes that's NumPy too.
    """
    return isinstance(x, (np.ndarray, np.generic)) and not isinstance(x, jax.Array)


def is_jax_array(x) -> bool:
    """Return True if ``x`` is a ``jax.Array``."""
    return isinstance(x, jax.Array)


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


def is_dask_array(x) -> bool:
    """Return True if ``x`` is a dask Array. False if dask is not installed."""
    da = _try_import("dask.array")
    if da is None:
        return False
    return isinstance(x, da.Array)


def is_ndonnx_array(x) -> bool:
    """Return True if ``x`` is an ndonnx Array. False if ndonnx is not installed."""
    ndonnx = _try_import("ndonnx")
    if ndonnx is None:
        return False
    return isinstance(x, ndonnx.Array)


def get_default_backend() -> Optional[BackendName]:
    """Return the currently configured default backend, or None if unset."""
    return _default_backend.get()


def set_default_backend(name: Optional[BackendName]) -> None:
    """Set the default backend used when input backend is ambiguous.

    Parameters
    ----------
    name : {'numpy', 'jax', 'cupy', 'torch', None}
        Pass ``None`` to clear the default (JAX wins on tie-breaker).
    """
    if name not in ("numpy", "jax", "cupy", "torch", "dask", None):
        raise ValueError(
            f"default backend must be 'numpy', 'jax', 'cupy', 'torch', 'dask', or None; got {name!r}"
        )
    _default_backend.set(name)


@contextmanager
def using_backend(name: BackendName) -> Iterator[None]:
    """Context manager that temporarily sets the default backend."""
    if name not in ("numpy", "jax", "cupy", "torch", "dask"):
        raise ValueError(
            f"backend must be 'numpy', 'jax', 'cupy', 'torch', or 'dask'; got {name!r}"
        )
    token = _default_backend.set(name)
    try:
        yield
    finally:
        _default_backend.reset(token)


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
    elif name == "dask":
        if _try_import("dask.array") is None:
            raise BackendError(
                "dask backend requested but dask is not installed. "
                "Install with: pip install saiunit[dask]"
            )
        import array_api_compat.dask.array as mod  # noqa: F811
    elif name == "ndonnx":
        ndonnx = _try_import("ndonnx")
        if ndonnx is None:
            raise BackendError(
                "ndonnx backend requested but ndonnx is not installed. "
                "Install with: pip install saiunit[ndonnx]"
            )
        mod = ndonnx  # ndonnx is itself array-API-compatible
    else:
        raise ValueError(f"unknown backend: {name!r}")
    _XP_CACHE[name] = mod
    return mod


def _name_to_xp(name: BackendName) -> ModuleType:
    """Deprecated alias retained for any external callers; prefer ``_xp_for``."""
    return _xp_for(name)


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
    has_dask = any(is_dask_array(x) for x in mantissas)
    has_ndonnx = any(is_ndonnx_array(x) for x in mantissas)

    kinds = [name for name, has in
             [("numpy", has_numpy), ("jax", has_jax),
              ("cupy", has_cupy), ("torch", has_torch),
              ("dask", has_dask), ("ndonnx", has_ndonnx)] if has]

    if len(kinds) == 1:
        return _xp_for(kinds[0])

    default = _default_backend.get()
    if default is not None:
        return _xp_for(default)
    return _xp_for("jax")


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
    raise ValueError(f"backend must be one of 'numpy', 'jax', 'cupy', 'torch', 'dask'; got {name!r}")
