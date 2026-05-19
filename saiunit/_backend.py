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
    "to_backend",
]

BackendName = Literal["numpy", "jax"]

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


def get_default_backend() -> Optional[BackendName]:
    """Return the currently configured default backend, or None if unset."""
    return _default_backend.get()


def set_default_backend(name: Optional[BackendName]) -> None:
    """Set the default backend used when input backend is ambiguous.

    Parameters
    ----------
    name : {'numpy', 'jax', None}
        Pass ``None`` to clear the default (JAX wins on tie-breaker).
    """
    if name not in ("numpy", "jax", None):
        raise ValueError(
            f"default backend must be 'numpy', 'jax', or None; got {name!r}"
        )
    _default_backend.set(name)


@contextmanager
def using_backend(name: BackendName) -> Iterator[None]:
    """Context manager that temporarily sets the default backend."""
    if name not in ("numpy", "jax"):
        raise ValueError(f"backend must be 'numpy' or 'jax'; got {name!r}")
    token = _default_backend.set(name)
    try:
        yield
    finally:
        _default_backend.reset(token)


def _name_to_xp(name: BackendName) -> ModuleType:
    return _jax_xp if name == "jax" else _numpy_xp


def get_backend(*arrays_or_quantities) -> ModuleType:
    """Return the ``xp`` namespace appropriate for the given inputs.

    Rules
    -----
    1. Flatten any ``Quantity`` inputs to their mantissas.
    2. If only NumPy arrays are present, return the numpy xp.
    3. If only JAX arrays are present, return the jax xp.
    4. If mixed (or only scalars / non-array inputs):
       - If a default backend is set, use it.
       - Otherwise, JAX wins.
    """
    from saiunit._base_quantity import Quantity  # local import to avoid cycle

    mantissas = [a.mantissa if isinstance(a, Quantity) else a for a in arrays_or_quantities]

    has_jax = any(is_jax_array(x) for x in mantissas)
    has_numpy = any(is_numpy_array(x) for x in mantissas)

    if has_jax and not has_numpy:
        return _jax_xp
    if has_numpy and not has_jax:
        return _numpy_xp

    # Mixed, or no arrays at all → consult the default.
    default = _default_backend.get()
    if default is not None:
        return _name_to_xp(default)
    return _jax_xp  # JAX wins on the tie-breaker.


def to_backend(x, name: BackendName):
    """Convert ``x`` to the given backend; no-op if already there."""
    if name == "numpy":
        if is_numpy_array(x):
            return x
        return np.asarray(x)
    if name == "jax":
        if is_jax_array(x):
            return x
        return jnp.asarray(x)
    raise ValueError(f"backend must be 'numpy' or 'jax'; got {name!r}")
