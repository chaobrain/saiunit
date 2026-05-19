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

"""Entry guards for JAX-only modules (lax, autograd, sparse).

These modules use JAX primitives (``jax.lax``, ``jax.grad``, JAX sparse
matrices) that have no NumPy equivalent. Calling them with a NumPy-backed
``Quantity`` should fail with a clear, actionable error rather than a
cryptic crash deep inside JAX.
"""

from __future__ import annotations

import functools

from saiunit._exceptions import BackendError


def require_jax_backend(func_name: str, *quantities_or_arrays) -> None:
    """Raise :class:`BackendError` if any input is not jax-compatible.

    - Quantity wrapping non-jax mantissa → reject, naming the backend.
    - Bare cupy / torch arrays → reject (jax cannot lift them).
    - Bare numpy arrays, python scalars → tolerated (jax auto-lifts).
    """
    from saiunit._base_quantity import Quantity
    from saiunit._backend import (
        is_cupy_array, is_torch_array,
    )

    for q in quantities_or_arrays:
        if isinstance(q, Quantity):
            backend = q.backend
            if backend != "jax":
                raise BackendError(
                    f"{func_name} requires the jax backend; got "
                    f"{backend}-backed Quantity. Call .to_jax() on the input first."
                )
            continue
        # Bare arrays: reject cupy/torch; tolerate numpy + jax + scalars.
        if is_cupy_array(q):
            raise BackendError(
                f"{func_name} requires the jax backend; got cupy array. "
                f"Convert to a JAX array first."
            )
        if is_torch_array(q):
            raise BackendError(
                f"{func_name} requires the jax backend; got torch tensor. "
                f"Convert to a JAX array first."
            )
        # Anything else (numpy ndarray, jax array, python scalar) is fine.


def jax_only(fn):
    """Decorator that wraps a function with :func:`require_jax_backend`.

    The wrapped function's name is used in the error message.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        require_jax_backend(fn.__qualname__, *args)
        return fn(*args, **kwargs)

    return wrapper


__all__ = ["require_jax_backend", "jax_only"]
