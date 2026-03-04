# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from __future__ import annotations

from typing import (Sequence, Callable)

from ._jacobian import jacrev, jacfwd
from ._misc import _check_callable

__all__ = [
    'hessian'
]


def hessian(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False
) -> Callable:
    """
    Physical unit-aware Hessian of ``fun`` as a dense array.

    This is the unit-aware counterpart of
    `jax.hessian <https://jax.readthedocs.io/en/latest/_autosummary/jax.hessian.html>`_.
    It computes the Hessian (matrix of second derivatives) while
    correctly propagating physical units. Internally it is implemented
    as ``jacfwd(jacrev(fun))``.

    Parameters
    ----------
    fun : callable
        Function whose Hessian is to be computed. Its arguments at
        positions specified by ``argnums`` should be arrays, scalars,
        or standard Python containers thereof (possibly carrying
        physical units). It should return a scalar output.
    argnums : int or tuple of int, optional
        Specifies which positional argument(s) to differentiate with
        respect to. Default is ``0``.
    has_aux : bool, optional
        If ``True``, ``fun`` is expected to return ``(output, aux)``
        where only ``output`` is differentiated. Default is ``False``.
    holomorphic : bool, optional
        Whether ``fun`` is promised to be holomorphic. Default is
        ``False``.

    Returns
    -------
    hess_fun : callable
        A function with the same arguments as ``fun`` that evaluates
        the Hessian. If ``has_aux=True``, it returns
        ``(hessian, aux)``. Each Hessian leaf carries the correct
        physical units (output unit / input_i unit / input_j unit).

    Notes
    -----
    ``hessian`` generalises to nested Python containers (pytrees).
    The tree structure of ``hessian(fun)(x)`` is formed by taking a
    tree product of the structure of ``fun(x)`` with two copies of
    the structure of ``x``.

    See Also
    --------
    jacrev : Reverse-mode Jacobian computation.
    jacfwd : Forward-mode Jacobian computation.

    Examples
    --------
    Hessian of a unitless quadratic function:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.autograd as suauto
        >>> def f(x):
        ...     return x ** 2 + 3 * x * su.ms + 2 * su.msecond2
        >>> hess_fn = suauto.hessian(f)
        >>> hess_fn(jnp.array(1.0) * su.ms)
        [2]

    Hessian of a cubic function where the result carries units:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.autograd as suauto
        >>> def g(x):
        ...     return x ** 3 + 3 * x * su.msecond2 + 2 * su.msecond3
        >>> hess_fn = suauto.hessian(g)
        >>> hess_fn(jnp.array(1.0) * su.ms)
        [6] * ms
    """
    _check_callable(fun)

    return jacfwd(
        jacrev(fun, argnums, has_aux=has_aux, holomorphic=holomorphic),
        argnums, has_aux=has_aux, holomorphic=holomorphic,
    )
