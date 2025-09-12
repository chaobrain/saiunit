# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import (Union, Optional)

import jax
import jax.numpy as jnp

from .._base import Quantity
from .._misc import set_module_as, maybe_custom_array
from ..math._fun_remove_unit import _fun_remove_unit_unary

__all__ = [
    # Norms and other numbers
    'cond', 'matrix_rank', 'slogdet',
]


@set_module_as('saiunit.linalg')
def cond(
    x: Union[jax.typing.ArrayLike, Quantity],
    p=None
) -> jax.Array:
    """Compute the condition number of a matrix.

    SaiUnit implementation of :func:`numpy.linalg.cond`.

    The condition number is defined as ``norm(x, p) * norm(inv(x), p)``. For ``p = 2``
    (the default), the condition number is the ratio of the largest to the smallest
    singular value.

    Args:
        x: quantity of shape ``(..., M, N)`` for which to compute the condition number.
        p: the order of the norm to use. One of ``{None, 1, -1, 2, -2, inf, -inf, 'fro'}``;
            see :func:`jax.numpy.linalg.norm` for the meaning of these. The default is ``p = None``,
            which is equivalent to ``p = 2``. If not in ``{None, 2, -2}`` then ``x`` must be square,
            i.e. ``M = N``.

    Returns:
        array of shape ``x.shape[:-2]`` containing the condition number.

    Examples:

        Well-conditioned matrix:

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1, 2],
        ...                [2, 1]]) * u.meter
        >>> u.linalg.cond(x)
        Array(3., dtype=float32)

        Ill-conditioned matrix:

        >>> x = jnp.array([[1, 2],
        ...                [0, 0]]) * u.meter
        >>> u.linalg.cond(x)
        Array(inf, dtype=float32)
    """
    return _fun_remove_unit_unary(jnp.linalg.cond, x, p=p)


@set_module_as('saiunit.linalg')
def matrix_rank(
    M: Union[jax.typing.ArrayLike, Quantity],
    rtol: Optional[Union[jax.typing.ArrayLike, Quantity]] = None,
    *,
    tol: jax.typing.ArrayLike | None = None
) -> jax.Array:
    """Compute the rank of a matrix.

    SaiUnit implementation of :func:`numpy.linalg.matrix_rank`.

    The rank is calculated via the Singular Value Decomposition (SVD), and determined
    by the number of singular values greater than the specified tolerance.

    Args:
        M: quantity of shape ``(..., N, K)`` whose rank is to be computed.
        rtol: optional array of shape ``(...)`` specifying the tolerance. Singular values
            smaller than `rtol * largest_singular_value` are considered to be zero. If
            ``rtol`` is None (the default), a reasonable default is chosen based the
            floating point precision of the input.
        tol: deprecated alias of the ``rtol`` argument. Will result in a
            :class:`DeprecationWarning` if used.

    Returns:
        array of shape ``a.shape[-2]`` giving the matrix rank.

    Notes:
    The rank calculation may be inaccurate for matrices with very small singular
    values or those that are numerically ill-conditioned. Consider adjusting the
    ``rtol`` parameter or using a more specialized rank computation method in such cases.

    Examples:
        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[1, 2],
        ...                [3, 4]]) * u.meter
        >>> u.linalg.matrix_rank(a)
        Array(2, dtype=int32)

        >>> b = jnp.array([[1, 0],  # Rank-deficient matrix
        ...                [0, 0]]) * u.meter
        >>> u.linalg.matrix_rank(b)
        Array(1, dtype=int32)
    """
    return _fun_remove_unit_unary(jnp.linalg.matrix_rank, M, rtol=rtol, tol=tol)


@set_module_as('saiunit.linalg')
def slogdet(
    a: Union[jax.typing.ArrayLike, Quantity],
    *,
    method: str | None = None
) -> tuple[jax.Array, jax.Array]:
    """
    Compute the sign and (natural) logarithm of the determinant of an array.

    SaiUnit implementation of :func:`numpy.linalg.slogdet`.

    Args:
        a: quantity of shape ``(..., M, M)`` for which to compute the sign and log determinant.
        method: the method to use for determinant computation. Options are

        - ``'lu'`` (default): use the LU decomposition.
        - ``'qr'``: use the QR decomposition.

    Returns:
        A tuple of arrays ``(sign, logabsdet)``, each of shape ``a.shape[:-2]``

        - ``sign`` is the sign of the determinant.
        - ``logabsdet`` is the natural log of the determinant's absolute value.

    Examples:
        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[1, 2],
        ...                [3, 4]]) * u.meter
        >>> sign, logabsdet = u.linalg.slogdet(a)
        >>> sign  # -1 indicates negative determinant
        Array(-1., dtype=float32)
        >>> jnp.exp(logabsdet)  # Absolute value of determinant
        Array(2., dtype=float32)
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return jnp.linalg.slogdet(a.mantissa, method=method)
    return jnp.linalg.slogdet(a, method=method)
