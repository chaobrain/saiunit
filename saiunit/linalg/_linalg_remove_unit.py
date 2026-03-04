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

from typing import (Union, Optional)

import jax
import jax.numpy as jnp

from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as, maybe_custom_array
from saiunit.math._fun_remove_unit import _fun_remove_unit_unary

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

    The condition number is defined as ``norm(x, p) * norm(inv(x), p)``.
    For ``p = 2`` (the default), the condition number is the ratio of the
    largest to the smallest singular value.  The unit is stripped before
    computation and the result is a dimensionless JAX array.

    Parameters
    ----------
    x : array_like or Quantity
        Input of shape ``(..., M, N)`` for which to compute the condition
        number.  If *x* carries a unit, the unit is removed before the
        computation.
    p : {None, 1, -1, 2, -2, inf, -inf, 'fro'}, optional
        Order of the norm used in the condition number computation; see
        :func:`jax.numpy.linalg.norm`.  The default ``p = None`` is
        equivalent to ``p = 2``.  If *p* is not in ``{None, 2, -2}``,
        then *x* must be square (``M = N``).

    Returns
    -------
    out : jax.Array
        Condition number(s) of shape ``x.shape[:-2]``.  Always
        dimensionless.

    See Also
    --------
    saiunit.linalg.matrix_rank : Rank of a matrix via SVD.
    saiunit.linalg.slogdet : Sign and log-determinant of a matrix.

    Notes
    -----
    The condition number is a scalar measure of how sensitive a matrix
    inversion is to numerical errors.  A large condition number indicates
    an ill-conditioned (nearly singular) matrix.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1., 2.],
        ...                [2., 1.]]) * su.meter
        >>> su.linalg.cond(x)
        Array(3., dtype=float32)

    Ill-conditioned (rank-deficient) matrix:

    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1., 2.],
        ...                [0., 0.]]) * su.meter
        >>> su.linalg.cond(x)
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

    The rank is calculated via the Singular Value Decomposition (SVD)
    and determined by the number of singular values greater than the
    specified tolerance.  The unit is stripped before computation and
    the result is always a dimensionless integer array.

    Parameters
    ----------
    M : array_like or Quantity
        Input of shape ``(..., N, K)`` whose rank is to be computed.
        If *M* carries a unit, the unit is removed before computation.
    rtol : float or array_like, optional
        Relative tolerance.  Singular values smaller than
        ``rtol * largest_singular_value`` are considered to be zero.
        If ``None`` (the default), a reasonable default is chosen based
        on the floating-point precision of the input.
    tol : float or None, optional
        Deprecated alias for *rtol*.  Will result in a
        :class:`DeprecationWarning` if used.

    Returns
    -------
    out : jax.Array
        Matrix rank of shape ``M.shape[:-2]``, as a dimensionless
        integer array.

    See Also
    --------
    saiunit.linalg.cond : Condition number of a matrix.
    saiunit.linalg.slogdet : Sign and log-determinant of a matrix.

    Notes
    -----
    The rank calculation may be inaccurate for matrices with very small
    singular values or those that are numerically ill-conditioned.
    Consider adjusting the *rtol* parameter or using a more specialised
    rank-computation method in such cases.

    Examples
    --------
    Full-rank matrix:

    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[1., 2.],
        ...                [3., 4.]]) * su.meter
        >>> su.linalg.matrix_rank(a)
        Array(2, dtype=int32)

    Rank-deficient matrix:

    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> b = jnp.array([[1., 0.],
        ...                [0., 0.]]) * su.meter
        >>> su.linalg.matrix_rank(b)
        Array(1, dtype=int32)
    """
    return _fun_remove_unit_unary(jnp.linalg.matrix_rank, M, rtol=rtol, tol=tol)


@set_module_as('saiunit.linalg')
def slogdet(
    a: Union[jax.typing.ArrayLike, Quantity],
    *,
    method: str | None = None
) -> tuple[jax.Array, jax.Array]:
    """Compute the sign and (natural) logarithm of the absolute determinant.

    SaiUnit implementation of :func:`numpy.linalg.slogdet`.

    The unit is stripped before computation.  Both returned arrays are
    always dimensionless.  This function is more numerically stable
    than computing ``log(det(a))`` directly because it avoids overflow
    and underflow for matrices with very large or very small
    determinants.

    Parameters
    ----------
    a : array_like or Quantity
        Square input of shape ``(..., M, M)`` for which to compute the
        sign and log-determinant.  If *a* carries a unit, the unit is
        removed before computation.
    method : {'lu', 'qr'} or None, optional
        Decomposition method used internally.

        - ``'lu'`` (default) -- use the LU decomposition.
        - ``'qr'`` -- use the QR decomposition.

    Returns
    -------
    sign : jax.Array
        Sign of the determinant (``+1.``, ``-1.``, or ``0.``), of shape
        ``a.shape[:-2]``.
    logabsdet : jax.Array
        Natural logarithm of the absolute value of the determinant, of
        shape ``a.shape[:-2]``.

    See Also
    --------
    saiunit.linalg.cond : Condition number of a matrix.
    saiunit.linalg.matrix_rank : Rank of a matrix via SVD.

    Notes
    -----
    The determinant can be reconstructed as ``sign * exp(logabsdet)``.
    Using :func:`slogdet` instead of :func:`det` avoids numerical
    issues when the determinant is extremely large or small.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[1., 2.],
        ...                [3., 4.]]) * su.meter
        >>> sign, logabsdet = su.linalg.slogdet(a)
        >>> sign
        Array(-1., dtype=float32)
        >>> jnp.exp(logabsdet)
        Array(2., dtype=float32)
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return jnp.linalg.slogdet(a.mantissa, method=method)
    return jnp.linalg.slogdet(a, method=method)
