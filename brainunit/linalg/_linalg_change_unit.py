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

from typing import Union

import jax
import jax.numpy as jnp

from .._base import Quantity, maybe_decimal, UNITLESS
from .._misc import set_module_as
from ..math._fun_change_unit import (
    dot, multi_dot, vdot, vecdot, inner,
    outer, kron, matmul, tensordot,
    matrix_power, det, cross, unit_change, _fun_change_unit_unary, _fun_change_unit_binary
)

__all__ = [
    # Matrix and vector products
    'dot', 'multi_dot', 'vdot', 'vecdot',
    'inner', 'kron', 'matmul',
    'tensordot', 'matrix_power', 'det',
    'cross',

    # Decompositions
    'cholesky', 'outer',

    # Norms and other numbers
    'det',

    # Solving equations and inverting matrices
    'solve', 'tensorsolve', 'lstsq',
    'inv', 'pinv', 'tensorinv',
]


@unit_change(lambda x: x ** 0.5)
@set_module_as('brainunit.linalg')
def cholesky(
    a: Union[jax.typing.ArrayLike, Quantity],
    *,
    upper: bool = False
) -> Union[Quantity, jax.Array]:
    """Compute the Cholesky decomposition of a matrix.

    JAX implementation of :func:`numpy.linalg.cholesky`.

    The Cholesky decomposition of a matrix `A` is:

    .. math::

        A = U^HU

    or

    .. math::

        A = LL^H

    where `U` is an upper-triangular matrix and `L` is a lower-triangular matrix, and
    :math:`X^H` is the Hermitian transpose of `X`.

    Args:
        a: input array, representing a (batched) positive-definite hermitian matrix.
            Must have shape ``(..., N, N)``.
        upper: if True, compute the upper Cholesky decomposition `L`. if False
            (default), compute the lower Cholesky decomposition `U`.

    Returns:
        array of shape ``(..., N, N)`` representing the Cholesky decomposition
        of the input. If the input is not Hermitian positive-definite, The result
        will contain NaN entries.

    Examples:
        A small real Hermitian positive-definite matrix:

        >>> x = jnp.array([[2., 1.],
        ...                [1., 2.]])

        Lower Cholesky factorization:

        >>> brainunit.linalg.cholesky(x)
        Array([[1.4142135 , 0.        ],
               [0.70710677, 1.2247449 ]], dtype=float32)

        Upper Cholesky factorization:

        >>> brainunit.linalg.cholesky(x, upper=True)
        Array([[1.4142135 , 0.70710677],
               [0.        , 1.2247449 ]], dtype=float32)

        Reconstructing ``x`` from its factorization:

        >>> L = brainunit.linalg.cholesky(x)
        >>> jnp.allclose(x, L @ L.T)
        Array(True, dtype=bool)
    """
    return _fun_change_unit_unary(jnp.linalg.cholesky,
                                  lambda u: u ** 0.5,
                                  a,
                                  upper=upper)


@unit_change(lambda x, y: y / x)
@set_module_as('brainunit.linalg')
def solve(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
) -> Union[jax.typing.ArrayLike, Quantity]:
    """Solve a linear system of equations

    JAX implementation of :func:`numpy.linalg.solve`.

    This solves a (batched) linear system of equations ``a @ x = b``
    for ``x`` given ``a`` and ``b``.

    Args:
        a: array of shape ``(..., N, N)``.
        b: array of shape ``(N,)`` (for 1-dimensional right-hand-side) or
            ``(..., N, M)`` (for batched 2-dimensional right-hand-side).

    Returns:
        An array containing the result of the linear solve. The result has shape ``(..., N)``
        if ``b`` is of shape ``(N,)``, and has shape ``(..., N, M)`` otherwise.

    Examples:
        A simple 3x3 linear system:

        >>> A = jnp.array([[1., 2., 3.],
        ...                [2., 4., 2.],
        ...                [3., 2., 1.]])
        >>> b = jnp.array([14., 16., 10.])
        >>> x = brainunit.linalg.solve(A, b)
        >>> x
        Array([1., 2., 3.], dtype=float32)

        Confirming that the result solves the system:

        >>> jnp.allclose(A @ x, b)
        Array(True, dtype=bool)
    """
    return _fun_change_unit_binary(jnp.linalg.solve,
                                   lambda a, b: b / a,
                                   a,
                                   b)


@unit_change(lambda x, y: y / x)
@set_module_as('brainunit.linalg')
def tensorsolve(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
    axes: tuple[int, ...] | None = None
) -> Union[jax.typing.ArrayLike, Quantity]:
    """Solve the tensor equation a x = b for x.

    JAX implementation of :func:`numpy.linalg.tensorsolve`.

    Args:
        a: input array. After reordering via ``axes`` (see below), shape must be
          ``(*b.shape, *x.shape)``.
        b: right-hand-side array.
        axes: optional tuple specifying axes of ``a`` that should be moved to the end

    Returns:
        array x such that after reordering of axes of ``a``, ``tensordot(a, x, x.ndim)``
        is equivalent to ``b``.

    Examples:
        >>> key1, key2 = jax.random.split(jax.random.key(8675309))
        >>> a = jax.random.normal(key1, shape=(2, 2, 4))
        >>> b = jax.random.normal(key2, shape=(2, 2))
        >>> x = brainunit.linalg.tensorsolve(a, b)
        >>> x.shape
        (4,)

        Now show that ``x`` can be used to reconstruct ``b`` using
        :func:`~jax.numpy.linalg.tensordot`:

        >>> b_reconstructed = brainunit.linalg.tensordot(a, x, axes=x.ndim)
        >>> jnp.allclose(b, b_reconstructed)
        Array(True, dtype=bool)
    """
    return _fun_change_unit_binary(jnp.linalg.tensorsolve,
                                   lambda a, b: b / a,
                                   a,
                                   b,
                                   axes=axes)


@unit_change(lambda x, y: y / x)
@set_module_as('brainunit.linalg')
def lstsq(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
    rcond: float | None = None,
    *,
    numpy_resid: bool = False
) -> tuple[Union[jax.typing.ArrayLike, Quantity], jax.Array, jax.Array, jax.Array]:
    """
    Return the least-squares solution to a linear equation.

    JAX implementation of :func:`numpy.linalg.lstsq`.

    Args:
        a: array of shape ``(M, N)`` representing the coefficient matrix.
        b: array of shape ``(M,)`` or ``(M, K)`` representing the right-hand side.
        rcond: Cut-off ratio for small singular values. Singular values smaller than
            ``rcond * largest_singular_value`` are treated as zero. If None (default),
            the optimal value will be used to reduce floating point errors.
        numpy_resid: If True, compute and return residuals in the same way as NumPy's
            `linalg.lstsq`. This is necessary if you want to precisely replicate NumPy's
            behavior. If False (default), a more efficient method is used to compute residuals.

    Returns:
        Tuple of arrays ``(x, resid, rank, s)`` where

        - ``x`` is a shape ``(N,)`` or ``(N, K)`` array containing the least-squares solution.
        - ``resid`` is the sum of squared residual of shape ``()`` or ``(K,)``.
        - ``rank`` is the rank of the matrix ``a``.
        - ``s`` is the singular values of the matrix ``a``.

    Examples:
        >>> a = jnp.array([[1, 2],
        ...                [3, 4]])
        >>> b = jnp.array([5, 6])
        >>> x, _, _, _ = brainunit.linalg.lstsq(a, b)
        >>> with jnp.printoptions(precision=3):
        ...   print(x)
        [-4.   4.5]
    """
    if isinstance(a, Quantity) and isinstance(b, Quantity):
        r = jnp.linalg.lstsq(a.mantissa, b.mantissa, rcond=rcond, numpy_resid=numpy_resid)
        return maybe_decimal(Quantity(r[0], unit=b.unit / a.unit)), r[1], r[2], r[3]
    elif isinstance(a, Quantity):
        r = jnp.linalg.lstsq(a.mantissa, b, rcond=rcond, numpy_resid=numpy_resid)
        return maybe_decimal(Quantity(r[0], unit=UNITLESS / a.unit)), r[1], r[2], r[3]
    elif isinstance(b, Quantity):
        r = jnp.linalg.lstsq(a, b.mantissa, rcond=rcond, numpy_resid=numpy_resid)
        return maybe_decimal(Quantity(r[0], unit=b.unit)), r[1], r[2], r[3]
    else:
        return jnp.linalg.lstsq(a, b, rcond=rcond, numpy_resid=numpy_resid)


@unit_change(lambda u: u ** -1)
@set_module_as('brainunit.linalg')
def inv(
    a: Union[jax.typing.ArrayLike, Quantity],
) -> Union[jax.typing.ArrayLike, Quantity]:
    """Return the inverse of a square matrix

    JAX implementation of :func:`numpy.linalg.inv`.

    Args:
        a: array of shape ``(..., N, N)`` specifying square array(s) to be inverted.

    Returns:
        Array of shape ``(..., N, N)`` containing the inverse of the input.

    Notes:
        In most cases, explicitly computing the inverse of a matrix is ill-advised. For
        example, to compute ``x = inv(A) @ b``, it is more performant and numerically
        precise to use a direct solve, such as :func:`jax.scipy.linalg.solve`.

    Examples:
        Compute the inverse of a 3x3 matrix

        >>> a = jnp.array([[1., 2., 3.],
        ...                [2., 4., 2.],
        ...                [3., 2., 1.]])
        >>> a_inv = brainunit.linalg.inv(a)
        >>> a_inv  # doctest: +SKIP
        Array([[ 0.        , -0.25      ,  0.5       ],
               [-0.25      ,  0.5       , -0.25000003],
               [ 0.5       , -0.25      ,  0.        ]], dtype=float32)

        Check that multiplying with the inverse gives the identity:

        >>> jnp.allclose(a @ a_inv, jnp.eye(3), atol=1E-5)
        Array(True, dtype=bool)

        Multiply the inverse by a vector ``b``, to find a solution to ``a @ x = b``

        >>> b = jnp.array([1., 4., 2.])
        >>> a_inv @ b
        Array([ 0.  ,  1.25, -0.5 ], dtype=float32)

        Note, however, that explicitly computing the inverse in such a case can lead
        to poor performance and loss of precision as the size of the problem grows.
        Instead, you should use a direct solver like :func:`jax.numpy.linalg.solve`:

        >>> brainunit.linalg.solve(a, b)
         Array([ 0.  ,  1.25, -0.5 ], dtype=float32)
    """
    return _fun_change_unit_unary(jnp.linalg.inv,
                                  lambda u: u ** -1,
                                  a)


@unit_change(lambda u: u ** -1)
@set_module_as('brainunit.linalg')
def pinv(
    a: Union[jax.typing.ArrayLike, Quantity],
    rtol: jax.typing.ArrayLike | None = None,
    hermitian: bool = False,
    *,
    rcond: jax.typing.ArrayLike | None = None,
) -> Union[jax.typing.ArrayLike, Quantity]:
    """Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    JAX implementation of :func:`numpy.linalg.pinv`.

    Args:
        a: array of shape ``(..., M, N)`` containing matrices to pseudo-invert.
        rtol: float or array_like of shape ``a.shape[:-2]``. Specifies the cutoff
            for small singular values.of shape ``(...,)``.
            Cutoff for small singular values; singular values smaller
            ``rtol * largest_singular_value`` are treated as zero. The default is
            determined based on the floating point precision of the dtype.
        hermitian: if True, then the input is assumed to be Hermitian, and a more
            efficient algorithm is used (default: False)
        rcond: deprecated alias of the ``rtol`` argument. Will result in a
            :class:`DeprecationWarning` if used.

    Returns:
        An array of shape ``(..., N, M)`` containing the pseudo-inverse of ``a``.

    Examples:
        >>> a = jnp.array([[1, 2],
        ...                [3, 4],
        ...                [5, 6]])
        >>> a_pinv = brainunit.linalg.pinv(a)
        >>> a_pinv  # doctest: +SKIP
        Array([[-1.333332  , -0.33333257,  0.6666657 ],
               [ 1.0833322 ,  0.33333272, -0.41666582]], dtype=float32)

        The pseudo-inverse operates as a multiplicative inverse so long as the
        output is not rank-deficient:

        >>> jnp.allclose(a_pinv @ a, jnp.eye(2), atol=1E-4)
        Array(True, dtype=bool)
    """
    return _fun_change_unit_unary(jnp.linalg.pinv,
                                  lambda u: u ** -1,
                                  a,
                                  rtol=rtol,
                                  hermitian=hermitian,
                                  rcond=rcond)


@unit_change(lambda u: u ** -1)
@set_module_as('brainunit.linalg')
def tensorinv(
    a: Union[jax.typing.ArrayLike, Quantity],
    ind: int = 2,
) -> Union[jax.typing.ArrayLike, Quantity]:
    """Compute the tensor inverse of an array.

    JAX implementation of :func:`numpy.linalg.tensorinv`.

    This computes the inverse of the :func:`~jax.numpy.linalg.tensordot`
    operation with the same ``ind`` value.

    Args:
        a: array to be inverted. Must have ``prod(a.shape[:ind]) == prod(a.shape[ind:])``
        ind: positive integer specifying the number of indices in the tensor product.

    Returns:
        array of shape ``(*a.shape[ind:], *a.shape[:ind])`` containing the
        tensor inverse of ``a``.

    Examples:
        >>> key = jax.random.key(1337)
        >>> x = jax.random.normal(key, shape=(2, 2, 4))
        >>> xinv = brainunit.linalg.tensorinv(x, 2)
        >>> xinv_x = brainunit.linalg.tensordot(xinv, x, axes=2)
        >>> jnp.allclose(xinv_x, jnp.eye(4), atol=1E-4)
        Array(True, dtype=bool)
        """
    return _fun_change_unit_unary(jnp.linalg.tensorinv,
                                  lambda u: u ** -1,
                                  a,
                                  ind=ind)
