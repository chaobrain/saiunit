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

from typing import Union

import jax
import jax.numpy as jnp

from saiunit._base_unit import UNITLESS
from saiunit._base_getters import maybe_decimal
from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as, maybe_custom_array
from saiunit.math._fun_change_unit import (
    dot, multi_dot, vdot, vecdot, inner,
    outer, kron, matmul, tensordot,
    matrix_power, det, cross, unit_change, _fun_change_unit_unary, _fun_change_unit_binary
)

__all__ = [
    # Matrix and vector products
    'dot', 'multi_dot', 'vdot', 'vecdot',
    'inner', 'kron', 'matmul',
    'tensordot', 'matrix_power',
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
@set_module_as('saiunit.linalg')
def cholesky(
    a: Union[jax.typing.ArrayLike, Quantity],
    *,
    upper: bool = False,
    symmetrize_input: bool = True,
) -> Union[Quantity, jax.Array]:
    """
    Compute the Cholesky decomposition of a matrix.

    SaiUnit implementation of :func:`numpy.linalg.cholesky`.

    The Cholesky decomposition of a positive-definite Hermitian matrix
    `A` is:

    .. math::

        A = LL^H \\quad \\text{(lower)} \\qquad \\text{or} \\qquad A = U^HU \\quad \\text{(upper)}

    where `L` is lower-triangular, `U` is upper-triangular, and
    :math:`X^H` is the Hermitian transpose of `X`.

    For a Quantity with unit *u*, the resulting unit is ``u ** 0.5``.

    Parameters
    ----------
    a : array_like or Quantity
        Input quantity representing a (batched) positive-definite Hermitian
        matrix. Must have shape ``(..., N, N)``.
    upper : bool, optional
        If ``True``, compute the upper Cholesky factor `U`. If ``False``
        (default), compute the lower Cholesky factor `L`.
    symmetrize_input : bool, optional
        If ``True`` (default), symmetrize the input before decomposition
        for improved autodiff behavior.

    Returns
    -------
    out : ndarray or Quantity
        Cholesky factor of shape ``(..., N, N)``. The resulting unit is
        ``a.unit ** 0.5``. If the input is not Hermitian positive-definite,
        the result will contain NaN entries.

    See Also
    --------
    saiunit.linalg.inv : Compute the inverse of a matrix.
    saiunit.linalg.solve : Solve a linear system of equations.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[2., 1.],
        ...                [1., 2.]]) * u.meter2
        >>> L = u.linalg.cholesky(x)
        >>> L.unit
        Unit("m")
        >>> u.math.allclose(x, L @ L.T)
        Array(True, dtype=bool)
    """
    return _fun_change_unit_unary(jnp.linalg.cholesky,
                                  lambda u: u ** 0.5,
                                  a,
                                  upper=upper,
                                  symmetrize_input=symmetrize_input)


@unit_change(lambda x, y: y / x)
@set_module_as('saiunit.linalg')
def solve(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
) -> Union[jax.typing.ArrayLike, Quantity]:
    """
    Solve a linear system of equations.

    SaiUnit implementation of :func:`numpy.linalg.solve`.

    This solves a (batched) linear system of equations ``a @ x = b``
    for ``x`` given ``a`` and ``b``. The resulting unit is
    ``b.unit / a.unit``.

    Parameters
    ----------
    a : array_like or Quantity
        Coefficient matrix of shape ``(..., N, N)``.
    b : array_like or Quantity
        Right-hand side of shape ``(N,)`` (for a 1-dimensional right-hand
        side) or ``(..., N, M)`` (for batched 2-dimensional right-hand side).

    Returns
    -------
    x : ndarray or Quantity
        Solution array. The result has shape ``(..., N)`` if ``b`` is of
        shape ``(N,)``, and shape ``(..., N, M)`` otherwise. The resulting
        unit is ``b.unit / a.unit``.

    See Also
    --------
    saiunit.linalg.inv : Compute the inverse of a matrix.
    saiunit.linalg.lstsq : Least-squares solution to a linear equation.
    saiunit.linalg.tensorsolve : Solve the tensor equation ``a x = b``.

    Notes
    -----
    Prefer ``solve`` over explicitly computing ``inv(a) @ b`` for better
    numerical precision and performance.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> A = jnp.array([[1., 2., 3.],
        ...                [2., 4., 2.],
        ...                [3., 2., 1.]]) * u.meter
        >>> b = jnp.array([14., 16., 10.]) * u.second
        >>> x = u.linalg.solve(A, b)
        >>> x.unit
        Unit("s / m")
        >>> u.math.allclose(A @ x, b)
        Array(True, dtype=bool)
    """
    return _fun_change_unit_binary(jnp.linalg.solve,
                                   lambda a, b: b / a,
                                   a,
                                   b)


@unit_change(lambda x, y: y / x)
@set_module_as('saiunit.linalg')
def tensorsolve(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
    axes: tuple[int, ...] | None = None
) -> Union[jax.typing.ArrayLike, Quantity]:
    """
    Solve the tensor equation ``a x = b`` for ``x``.

    SaiUnit implementation of :func:`numpy.linalg.tensorsolve`.

    The resulting unit is ``b.unit / a.unit``.

    Parameters
    ----------
    a : array_like or Quantity
        Coefficient tensor. After reordering via ``axes`` (see below),
        shape must be ``(*b.shape, *x.shape)``.
    b : array_like or Quantity
        Right-hand side tensor.
    axes : tuple of int or None, optional
        Axes of ``a`` that should be moved to the end before solving.

    Returns
    -------
    x : ndarray or Quantity
        Solution ``x`` such that after reordering of axes of ``a``,
        ``tensordot(a, x, x.ndim)`` is equivalent to ``b``. The
        resulting unit is ``b.unit / a.unit``.

    See Also
    --------
    saiunit.linalg.solve : Solve a linear system of equations.
    saiunit.linalg.tensorinv : Compute the tensor inverse.
    saiunit.linalg.tensordot : Compute tensor dot product.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax
        >>> key1, key2 = jax.random.split(jax.random.key(8675309))
        >>> a = jax.random.normal(key1, shape=(2, 2, 4)) * u.meter
        >>> b = jax.random.normal(key2, shape=(2, 2)) * u.second
        >>> x = u.linalg.tensorsolve(a, b)
        >>> x.shape
        (4,)
        >>> b_reconstructed = u.linalg.tensordot(a, x, axes=x.ndim)
        >>> u.math.allclose(b, b_reconstructed)
        Array(True, dtype=bool)
    """
    return _fun_change_unit_binary(jnp.linalg.tensorsolve,
                                   lambda a, b: b / a,
                                   a,
                                   b,
                                   axes=axes)


@unit_change(lambda x, y: y / x)
@set_module_as('saiunit.linalg')
def lstsq(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
    rcond: float | None = None,
    *,
    numpy_resid: bool = False
) -> tuple[Union[jax.typing.ArrayLike, Quantity], jax.Array, jax.Array, jax.Array]:
    """
    Return the least-squares solution to a linear equation.

    SaiUnit implementation of :func:`numpy.linalg.lstsq`.

    Finds ``x`` that minimizes ``||a @ x - b||``. The resulting unit of
    ``x`` is ``b.unit / a.unit``.

    Parameters
    ----------
    a : array_like or Quantity
        Coefficient matrix of shape ``(M, N)``.
    b : array_like or Quantity
        Right-hand side of shape ``(M,)`` or ``(M, K)``.
    rcond : float or None, optional
        Cut-off ratio for small singular values. Singular values smaller
        than ``rcond * largest_singular_value`` are treated as zero. If
        ``None`` (default), the optimal value is used to reduce floating
        point errors.
    numpy_resid : bool, optional
        If ``True``, compute and return residuals in the same way as
        NumPy's ``linalg.lstsq``. If ``False`` (default), a more
        efficient method is used to compute residuals.

    Returns
    -------
    x : ndarray or Quantity
        Least-squares solution of shape ``(N,)`` or ``(N, K)``. The
        resulting unit is ``b.unit / a.unit``.
    residuals : ndarray
        Sum of squared residuals of shape ``()`` or ``(K,)``.
    rank : ndarray
        Effective rank of ``a``.
    s : ndarray
        Singular values of ``a``.

    See Also
    --------
    saiunit.linalg.solve : Solve a square linear system exactly.
    saiunit.linalg.pinv : Compute the Moore-Penrose pseudo-inverse.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[1, 2],
        ...                [3, 4]]) * u.second
        >>> b = jnp.array([5, 6]) * u.meter
        >>> x, residuals, rank, s = u.linalg.lstsq(a, b)
        >>> x.unit
        Unit("m / s")
    """
    a = maybe_custom_array(a)
    b = maybe_custom_array(b)
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
@set_module_as('saiunit.linalg')
def inv(
    a: Union[jax.typing.ArrayLike, Quantity],
) -> Union[jax.typing.ArrayLike, Quantity]:
    """
    Return the inverse of a square matrix.

    SaiUnit implementation of :func:`numpy.linalg.inv`.

    The resulting unit is ``a.unit ** -1``.

    Parameters
    ----------
    a : array_like or Quantity
        Square input of shape ``(..., N, N)`` specifying square matrix(es)
        to be inverted.

    Returns
    -------
    out : ndarray or Quantity
        Inverse matrix of shape ``(..., N, N)``. The resulting unit is
        ``a.unit ** -1``.

    See Also
    --------
    saiunit.linalg.solve : Solve a linear system (preferred over explicit inverse).
    saiunit.linalg.pinv : Compute the Moore-Penrose pseudo-inverse.

    Notes
    -----
    In most cases, explicitly computing the inverse of a matrix is
    ill-advised. For example, to compute ``x = inv(A) @ b``, it is more
    performant and numerically precise to use a direct solve, such as
    :func:`saiunit.linalg.solve`.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[1., 2., 3.],
        ...                [2., 4., 2.],
        ...                [3., 2., 1.]]) * u.second
        >>> a_inv = u.linalg.inv(a)
        >>> u.math.allclose(a @ a_inv, jnp.eye(3), atol=1e-5)
        Array(True, dtype=bool)
    """
    return _fun_change_unit_unary(jnp.linalg.inv,
                                  lambda u: u ** -1,
                                  a)


@unit_change(lambda u: u ** -1)
@set_module_as('saiunit.linalg')
def pinv(
    a: Union[jax.typing.ArrayLike, Quantity],
    rtol: jax.typing.ArrayLike | None = None,
    hermitian: bool = False,
    *,
    rcond: jax.typing.ArrayLike | None = None,
) -> Union[jax.typing.ArrayLike, Quantity]:
    """
    Compute the Moore-Penrose pseudo-inverse of a matrix.

    SaiUnit implementation of :func:`numpy.linalg.pinv`.

    The resulting unit is ``a.unit ** -1``.

    Parameters
    ----------
    a : array_like or Quantity
        Input matrix of shape ``(..., M, N)`` to pseudo-invert.
    rtol : float or array_like, optional
        Cutoff for small singular values of shape ``a.shape[:-2]``.
        Singular values smaller than ``rtol * largest_singular_value``
        are treated as zero. The default is determined based on the
        floating point precision of the dtype.
    hermitian : bool, optional
        If ``True``, the input is assumed to be Hermitian, and a more
        efficient algorithm is used (default: ``False``).
    rcond : float or None, optional
        Deprecated alias for ``rtol``. Will result in a
        :class:`DeprecationWarning` if used.

    Returns
    -------
    out : ndarray or Quantity
        Pseudo-inverse of shape ``(..., N, M)``. The resulting unit is
        ``a.unit ** -1``.

    See Also
    --------
    saiunit.linalg.inv : Compute the inverse of a square matrix.
    saiunit.linalg.lstsq : Least-squares solution to a linear equation.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[1, 2],
        ...                [3, 4],
        ...                [5, 6]]) * u.second
        >>> a_pinv = u.linalg.pinv(a)
        >>> a_pinv.shape
        (2, 3)
        >>> u.math.allclose(a_pinv @ a, jnp.eye(2), atol=1e-4)
        Array(True, dtype=bool)
    """
    return _fun_change_unit_unary(jnp.linalg.pinv,
                                  lambda u: u ** -1,
                                  a,
                                  rtol=rtol,
                                  hermitian=hermitian,
                                  rcond=rcond)


@unit_change(lambda u: u ** -1)
@set_module_as('saiunit.linalg')
def tensorinv(
    a: Union[jax.typing.ArrayLike, Quantity],
    ind: int = 2,
) -> Union[jax.typing.ArrayLike, Quantity]:
    """
    Compute the tensor inverse of an array.

    SaiUnit implementation of :func:`numpy.linalg.tensorinv`.

    This computes the inverse of the :func:`~saiunit.linalg.tensordot`
    operation with the same ``ind`` value. The resulting unit is
    ``a.unit ** -1``.

    Parameters
    ----------
    a : array_like or Quantity
        Quantity to be inverted. Must satisfy
        ``prod(a.shape[:ind]) == prod(a.shape[ind:])``.
    ind : int, optional
        Positive integer specifying the number of indices in the tensor
        product (default: 2).

    Returns
    -------
    out : ndarray or Quantity
        Tensor inverse of shape ``(*a.shape[ind:], *a.shape[:ind])``.
        The resulting unit is ``a.unit ** -1``.

    See Also
    --------
    saiunit.linalg.tensorsolve : Solve the tensor equation ``a x = b``.
    saiunit.linalg.tensordot : Compute tensor dot product.
    saiunit.linalg.inv : Compute the inverse of a square matrix.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.key(1337)
        >>> x = jax.random.normal(key, shape=(2, 2, 4)) * u.second
        >>> xinv = u.linalg.tensorinv(x, 2)
        >>> xinv.shape
        (4, 2, 2)
        >>> xinv_x = u.linalg.tensordot(xinv, x, axes=2)
        >>> u.math.allclose(xinv_x, jnp.eye(4), atol=1e-4)
        Array(True, dtype=bool)
    """
    return _fun_change_unit_unary(jnp.linalg.tensorinv,
                                  lambda u: u ** -1,
                                  a,
                                  ind=ind)
