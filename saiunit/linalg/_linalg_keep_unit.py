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

from typing import (Union)

import jax
import jax.numpy as jnp
from jax import Array

from .._base import Quantity, maybe_decimal
from .._misc import set_module_as
from ..lax import _lax_linalg as lax_linalg
from ..math._fun_keep_unit import (
    _fun_keep_unit_unary, trace, diagonal
)

__all__ = [
    # Decompositions
    'qr', 'svd', 'svdvals',
    # Matrix eigenvalues
    'eig', 'eigh', 'eigvals', 'eigvalsh',
    # Norms and other numbers
    'norm', 'matrix_norm', 'vector_norm',
    'trace',
    # Other matrix operations
    'diagonal', 'matrix_transpose',
]


@set_module_as('saiunit.linalg')
def norm(
    x: Union[jax.typing.ArrayLike, Quantity],
    ord: int | str | None = None,
    axis: None | tuple[int, ...] | int = None,
    keepdims: bool = False,
) -> Union[jax.Array, Quantity]:
    """Compute the norm of a matrix or vector.

    Args:
        x: N-dimensional quantity for which the norm will be computed.
        ord: specify the kind of norm to take. Default is Frobenius norm for matrices,
            and the 2-norm for vectors. For other options, see Notes below.
        axis: integer or sequence of integers specifying the axes over which the norm
            will be computed. Defaults to all axes of ``x``.
        keepdims: if True, the output array will have the same number of dimensions as
            the input, with the size of reduced axes replaced by ``1`` (default: False).

    Returns:
        quantity containing the specified norm of x.

    Notes:
    The flavor of norm computed depends on the value of ``ord`` and the number of
    axes being reduced.

    For **vector norms** (i.e. a single axis reduction):

    - ``ord=None`` (default) computes the 2-norm
    - ``ord=inf`` computes ``max(abs(x))``
    - ``ord=-inf`` computes min(abs(x))``
    - ``ord=0`` computes ``sum(x!=0)``
    - for other numerical values, computes ``sum(abs(x) ** ord)**(1/ord)``

    For **matrix norms** (i.e. two axes reductions):

    - ``ord='fro'`` or ``ord=None`` (default) computes the Frobenius norm
    - ``ord='nuc'`` computes the nuclear norm, or the sum of the singular values
    - ``ord=1`` computes ``max(abs(x).sum(0))``
    - ``ord=-1`` computes ``min(abs(x).sum(0))``
    - ``ord=2`` computes the 2-norm, i.e. the largest singular value
    - ``ord=-2`` computes the smallest singular value

    Examples:
        >>> import saiunit as u
        >>> import jax.numpy as jnp

        Vector norms:

        >>> x = jnp.array([3., 4., 12.]) * u.meter
        >>> u.linalg.norm(x)
        13. * meter
        >>> u.linalg.norm(x, ord=1)
        19. * meter
        >>> u.linalg.norm(x, ord=0)
        3. * meter

        Matrix norms:

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1., 2., 3.],
        ...                [4., 5., 7.]]) * u.meter
        >>> u.linalg.norm(x)  # Frobenius norm
        10.198039 * meter
        >>> u.linalg.norm(x, ord='nuc')  # nuclear norm
        10.762534 * meter
        >>> u.linalg.norm(x, ord=1)  # 1-norm
        10. * meter

        Batched vector norm:

        >>> u.linalg.norm(x, axis=1)
        ArrayImpl([3.7416575, 9.48683262], dtype=float32) * meter
    """
    return _fun_keep_unit_unary(jnp.linalg.norm, x, ord=ord, axis=axis, keepdims=keepdims)


@set_module_as('saiunit.linalg')
def matrix_norm(
    x: Union[jax.typing.ArrayLike, Quantity],
    *,
    keepdims: bool = False,
    ord: str = 'fro'
) -> Union[jax.Array, Quantity]:
    """Compute the norm of a matrix or stack of matrices.

    SaiUnit implementation of :func:`numpy.linalg.matrix_norm`

    Args:
        x: quantity of shape ``(..., M, N)`` for which to take the norm.
        keepdims: if True, keep the reduced dimensions in the output.
        ord: A string or int specifying the type of norm; default is the Frobenius norm.
            See :func:`numpy.linalg.norm` for details on available options.

    Returns:
        quantity containing the norm of ``x``. Has shape ``x.shape[:-2]`` if ``keepdims`` is
        False, or shape ``(..., 1, 1)`` if ``keepdims`` is True.

    Examples:

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1, 2, 3],
        ...                [4, 5, 6],
        ...                [7, 8, 9]]) * u.second
        >>> u.linalg.matrix_norm(x)
        16.881943 * second
    """
    return _fun_keep_unit_unary(jnp.linalg.norm,
                                x,
                                keepdims=keepdims,
                                ord=ord)


@set_module_as('saiunit.linalg')
def vector_norm(
    x: Union[jax.typing.ArrayLike, Quantity],
    *, axis: int | None = None,
    keepdims: bool = False,
    ord: int | str = 2
) -> Union[jax.Array, Quantity]:
    """Compute the vector norm of a vector or batch of vectors.

    SaiUnit implementation of :func:`numpy.linalg.vector_norm`.

    Args:
        x: N-dimensional quantity for which to take the norm.
        axis: optional axis along which to compute the vector norm. If None (default)
            then ``x`` is flattened and the norm is taken over all values.
        keepdims: if True, keep the reduced dimensions in the output.
        ord: A string or int specifying the type of norm; default is the 2-norm.
            See :func:`numpy.linalg.norm` for details on available options.

    Returns:
        quantity containing the norm of ``x``.

    Examples:
        >>> import saiunit as u
        >>> import jax.numpy as jnp

        Norm of a single vector:

        >>> x = jnp.array([1., 2., 3.]) * u.meter
        >>> u.linalg.vector_norm(x)
        3.7416575 * meter

        Norm of a batch of vectors:

        >>> x = jnp.array([[1., 2., 3.],
        ...                [4., 5., 7.]]) * u.meter
        >>> u.linalg.vector_norm(x, axis=1)
        ArrayImpl([3.7416575, 9.48683262], dtype=float32) * meter
    """
    return _fun_keep_unit_unary(jnp.linalg.norm,
                                x,
                                axis=axis,
                                keepdims=keepdims,
                                ord=ord)


@set_module_as('saiunit.linalg')
def qr(
    a: Union[Quantity, jax.typing.ArrayLike],
    mode: str = "reduced"
) -> Array | Quantity:
    """Compute the QR decomposition of a quantity

    SaiUnit implementation of :func:`numpy.linalg.qr`.

    The QR decomposition of a matrix `A` is given by

    .. math::

        A = QR

    Where `Q` is a unitary matrix (i.e. :math:`Q^HQ=I`) and `R` is an upper-triangular
    matrix.

    Args:
        a: quantity of shape (..., M, N)
        mode: Computational mode. Supported values are:

        - ``"reduced"`` (default): return `Q` of shape ``(..., M, K)`` and `R` of shape
          ``(..., K, N)``, where ``K = min(M, N)``.
        - ``"complete"``: return `Q` of shape ``(..., M, M)`` and `R` of shape ``(..., M, N)``.
        - ``"raw"``: return lapack-internal representations of shape ``(..., M, N)`` and ``(..., K)``.
        - ``"r"``: return `R` only.

    Returns:
        A tuple ``(Q, R)`` (if ``mode`` is not ``"r"``) otherwise an quantity ``R``,
        where:

        - ``Q`` is an orthogonal matrix of shape ``(..., M, K)`` (if ``mode`` is ``"reduced"``)
          or ``(..., M, M)`` (if ``mode`` is ``"complete"``).
        - ``R`` is an upper-triangular matrix of shape ``(..., M, N)`` (if ``mode`` is
          ``"r"`` or ``"complete"``) or ``(..., K, N)`` (if ``mode`` is ``"reduced"``)

        with ``K = min(M, N)``.

    Examples:
        >>> import saiunit as u
        >>> import jax.numpy as jnp

        Compute the QR decomposition of a matrix:

        >>> a = jnp.array([[1., 2., 3., 4.],
        ...                [5., 4., 2., 1.],
        ...                [6., 3., 1., 5.]]) * u.meter
        >>> Q, R = u.linalg.qr(a)
        >>> Q  # doctest: +SKIP
        Array([[-0.12700021, -0.7581426 , -0.6396022 ],
               [-0.63500065, -0.43322435,  0.63960224],
               [-0.7620008 ,  0.48737738, -0.42640156]], dtype=float32)
        >>> R  # doctest: +SKIP
        ArrayImpl([[-7.8740077, -5.08000517, -2.41300249, -4.95300531],
                   [ 0.       , -1.78704989, -2.65349889, -1.02890778],
                   [ 0.       ,  0.       , -1.06600308, -4.05081367]],
                  dtype=float32) * meter

        Check that ``Q`` is orthonormal:

        >>> u.math.allclose(Q.T @ Q, jnp.eye(3), atol=1E-5)
        Array(True, dtype=bool)

        Reconstruct the input:

        >>> u.math.allclose(Q @ R, a)
        Array(True, dtype=bool)
    """
    if isinstance(a, Quantity):
        result = jnp.linalg.qr(a.mantissa, mode=mode)
        if mode == "r":
            return maybe_decimal(Quantity(result, unit=a.unit))
        else:
            Q, R = result
            return Q, maybe_decimal(Quantity(R, unit=a.unit))
    else:
        result = jnp.linalg.qr(a, mode=mode)
        if mode == "r":
            return result
        else:
            return result


svd = lax_linalg.svd


@set_module_as('saiunit.linalg')
def svdvals(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> Union[jax.Array, Quantity]:
    """Compute the singular values of a matrix.

    SaiUnit implementation of :func:`numpy.linalg.svdvals`.

    Args:
        x: quantity of shape ``(..., M, N)`` for which singular values will be computed.

    Returns:
        quantity of singular values of shape ``(..., K)`` with ``K = min(M, N)``.

    Examples:
        >>> import saiunit as u
        >>> import jax.numpy as jnp

        >>> x = jnp.array([[1., 2., 3.],
        ...                [4., 5., 6.]]) * u.meter
        >>> u.linalg.svdvals(x)
        ArrayImpl([9.50803089, 0.77286941], dtype=float32) * meter
    """
    return svd(x, compute_uv=False)


@set_module_as('saiunit.linalg')
def eig(
    a: Union[Quantity, jax.typing.ArrayLike],
) -> tuple[Union[jax.Array, Quantity], Union[jax.Array, Quantity]]:
    """
    Compute the eigenvalues and eigenvectors of a square quantity.

    SaiUnit implementation of :func:`numpy.linalg.eig`.

    Args:
        a: quantity of shape ``(..., M, M)`` for which to compute the eigenvalues and vectors.

    Returns:
        A tuple ``(eigenvalues, eigenvectors)`` with

        - ``eigenvalues``: a quantity of shape ``(..., M)`` containing the eigenvalues.
        - ``eigenvectors``: an quantity of shape ``(..., M, M)``, where column ``v[:, i]`` is the
          eigenvector corresponding to the eigenvalue ``w[i]``.

    Examples:
        >>> import saiunit as u
        >>> import jax.numpy as jnp

        >>> a = jnp.array([[1., 2.],
        ...                [2., 1.]]) * u.meter
        >>> w, v = u.linalg.eig(a)
        >>> with jax.numpy.printoptions(precision=4):
        ...   print(w)
        ArrayImpl([ 3.+0.j, -1.+0.j], dtype=complex64) * meter
        >>> v
        Array([[ 0.70710677+0.j, -0.70710677+0.j],
               [ 0.70710677+0.j,  0.70710677+0.j]], dtype=complex64)
    """
    return lax_linalg.eig(a, compute_left_eigenvectors=False)


@set_module_as('saiunit.linalg')
def eigh(
    a: Union[Quantity, jax.typing.ArrayLike],
    UPLO: str | None = None,
    symmetrize_input: bool = True
) -> tuple[Union[jax.Array, Quantity], Union[jax.Array, Quantity]]:
    """
    Compute the eigenvalues and eigenvectors of a Hermitian matrix.

    SaiUnit implementation of :func:`numpy.linalg.eigh`.

    Args:
        a: quantity of shape ``(..., M, M)``, containing the Hermitian (if complex)
            or symmetric (if real) matrix.
        UPLO: specifies whether the calculation is done with the lower triangular
            part of ``a`` (``'L'``, default) or the upper triangular part (``'U'``).
        symmetrize_input: if True (default) then input is symmetrized, which leads
            to better behavior under automatic differentiation.

    Returns:
        A namedtuple ``(eigenvalues, eigenvectors)`` where

        - ``eigenvalues``: a quantity of shape ``(..., M)`` containing the eigenvalues,
            sorted in ascending order.
        - ``eigenvectors``: a quantity of shape ``(..., M, M)``, where column ``v[:, i]`` is the
            normalized eigenvector corresponding to the eigenvalue ``w[i]``.

    Examples:
        >>> import saiunit as u
        >>> import jax.numpy as jnp

        >>> a = jnp.array([[1, -2j],
        ...                [2j, 1]]) * u.meter
        >>> w, v = u.linalg.eigh(a)
        >>> w
        Array([-1.,  3.], dtype=float32)
        >>> with jnp.printoptions(precision=3):
        ...   print(v)
        ArrayImpl([[-0.707-0.j   , -0.707+0.j   ],
                   [ 0.   +0.707j,  0.   -0.707j]], dtype=complex64)
    """
    if UPLO is None or UPLO == "L":
        lower = True
    elif UPLO == "U":
        lower = False
    else:
        msg = f"UPLO must be one of None, 'L', or 'U', got {UPLO}"
        raise ValueError(msg)
    return lax_linalg.eigh(a, lower=lower, symmetrize_input=symmetrize_input)


@set_module_as('saiunit.linalg')
def eigvals(
    a: Union[Quantity, jax.typing.ArrayLike],
) -> Union[jax.Array, Quantity]:
    """
    Compute the eigenvalues of a general matrix.

    SaiUnit implementation of :func:`numpy.linalg.eigvals`.

    Args:
        a: quantity of shape ``(..., M, M)`` for which to compute the eigenvalues.

    Returns:
        An quantity of shape ``(..., M)`` containing the eigenvalues.

    Examples:
        >>> import saiunit as u
        >>> import jax.numpy as jnp

        >>> a = jnp.array([[1., 2.],
        ...                [2., 1.]]) * u.meter
        >>> w = u.linalg.eigvals(a)
        >>> with jnp.printoptions(precision=2):
        ...  print(w)
        ArrayImpl([ 3.+0.j, -1.+0.j], dtype=complex64) * meter
    """
    return eig(a)[0]


@set_module_as('saiunit.linalg')
def eigvalsh(
    a: Union[Quantity, jax.typing.ArrayLike],
    UPLO: str = 'L',
) -> Union[jax.Array, Quantity]:
    """
    Compute the eigenvalues of a Hermitian matrix.

    SaiUnit implementation of :func:`numpy.linalg.eigvalsh`.

    Args:
        a: quantity of shape ``(..., M, M)``, containing the Hermitian (if complex)
            or symmetric (if real) matrix.
        UPLO: specifies whether the calculation is done with the lower triangular
            part of ``a`` (``'L'``, default) or the upper triangular part (``'U'``).

    Returns:
        A quantity of shape ``(..., M)`` containing the eigenvalues, sorted in
        ascending order.

    Examples:
        >>> import saiunit as u
        >>> import jax.numpy as jnp

        >>> a = jnp.array([[1, -2j],
        ...                [2j, 1]]) * u.meter
        >>> w = u.linalg.eigvalsh(a)
        >>> w
        Array([-1.,  3.], dtype=float32)
    """
    return eigh(a, UPLO=UPLO)[0]


@set_module_as('saiunit.linalg')
def matrix_transpose(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Transpose a matrix or stack of matrices.

    SaiUnit implementation of :func:`numpy.linalg.matrix_transpose`.

    Args:
        x: quantity of shape ``(..., M, N)``

    Returns:
        quantity of shape ``(..., N, M)`` containing the matrix transpose of ``x``.

    Examples:
        Transpose of a single matrix:
        >>> import saiunit as u
        >>> import jax.numpy as jnp

        >>> x = jnp.array([[1, 2, 3],
        ...                [4, 5, 6]]) * u.meter
        >>> u.linalg.matrix_transpose(x)
        ArrayImpl([[1, 4],
                   [2, 5],
                   [3, 6]], dtype=int32) * meter

        Transpose of a stack of matrices:

        >>> x = jnp.array([[[1, 2],
        ...                 [3, 4]],
        ...                [[5, 6],
        ...                 [7, 8]]]) * u.meter
        >>> u.linalg.matrix_transpose(x)
        ArrayImpl([[[1, 3],
                    [2, 4]],
        <BLANKLINE>
                   [[5, 7],
                    [6, 8]]], dtype=int32) * meter

        For convenience, the same computation can be done via the
        :attr:`~jax.Array.mT` property of SaiUnit quantity objects:

        >>> x.mT
        Array([[[1, 3],
                [2, 4]],
        <BLANKLINE>
               [[5, 7],
                [6, 8]]], dtype=int32)
    """
    return _fun_keep_unit_unary(jnp.linalg.matrix_transpose, x)
