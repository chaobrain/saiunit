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

from saiunit._jax_compat import HAS_JAX, jax, jnp, Array, require_jax, ArrayLike

from saiunit._backend import get_backend
from saiunit._base_getters import maybe_decimal
from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as, maybe_custom_array
from saiunit.math._fun_keep_unit import (
    _fun_keep_unit_unary, trace, diagonal
)


def _lax_linalg():
    require_jax("saiunit.linalg.svd / eig / eigh")
    from saiunit.lax import _lax_linalg as _m
    return _m

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
    x: Union[ArrayLike, Quantity],
    ord: int | str | None = None,
    axis: None | tuple[int, ...] | int = None,
    keepdims: bool = False,
    **kwargs,
) -> Union[jax.Array, Quantity]:
    """Compute the norm of a matrix or vector.

    Computes a variety of vector and matrix norms, preserving the physical
    unit of the input.

    Parameters
    ----------
    x : array_like or Quantity
        N-dimensional input for which the norm will be computed.
    ord : {int, float, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm.  Default is Frobenius norm for matrices and
        the 2-norm for vectors.  See *Notes* for details.
    axis : None or int or tuple of int, optional
        Axes over which the norm is computed.  Defaults to all axes.
    keepdims : bool, optional
        If ``True``, reduced axes are kept as size-1 dimensions
        (default: ``False``).

    Returns
    -------
    out : ndarray or Quantity
        Norm of *x*.  Carries the same unit as *x*.

    Notes
    -----
    The flavor of norm computed depends on the value of *ord* and the
    number of axes being reduced.

    For **vector norms** (single-axis reduction):

    * ``ord=None`` (default) -- 2-norm
    * ``ord=inf`` -- ``max(abs(x))``
    * ``ord=-inf`` -- ``min(abs(x))``
    * ``ord=0`` -- ``sum(x != 0)``
    * other numeric -- ``sum(abs(x)**ord)**(1/ord)``

    For **matrix norms** (two-axis reduction):

    * ``ord='fro'`` or ``None`` (default) -- Frobenius norm
    * ``ord='nuc'`` -- nuclear norm (sum of singular values)
    * ``ord=1`` -- ``max(abs(x).sum(axis=0))``
    * ``ord=-1`` -- ``min(abs(x).sum(axis=0))``
    * ``ord=2`` -- largest singular value
    * ``ord=-2`` -- smallest singular value

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp

        Vector 2-norm:

        >>> x = jnp.array([3., 4., 12.]) * u.meter
        >>> u.linalg.norm(x)
        13. * meter

        L1 vector norm:

        >>> u.linalg.norm(x, ord=1)
        19. * meter

        Frobenius matrix norm:

        >>> m = jnp.array([[1., 2., 3.],
        ...                [4., 5., 7.]]) * u.meter
        >>> u.linalg.norm(m)
        10.198039 * meter
    """
    return _fun_keep_unit_unary('linalg.norm', x, ord=ord, axis=axis, keepdims=keepdims, **kwargs)


@set_module_as('saiunit.linalg')
def matrix_norm(
    x: Union[ArrayLike, Quantity],
    *,
    keepdims: bool = False,
    ord: int | str = 'fro',
    **kwargs,
) -> Union[jax.Array, Quantity]:
    """Compute the norm of a matrix or stack of matrices.

    SaiUnit implementation of :func:`numpy.linalg.matrix_norm`.

    Parameters
    ----------
    x : array_like or Quantity
        Input of shape ``(..., M, N)``.
    keepdims : bool, optional
        If ``True``, reduced axes are kept as size-1 dimensions
        (default: ``False``).
    ord : {int, float, inf, -inf, 'fro', 'nuc'}, optional
        Type of matrix norm (default: ``'fro'``).  See
        :func:`numpy.linalg.norm` for available options.

    Returns
    -------
    out : ndarray or Quantity
        Matrix norm with shape ``x.shape[:-2]`` (or ``(..., 1, 1)`` when
        *keepdims* is ``True``).  Carries the same unit as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1, 2, 3],
        ...                [4, 5, 6],
        ...                [7, 8, 9]]) * u.second
        >>> u.linalg.matrix_norm(x)
        16.881943 * second
    """
    return _fun_keep_unit_unary('linalg.matrix_norm',
                                x,
                                keepdims=keepdims,
                                ord=ord, **kwargs)


@set_module_as('saiunit.linalg')
def vector_norm(
    x: Union[ArrayLike, Quantity],
    *, axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: int | str = 2,
    **kwargs,
) -> Union[jax.Array, Quantity]:
    """Compute the vector norm of a vector or batch of vectors.

    SaiUnit implementation of :func:`numpy.linalg.vector_norm`.

    Parameters
    ----------
    x : array_like or Quantity
        N-dimensional input.
    axis : int or tuple of int, optional
        Axis (or axes) along which to compute the norm.  If ``None``
        (default), *x* is flattened first.
    keepdims : bool, optional
        If ``True``, reduced axes are kept as size-1 dimensions
        (default: ``False``).
    ord : int or str, optional
        Type of norm (default: 2).  See :func:`numpy.linalg.norm` for
        available options.

    Returns
    -------
    out : ndarray or Quantity
        Vector norm of *x*.  Carries the same unit as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp

        Norm of a single vector:

        >>> x = jnp.array([1., 2., 3.]) * u.meter
        >>> u.linalg.vector_norm(x)
        3.7416575 * meter

        Batch of vectors along axis 1:

        >>> x = jnp.array([[1., 2., 3.],
        ...                [4., 5., 7.]]) * u.meter
        >>> u.linalg.vector_norm(x, axis=1)
        ArrayImpl([3.7416575, 9.48683262], dtype=float32) * meter
    """
    return _fun_keep_unit_unary('linalg.vector_norm',
                                x,
                                axis=axis,
                                keepdims=keepdims,
                                ord=ord, **kwargs)


@set_module_as('saiunit.linalg')
def qr(
    a: Union[Quantity, ArrayLike],
    mode: str = "reduced",
    **kwargs,
) -> Array | Quantity:
    """Compute the QR decomposition of a matrix.

    SaiUnit implementation of :func:`numpy.linalg.qr`.

    The QR decomposition of a matrix *A* is:

    .. math::

        A = QR

    where *Q* is a unitary matrix (:math:`Q^HQ=I`) and *R* is
    upper-triangular.  The unit of *A* is carried by *R*; *Q* is
    always dimensionless.

    Parameters
    ----------
    a : array_like or Quantity
        Input of shape ``(..., M, N)``.
    mode : {'reduced', 'complete', 'raw', 'r'}, optional
        Computational mode (default: ``"reduced"``).

        * ``"reduced"`` -- *Q* has shape ``(..., M, K)``, *R* has shape
          ``(..., K, N)`` with ``K = min(M, N)``.
        * ``"complete"`` -- *Q* has shape ``(..., M, M)``, *R* has shape
          ``(..., M, N)``.
        * ``"r"`` -- only *R* is returned.

    Returns
    -------
    Q : ndarray
        Orthogonal factor (omitted when ``mode="r"``).
    R : ndarray or Quantity
        Upper-triangular factor carrying the unit of *a*.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[1., 2., 3., 4.],
        ...                [5., 4., 2., 1.],
        ...                [6., 3., 1., 5.]]) * u.meter
        >>> Q, R = u.linalg.qr(a)
        >>> Q.shape
        (3, 3)
        >>> R.unit
        meter
    """
    a = maybe_custom_array(a)
    mantissa = a.mantissa if isinstance(a, Quantity) else a
    xp = get_backend(mantissa)
    result = xp.linalg.qr(mantissa, mode=mode, **kwargs)
    if not isinstance(a, Quantity):
        return result  # type: ignore[return-value]
    if mode == "r":
        return maybe_decimal(Quantity(result, unit=a.unit))
    Q, R = result
    return Q, maybe_decimal(Quantity(R, unit=a.unit))  # type: ignore[return-value]


@set_module_as('saiunit.linalg')
def svd(
    x: Union[Quantity, ArrayLike],
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
    algorithm: jax.lax.linalg.SvdAlgorithm | None = None,
    **kwargs,
) -> Union[Quantity, ArrayLike] | tuple[jax.Array, Quantity | jax.Array, jax.Array]:
    """Singular value decomposition.

    SaiUnit implementation of :func:`numpy.linalg.svd`.

    Decomposes a matrix *A* into ``U @ diag(S) @ Vh``.  The singular
    values *S* carry the unit of *A*; *U* and *Vh* are dimensionless.

    Parameters
    ----------
    x : array_like or Quantity
        Input of shape ``(..., M, N)``.
    full_matrices : bool, optional
        If ``True`` (default), *U* and *Vh* have shapes ``(..., M, M)``
        and ``(..., N, N)``.  If ``False``, the shapes are
        ``(..., M, K)`` and ``(..., K, N)`` with ``K = min(M, N)``.
    compute_uv : bool, optional
        If ``True`` (default), return ``(U, S, Vh)``.  If ``False``,
        return only *S*.
    hermitian : bool, optional
        If ``True``, *x* is assumed to be Hermitian, enabling a more
        efficient algorithm.
    subset_by_index : tuple of int, optional
        Two-element tuple ``(start, end)`` selecting a subset of
        singular values.
    algorithm : SvdAlgorithm, optional
        SVD backend algorithm.

    Returns
    -------
    U : ndarray
        Left singular vectors (omitted when ``compute_uv=False``).
    S : ndarray or Quantity
        Singular values carrying the unit of *x*.
    Vh : ndarray
        Right singular vectors (omitted when ``compute_uv=False``).

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1., 2., 3.],
        ...                [4., 5., 6.]]) * u.meter
        >>> U, S, Vh = u.linalg.svd(x, full_matrices=False)
        >>> S
        ArrayImpl([9.50803089, 0.77286941], dtype=float32) * meter
    """
    x = maybe_custom_array(x)

    # `hermitian` is currently available on jnp.linalg.svd (not lax.linalg.svd).
    if hermitian:
        if algorithm is not None:
            raise TypeError('"algorithm" is not supported when "hermitian=True".')
        mantissa = x.mantissa if isinstance(x, Quantity) else x
        xp = get_backend(mantissa)
        svd_kwargs = dict(
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            hermitian=hermitian,
        )
        if xp is jnp and subset_by_index is not None:
            svd_kwargs['subset_by_index'] = subset_by_index
        elif subset_by_index is not None:
            raise TypeError(
                '"subset_by_index" is only supported on the jax backend.'
            )
        result = xp.linalg.svd(mantissa, **svd_kwargs, **kwargs)
        if not isinstance(x, Quantity):
            return result
        if compute_uv:
            u, s, vh = result
            return u, maybe_decimal(Quantity(s, unit=x.unit)), vh
        return maybe_decimal(Quantity(result, unit=x.unit))

    return _lax_linalg().svd(
        x,
        full_matrices=full_matrices,
        compute_uv=compute_uv,
        subset_by_index=subset_by_index,
        algorithm=algorithm,
        **kwargs,
    )


@set_module_as('saiunit.linalg')
def svdvals(
    x: Union[Quantity, ArrayLike],
    **kwargs,
) -> Union[jax.Array, Quantity]:
    """Compute the singular values of a matrix.

    SaiUnit implementation of :func:`numpy.linalg.svdvals`.

    Parameters
    ----------
    x : array_like or Quantity
        Input of shape ``(..., M, N)``.

    Returns
    -------
    out : ndarray or Quantity
        Singular values of shape ``(..., K)`` with ``K = min(M, N)``.
        Carries the same unit as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1., 2., 3.],
        ...                [4., 5., 6.]]) * u.meter
        >>> u.linalg.svdvals(x)
        ArrayImpl([9.50803089, 0.77286941], dtype=float32) * meter
    """
    return svd(x, compute_uv=False, **kwargs)


@set_module_as('saiunit.linalg')
def eig(
    a: Union[Quantity, ArrayLike],
    **kwargs,
) -> tuple[Union[jax.Array, Quantity], Union[jax.Array, Quantity]]:
    """Compute the eigenvalues and eigenvectors of a square matrix.

    SaiUnit implementation of :func:`numpy.linalg.eig`.

    The eigenvalues carry the same unit as *a*; the eigenvectors are
    dimensionless.

    Parameters
    ----------
    a : array_like or Quantity
        Square input of shape ``(..., M, M)``.

    Returns
    -------
    eigenvalues : ndarray or Quantity
        Shape ``(..., M)``.  Carries the same unit as *a*.
    eigenvectors : ndarray
        Shape ``(..., M, M)``.  Column ``v[:, i]`` corresponds to
        ``eigenvalues[i]``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[1., 2.],
        ...                [2., 1.]]) * u.meter
        >>> w, v = u.linalg.eig(a)
        >>> w  # eigenvalues carry the unit
        ArrayImpl([ 3.+0.j, -1.+0.j], dtype=complex64) * meter
    """
    a = maybe_custom_array(a)
    return _lax_linalg().eig(a, compute_left_eigenvectors=False, **kwargs)


@set_module_as('saiunit.linalg')
def eigh(
    a: Union[Quantity, ArrayLike],
    UPLO: str | None = None,
    symmetrize_input: bool = True,
    **kwargs,
) -> tuple[Union[jax.Array, Quantity], Union[jax.Array, Quantity]]:
    """Compute eigenvalues and eigenvectors of a Hermitian matrix.

    SaiUnit implementation of :func:`numpy.linalg.eigh`.

    Eigenvalues carry the same unit as *a*; eigenvectors are
    dimensionless.

    Parameters
    ----------
    a : array_like or Quantity
        Hermitian (or symmetric) input of shape ``(..., M, M)``.
    UPLO : {'L', 'U'}, optional
        Use the lower (``'L'``, default) or upper (``'U'``) triangle.
    symmetrize_input : bool, optional
        If ``True`` (default), symmetrise the input for better autodiff
        behaviour.

    Returns
    -------
    eigenvalues : ndarray or Quantity
        Shape ``(..., M)``, sorted ascending.  Same unit as *a*.
    eigenvectors : ndarray
        Shape ``(..., M, M)``.  Column ``v[:, i]`` is the eigenvector
        for ``eigenvalues[i]``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[1, -2j],
        ...                [2j, 1]]) * u.meter
        >>> w, v = u.linalg.eigh(a)
        >>> w
        Array([-1.,  3.], dtype=float32)
    """
    a = maybe_custom_array(a)
    if UPLO is None or UPLO == "L":
        lower = True
    elif UPLO == "U":
        lower = False
    else:
        msg = f"UPLO must be one of None, 'L', or 'U', got {UPLO}"
        raise ValueError(msg)
    v, w = _lax_linalg().eigh(a, lower=lower, symmetrize_input=symmetrize_input, **kwargs)
    return w, v


@set_module_as('saiunit.linalg')
def eigvals(
    a: Union[Quantity, ArrayLike],
    **kwargs,
) -> Union[jax.Array, Quantity]:
    """Compute the eigenvalues of a general matrix.

    SaiUnit implementation of :func:`numpy.linalg.eigvals`.

    Parameters
    ----------
    a : array_like or Quantity
        Square input of shape ``(..., M, M)``.

    Returns
    -------
    out : ndarray or Quantity
        Eigenvalues of shape ``(..., M)``.  Same unit as *a*.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[1., 2.],
        ...                [2., 1.]]) * u.meter
        >>> w = u.linalg.eigvals(a)
        >>> w  # eigenvalues carry the unit
        ArrayImpl([ 3.+0.j, -1.+0.j], dtype=complex64) * meter
    """
    return eig(a, **kwargs)[0]


@set_module_as('saiunit.linalg')
def eigvalsh(
    a: Union[Quantity, ArrayLike],
    UPLO: str = 'L',
    *,
    symmetrize_input: bool = True,
    **kwargs,
) -> Union[jax.Array, Quantity]:
    """Compute the eigenvalues of a Hermitian matrix.

    SaiUnit implementation of :func:`numpy.linalg.eigvalsh`.

    Parameters
    ----------
    a : array_like or Quantity
        Hermitian (or symmetric) input of shape ``(..., M, M)``.
    UPLO : {'L', 'U'}, optional
        Use the lower (``'L'``, default) or upper (``'U'``) triangle.
    symmetrize_input : bool, optional
        If ``True`` (default), symmetrise the input for better autodiff
        behaviour.

    Returns
    -------
    out : ndarray or Quantity
        Eigenvalues of shape ``(..., M)``, sorted ascending.  Same unit
        as *a*.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[1, -2j],
        ...                [2j, 1]]) * u.meter
        >>> w = u.linalg.eigvalsh(a)
        >>> w
        Array([-1.,  3.], dtype=float32)
    """
    return eigh(a, UPLO=UPLO, symmetrize_input=symmetrize_input, **kwargs)[0]


@set_module_as('saiunit.linalg')
def matrix_transpose(
    x: Union[Quantity, ArrayLike],
    **kwargs,
) -> Union[Quantity, ArrayLike]:
    """Transpose a matrix or stack of matrices.

    SaiUnit implementation of :func:`numpy.linalg.matrix_transpose`.

    Parameters
    ----------
    x : array_like or Quantity
        Input of shape ``(..., M, N)``.

    Returns
    -------
    out : ndarray or Quantity
        Transposed array of shape ``(..., N, M)``.  Carries the same
        unit as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1, 2, 3],
        ...                [4, 5, 6]]) * u.meter
        >>> u.linalg.matrix_transpose(x)
        ArrayImpl([[1, 4],
                   [2, 5],
                   [3, 6]], dtype=int32) * meter
    """
    return _fun_keep_unit_unary('linalg.matrix_transpose', x, **kwargs)
