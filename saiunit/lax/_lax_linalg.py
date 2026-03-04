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

from typing import Union, Callable, Any

import jax
from jax import lax, Array

from saiunit.lax._lax_change_unit import unit_change
from saiunit._base_getters import fail_for_unit_mismatch, maybe_decimal
from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as, maybe_custom_array, maybe_custom_array_tree
from saiunit.math._fun_change_unit import _fun_change_unit_unary

__all__ = [
    # linear algebra unary
    'cholesky', 'eig', 'eigh', 'hessenberg', 'lu',
    'qdwh', 'qr', 'schur', 'svd',
    'tridiagonal',

    # linear algebra binary
    'householder_product', 'triangular_solve',

    # linear algebra nary
    'tridiagonal_solve',
]


# linear algebra
@unit_change(lambda x: x ** 0.5)
def cholesky(
    x: Union[Quantity, jax.typing.ArrayLike],
    symmetrize_input: bool = True,
) -> Union[Quantity, jax.typing.ArrayLike]:
    r"""Cholesky decomposition.

    Compute the Cholesky decomposition :math:`A = L \cdot L^H` of square
    positive-definite matrices such that :math:`L` is lower triangular.
    The matrices must be Hermitian (if complex) or symmetric (if real).

    Parameters
    ----------
    x : array_like or Quantity
        A batch of square positive-definite matrices with shape
        ``[..., n, n]``.
    symmetrize_input : bool, optional
        If ``True``, the matrix is symmetrized before decomposition by
        computing :math:`\frac{1}{2}(x + x^H)`.  If ``False``, only the
        lower triangle of ``x`` is used. Default is ``True``.

    Returns
    -------
    L : jax.Array or Quantity
        The lower-triangular Cholesky factor with shape ``[..., n, n]``.
        If ``x`` carries a unit ``u``, the result has unit ``u ** 0.5``.
        If decomposition fails, the result is filled with NaNs.

    See Also
    --------
    saiunit.linalg.cholesky : Higher-level Cholesky wrapper.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> A = jnp.array([[4.0, 2.0], [2.0, 3.0]]) * (su.meter ** 2)
        >>> L = sulax.cholesky(A)
        >>> su.get_unit(L) == su.meter
        True
    """
    return _fun_change_unit_unary(lax.linalg.cholesky,
                                  lambda u: u ** 0.5,
                                  x,
                                  symmetrize_input=symmetrize_input)


@set_module_as('saiunit.lax')
def eig(
    x: Union[Quantity, jax.typing.ArrayLike],
    compute_left_eigenvectors: bool = True,
    compute_right_eigenvectors: bool = True
) -> tuple[Array | Quantity, Array, Array] | list[Array] | tuple[Array | Quantity, Array] | tuple[Array | Quantity]:
    """Eigendecomposition of a general matrix.

    Compute the eigenvalues and (optionally) left/right eigenvectors of a
    general square matrix.  Non-symmetric eigendecomposition is currently
    only implemented on CPU.

    Parameters
    ----------
    x : array_like or Quantity
        A batch of square matrices with shape ``[..., n, n]``.
    compute_left_eigenvectors : bool, optional
        If ``True``, compute the left eigenvectors. Default is ``True``.
    compute_right_eigenvectors : bool, optional
        If ``True``, compute the right eigenvectors. Default is ``True``.

    Returns
    -------
    w : jax.Array or Quantity
        The eigenvalues.  If ``x`` has a unit, ``w`` preserves that unit.
    vl : jax.Array, optional
        The left eigenvectors (unitless).  Only returned when
        ``compute_left_eigenvectors`` is ``True``.
    vr : jax.Array, optional
        The right eigenvectors (unitless).  Only returned when
        ``compute_right_eigenvectors`` is ``True``.

    Notes
    -----
    If the eigendecomposition fails, arrays full of NaNs are returned for
    that batch element.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> A = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * su.second
        >>> w, vl, vr = sulax.eig(A)
        >>> su.get_unit(w) == su.second
        True
    """
    x = maybe_custom_array_tree(x)
    if compute_left_eigenvectors and compute_right_eigenvectors:
        if isinstance(x, Quantity):
            w, vl, vr = lax.linalg.eig(x.mantissa, compute_left_eigenvectors=compute_left_eigenvectors,
                                       compute_right_eigenvectors=compute_right_eigenvectors)
            return maybe_decimal(Quantity(w, unit=x.unit)), vl, vr
        else:
            return lax.linalg.eig(x, compute_left_eigenvectors=compute_left_eigenvectors,
                                  compute_right_eigenvectors=compute_right_eigenvectors)
    elif compute_left_eigenvectors:
        if isinstance(x, Quantity):
            w, vl = lax.linalg.eig(x.mantissa, compute_left_eigenvectors=compute_left_eigenvectors,
                                   compute_right_eigenvectors=compute_right_eigenvectors)
            return maybe_decimal(Quantity(w, unit=x.unit)), vl
        else:
            return lax.linalg.eig(x, compute_left_eigenvectors=compute_left_eigenvectors,
                                  compute_right_eigenvectors=compute_right_eigenvectors)

    elif compute_right_eigenvectors:
        if isinstance(x, Quantity):
            w, vr = lax.linalg.eig(x.mantissa, compute_left_eigenvectors=compute_left_eigenvectors,
                                   compute_right_eigenvectors=compute_right_eigenvectors)
            return maybe_decimal(Quantity(w, unit=x.unit)), vr
        else:
            return lax.linalg.eig(x, compute_left_eigenvectors=compute_left_eigenvectors,
                                  compute_right_eigenvectors=compute_right_eigenvectors)
    else:
        if isinstance(x, Quantity):
            w = lax.linalg.eig(x.mantissa, compute_left_eigenvectors=compute_left_eigenvectors,
                               compute_right_eigenvectors=compute_right_eigenvectors)
            return (maybe_decimal(Quantity(w, unit=x.unit)),)
        else:
            return lax.linalg.eig(x, compute_left_eigenvectors=compute_left_eigenvectors,
                                  compute_right_eigenvectors=compute_right_eigenvectors)


@set_module_as('saiunit.lax')
def eigh(
    x: Union[Quantity, jax.typing.ArrayLike],
    lower: bool = True,
    symmetrize_input: bool = True,
    sort_eigenvalues: bool = True,
    subset_by_index: tuple[int, int] | None = None,
) -> tuple[Quantity | jax.Array, jax.Array]:
    r"""Eigendecomposition of a Hermitian matrix.

    Compute the eigenvectors and eigenvalues of a complex Hermitian or
    real symmetric square matrix.

    Parameters
    ----------
    x : array_like or Quantity
        A batch of square Hermitian (or real symmetric) matrices with shape
        ``[..., n, n]``.
    lower : bool, optional
        When ``symmetrize_input`` is ``False``, selects which triangle of the
        input to use.  Default is ``True``.
    symmetrize_input : bool, optional
        If ``True``, the matrix is symmetrized before the eigendecomposition
        by computing :math:`\frac{1}{2}(x + x^H)`.  Default is ``True``.
    sort_eigenvalues : bool, optional
        If ``True``, eigenvalues are sorted in ascending order.
        Default is ``True``.
    subset_by_index : tuple of int or None, optional
        A ``(start, end)`` pair selecting a range of eigenvalue indices to
        compute.  ``None`` means all eigenvalues.  Default is ``None``.

    Returns
    -------
    v : jax.Array
        Eigenvectors (unitless).  ``v[..., :, i]`` is the normalised
        eigenvector for eigenvalue ``w[..., i]``.
    w : jax.Array or Quantity
        Eigenvalues in ascending order.  If ``x`` has a unit, ``w``
        preserves that unit.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> A = jnp.array([[2.0, 1.0], [1.0, 3.0]]) * su.second
        >>> v, w = sulax.eigh(A)
        >>> su.get_unit(w) == su.second
        True
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        v, w = lax.linalg.eigh(x.mantissa, lower=lower, symmetrize_input=symmetrize_input,
                               sort_eigenvalues=sort_eigenvalues, subset_by_index=subset_by_index)
        return v, maybe_decimal(Quantity(w, unit=x.unit))
    else:
        return lax.linalg.eigh(x, lower=lower, symmetrize_input=symmetrize_input,
                               sort_eigenvalues=sort_eigenvalues, subset_by_index=subset_by_index)


@set_module_as('saiunit.lax')
def hessenberg(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> tuple[Quantity | jax.Array, jax.Array]:
    """Reduce a square matrix to upper Hessenberg form.

    Currently implemented on CPU only.

    Parameters
    ----------
    x : array_like or Quantity
        A floating-point or complex square matrix (or batch of matrices)
        with shape ``[..., n, n]``.

    Returns
    -------
    h : jax.Array or Quantity
        The upper Hessenberg form.  The upper triangle and first
        sub-diagonal contain the Hessenberg matrix; elements below the
        first sub-diagonal hold the Householder reflectors.  If ``x``
        has a unit, ``h`` preserves that unit.
    taus : jax.Array
        Scalar factors of the elementary Householder reflectors (unitless).

    See Also
    --------
    saiunit.lax.householder_product : Reconstruct Q from reflectors.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> A = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * su.second
        >>> h, taus = sulax.hessenberg(A)
        >>> su.get_unit(h) == su.second
        True
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        h, q = lax.linalg.hessenberg(x.mantissa)
        return maybe_decimal(Quantity(h, unit=x.unit)), q
    else:
        return lax.linalg.hessenberg(x)


@set_module_as('saiunit.lax')
def lu(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> tuple[Quantity | jax.Array, jax.Array, jax.Array]:
    r"""LU decomposition with partial pivoting.

    Compute the matrix decomposition :math:`P \cdot A = L \cdot U` where
    :math:`P` is a permutation matrix, :math:`L` is lower-triangular with
    unit diagonal, and :math:`U` is upper-triangular.

    Parameters
    ----------
    x : array_like or Quantity
        A batch of matrices with shape ``[..., m, n]``.

    Returns
    -------
    lu : jax.Array or Quantity
        A matrix containing :math:`L` in its lower triangle and :math:`U`
        in its upper triangle (the unit diagonal of :math:`L` is implicit).
        If ``x`` has a unit, ``lu`` preserves that unit.
    pivots : jax.Array
        An ``int32`` array with shape ``[..., min(m, n)]`` encoding row
        swaps.
    permutation : jax.Array
        An ``int32`` array with shape ``[..., m]`` representing the row
        permutation.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> A = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * su.second
        >>> lu_mat, pivots, perm = sulax.lu(A)
        >>> su.get_unit(lu_mat) == su.second
        True
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        p, l, u = lax.linalg.lu(x.mantissa)
        return maybe_decimal(Quantity(p, unit=x.unit)), l, u
    else:
        return lax.linalg.lu(x)


@set_module_as('saiunit.lax')
def householder_product(
    a: Union[Quantity, jax.typing.ArrayLike],
    taus: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    """Product of elementary Householder reflectors.

    Reconstruct the orthogonal (unitary) matrix :math:`Q` from a set of
    elementary Householder reflectors and their scalar factors.

    Parameters
    ----------
    a : array_like or Quantity
        A matrix with shape ``[..., m, n]`` whose lower triangle contains
        the elementary Householder reflectors.  Units, if present, are
        stripped before computation.
    taus : array_like or Quantity
        A vector with shape ``[..., k]`` (``k < min(m, n)``) containing
        the scalar factors.  Units, if present, are stripped.

    Returns
    -------
    Q : jax.Array
        The orthogonal (unitary) matrix with the same shape as ``a``
        (always unitless).

    See Also
    --------
    saiunit.lax.hessenberg : Produces reflectors consumed by this function.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> taus = jnp.array([1.0])
        >>> Q = sulax.householder_product(a, taus)
        >>> Q.shape
        (2, 2)
    """
    # TODO: more proper handling of Quantity?
    a = maybe_custom_array(a)
    taus = maybe_custom_array(taus)
    if isinstance(a, Quantity) and isinstance(taus, Quantity):
        return lax.linalg.householder_product(a.mantissa, taus.mantissa)
    elif isinstance(a, Quantity):
        return lax.linalg.householder_product(a.mantissa, taus)
    elif isinstance(taus, Quantity):
        return lax.linalg.householder_product(a, taus.mantissa)
    else:
        return lax.linalg.householder_product(a, taus)


@set_module_as('saiunit.lax')
def qdwh(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> tuple[jax.Array, Quantity | jax.Array, int, bool]:
    r"""Polar decomposition via QR-based dynamically weighted Halley iteration.

    Compute the polar decomposition :math:`x = U \cdot H` where :math:`U`
    is unitary and :math:`H` is Hermitian positive semi-definite.

    Parameters
    ----------
    x : array_like or Quantity
        A full-rank matrix with shape ``(M, N)``.

    Returns
    -------
    u : jax.Array
        The unitary factor (unitless).
    h : jax.Array or Quantity
        The Hermitian positive semi-definite factor.  If ``x`` has a unit,
        ``h`` preserves that unit.
    num_iters : int
        Number of iterations performed.
    is_converged : bool
        ``True`` if the algorithm converged within the maximum number of
        iterations.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> A = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * su.second
        >>> u, h, num_iters, is_converged = sulax.qdwh(A)
        >>> su.get_unit(h) == su.second
        True
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        u, h, num_iters, is_converged = lax.linalg.qdwh(x.mantissa)
        return u, maybe_decimal(Quantity(h, unit=x.unit)), num_iters, is_converged
    else:
        return lax.linalg.qdwh(x)


@set_module_as('saiunit.lax')
def qr(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> tuple[jax.Array, Quantity | jax.Array]:
    r"""QR decomposition.

    Compute the QR decomposition :math:`A = Q \cdot R` where :math:`Q` is
    a unitary (orthogonal) matrix and :math:`R` is upper-triangular.

    Parameters
    ----------
    x : array_like or Quantity
        A batch of matrices with shape ``[..., m, n]``.

    Returns
    -------
    q : jax.Array
        The unitary factor (unitless) with shape ``[..., m, min(m, n)]``.
    r : jax.Array or Quantity
        The upper-triangular factor with shape ``[..., min(m, n), n]``.
        If ``x`` has a unit, ``r`` preserves that unit.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> A = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * su.meter
        >>> q, r = sulax.qr(A)
        >>> su.get_unit(r) == su.meter
        True
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        q, r = lax.linalg.qr(x.mantissa)
        return q, maybe_decimal(Quantity(r, unit=x.unit))
    else:
        return lax.linalg.qr(x)


@set_module_as('saiunit.lax')
def schur(
    x: Union[Quantity, jax.typing.ArrayLike],
    compute_schur_vectors: bool = True,
    sort_eig_vals: bool = False,
    select_callable: Callable[..., Any] | None = None
) -> tuple[jax.Array, Quantity | jax.Array]:
    r"""Schur decomposition.

    Compute the Schur decomposition :math:`A = Q \cdot T \cdot Q^H` where
    :math:`T` is upper-triangular (quasi-triangular for real matrices) and
    :math:`Q` is unitary.

    Parameters
    ----------
    x : array_like or Quantity
        A batch of square matrices with shape ``[..., n, n]``.
    compute_schur_vectors : bool, optional
        If ``True``, compute the Schur vectors ``Q``. Default is ``True``.
    sort_eig_vals : bool, optional
        If ``True``, sort eigenvalues. Default is ``False``.
    select_callable : callable or None, optional
        A function used to select eigenvalues for reordering.
        Default is ``None``.

    Returns
    -------
    t : jax.Array
        The Schur form (unitless).
    q : jax.Array or Quantity
        The Schur vectors.  If ``x`` has a unit, ``q`` preserves that unit.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> A = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * su.second
        >>> t, q = sulax.schur(A)
        >>> su.get_unit(q) == su.second
        True
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        t, q = lax.linalg.schur(x.mantissa, compute_schur_vectors=compute_schur_vectors,
                                sort_eig_vals=sort_eig_vals, select_callable=select_callable)
        return t, maybe_decimal(Quantity(q, unit=x.unit))
    else:
        return lax.linalg.schur(x, compute_schur_vectors=compute_schur_vectors,
                                sort_eig_vals=sort_eig_vals, select_callable=select_callable)


@set_module_as('saiunit.lax')
def svd(
    x: Union[Quantity, jax.typing.ArrayLike],
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
    subset_by_index: tuple[int, int] | None = None,
    algorithm: jax.lax.linalg.SvdAlgorithm | None = None,
) -> Union[Quantity, jax.typing.ArrayLike] | tuple[jax.Array, Quantity | jax.Array, jax.Array]:
    """Singular value decomposition.

    Compute the SVD of a matrix.  When ``compute_uv`` is ``True``, return
    ``(u, s, vh)``; otherwise return only the singular values ``s``.

    Parameters
    ----------
    x : array_like or Quantity
        A batch of matrices with shape ``[..., m, n]``.
    full_matrices : bool, optional
        If ``True``, compute full-size ``U`` and ``Vh``.
        Default is ``True``.
    compute_uv : bool, optional
        If ``True``, compute ``U`` and ``Vh`` in addition to ``S``.
        Default is ``True``.
    subset_by_index : tuple of int or None, optional
        Optional ``(start, end)`` range of singular-value indices.
        Default is ``None``.
    algorithm : SvdAlgorithm or None, optional
        The SVD algorithm to use. Default is ``None``.

    Returns
    -------
    u : jax.Array
        Left singular vectors (unitless).  Only returned when
        ``compute_uv=True``.
    s : jax.Array or Quantity
        Singular values.  If ``x`` has a unit, ``s`` preserves that unit.
    vh : jax.Array
        Right singular vectors (unitless).  Only returned when
        ``compute_uv=True``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> A = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * su.meter
        >>> u, s, vh = sulax.svd(A)
        >>> su.get_unit(s) == su.meter
        True
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        if compute_uv:
            u, s, vh = lax.linalg.svd(x.mantissa, full_matrices=full_matrices, compute_uv=compute_uv,
                                      subset_by_index=subset_by_index, algorithm=algorithm)
            return u, maybe_decimal(Quantity(s, unit=x.unit)), vh
        else:
            s = lax.linalg.svd(x.mantissa, full_matrices=full_matrices, compute_uv=compute_uv,
                               subset_by_index=subset_by_index, algorithm=algorithm)
            return maybe_decimal(Quantity(s, unit=x.unit))
    else:
        return lax.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv,
                              subset_by_index=subset_by_index, algorithm=algorithm)


@set_module_as('saiunit.lax')
def triangular_solve(
    a: Union[Quantity, jax.typing.ArrayLike],
    b: Union[Quantity, jax.typing.ArrayLike],
    left_side: bool = False, lower: bool = False,
    transpose_a: bool = False, conjugate_a: bool = False,
    unit_diagonal: bool = False,
) -> Quantity | jax.Array:
    r"""Triangular solve.

    Solve the matrix equation :math:`\mathit{op}(A) \cdot X = B` (when
    ``left_side=True``) or :math:`X \cdot \mathit{op}(A) = B` (when
    ``left_side=False``), where :math:`A` is triangular.

    Parameters
    ----------
    a : array_like or Quantity
        A batch of triangular matrices with shape ``[..., m, m]``.
    b : array_like or Quantity
        A batch of right-hand-side matrices with shape ``[..., m, n]``
        (if ``left_side=True``) or ``[..., n, m]`` otherwise.
    left_side : bool, optional
        Selects which equation to solve. Default is ``False``.
    lower : bool, optional
        If ``True``, use the lower triangle of ``a``. Default is ``False``.
    transpose_a : bool, optional
        If ``True``, transpose ``a`` before solving. Default is ``False``.
    conjugate_a : bool, optional
        If ``True``, use the complex conjugate of ``a``.
        Default is ``False``.
    unit_diagonal : bool, optional
        If ``True``, the diagonal of ``a`` is assumed to be all ones.
        Default is ``False``.

    Returns
    -------
    X : jax.Array or Quantity
        The solution with the same shape and dtype as ``b``.  If ``b``
        carries a unit, ``X`` preserves that unit.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> A = jnp.array([[2.0, 0.0], [1.0, 3.0]])
        >>> b = jnp.array([[4.0], [7.0]]) * su.meter
        >>> X = sulax.triangular_solve(A, b, left_side=True, lower=True)
        >>> su.get_unit(X) == su.meter
        True
    """
    a = maybe_custom_array(a)
    b = maybe_custom_array(b)
    if isinstance(a, Quantity) and isinstance(b, Quantity):
        return maybe_decimal(Quantity(lax.linalg.triangular_solve(a.mantissa, b.mantissa, left_side=left_side,
                                                                  lower=lower, transpose_a=transpose_a,
                                                                  conjugate_a=conjugate_a,
                                                                  unit_diagonal=unit_diagonal), unit=b.unit))
    elif isinstance(a, Quantity):
        return lax.linalg.triangular_solve(a.mantissa, b, left_side=left_side,
                                           lower=lower, transpose_a=transpose_a, conjugate_a=conjugate_a,
                                           unit_diagonal=unit_diagonal)
    elif isinstance(b, Quantity):
        return maybe_decimal(Quantity(lax.linalg.triangular_solve(a, b.mantissa, left_side=left_side,
                                                                  lower=lower, transpose_a=transpose_a,
                                                                  conjugate_a=conjugate_a,
                                                                  unit_diagonal=unit_diagonal), unit=b.unit))
    else:
        return lax.linalg.triangular_solve(a, b, left_side=left_side,
                                           lower=lower, transpose_a=transpose_a, conjugate_a=conjugate_a,
                                           unit_diagonal=unit_diagonal)


@set_module_as('saiunit.lax')
def tridiagonal(
    a: Union[Quantity, jax.typing.ArrayLike],
    lower: bool = True,
) -> tuple[Quantity | jax.Array, Quantity | jax.Array, Quantity | jax.Array, jax.Array]:
    """Reduce a symmetric/Hermitian matrix to tridiagonal form.

    Currently implemented on CPU and GPU only.

    Parameters
    ----------
    a : array_like or Quantity
        A floating-point or complex symmetric/Hermitian matrix (or batch of
        matrices) with shape ``[..., n, n]``.
    lower : bool, optional
        Selects which triangle of the input to use. Default is ``True``.

    Returns
    -------
    a_out : jax.Array or Quantity
        The matrix with the tridiagonal representation stored in its
        diagonal and first sub/super-diagonal; remaining elements hold the
        Householder reflectors.  If ``a`` has a unit, ``a_out`` preserves
        that unit.
    d : jax.Array or Quantity
        The diagonal of the tridiagonal matrix.  Preserves the unit of ``a``
        if present.
    e : jax.Array or Quantity
        The first sub-diagonal (``lower=True``) or super-diagonal
        (``lower=False``).  Preserves the unit of ``a`` if present.
    taus : jax.Array
        Scalar factors of the elementary Householder reflectors (unitless).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> A = jnp.array([[2.0, 1.0], [1.0, 3.0]]) * su.second
        >>> a_out, d, e, taus = sulax.tridiagonal(A)
        >>> su.get_unit(d) == su.second
        True
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        arr, d, e, taus = lax.linalg.tridiagonal(a.mantissa, lower=lower)
        return maybe_decimal(Quantity(a, unit=a.unit)), maybe_decimal(Quantity(d, unit=a.unit)), \
            maybe_decimal(Quantity(e, unit=a.unit)), taus
    else:
        return lax.linalg.tridiagonal(a, lower=lower)


@set_module_as('saiunit.lax')
def tridiagonal_solve(
    dl: Union[Quantity, jax.typing.ArrayLike],
    d: Union[Quantity, jax.typing.ArrayLike],
    du: Union[Quantity, jax.typing.ArrayLike],
    b: Union[Quantity, jax.typing.ArrayLike],
) -> Quantity | jax.Array:
    r"""Solve a tridiagonal linear system.

    Compute the solution :math:`X` of the tridiagonal system
    :math:`A \cdot X = B`, where the tridiagonal matrix :math:`A` is
    specified by its three diagonals.

    Parameters
    ----------
    dl : array_like or Quantity
        Lower diagonal with shape ``[..., m]``.
        ``dl[i] = A[i, i-1]``; ``dl[0]`` is unused.  Must have the same
        unit as ``d`` and ``du``.
    d : array_like or Quantity
        Main diagonal with shape ``[..., m]``.
        ``d[i] = A[i, i]``.
    du : array_like or Quantity
        Upper diagonal with shape ``[..., m]``.
        ``du[i] = A[i, i+1]``; ``du[m-1]`` is unused.  Must have the same
        unit as ``dl`` and ``d``.
    b : array_like or Quantity
        Right-hand-side matrix.

    Returns
    -------
    X : jax.Array or Quantity
        The solution of the tridiagonal system.  If ``b`` has a unit, ``X``
        preserves that unit.

    Raises
    ------
    saiunit.DimensionMismatchError
        If ``dl``, ``d``, and ``du`` do not share the same unit.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> dl = jnp.array([0.0, 1.0, 1.0])
        >>> d  = jnp.array([2.0, 2.0, 2.0])
        >>> du = jnp.array([1.0, 1.0, 0.0])
        >>> b  = jnp.array([[1.0], [2.0], [3.0]]) * su.meter
        >>> X = sulax.tridiagonal_solve(dl, d, du, b)
        >>> su.get_unit(X) == su.meter
        True
    """
    dl = maybe_custom_array(dl)
    d = maybe_custom_array(d)
    du = maybe_custom_array(du)
    b = maybe_custom_array(b)
    fail_for_unit_mismatch(dl, d)
    fail_for_unit_mismatch(dl, du)
    if isinstance(b, Quantity):
        try:
            return maybe_decimal(
                Quantity(lax.linalg.tridiagonal_solve(dl.mantissa, d.mantissa, du.mantissa, b.mantissa), unit=b.unit))
        except:
            return Quantity(lax.linalg.tridiagonal_solve(dl, d, du, b.mantissa), unit=b.unit)
    else:
        try:
            return lax.linalg.tridiagonal_solve(dl.mantissa, d.mantissa, du.mantissa, b)
        except:
            return lax.linalg.tridiagonal_solve(dl, d, du, b)
