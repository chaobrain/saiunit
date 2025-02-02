from __future__ import annotations

import sys
from typing import Union, Callable, Any

import jax
from jax import lax, Array

from saiunit.lax._lax_change_unit import unit_change
from .._base import Quantity, maybe_decimal, fail_for_unit_mismatch
from .._misc import set_module_as
from ..math._fun_change_unit import _fun_change_unit_unary

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
    """Cholesky decomposition.

    Computes the Cholesky decomposition

    .. math::
        A = L . L^H

    of square matrices, :math:`A`, such that :math:`L`
    is lower triangular. The matrices of :math:`A` must be positive-definite and
    either Hermitian, if complex, or symmetric, if real.

    Args:
        x: A batch of square Hermitian (symmetric if real) positive-definite
            matrices with shape ``[..., n, n]``.
        symmetrize_input: If ``True``, the matrix is symmetrized before Cholesky
              decomposition by computing :math:`\\frac{1}{2}(x + x^H)`. If ``False``,
              only the lower triangle of ``x`` is used; the upper triangle is ignored
              and not accessed.

    Returns:
        The Cholesky decomposition as a matrix with the same dtype as ``x`` and
        shape ``[..., n, n]``. If Cholesky decomposition fails, returns a matrix
        full of NaNs. The behavior on failure may change in the future.
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

    Nonsymmetric eigendecomposition is at present only implemented on CPU.

    Args:
        x: A batch of square matrices with shape ``[..., n, n]``.
        compute_left_eigenvectors: If true, the left eigenvectors will be computed.
        compute_right_eigenvectors: If true, the right eigenvectors will be
            computed.
    Returns:
        The eigendecomposition of ``x``, which is a tuple of the form
        ``(w, vl, vr)`` where ``w`` are the eigenvalues, ``vl`` are the left
        eigenvectors, and ``vr`` are the right eigenvectors. ``vl`` and ``vr`` are
        optional and will only be included if ``compute_left_eigenvectors`` or
        ``compute_right_eigenvectors`` respectively are ``True``.

    If the eigendecomposition fails, then arrays full of NaNs will be returned
    for that batch element.
    """
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

    Computes the eigenvectors and eigenvalues of a complex Hermitian or real
    symmetric square matrix.

    Args:
        x: A batch of square complex Hermitian or real symmetric matrices with shape
            ``[..., n, n]``.
        lower: If ``symmetrize_input`` is ``False``, describes which triangle of the
              input matrix to use. If ``symmetrize_input`` is ``False``, only the
              triangle given by ``lower`` is accessed; the other triangle is ignored and
              not accessed.
        symmetrize_input: If ``True``, the matrix is symmetrized before the
            eigendecomposition by computing :math:`\frac{1}{2}(x + x^H)`.
        sort_eigenvalues: If ``True``, the eigenvalues will be sorted in ascending
              order. If ``False`` the eigenvalues are returned in an
              implementation-defined order.
        subset_by_index: Optional 2-tuple [start, end] indicating the range of
               indices of eigenvalues to compute. For example, is ``range_select`` =
               [n-2,n], then ``eigh`` computes the two largest eigenvalues and their
               eigenvectors.

    Returns:
        A tuple ``(v, w)``.

        ``v`` is an array with the same dtype as ``x`` such that ``v[..., :, i]`` is
        the normalized eigenvector corresponding to eigenvalue ``w[..., i]``.

        ``w`` is an array with the same dtype as ``x`` (or its real counterpart if
        complex) with shape ``[..., d]`` containing the eigenvalues of ``x`` in
        ascending order(each repeated according to its multiplicity).
        If ``subset_by_index`` is ``None`` then ``d`` is equal to ``n``. Otherwise
        ``d`` is equal to ``subset_by_index[1] - subset_by_index[0]``.
    """
    if isinstance(x, Quantity):
        w, v = lax.linalg.eigh(x.mantissa, lower=lower, symmetrize_input=symmetrize_input,
                               sort_eigenvalues=sort_eigenvalues, subset_by_index=subset_by_index)
        return maybe_decimal(Quantity(w, unit=x.unit)), v
    else:
        return lax.linalg.eigh(x, lower=lower, symmetrize_input=symmetrize_input,
                               sort_eigenvalues=sort_eigenvalues, subset_by_index=subset_by_index)


@set_module_as('saiunit.lax')
def hessenberg(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> tuple[Quantity | jax.Array, jax.Array]:
    """Reduces a square matrix to upper Hessenberg form.

    Currently implemented on CPU only.

    Args:
        a: A floating point or complex square matrix or batch of matrices.

    Returns:
        A ``(a, taus)`` pair, where the upper triangle and first subdiagonal of ``a``
        contain the upper Hessenberg matrix, and the elements below the first
        subdiagonal contain the Householder reflectors. For each Householder
        reflector ``taus`` contains the scalar factors of the elementary Householder
        reflectors.
    """
    if isinstance(x, Quantity):
        h, q = lax.linalg.hessenberg(x.mantissa)
        return maybe_decimal(Quantity(h, unit=x.unit)), q
    else:
        return lax.linalg.hessenberg(x)


@set_module_as('saiunit.lax')
def lu(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> tuple[Quantity | jax.Array, jax.Array, jax.Array]:
    """LU decomposition with partial pivoting.

    Computes the matrix decomposition:

    .. math::
        P.A = L.U

    where :math:`P` is a permutation of the rows of :math:`A`, :math:`L` is a
    lower-triangular matrix with unit-diagonal elements, and :math:`U` is an
    upper-triangular matrix.

    Args:
        x: A batch of matrices with shape ``[..., m, n]``.

    Returns:
        A tuple ``(lu, pivots, permutation)``.

        ``lu`` is a batch of matrices with the same shape and dtype as ``x``
        containing the :math:`L` matrix in its lower triangle and the :math:`U`
        matrix in its upper triangle. The (unit) diagonal elements of :math:`L` are
        not represented explicitly.

        ``pivots`` is an int32 array with shape ``[..., min(m, n)]`` representing a
        sequence of row swaps that should be performed on :math:`A`.

        ``permutation`` is an alternative representation of the sequence of row
        swaps as a permutation, represented as an int32 array with shape
        ``[..., m]``.
    """
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

    Args:
        a: A matrix with shape ``[..., m, n]``, whose lower triangle contains
            elementary Householder reflectors.
        taus: A vector with shape ``[..., k]``, where ``k < min(m, n)``, containing
            the scalar factors of the elementary Householder reflectors.

    Returns:
        A batch of orthogonal (unitary) matrices with the same shape as ``a``,
        containing the products of the elementary Householder reflectors.
    """
    # TODO: more proper handling of Quantity?
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
    """QR-based dynamically weighted Halley iteration for polar decomposition.

    Args:
        x: A full-rank matrix, with shape `M x N`. The matrix may be padded up to
            that size from a smaller true shape (``dynamic_shape``).
        is_hermitian: True if `x` is Hermitian. Default to `False`. This parameter
            is currently unused, but exists for backward compatibility.
        eps: The final result will satisfy ``|x_k - x_k-1| < |x_k| *
            (4*eps)**(1/3)`` where `x_k` is the iterate.
        max_iterations: Iterations will terminate after this many steps even if the
            above is unsatisfied.
        dynamic_shape: the unpadded shape as an ``(m, n)`` tuple; optional.

    Returns:
        A four-tuple of (u, h, num_iters, is_converged) containing the
        polar decomposition of `x = u * h`, the number of iterations to compute `u`,
        and `is_converged`, whose value is `True` when the convergence is achieved
        within the maximum number of iterations.
    """
    if isinstance(x, Quantity):
        u, h, num_iters, is_converged = lax.linalg.qdwh(x.mantissa)
        return u, maybe_decimal(Quantity(h, unit=x.unit)), num_iters, is_converged
    else:
        return lax.linalg.qdwh(x)


@set_module_as('saiunit.lax')
def qr(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> tuple[jax.Array, Quantity | jax.Array]:
    """QR decomposition.

    Computes the QR decomposition

    .. math::
        A = Q . R

    of matrices :math:`A`, such that :math:`Q` is a unitary (orthogonal) matrix,
    and :math:`R` is an upper-triangular matrix.

    Args:
        x: A batch of matrices with shape ``[..., m, n]``.
        full_matrices: Determines if full or reduced matrices are returned; see
            below.

    Returns:
        A pair of arrays ``(q, r)``.

        Array ``q`` is a unitary (orthogonal) matrix,
        with shape ``[..., m, m]`` if ``full_matrices=True``, or
        ``[..., m, min(m, n)]`` if ``full_matrices=False``.

        Array ``r`` is an upper-triangular matrix with shape ``[..., m, n]`` if
        ``full_matrices=True``, or ``[..., min(m, n), n]`` if
        ``full_matrices=False``.
    """
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

    Returns the singular values if compute_uv is False, otherwise returns a triple
    containing the left singular vectors, the singular values and the adjoint of
    the right singular vectors.
    """
    if sys.version_info >= (3, 10):
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
    else:
        if isinstance(x, Quantity):
            if compute_uv:
                u, s, vh = lax.linalg.svd(x.mantissa, full_matrices=full_matrices, compute_uv=compute_uv,
                                          subset_by_index=subset_by_index)
                return u, maybe_decimal(Quantity(s, unit=x.unit)), vh
            else:
                s = lax.linalg.svd(x.mantissa, full_matrices=full_matrices, compute_uv=compute_uv,
                                   subset_by_index=subset_by_index)
                return maybe_decimal(Quantity(s, unit=x.unit))
        else:
            return lax.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv,
                                  subset_by_index=subset_by_index)


@set_module_as('saiunit.lax')
def triangular_solve(
    a: Union[Quantity, jax.typing.ArrayLike],
    b: Union[Quantity, jax.typing.ArrayLike],
    left_side: bool = False, lower: bool = False,
    transpose_a: bool = False, conjugate_a: bool = False,
    unit_diagonal: bool = False,
) -> Quantity | jax.Array:
    r"""Triangular solve.

    Solves either the matrix equation

    .. math::
        \mathit{op}(A) . X = B

    if ``left_side`` is ``True`` or

    .. math::
        X . \mathit{op}(A) = B

    if ``left_side`` is ``False``.

    ``A`` must be a lower or upper triangular square matrix, and where
    :math:`\mathit{op}(A)` may either transpose :math:`A` if ``transpose_a``
    is ``True`` and/or take its complex conjugate if ``conjugate_a`` is ``True``.

    Args:
        a: A batch of matrices with shape ``[..., m, m]``.
        b: A batch of matrices with shape ``[..., m, n]`` if ``left_side`` is
            ``True`` or shape ``[..., n, m]`` otherwise.
        left_side: describes which of the two matrix equations to solve; see above.
        lower: describes which triangle of ``a`` should be used. The other triangle
            is ignored.
        transpose_a: if ``True``, the value of ``a`` is transposed.
        conjugate_a: if ``True``, the complex conjugate of ``a`` is used in the
            solve. Has no effect if ``a`` is real.
        unit_diagonal: if ``True``, the diagonal of ``a`` is assumed to be unit
            (all 1s) and not accessed.

    Returns:
    A batch of matrices the same shape and dtype as ``b``.
    """
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
    """Reduces a symmetric/Hermitian matrix to tridiagonal form.

    Currently implemented on CPU and GPU only.

    Args:
        a: A floating point or complex matrix or batch of matrices.
        lower: Describes which triangle of the input matrices to use.
            The other triangle is ignored and not accessed.

    Returns:
    A ``(a, d, e, taus)`` pair. If ``lower=True``, the diagonal and first subdiagonal of
    matrix (or batch of matrices) ``a`` contain the tridiagonal representation,
    and elements below the first subdiagonal contain the elementary Householder
    reflectors, where additionally ``d`` contains the diagonal of the matrix and ``e`` contains
    the first subdiagonal.If ``lower=False`` the diagonal and first superdiagonal of the
    matrix contains the tridiagonal representation, and elements above the first
    superdiagonal contain the elementary Householder reflectors, where
    additionally ``d`` contains the diagonal of the matrix and ``e`` contains the
    first superdiagonal. ``taus`` contains the scalar factors of the elementary
    Householder reflectors.
    """
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
    r"""Computes the solution of a tridiagonal linear system.

    This function computes the solution of a tridiagonal linear system:

    .. math::
        A . X = B

    Args:

        dl: A batch of vectors with shape ``[..., m]``.
              The lower diagonal of A: ``dl[i] := A[i, i-1]`` for i in ``[0,m)``.
              Note that ``dl[0] = 0``.
        d: A batch of vectors with shape ``[..., m]``.
            The middle diagonal of A: ``d[i]  := A[i, i]`` for i in ``[0,m)``.
        du: A batch of vectors with shape ``[..., m]``.
              The upper diagonal of A: ``du[i] := A[i, i+1]`` for i in ``[0,m)``.
              Note that ``dl[m - 1] = 0``.
        b: Right hand side matrix.

    Returns:
        Solution ``X`` of tridiagonal system.
    """
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
