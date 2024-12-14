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

import operator
from typing import Any, Tuple, Sequence, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import tree_util
from jax._src.lax.lax import _const
from jax.experimental.sparse import JAXSparse, coo_todense_p, coo_fromdense_p, coo_matmat_p, coo_matvec_p

from brainunit._base import Quantity, split_mantissa_unit, maybe_decimal, get_mantissa, get_unit
from brainunit._sparse_base import SparseMatrix
from brainunit.math._fun_array_creation import asarray
from brainunit.math._fun_keep_unit import promote_dtypes

__all__ = [
    'COO', 'coo_todense', 'coo_fromdense',
]

Dtype = Any
Shape = tuple[int, ...]


class COOInfo(NamedTuple):
    shape: Shape
    rows_sorted: bool = False
    cols_sorted: bool = False


@tree_util.register_pytree_node_class
class COO(SparseMatrix):
    """Experimental COO matrix implemented in JAX.

    Note: this class has minimal compatibility with JAX transforms such as
    grad and autodiff, and offers very little functionality. In general you
    should prefer :class:`jax.experimental.sparse.BCOO`.

    Additionally, there are known failures in the case that `nse` is larger
    than the true number of nonzeros in the represented matrix. This situation
    is better handled in BCOO.
    """
    data: jax.Array
    row: jax.Array
    col: jax.Array
    shape: tuple[int, int]
    nse = property(lambda self: self.data.size)
    dtype = property(lambda self: self.data.dtype)
    _info = property(
        lambda self: COOInfo(
            shape=self.shape,
            rows_sorted=self._rows_sorted,
            cols_sorted=self._cols_sorted)
    )
    _bufs = property(lambda self: (self.data, self.row, self.col))
    _rows_sorted: bool
    _cols_sorted: bool

    def __init__(
        self,
        args: Tuple[jax.Array | Quantity, jax.Array, jax.Array],
        *,
        shape: Shape,
        rows_sorted: bool = False,
        cols_sorted: bool = False
    ):
        self.data, self.row, self.col = map(asarray, args)
        self._rows_sorted = rows_sorted
        self._cols_sorted = cols_sorted
        super().__init__(args, shape=shape)

    @classmethod
    def fromdense(
        cls,
        mat: jax.Array,
        *,
        nse: int | None = None,
        index_dtype: jax.typing.DTypeLike = np.int32
    ) -> COO:
        return coo_fromdense(mat, nse=nse, index_dtype=index_dtype)

    def _sort_indices(self) -> COO:
        """Return a copy of the COO matrix with sorted indices.

        The matrix is sorted by row indices and column indices per row.
        If self._rows_sorted is True, this returns ``self`` without a copy.
        """
        # TODO(jakevdp): would be benefit from lowering this to cusparse sort_rows utility?
        if self._rows_sorted:
            return self
        row, col, data = lax.sort((self.row, self.col, self.data), num_keys=2)
        return self.__class__(
            (data, row, col),
            shape=self.shape,
            rows_sorted=True
        )

    @classmethod
    def _empty(
        cls,
        shape: Sequence[int],
        *,
        dtype: jax.typing.DTypeLike | None = None,
        index_dtype: jax.typing.DTypeLike = 'int32'
    ) -> COO:
        """Create an empty COO instance. Public method is sparse.empty()."""
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError(f"COO must have ndim=2; got {shape=}")
        data = jnp.empty(0, dtype)
        row = col = jnp.empty(0, index_dtype)
        return cls(
            (data, row, col),
            shape=shape,
            rows_sorted=True,
            cols_sorted=True
        )

    @classmethod
    def _eye(
        cls,
        N: int,
        M: int,
        k: int,
        *,
        dtype: jax.typing.DTypeLike | None = None,
        index_dtype: jax.typing.DTypeLike = 'int32'
    ) -> COO:
        if k > 0:
            diag_size = min(N, M - k)
        else:
            diag_size = min(N + k, M)

        if diag_size <= 0:
            # if k is out of range, return an empty matrix.
            return cls._empty((N, M), dtype=dtype, index_dtype=index_dtype)

        data = jnp.ones(diag_size, dtype=dtype)
        idx = jnp.arange(diag_size, dtype=index_dtype)
        zero = _const(idx, 0)
        k = _const(idx, k)
        row = lax.sub(idx, lax.cond(k >= 0, lambda: zero, lambda: k))
        col = lax.add(idx, lax.cond(k <= 0, lambda: zero, lambda: k))
        return cls((data, row, col), shape=(N, M), rows_sorted=True, cols_sorted=True)

    def with_data(self, data: jax.Array | Quantity) -> COO:
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert get_unit(data) == get_unit(self.data)
        return COO((data, self.row, self.col), shape=self.shape)

    def todense(self) -> jax.Array:
        return coo_todense(self)

    def transpose(self, axes: Tuple[int, ...] | None = None) -> COO:
        if axes is not None:
            raise NotImplementedError("axes argument to transpose()")
        return COO((self.data, self.col, self.row), shape=self.shape[::-1],
                   rows_sorted=self._cols_sorted, cols_sorted=self._rows_sorted)

    def tree_flatten(self) -> Tuple[
        Tuple[jax.Array | Quantity,], dict[str, Any]
    ]:
        aux = self._info._asdict()
        aux['row'] = self.row
        aux['col'] = self.col
        return (self.data,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.data, = children
        if aux_data.keys() != {'shape', 'rows_sorted', 'cols_sorted', 'row', 'col'}:
            raise ValueError(f"COO.tree_unflatten: invalid {aux_data=}")
        obj.shape = aux_data['shape']
        obj._rows_sorted = aux_data['rows_sorted']
        obj._cols_sorted = aux_data['cols_sorted']
        obj.row = aux_data['row']
        obj.col = aux_data['col']
        return obj

    def __abs__(self):
        return COO(
            (self.data.__abs__(), self.row, self.col),
            shape=self.shape,
            rows_sorted=self._rows_sorted,
            cols_sorted=self._cols_sorted
        )

    def __neg__(self):
        return COO(
            (-self.data, self.row, self.col),
            shape=self.shape,
            rows_sorted=self._rows_sorted,
            cols_sorted=self._cols_sorted
        )

    def __pos__(self):
        return COO(
            (self.data.__pos__(), self.row, self.col),
            shape=self.shape,
            rows_sorted=self._rows_sorted,
            cols_sorted=self._cols_sorted
        )

    def _binary_op(self, other, op):
        if isinstance(other, COO):
            if id(self.row) == id(other.row) and id(self.col) == id(other.col):
                return COO(
                    (op(self.data, other.data), self.row, self.col),
                    shape=self.shape,
                    rows_sorted=self._rows_sorted,
                    cols_sorted=self._cols_sorted
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = asarray(other)
        if other.size == 1:
            return COO(
                (op(self.data, other), self.row, self.col),
                shape=self.shape,
                rows_sorted=self._rows_sorted,
                cols_sorted=self._cols_sorted
            )
        elif other.ndim == 2 and other.shape == self.shape:
            other = other[self.row, self.col]
            return COO(
                (op(self.data, other), self.row, self.col),
                shape=self.shape,
                rows_sorted=self._rows_sorted,
                cols_sorted=self._cols_sorted
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, COO):
            if id(self.row) == id(other.row) and id(self.col) == id(other.col):
                return COO(
                    (op(other.data, self.data), self.row, self.col),
                    shape=self.shape,
                    rows_sorted=self._rows_sorted,
                    cols_sorted=self._cols_sorted
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = asarray(other)
        if other.size == 1:
            return COO(
                (op(other, self.data), self.row, self.col),
                shape=self.shape,
                rows_sorted=self._rows_sorted,
                cols_sorted=self._cols_sorted
            )
        elif other.ndim == 2 and other.shape == self.shape:
            other = other[self.row, self.col]
            return COO(
                (op(other, self.data), self.row, self.col),
                shape=self.shape,
                rows_sorted=self._rows_sorted,
                cols_sorted=self._cols_sorted
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __mul__(self, other: jax.Array | Quantity) -> COO:
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: jax.Array | Quantity) -> COO:
        return self._binary_rop(other, operator.mul)

    def __div__(self, other: jax.Array | Quantity) -> COO:
        return self._binary_op(other, operator.truediv)

    def __rdiv__(self, other: jax.Array | Quantity) -> COO:
        return self._binary_rop(other, operator.truediv)

    def __truediv__(self, other) -> COO:
        return self.__div__(other)

    def __rtruediv__(self, other) -> COO:
        return self.__rdiv__(other)

    def __add__(self, other) -> COO:
        return self._binary_op(other, operator.add)

    def __radd__(self, other) -> COO:
        return self._binary_rop(other, operator.add)

    def __sub__(self, other) -> COO:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other) -> COO:
        return self._binary_rop(other, operator.sub)

    def __mod__(self, other) -> COO:
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other) -> COO:
        return self._binary_rop(other, operator.mod)

    def __matmul__(
        self, other: jax.typing.ArrayLike
    ) -> jax.Array | Quantity:
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        other = asarray(other)
        data, other = promote_dtypes(self.data, other)
        self_promoted = COO((data, self.row, self.col), **self._info._asdict())
        if other.ndim == 1:
            return coo_matvec(self_promoted, other)
        elif other.ndim == 2:
            return coo_matmat(self_promoted, other)
        else:
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(
        self,
        other: jax.typing.ArrayLike
    ) -> jax.Array | Quantity:
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        other = asarray(other)
        data, other = promote_dtypes(self.data, other)
        self_promoted = COO((data, self.row, self.col), **self._info._asdict())
        if other.ndim == 1:
            return coo_matvec(self_promoted, other, transpose=True)
        elif other.ndim == 2:
            other = other.T
            return coo_matmat(self_promoted, other, transpose=True).T
        else:
            raise NotImplementedError(f"matmul with object of shape {other.shape}")


def coo_todense(mat: COO) -> jax.Array | Quantity:
    """Convert a COO-format sparse matrix to a dense matrix.

    Args:
      mat : COO matrix
    Returns:
      mat_dense: dense version of ``mat``
    """
    return _coo_todense(mat.data, mat.row, mat.col, spinfo=mat._info)


def coo_fromdense(
    mat: jax.Array | Quantity,
    *,
    nse: int | None = None,
    index_dtype: jax.typing.DTypeLike = jnp.int32
) -> COO:
    """Create a COO-format sparse matrix from a dense matrix.

    Args:
      mat : array to be converted to COO.
      nse : number of specified entries in ``mat``. If not specified,
        it will be computed from the input matrix.
      index_dtype : dtype of sparse indices

    Returns:
      mat_coo : COO representation of the matrix.
    """
    if nse is None:
        nse = int((get_mantissa(mat) != 0).sum())
    nse_int = jax.core.concrete_or_error(operator.index, nse, "coo_fromdense nse argument")
    return COO(
        _coo_fromdense(mat, nse=nse_int, index_dtype=index_dtype),
        shape=mat.shape,
        rows_sorted=True
    )


def _coo_todense(
    data: jax.Array | Quantity,
    row: jax.Array,
    col: jax.Array,
    *,
    spinfo: COOInfo
) -> jax.Array | Quantity:
    """Convert CSR-format sparse matrix to a dense matrix.

    Args:
      data : array of shape ``(nse,)``.
      row : array of shape ``(nse,)``
      col : array of shape ``(nse,)`` and dtype ``row.dtype``
      spinfo : COOInfo object containing matrix metadata

    Returns:
      mat : array with specified shape and dtype matching ``data``
    """
    data, unit = split_mantissa_unit(data)
    r = coo_todense_p.bind(data, row, col, spinfo=spinfo)
    return maybe_decimal(r * unit)


def _coo_fromdense(
    mat: jax.Array | Quantity,
    *,
    nse: int,
    index_dtype: jax.typing.DTypeLike = jnp.int32
) -> Tuple[jax.Array | Quantity, jax.Array, jax.Array]:
    """Create COO-format sparse matrix from a dense matrix.

    Args:
      mat : array to be converted to COO.
      nse : number of specified entries in ``mat``
      index_dtype : dtype of sparse indices

    Returns:
      data : array of shape ``(nse,)`` and dtype ``mat.dtype``
      row : array of shape ``(nse,)`` and dtype ``index_dtype``
      col : array of shape ``(nse,)`` and dtype ``index_dtype``
    """
    mat = asarray(mat)
    mat, unit = split_mantissa_unit(mat)
    nse = jax.core.concrete_or_error(operator.index, nse, "nse argument of coo_fromdense()")
    r = coo_fromdense_p.bind(mat, nse=nse, index_dtype=index_dtype)
    if unit.is_unitless:
        return r
    return r[0] * unit, r[1], r[2]


def coo_matvec(
    mat: COO,
    v: jax.Array | Quantity,
    transpose: bool = False
) -> jax.Array | Quantity:
    """Product of COO sparse matrix and a dense vector.

    Args:
      mat : COO matrix
      v : one-dimensional array of size ``(shape[0] if transpose else shape[1],)`` and
        dtype ``mat.dtype``
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      y : array of shape ``(mat.shape[1] if transpose else mat.shape[0],)`` representing
        the matrix vector product.
    """
    data, row, col = mat._bufs
    return _coo_matvec(data, row, col, v, spinfo=mat._info, transpose=transpose)


def _coo_matvec(
    data: jax.Array | Quantity,
    row: jax.Array,
    col: jax.Array,
    v: jax.Array | Quantity,
    *,
    spinfo: COOInfo,
    transpose: bool = False
) -> jax.Array | Quantity:
    """Product of COO sparse matrix and a dense vector.

    Args:
      data : array of shape ``(nse,)``.
      row : array of shape ``(nse,)``
      col : array of shape ``(nse,)`` and dtype ``row.dtype``
      v : array of shape ``(shape[0] if transpose else shape[1],)`` and
        dtype ``data.dtype``
      shape : length-2 tuple representing the matrix shape
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      y : array of shape ``(shape[1] if transpose else shape[0],)`` representing
        the matrix vector product.
    """
    data, unita = split_mantissa_unit(data)
    v, unitv = split_mantissa_unit(v)
    r = coo_matvec_p.bind(data, row, col, v, spinfo=spinfo, transpose=transpose)
    return maybe_decimal(r * unita * unitv)


def coo_matmat(
    mat: COO,
    B: jax.Array | Quantity,
    *,
    transpose: bool = False
) -> jax.Array | Quantity:
    """Product of COO sparse matrix and a dense matrix.

    Args:
      mat : COO matrix
      B : array of shape ``(mat.shape[0] if transpose else mat.shape[1], cols)`` and
        dtype ``mat.dtype``
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      C : array of shape ``(mat.shape[1] if transpose else mat.shape[0], cols)``
        representing the matrix vector product.
    """
    data, row, col = mat._bufs
    return _coo_matmat(data, row, col, B, spinfo=mat._info, transpose=transpose)


def _coo_matmat(
    data: jax.Array | Quantity,
    row: jax.Array,
    col: jax.Array,
    B: jax.Array | Quantity,
    *,
    spinfo: COOInfo,
    transpose: bool = False
) -> jax.Array:
    """Product of COO sparse matrix and a dense matrix.

    Args:
      data : array of shape ``(nse,)``.
      row : array of shape ``(nse,)``
      col : array of shape ``(nse,)`` and dtype ``row.dtype``
      B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
        dtype ``data.dtype``
      shape : length-2 tuple representing the matrix shape
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      C : array of shape ``(shape[1] if transpose else shape[0], cols)``
        representing the matrix vector product.
    """
    data, unita = split_mantissa_unit(data)
    B, unitb = split_mantissa_unit(B)
    res = coo_matmat_p.bind(data, row, col, B, spinfo=spinfo, transpose=transpose)
    return maybe_decimal(res * unita * unitb)
