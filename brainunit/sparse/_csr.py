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
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax._src.lax.lax import _const
from jax.experimental.sparse import (
    JAXSparse, csr_fromdense_p, csr_todense_p, csr_matvec_p, csr_matmat_p
)

from brainunit._base import Quantity, split_mantissa_unit, maybe_decimal, get_mantissa, get_unit
from brainunit._sparse_base import SparseMatrix
from brainunit.math._fun_array_creation import asarray
from brainunit.math._fun_keep_unit import promote_dtypes

__all__ = [
    'CSR', 'CSC',
    'csr_fromdense', 'csr_todense',
    'csc_fromdense', 'csc_todense',
]

Shape = tuple[int, ...]


@tree_util.register_pytree_node_class
class CSR(SparseMatrix):
    """
    Unit-aware CSR matrix.
    """
    data: jax.Array | Quantity
    indices: jax.Array
    indptr: jax.Array
    shape: tuple[int, int]
    nse = property(lambda self: self.data.size)
    dtype = property(lambda self: self.data.dtype)
    _bufs = property(lambda self: (self.data, self.indices, self.indptr))

    def __init__(self, args, *, shape):
        self.data, self.indices, self.indptr = map(asarray, args)
        super().__init__(args, shape=shape)

    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=np.int32):
        if nse is None:
            nse = (get_mantissa(mat) != 0).sum()
        return csr_fromdense(mat, nse=nse, index_dtype=index_dtype)

    @classmethod
    def _empty(cls, shape, *, dtype=None, index_dtype='int32'):
        """Create an empty CSR instance. Public method is sparse.empty()."""
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError(f"CSR must have ndim=2; got {shape=}")
        data = jnp.empty(0, dtype)
        indices = jnp.empty(0, index_dtype)
        indptr = jnp.zeros(shape[0] + 1, index_dtype)
        return cls((data, indices, indptr), shape=shape)

    @classmethod
    def _eye(cls, N, M, k, *, dtype=None, index_dtype='int32'):
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
        col = jax.lax.add(idx, jax.lax.cond(k <= 0, lambda: zero, lambda: k))
        indices = col.astype(index_dtype)
        # TODO(jakevdp): this can be done more efficiently.
        row = jax.lax.sub(idx, jax.lax.cond(k >= 0, lambda: zero, lambda: k))
        indptr = jnp.zeros(N + 1, dtype=index_dtype).at[1:].set(
            jnp.cumsum(jnp.bincount(row, length=N).astype(index_dtype)))
        return cls((data, indices, indptr), shape=(N, M))

    def with_data(self, data: jax.Array | Quantity) -> CSR:
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert get_unit(data) == get_unit(self.data)
        return CSR((data, self.indices, self.indptr), shape=self.shape)

    def todense(self):
        return csr_todense(self)

    def transpose(self, axes=None):
        assert axes is None
        return CSC((self.data, self.indices, self.indptr), shape=self.shape[::-1])

    def __abs__(self):
        return CSR((abs(self.data), self.indices, self.indptr), shape=self.shape)

    def __neg__(self):
        return CSR((-self.data, self.indices, self.indptr), shape=self.shape)

    def __pos__(self):
        return CSR((self.data.__pos__(), self.indices, self.indptr), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSR(
                    (op(self.data, other.data), self.indices, self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = asarray(other)
        if other.size == 1:
            return CSR(
                (op(self.data, other), self.indices, self.indptr),
                shape=self.shape
            )
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR(
                (op(self.data, other), self.indices, self.indptr),
                shape=self.shape
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSR(
                    (op(other.data, self.data), self.indices, self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = asarray(other)
        if other.size == 1:
            return CSR(
                (op(other, self.data), self.indices, self.indptr),
                shape=self.shape
            )
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR(
                (op(other, self.data), self.indices, self.indptr),
                shape=self.shape
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __mul__(self, other: jax.Array | Quantity) -> CSR:
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: jax.Array | Quantity) -> CSR:
        return self._binary_rop(other, operator.mul)

    def __div__(self, other: jax.Array | Quantity) -> CSR:
        return self._binary_op(other, operator.truediv)

    def __rdiv__(self, other: jax.Array | Quantity) -> CSR:
        return self._binary_rop(other, operator.truediv)

    def __truediv__(self, other) -> CSR:
        return self.__div__(other)

    def __rtruediv__(self, other) -> CSR:
        return self.__rdiv__(other)

    def __add__(self, other) -> CSR:
        return self._binary_op(other, operator.add)

    def __radd__(self, other) -> CSR:
        return self._binary_rop(other, operator.add)

    def __sub__(self, other) -> CSR:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other) -> CSR:
        return self._binary_rop(other, operator.sub)

    def __mod__(self, other) -> CSR:
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other) -> CSR:
        return self._binary_rop(other, operator.mod)

    def __matmul__(self, other):
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        other = asarray(other)
        data, other = promote_dtypes(self.data, other)
        if other.ndim == 1:
            return _csr_matvec(data, self.indices, self.indptr, other, shape=self.shape)
        elif other.ndim == 2:
            return _csr_matmat(data, self.indices, self.indptr, other, shape=self.shape)
        else:
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        other = asarray(other)
        data, other = promote_dtypes(self.data, other)
        if other.ndim == 1:
            return _csr_matvec(data, self.indices, self.indptr, other, shape=self.shape, transpose=True)
        elif other.ndim == 2:
            other = other.T
            r = _csr_matmat(data, self.indices, self.indptr, other, shape=self.shape, transpose=True)
            return r.T
        else:
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def tree_flatten(self):
        return (self.data,), {"shape": self.shape, "indices": self.indices, "indptr": self.indptr}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.data, = children
        if aux_data.keys() != {'shape', 'indices', 'indptr'}:
            raise ValueError(f"CSR.tree_unflatten: invalid {aux_data=}")
        obj.__dict__.update(**aux_data)
        return obj


@tree_util.register_pytree_node_class
class CSC(SparseMatrix):
    """
    Unit-aware CSC matrix.
    """
    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    shape: tuple[int, int]
    nse = property(lambda self: self.data.size)
    dtype = property(lambda self: self.data.dtype)

    __array_priority__ = 2000

    def __init__(self, args, *, shape):
        self.data, self.indices, self.indptr = map(asarray, args)
        super().__init__(args, shape=shape)

    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=np.int32):
        if nse is None:
            nse = (get_mantissa(mat) != 0).sum()
        return csr_fromdense(mat.T, nse=nse, index_dtype=index_dtype).T

    @classmethod
    def _empty(cls, shape, *, dtype=None, index_dtype='int32'):
        """Create an empty CSC instance. Public method is sparse.empty()."""
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError(f"CSC must have ndim=2; got {shape=}")
        data = jnp.empty(0, dtype)
        indices = jnp.empty(0, index_dtype)
        indptr = jnp.zeros(shape[1] + 1, index_dtype)
        return cls((data, indices, indptr), shape=shape)

    @classmethod
    def _eye(cls, N, M, k, *, dtype=None, index_dtype='int32'):
        return CSR._eye(M, N, -k, dtype=dtype, index_dtype=index_dtype).T

    def with_data(self, data: jax.Array | Quantity) -> CSC:
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert get_unit(data) == get_unit(self.data)
        return CSC((data, self.indices, self.indptr), shape=self.shape)

    def todense(self):
        return csr_todense(self.T).T

    def transpose(self, axes=None):
        assert axes is None
        return CSR((self.data, self.indices, self.indptr), shape=self.shape[::-1])

    def __abs__(self):
        return CSC((abs(self.data), self.indices, self.indptr), shape=self.shape)

    def __neg__(self):
        return CSC((-self.data, self.indices, self.indptr), shape=self.shape)

    def __pos__(self):
        return CSC((self.data.__pos__(), self.indices, self.indptr), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, CSC):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSC(
                    (op(self.data, other.data), self.indices, self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = asarray(other)
        if other.size == 1:
            return CSC(
                (op(self.data, other), self.indices, self.indptr),
                shape=self.shape
            )
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC(
                (op(self.data, other), self.indices, self.indptr),
                shape=self.shape
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, CSC):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSC(
                    (op(other.data, self.data), self.indices, self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = asarray(other)
        if other.size == 1:
            return CSC(
                (op(other, self.data), self.indices, self.indptr),
                shape=self.shape
            )
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC(
                (op(other, self.data), self.indices, self.indptr),
                shape=self.shape
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __mul__(self, other: jax.Array | Quantity) -> CSC:
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: jax.Array | Quantity) -> CSC:
        return self._binary_rop(other, operator.mul)

    def __div__(self, other: jax.Array | Quantity) -> CSC:
        return self._binary_op(other, operator.truediv)

    def __rdiv__(self, other: jax.Array | Quantity) -> CSC:
        return self._binary_rop(other, operator.truediv)

    def __truediv__(self, other) -> CSC:
        return self.__div__(other)

    def __rtruediv__(self, other) -> CSC:
        return self.__rdiv__(other)

    def __add__(self, other) -> CSC:
        return self._binary_op(other, operator.add)

    def __radd__(self, other) -> CSC:
        return self._binary_rop(other, operator.add)

    def __sub__(self, other) -> CSC:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other) -> CSC:
        return self._binary_rop(other, operator.sub)

    def __mod__(self, other) -> CSC:
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other) -> CSC:
        return self._binary_rop(other, operator.mod)

    def __matmul__(self, other):
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        other = asarray(other)
        data, other = promote_dtypes(self.data, other)
        if other.ndim == 1:
            return _csr_matvec(
                data,
                self.indices,
                self.indptr,
                other,
                shape=self.shape[::-1],
                transpose=True
            )
        elif other.ndim == 2:
            return _csr_matmat(
                data,
                self.indices,
                self.indptr,
                other,
                shape=self.shape[::-1],
                transpose=True
            )
        else:
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        other = asarray(other)
        data, other = promote_dtypes(self.data, other)
        if other.ndim == 1:
            return _csr_matvec(data, self.indices, self.indptr, other,
                               shape=self.shape[::-1], transpose=False)
        elif other.ndim == 2:
            other = other.T
            r = _csr_matmat(data, self.indices, self.indptr, other,
                            shape=self.shape[::-1], transpose=False)
            return r.T
        else:
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def tree_flatten(self):
        return (self.data,), {"shape": self.shape, "indices": self.indices, "indptr": self.indptr}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.data, = children
        if aux_data.keys() != {'shape', 'indices', 'indptr'}:
            raise ValueError(f"CSR.tree_unflatten: invalid {aux_data=}")
        obj.__dict__.update(**aux_data)
        return obj


Data = Union[jax.Array, Quantity]
Indices = jax.Array
Indptr = jax.Array


def csr_fromdense(
    mat: jax.Array | Quantity,
    *, nse: int | None = None,
    index_dtype: jax.typing.DTypeLike = np.int32
) -> CSR:
    """Create a CSR-format sparse matrix from a dense matrix.

    Args:
      mat : array to be converted to CSR.
      nse : number of specified entries in ``mat``. If not specified,
        it will be computed from the input matrix.
      index_dtype : dtype of sparse indices

    Returns:
      mat_coo : CSR representation of the matrix.
    """
    if nse is None:
        nse = int((get_mantissa(mat) != 0).sum())
    nse_int = jax.core.concrete_or_error(operator.index, nse, "coo_fromdense nse argument")
    return CSR(_csr_fromdense(mat, nse=nse_int, index_dtype=index_dtype), shape=mat.shape)


def csr_todense(mat: CSR) -> jax.Array | Quantity:
    """Convert a CSR-format sparse matrix to a dense matrix.

    Args:
      mat : CSR matrix
    Returns:
      mat_dense: dense version of ``mat``
    """
    assert isinstance(mat, CSR), f"Expected CSR, got {type(mat)}"
    return _csr_todense(mat.data, mat.indices, mat.indptr, shape=mat.shape)


def csc_todense(mat: CSC) -> jax.Array | Quantity:
    """Convert a CSR-format sparse matrix to a dense matrix.

    Args:
      mat : CSR matrix
    Returns:
      mat_dense: dense version of ``mat``
    """
    assert isinstance(mat, CSC), f"Expected CSC, got {type(mat)}"
    return mat.todense()


def csc_fromdense(
    mat: jax.Array | Quantity,
    *,
    nse: int | None = None,
    index_dtype: jax.typing.DTypeLike = np.int32
) -> CSC:
    assert nse is None, "nse argument is not supported for CSC"
    return CSC.fromdense(mat, nse=nse, index_dtype=index_dtype)


def _csr_fromdense(
    mat: jax.Array | Quantity,
    *,
    nse: int,
    index_dtype: jax.typing.DTypeLike = np.int32
) -> Tuple[Data, Indices, Indptr]:
    """Create CSR-format sparse matrix from a dense matrix.

    Args:
      mat : array to be converted to CSR.
      nse : number of specified entries in ``mat``
      index_dtype : dtype of sparse indices

    Returns:
      data : array of shape ``(nse,)`` and dtype ``mat.dtype``.
      indices : array of shape ``(nse,)`` and dtype ``index_dtype``
      indptr : array of shape ``(mat.shape[0] + 1,)`` and dtype ``index_dtype``
    """
    mat = asarray(mat)
    mat, unit = split_mantissa_unit(mat)
    nse = jax.core.concrete_or_error(operator.index, nse, "nse argument of csr_fromdense()")
    r = csr_fromdense_p.bind(mat, nse=nse, index_dtype=np.dtype(index_dtype))
    if unit.is_unitless:
        return r
    else:
        return maybe_decimal(r[0] * unit), r[1], r[2]


def _csr_todense(
    data: jax.Array | Quantity,
    indices: jax.Array,
    indptr: jax.Array, *,
    shape: Shape
) -> jax.Array:
    """Convert CSR-format sparse matrix to a dense matrix.

    Args:
      data : array of shape ``(nse,)``.
      indices : array of shape ``(nse,)``
      indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
      shape : length-2 tuple representing the matrix shape

    Returns:
      mat : array with specified shape and dtype matching ``data``
    """
    data, unit = split_mantissa_unit(data)
    mat = csr_todense_p.bind(data, indices, indptr, shape=shape)
    return maybe_decimal(mat * unit)


def _csr_matvec(
    data: jax.Array | Quantity,
    indices: jax.Array,
    indptr: jax.Array,
    v: jax.Array | Quantity,
    *,
    shape: Shape,
    transpose: bool = False
) -> jax.Array | Quantity:
    """Product of CSR sparse matrix and a dense vector.

    Args:
      data : array of shape ``(nse,)``.
      indices : array of shape ``(nse,)``
      indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
      v : array of shape ``(shape[0] if transpose else shape[1],)``
        and dtype ``data.dtype``
      shape : length-2 tuple representing the matrix shape
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      y : array of shape ``(shape[1] if transpose else shape[0],)`` representing
        the matrix vector product.
    """
    data, unitd = split_mantissa_unit(data)
    v, unitv = split_mantissa_unit(v)
    res = csr_matvec_p.bind(data, indices, indptr, v, shape=shape, transpose=transpose)
    return maybe_decimal(res * unitd * unitv)


def _csr_matmat(
    data: jax.Array | Quantity,
    indices: jax.Array,
    indptr: jax.Array,
    B: jax.Array | Quantity,
    *,
    shape: Shape,
    transpose: bool = False
) -> jax.Array | Quantity:
    """Product of CSR sparse matrix and a dense matrix.

    Args:
      data : array of shape ``(nse,)``.
      indices : array of shape ``(nse,)``
      indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
      B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
        dtype ``data.dtype``
      shape : length-2 tuple representing the matrix shape
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      C : array of shape ``(shape[1] if transpose else shape[0], cols)``
        representing the matrix-matrix product.
    """
    data, unitd = split_mantissa_unit(data)
    B, unitb = split_mantissa_unit(B)
    res = csr_matmat_p.bind(data, indices, indptr, B, shape=shape, transpose=transpose)
    return maybe_decimal(res * unitd * unitb)


@jax.jit
def _csr_to_coo(indices: jax.Array, indptr: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Given CSR (indices, indptr) return COO (row, col)"""
    return jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1, indices
