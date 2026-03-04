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

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from saiunit._base_getters import get_mantissa, get_unit, maybe_decimal
from saiunit._base_quantity import Quantity
from saiunit._sparse_base import SparseMatrix

__all__ = [
    'BlockELL',
]


@jax.tree_util.register_pytree_node_class
class BlockELL(SparseMatrix):
    """
    Unit-aware Block-ELL sparse matrix.
    """
    data: jax.Array | Quantity  # float32[n_blocks, *block_size]
    blocks_per_row: jax.Array  # int32[n_rows]
    indices: jax.Array  # int32[n_rows, max_num_blocks_per_row, 2]
    shape: tuple[int, int]  # (n_rows * block_size[0], n_cols * block_size[1])

    ndim: int = property(lambda self: len(self.shape))
    num_blocks = property(lambda self: self.data.shape[0])
    block_size = property(lambda self: self.data.shape[1:])
    dtype = property(lambda self: self.data.dtype)

    def __init__(self, args, *, shape):
        blocks, blocks_per_row, indices = args
        self.data = blocks
        self.blocks_per_row = blocks_per_row
        self.indices = indices
        super().__init__(args, shape=shape)

    def tree_flatten(self):
        return (self.data,), (self.blocks_per_row, self.indices, self.shape,)

    @classmethod
    def tree_unflatten(cls, data, xs):
        blocks, = xs
        blocks_per_row, indices, shape, = data
        return BlockELL((blocks, blocks_per_row, indices), shape=shape)

    def _validate(self):
        _nblocks, n, m = self.data.shape
        nrows = self.blocks_per_row.shape[0]
        assert self.indices.shape[0] == nrows
        assert len(self.shape) == 2
        assert self.shape[0] == n * nrows
        assert self.shape[1] % m == 0

    @jax.jit
    def todense(self) -> jax.Array:
        self._validate()
        return _sdd_todense(self)

    @classmethod
    def fromdense(cls, dense: jax.Array | Quantity, *, block_size) -> 'BlockELL':
        dense_mantissa = get_mantissa(dense)
        dense_unit = get_unit(dense)

        nrows, ncols = dense_mantissa.shape
        n, m = block_size
        if nrows % n != 0:
            raise ValueError(f"Row size {nrows} is not divisible by block row size {n}")
        if ncols % m != 0:
            raise ValueError(f"Column size {ncols} is not divisible by block column size {m}")
        nrows //= n
        ncols //= m

        blocks = []
        blocks_per_row = []
        indices_by_row = []
        global_block_idx = 0
        for i in range(nrows):
            row_indices = []
            for j in range(ncols):
                block = dense_mantissa[i * n:(i + 1) * n, j * m:(j + 1) * m]
                if not jnp.all(block == 0):
                    blocks.append(block)
                    row_indices.append([j, global_block_idx])
                    global_block_idx += 1
            blocks_per_row.append(len(row_indices))
            indices_by_row.append(row_indices)

        max_num_blocks_per_row = max(blocks_per_row, default=0)
        indices = np.zeros((nrows, max_num_blocks_per_row, 2), dtype=np.int32)
        for i, row_indices in enumerate(indices_by_row):
            for j, index_pair in enumerate(row_indices):
                indices[i, j] = index_pair

        if blocks:
            blocks = jnp.stack(blocks)
        else:
            blocks = jnp.empty((0, n, m), dtype=dense_mantissa.dtype)

        blocks = maybe_decimal(Quantity(blocks, unit=dense_unit))

        return cls(
            (blocks, jnp.asarray(blocks_per_row, dtype=jnp.int32), jnp.asarray(indices, dtype=jnp.int32)),
            shape=dense_mantissa.shape
        )

    def transpose(self, axes=None):
        if axes is not None:
            raise NotImplementedError("axes argument to transpose()")
        raise NotImplementedError("BlockELL.transpose is not implemented.")

    def __matmul__(self, other) -> jax.Array:
        self._validate()
        return sdd_matmul(self, other)


@jax.jit
def _sdd_todense(mat: BlockELL) -> jax.Array:
    _, n, m = mat.data.shape
    nrows = mat.shape[0] // n
    unit = get_unit(mat.data)
    blocks = get_mantissa(mat.data)
    out = jnp.zeros(mat.shape, mat.dtype)

    def i_body(i, val1):  # each row
        def j_body(j, val2):  # each block in the row
            i_col, i_block = mat.indices[i, j]
            val2 = jax.lax.dynamic_update_slice(val2, blocks[i_block], (i * n, i_col * m))
            return val2

        return jax.lax.fori_loop(0, mat.blocks_per_row[i], j_body, val1)

    out = jax.lax.fori_loop(0, nrows, i_body, out)
    return maybe_decimal(Quantity(out, unit=unit))


def _check_shape_consistency(x, y):
    y_mantissa = get_mantissa(y)
    if not isinstance(y_mantissa, jax.Array):
        raise TypeError(f"Only support array-like matmul inputs. Got {type(y)}")
    if x.ndim != 2 or y_mantissa.ndim != 2:
        raise ValueError(f"Expected rank-2 matmul inputs, got {x.ndim} and {y_mantissa.ndim}")
    if x.shape[1] != y_mantissa.shape[0]:
        raise ValueError(f"Dimension mismatch: {x.shape} @ {y_mantissa.shape}")


def _sdd_kernel(
    x_ref,  # [n_blocks, bm, bn]
    indices_ref,  # [n_rows, max_num_blocks_per_row, 2]
    blocks_per_row_ref,  # [n_rows]
    y_ref,  # [n, k]
    o_ref,  # [m, k]
    *,
    bm: int,
    bn: int,
    bk: int,
):
    i_m = pl.program_id(axis=0)
    i_k = pl.program_id(axis=1)
    n_block_this_row = blocks_per_row_ref[i_m]

    def body(k, val):
        i_x_col = indices_ref[i_m, k, 0]
        i_block = indices_ref[i_m, k, 1]
        # block = x_ref[i_block, ...]  # [bm, bn]
        # chunk = y_ref[i_x_col * bn:(i_x_col + 1) * bn, i_k * bk:(i_k + 1) * bk]  # [bn, bk]
        block = pl.load(x_ref, (i_block, pl.dslice(None), pl.dslice(None)))  # [bm, bn]
        chunk = pl.load(y_ref, (pl.dslice(i_x_col * bn, bn), pl.dslice(i_k * bk, bk)))  # [bn, bk]
        return val + jnp.dot(block, chunk).astype(o_ref.dtype)

    acc = jax.lax.fori_loop(0, n_block_this_row, body, jnp.zeros([bm, bk], dtype=o_ref.dtype))
    pl.store(o_ref, (pl.dslice(bm * i_m, bm), pl.dslice(bk * i_k, bk)), acc)  # [bm, bk]
    # o_ref[i_m * bm:(i_m + 1) * bm, i_k * bk:(i_k + 1) * bk] = acc


@functools.partial(jax.jit, static_argnames=["debug", 'interpret', 'block_size'])
def sdd_matmul(
    mat1: BlockELL,
    mat2: jax.Array | Quantity,
    *,
    debug: bool = False,
    interpret: bool = False,
    block_size: int = 256,
) -> jax.Array:
    _check_shape_consistency(mat1, mat2)

    mat2_mantissa = get_mantissa(mat2)

    # shape and dtype
    m, n, k = mat1.shape[0], mat1.shape[1], mat2_mantissa.shape[1]
    _, bm, bn = mat1.data.shape
    dtype = jnp.result_type(mat1.dtype, mat2_mantissa.dtype)

    # kernel
    fn = pl.pallas_call(
        functools.partial(_sdd_kernel, bm=bm, bn=bn, bk=block_size),
        out_shape=jax.ShapeDtypeStruct(shape=(m, k), dtype=dtype),
        grid=(pl.cdiv(m, bm), pl.cdiv(k, block_size)),
        debug=debug,
        interpret=interpret
    )

    # call
    unita = get_unit(mat1.data)
    unitb = get_unit(mat2)
    data = get_mantissa(mat1.data)
    r = fn(data, mat1.indices, mat1.blocks_per_row, mat2_mantissa)
    return maybe_decimal(Quantity(r, unit=unita * unitb))


@jax.jit
def native_sdd_matmul(
    mat1: BlockELL,
    mat2: jax.Array | Quantity,
):
    _check_shape_consistency(mat1, mat2)

    dtype = jnp.result_type(mat1.dtype, get_mantissa(mat2).dtype)
    _, n, m = mat1.data.shape
    nrows = mat1.shape[0] // n

    def i_body(i):  # each row
        num_blocks_in_row = mat1.blocks_per_row[i]

        def k_body(k, val):
            i_col, i_block = mat1.indices[i, k]
            chunk = jax.lax.dynamic_slice(mat2, [i_col * m, 0], (m, mat2.shape[1]))  # [m, mat2.shape[1]]
            block = data[i_block]
            return val + block.dot(chunk)

        acc = jax.lax.fori_loop(0, num_blocks_in_row, k_body, jnp.zeros((n, mat2.shape[1]), dtype=jnp.float32))
        return acc.astype(dtype)

    unita = get_unit(mat1.data)
    unitb = get_unit(mat2)
    data = get_mantissa(mat1.data)
    mat2 = get_mantissa(mat2)

    out = jax.vmap(i_body)(jnp.arange(nrows))
    out = out.reshape((mat1.shape[0], mat2.shape[1]))
    return maybe_decimal(Quantity(out, unit=unita * unitb))


def sample_sparse_matrix(
    m, n, bm, bn, *,
    sparse_prob=0.2,
    dtype=jnp.float32
) -> BlockELL:
    num_rows = m // bm  # number of rows in the Block-ELL matrix
    num_cols = n // bn  # number of columns in the Block-ELL matrix
    blocks_per_row = np.random.binomial(num_cols, sparse_prob,
                                        size=[num_rows])  # [n_rows], number of data in each row
    num_blocks = blocks_per_row.sum()
    blocks = np.random.randn(num_blocks, bm, bn).astype(dtype)  # [n_blocks, bm, bk], block values

    indices = []
    block_index = 0
    max_num_blocks = blocks_per_row.max(axis=0)
    for i in range(num_rows):
        row = []
        num_blocks_in_row = blocks_per_row[i]
        block_indices = np.sort(np.random.permutation(np.arange(num_cols))[:max_num_blocks])
        for j, b in zip(range(max_num_blocks), block_indices):
            if j < num_blocks_in_row:
                index = [b, block_index]
                block_index += 1
            else:
                index = [0, 0]
            row.append(index)
        indices.append(row)
    indices = jnp.array(indices)  # [n_rows, max_num_blocks_per_row, 2], block indices

    return BlockELL(
        (jnp.asarray(blocks), jnp.asarray(blocks_per_row), jnp.asarray(indices)),
        shape=(m, n)
    )
