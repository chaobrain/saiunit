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
import sys
import unittest

import brainstate as bst
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

import brainunit as u
import brainunit.lax as ulax
from brainunit import meter, second
from brainunit._base import assert_quantity

lax_array_manipulation = [
    'slice', 'dynamic_slice', 'dynamic_update_slice', 'gather',
    'index_take', 'slice_in_dim', 'index_in_dim', 'dynamic_slice_ind_dim', 'dynamic_index_in_dim',
    'dynamic_update_slice_in_dim', 'dynamic_update_index_in_dim',
    'sort', 'sort_key_val',
]

lax_keep_unit_unary = [
    'neg',
    'cummax', 'cummin', 'cumsum',
]

lax_keep_unit_binary = [
    'sub', 'complex',
]
lax_keep_unit_nary = [
    'clamp',
]

lax_type_conversion = [
    'convert_element_type', 'bitcast_convert_type',
]

lax_keep_unit_return_Quantity_index = [
    'approx_max_k', 'approx_min_k', 'top_k',
]

lax_broadcasting_arrays = [
    'broadcast', 'broadcast_in_dim', 'broadcast_to_rank',
]


class TestLaxKeepUnitArrayManipulation(parameterized.TestCase):
    @parameterized.product(
        [
            dict(
                shape=shape, starts=indices, limits=limit_indices, strides=strides
            )
            for shape, indices, limit_indices, strides in [
            [(3,), (1,), (2,), None],
            [(7,), (4,), (7,), None],
            [(5,), (1,), (5,), (2,)],
            [(8,), (1,), (6,), (2,)],
            [(5, 3), (1, 1), (3, 2), None],
            [(5, 3), (1, 1), (3, 1), None],
            [(7, 5, 3), (4, 0, 1), (7, 1, 3), None],
            [(5, 3), (1, 1), (2, 1), (1, 1)],
            [(5, 3), (1, 1), (5, 3), (2, 1)],
        ]
        ],
    )
    def test_slice(self, shape, starts, limits, strides):
        array = bst.random.random(shape)
        result = ulax.slice(array, start_indices=starts, limit_indices=limits, strides=strides)
        expected = lax.slice(array, start_indices=starts, limit_indices=limits, strides=strides)
        self.assertTrue(jnp.all(result == expected))

        array = array * u.second
        result_q = ulax.slice(array, start_indices=starts, limit_indices=limits, strides=strides)
        assert_quantity(result_q, expected, u.second)

    @parameterized.product(
        [
            dict(shape=shape, indices=indices, size_indices=size_indices)
            for shape, indices, size_indices in [
            [(3,), np.array((1,)), (1,)],
            [(5, 3), (1, 1), (3, 1)],
            [(5, 3), np.array((1, 1)), (3, 1)],
            [(7, 5, 3), np.array((4, 1, 0)), (2, 0, 1)],
        ]
        ],
    )
    def test_dynamic_slice(self, shape, indices, size_indices):
        array = bst.random.random(shape)
        result = ulax.dynamic_slice(array, start_indices=indices, slice_sizes=size_indices)
        expected = lax.dynamic_slice(array, start_indices=indices, slice_sizes=size_indices)
        self.assertTrue(jnp.all(result == expected))

        array = array * u.second
        result_q = ulax.dynamic_slice(array, start_indices=indices, slice_sizes=size_indices)
        assert_quantity(result_q, expected, u.second)

    @parameterized.product(
        [
            dict(shape=shape, indices=indices, update_shape=update_shape)
            for shape, indices, update_shape in [
            [(3,), (1,), (1,)],
            [(5, 3), (1, 1), (3, 1)],
            [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
        ]
        ],
    )
    def test_dynamic_update_slice(self, shape, indices, update_shape):
        array = bst.random.random(shape)
        start_indices = bst.random.random_integers(indices)
        update = bst.random.random(update_shape)
        result = ulax.dynamic_update_slice(array, start_indices=start_indices, update=update)
        expected = lax.dynamic_update_slice(array, start_indices=start_indices, update=update)
        self.assertTrue(jnp.all(result == expected))

        array = array * u.second
        with pytest.raises(AssertionError):
            result_q = ulax.dynamic_update_slice(array, start_indices=start_indices, update=update)
        update = update * u.second
        result_q = ulax.dynamic_update_slice(array, start_indices=start_indices, update=update)
        assert_quantity(result_q, expected, u.second)


    @parameterized.product(
        [dict(shape=shape, idxs=idxs, dnums=dnums, slice_sizes=slice_sizes)
         for shape, idxs, dnums, slice_sizes in [
             ((5,), np.array([[0], [2]]), lax.GatherDimensionNumbers(
                 offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
              (1,)),
             ((10,), np.array([[0], [0], [0]]), lax.GatherDimensionNumbers(
                 offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
              (2,)),
             ((10, 5,), np.array([[0], [2], [1]]), lax.GatherDimensionNumbers(
                 offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
              (1, 3)),
             ((10, 5), np.array([[0, 2], [1, 0]]), lax.GatherDimensionNumbers(
                 offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)),
              (1, 3)),
             ((2, 5), np.array([[[0], [2]], [[1], [1]]]),
              lax.GatherDimensionNumbers(
                  offset_dims=(), collapsed_slice_dims=(1,),
                  start_index_map=(1,), operand_batching_dims=(0,),
                  start_indices_batching_dims=(0,)),
              (1, 1)),
             ((2, 3, 10), np.array([[[0], [1]], [[2], [3]], [[4], [5]]]),
              lax.GatherDimensionNumbers(
                  offset_dims=(2,), collapsed_slice_dims=(),
                  start_index_map=(2,), operand_batching_dims=(0, 1),
                  start_indices_batching_dims=(1, 0)),
              (1, 1, 3))
         ]] if sys.version_info >= (3, 10) else [
            dict(shape=shape, idxs=idxs, dnums=dnums, slice_sizes=slice_sizes)
            for shape, idxs, dnums, slice_sizes in [
                 ((5,), np.array([[0], [2]]), lax.GatherDimensionNumbers(
                     offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)),
                  (1,)),
                 ((10,), np.array([[0], [0], [0]]), lax.GatherDimensionNumbers(
                     offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)),
                  (2,)),
                 ((10, 5,), np.array([[0], [2], [1]]), lax.GatherDimensionNumbers(
                     offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)),
                  (1, 3)),
                 ((10, 5), np.array([[0, 2], [1, 0]]), lax.GatherDimensionNumbers(
                     offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0, 1)),
                  (1, 3)),
         ]],
    )
    def test_gather(self, shape, idxs, dnums, slice_sizes):
        rand_idxs = bst.random.randint(0., high=max(shape), size=idxs.shape)
        array = bst.random.random(shape)

        result = ulax.gather(array, rand_idxs, dimension_numbers=dnums, slice_sizes=slice_sizes)
        expected = lax.gather(array, rand_idxs, dimension_numbers=dnums, slice_sizes=slice_sizes)
        self.assertTrue(jnp.all(result == expected))

        array = array * u.second
        result_q = ulax.gather(array, rand_idxs, dimension_numbers=dnums, slice_sizes=slice_sizes)
        assert_quantity(result_q, expected, u.second)

    @parameterized.product(
        [dict(shape=shape, idxs=idxs, axes=axes)
         for shape, idxs, axes in [
             [(3, 4, 5), (np.array([0, 2, 1]),), (0,)],
             [(3, 4, 5), (np.array([-1, -2]),), (0,)],
             [(3, 4, 5), (np.array([0, 2]), np.array([1, 3])), (0, 1)],
             [(3, 4, 5), (np.array([0, 2]), np.array([1, 3])), [0, 2]],
         ]],
    )
    def test_index_take(self, shape, idxs, axes):
        array = bst.random.random(shape)
        rand_idxs = jnp.array([bst.random.randint(e.shape) for e in idxs])

        result = ulax.index_take(array, idxs=rand_idxs, axes=axes)
        expected = lax.index_take(array, idxs=rand_idxs, axes=axes)
        self.assertTrue(jnp.all(result == expected))

        array = array * u.second
        result_q = ulax.index_take(array, idxs=rand_idxs, axes=axes)
        assert_quantity(result_q, expected, u.second)

    def test_slice_in_dim(self):
        array = jnp.array([[0, 1, 2],
                           [3, 4, 5],
                           [6, 7, 8],
                           [9, 10, 11]])
        start_index = 1
        limit_index = 3

        result = ulax.slice_in_dim(array, start_index=start_index, limit_index=limit_index)
        expected = lax.slice_in_dim(array, start_index=start_index, limit_index=limit_index)
        self.assertTrue(jnp.all(result == expected))

        array = array * u.second
        result_q = ulax.slice_in_dim(array, start_index=start_index, limit_index=limit_index)
        assert_quantity(result_q, expected, u.second)

    def test_index_in_dim(self):
        # TODO: No test in JAX
        ...

    def test_dynamic_slice_ind_dim(self):
        # TODO: No test in JAX
        ...

    def test_dynamic_index_in_dim(self):
        # TODO: No test in JAX
        ...

    def test_dynamic_update_slice_in_dim(self):
        x = jnp.ones((6, 7), jnp.int32)
        with self.assertRaises(TypeError):
            ulax.dynamic_update_slice_in_dim(x, jnp.ones((2, 7), jnp.int32),
                                             jnp.array([2, 2]), axis=0)

    def test_dynamic_update_index_in_dim(self):
        ...

    @parameterized.product(
        [dict(shape=shape, axis=axis)
         for shape in [(5,), (5, 7)] for axis in [-1, len(shape) - 1]],
        is_stable=[False, True],
    )
    def test_sort(self, shape, axis, is_stable):
        array = bst.random.random(shape)

        result = ulax.sort(array, dimension=axis, is_stable=is_stable)
        expected = lax.sort(array, dimension=axis, is_stable=is_stable)
        self.assertTrue(jnp.all(result == expected))

        array = array * u.second
        result_q = ulax.sort(array, dimension=axis, is_stable=is_stable)
        assert_quantity(result_q, expected, u.second)

    def test_sort_key_val(self):
        x = jnp.array([-np.inf, 0.0, -0.0, np.inf, np.nan, -np.nan])
        index = lax.iota(jnp.int64, x.size)

        result = ulax.sort_key_val(x, index, is_stable=True)[1]
        expected = lax.sort_key_val(x, index, is_stable=True)[1]
        self.assertTrue(jnp.all(result == expected))

        x = x * u.second
        result_q = ulax.sort_key_val(x, index, is_stable=True)[1]
        self.assertTrue(jnp.all(result_q == expected))


class TestLaxKeepUnit(parameterized.TestCase):

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[second, meter]
    )
    def test_lax_keep_unit_math_unary(self, value, unit):
        ulax_fun_list = [getattr(ulax, fun) for fun in lax_keep_unit_unary]
        lax_fun_list = [getattr(lax, fun) for fun in lax_keep_unit_unary]

        for ulax_fun, lax_fun in zip(ulax_fun_list, lax_fun_list):
            print(f'fun: {ulax_fun.__name__}')

            result = ulax_fun(jnp.array(value))
            expected = lax_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = ulax_fun(q)
            assert_quantity(result, expected, unit=unit)

    @parameterized.product(
        value=[((1.0, 2.0), (3.0, 4.0)),
               ((1.23, 2.34, 3.45), (4.56, 5.67, 6.78))],
        unit=[second, meter]
    )
    def test_lax_keep_unit_math_binary(self, value, unit):
        ulax_fun_list = [getattr(ulax, fun) for fun in lax_keep_unit_binary]
        lax_fun_list = [getattr(lax, fun) for fun in lax_keep_unit_binary]

        for ulax_fun, lax_fun in zip(ulax_fun_list, lax_fun_list):
            print(f'fun: {ulax_fun.__name__}')

            x1, x2 = value

            result = ulax_fun(jnp.array(x1), jnp.array(x2))
            expected = lax_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            q1 = x1 * unit
            q2 = x2 * unit
            result = ulax_fun(q1, q2)
            expected = lax_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected, unit=unit)

            with pytest.raises(AssertionError):
                result = ulax_fun(q1, jnp.array(x2))

            with pytest.raises(AssertionError):
                result = ulax_fun(jnp.array(x1), q2)

    @parameterized.product(
        [dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
              dnums=dnums)
         for arg_shape, idxs, update_shape, dnums in [
             ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
                 update_window_dims=(), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
             ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(),
                 scatter_dims_to_operand_dims=(0,))),
             ((10, 5), np.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
             ((2, 5), np.array([[[0], [2]], [[1], [1]]]), (2, 2),
              lax.ScatterDimensionNumbers(
                  update_window_dims=(), inserted_window_dims=(1,),
                  scatter_dims_to_operand_dims=(1,), operand_batching_dims=(0,),
                  scatter_indices_batching_dims=(0,))),
             ((2, 3, 10), np.array([[[0], [1]], [[2], [3]], [[4], [5]]]),
              (3, 2, 3), lax.ScatterDimensionNumbers(
                 update_window_dims=(2,), inserted_window_dims=(),
                 scatter_dims_to_operand_dims=(2,), operand_batching_dims=(0, 1),
                 scatter_indices_batching_dims=(1, 0)))
         ]] if sys.version_info >= (3, 10) else [
            dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
              dnums=dnums)
             for arg_shape, idxs, update_shape, dnums in [
                 ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
                     update_window_dims=(), inserted_window_dims=(0,),
                     scatter_dims_to_operand_dims=(0,))),
                 ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
                     update_window_dims=(1,), inserted_window_dims=(),
                     scatter_dims_to_operand_dims=(0,))),
                 ((10, 5), np.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
                     update_window_dims=(1,), inserted_window_dims=(0,),
                     scatter_dims_to_operand_dims=(0,))),
            ]]

    )
    def test_scatter(self, arg_shape, idxs, update_shape, dnums):
        array = bst.random.random(arg_shape)
        rand_idx = bst.random.randint(0, max(arg_shape), size=idxs.shape)
        update = bst.random.random(update_shape)

        result = ulax.scatter(array, rand_idx, update, dimension_numbers=dnums)
        expected = lax.scatter(array, rand_idx, update, dimension_numbers=dnums)
        self.assertTrue(jnp.all(result == expected))

        array = array * u.second
        with pytest.raises(AssertionError):
            result_q = ulax.scatter(array, rand_idx, update, dimension_numbers=dnums)
        update = update * u.second
        result_q = ulax.scatter(array, rand_idx, update, dimension_numbers=dnums)

        assert_quantity(result_q, expected, u.second)

    @parameterized.product(
        [dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
              dnums=dnums)
         for arg_shape, idxs, update_shape, dnums in [
             ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
                 update_window_dims=(), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
             ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(),
                 scatter_dims_to_operand_dims=(0,))),
             ((10, 5), np.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
             ((2, 5), np.array([[[0], [2]], [[1], [1]]]), (2, 2),
              lax.ScatterDimensionNumbers(
                  update_window_dims=(), inserted_window_dims=(1,),
                  scatter_dims_to_operand_dims=(1,), operand_batching_dims=(0,),
                  scatter_indices_batching_dims=(0,))),
             ((2, 3, 10), np.array([[[0], [1]], [[2], [3]], [[4], [5]]]),
              (3, 2, 3), lax.ScatterDimensionNumbers(
                 update_window_dims=(2,), inserted_window_dims=(),
                 scatter_dims_to_operand_dims=(2,), operand_batching_dims=(0, 1),
                 scatter_indices_batching_dims=(1, 0)))
         ]] if sys.version_info >= (3, 10) else [
            dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
              dnums=dnums)
             for arg_shape, idxs, update_shape, dnums in [
                 ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
                     update_window_dims=(), inserted_window_dims=(0,),
                     scatter_dims_to_operand_dims=(0,))),
                 ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
                     update_window_dims=(1,), inserted_window_dims=(),
                     scatter_dims_to_operand_dims=(0,))),
                 ((10, 5), np.array([[0], [2], [1]]), (3, 3), lax.ScatterDimensionNumbers(
                     update_window_dims=(1,), inserted_window_dims=(0,),
                     scatter_dims_to_operand_dims=(0,))),
             ]],

        mode=["clip", "fill", None],
        op=['scatter_add', 'scatter_sub'] if sys.version_info >= (3, 10) else ['scatter_add'],
    )
    def test_scatter_add_sub(self, arg_shape, idxs, update_shape, dnums, mode, op):
        ulax_op = getattr(ulax, op)
        lax_op = getattr(lax, op)

        array = bst.random.random(arg_shape)
        rand_idx = bst.random.randint(0, max(arg_shape), size=idxs.shape)
        update = bst.random.random(update_shape)

        result = ulax_op(array, rand_idx, update, dimension_numbers=dnums, mode=mode)
        expected = lax_op(array, rand_idx, update, dimension_numbers=dnums, mode=mode)
        self.assertTrue(jnp.all(result == expected))

        array = array * u.second
        with pytest.raises(AssertionError):
            result_q = ulax_op(array, rand_idx, update, dimension_numbers=dnums, mode=mode)
        update = update * u.second
        result_q = ulax_op(array, rand_idx, update, dimension_numbers=dnums, mode=mode)

        assert_quantity(result_q, expected, u.second)

    def test_scatter_mul(self):
        # TODO: no test in JAX
        ...

    @parameterized.product(
        [dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
              dnums=dnums)
         for arg_shape, idxs, update_shape, dnums in [
             ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
                 update_window_dims=(), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
             ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(),
                 scatter_dims_to_operand_dims=(0,))),
             ((10, 5), np.array([[0], [2], [1]], dtype=np.uint64), (3, 3), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
             ((2, 5), np.array([[[0], [2]], [[1], [1]]]), (2, 2),
              lax.ScatterDimensionNumbers(
                  update_window_dims=(), inserted_window_dims=(1,),
                  scatter_dims_to_operand_dims=(1,), operand_batching_dims=(0,),
                  scatter_indices_batching_dims=(0,))),
             ((2, 3, 10), np.array([[[0], [1]], [[2], [3]], [[4], [5]]]),
              (3, 2, 3), lax.ScatterDimensionNumbers(
                 update_window_dims=(2,), inserted_window_dims=(),
                 scatter_dims_to_operand_dims=(2,), operand_batching_dims=(0, 1),
                 scatter_indices_batching_dims=(1, 0)))
         ]] if sys.version_info >= (3, 10) else [dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
              dnums=dnums)
         for arg_shape, idxs, update_shape, dnums in [
             ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
                 update_window_dims=(), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
             ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(),
                 scatter_dims_to_operand_dims=(0,))),
             ((10, 5), np.array([[0], [2], [1]], dtype=np.uint64), (3, 3), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
         ]]
    )
    def test_scatter_min(self, arg_shape, idxs, update_shape, dnums):
        array = bst.random.random(arg_shape)
        rand_idx = bst.random.randint(0, max(arg_shape), size=idxs.shape)
        update = bst.random.random(update_shape)

        result = ulax.scatter_min(array, rand_idx, update, dimension_numbers=dnums)
        expected = lax.scatter_min(array, rand_idx, update, dimension_numbers=dnums)
        self.assertTrue(jnp.all(result == expected))

        array = array * u.second
        with pytest.raises(AssertionError):
            result_q = ulax.scatter_min(array, rand_idx, update, dimension_numbers=dnums)
        update = update * u.second
        result_q = ulax.scatter_min(array, rand_idx, update, dimension_numbers=dnums)

        assert_quantity(result_q, expected, u.second)

    @parameterized.product(
        [dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
              dnums=dnums)
         for arg_shape, idxs, update_shape, dnums in [
             ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
                 update_window_dims=(), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
             ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(),
                 scatter_dims_to_operand_dims=(0,))),
             ((10, 5), np.array([[0], [2], [1]], dtype=np.uint64), (3, 3), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
             ((2, 5), np.array([[[0], [2]], [[1], [1]]]), (2, 2),
              lax.ScatterDimensionNumbers(
                  update_window_dims=(), inserted_window_dims=(1,),
                  scatter_dims_to_operand_dims=(1,), operand_batching_dims=(0,),
                  scatter_indices_batching_dims=(0,))),
             ((2, 3, 10), np.array([[[0], [1]], [[2], [3]], [[4], [5]]]),
              (3, 2, 3), lax.ScatterDimensionNumbers(
                 update_window_dims=(2,), inserted_window_dims=(),
                 scatter_dims_to_operand_dims=(2,), operand_batching_dims=(0, 1),
                 scatter_indices_batching_dims=(1, 0)))
         ]] if sys.version_info >= (3, 10) else [
            dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
              dnums=dnums)
         for arg_shape, idxs, update_shape, dnums in [
             ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
                 update_window_dims=(), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
             ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(),
                 scatter_dims_to_operand_dims=(0,))),
             ((10, 5), np.array([[0], [2], [1]], dtype=np.uint64), (3, 3), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
         ]]
    )
    def test_scatter_max(self, arg_shape, idxs, update_shape, dnums):
        array = bst.random.random(arg_shape)
        rand_idx = bst.random.randint(0, max(arg_shape), size=idxs.shape)
        update = bst.random.random(update_shape)

        result = ulax.scatter_max(array, rand_idx, update, dimension_numbers=dnums)
        expected = lax.scatter_max(array, rand_idx, update, dimension_numbers=dnums)
        self.assertTrue(jnp.all(result == expected))

        array = array * u.second
        with pytest.raises(AssertionError):
            result_q = ulax.scatter_max(array, rand_idx, update, dimension_numbers=dnums)
        update = update * u.second
        result_q = ulax.scatter_max(array, rand_idx, update, dimension_numbers=dnums)

        assert_quantity(result_q, expected, u.second)

    @parameterized.product(
        [dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
              dnums=dnums)
         for arg_shape, idxs, update_shape, dnums in [
             ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
                 update_window_dims=(), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
             ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(),
                 scatter_dims_to_operand_dims=(0,))),
             ((10, 5), np.array([[0], [2], [1]], dtype=np.uint64), (3, 3), lax.ScatterDimensionNumbers(
                 update_window_dims=(1,), inserted_window_dims=(0,),
                 scatter_dims_to_operand_dims=(0,))),
             ((2, 5), np.array([[[0], [2]], [[1], [1]]]), (2, 2),
              lax.ScatterDimensionNumbers(
                  update_window_dims=(), inserted_window_dims=(1,),
                  scatter_dims_to_operand_dims=(1,), operand_batching_dims=(0,),
                  scatter_indices_batching_dims=(0,))),
             ((2, 3, 10), np.array([[[0], [1]], [[2], [3]], [[4], [5]]]),
              (3, 2, 3), lax.ScatterDimensionNumbers(
                 update_window_dims=(2,), inserted_window_dims=(),
                 scatter_dims_to_operand_dims=(2,), operand_batching_dims=(0, 1),
                 scatter_indices_batching_dims=(1, 0)))
         ]] if sys.version_info >= (3, 10) else [
            dict(arg_shape=arg_shape, idxs=idxs, update_shape=update_shape,
              dnums=dnums)
             for arg_shape, idxs, update_shape, dnums in [
                 ((5,), np.array([[0], [2]]), (2,), lax.ScatterDimensionNumbers(
                     update_window_dims=(), inserted_window_dims=(0,),
                     scatter_dims_to_operand_dims=(0,))),
                 ((10,), np.array([[0], [0], [0]]), (3, 2), lax.ScatterDimensionNumbers(
                     update_window_dims=(1,), inserted_window_dims=(),
                     scatter_dims_to_operand_dims=(0,))),
                 ((10, 5), np.array([[0], [2], [1]], dtype=np.uint64), (3, 3), lax.ScatterDimensionNumbers(
                     update_window_dims=(1,), inserted_window_dims=(0,),
                     scatter_dims_to_operand_dims=(0,))),
             ]]
    )
    def test_scatter_apply(self, arg_shape, idxs, update_shape, dnums):
        array = bst.random.random(arg_shape)
        rand_idx = bst.random.randint(0, max(arg_shape), size=idxs.shape)

        result = ulax.scatter_apply(array, rand_idx, jnp.sin, dimension_numbers=dnums, update_shape=update_shape)
        expected = lax.scatter_apply(array, rand_idx, jnp.sin, dimension_numbers=dnums, update_shape=update_shape)
        self.assertTrue(jnp.all(result == expected))

        array = array * u.second
        result_q = ulax.scatter_apply(array, rand_idx, jnp.sin, dimension_numbers=dnums, update_shape=update_shape)

        assert_quantity(result_q, expected, u.second)

    @parameterized.product(
        [dict(shape=shape, pads=pads) for shape, pads in [
            ((0, 2), [(1, 2, 1), (0, 1, 0)]),
            ((2, 3), [(1, 2, 1), (0, 1, 0)]),
            ((2,), [(1, 2, 0)]),
            ((1, 2), [(1, 2, 0), (3, 4, 0)]),
            ((1, 2), [(0, 0, 0), (0, 0, 0)]),
            ((2,), [(1, 2, 3), ]),
            ((3, 2), [(1, 2, 1), (3, 4, 2)]),
            ((2,), [(-1, 2, 0), ]),
            ((4, 2), [(-1, -2, 0), (1, 2, 0)]),
            ((4, 2), [(-1, 2, 0), (1, 2, 2)]),
            ((5,), [(-1, -2, 2), ]),
            ((4, 2), [(-1, -2, 1), (1, 2, 2)])
        ]
         ],
    )
    def test_pad(self, shape, pads):
        array = bst.random.random(shape)
        padding = np.array(0, dtype=array.dtype)

        result = ulax.pad(array, padding, pads)
        expected = lax.pad(array, padding, pads)
        self.assertTrue(jnp.all(result == expected))

        array = array * u.second
        with pytest.raises(AssertionError):
            result_q = ulax.pad(array, padding, pads)
        padding = padding * u.second
        result_q = ulax.pad(array, padding, pads)
        assert_quantity(result_q, expected, u.second)


class TestLaxKeepUnitNary(parameterized.TestCase):

    @parameterized.product(
        [dict(min_shape=min_shape, operand_shape=operand_shape, max_shape=max_shape)
         for min_shape, operand_shape, max_shape in [
             [(), (2, 3), ()],
             [(2, 3), (2, 3), ()],
             [(), (2, 3), (2, 3)],
             [(2, 3), (2, 3), (2, 3)],
         ]],
    )
    def test_clamp(self, min_shape, operand_shape, max_shape):
        array1 = bst.random.random(min_shape)
        array2 = bst.random.random(operand_shape)
        array3 = bst.random.random(max_shape)

        result = ulax.clamp(array1, array2, array3)
        expected = lax.clamp(array1, array2, array3)
        self.assertTrue(jnp.all(result == expected))

        array1 = array1 * u.second
        array2 = array2 * u.second
        with pytest.raises(AssertionError):
            result_q = ulax.clamp(array1, array2, array3)
        array3 = array3 * u.second
        result_q = ulax.clamp(array1, array2, array3)
        assert_quantity(result_q, expected, u.second)


class TestLaxTypeConversion(parameterized.TestCase):

    @parameterized.product(
        input_type=[int, float, np.int32, np.float32, np.array],
        dtype=[np.int32, np.float32],
        value=[0, 1],
    )
    def test_convert_element_type(self, input_type, dtype, value):
        ulax_op = lambda x: ulax.convert_element_type(x, dtype)
        lax_op = lambda x: lax.convert_element_type(x, dtype)

        result = ulax_op(input_type(value))
        expected = lax_op(input_type(value))
        self.assertTrue(jnp.all(result == expected))

        result_q = ulax_op(input_type(value) * u.second)
        assert_quantity(result_q, expected, u.second)

    def test_bitcast_convert_type(self):
        # TODO: dtypes.bit_width need the source code of JAX
        ...


def compute_recall(result_neighbors, ground_truth_neighbors) -> float:
    """Computes the recall of an approximate nearest neighbor search.

    Args:
      result_neighbors: int32 numpy array of the shape [num_queries,
        neighbors_per_query] where the values are the indices of the dataset.
      ground_truth_neighbors: int32 numpy array of with shape [num_queries,
        ground_truth_neighbors_per_query] where the values are the indices of the
        dataset.

    Returns:
      The recall.
    """
    assert len(
        result_neighbors.shape) == 2, "shape = [num_queries, neighbors_per_query]"
    assert len(ground_truth_neighbors.shape
               ) == 2, "shape = [num_queries, ground_truth_neighbors_per_query]"
    assert result_neighbors.shape[0] == ground_truth_neighbors.shape[0]
    gt_sets = [set(np.asarray(x)) for x in ground_truth_neighbors]
    hits = sum(len([x
                    for x in nn_per_q
                    if x.item() in gt_sets[q]])
               for q, nn_per_q in enumerate(result_neighbors))
    return hits / ground_truth_neighbors.size


class TestLaxKeepUnitReturnQuantityIndex(parameterized.TestCase):

    @parameterized.product(
        qy_shape=[(200, 128), (128, 128)],
        db_shape=[(128, 500), (128, 3000)],
        k=[1, 10],
        recall=[0.95],
    )
    def test_approx_max_k(self, qy_shape, db_shape, k, recall):
        qy = bst.random.random(qy_shape)
        db = bst.random.random(db_shape)
        scores = lax.dot(qy, db)
        _, results = ulax.approx_max_k(scores, k, recall_target=recall)
        _, expecteds = lax.approx_max_k(scores, k, recall_target=recall)
        for result, expected in zip(results, expecteds):
            self.assertTrue(jnp.all(result == expected))

        scores = scores * u.second
        _, results_q = ulax.approx_max_k(scores, k, recall_target=recall)
        for result, expected in zip(results_q, expecteds):
            self.assertTrue(jnp.all(result == expected))

    @parameterized.product(
        qy_shape=[(200, 128), (128, 128)],
        db_shape=[(128, 500), (128, 3000)],
        k=[1, 10],
        recall=[0.95],
    )
    def test_approx_min_k(self, qy_shape, db_shape, k, recall):
        qy = bst.random.random(qy_shape)
        db = bst.random.random(db_shape)
        scores = lax.dot(qy, db)
        _, results = ulax.approx_min_k(scores, k, recall_target=recall)
        _, expecteds = lax.approx_min_k(scores, k, recall_target=recall)
        for result, expected in zip(results, expecteds):
            self.assertTrue(jnp.all(result == expected))

        scores = scores * u.second
        _, results_q = ulax.approx_min_k(scores, k, recall_target=recall)
        for result, expected in zip(results_q, expecteds):
            self.assertTrue(jnp.all(result == expected))

    @parameterized.product(
        qy_shape=[(200, 128), (128, 128)],
        db_shape=[(128, 500), (128, 3000)],
        k=[1, 10],
        recall=[0.95],
    )
    def test_top_k(self, qy_shape, db_shape, k, recall):
        qy = bst.random.random(qy_shape)
        db = bst.random.random(db_shape)
        scores = lax.dot(qy, db)
        _, results = ulax.top_k(-scores, k)
        _, expecteds = lax.top_k(-scores, k)
        for result, expected in zip(results, expecteds):
            self.assertTrue(jnp.all(result == expected))

        scores = scores * u.second
        _, results_q = ulax.top_k(-scores, k)
        for result, expected in zip(results_q, expecteds):
            self.assertTrue(jnp.all(result == expected))


class TestLaxBroadcastingArrays(parameterized.TestCase):

    @parameterized.product(
        shape=[(), (2, 3)],
        broadcast_sizes=[(), (2,), (1, 2)],
    )
    def test_broadcast(self, shape, broadcast_sizes):
        x = bst.random.random(shape)

        result = ulax.broadcast(x, broadcast_sizes)
        expected = lax.broadcast(x, broadcast_sizes)
        self.assertTrue(jnp.all(result == expected))

        x = x * u.second
        result_q = ulax.broadcast(x, broadcast_sizes)
        assert_quantity(result_q, expected, u.second)

    @parameterized.product(
        [
            dict(inshape=inshape, outshape=outshape, dimensions=dimensions)
            for inshape, outshape, dimensions in [
            ([2], [2, 2], [0]),
            ([2], [2, 2], [1]),
            ([2], [2, 3], [0]),
            ([], [2, 3], []),
            ([1], [2, 3], [1]),
        ]
        ],
    )
    def test_broadcast_in_dim(self, inshape, outshape, dimensions):
        x = bst.random.random(inshape)
        result = ulax.broadcast_in_dim(x, outshape, dimensions)
        expected = lax.broadcast_in_dim(x, outshape, dimensions)
        self.assertTrue(jnp.all(result == expected))

        x = x * u.second
        result_q = ulax.broadcast_in_dim(x, outshape, dimensions)
        assert_quantity(result_q, expected, u.second)

    def test_broadcast_to_rank(self):
        # TODO: no test in JAX
        ...
