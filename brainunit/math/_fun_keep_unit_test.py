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

import jax.numpy as jnp
import pytest
from absl.testing import parameterized

import brainunit as bu
import brainunit.math as bm
from brainunit import second, meter, ms
from brainunit._base import assert_quantity

fun_keep_unit_squence_inputs = [
    'row_stack', 'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack', 'block', 'append',
]
fun_keep_unit_squence_outputs = [
    'split', 'array_split', 'dsplit', 'hsplit', 'vsplit',
]
fun_keep_unit_broadcasting_arrays = [
    'atleast_1d', 'atleast_2d', 'atleast_3d', 'broadcast_arrays',
]
fun_keep_unit_array_manipulation = [
    'reshape', 'moveaxis', 'transpose', 'swapaxes', 'tile', 'repeat',
    'flip', 'fliplr', 'flipud', 'roll', 'expand_dims', 'squeeze',
    'sort', 'max', 'min', 'amax', 'amin', 'diagflat', 'diagonal', 'choose', 'ravel',
    'flatten', 'unflatten', 'remove_diag',
]
fun_keep_unit_selection = [
    'compress', 'extract', 'take', 'select', 'where', 'unique',
]
fun_keep_unit_math_other = [
    'interp', 'clip', 'histogram',
]
fun_keep_unit_math_unary = [
    'real', 'imag', 'conj', 'conjugate', 'negative', 'positive',
    'abs', 'sum', 'nancumsum', 'nansum',
    'cumsum', 'ediff1d', 'absolute', 'fabs', 'median',
    'nanmin', 'nanmax', 'ptp', 'average', 'mean', 'std',
    'nanmedian', 'nanmean', 'nanstd', 'diff', 'nan_to_num',
]

fun_accept_unitless_unary_can_return_quantity = [
    'round', 'around', 'rint',
    'floor', 'ceil', 'trunc', 'fix',
]
fun_keep_unit_math_binary = [
    'fmod', 'mod', 'remainder',
    'maximum', 'minimum', 'fmax', 'fmin',
    'add', 'subtract', 'nextafter',
]
fun_keep_unit_percentile = [
    'percentile', 'nanpercentile',
]
fun_keep_unit_quantile = [
    'quantile', 'nanquantile',
]
fun_keep_unit_math_unary_misc = [
    'trace', 'lcm', 'gcd', 'copysign', 'rot90', 'intersect1d',
]
fun_accept_unitless_unary_2_results = [
    'modf',
]


class TestFunKeepUnitSquenceInputs(parameterized.TestCase):
    def test_row_stack(self):
        a = jnp.array([1, 2, 3])
        b = jnp.array([4, 5, 6])
        result = bu.math.row_stack((a, b))
        self.assertTrue(jnp.all(result == jnp.vstack((a, b))))

        q1 = [1, 2, 3] * bu.second
        q2 = [4, 5, 6] * bu.second
        result_q = bu.math.row_stack((q1, q2))
        expected_q = jnp.vstack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
        assert_quantity(result_q, expected_q, bu.second)

    def test_concatenate(self):
        a = jnp.array([[1, 2], [3, 4]])
        b = jnp.array([[5, 6]])
        result = bu.math.concatenate((a, b), axis=0)
        self.assertTrue(jnp.all(result == jnp.concatenate((a, b), axis=0)))

        q1 = [[1, 2], [3, 4]] * bu.second
        q2 = [[5, 6]] * bu.second
        result_q = bu.math.concatenate((q1, q2), axis=0)
        expected_q = jnp.concatenate((jnp.array([[1, 2], [3, 4]]), jnp.array([[5, 6]])), axis=0)
        assert_quantity(result_q, expected_q, bu.second)

    def test_stack(self):
        a = jnp.array([1, 2, 3])
        b = jnp.array([4, 5, 6])
        result = bu.math.stack((a, b), axis=1)
        self.assertTrue(jnp.all(result == jnp.stack((a, b), axis=1)))

        q1 = [1, 2, 3] * bu.second
        q2 = [4, 5, 6] * bu.second
        result_q = bu.math.stack((q1, q2), axis=1)
        expected_q = jnp.stack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])), axis=1)
        assert_quantity(result_q, expected_q, bu.second)

    def test_vstack(self):
        a = jnp.array([1, 2, 3])
        b = jnp.array([4, 5, 6])
        result = bu.math.vstack((a, b))
        self.assertTrue(jnp.all(result == jnp.vstack((a, b))))

        q1 = [1, 2, 3] * bu.second
        q2 = [4, 5, 6] * bu.second
        result_q = bu.math.vstack((q1, q2))
        expected_q = jnp.vstack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
        assert_quantity(result_q, expected_q, bu.second)

    def test_hstack(self):
        a = jnp.array((1, 2, 3))
        b = jnp.array((4, 5, 6))
        result = bu.math.hstack((a, b))
        self.assertTrue(jnp.all(result == jnp.hstack((a, b))))

        q1 = [1, 2, 3] * bu.second
        q2 = [4, 5, 6] * bu.second
        result_q = bu.math.hstack((q1, q2))
        expected_q = jnp.hstack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
        assert_quantity(result_q, expected_q, bu.second)

    def test_dstack(self):
        a = jnp.array([[1], [2], [3]])
        b = jnp.array([[4], [5], [6]])
        result = bu.math.dstack((a, b))
        self.assertTrue(jnp.all(result == jnp.dstack((a, b))))

        q1 = [[1], [2], [3]] * bu.second
        q2 = [[4], [5], [6]] * bu.second
        result_q = bu.math.dstack((q1, q2))
        expected_q = jnp.dstack((jnp.array([[1], [2], [3]]), jnp.array([[4], [5], [6]])))
        assert_quantity(result_q, expected_q, bu.second)

    def test_column_stack(self):
        a = jnp.array((1, 2, 3))
        b = jnp.array((4, 5, 6))
        result = bu.math.column_stack((a, b))
        self.assertTrue(jnp.all(result == jnp.column_stack((a, b))))

        q1 = [1, 2, 3] * bu.second
        q2 = [4, 5, 6] * bu.second
        result_q = bu.math.column_stack((q1, q2))
        expected_q = jnp.column_stack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
        assert_quantity(result_q, expected_q, bu.second)

    def test_block(self):
        array = jnp.array([[1, 2], [3, 4]])
        result = bu.math.block(array)
        self.assertTrue(jnp.all(result == jnp.block(array)))

        q = [[1, 2], [3, 4]] * bu.second
        result_q = bu.math.block(q)
        expected_q = jnp.block(jnp.array([[1, 2], [3, 4]]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_append(self):
        array = jnp.array([0, 1, 2])
        result = bu.math.append(array, 3)
        self.assertTrue(jnp.all(result == jnp.append(array, 3)))

        q = [0, 1, 2] * bu.second
        result_q = bu.math.append(q, 3 * bu.second)
        expected_q = jnp.append(jnp.array([0, 1, 2]), 3)
        assert_quantity(result_q, expected_q, bu.second)


class TestFunKeepUnitSquenceOutputs(parameterized.TestCase):
    def test_split(self):
        array = jnp.arange(9)
        result = bu.math.split(array, 3)
        expected = jnp.split(array, 3)
        for r, e in zip(result, expected):
            self.assertTrue(jnp.all(r == e))

        q = jnp.arange(9) * bu.second
        result_q = bu.math.split(q, 3)
        expected_q = jnp.split(jnp.arange(9), 3)
        for r, e in zip(result_q, expected_q):
            assert_quantity(r, e, bu.second)

    def test_array_split(self):
        array = jnp.arange(9)
        result = bu.math.array_split(array, 3)
        expected = jnp.array_split(array, 3)
        for r, e in zip(result, expected):
            self.assertTrue(jnp.all(r == e))

        q = jnp.arange(9) * bu.second
        result_q = bu.math.array_split(q, 3)
        expected_q = jnp.array_split(jnp.arange(9), 3)
        for r, e in zip(result_q, expected_q):
            assert_quantity(r, e, bu.second)

    def test_dsplit(self):
        array = jnp.arange(16.0).reshape(2, 2, 4)
        result = bu.math.dsplit(array, 2)
        expected = jnp.dsplit(array, 2)
        for r, e in zip(result, expected):
            self.assertTrue(jnp.all(r == e))

        q = jnp.arange(16.0).reshape(2, 2, 4) * bu.second
        result_q = bu.math.dsplit(q, 2)
        expected_q = jnp.dsplit(jnp.arange(16.0).reshape(2, 2, 4), 2)
        for r, e in zip(result_q, expected_q):
            assert_quantity(r, e, bu.second)

    def test_hsplit(self):
        array = jnp.arange(16.0).reshape(4, 4)
        result = bu.math.hsplit(array, 2)
        expected = jnp.hsplit(array, 2)
        for r, e in zip(result, expected):
            self.assertTrue(jnp.all(r == e))

        q = jnp.arange(16.0).reshape(4, 4) * bu.second
        result_q = bu.math.hsplit(q, 2)
        expected_q = jnp.hsplit(jnp.arange(16.0).reshape(4, 4), 2)
        for r, e in zip(result_q, expected_q):
            assert_quantity(r, e, bu.second)

    def test_vsplit(self):
        array = jnp.arange(16.0).reshape(4, 4)
        result = bu.math.vsplit(array, 2)
        expected = jnp.vsplit(array, 2)
        for r, e in zip(result, expected):
            self.assertTrue(jnp.all(r == e))

        q = jnp.arange(16.0).reshape(4, 4) * bu.second
        result_q = bu.math.vsplit(q, 2)
        expected_q = jnp.vsplit(jnp.arange(16.0).reshape(4, 4), 2)
        for r, e in zip(result_q, expected_q):
            assert_quantity(r, e, bu.second)


class TestFunKeepUnitBroadcastingArrays(parameterized.TestCase):
    def test_atleast_1d(self):
        array = jnp.array(0)
        result = bu.math.atleast_1d(array)
        self.assertTrue(jnp.all(result == jnp.atleast_1d(array)))

        q = 0 * bu.second
        result_q = bu.math.atleast_1d(q)
        expected_q = jnp.atleast_1d(jnp.array(0))
        assert_quantity(result_q, expected_q, bu.second)

    def test_atleast_2d(self):
        array = jnp.array([0, 1, 2])
        result = bu.math.atleast_2d(array)
        self.assertTrue(jnp.all(result == jnp.atleast_2d(array)))

        q = [0, 1, 2] * bu.second
        result_q = bu.math.atleast_2d(q)
        expected_q = jnp.atleast_2d(jnp.array([0, 1, 2]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_atleast_3d(self):
        array = jnp.array([[0, 1, 2], [3, 4, 5]])
        result = bu.math.atleast_3d(array)
        self.assertTrue(jnp.all(result == jnp.atleast_3d(array)))

        q = [[0, 1, 2], [3, 4, 5]] * bu.second
        result_q = bu.math.atleast_3d(q)
        expected_q = jnp.atleast_3d(jnp.array([[0, 1, 2], [3, 4, 5]]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_broadcast_arrays(self):
        a = jnp.array([1, 2, 3])
        b = jnp.array([[4], [5]])
        result = bu.math.broadcast_arrays(a, b)
        self.assertTrue(jnp.all(result[0] == jnp.broadcast_arrays(a, b)[0]))
        self.assertTrue(jnp.all(result[1] == jnp.broadcast_arrays(a, b)[1]))

        q1 = [1, 2, 3] * bu.second
        q2 = [[4], [5]] * bu.second
        result_q = bu.math.broadcast_arrays(q1, q2)
        expected_q = jnp.broadcast_arrays(jnp.array([1, 2, 3]), jnp.array([[4], [5]]))
        for r, e in zip(result_q, expected_q):
            assert_quantity(r, e, bu.second)


class TestFunKeepUnitArrayManipulation(parameterized.TestCase):
    def test_reshape(self):
        array = jnp.array([1, 2, 3, 4])
        result = bu.math.reshape(array, (2, 2))
        self.assertTrue(jnp.all(result == jnp.reshape(array, (2, 2))))

        q = [1, 2, 3, 4] * bu.second
        result_q = bu.math.reshape(q, (2, 2))
        expected_q = jnp.reshape(jnp.array([1, 2, 3, 4]), (2, 2))
        assert_quantity(result_q, expected_q, bu.second)

    def test_moveaxis(self):
        array = jnp.zeros((3, 4, 5))
        result = bu.math.moveaxis(array, 0, -1)
        self.assertTrue(jnp.all(result == jnp.moveaxis(array, 0, -1)))

        q = jnp.zeros((3, 4, 5)) * bu.second
        result_q = bu.math.moveaxis(q, 0, -1)
        expected_q = jnp.moveaxis(jnp.zeros((3, 4, 5)), 0, -1)
        assert_quantity(result_q, expected_q, bu.second)

    def test_transpose(self):
        array = jnp.ones((2, 3))
        result = bu.math.transpose(array)
        self.assertTrue(jnp.all(result == jnp.transpose(array)))

        q = jnp.ones((2, 3)) * bu.second
        result_q = bu.math.transpose(q)
        expected_q = jnp.transpose(jnp.ones((2, 3)))
        assert_quantity(result_q, expected_q, bu.second)

    def test_swapaxes(self):
        array = jnp.zeros((3, 4, 5))
        result = bu.math.swapaxes(array, 0, 2)
        self.assertTrue(jnp.all(result == jnp.swapaxes(array, 0, 2)))

        q = jnp.zeros((3, 4, 5)) * bu.second
        result_q = bu.math.swapaxes(q, 0, 2)
        expected_q = jnp.swapaxes(jnp.zeros((3, 4, 5)), 0, 2)
        assert_quantity(result_q, expected_q, bu.second)

    def test_tile(self):
        array = jnp.array([0, 1, 2])
        result = bu.math.tile(array, 2)
        self.assertTrue(jnp.all(result == jnp.tile(array, 2)))

        q = jnp.array([0, 1, 2]) * bu.second
        result_q = bu.math.tile(q, 2)
        expected_q = jnp.tile(jnp.array([0, 1, 2]), 2)
        assert_quantity(result_q, expected_q, bu.second)

    def test_repeat(self):
        array = jnp.array([0, 1, 2])
        result = bu.math.repeat(array, 2)
        self.assertTrue(jnp.all(result == jnp.repeat(array, 2)))

        q = [0, 1, 2] * bu.second
        result_q = bu.math.repeat(q, 2)
        expected_q = jnp.repeat(jnp.array([0, 1, 2]), 2)
        assert_quantity(result_q, expected_q, bu.second)

    def test_flip(self):
        array = jnp.array([0, 1, 2])
        result = bu.math.flip(array)
        self.assertTrue(jnp.all(result == jnp.flip(array)))

        q = [0, 1, 2] * bu.second
        result_q = bu.math.flip(q)
        expected_q = jnp.flip(jnp.array([0, 1, 2]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_fliplr(self):
        array = jnp.array([[0, 1, 2], [3, 4, 5]])
        result = bu.math.fliplr(array)
        self.assertTrue(jnp.all(result == jnp.fliplr(array)))

        q = [[0, 1, 2], [3, 4, 5]] * bu.second
        result_q = bu.math.fliplr(q)
        expected_q = jnp.fliplr(jnp.array([[0, 1, 2], [3, 4, 5]]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_flipud(self):
        array = jnp.array([[0, 1, 2], [3, 4, 5]])
        result = bu.math.flipud(array)
        self.assertTrue(jnp.all(result == jnp.flipud(array)))

        q = [[0, 1, 2], [3, 4, 5]] * bu.second
        result_q = bu.math.flipud(q)
        expected_q = jnp.flipud(jnp.array([[0, 1, 2], [3, 4, 5]]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_roll(self):
        array = jnp.array([0, 1, 2])
        result = bu.math.roll(array, 1)
        self.assertTrue(jnp.all(result == jnp.roll(array, 1)))

        q = [0, 1, 2] * bu.second
        result_q = bu.math.roll(q, 1)
        expected_q = jnp.roll(jnp.array([0, 1, 2]), 1)
        assert_quantity(result_q, expected_q, bu.second)

    def test_expand_dims(self):
        array = jnp.array([1, 2, 3])
        result = bu.math.expand_dims(array, axis=0)
        self.assertTrue(jnp.all(result == jnp.expand_dims(array, axis=0)))

        q = [1, 2, 3] * bu.second
        result_q = bu.math.expand_dims(q, axis=0)
        expected_q = jnp.expand_dims(jnp.array([1, 2, 3]), axis=0)
        assert_quantity(result_q, expected_q, bu.second)

    def test_squeeze(self):
        array = jnp.array([[[0], [1], [2]]])
        result = bu.math.squeeze(array)
        self.assertTrue(jnp.all(result == jnp.squeeze(array)))

        q = [[[0], [1], [2]]] * bu.second
        result_q = bu.math.squeeze(q)
        expected_q = jnp.squeeze(jnp.array([[[0], [1], [2]]]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_sort(self):
        array = jnp.array([2, 3, 1])
        result = bu.math.sort(array)
        self.assertTrue(jnp.all(result == jnp.sort(array)))

        q = [2, 3, 1] * bu.second
        result_q = bu.math.sort(q)
        expected_q = jnp.sort(jnp.array([2, 3, 1]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_max(self):
        array = jnp.array([1, 2, 3])
        result = bu.math.max(array)
        self.assertTrue(result == jnp.max(array))

        q = [1, 2, 3] * bu.second
        result_q = bu.math.max(q)
        expected_q = jnp.max(jnp.array([1, 2, 3]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_min(self):
        array = jnp.array([1, 2, 3])
        result = bu.math.min(array)
        self.assertTrue(result == jnp.min(array))

        q = [1, 2, 3] * bu.second
        result_q = bu.math.min(q)
        expected_q = jnp.min(jnp.array([1, 2, 3]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_amin(self):
        array = jnp.array([1, 2, 3])
        result = bu.math.amin(array)
        self.assertTrue(result == jnp.min(array))

        q = [1, 2, 3] * bu.second
        result_q = bu.math.amin(q)
        expected_q = jnp.min(jnp.array([1, 2, 3]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_amax(self):
        array = jnp.array([1, 2, 3])
        result = bu.math.amax(array)
        self.assertTrue(result == jnp.max(array))

        q = [1, 2, 3] * bu.second
        result_q = bu.math.amax(q)
        expected_q = jnp.max(jnp.array([1, 2, 3]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_diagflat(self):
        array = jnp.array([1, 2, 3])
        result = bu.math.diagflat(array)
        self.assertTrue(jnp.all(result == jnp.diagflat(array)))

        q = [1, 2, 3] * bu.second
        result_q = bu.math.diagflat(q)
        expected_q = jnp.diagflat(jnp.array([1, 2, 3]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_diagonal(self):
        array = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        result = bu.math.diagonal(array)
        self.assertTrue(jnp.all(result == jnp.diagonal(array)))

        q = [[0, 1, 2], [3, 4, 5], [6, 7, 8]] * bu.second
        result_q = bu.math.diagonal(q)
        expected_q = jnp.diagonal(jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_choose(self):
        choices = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6]), jnp.array([7, 8, 9])]
        result = bu.math.choose(jnp.array([0, 1, 2]), choices)
        self.assertTrue(jnp.all(result == jnp.choose(jnp.array([0, 1, 2]), choices)))

        q = [0, 1, 2] * bu.second
        q = q.astype(jnp.int64)
        result_q = bu.math.choose(q, choices)
        expected_q = jnp.choose(jnp.array([0, 1, 2]), choices)
        assert_quantity(result_q, expected_q, bu.second)

    def test_ravel(self):
        array = jnp.array([[1, 2, 3], [4, 5, 6]])
        result = bu.math.ravel(array)
        self.assertTrue(jnp.all(result == jnp.ravel(array)))

        q = [[1, 2, 3], [4, 5, 6]] * bu.second
        result_q = bu.math.ravel(q)
        expected_q = jnp.ravel(jnp.array([[1, 2, 3], [4, 5, 6]]))
        assert_quantity(result_q, expected_q, bu.second)


class TestFunKeepUnitSelection(parameterized.TestCase):
    def test_compress(self):
        array = jnp.array([1, 2, 3, 4])
        result = bu.math.compress(jnp.array([0, 1, 1, 0]), array)
        self.assertTrue(jnp.all(result == jnp.compress(jnp.array([0, 1, 1, 0]), array)))

        q = jnp.array([1, 2, 3, 4]) * bu.second
        result_q = bu.math.compress(jnp.array([0, 1, 1, 0]), q)
        expected_q = jnp.compress(jnp.array([0, 1, 1, 0]), q.mantissa)
        assert_quantity(result_q, expected_q, bu.second)

    def test_extract(self):
        array = jnp.array([1, 2, 3])
        result = bu.math.extract(array > 1, array)
        self.assertTrue(jnp.all(result == jnp.extract(array > 1, array)))

        q = jnp.array([1, 2, 3])
        a = array * bu.second
        result_q = bu.math.extract(q > 1, a)
        expected_q = jnp.extract(q > 1, jnp.array([1, 2, 3])) * bu.second
        assert bu.math.allclose(result_q, expected_q)

    def test_take(self):
        array = jnp.array([4, 3, 5, 7, 6, 8])
        indices = jnp.array([0, 1, 4])
        result = bu.math.take(array, indices)
        self.assertTrue(jnp.all(result == jnp.take(array, indices)))

        q = [4, 3, 5, 7, 6, 8] * bu.second
        i = jnp.array([0, 1, 4])
        result_q = bu.math.take(q, i)
        expected_q = jnp.take(jnp.array([4, 3, 5, 7, 6, 8]), jnp.array([0, 1, 4]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_select(self):
        condlist = [jnp.array([True, False, True]), jnp.array([False, True, False])]
        choicelist = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6])]
        result = bu.math.select(condlist, choicelist, default=0)
        self.assertTrue(jnp.all(result == jnp.select(condlist, choicelist, default=0)))

        c = [jnp.array([True, False, True]), jnp.array([False, True, False])]
        ch = [[1, 2, 3] * bu.second, [4, 5, 6] * bu.second]
        result_q = bu.math.select(c, ch, default=0)
        expected_q = jnp.select([jnp.array([True, False, True]), jnp.array([False, True, False])],
                                [jnp.array([1, 2, 3]), jnp.array([4, 5, 6])], default=0)
        assert_quantity(result_q, expected_q, bu.second)

    def test_where(self):
        array = jnp.array([1, 2, 3, 4, 5])
        result = bu.math.where(array > 2, array, 0)
        self.assertTrue(jnp.all(result == jnp.where(array > 2, array, 0)))

        q = [1, 2, 3, 4, 5] * bu.second
        result_q = bu.math.where(q > 2 * bu.second, q, 0 * bu.second)
        expected_q = jnp.where(jnp.array([1, 2, 3, 4, 5]) > 2, jnp.array([1, 2, 3, 4, 5]), 0)
        assert_quantity(result_q, expected_q, bu.second)

    def test_unique(self):
        array = jnp.array([0, 1, 2, 1, 0])
        result = bu.math.unique(array)
        self.assertTrue(jnp.all(result == jnp.unique(array)))

        q = [0, 1, 2, 1, 0] * bu.second
        result_q = bu.math.unique(q)
        expected_q = jnp.unique(jnp.array([0, 1, 2, 1, 0]))
        assert_quantity(result_q, expected_q, bu.second)


class TestFunKeepUnitOther(parameterized.TestCase):
    def test_interp(self):
        x = jnp.array([1, 2, 3])
        xp = jnp.array([0, 1, 2, 3, 4])
        fp = jnp.array([0, 1, 2, 3, 4])
        result = bu.math.interp(x, xp, fp)
        self.assertTrue(jnp.all(result == jnp.interp(x, xp, fp)))

        x = [1, 2, 3] * bu.second
        xp = [0, 1, 2, 3, 4] * bu.second
        fp = [0, 1, 2, 3, 4] * bu.second
        result_q = bu.math.interp(x, xp, fp)
        expected_q = jnp.interp(jnp.array([1, 2, 3]), jnp.array([0, 1, 2, 3, 4]),
                                jnp.array([0, 1, 2, 3, 4])) * bu.second
        assert bu.math.allclose(result_q, expected_q)

    def test_clip(self):
        array = jnp.array([1, 2, 3, 4, 5])
        result = bu.math.clip(array, 2, 4)
        self.assertTrue(jnp.all(result == jnp.clip(array, 2, 4)))

        q = [1, 2, 3, 4, 5] * bu.ms
        result_q = bu.math.clip(q, 2 * bu.ms, 4 * bu.ms)
        expected_q = jnp.clip(jnp.array([1, 2, 3, 4, 5]), 2, 4) * bu.ms
        assert bu.math.allclose(result_q, expected_q)

    def test_histogram(self):
        array = jnp.array([1, 2, 1])
        result, _ = bu.math.histogram(array)
        expected, _ = jnp.histogram(array)
        self.assertTrue(jnp.all(result == expected))

        q = [1, 2, 1] * bu.second
        result_q, _ = bu.math.histogram(q)
        expected_q, _ = jnp.histogram(jnp.array([1, 2, 1]))
        assert_quantity(result_q, expected_q, None)


class TestFunKeepUnit(parameterized.TestCase):

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[second, meter]
    )
    def test_fun_keep_unit_math_unary(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_keep_unit_math_unary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_keep_unit_math_unary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected, unit=unit)
    @parameterized.product(
        value=[((1.0, 2.0), (3.0, 4.0)),
               ((1.23, 2.34, 3.45), (4.56, 5.67, 6.78))],
        unit=[second, meter]
    )
    def test_fun_keep_unit_math_binary(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_keep_unit_math_binary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_keep_unit_math_binary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            x1, x2 = value

            result = bm_fun(jnp.array(x1), jnp.array(x2))
            expected = jnp_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            q1 = x1 * unit
            q2 = x2 * unit
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected, unit=unit)

            with pytest.raises(AssertionError):
                result = bm_fun(q1, jnp.array(x2))

            with pytest.raises(AssertionError):
                result = bm_fun(jnp.array(x1), q2)

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, jnp.nan, 3.45)],
        q=[25, 50, 75],
        unit=[second, meter]
    )
    def test_fun_keep_unit_percentile(self, value, q, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_keep_unit_percentile]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_keep_unit_percentile]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value), q)
            expected = jnp_fun(jnp.array(value), q)
            assert_quantity(result, expected)

            q_value = value * unit
            result = bm_fun(q_value, q)
            expected = jnp_fun(jnp.array(value), q)
            assert_quantity(result, expected, unit=unit)

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, jnp.nan, 3.45)],
        q=[0.25, 0.5, 0.75],
        unit=[second, meter]
    )
    def test_fun_keep_unit_quantile(self, value, q, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_keep_unit_percentile]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_keep_unit_percentile]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value), q)
            expected = jnp_fun(jnp.array(value), q)
            assert_quantity(result, expected)

            q_value = value * unit
            result = bm_fun(q_value, q)
            expected = jnp_fun(jnp.array(value), q)
            assert_quantity(result, expected, unit=unit)

    @parameterized.product(
        value=[(1.123, 2.567, 3.891), (1.23, 2.34, 3.45)]
    )
    def test_fun_accept_unitless_binary_2_results(self, value):
        bm_fun_list = [getattr(bm, fun) for fun in fun_accept_unitless_unary_2_results]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_accept_unitless_unary_2_results]

        for fun in fun_accept_unitless_unary_2_results:
            bm_fun = getattr(bm, fun)
            jnp_fun = getattr(jnp, fun)

            print(f'fun: {bm_fun.__name__}')
            result1, result2 = bm_fun(jnp.array(value))
            expected1, expected2 = jnp_fun(jnp.array(value))
            assert_quantity(result1, expected1)
            assert_quantity(result2, expected2)

            for unit in [meter, ms]:
                q = value * unit
                result1, result2 = bm_fun(q)
                expected1, expected2 = jnp_fun(jnp.array(value))
                assert_quantity(result1, expected1, unit)
                assert_quantity(result2, expected2, unit)

    @parameterized.product(
        value=[(1.123, 2.567, 3.891), (1.23, 2.34, 3.45)]
    )
    def test_fun_accept_unitless_unary_can_return_quantity(self, value):
        for fun in fun_accept_unitless_unary_can_return_quantity:
            bm_fun = getattr(bm, fun)
            jnp_fun = getattr(jnp, fun)

            print(f'fun: {bm_fun.__name__}')
            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            for unit in [meter, ms]:
                q = value * unit
                result = bm_fun(q)
                expected = jnp_fun(jnp.array(value))
                assert_quantity(result, expected, unit)


class TestFunKeepUnitMathFunMisc(parameterized.TestCase):
    def test_trace(self):
        a = jnp.array([[1, 2], [3, 4]])
        result = bu.math.trace(a)
        self.assertTrue(result == jnp.trace(a))

        q = [[1, 2], [3, 4]] * bu.second
        result_q = bu.math.trace(q)
        expected_q = jnp.trace(jnp.array([[1, 2], [3, 4]]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_lcm(self):
        result = bu.math.lcm(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))
        self.assertTrue(jnp.all(result == jnp.lcm(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))))

        q1 = [4, 5, 6] * bu.second
        q2 = [2, 3, 4] * bu.second
        q1 = q1.astype(jnp.int64)
        q2 = q2.astype(jnp.int64)
        result_q = bu.math.lcm(q1, q2)
        expected_q = jnp.lcm(jnp.array([4, 5, 6]), jnp.array([2, 3, 4])) * bu.second
        assert bu.math.allclose(result_q, expected_q)

    def test_gcd(self):
        result = bu.math.gcd(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))
        self.assertTrue(jnp.all(result == jnp.gcd(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))))

        q1 = [4, 5, 6] * bu.second
        q2 = [2, 3, 4] * bu.second
        q1 = q1.astype(jnp.int64)
        q2 = q2.astype(jnp.int64)
        result_q = bu.math.gcd(q1, q2)
        expected_q = jnp.gcd(jnp.array([4, 5, 6]), jnp.array([2, 3, 4])) * bu.second
        assert bu.math.allclose(result_q, expected_q)

    def test_copysign(self):
        result = bu.math.copysign(jnp.array([-1, 2]), jnp.array([1, -3]))
        self.assertTrue(jnp.all(result == jnp.copysign(jnp.array([-1, 2]), jnp.array([1, -3]))))

        q1 = [-1, 2] * ms
        q2 = [1, -3] * ms
        result_q = bu.math.copysign(q1, q2)
        expected_q = jnp.copysign(jnp.array([-1, 2]), jnp.array([1, -3])) * ms
        assert bu.math.allclose(result_q, expected_q)

    def test_rot90(self):
        a = jnp.array([[1, 2], [3, 4]])
        result = bu.math.rot90(a)
        self.assertTrue(jnp.all(result == jnp.rot90(a)))

        q = [[1, 2], [3, 4]] * bu.second
        result_q = bu.math.rot90(q)
        expected_q = jnp.rot90(jnp.array([[1, 2], [3, 4]]))
        assert_quantity(result_q, expected_q, bu.second)

    def test_intersect1d(self):
        a = jnp.array([1, 2, 3, 4, 5])
        b = jnp.array([3, 4, 5, 6, 7])
        result = bu.math.intersect1d(a, b)
        self.assertTrue(jnp.all(result == jnp.intersect1d(a, b)))

        q1 = [1, 2, 3, 4, 5] * bu.second
        q2 = [3, 4, 5, 6, 7] * bu.second
        result_q = bu.math.intersect1d(q1, q2)
        expected_q = jnp.intersect1d(jnp.array([1, 2, 3, 4, 5]), jnp.array([3, 4, 5, 6, 7]))
        assert_quantity(result_q, expected_q, bu.second)
