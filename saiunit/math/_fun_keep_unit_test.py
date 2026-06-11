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

import jax.numpy as jnp
import pytest
from absl.testing import parameterized

import saiunit as u
import saiunit.math as um
from saiunit import second, meter, ms
from saiunit._base_getters import assert_quantity


class Array(u.CustomArray):
    def __init__(self, value):
        self.data = value


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


class TestFunKeepUnitWithArrayCustomArray(parameterized.TestCase):

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[second, meter]
    )
    def test_fun_keep_unit_math_unary_with_array(self, value, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_keep_unit_math_unary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_keep_unit_math_unary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            q = jnp.array(value) * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected, unit=unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=unit)

            array_input = Array(q)
            result = bm_fun(array_input.data)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=unit)

    @parameterized.product(
        value=[((1.0, 2.0), (3.0, 4.0)),
               ((1.23, 2.34, 3.45), (4.56, 5.67, 6.78))],
        unit=[second, meter]
    )
    def test_fun_keep_unit_math_binary_with_array(self, value, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_keep_unit_math_binary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_keep_unit_math_binary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            x1, x2 = value

            result = bm_fun(jnp.array(x1), jnp.array(x2))
            expected = jnp_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            q1 = jnp.array(x1) * unit
            q2 = jnp.array(x2) * unit
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected, unit=unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=unit)

            array_input1 = Array(q1)
            array_input2 = Array(q2)
            result = bm_fun(array_input1.data, array_input2.data)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=unit)

    def test_fun_keep_unit_array_manipulation_with_array(self):
        data = jnp.array([1.0, 2.0, 3.0, 4.0]) * meter
        test_array = Array(data)

        assert isinstance(test_array, u.CustomArray)
        assert hasattr(test_array, 'data')
        assert_quantity(test_array.data, jnp.array([1.0, 2.0, 3.0, 4.0]), unit=meter)

        reshape_result = um.reshape(test_array.data, (2, 2))
        reshape_array = Array(reshape_result)
        assert isinstance(reshape_array, u.CustomArray)
        assert_quantity(reshape_array.data, jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=meter)

        flip_result = um.flip(test_array.data)
        flip_array = Array(flip_result)
        assert isinstance(flip_array, u.CustomArray)
        assert_quantity(flip_array.data, jnp.array([4.0, 3.0, 2.0, 1.0]), unit=meter)

    def test_fun_keep_unit_sequence_operations_with_array(self):
        data1 = jnp.array([1.0, 2.0, 3.0]) * second
        data2 = jnp.array([4.0, 5.0, 6.0]) * second

        array1 = Array(data1)
        array2 = Array(data2)

        assert isinstance(array1, u.CustomArray)
        assert isinstance(array2, u.CustomArray)

        vstack_result = um.vstack((array1.data, array2.data))
        vstack_array = Array(vstack_result)
        assert isinstance(vstack_array, u.CustomArray)
        assert_quantity(vstack_array.data, jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), unit=second)

        hstack_result = um.hstack((array1.data, array2.data))
        hstack_array = Array(hstack_result)
        assert isinstance(hstack_array, u.CustomArray)
        assert_quantity(hstack_array.data, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), unit=second)

    def test_fun_keep_unit_selection_with_array(self):
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]) * meter
        test_array = Array(data)

        assert isinstance(test_array, u.CustomArray)

        where_result = um.where(test_array.data > 3.0 * meter, test_array.data, 0.0 * meter)
        where_array = Array(where_result)
        assert isinstance(where_array, u.CustomArray)
        assert_quantity(where_array.data, jnp.array([0.0, 0.0, 0.0, 4.0, 5.0]), unit=meter)

        sort_result = um.sort(test_array.data)
        sort_array = Array(sort_result)
        assert isinstance(sort_array, u.CustomArray)
        assert_quantity(sort_array.data, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), unit=meter)

    def test_fun_keep_unit_statistical_operations_with_array(self):
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]) * second
        test_array = Array(data)

        assert isinstance(test_array, u.CustomArray)

        sum_result = um.sum(test_array.data)
        sum_array = Array(sum_result)
        assert isinstance(sum_array, u.CustomArray)
        assert_quantity(sum_array.data, 15.0, unit=second)

        mean_result = um.mean(test_array.data)
        mean_array = Array(mean_result)
        assert isinstance(mean_array, u.CustomArray)
        assert_quantity(mean_array.data, 3.0, unit=second)

        max_result = um.max(test_array.data)
        max_array = Array(max_result)
        assert isinstance(max_array, u.CustomArray)
        assert_quantity(max_array.data, 5.0, unit=second)

        min_result = um.min(test_array.data)
        min_array = Array(min_result)
        assert isinstance(min_array, u.CustomArray)
        assert_quantity(min_array.data, 1.0, unit=second)

    def test_fun_keep_unit_percentile_quantile_with_array(self):
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]) * meter
        test_array = Array(data)

        assert isinstance(test_array, u.CustomArray)

        percentile_result = um.percentile(test_array.data, 50)
        percentile_array = Array(percentile_result)
        assert isinstance(percentile_array, u.CustomArray)
        assert_quantity(percentile_array.data, 3.0, unit=meter)

        quantile_result = um.quantile(test_array.data, 0.5)
        quantile_array = Array(quantile_result)
        assert isinstance(quantile_array, u.CustomArray)
        assert_quantity(quantile_array.data, 3.0, unit=meter)

    def test_fun_keep_unit_broadcasting_with_array(self):
        data = jnp.array([1.0, 2.0, 3.0]) * second
        test_array = Array(data)

        assert isinstance(test_array, u.CustomArray)

        atleast_2d_result = um.atleast_2d(test_array.data)
        atleast_2d_array = Array(atleast_2d_result)
        assert isinstance(atleast_2d_array, u.CustomArray)
        assert_quantity(atleast_2d_array.data, jnp.array([[1.0, 2.0, 3.0]]), unit=second)

        expand_dims_result = um.expand_dims(test_array.data, axis=0)
        expand_dims_array = Array(expand_dims_result)
        assert isinstance(expand_dims_array, u.CustomArray)
        assert_quantity(expand_dims_array.data, jnp.array([[1.0, 2.0, 3.0]]), unit=second)

    def test_fun_keep_unit_rounding_functions_with_array(self):
        data = jnp.array([1.2, 2.7, 3.1, 4.9]) * meter
        test_array = Array(data)

        assert isinstance(test_array, u.CustomArray)

        round_result = um.round(test_array.data)
        round_array = Array(round_result)
        assert isinstance(round_array, u.CustomArray)
        assert_quantity(round_array.data, jnp.array([1.0, 3.0, 3.0, 5.0]), unit=meter)

        floor_result = um.floor(test_array.data)
        floor_array = Array(floor_result)
        assert isinstance(floor_array, u.CustomArray)
        assert_quantity(floor_array.data, jnp.array([1.0, 2.0, 3.0, 4.0]), unit=meter)

        ceil_result = um.ceil(test_array.data)
        ceil_array = Array(ceil_result)
        assert isinstance(ceil_array, u.CustomArray)
        assert_quantity(ceil_array.data, jnp.array([2.0, 3.0, 4.0, 5.0]), unit=meter)

    def test_fun_keep_unit_complex_operations_with_array(self):
        real_data = jnp.array([1.0, 2.0, 3.0]) * second
        imag_data = jnp.array([4.0, 5.0, 6.0]) * second
        complex_data = real_data + 1j * imag_data

        test_array = Array(complex_data)
        assert isinstance(test_array, u.CustomArray)

        real_result = um.real(test_array.data)
        real_array = Array(real_result)
        assert isinstance(real_array, u.CustomArray)
        assert_quantity(real_array.data, jnp.array([1.0, 2.0, 3.0]), unit=second)

        imag_result = um.imag(test_array.data)
        imag_array = Array(imag_result)
        assert isinstance(imag_array, u.CustomArray)
        assert_quantity(imag_array.data, jnp.array([4.0, 5.0, 6.0]), unit=second)

        abs_result = um.abs(test_array.data)
        abs_array = Array(abs_result)
        assert isinstance(abs_array, u.CustomArray)
        expected_abs = jnp.sqrt(jnp.array([1.0, 2.0, 3.0]) ** 2 + jnp.array([4.0, 5.0, 6.0]) ** 2)
        assert_quantity(abs_array.data, expected_abs, unit=second)


class TestFunKeepUnitSquenceInputs(parameterized.TestCase):
    def test_row_stack(self):
        a = jnp.array([1, 2, 3])
        b = jnp.array([4, 5, 6])
        result = u.math.row_stack((a, b))
        self.assertTrue(jnp.all(result == jnp.vstack((a, b))))

        q1 = [1, 2, 3] * u.second
        q2 = [4, 5, 6] * u.second
        result_q = u.math.row_stack((q1, q2))
        expected_q = jnp.vstack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
        assert_quantity(result_q, expected_q, u.second)

    def test_concatenate(self):
        a = jnp.array([[1, 2], [3, 4]])
        b = jnp.array([[5, 6]])
        result = u.math.concatenate((a, b), axis=0)
        self.assertTrue(jnp.all(result == jnp.concatenate((a, b), axis=0)))

        q1 = [[1, 2], [3, 4]] * u.second
        q2 = [[5, 6]] * u.second
        result_q = u.math.concatenate((q1, q2), axis=0)
        expected_q = jnp.concatenate((jnp.array([[1, 2], [3, 4]]), jnp.array([[5, 6]])), axis=0)
        assert_quantity(result_q, expected_q, u.second)

    def test_stack(self):
        a = jnp.array([1, 2, 3])
        b = jnp.array([4, 5, 6])
        result = u.math.stack((a, b), axis=1)
        self.assertTrue(jnp.all(result == jnp.stack((a, b), axis=1)))

        q1 = [1, 2, 3] * u.second
        q2 = [4, 5, 6] * u.second
        result_q = u.math.stack((q1, q2), axis=1)
        expected_q = jnp.stack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])), axis=1)
        assert_quantity(result_q, expected_q, u.second)

    def test_vstack(self):
        a = jnp.array([1, 2, 3])
        b = jnp.array([4, 5, 6])
        result = u.math.vstack((a, b))
        self.assertTrue(jnp.all(result == jnp.vstack((a, b))))

        q1 = [1, 2, 3] * u.second
        q2 = [4, 5, 6] * u.second
        result_q = u.math.vstack((q1, q2))
        expected_q = jnp.vstack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
        assert_quantity(result_q, expected_q, u.second)

    def test_hstack(self):
        a = jnp.array((1, 2, 3))
        b = jnp.array((4, 5, 6))
        result = u.math.hstack((a, b))
        self.assertTrue(jnp.all(result == jnp.hstack((a, b))))

        q1 = [1, 2, 3] * u.second
        q2 = [4, 5, 6] * u.second
        result_q = u.math.hstack((q1, q2))
        expected_q = jnp.hstack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
        assert_quantity(result_q, expected_q, u.second)

    def test_dstack(self):
        a = jnp.array([[1], [2], [3]])
        b = jnp.array([[4], [5], [6]])
        result = u.math.dstack((a, b))
        self.assertTrue(jnp.all(result == jnp.dstack((a, b))))

        q1 = [[1], [2], [3]] * u.second
        q2 = [[4], [5], [6]] * u.second
        result_q = u.math.dstack((q1, q2))
        expected_q = jnp.dstack((jnp.array([[1], [2], [3]]), jnp.array([[4], [5], [6]])))
        assert_quantity(result_q, expected_q, u.second)

    def test_column_stack(self):
        a = jnp.array((1, 2, 3))
        b = jnp.array((4, 5, 6))
        result = u.math.column_stack((a, b))
        self.assertTrue(jnp.all(result == jnp.column_stack((a, b))))

        q1 = [1, 2, 3] * u.second
        q2 = [4, 5, 6] * u.second
        result_q = u.math.column_stack((q1, q2))
        expected_q = jnp.column_stack((jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))
        assert_quantity(result_q, expected_q, u.second)

    def test_block(self):
        array = jnp.array([[1, 2], [3, 4]])
        result = u.math.block(array)
        self.assertTrue(jnp.all(result == jnp.block(array)))

        q = [[1, 2], [3, 4]] * u.second
        result_q = u.math.block(q)
        expected_q = jnp.block(jnp.array([[1, 2], [3, 4]]))
        assert_quantity(result_q, expected_q, u.second)

    def test_append(self):
        array = jnp.array([0, 1, 2])
        result = u.math.append(array, 3)
        self.assertTrue(jnp.all(result == jnp.append(array, 3)))

        q = [0, 1, 2] * u.second
        result_q = u.math.append(q, 3 * u.second)
        expected_q = jnp.append(jnp.array([0, 1, 2]), 3)
        assert_quantity(result_q, expected_q, u.second)


class TestFunKeepUnitSquenceOutputs(parameterized.TestCase):
    def test_split(self):
        array = jnp.arange(9)
        result = u.math.split(array, 3)
        expected = jnp.split(array, 3)
        for r, e in zip(result, expected):
            self.assertTrue(jnp.all(r == e))

        q = jnp.arange(9) * u.second
        result_q = u.math.split(q, 3)
        expected_q = jnp.split(jnp.arange(9), 3)
        for r, e in zip(result_q, expected_q):
            assert_quantity(r, e, u.second)

    def test_array_split(self):
        array = jnp.arange(9)
        result = u.math.array_split(array, 3)
        expected = jnp.array_split(array, 3)
        for r, e in zip(result, expected):
            self.assertTrue(jnp.all(r == e))

        q = jnp.arange(9) * u.second
        result_q = u.math.array_split(q, 3)
        expected_q = jnp.array_split(jnp.arange(9), 3)
        for r, e in zip(result_q, expected_q):
            assert_quantity(r, e, u.second)

    def test_dsplit(self):
        array = jnp.arange(16.0).reshape(2, 2, 4)
        result = u.math.dsplit(array, 2)
        expected = jnp.dsplit(array, 2)
        for r, e in zip(result, expected):
            self.assertTrue(jnp.all(r == e))

        q = jnp.arange(16.0).reshape(2, 2, 4) * u.second
        result_q = u.math.dsplit(q, 2)
        expected_q = jnp.dsplit(jnp.arange(16.0).reshape(2, 2, 4), 2)
        for r, e in zip(result_q, expected_q):
            assert_quantity(r, e, u.second)

    def test_hsplit(self):
        array = jnp.arange(16.0).reshape(4, 4)
        result = u.math.hsplit(array, 2)
        expected = jnp.hsplit(array, 2)
        for r, e in zip(result, expected):
            self.assertTrue(jnp.all(r == e))

        q = jnp.arange(16.0).reshape(4, 4) * u.second
        result_q = u.math.hsplit(q, 2)
        expected_q = jnp.hsplit(jnp.arange(16.0).reshape(4, 4), 2)
        for r, e in zip(result_q, expected_q):
            assert_quantity(r, e, u.second)

    def test_vsplit(self):
        array = jnp.arange(16.0).reshape(4, 4)
        result = u.math.vsplit(array, 2)
        expected = jnp.vsplit(array, 2)
        for r, e in zip(result, expected):
            self.assertTrue(jnp.all(r == e))

        q = jnp.arange(16.0).reshape(4, 4) * u.second
        result_q = u.math.vsplit(q, 2)
        expected_q = jnp.vsplit(jnp.arange(16.0).reshape(4, 4), 2)
        for r, e in zip(result_q, expected_q):
            assert_quantity(r, e, u.second)


class TestFunKeepUnitBroadcastingArrays(parameterized.TestCase):
    def test_atleast_1d(self):
        array = jnp.array(0)
        result = u.math.atleast_1d(array)
        self.assertTrue(jnp.all(result == jnp.atleast_1d(array)))

        q = 0 * u.second
        result_q = u.math.atleast_1d(q)
        expected_q = jnp.atleast_1d(jnp.array(0))
        assert_quantity(result_q, expected_q, u.second)

    def test_atleast_2d(self):
        array = jnp.array([0, 1, 2])
        result = u.math.atleast_2d(array)
        self.assertTrue(jnp.all(result == jnp.atleast_2d(array)))

        q = [0, 1, 2] * u.second
        result_q = u.math.atleast_2d(q)
        expected_q = jnp.atleast_2d(jnp.array([0, 1, 2]))
        assert_quantity(result_q, expected_q, u.second)

    def test_atleast_3d(self):
        array = jnp.array([[0, 1, 2], [3, 4, 5]])
        result = u.math.atleast_3d(array)
        self.assertTrue(jnp.all(result == jnp.atleast_3d(array)))

        q = [[0, 1, 2], [3, 4, 5]] * u.second
        result_q = u.math.atleast_3d(q)
        expected_q = jnp.atleast_3d(jnp.array([[0, 1, 2], [3, 4, 5]]))
        assert_quantity(result_q, expected_q, u.second)

    def test_broadcast_arrays(self):
        a = jnp.array([1, 2, 3])
        b = jnp.array([[4], [5]])
        result = u.math.broadcast_arrays(a, b)
        self.assertTrue(jnp.all(result[0] == jnp.broadcast_arrays(a, b)[0]))
        self.assertTrue(jnp.all(result[1] == jnp.broadcast_arrays(a, b)[1]))

        q1 = [1, 2, 3] * u.second
        q2 = [[4], [5]] * u.second
        result_q = u.math.broadcast_arrays(q1, q2)
        expected_q = jnp.broadcast_arrays(jnp.array([1, 2, 3]), jnp.array([[4], [5]]))
        for r, e in zip(result_q, expected_q):
            assert_quantity(r, e, u.second)


class TestFunKeepUnitArrayManipulation(parameterized.TestCase):
    def test_reshape(self):
        array = jnp.array([1, 2, 3, 4])
        result = u.math.reshape(array, (2, 2))
        self.assertTrue(jnp.all(result == jnp.reshape(array, (2, 2))))

        q = [1, 2, 3, 4] * u.second
        result_q = u.math.reshape(q, (2, 2))
        expected_q = jnp.reshape(jnp.array([1, 2, 3, 4]), (2, 2))
        assert_quantity(result_q, expected_q, u.second)

    def test_moveaxis(self):
        array = jnp.zeros((3, 4, 5))
        result = u.math.moveaxis(array, 0, -1)
        self.assertTrue(jnp.all(result == jnp.moveaxis(array, 0, -1)))

        q = jnp.zeros((3, 4, 5)) * u.second
        result_q = u.math.moveaxis(q, 0, -1)
        expected_q = jnp.moveaxis(jnp.zeros((3, 4, 5)), 0, -1)
        assert_quantity(result_q, expected_q, u.second)

    def test_transpose(self):
        array = jnp.ones((2, 3))
        result = u.math.transpose(array)
        self.assertTrue(jnp.all(result == jnp.transpose(array)))

        q = jnp.ones((2, 3)) * u.second
        result_q = u.math.transpose(q)
        expected_q = jnp.transpose(jnp.ones((2, 3)))
        assert_quantity(result_q, expected_q, u.second)

    def test_swapaxes(self):
        array = jnp.zeros((3, 4, 5))
        result = u.math.swapaxes(array, 0, 2)
        self.assertTrue(jnp.all(result == jnp.swapaxes(array, 0, 2)))

        q = jnp.zeros((3, 4, 5)) * u.second
        result_q = u.math.swapaxes(q, 0, 2)
        expected_q = jnp.swapaxes(jnp.zeros((3, 4, 5)), 0, 2)
        assert_quantity(result_q, expected_q, u.second)

    def test_tile(self):
        array = jnp.array([0, 1, 2])
        result = u.math.tile(array, 2)
        self.assertTrue(jnp.all(result == jnp.tile(array, 2)))

        q = jnp.array([0, 1, 2]) * u.second
        result_q = u.math.tile(q, 2)
        expected_q = jnp.tile(jnp.array([0, 1, 2]), 2)
        assert_quantity(result_q, expected_q, u.second)

    def test_repeat(self):
        array = jnp.array([0, 1, 2])
        result = u.math.repeat(array, 2)
        self.assertTrue(jnp.all(result == jnp.repeat(array, 2)))

        q = [0, 1, 2] * u.second
        result_q = u.math.repeat(q, 2)
        expected_q = jnp.repeat(jnp.array([0, 1, 2]), 2)
        assert_quantity(result_q, expected_q, u.second)

    def test_flip(self):
        array = jnp.array([0, 1, 2])
        result = u.math.flip(array)
        self.assertTrue(jnp.all(result == jnp.flip(array)))

        q = [0, 1, 2] * u.second
        result_q = u.math.flip(q)
        expected_q = jnp.flip(jnp.array([0, 1, 2]))
        assert_quantity(result_q, expected_q, u.second)

    def test_fliplr(self):
        array = jnp.array([[0, 1, 2], [3, 4, 5]])
        result = u.math.fliplr(array)
        self.assertTrue(jnp.all(result == jnp.fliplr(array)))

        q = [[0, 1, 2], [3, 4, 5]] * u.second
        result_q = u.math.fliplr(q)
        expected_q = jnp.fliplr(jnp.array([[0, 1, 2], [3, 4, 5]]))
        assert_quantity(result_q, expected_q, u.second)

    def test_flipud(self):
        array = jnp.array([[0, 1, 2], [3, 4, 5]])
        result = u.math.flipud(array)
        self.assertTrue(jnp.all(result == jnp.flipud(array)))

        q = [[0, 1, 2], [3, 4, 5]] * u.second
        result_q = u.math.flipud(q)
        expected_q = jnp.flipud(jnp.array([[0, 1, 2], [3, 4, 5]]))
        assert_quantity(result_q, expected_q, u.second)

    def test_roll(self):
        array = jnp.array([0, 1, 2])
        result = u.math.roll(array, 1)
        self.assertTrue(jnp.all(result == jnp.roll(array, 1)))

        q = [0, 1, 2] * u.second
        result_q = u.math.roll(q, 1)
        expected_q = jnp.roll(jnp.array([0, 1, 2]), 1)
        assert_quantity(result_q, expected_q, u.second)

    def test_expand_dims(self):
        array = jnp.array([1, 2, 3])
        result = u.math.expand_dims(array, axis=0)
        self.assertTrue(jnp.all(result == jnp.expand_dims(array, axis=0)))

        q = [1, 2, 3] * u.second
        result_q = u.math.expand_dims(q, axis=0)
        expected_q = jnp.expand_dims(jnp.array([1, 2, 3]), axis=0)
        assert_quantity(result_q, expected_q, u.second)

    def test_squeeze(self):
        array = jnp.array([[[0], [1], [2]]])
        result = u.math.squeeze(array)
        self.assertTrue(jnp.all(result == jnp.squeeze(array)))

        q = [[[0], [1], [2]]] * u.second
        result_q = u.math.squeeze(q)
        expected_q = jnp.squeeze(jnp.array([[[0], [1], [2]]]))
        assert_quantity(result_q, expected_q, u.second)

    def test_sort(self):
        array = jnp.array([2, 3, 1])
        result = u.math.sort(array)
        self.assertTrue(jnp.all(result == jnp.sort(array)))

        q = [2, 3, 1] * u.second
        result_q = u.math.sort(q)
        expected_q = jnp.sort(jnp.array([2, 3, 1]))
        assert_quantity(result_q, expected_q, u.second)

    def test_max(self):
        array = jnp.array([1, 2, 3])
        result = u.math.max(array)
        self.assertTrue(result == jnp.max(array))

        q = [1, 2, 3] * u.second
        result_q = u.math.max(q)
        expected_q = jnp.max(jnp.array([1, 2, 3]))
        assert_quantity(result_q, expected_q, u.second)

    def test_min(self):
        array = jnp.array([1, 2, 3])
        result = u.math.min(array)
        self.assertTrue(result == jnp.min(array))

        q = [1, 2, 3] * u.second
        result_q = u.math.min(q)
        expected_q = jnp.min(jnp.array([1, 2, 3]))
        assert_quantity(result_q, expected_q, u.second)

    def test_amin(self):
        array = jnp.array([1, 2, 3])
        result = u.math.amin(array)
        self.assertTrue(result == jnp.min(array))

        q = [1, 2, 3] * u.second
        result_q = u.math.amin(q)
        expected_q = jnp.min(jnp.array([1, 2, 3]))
        assert_quantity(result_q, expected_q, u.second)

    def test_amax(self):
        array = jnp.array([1, 2, 3])
        result = u.math.amax(array)
        self.assertTrue(result == jnp.max(array))

        q = [1, 2, 3] * u.second
        result_q = u.math.amax(q)
        expected_q = jnp.max(jnp.array([1, 2, 3]))
        assert_quantity(result_q, expected_q, u.second)

    def test_diagflat(self):
        array = jnp.array([1, 2, 3])
        result = u.math.diagflat(array)
        self.assertTrue(jnp.all(result == jnp.diagflat(array)))

        q = [1, 2, 3] * u.second
        result_q = u.math.diagflat(q)
        expected_q = jnp.diagflat(jnp.array([1, 2, 3]))
        assert_quantity(result_q, expected_q, u.second)

    def test_diagonal(self):
        array = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        result = u.math.diagonal(array)
        self.assertTrue(jnp.all(result == jnp.diagonal(array)))

        q = [[0, 1, 2], [3, 4, 5], [6, 7, 8]] * u.second
        result_q = u.math.diagonal(q)
        expected_q = jnp.diagonal(jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
        assert_quantity(result_q, expected_q, u.second)

    def test_choose(self):
        choices = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6]), jnp.array([7, 8, 9])]
        result = u.math.choose(jnp.array([0, 1, 2]), choices)
        self.assertTrue(jnp.all(result == jnp.choose(jnp.array([0, 1, 2]), choices)))

        # the unit lives on the choices, not on the index; a dimensioned
        # Quantity index is rejected
        q = [0, 1, 2] * u.second
        q = q.astype(jnp.int64)
        with pytest.raises(TypeError):
            u.math.choose(q, choices)

        q_choices = [c * u.second for c in choices]
        result_q = u.math.choose(jnp.array([0, 1, 2]), q_choices)
        expected_q = jnp.choose(jnp.array([0, 1, 2]), choices)
        assert_quantity(result_q, expected_q, u.second)

    def test_ravel(self):
        array = jnp.array([[1, 2, 3], [4, 5, 6]])
        result = u.math.ravel(array)
        self.assertTrue(jnp.all(result == jnp.ravel(array)))

        q = [[1, 2, 3], [4, 5, 6]] * u.second
        result_q = u.math.ravel(q)
        expected_q = jnp.ravel(jnp.array([[1, 2, 3], [4, 5, 6]]))
        assert_quantity(result_q, expected_q, u.second)


class TestFunKeepUnitSelection(parameterized.TestCase):
    def test_compress(self):
        array = jnp.array([1, 2, 3, 4])
        result = u.math.compress(jnp.array([0, 1, 1, 0]), array)
        self.assertTrue(jnp.all(result == jnp.compress(jnp.array([0, 1, 1, 0]), array)))

        q = jnp.array([1, 2, 3, 4]) * u.second
        result_q = u.math.compress(jnp.array([0, 1, 1, 0]), q)
        expected_q = jnp.compress(jnp.array([0, 1, 1, 0]), q.mantissa)
        assert_quantity(result_q, expected_q, u.second)

    def test_extract(self):
        array = jnp.array([1, 2, 3])
        result = u.math.extract(array > 1, array)
        self.assertTrue(jnp.all(result == jnp.extract(array > 1, array)))

        q = jnp.array([1, 2, 3])
        a = array * u.second
        result_q = u.math.extract(q > 1, a)
        expected_q = jnp.extract(q > 1, jnp.array([1, 2, 3])) * u.second
        assert u.math.allclose(result_q, expected_q)

    def test_take(self):
        array = jnp.array([4, 3, 5, 7, 6, 8])
        indices = jnp.array([0, 1, 4])
        result = u.math.take(array, indices)
        self.assertTrue(jnp.all(result == jnp.take(array, indices)))

        q = [4, 3, 5, 7, 6, 8] * u.second
        i = jnp.array([0, 1, 4])
        result_q = u.math.take(q, i)
        expected_q = jnp.take(jnp.array([4, 3, 5, 7, 6, 8]), jnp.array([0, 1, 4]))
        assert_quantity(result_q, expected_q, u.second)

    def test_select(self):
        condlist = [jnp.array([True, False, True]), jnp.array([False, True, False])]
        choicelist = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6])]
        result = u.math.select(condlist, choicelist, default=0)
        self.assertTrue(jnp.all(result == jnp.select(condlist, choicelist, default=0)))

        c = [jnp.array([True, False, True]), jnp.array([False, True, False])]
        ch = [[1, 2, 3] * u.second, [4, 5, 6] * u.second]
        result_q = u.math.select(c, ch, default=0)
        expected_q = jnp.select([jnp.array([True, False, True]), jnp.array([False, True, False])],
                                [jnp.array([1, 2, 3]), jnp.array([4, 5, 6])], default=0)
        assert_quantity(result_q, expected_q, u.second)

    def test_where(self):
        array = jnp.array([1, 2, 3, 4, 5])
        result = u.math.where(array > 2, array, 0)
        self.assertTrue(jnp.all(result == jnp.where(array > 2, array, 0)))

        q = [1, 2, 3, 4, 5] * u.second
        result_q = u.math.where(q > 2 * u.second, q, 0 * u.second)
        expected_q = jnp.where(jnp.array([1, 2, 3, 4, 5]) > 2, jnp.array([1, 2, 3, 4, 5]), 0)
        assert_quantity(result_q, expected_q, u.second)

    def test_unique(self):
        array = jnp.array([0, 1, 2, 1, 0])
        result = u.math.unique(array)
        self.assertTrue(jnp.all(result == jnp.unique(array)))

        q = [0, 1, 2, 1, 0] * u.second
        result_q = u.math.unique(q)
        expected_q = jnp.unique(jnp.array([0, 1, 2, 1, 0]))
        assert_quantity(result_q, expected_q, u.second)


class TestFunKeepUnitOther(parameterized.TestCase):
    def test_interp(self):
        x = jnp.array([1, 2, 3])
        xp = jnp.array([0, 1, 2, 3, 4])
        fp = jnp.array([0, 1, 2, 3, 4])
        result = u.math.interp(x, xp, fp)
        self.assertTrue(jnp.all(result == jnp.interp(x, xp, fp)))

        x = [1, 2, 3] * u.second
        xp = [0, 1, 2, 3, 4] * u.second
        fp = [0, 1, 2, 3, 4] * u.mvolt
        result_q = u.math.interp(x, xp, fp)
        expected_q = jnp.interp(jnp.array([1, 2, 3]),
                                jnp.array([0, 1, 2, 3, 4]),
                                jnp.array([0, 1, 2, 3, 4])) * u.mvolt
        assert u.math.allclose(result_q, expected_q)

    def test_clip(self):
        array = jnp.array([1, 2, 3, 4, 5])
        result = u.math.clip(array, 2, 4)
        self.assertTrue(jnp.all(result == jnp.clip(array, 2, 4)))

        q = [1, 2, 3, 4, 5] * u.ms
        result_q = u.math.clip(q, 2 * u.ms, 4 * u.ms)
        expected_q = jnp.clip(jnp.array([1, 2, 3, 4, 5]), 2, 4) * u.ms
        assert u.math.allclose(result_q, expected_q)

    def test_histogram(self):
        array = jnp.array([1, 2, 1])
        result, _ = u.math.histogram(array)
        expected, _ = jnp.histogram(array)
        self.assertTrue(jnp.all(result == expected))

        q = [1, 2, 1] * u.second
        result_q, _ = u.math.histogram(q)
        expected_q, _ = jnp.histogram(jnp.array([1, 2, 1]))
        assert_quantity(result_q, expected_q, None)


class TestFunKeepUnit(parameterized.TestCase):

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[second, meter]
    )
    def test_fun_keep_unit_math_unary(self, value, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_keep_unit_math_unary]
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
        bm_fun_list = [getattr(um, fun) for fun in fun_keep_unit_math_binary]
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

            with pytest.raises(TypeError):
                result = bm_fun(q1, jnp.array(x2))

            with pytest.raises(TypeError):
                result = bm_fun(jnp.array(x1), q2)

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, jnp.nan, 3.45)],
        q=[25, 50, 75],
        unit=[second, meter]
    )
    def test_fun_keep_unit_percentile(self, value, q, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_keep_unit_percentile]
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
        bm_fun_list = [getattr(um, fun) for fun in fun_keep_unit_percentile]
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
        bm_fun_list = [getattr(um, fun) for fun in fun_accept_unitless_unary_2_results]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_accept_unitless_unary_2_results]

        for fun in fun_accept_unitless_unary_2_results:
            bm_fun = getattr(um, fun)
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
            bm_fun = getattr(um, fun)
            jnp_fun = jnp.trunc if fun == 'fix' else getattr(jnp, fun)

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
        result = u.math.trace(a)
        self.assertTrue(result == jnp.trace(a))

        q = [[1, 2], [3, 4]] * u.second
        result_q = u.math.trace(q)
        expected_q = jnp.trace(jnp.array([[1, 2], [3, 4]]))
        assert_quantity(result_q, expected_q, u.second)

    def test_lcm(self):
        result = u.math.lcm(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))
        self.assertTrue(jnp.all(result == jnp.lcm(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))))

        q1 = [4, 5, 6] * u.second
        q2 = [2, 3, 4] * u.second
        q1 = q1.astype(jnp.int64)
        q2 = q2.astype(jnp.int64)
        result_q = u.math.lcm(q1, q2)
        expected_q = jnp.lcm(jnp.array([4, 5, 6]), jnp.array([2, 3, 4])) * u.second
        assert u.math.allclose(result_q, expected_q)

    def test_gcd(self):
        result = u.math.gcd(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))
        self.assertTrue(jnp.all(result == jnp.gcd(jnp.array([4, 5, 6]), jnp.array([2, 3, 4]))))

        q1 = [4, 5, 6] * u.second
        q2 = [2, 3, 4] * u.second
        q1 = q1.astype(jnp.int64)
        q2 = q2.astype(jnp.int64)
        result_q = u.math.gcd(q1, q2)
        expected_q = jnp.gcd(jnp.array([4, 5, 6]), jnp.array([2, 3, 4])) * u.second
        assert u.math.allclose(result_q, expected_q)

    def test_copysign(self):
        result = u.math.copysign(jnp.array([-1, 2]), jnp.array([1, -3]))
        self.assertTrue(jnp.all(result == jnp.copysign(jnp.array([-1, 2]), jnp.array([1, -3]))))

        q1 = [-1, 2] * ms
        q2 = [1, -3] * ms
        result_q = u.math.copysign(q1, q2)
        expected_q = jnp.copysign(jnp.array([-1, 2]), jnp.array([1, -3])) * ms
        assert u.math.allclose(result_q, expected_q)

    def test_rot90(self):
        a = jnp.array([[1, 2], [3, 4]])
        result = u.math.rot90(a)
        self.assertTrue(jnp.all(result == jnp.rot90(a)))

        q = [[1, 2], [3, 4]] * u.second
        result_q = u.math.rot90(q)
        expected_q = jnp.rot90(jnp.array([[1, 2], [3, 4]]))
        assert_quantity(result_q, expected_q, u.second)

    def test_intersect1d(self):
        a = jnp.array([1, 2, 3, 4, 5])
        b = jnp.array([3, 4, 5, 6, 7])
        result = u.math.intersect1d(a, b)
        self.assertTrue(jnp.all(result == jnp.intersect1d(a, b)))

        q1 = [1, 2, 3, 4, 5] * u.second
        q2 = [3, 4, 5, 6, 7] * u.second
        result_q = u.math.intersect1d(q1, q2)
        expected_q = jnp.intersect1d(jnp.array([1, 2, 3, 4, 5]), jnp.array([3, 4, 5, 6, 7]))
        assert_quantity(result_q, expected_q, u.second)


class TestGather:
    def test(self):
        # Test 1: Basic 2D example (matches PyTorch documentation)
        input_tensor = jnp.array([[1, 2], [3, 4]])
        index_tensor = jnp.array([[0, 0], [1, 0]])
        result1 = u.math.gather(input_tensor, 1, index_tensor)
        print("Test 1:")
        print("Input:", input_tensor)
        print("Index:", index_tensor)
        print("Result:", result1)
        print()

        # Test 2: 3D example
        input_3d = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        index_3d = jnp.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
        result2 = u.math.gather(input_3d, 2, index_3d)
        print("Test 2:")
        print("Input shape:", input_3d.shape)
        print("Index shape:", index_3d.shape)
        print("Result:", result2)
        print()

        # Test 3: Gather along dim=0
        result3 = u.math.gather(input_tensor, 0, jnp.array([[1, 0], [0, 1]]))
        print("Test 3 (dim=0):")
        print("Result:", result3)

        # Test 4: Gather along dim=0
        result3 = u.math.gather(input_tensor * u.mV, 0, jnp.array([[1, 0], [0, 1]]))
        print("Test 3 (dim=0):")
        print("Result:", result3)


# ---------------------------------------------------------------
# Docstring example tests
# ---------------------------------------------------------------


class TestDocstringExamples:
    """Tests mirroring the examples shown in the docstrings."""

    # -- concatenate --
    def test_concatenate_with_quantity(self):
        a = [1, 2] * u.second
        b = [3, 4] * u.second
        result = u.math.concatenate([a, b])
        assert isinstance(result, u.Quantity)
        assert result.unit == u.second
        expected = jnp.concatenate([jnp.array([1, 2]), jnp.array([3, 4])])
        assert_quantity(result, expected, u.second)

    def test_concatenate_plain_array(self):
        result = u.math.concatenate([jnp.array([1, 2]), jnp.array([3, 4])])
        expected = jnp.concatenate([jnp.array([1, 2]), jnp.array([3, 4])])
        assert jnp.array_equal(result, expected)

    # -- stack --
    def test_stack_with_quantity(self):
        a = [1, 2, 3] * u.second
        b = [4, 5, 6] * u.second
        result = u.math.stack([a, b])
        assert isinstance(result, u.Quantity)
        expected = jnp.stack([jnp.array([1, 2, 3]), jnp.array([4, 5, 6])])
        assert_quantity(result, expected, u.second)

    def test_stack_plain_array(self):
        result = u.math.stack([jnp.array([1, 2, 3]), jnp.array([4, 5, 6])])
        expected = jnp.stack([jnp.array([1, 2, 3]), jnp.array([4, 5, 6])])
        assert jnp.array_equal(result, expected)

    # -- reshape --
    def test_reshape_with_quantity(self):
        a = [1, 2, 3, 4] * u.second
        result = u.math.reshape(a, (2, 2))
        assert isinstance(result, u.Quantity)
        assert result.shape == (2, 2)
        expected = jnp.reshape(jnp.array([1, 2, 3, 4]), (2, 2))
        assert_quantity(result, expected, u.second)

    def test_reshape_plain_array(self):
        result = u.math.reshape(jnp.array([1, 2, 3, 4]), (2, 2))
        expected = jnp.reshape(jnp.array([1, 2, 3, 4]), (2, 2))
        assert jnp.array_equal(result, expected)

    # -- sum --
    def test_sum_with_quantity(self):
        a = [1.0, 2.0, 3.0] * u.second
        result = u.math.sum(a)
        assert isinstance(result, u.Quantity)
        assert_quantity(result, 6.0, u.second)

    def test_sum_with_axis(self):
        a = [[1.0, 2.0], [3.0, 4.0]] * u.meter
        result = u.math.sum(a, axis=0)
        assert isinstance(result, u.Quantity)
        expected = jnp.array([4.0, 6.0])
        assert_quantity(result, expected, u.meter)

    def test_sum_plain_array(self):
        result = u.math.sum(jnp.array([1.0, 2.0, 3.0]))
        expected = jnp.sum(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.array_equal(result, expected)

    # -- mean --
    def test_mean_with_quantity(self):
        a = [1.0, 2.0, 3.0] * u.second
        result = u.math.mean(a)
        assert isinstance(result, u.Quantity)
        assert_quantity(result, 2.0, u.second)

    def test_mean_plain_array(self):
        result = u.math.mean(jnp.array([1.0, 2.0, 3.0]))
        expected = jnp.mean(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.array_equal(result, expected)

    # -- abs --
    def test_abs_with_quantity(self):
        a = [-1, -2, 3] * u.meter
        result = u.math.abs(a)
        assert isinstance(result, u.Quantity)
        expected = jnp.array([1, 2, 3])
        assert_quantity(result, expected, u.meter)

    def test_abs_plain_array(self):
        result = u.math.abs(jnp.array([-1, -2, 3]))
        expected = jnp.abs(jnp.array([-1, -2, 3]))
        assert jnp.array_equal(result, expected)

    # -- add --
    def test_add_with_quantity(self):
        a = [1, 2, 3] * u.meter
        b = [4, 5, 6] * u.meter
        result = u.math.add(a, b)
        assert isinstance(result, u.Quantity)
        expected = jnp.array([5, 7, 9])
        assert_quantity(result, expected, u.meter)

    def test_add_plain_array(self):
        result = u.math.add(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
        expected = jnp.add(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
        assert jnp.array_equal(result, expected)

    def test_add_mismatched_raises(self):
        a = [1, 2, 3] * u.meter
        b = jnp.array([4, 5, 6])
        with pytest.raises(TypeError):
            u.math.add(a, b)

    # -- maximum --
    def test_maximum_with_quantity(self):
        a = [1, 3, 5] * u.second
        b = [2, 2, 4] * u.second
        result = u.math.maximum(a, b)
        assert isinstance(result, u.Quantity)
        expected = jnp.array([2, 3, 5])
        assert_quantity(result, expected, u.second)

    def test_maximum_plain_array(self):
        result = u.math.maximum(jnp.array([1, 3, 5]), jnp.array([2, 2, 4]))
        expected = jnp.maximum(jnp.array([1, 3, 5]), jnp.array([2, 2, 4]))
        assert jnp.array_equal(result, expected)

    # -- where --
    def test_where_with_quantity(self):
        a = [1, 2, 3, 4, 5] * u.meter
        result = u.math.where(a > 3 * u.meter, a, 0 * u.meter)
        assert isinstance(result, u.Quantity)
        expected = jnp.array([0, 0, 0, 4, 5])
        assert_quantity(result, expected, u.meter)

    def test_where_plain_array(self):
        a = jnp.array([1, 2, 3, 4, 5])
        result = u.math.where(a > 3, a, 0)
        expected = jnp.where(a > 3, a, 0)
        assert jnp.array_equal(result, expected)

    def test_where_condition_only(self):
        a = jnp.array([True, False, True])
        result = u.math.where(a)
        expected = jnp.where(a)
        for r, e in zip(result, expected):
            assert jnp.array_equal(r, e)


def test_concatenate_numpy_backend():
    import numpy as np
    a = u.Quantity(np.array([1.0, 2.0]), unit=meter)
    b = u.Quantity(np.array([3.0, 4.0]), unit=meter)
    r = u.math.concatenate([a, b])
    assert r.backend == "numpy"
    assert r.unit == meter
    assert np.allclose(r.mantissa, [1.0, 2.0, 3.0, 4.0])


def test_concatenate_jax_backend():
    import numpy as np
    a = u.Quantity(jnp.array([1.0, 2.0]), unit=meter)
    b = u.Quantity(jnp.array([3.0, 4.0]), unit=meter)
    r = u.math.concatenate([a, b])
    assert r.backend == "jax"
    assert np.allclose(np.asarray(r.mantissa), [1.0, 2.0, 3.0, 4.0])


def test_reshape_numpy_backend():
    import numpy as np
    q = u.Quantity(np.arange(6.0), unit=meter)
    r = u.math.reshape(q, (2, 3))
    assert r.backend == "numpy"
    assert r.shape == (2, 3)


def test_promote_dtypes_common_type_and_unit():
    a = [1, 2, 3] * u.second          # integer mantissa
    b = [4.0, 5.0, 6.0] * u.second    # float mantissa
    out = u.math.promote_dtypes(a, b)
    assert isinstance(out, list)
    assert len(out) == 2
    # Both promoted to the common (float) dtype.
    assert jnp.issubdtype(out[0].mantissa.dtype, jnp.floating)
    assert jnp.issubdtype(out[1].mantissa.dtype, jnp.floating)
    # Unit is preserved on both.
    assert out[0].unit == u.second
    assert out[1].unit == u.second
    # Values are unchanged.
    assert jnp.allclose(out[0].mantissa, jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(out[1].mantissa, jnp.array([4.0, 5.0, 6.0]))


# ---------------------------------------------------------------
# Regression tests (math audit)
# ---------------------------------------------------------------


def test_unit_scale_align_to_first_unitless_first_rescales():
    # A scaled-dimensionless Quantity following a plain first item must be
    # rescaled onto the (unitless) first unit, not passed through raw.
    aligned = u.unit_scale_align_to_first(jnp.array([1., 2.]), jnp.array([1., 2.]) * (u.mV / u.volt))
    assert jnp.allclose(aligned[1].mantissa, jnp.array([0.001, 0.002]))

    r = u.math.concatenate([jnp.array([1., 2.]), jnp.array([1., 2.]) * (u.mV / u.volt)])
    assert not isinstance(r, u.Quantity)
    assert jnp.allclose(r, jnp.array([1., 2., 0.001, 0.002]))

    r = u.math.stack([jnp.array([1., 2.]), jnp.array([1., 2.]) * (u.mV / u.volt)])
    assert not isinstance(r, u.Quantity)
    assert jnp.allclose(r, jnp.array([[1., 2.], [0.001, 0.002]]))

    # a dimensioned later item still raises
    with pytest.raises(u.UnitMismatchError):
        u.unit_scale_align_to_first(jnp.array([1., 2.]), jnp.array([1., 2.]) * u.meter)


def test_unit_scale_align_to_first_all_plain_unchanged():
    aligned = u.unit_scale_align_to_first(jnp.array([1, 2]), jnp.array([3, 4]))
    assert all(isinstance(q, u.Quantity) for q in aligned)
    assert jnp.array_equal(aligned[1].mantissa, jnp.array([3, 4]))
    # consumers still unwrap plain inputs to plain arrays
    r = u.math.concatenate([jnp.array([1, 2]), jnp.array([3, 4])])
    assert not isinstance(r, u.Quantity)
    assert jnp.array_equal(r, jnp.array([1, 2, 3, 4]))


def test_choose_quantity_choices():
    index = jnp.array([0, 1, 0])
    choices = [jnp.array([1., 2., 3.]) * u.meter, jnp.array([4., 5., 6.]) * u.meter]
    result = u.math.choose(index, choices)
    assert isinstance(result, u.Quantity)
    assert_quantity(result, jnp.array([1., 5., 3.]), u.meter)


def test_choose_dimensioned_index_raises():
    choices = [jnp.array([1., 2., 3.]), jnp.array([4., 5., 6.])]
    with pytest.raises(TypeError, match='choose'):
        u.math.choose(jnp.array([0, 1, 0]) * u.meter, choices)


def test_choose_unitless_quantity_index():
    index = u.Quantity(jnp.array([0, 1, 0]))
    choices = [jnp.array([1., 2., 3.]) * u.meter, jnp.array([4., 5., 6.]) * u.meter]
    result = u.math.choose(index, choices)
    assert_quantity(result, jnp.array([1., 5., 3.]), u.meter)


def test_interp_left_right_use_fp_unit():
    x = jnp.array([0.5, 2.5]) * u.second
    xs = jnp.array([1., 2.]) * u.second
    fp = jnp.array([10., 20.]) * u.meter
    result = u.math.interp(x, xs, fp, left=0 * u.meter, right=30 * u.meter)
    assert_quantity(result, jnp.array([0., 30.]), u.meter)


def test_interp_left_scaled_fp_unit():
    # x and fp share the dimension at different scales: left must be
    # converted into fp's unit, not x's.
    x = jnp.array([0.5]) * u.meter
    xs = jnp.array([1., 2.]) * u.meter
    fp = jnp.array([10., 20.]) * u.kmeter
    result = u.math.interp(x, xs, fp, left=1 * u.kmeter)
    assert_quantity(result, jnp.array([1.]), u.kmeter)


def test_average_returned_tuple():
    a = jnp.array([1., 2., 3.]) * u.meter
    avg, wsum = u.math.average(a, weights=jnp.array([1., 1., 2.]), returned=True)
    assert isinstance(avg, u.Quantity)
    assert_quantity(avg, 2.25, u.meter)
    assert not isinstance(wsum, u.Quantity)
    assert jnp.allclose(wsum, 4.0)

    # 2-D shapes survive
    a2 = jnp.array([[1., 2.], [3., 4.]]) * u.meter
    avg2, wsum2 = u.math.average(a2, axis=0, returned=True)
    assert avg2.shape == (2,)
    assert wsum2.shape == (2,)
    assert_quantity(avg2, jnp.array([2., 3.]), u.meter)

    # plain input stays plain
    avg3, wsum3 = u.math.average(jnp.array([1., 2., 3.]), returned=True)
    assert not isinstance(avg3, u.Quantity)
    assert jnp.allclose(avg3, 2.0)
    assert jnp.allclose(wsum3, 3.0)


def test_average_quantity_weights():
    a = jnp.array([1., 2., 3.]) * u.meter
    result = u.math.average(a, weights=u.Quantity(jnp.array([1., 1., 2.])))
    assert_quantity(result, 2.25, u.meter)
    # the weights' unit cancels in a weighted average
    result = u.math.average(a, weights=jnp.array([1., 1., 2.]) * u.second)
    assert_quantity(result, 2.25, u.meter)


def test_max_min_quantity_initial():
    a = jnp.array([1., 2.]) * u.meter
    assert_quantity(u.math.max(a, initial=5 * u.meter), 5.0, u.meter)
    assert_quantity(u.math.min(a, initial=0 * u.meter), 0.0, u.meter)
    # plain data + plain initial still works
    r = u.math.max(jnp.array([1., 2.]), initial=5.0)
    assert not isinstance(r, u.Quantity)
    assert r == 5.0


def test_max_min_plain_initial_on_unitful_data_raises():
    a = jnp.array([1., 2.]) * u.meter
    with pytest.raises(u.UnitMismatchError):
        u.math.max(a, initial=5.0)
    with pytest.raises(u.UnitMismatchError):
        u.math.min(a, initial=5.0)


def test_unflatten_negative_axis():
    a = jnp.arange(6.) * u.meter
    result = u.math.unflatten(a, -1, (2, 3))
    assert result.shape == (2, 3)
    assert_quantity(result, jnp.arange(6.).reshape(2, 3), u.meter)

    result = u.math.unflatten(jnp.array([5.]) * u.meter, -1, (1, 1))
    assert result.shape == (1, 1)


def test_unflatten_axis_out_of_bounds():
    a = jnp.arange(6.) * u.meter
    with pytest.raises(ValueError):
        u.math.unflatten(a, 1, (2, 3))
    with pytest.raises(ValueError):
        u.math.unflatten(a, -2, (2, 3))


def test_concatenate_axis_none_flattens():
    a = jnp.array([[1., 2.], [3., 4.]]) * u.meter
    b = jnp.array([5., 6.]) * u.meter
    result = u.math.concatenate([a, b], axis=None)
    expected = jnp.concatenate([jnp.array([[1., 2.], [3., 4.]]), jnp.array([5., 6.])], axis=None)
    assert_quantity(result, expected, u.meter)

    r = u.math.concatenate([jnp.array([[1, 2], [3, 4]]), jnp.array([5, 6])], axis=None)
    assert jnp.array_equal(r, jnp.array([1, 2, 3, 4, 5, 6]))

    # the default stays axis=0
    result = u.math.concatenate([a, a])
    assert result.shape == (4, 2)


def test_concatenate_empty_raises_value_error():
    with pytest.raises(ValueError, match='at least one array'):
        u.math.concatenate([])


def test_repeat_total_repeat_length_numpy_backend():
    import numpy as np
    # truncation, as jnp.repeat does
    r = u.math.repeat(np.array([1, 2]), 3, total_repeat_length=4)
    assert r.shape == (4,)
    assert np.array_equal(r, np.array([1, 1, 1, 2]))
    # padding with the final element, as jnp.repeat does
    r = u.math.repeat(np.array([1, 2]), 3, total_repeat_length=8)
    assert np.array_equal(r, np.array([1, 1, 1, 2, 2, 2, 2, 2]))
    # along an axis
    r = u.math.repeat(np.array([[1, 2], [3, 4]]), 2, axis=1, total_repeat_length=3)
    assert np.array_equal(r, np.array([[1, 1, 2], [3, 3, 4]]))
    # Quantity input keeps the unit
    q = u.Quantity(np.array([1., 2.]), unit=meter)
    rq = u.math.repeat(q, 3, total_repeat_length=4)
    assert isinstance(rq, u.Quantity)
    assert rq.unit == meter
    assert np.array_equal(rq.mantissa, np.array([1., 1., 1., 2.]))
    # the jax path is unchanged
    rj = u.math.repeat(jnp.array([1, 2]) * meter, 3, total_repeat_length=4)
    assert_quantity(rj, jnp.array([1, 1, 1, 2]), meter)


def test_select_quantity_default():
    conds = [jnp.array([True, False, False])]
    choices = [jnp.array([1., 2., 3.]) * u.mV]
    result = u.math.select(conds, choices, default=5 * u.mV)
    assert_quantity(result, jnp.array([1., 5., 5.]), u.mV)
    # a default in another scale of the same dimension is rescaled
    result = u.math.select(conds, choices, default=0.005 * u.volt)
    assert_quantity(result, jnp.array([1., 5., 5.]), u.mV)


def test_select_plain_default_with_unitful_choices():
    conds = [jnp.array([True, False])]
    choices = [jnp.array([1., 2.]) * u.mV]
    # a plain non-zero default would silently acquire the choicelist's unit
    with pytest.raises(TypeError):
        u.math.select(conds, choices, default=5.0)
    # plain zero stays allowed (jnp's documented default, unit-neutral)
    result = u.math.select(conds, choices, default=0)
    assert_quantity(result, jnp.array([1., 0.]), u.mV)


def test_histogram_quantity_bins_and_weights():
    x = jnp.array([1., 2., 3.]) * u.second
    hist, edges = u.math.histogram(x, bins=jnp.array([0., 2., 4.]) * u.second)
    assert jnp.array_equal(hist, jnp.array([1., 2.]))
    assert isinstance(edges, u.Quantity)
    assert_quantity(edges, jnp.array([0., 2., 4.]), u.second)
    # bins in a different scale of the same dimension are rescaled
    hist, edges = u.math.histogram(x, bins=jnp.array([0., 2000., 4000.]) * u.ms)
    assert jnp.array_equal(hist, jnp.array([1., 2.]))
    # Quantity weights are unwrapped
    hist, _ = u.math.histogram(x, bins=jnp.array([0., 2., 4.]) * u.second,
                               weights=u.Quantity(jnp.array([1., 1., 2.])))
    assert jnp.allclose(hist, jnp.array([1., 3.]))


def test_gather_index_smaller_than_input():
    a = jnp.arange(1., 10.).reshape(3, 3) * u.mV
    index = jnp.array([[0, 1], [2, 0]])
    result = u.math.gather(a, 1, index)
    assert isinstance(result, u.Quantity)
    assert_quantity(result, jnp.array([[1., 2.], [6., 4.]]), u.mV)

    result = u.math.gather(a, 0, index)
    assert_quantity(result, jnp.array([[1., 5.], [7., 2.]]), u.mV)


def test_intersect1d_scaled_dimensionless_vs_plain():
    q = jnp.array([1., 2., 3.]) * (u.mV / u.volt)   # values 0.001, 0.002, 0.003
    plain = jnp.array([0.001, 0.002, 5.0])
    result = u.math.intersect1d(q, plain)
    assert isinstance(result, u.Quantity)
    assert jnp.allclose(result.to_decimal(u.mV / u.volt), jnp.array([1., 2.]))
    # symmetric: plain first
    result2 = u.math.intersect1d(plain, q)
    assert not isinstance(result2, u.Quantity)
    assert jnp.allclose(result2, jnp.array([0.001, 0.002]))


def test_remove_diag_non_square_raises():
    with pytest.raises(ValueError, match='square'):
        u.math.remove_diag(jnp.arange(6.).reshape(3, 2) * u.second)
    with pytest.raises(ValueError, match='square'):
        u.math.remove_diag(jnp.arange(6.).reshape(2, 3))


def test_nan_to_num_plain_array_quantity_replacement_raises():
    x = jnp.array([1.0, jnp.nan])
    with pytest.raises(TypeError, match='nan_to_num'):
        u.math.nan_to_num(x, nan=1 * u.meter)
    with pytest.raises(TypeError, match='nan_to_num'):
        u.math.nan_to_num(x, posinf=1 * u.meter)
    with pytest.raises(TypeError, match='nan_to_num'):
        u.math.nan_to_num(x, neginf=1 * u.meter)


def test_filter_unsupported_kwargs_raises_on_unknown():
    """Unknown kwargs must raise, not be silently dropped."""
    from saiunit.math._fun_keep_unit import _filter_unsupported_kwargs

    def fn(a, axis=-1):
        return a

    with pytest.raises(TypeError, match="unexpected keyword argument 'axes'"):
        _filter_unsupported_kwargs(fn, {'axes': 0})
    # accepted kwargs pass through unchanged
    assert _filter_unsupported_kwargs(fn, {'axis': 0}) == {'axis': 0}

    def fn_varkw(a, **kw):
        return a

    # **kwargs / un-introspectable functions are left for _safe_call
    assert _filter_unsupported_kwargs(fn_varkw, {'whatever': 1}) == {'whatever': 1}
