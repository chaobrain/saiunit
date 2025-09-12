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


import unittest

import brainstate as bst
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

import saiunit as bu
import saiunit.math as bm
from saiunit._base import assert_quantity


class Array(bu.CustomArray):
    def __init__(self, value):
        self.value = value

fun_remove_unit_unary = [
    'signbit', 'sign',
]

fun_remove_unit_heaviside = [
    'heaviside',
]
fun_remove_unit_bincount = [
    'bincount',
]
fun_remove_unit_digitize = [
    'digitize',
]

fun_remove_unit_logic_unary = [
    'all', 'any', 'logical_not',
]

fun_remove_unit_logic_binary = [
    'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
    'array_equal', 'isclose', 'allclose', 'logical_and',
    'logical_or', 'logical_xor',
]

fun_remove_unit_indexing_1d = [
    'argsort', 'argmax', 'argmin', 'nanargmax', 'nanargmin', 'argwhere',
    'count_nonzero',
]

fun_remove_unit_indexing_nd = [
    'diag_indices_from',
]

fun_remove_unit_indexing_return_tuple = [
    'nonzero', 'flatnonzero',
]
fun_remove_unit_searchsorted = [
    'searchsorted',
]


class TestFunRemoveUnitWithArrayCustomArray(parameterized.TestCase):

    @parameterized.product(
        value=[(-1.0, 2.0), (-1.23, 2.34, 3.45)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_unary_with_array(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_unary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_unary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected)

            q = jnp.array(value) * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected)

            array_input = Array(q)
            result = bm_fun(array_input.value)
            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected)

    @parameterized.product(
        value=[((1.0, 2.0), (3.0, 4.0)),
               ((1.23, 2.34, 3.45), (4.56, 5.67, 6.78))],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_logic_binary_with_array(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_logic_binary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_logic_binary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            x1, x2 = value
            result = bm_fun(jnp.array(x1), jnp.array(x2))
            expected = jnp_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected)

            q1 = jnp.array(x1) * unit
            q2 = jnp.array(x2) * unit
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected)

            array_input1 = Array(q1)
            array_input2 = Array(q2)
            result = bm_fun(array_input1.value, array_input2.value)
            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected)

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_indexing_1d_with_array(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_indexing_1d]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_indexing_1d]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected)

            q = jnp.array(value) * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected)

            array_input = Array(q)
            result = bm_fun(array_input.value)
            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected)

    def test_fun_remove_unit_sign_operations_with_array(self):
        data = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * bu.meter
        test_array = Array(data)
        
        assert isinstance(test_array, bu.CustomArray)
        assert hasattr(test_array, 'value')
        
        sign_result = bm.sign(test_array.value)
        sign_array = Array(sign_result)
        assert isinstance(sign_array, bu.CustomArray)
        assert_quantity(sign_array.value, jnp.array([-1.0, -1.0, 0.0, 1.0, 1.0]))
        
        signbit_result = bm.signbit(test_array.value)
        signbit_array = Array(signbit_result)
        assert isinstance(signbit_array, bu.CustomArray)
        assert_quantity(signbit_array.value, jnp.array([True, True, False, False, False]))

    def test_fun_remove_unit_comparison_operations_with_array(self):
        data1 = jnp.array([1.0, 2.0, 3.0]) * bu.second
        data2 = jnp.array([2.0, 2.0, 2.0]) * bu.second
        
        array1 = Array(data1)
        array2 = Array(data2)
        
        assert isinstance(array1, bu.CustomArray)
        assert isinstance(array2, bu.CustomArray)
        
        equal_result = bm.equal(array1.value, array2.value)
        equal_array = Array(equal_result)
        assert isinstance(equal_array, bu.CustomArray)
        assert_quantity(equal_array.value, jnp.array([False, True, False]))
        
        greater_result = bm.greater(array1.value, array2.value)
        greater_array = Array(greater_result)
        assert isinstance(greater_array, bu.CustomArray)
        assert_quantity(greater_array.value, jnp.array([False, False, True]))
        
        less_result = bm.less(array1.value, array2.value)
        less_array = Array(less_result)
        assert isinstance(less_array, bu.CustomArray)
        assert_quantity(less_array.value, jnp.array([True, False, False]))

    def test_fun_remove_unit_indexing_operations_with_array(self):
        data = jnp.array([3.0, 1.0, 4.0, 2.0, 5.0]) * bu.meter
        test_array = Array(data)
        
        assert isinstance(test_array, bu.CustomArray)
        
        argsort_result = bm.argsort(test_array.value)
        argsort_array = Array(argsort_result)
        assert isinstance(argsort_array, bu.CustomArray)
        assert_quantity(argsort_array.value, jnp.array([1, 3, 0, 2, 4]))
        
        argmax_result = bm.argmax(test_array.value)
        argmax_array = Array(argmax_result)
        assert isinstance(argmax_array, bu.CustomArray)
        assert_quantity(argmax_array.value, 4)
        
        argmin_result = bm.argmin(test_array.value)
        argmin_array = Array(argmin_result)
        assert isinstance(argmin_array, bu.CustomArray)
        assert_quantity(argmin_array.value, 1)

    def test_fun_remove_unit_searchsorted_with_array(self):
        sorted_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]) * bu.second
        values = jnp.array([2.5, 3.5, 4.5]) * bu.second
        
        sorted_array = Array(sorted_data)
        values_array = Array(values)
        
        assert isinstance(sorted_array, bu.CustomArray)
        assert isinstance(values_array, bu.CustomArray)
        
        searchsorted_result = bm.searchsorted(sorted_array.value, values_array.value)
        searchsorted_array = Array(searchsorted_result)
        assert isinstance(searchsorted_array, bu.CustomArray)
        assert_quantity(searchsorted_array.value, jnp.array([2, 3, 4]))

    def test_fun_remove_unit_nonzero_operations_with_array(self):
        data = jnp.array([0.0, 1.0, 0.0, 2.0, 0.0]) * bu.meter
        test_array = Array(data)
        
        assert isinstance(test_array, bu.CustomArray)
        
        nonzero_result = bm.nonzero(test_array.value)
        assert len(nonzero_result) == 1
        nonzero_array = Array(nonzero_result[0])
        assert isinstance(nonzero_array, bu.CustomArray)
        assert_quantity(nonzero_array.value, jnp.array([1, 3]))
        
        count_nonzero_result = bm.count_nonzero(test_array.value)
        count_nonzero_array = Array(count_nonzero_result)
        assert isinstance(count_nonzero_array, bu.CustomArray)
        assert_quantity(count_nonzero_array.value, 2)

    def test_fun_remove_unit_digitize_with_array(self):
        data = jnp.array([0.2, 1.5, 2.8, 3.1]) * bu.meter
        bins = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]) * bu.meter
        
        data_array = Array(data)
        bins_array = Array(bins)
        
        assert isinstance(data_array, bu.CustomArray)
        assert isinstance(bins_array, bu.CustomArray)
        
        digitize_result = bm.digitize(data_array.value, bins_array.value)
        digitize_array = Array(digitize_result)
        assert isinstance(digitize_array, bu.CustomArray)
        assert_quantity(digitize_array.value, jnp.array([1, 2, 3, 4]))

    def test_fun_remove_unit_heaviside_with_array(self):
        x_data = jnp.array([-1.0, 0.0, 1.0]) * bu.second
        h_data = jnp.array([0.5, 0.5, 0.5])
        
        x_array = Array(x_data)
        h_array = Array(h_data)
        
        assert isinstance(x_array, bu.CustomArray)
        assert isinstance(h_array, bu.CustomArray)
        
        heaviside_result = bm.heaviside(x_array.value, h_array.value)
        heaviside_array = Array(heaviside_result)
        assert isinstance(heaviside_array, bu.CustomArray)
        assert_quantity(heaviside_array.value, jnp.array([0.0, 0.5, 1.0]))


class TestFunRemoveUnit(parameterized.TestCase):

    @parameterized.product(
        value=[(-1.0, 2.0), (-1.23, 2.34, 3.45)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_unary(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_unary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_unary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

    @parameterized.product(
        value=[((1.0, 2.0), (3.0, 4.0)),
               ((1.23, 2.34, 3.45), (4.56, 5.67, 6.78))],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_heaviside(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_heaviside]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_heaviside]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            x1, x2 = value

            result = bm_fun(jnp.array(x1), jnp.array(x2))
            expected = jnp_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            q1 = x1 * unit
            q2 = x2 * unit
            result = bm_fun(q1, jnp.array(x2))
            expected = jnp_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(jnp.array(x1), q2)
                expected = jnp_fun(jnp.array(x1), jnp.array(x2))
                assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(q1, q2)
                expected = jnp_fun(jnp.array(x1), jnp.array(x2))
                assert_quantity(result, expected)

    @parameterized.product(
        value=[(1, 2), (1, 2, 3)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_bincount(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_bincount]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_bincount]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q.astype(jnp.int32))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

    @parameterized.product(
        array=[(1, 2, 3), (1, 2, 3, 4, 5)],
        bins=[(0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_digitize(self, array, bins, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_digitize]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_digitize]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(array), jnp.array(bins))
            expected = jnp_fun(jnp.array(array), jnp.array(bins))
            assert_quantity(result, expected)

            q_array = array * unit
            q_bins = bins * unit
            result = bm_fun(q_array, q_bins)
            expected = jnp_fun(jnp.array(array), jnp.array(bins))
            assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(jnp.array(array), q_bins)

            with pytest.raises(AssertionError):
                result = bm_fun(q_array, jnp.array(bins))

    @parameterized.product(
        value=[(True, True), (False, True, False)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_logic_unary(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_logic_unary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_logic_unary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit

            with pytest.raises(AssertionError):
                result = bm_fun(q)

    @parameterized.product(
        value=[((1.0, 2.0), (3.0, 4.0)),
               ((1.23, 2.34, 3.45), (1.23, 2.34, 3.45))],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_logic_binary(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_logic_binary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_logic_binary]

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
            assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(jnp.array(x1), q2)

            with pytest.raises(AssertionError):
                result = bm_fun(q1, jnp.array(x2))

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_indexing_1d(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_indexing_1d]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_indexing_1d]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

    @parameterized.product(
        value=[(
                [1.0, 2.0],
                [3.0, 4.0],
        ),
            (
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]
            ),
        ],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_indexing_nd(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_indexing_nd]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_indexing_nd]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert (result[0] == expected[0]).all()
            assert (result[1] == expected[1]).all()

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert (result[0] == expected[0]).all()
            assert (result[1] == expected[1]).all()

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_indexing_return_tuple(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_indexing_return_tuple]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_indexing_return_tuple]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            for r, e in zip(result, expected):
                assert_quantity(r, e)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            for r, e in zip(result, expected):
                assert_quantity(r, e)

    @parameterized.product(
        value=[((1.0, 2.0), (3.0, 4.0)),
               ((1.23, 2.34, 3.45), (1.23, 2.34, 3.45))],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_searchsorted(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_searchsorted]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_searchsorted]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            x, v = value
            result = bm_fun(jnp.array(x), jnp.array(v))
            expected = jnp_fun(jnp.array(x), jnp.array(v))
            assert_quantity(result, expected)

            q_x = x * unit
            q_v = v * unit
            result = bm_fun(q_x, q_v)
            expected = jnp_fun(jnp.array(x), jnp.array(v))
            assert_quantity(result, expected)

            with pytest.raises(bu.UnitMismatchError):
                result = bm_fun(jnp.array(x), q_v)

            with pytest.raises(bu.UnitMismatchError):
                result = bm_fun(q_x, jnp.array(v))


class Test_allclose(unittest.TestCase):
    def test1(self):
        a = bst.random.random((10, 10))
        b = a + 1e-4
        assert bu.math.allclose(a, b, atol=1e-3)

        a = a * bu.ms
        b = b * bu.ms
        with pytest.raises(bu.UnitMismatchError):
            assert bu.math.allclose(a, b, atol=1e-3)
        assert bu.math.allclose(a, b, atol=1e-3 * bu.ms)

        val = bst.random.random((10, 10))
        a = val * bu.mV
        b = val * bu.ms
        with pytest.raises(bu.UnitMismatchError):
            assert bu.math.allclose(a, b)

        b = val * bu.volt
        assert not bu.math.allclose(a, b)

        b = val
        with pytest.raises(AssertionError):
            assert bu.math.allclose(a, b)

        b = val * bu.mV
        a = val
        with pytest.raises(AssertionError):
            assert bu.math.allclose(a, b)

    def test_tol(self):
        val = bst.random.random((10, 10))

        a = val * bu.mV
        b = val * bu.mV
        with pytest.raises(bu.UnitMismatchError):
            assert bu.math.allclose(a, b, atol=1e-3)
        with pytest.raises(bu.UnitMismatchError):
            assert bu.math.allclose(a, b, atol=1e-3 * bu.ms)
        assert bu.math.allclose(a, b, atol=1e-3 * bu.mV)

        with pytest.raises(bu.UnitMismatchError):
            assert bu.math.allclose(a, b, rtol=1e-8)
        with pytest.raises(bu.UnitMismatchError):
            assert bu.math.allclose(a, b, rtol=1e-8 * bu.ms)
        assert bu.math.allclose(a, b, rtol=1e-8 * bu.mV)
