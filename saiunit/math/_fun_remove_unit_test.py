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


import unittest

import brainstate as bst  # type: ignore[import-untyped]
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

import saiunit as u
import saiunit.math as bm
from saiunit._base_getters import assert_quantity


class Array(u.CustomArray):
    def __init__(self, value):
        self.data = value


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
        unit=[u.meter, u.second]
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
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            q = jnp.array(value) * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            array_input = Array(q)
            result = bm_fun(array_input.data)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

    @parameterized.product(
        value=[((1.0, 2.0), (3.0, 4.0)),
               ((1.23, 2.34, 3.45), (4.56, 5.67, 6.78))],
        unit=[u.meter, u.second]
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
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            q1 = jnp.array(x1) * unit
            q2 = jnp.array(x2) * unit
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            array_input1 = Array(q1)
            array_input2 = Array(q2)
            result = bm_fun(array_input1.data, array_input2.data)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[u.meter, u.second]
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
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            q = jnp.array(value) * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            array_input = Array(q)
            result = bm_fun(array_input.data)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

    def test_fun_remove_unit_sign_operations_with_array(self):
        data = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * u.meter
        test_array = Array(data)

        assert isinstance(test_array, u.CustomArray)
        assert hasattr(test_array, 'data')

        sign_result = bm.sign(test_array.data)
        sign_array = Array(sign_result)
        assert isinstance(sign_array, u.CustomArray)
        assert_quantity(sign_array.data, jnp.array([-1.0, -1.0, 0.0, 1.0, 1.0]))

        signbit_result = bm.signbit(test_array.data)
        signbit_array = Array(signbit_result)
        assert isinstance(signbit_array, u.CustomArray)
        assert_quantity(signbit_array.data, jnp.array([True, True, False, False, False]))

    def test_fun_remove_unit_comparison_operations_with_array(self):
        data1 = jnp.array([1.0, 2.0, 3.0]) * u.second
        data2 = jnp.array([2.0, 2.0, 2.0]) * u.second

        array1 = Array(data1)
        array2 = Array(data2)

        assert isinstance(array1, u.CustomArray)
        assert isinstance(array2, u.CustomArray)

        equal_result = bm.equal(array1.data, array2.data)
        equal_array = Array(equal_result)
        assert isinstance(equal_array, u.CustomArray)
        assert_quantity(equal_array.data, jnp.array([False, True, False]))

        greater_result = bm.greater(array1.data, array2.data)
        greater_array = Array(greater_result)
        assert isinstance(greater_array, u.CustomArray)
        assert_quantity(greater_array.data, jnp.array([False, False, True]))

        less_result = bm.less(array1.data, array2.data)
        less_array = Array(less_result)
        assert isinstance(less_array, u.CustomArray)
        assert_quantity(less_array.data, jnp.array([True, False, False]))

    def test_fun_remove_unit_indexing_operations_with_array(self):
        data = jnp.array([3.0, 1.0, 4.0, 2.0, 5.0]) * u.meter
        test_array = Array(data)

        assert isinstance(test_array, u.CustomArray)

        argsort_result = bm.argsort(test_array.data)
        argsort_array = Array(argsort_result)
        assert isinstance(argsort_array, u.CustomArray)
        assert_quantity(argsort_array.data, jnp.array([1, 3, 0, 2, 4]))

        argmax_result = bm.argmax(test_array.data)
        argmax_array = Array(argmax_result)
        assert isinstance(argmax_array, u.CustomArray)
        assert_quantity(argmax_array.data, 4)

        argmin_result = bm.argmin(test_array.data)
        argmin_array = Array(argmin_result)
        assert isinstance(argmin_array, u.CustomArray)
        assert_quantity(argmin_array.data, 1)

    def test_fun_remove_unit_searchsorted_with_array(self):
        sorted_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]) * u.second
        values = jnp.array([2.5, 3.5, 4.5]) * u.second

        sorted_array = Array(sorted_data)
        values_array = Array(values)

        assert isinstance(sorted_array, u.CustomArray)
        assert isinstance(values_array, u.CustomArray)

        searchsorted_result = bm.searchsorted(sorted_array.data, values_array.data)
        searchsorted_array = Array(searchsorted_result)
        assert isinstance(searchsorted_array, u.CustomArray)
        assert_quantity(searchsorted_array.data, jnp.array([2, 3, 4]))

    def test_fun_remove_unit_nonzero_operations_with_array(self):
        data = jnp.array([0.0, 1.0, 0.0, 2.0, 0.0]) * u.meter
        test_array = Array(data)

        assert isinstance(test_array, u.CustomArray)

        nonzero_result = bm.nonzero(test_array.data)
        assert len(nonzero_result) == 1
        nonzero_array = Array(nonzero_result[0])
        assert isinstance(nonzero_array, u.CustomArray)
        assert_quantity(nonzero_array.data, jnp.array([1, 3]))

        count_nonzero_result = bm.count_nonzero(test_array.data)
        count_nonzero_array = Array(count_nonzero_result)
        assert isinstance(count_nonzero_array, u.CustomArray)
        assert_quantity(count_nonzero_array.data, 2)

    def test_fun_remove_unit_digitize_with_array(self):
        data = jnp.array([0.2, 1.5, 2.8, 3.1]) * u.meter
        bins = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]) * u.meter

        data_array = Array(data)
        bins_array = Array(bins)

        assert isinstance(data_array, u.CustomArray)
        assert isinstance(bins_array, u.CustomArray)

        digitize_result = bm.digitize(data_array.data, bins_array.data)
        digitize_array = Array(digitize_result)
        assert isinstance(digitize_array, u.CustomArray)
        assert_quantity(digitize_array.data, jnp.array([1, 2, 3, 4]))

    def test_fun_remove_unit_heaviside_with_array(self):
        x_data = jnp.array([-1.0, 0.0, 1.0]) * u.second
        h_data = jnp.array([0.5, 0.5, 0.5])

        x_array = Array(x_data)
        h_array = Array(h_data)

        assert isinstance(x_array, u.CustomArray)
        assert isinstance(h_array, u.CustomArray)

        heaviside_result = bm.heaviside(x_array.data, h_array.data)
        heaviside_array = Array(heaviside_result)
        assert isinstance(heaviside_array, u.CustomArray)
        assert_quantity(heaviside_array.data, jnp.array([0.0, 0.5, 1.0]))


class TestFunRemoveUnit(parameterized.TestCase):

    @parameterized.product(
        value=[(-1.0, 2.0), (-1.23, 2.34, 3.45)],
        unit=[u.meter, u.second]
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
        unit=[u.meter, u.second]
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

            with pytest.raises(TypeError):
                result = bm_fun(jnp.array(x1), q2)
                expected = jnp_fun(jnp.array(x1), jnp.array(x2))
                assert_quantity(result, expected)

            with pytest.raises(TypeError):
                result = bm_fun(q1, q2)
                expected = jnp_fun(jnp.array(x1), jnp.array(x2))
                assert_quantity(result, expected)

    @parameterized.product(
        value=[(1, 2), (1, 2, 3)],
        unit=[u.meter, u.second]
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
        unit=[u.meter, u.second]
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

            with pytest.raises(TypeError):
                result = bm_fun(jnp.array(array), q_bins)

            with pytest.raises(TypeError):
                result = bm_fun(q_array, jnp.array(bins))

    @parameterized.product(
        value=[(True, True), (False, True, False)],
        unit=[u.meter, u.second]
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

            with pytest.raises(TypeError):
                result = bm_fun(q)

    @parameterized.product(
        value=[((1.0, 2.0), (3.0, 4.0)),
               ((1.23, 2.34, 3.45), (1.23, 2.34, 3.45))],
        unit=[u.meter, u.second]
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

            with pytest.raises(TypeError):
                result = bm_fun(jnp.array(x1), q2)

            with pytest.raises(TypeError):
                result = bm_fun(q1, jnp.array(x2))

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[u.meter, u.second]
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
        unit=[u.meter, u.second]
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
        unit=[u.meter, u.second]
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
        unit=[u.meter, u.second]
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

            with pytest.raises(u.UnitMismatchError):
                result = bm_fun(jnp.array(x), q_v)

            with pytest.raises(u.UnitMismatchError):
                result = bm_fun(q_x, jnp.array(v))


class Test_allclose(unittest.TestCase):
    def test1(self):
        a = bst.random.random((10, 10))
        b = a + 1e-4
        assert u.math.allclose(a, b, atol=1e-3)

        a = a * u.ms
        b = b * u.ms
        with pytest.raises(u.UnitMismatchError):
            assert u.math.allclose(a, b, atol=1e-3)
        assert u.math.allclose(a, b, atol=1e-3 * u.ms)

        val = bst.random.random((10, 10))
        a = val * u.mV
        b = val * u.ms
        with pytest.raises(u.UnitMismatchError):
            assert u.math.allclose(a, b)

        b = val * u.volt
        assert not u.math.allclose(a, b)

        b = val
        with pytest.raises(TypeError):
            assert u.math.allclose(a, b)

        b = val * u.mV
        a = val
        with pytest.raises(TypeError):
            assert u.math.allclose(a, b)

    def test_tol(self):
        val = bst.random.random((10, 10))

        a = val * u.mV
        b = val * u.mV
        # atol carries the data's dimension: plain or mismatched units raise
        with pytest.raises(u.UnitMismatchError):
            assert u.math.allclose(a, b, atol=1e-3)
        with pytest.raises(u.UnitMismatchError):
            assert u.math.allclose(a, b, atol=1e-3 * u.ms)
        assert u.math.allclose(a, b, atol=1e-3 * u.mV)

        # rtol is dimensionless: plain floats and dimensionless Quantities
        # work, unitful rtol raises regardless of the data's unit
        assert u.math.allclose(a, b, rtol=1e-8)
        assert u.math.allclose(a, b, rtol=u.Quantity(1e-8))
        with pytest.raises(u.UnitMismatchError):
            assert u.math.allclose(a, b, rtol=1e-8 * u.ms)
        with pytest.raises(u.UnitMismatchError):
            assert u.math.allclose(a, b, rtol=1e-8 * u.mV)

    def test_rtol_dimensionless(self):
        x = jnp.array([1.0, 2.0]) * u.volt
        y = x * 1.005
        # plain float rtol works on unitful data
        assert u.math.allclose(x, y, rtol=1e-2)
        assert not u.math.allclose(x, y, rtol=1e-3)
        assert u.math.isclose(x, y, rtol=1e-2).all()
        # dimensionless Quantity rtol works
        assert u.math.allclose(x, y, rtol=u.Quantity(1e-2))
        # unitful rtol is rejected
        with pytest.raises(u.UnitMismatchError):
            u.math.allclose(x, y, rtol=2e-4 * u.mV)
        with pytest.raises(u.UnitMismatchError):
            u.math.isclose(x, y, rtol=2e-4 * u.mV)

    def test_rtol_scale_invariance(self):
        # identical physical data must give identical results regardless of
        # the display unit
        x_v = jnp.array([1.0, 2.0]) * u.volt
        y_v = x_v * 1.005
        x_mv = x_v.in_unit(u.mV)
        y_mv = y_v.in_unit(u.mV)
        assert bool(u.math.allclose(x_v, y_v, rtol=1e-2)) == bool(u.math.allclose(x_mv, y_mv, rtol=1e-2))
        assert bool(u.math.allclose(x_v, y_v, rtol=1e-3)) == bool(u.math.allclose(x_mv, y_mv, rtol=1e-3))
        assert jnp.array_equal(u.math.isclose(x_v, y_v, rtol=1e-2),
                               u.math.isclose(x_mv, y_mv, rtol=1e-2))

    def test_atol_compatible_unit_rescales(self):
        x = jnp.array([1.0, 2.0]) * u.meter
        y = x + 1e-3 * u.meter
        assert u.math.allclose(x, y, rtol=0.0, atol=2 * u.mmeter)
        assert not u.math.allclose(x, y, rtol=0.0, atol=0.5 * u.mmeter)
        assert u.math.isclose(x, y, rtol=0.0, atol=2 * u.mmeter).all()

    def test_atol_zero_is_dimension_neutral(self):
        # A concrete zero is the universal "pure relative tolerance" idiom and
        # is dimensionally neutral (0 is the same in every unit), matching
        # saiunit's zero-compatibility convention used by +, -, ==, <, > ...
        # So `atol=0` must work on unitful data even though a nonzero bare
        # float does not.
        a = jnp.array([1.0, 2.0]) * u.mV
        b = jnp.array([1.0, 2.0]) * u.mV
        # bare zero (int and float) accepted on unitful data
        assert u.math.allclose(a, b, atol=0)
        assert u.math.allclose(a, b, atol=0.0)
        assert u.math.isclose(a, b, atol=0).all()
        # the canonical pure-relative idiom
        assert u.math.allclose(a, b, rtol=1e-5, atol=0)
        # an all-zero array tolerance is likewise neutral
        assert u.math.allclose(a, b, atol=jnp.zeros(2))
        # numeric correctness: with rtol=0 and atol=0 only exact equality is close
        c = jnp.array([1.0, 2.0 + 1e-6]) * u.mV
        assert not u.math.allclose(a, c, rtol=0.0, atol=0)
        assert u.math.isclose(a, c, rtol=0.0, atol=0).tolist() == [True, False]
        # contract preserved: a *nonzero* bare-float atol still raises
        with pytest.raises(u.UnitMismatchError):
            u.math.allclose(a, b, atol=1e-3)
        with pytest.raises(u.UnitMismatchError):
            u.math.isclose(a, b, atol=1e-3)


# =========================================================================
# Docstring example tests
# =========================================================================


class TestDocstringExamples(unittest.TestCase):
    """Tests verifying the examples shown in function docstrings."""

    # --- equal ---

    def test_equal_plain_arrays(self):
        result = bm.equal(jnp.array([1, 2, 3]), jnp.array([1, 0, 3]))
        expected = jnp.array([True, False, True])
        assert jnp.array_equal(result, expected)

    def test_equal_quantity_arrays(self):
        a = jnp.array([1.0, 2.0]) * u.meter
        b = jnp.array([1.0, 2.0]) * u.meter
        result = bm.equal(a, b)
        expected = jnp.array([True, True])
        assert jnp.array_equal(result, expected)

    def test_equal_quantity_mismatch_raises(self):
        a = jnp.array([1.0]) * u.meter
        b = jnp.array([1.0])
        with pytest.raises(TypeError):
            bm.equal(a, b)

    # --- greater ---

    def test_greater_plain_arrays(self):
        result = bm.greater(jnp.array([3, 2, 1]), jnp.array([1, 2, 3]))
        expected = jnp.array([True, False, False])
        assert jnp.array_equal(result, expected)

    def test_greater_quantity_arrays(self):
        a = jnp.array([2.0, 1.0]) * u.meter
        b = jnp.array([1.0, 2.0]) * u.meter
        result = bm.greater(a, b)
        expected = jnp.array([True, False])
        assert jnp.array_equal(result, expected)

    # --- all ---

    def test_all_true(self):
        result = bm.all(jnp.array([True, True, True]))
        assert result == True

    def test_all_false(self):
        result = bm.all(jnp.array([True, False, True]))
        assert result == False

    def test_all_axis(self):
        result = bm.all(jnp.array([[True, False], [True, True]]), axis=1)
        expected = jnp.array([False, True])
        assert jnp.array_equal(result, expected)

    # --- argmax ---

    def test_argmax_plain(self):
        result = bm.argmax(jnp.array([1.0, 3.0, 2.0]))
        assert result == 1

    def test_argmax_quantity(self):
        q = jnp.array([1.0, 3.0, 2.0]) * u.meter
        result = bm.argmax(q)
        assert result == 1

    # --- argsort ---

    def test_argsort_plain(self):
        result = bm.argsort(jnp.array([3.0, 1.0, 2.0]))
        expected = jnp.array([1, 2, 0])
        assert jnp.array_equal(result, expected)

    def test_argsort_quantity(self):
        q = jnp.array([3.0, 1.0, 2.0]) * u.meter
        result = bm.argsort(q)
        expected = jnp.array([1, 2, 0])
        assert jnp.array_equal(result, expected)

    # --- sign ---

    def test_sign_plain(self):
        result = bm.sign(jnp.array([-5.0, 0.0, 3.0]))
        expected = jnp.array([-1.0, 0.0, 1.0])
        assert jnp.array_equal(result, expected)

    def test_sign_quantity(self):
        q = jnp.array([-2.0, 0.0, 4.0]) * u.second
        result = bm.sign(q)
        expected = jnp.array([-1.0, 0.0, 1.0])
        assert jnp.array_equal(result, expected)


def test_argmax_numpy_backend():
    import numpy as np
    import saiunit as u
    from saiunit import meter
    q = u.Quantity(np.array([3.0, 1.0, 2.0]), unit=meter)
    r = u.math.argmax(q)
    # argmax strips units; numpy/jax returns an integer-like result
    assert int(r) == 0


def test_argwhere_numpy_backend_default_kwargs():
    """Regression: argwhere must not forward JAX-only kwargs to NumPy."""
    import numpy as np
    import saiunit as u
    from saiunit import meter
    q = u.Quantity(np.array([0.0, 1.0, 0.0, 2.0]), unit=meter)
    r = u.math.argwhere(q)
    assert isinstance(r, np.ndarray) and not isinstance(r, type(np.asarray)) and r.dtype.kind in 'iu'
    assert np.array_equal(r, np.array([[1], [3]]))


def test_flatnonzero_numpy_backend_default_kwargs():
    """Regression: flatnonzero must not forward JAX-only kwargs to NumPy."""
    import numpy as np
    import saiunit as u
    from saiunit import meter
    q = u.Quantity(np.array([0.0, 1.0, 0.0, 2.0]), unit=meter)
    r = u.math.flatnonzero(q)
    assert isinstance(r, np.ndarray)
    assert np.array_equal(r, np.array([1, 3]))


def test_flatnonzero_fill_value_is_unitless_index():
    """Regression: fill_value pads the returned index array, so it must be
    a plain (unitless) value even when the input carries a unit."""
    q = jnp.array([0.0, 1.0, 0.0, 2.0]) * u.meter
    r = u.math.flatnonzero(q, size=4, fill_value=0)
    assert jnp.array_equal(r, jnp.array([1, 3, 0, 0]))
    # an index cannot carry a unit
    with pytest.raises(TypeError):
        u.math.flatnonzero(q, size=4, fill_value=2 * u.meter)


def test_nonzero_numpy_backend_default_kwargs():
    """Regression: nonzero must not forward JAX-only kwargs to NumPy."""
    import numpy as np
    import saiunit as u
    from saiunit import meter
    q = u.Quantity(np.array([0.0, 1.0, 0.0, 2.0]), unit=meter)
    r = u.math.nonzero(q)
    assert isinstance(r, tuple)
    assert np.array_equal(r[0], np.array([1, 3]))


def test_get_promote_dtypes_variadic():
    """Regression: get_promote_dtypes must accept one or three-plus args,
    not just exactly two."""
    import numpy as np
    assert bm.get_promote_dtypes(jnp.float32, jnp.int32) == np.dtype('float32')
    assert bm.get_promote_dtypes(jnp.float32) == np.dtype('float32')
    assert bm.get_promote_dtypes(jnp.float32, jnp.int32, jnp.float64) == np.dtype('float64')


def test_iscomplexobj_matches_jnp():
    real = jnp.array([1.0, 2.0])
    comp = jnp.array([1.0 + 2.0j])
    assert bm.iscomplexobj(real) == jnp.iscomplexobj(real)
    assert bm.iscomplexobj(comp) == jnp.iscomplexobj(comp)
    # Unit is stripped before the check; result is unaffected by the unit.
    assert bm.iscomplexobj(real * u.meter) is False
    assert bm.iscomplexobj(comp * u.meter) is True


def test_alltrue_is_all_alias():
    assert bm.alltrue is bm.all
    x = jnp.array([True, True, False])
    assert bool(bm.alltrue(x)) == bool(jnp.all(x))


def test_sometrue_is_any_alias():
    assert bm.sometrue is bm.any
    x = jnp.array([False, False, True])
    assert bool(bm.sometrue(x)) == bool(jnp.any(x))
