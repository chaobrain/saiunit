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


import itertools

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

import saiunit as bu
import saiunit.lax as bulax
from saiunit import meter, second, volt
from saiunit._base import assert_quantity


class Array(bu.CustomArray):
    def __init__(self, value):
        self.value = value


lax_change_unit_unary = [
    'rsqrt',
]

lax_change_unit_binary = [
    'div', 'mul',
]

lax_change_unit_batch_matmul = [
    'batch_matmul',
]

lax_change_unit_rem = [
    'rem',
]

lax_change_unit_pow = [
    'pow', 'integer_pow',
]

lax_change_unit_conv = [
    'conv',
]

lax_change_unit_conv_transpose = [
    'conv_transpose',
]

lax_change_unit_misc = [
    'dot_general',
]


class TestLaxChangeUnitWithArrayCustomArray(parameterized.TestCase):

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[meter, second]
    )
    def test_lax_change_unit_unary_with_array(self, value, unit):
        bulax_fun_list = [getattr(bulax, fun) for fun in lax_change_unit_unary]
        lax_fun_list = [getattr(lax, fun) for fun in lax_change_unit_unary]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')

            result = bulax_fun(jnp.array(value))
            expected = lax_fun(jnp.array(value))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected)

            q = jnp.array(value) * unit
            result = bulax_fun(q)
            expected = lax_fun(jnp.array(value))
            expected_unit = bulax_fun._unit_change_fun(unit)
            assert_quantity(result, expected, unit=expected_unit)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected, unit=expected_unit)

            array_input = Array(q)
            result = bulax_fun(array_input.value)
            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected, unit=expected_unit)

    @parameterized.product(
        value=[((1.123, 2.567, 3.891), (1.23, 2.34, 3.45)),
               ((1.0, 2.0), (3.0, 4.0))],
        unit1=[meter, second],
        unit2=[volt, second]
    )
    def test_lax_change_unit_binary_with_array(self, value, unit1, unit2):
        bulax_fun_list = [getattr(bulax, fun) for fun in lax_change_unit_binary]
        lax_fun_list = [getattr(lax, fun) for fun in lax_change_unit_binary]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')
            value1, value2 = value

            result = bulax_fun(jnp.array(value1), jnp.array(value2))
            expected = lax_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected)

            q1 = jnp.array(value1) * unit1
            q2 = jnp.array(value2) * unit2
            result = bulax_fun(q1, q2)
            expected = lax_fun(jnp.array(value1), jnp.array(value2))
            expected_unit = bulax_fun._unit_change_fun(bu.get_unit(unit1), bu.get_unit(unit2))
            assert_quantity(result, expected, unit=expected_unit)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected, unit=expected_unit)

            array_input1 = Array(q1)
            array_input2 = Array(q2)
            result = bulax_fun(array_input1.value, array_input2.value)
            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.value, expected, unit=expected_unit)

    def test_rsqrt_operations_with_array(self):
        data = jnp.array([4.0, 9.0, 16.0]) * (meter ** 2)
        test_array = Array(data)
        
        assert isinstance(test_array, bu.CustomArray)
        
        rsqrt_result = bulax.rsqrt(test_array.value)
        rsqrt_array = Array(rsqrt_result)
        assert isinstance(rsqrt_array, bu.CustomArray)
        expected = lax.rsqrt(jnp.array([4.0, 9.0, 16.0]))
        assert_quantity(rsqrt_array.value, expected, unit=meter**-1)

    def test_div_mul_operations_with_array(self):
        data1 = jnp.array([6.0, 8.0, 10.0]) * meter
        data2 = jnp.array([2.0, 4.0, 5.0]) * second
        
        array1 = Array(data1)
        array2 = Array(data2)
        
        assert isinstance(array1, bu.CustomArray)
        assert isinstance(array2, bu.CustomArray)
        
        # Test div
        div_result = bulax.div(array1.value, array2.value)
        div_array = Array(div_result)
        assert isinstance(div_array, bu.CustomArray)
        expected_div = lax.div(jnp.array([6.0, 8.0, 10.0]), jnp.array([2.0, 4.0, 5.0]))
        assert_quantity(div_array.value, expected_div, unit=meter / second)
        
        # Test mul
        mul_result = bulax.mul(array1.value, array2.value)
        mul_array = Array(mul_result)
        assert isinstance(mul_array, bu.CustomArray)
        expected_mul = lax.mul(jnp.array([6.0, 8.0, 10.0]), jnp.array([2.0, 4.0, 5.0]))
        assert_quantity(mul_array.value, expected_mul, unit=meter * second)

    def test_array_custom_array_compatibility_with_lax_change_unit(self):
        data = jnp.array([4.0, 9.0, 16.0]) * (meter ** 2)
        test_array = Array(data)
        
        assert isinstance(test_array, bu.CustomArray)
        assert hasattr(test_array, 'value')
        
        # Test that we can use the array value in unit-changing lax functions
        result = bulax.rsqrt(test_array.value)
        result_array = Array(result)
        
        assert isinstance(result_array, bu.CustomArray)
        
        # Compare with direct computation
        direct_result = bulax.rsqrt(data)
        assert_quantity(result_array.value, direct_result.mantissa, unit=meter**-1)


class TestLaxChangeUnit(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLaxChangeUnit, self).__init__(*args, **kwargs)

        print()

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[meter, second]
    )
    def test_lax_change_unit_unary(self, value, unit):
        bulax_fun_list = [getattr(bulax, fun) for fun in lax_change_unit_unary]
        lax_fun_list = [getattr(lax, fun) for fun in lax_change_unit_unary]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')

            result = bulax_fun(jnp.array(value))
            expected = lax_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bulax_fun(q)
            expected = lax_fun(jnp.array(value))
            assert_quantity(result, expected, unit=bulax_fun._unit_change_fun(unit))

    @parameterized.product(
        value=[((1.123, 2.567, 3.891), (1.23, 2.34, 3.45)),
               ((1.0, 2.0), (3.0, 4.0),)],
        unit1=[meter, second],
        unit2=[volt, second]
    )
    def test_lax_change_unit_binary(self, value, unit1, unit2):
        bulax_fun_list = [getattr(bulax, fun) for fun in lax_change_unit_binary]
        lax_fun_list = [getattr(lax, fun) for fun in lax_change_unit_binary]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')
            value1, value2 = value

            result = bulax_fun(jnp.array(value1), jnp.array(value2))
            expected = lax_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * unit1
            q2 = value2 * unit2
            result = bulax_fun(q1, q2)
            expected = lax_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected, unit=bulax_fun._unit_change_fun(bu.get_unit(unit1), bu.get_unit(unit2)))

    @parameterized.product(
        value=[(
                [[[1.0, 2.0], [3.0, 4.0]],
                 [[1.0, 2.0], [3.0, 4.0]]],
                [[[1.0, 2.0], [3.0, 4.0]],
                 [[1.0, 2.0], [3.0, 4.0]]]
        )],
        unit1=[meter, second],
        unit2=[volt, second]
    )
    def test_lax_change_unit_batch_matmul(self, value, unit1, unit2):
        bulax_fun_list = [getattr(bulax, fun) for fun in lax_change_unit_batch_matmul]
        lax_fun_list = [getattr(lax, fun) for fun in lax_change_unit_batch_matmul]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')
            value1, value2 = value

            result = bulax_fun(jnp.array(value1), jnp.array(value2))
            expected = lax_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * unit1
            q2 = value2 * unit2
            result = bulax_fun(q1, q2)
            expected = lax_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected, unit=bulax_fun._unit_change_fun(bu.get_unit(unit1), bu.get_unit(unit2)))

    @parameterized.product(
        value=[((1.0, 2.0), (1.23, 2.34)),
               ((1.0, 2.0), (3.0, 4.0))],
        unit1=[meter, second],
        unit2=[volt, second]
    )
    def test_lax_change_unit_rem(self, value, unit1, unit2):
        bulax_fun_list = [getattr(bulax, fun) for fun in lax_change_unit_rem]
        lax_fun_list = [getattr(lax, fun) for fun in lax_change_unit_rem]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')
            value1, value2 = value

            result = bulax_fun(jnp.array(value1), jnp.array(value2))
            expected = lax_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * unit1
            q2 = value2 * unit2
            result = bulax_fun(q1, q2)
            expected = lax_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected, unit=bu.get_unit(unit1))

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        power_exponents=[2, 3],
        unit=[meter, second]
    )
    def test_lax_change_unit_power(self, value, power_exponents, unit):
        bulax_fun_list = [getattr(bulax, fun) for fun in lax_change_unit_pow]
        lax_fun_list = [getattr(lax, fun) for fun in lax_change_unit_pow]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')

            result = bulax_fun(jnp.array(value), power_exponents)
            expected = lax_fun(jnp.array(value), power_exponents)
            assert_quantity(result, expected)

            q = value * unit
            result = bulax_fun(q, power_exponents)
            expected = lax_fun(jnp.array(value), power_exponents)
            result_unit = unit ** power_exponents
            assert_quantity(result, expected, unit=result_unit)

    @parameterized.product(
        shape=[
            dict(lhs_shape=(b, i, 9, 10), rhs_shape=(j, i, 4, 5))
            for b, i, j in itertools.product([2, 3], repeat=3)
        ],
        window_strides=[(1, 1), (1, 2), (2, 1)],
        padding=["VALID", "SAME", "SAME_LOWER"],
    )
    def test_lax_change_unit_conv(self, shape, window_strides, padding):
        lhs_shape = shape['lhs_shape']
        rhs_shape = shape['rhs_shape']

        lhs = np.random.rand(*lhs_shape).astype(np.float32)
        rhs = np.random.rand(*rhs_shape).astype(np.float32)

        bulax_fun_list = [getattr(bulax, fun) for fun in lax_change_unit_conv]
        lax_fun_list = [getattr(lax, fun) for fun in lax_change_unit_conv]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')

            result = bulax_fun(jnp.array(lhs), jnp.array(rhs), window_strides, padding)
            expected = lax_fun(jnp.array(lhs), jnp.array(rhs), window_strides, padding)
            assert_quantity(result, expected)

            q1 = lhs * meter
            q2 = rhs * meter
            result = bulax_fun(q1, q2, window_strides, padding)
            expected = lax_fun(jnp.array(lhs), jnp.array(rhs), window_strides, padding)
            assert_quantity(result, expected, unit=bulax_fun._unit_change_fun(bu.get_unit(q1), bu.get_unit(q2)))

    @parameterized.product(
        shapes=[
            dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
            for lhs_shape, rhs_shape in [
                ((b, 10, i), (k, i, j))
                for b, i, j, k in itertools.product(
                    [2, 3], [2, 3], [2, 3], [3, ]
                )
            ]
        ],
        strides=[(1,), (2,)],
        padding=["VALID", "SAME"],
    )
    def test_lax_change_unit_conv_transpose(self, shapes, strides, padding):
        lhs_shape = shapes['lhs_shape']
        rhs_shape = shapes['rhs_shape']

        lhs = np.random.rand(*lhs_shape).astype(np.float32)
        rhs = np.random.rand(*rhs_shape).astype(np.float32)

        bulax_fun_list = [getattr(bulax, fun) for fun in lax_change_unit_conv_transpose]
        lax_fun_list = [getattr(lax, fun) for fun in lax_change_unit_conv_transpose]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')

            result = bulax_fun(jnp.array(lhs), jnp.array(rhs), strides, padding)
            expected = lax_fun(jnp.array(lhs), jnp.array(rhs), strides, padding)
            assert_quantity(result, expected)

            q1 = lhs * meter
            q2 = rhs * meter
            result = bulax_fun(q1, q2, strides, padding)
            expected = lax_fun(jnp.array(lhs), jnp.array(rhs), strides, padding)
            assert_quantity(result, expected, unit=bulax_fun._unit_change_fun(bu.get_unit(q1), bu.get_unit(q2)))

    @parameterized.product(
        shapes=[
            dict(
                lhs_shape=lhs_shape,
                rhs_shape=rhs_shape,
                dimension_numbers=dimension_numbers,
            )
            for lhs_shape, rhs_shape, dimension_numbers in [
                ((3, 3, 2), (3, 2, 4), (([2], [1]), ([0], [0]))),
                ((3, 3, 2), (2, 3, 4), (([2], [0]), ([0], [1]))),
                ((3, 4, 2, 4), (3, 4, 3, 2), (([2], [3]), ([0, 1], [0, 1]))),
            ]
        ],
    )
    def test_lax_change_unit_dot_general(self, shapes):
        lhs_shape = shapes['lhs_shape']
        rhs_shape = shapes['rhs_shape']
        dimension_numbers = shapes['dimension_numbers']

        lhs = np.random.rand(*lhs_shape).astype(np.float32)
        rhs = np.random.rand(*rhs_shape).astype(np.float32)

        bulax_fun_list = [getattr(bulax, fun) for fun in lax_change_unit_misc]
        lax_fun_list = [getattr(lax, fun) for fun in lax_change_unit_misc]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')

            result = bulax_fun(jnp.array(lhs), jnp.array(rhs), dimension_numbers=dimension_numbers)
            expected = lax_fun(jnp.array(lhs), jnp.array(rhs), dimension_numbers=dimension_numbers)
            assert_quantity(result, expected)

            q1 = lhs * meter
            q2 = rhs * meter
            result = bulax_fun(q1, q2, dimension_numbers=dimension_numbers)
            expected = lax_fun(jnp.array(lhs), jnp.array(rhs), dimension_numbers=dimension_numbers)
            assert_quantity(result, expected, unit=bulax_fun._unit_change_fun(bu.get_unit(q1), bu.get_unit(q2)))
