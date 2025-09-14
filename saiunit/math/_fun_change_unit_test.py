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
from absl.testing import parameterized

import saiunit as u
import saiunit.math as um
from saiunit import meter, second, volt
from saiunit._base import assert_quantity


class Array(u.CustomArray):
    def __init__(self, value):
        self.data = value

fun_change_unit_unary = [
    'reciprocal', 'var', 'nanvar', 'cbrt', 'square', 'sqrt',
]
fun_change_unit_unary_prod_cumprod = [
    'prod', 'nanprod', 'cumprod', 'nancumprod',
]
fun_change_unit_power = [
    'power', 'float_power',
]
fun_change_unit_binary = [
    'multiply', 'divide', 'cross',
    'true_divide', 'floor_divide', 'convolve',
]
fun_change_unit_binary_divmod = [
    'divmod',
]


class TestFunChangeUnitWithArrayCustomArray(parameterized.TestCase):

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[meter, second]
    )
    def test_fun_change_unit_unary_with_array(self, value, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_change_unit_unary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_unary]

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
            expected_unit = bm_fun._unit_change_fun(unit)
            assert_quantity(result, expected, unit=expected_unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=expected_unit)

            array_input = Array(q)
            result = bm_fun(array_input.data)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=expected_unit)

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[meter, second]
    )
    def test_fun_change_unit_unary_prod_cumprod_with_array(self, value, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_change_unit_unary_prod_cumprod]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_unary_prod_cumprod]

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
            
            size = len(value)
            result_unit = unit ** size
            assert_quantity(result, expected, unit=result_unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=result_unit)

            array_input = Array(q)
            result = bm_fun(array_input.data)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=result_unit)

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        power_exponents=[2, 3],
        unit=[meter, second]
    )
    def test_fun_change_unit_power_with_array(self, value, power_exponents, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_change_unit_power]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_power]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value), power_exponents)
            expected = jnp_fun(jnp.array(value), power_exponents)
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            q = jnp.array(value) * unit
            result = bm_fun(q, power_exponents)
            expected = jnp_fun(jnp.array(value), power_exponents)
            result_unit = unit ** power_exponents
            assert_quantity(result, expected, unit=result_unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=result_unit)

            array_input = Array(q)
            result = bm_fun(array_input.data, power_exponents)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=result_unit)

    @parameterized.product(
        value=[((1.123, 2.567, 3.891), (1.23, 2.34, 3.45)),
               ((1.0, 2.0), (3.0, 4.0))],
        unit1=[meter, second],
        unit2=[volt, second]
    )
    def test_fun_change_unit_binary_with_array(self, value, unit1, unit2):
        bm_fun_list = [getattr(um, fun) for fun in fun_change_unit_binary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_binary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')
            value1, value2 = value

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            q1 = jnp.array(value1) * unit1
            q2 = jnp.array(value2) * unit2
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            expected_unit = bm_fun._unit_change_fun(u.get_unit(unit1), u.get_unit(unit2))
            assert_quantity(result, expected, unit=expected_unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=expected_unit)

            array_input1 = Array(q1)
            array_input2 = Array(q2)
            result = bm_fun(array_input1.data, array_input2.data)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=expected_unit)

    @parameterized.product(
        value=[((1.123, 2.567, 3.891), (1.23, 2.34, 3.45)),
               ((1.0, 2.0), (3.0, 4.0))],
        unit1=[meter, second],
        unit2=[meter, second]
    )
    def test_fun_change_unit_binary_divmod_with_array(self, value, unit1, unit2):
        bm_fun_list = [getattr(um, fun) for fun in fun_change_unit_binary_divmod]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_binary_divmod]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')
            value1, value2 = value

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            for r, e in zip(result, expected):
                assert_quantity(r, e)

            for i, r in enumerate(result):
                array_result = Array(r)
                assert isinstance(array_result, u.CustomArray)
                assert_quantity(array_result.data, expected[i])

            q1 = jnp.array(value1) * unit1
            q2 = jnp.array(value2) * unit2
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result[0], expected[0], unit=unit1 / unit2)
            assert_quantity(result[1], expected[1], unit=unit1)

            array_result1 = Array(result[0])
            array_result2 = Array(result[1])
            assert isinstance(array_result1, u.CustomArray)
            assert isinstance(array_result2, u.CustomArray)
            assert_quantity(array_result1.data, expected[0], unit=unit1 / unit2)
            assert_quantity(array_result2.data, expected[1], unit=unit1)

            array_input1 = Array(q1)
            array_input2 = Array(q2)
            result = bm_fun(array_input1.data, array_input2.data)
            array_result1 = Array(result[0])
            array_result2 = Array(result[1])
            assert isinstance(array_result1, u.CustomArray)
            assert isinstance(array_result2, u.CustomArray)
            assert_quantity(array_result1.data, expected[0], unit=unit1 / unit2)
            assert_quantity(array_result2.data, expected[1], unit=unit1)

    def test_array_with_unit_change_functions(self):
        data = jnp.array([4.0, 9.0, 16.0]) * (meter ** 2)
        test_array = Array(data)
        
        assert isinstance(test_array, u.CustomArray)
        assert hasattr(test_array, 'data')
        assert_quantity(test_array.data, jnp.array([4.0, 9.0, 16.0]), unit=meter ** 2)
        
        sqrt_result = um.sqrt(test_array.data)
        sqrt_array = Array(sqrt_result)
        assert isinstance(sqrt_array, u.CustomArray)
        assert_quantity(sqrt_array.data, jnp.array([2.0, 3.0, 4.0]), unit=meter)
        
        square_result = um.square(test_array.data)
        square_array = Array(square_result)
        assert isinstance(square_array, u.CustomArray)
        assert_quantity(square_array.data, jnp.array([16.0, 81.0, 256.0]), unit=meter ** 4)

    def test_array_with_custom_array_binary_operations(self):
        data1 = jnp.array([2.0, 4.0, 6.0]) * meter
        data2 = jnp.array([1.0, 2.0, 3.0]) * second
        
        array1 = Array(data1)
        array2 = Array(data2)
        
        assert isinstance(array1, u.CustomArray)
        assert isinstance(array2, u.CustomArray)
        
        multiply_result = um.multiply(array1.data, array2.data)
        multiply_array = Array(multiply_result)
        assert isinstance(multiply_array, u.CustomArray)
        assert_quantity(multiply_array.data, jnp.array([2.0, 8.0, 18.0]), unit=meter * second)
        
        divide_result = um.divide(array1.data, array2.data)
        divide_array = Array(divide_result)
        assert isinstance(divide_array, u.CustomArray)
        assert_quantity(divide_array.data, jnp.array([2.0, 2.0, 2.0]), unit=meter / second)

    def test_array_with_power_operations(self):
        base_data = jnp.array([2.0, 3.0, 4.0]) * meter
        test_array = Array(base_data)
        
        assert isinstance(test_array, u.CustomArray)
        
        power_result = um.power(test_array.data, 3)
        power_array = Array(power_result)
        assert isinstance(power_array, u.CustomArray)
        assert_quantity(power_array.data, jnp.array([8.0, 27.0, 64.0]), unit=meter ** 3)
        
        float_power_result = um.float_power(test_array.data, 2)
        float_power_array = Array(float_power_result)
        assert isinstance(float_power_array, u.CustomArray)
        assert_quantity(float_power_array.data, jnp.array([4.0, 9.0, 16.0]), unit=meter ** 2)


class TestFunChangeUnit(parameterized.TestCase):

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[meter, second]
    )
    def test_fun_change_unit_unary(self, value, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_change_unit_unary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_unary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected, unit=bm_fun._unit_change_fun(unit))

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[meter, second]
    )
    def test_fun_change_unit_unary_prod_cumprod(self, value, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_change_unit_unary_prod_cumprod]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_unary_prod_cumprod]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))

            size = len(value)
            result_unit = unit ** size
            assert_quantity(result, expected, unit=result_unit)

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        power_exponents=[2, 3],
        unit=[meter, second]
    )
    def test_fun_change_unit_power(self, value, power_exponents, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_change_unit_power]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_power]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value), power_exponents)
            expected = jnp_fun(jnp.array(value), power_exponents)
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q, power_exponents)
            expected = jnp_fun(jnp.array(value), power_exponents)
            result_unit = unit ** power_exponents
            assert_quantity(result, expected, unit=result_unit)

    @parameterized.product(
        value=[((1.123, 2.567, 3.891), (1.23, 2.34, 3.45)),
               ((1.0, 2.0), (3.0, 4.0),)],
        unit1=[meter, second],
        unit2=[volt, second]
    )
    def test_fun_change_unit_binary(self, value, unit1, unit2):
        bm_fun_list = [getattr(um, fun) for fun in fun_change_unit_binary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_binary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')
            value1, value2 = value

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * unit1
            q2 = value2 * unit2
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected, unit=bm_fun._unit_change_fun(u.get_unit(unit1), u.get_unit(unit2)))

    @parameterized.product(
        value=[((1.123, 2.567, 3.891), (1.23, 2.34, 3.45)),
               ((1.0, 2.0), (3.0, 4.0),)],
        unit1=[meter, second],
        unit2=[meter, second]
    )
    def test_fun_change_unit_binary_divmod(self, value, unit1, unit2):
        bm_fun_list = [getattr(um, fun) for fun in fun_change_unit_binary_divmod]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_binary_divmod]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')
            value1, value2 = value

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            for r, e in zip(result, expected):
                assert_quantity(r, e)

            q1 = value1 * unit1
            q2 = value2 * unit2
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result[0], expected[0], unit=unit1 / unit2)
            assert_quantity(result[1], expected[1], unit=unit1)
