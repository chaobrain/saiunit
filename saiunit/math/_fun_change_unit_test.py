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


import jax
import jax.numpy as jnp
from absl.testing import parameterized

import saiunit as u
import saiunit.math as um
from saiunit import meter, second, volt
from saiunit._base import assert_quantity

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

