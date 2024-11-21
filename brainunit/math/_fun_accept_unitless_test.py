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
from brainunit import meter
from brainunit._base import assert_quantity

fun_accept_unitless_unary = [
    'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2',
    'deg2rad', 'rad2deg', 'degrees', 'radians', 'angle',
    'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
    'arctanh', 'cos', 'cosh', 'sin', 'sinc', 'sinh', 'tan',
    'tanh',
]

fun_accept_unitless_binary = [
    'hypot', 'arctan2', 'logaddexp', 'logaddexp2',
    'corrcoef', 'correlate', 'cov',
]
fun_accept_unitless_binary_ldexp = [
    'ldexp',
]

fun_elementwise_bit_operation_unary = [
    'bitwise_not', 'invert',
]
fun_elementwise_bit_operation_binary = [
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'left_shift', 'right_shift',
]


class TestFunAcceptUnitless(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFunAcceptUnitless, self).__init__(*args, **kwargs)

        print()

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)]
    )
    def test_fun_accept_unitless_unary_1(self, value):
        for fun_name in fun_accept_unitless_unary:
            fun = getattr(bu.math, fun_name)
            jnp_fun = getattr(jnp, fun_name)
            print(f'fun: {fun}')

            result = fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            for unit, unit2scale in [(bu.ms, bu.second),
                                     (bu.mV, bu.volt),
                                     (bu.mV, bu.mV),
                                     (bu.nA, bu.amp)]:
                q = value * unit
                result = fun(q, unit_to_scale=unit2scale)
                expected = jnp_fun(q.to_decimal(unit2scale))
                assert_quantity(result, expected)

                with pytest.raises(AssertionError):
                    result = fun(q)

                with pytest.raises(bu.UnitMismatchError):
                    result = fun(q, unit_to_scale=bu.nS)

    @parameterized.product(
        value=[[(1.0, 2.0), (3.0, 4.0), ],
               [(1.23, 2.34, 3.45), (4.56, 5.67, 6.78)]]
    )
    def test_func_accept_unitless_binary(self, value):
        value1, value2 = value
        bm_fun_list = [getattr(bm, fun) for fun in fun_accept_unitless_binary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_accept_unitless_binary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * meter
            q2 = value2 * meter
            result = bm_fun(q1, q2, unit_to_scale=bu.dametre)
            expected = jnp_fun(q1.to_decimal(bu.dametre), q2.to_decimal(bu.dametre))
            assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(q1, q2)

            with pytest.raises(bu.UnitMismatchError):
                result = bm_fun(q1, q2, unit_to_scale=bu.second)

    @parameterized.product(
        value=[[(1.0, 2.0), (3, 4), ],
               [(1.23, 2.34, 3.45), (4, 5, 6)]]
    )
    def test_func_accept_unitless_binary_ldexp(self, value):
        value1, value2 = value
        bm_fun_list = [getattr(bm, fun) for fun in fun_accept_unitless_binary_ldexp]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_accept_unitless_binary_ldexp]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * meter
            q2 = value2 * meter
            result = bm_fun(q1.to_decimal(meter), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(q1, q2)

    @parameterized.product(
        value=[(1, 2), (1, 2, 3)]
    )
    def test_elementwise_bit_operation_unary(self, value):
        bm_fun_list = [getattr(bm, fun) for fun in fun_elementwise_bit_operation_unary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_elementwise_bit_operation_unary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * meter
            # result = bm_fun(q.astype(jnp.int32).to_value())
            # expected = jnp_fun(jnp.array(value))
            # assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(q)

    @parameterized.product(
        value=[[(0, 1), (1, 1)],
               [(True, False, True, False), (False, False, True, True)]]
    )
    def test_elementwise_bit_operation_binary(self, value):
        value1, value2 = value
        bm_fun_list = [getattr(bm, fun) for fun in fun_elementwise_bit_operation_binary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_elementwise_bit_operation_binary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * meter
            q2 = value2 * meter
            # result = bm_fun(q1.astype(jnp.bool_).to_value(), q2.astype(jnp.bool_).to_value())
            # expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            # assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(q1, q2)

    def test_dimensionless(self):
        a = bu.Quantity(1.0)

        for fun_name in fun_accept_unitless_unary:
            r1 = getattr(bu.math, fun_name)(a)
            r2 = getattr(jnp, fun_name)(a.to_decimal())
            print(fun_name, r1, r2)
            self.assertTrue(jnp.allclose(r1, r2, equal_nan=True))

        b = bu.Quantity(2.0)

        for fun_name in ['hypot', 'arctan2', 'logaddexp', 'logaddexp2', ]:
            r1 = getattr(bu.math, fun_name)(a, b)
            r2 = getattr(jnp, fun_name)(a.to_decimal(), b.to_decimal())
            print(fun_name, r1, r2)
            self.assertTrue(jnp.allclose(r1, r2, equal_nan=True))
