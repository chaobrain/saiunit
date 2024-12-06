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


import jax.lax as lax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

import brainunit as bu
import brainunit.lax as bulax
from brainunit import meter
from brainunit._base import assert_quantity

# math funcs only accept unitless (unary)
lax_accept_unitless_unary = [
    'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh',
    'cumlogsumexp',
    'bessel_i0e', 'bessel_i1e', 'digamma', 'lgamma', 'erf', 'erfc',
    'erf_inv', 'logistic',
]

# math funcs only accept unitless (binary)
lax_accept_unitless_binary = [
    'atan2', 'polygamma', 'igamma', 'igammac', 'igamma_grad_a', 'random_gamma_grad',
    'zeta',
]

# Elementwise bit operations (binary)
lax_bit_operation_binary = [
    'shift_left', 'shift_right_arithmetic', 'shift_right_logical',
]

# fft
lax_fft = [
    'fft',
]

# misc
lax_accept_unitless_misc = [
    'betainc', 'collapse',
]


class TestLaxAcceptUnitless(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLaxAcceptUnitless, self).__init__(*args, **kwargs)

        print()

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)]
    )
    def test_lax_accept_unitless_unary(self, value):
        for fun_name in lax_accept_unitless_unary:
            bulax_fun = getattr(bulax, fun_name)
            lax_fun = getattr(lax, fun_name)
            print(f'fun: {bulax_fun}')

            result = bulax_fun(jnp.array(value))
            expected = lax_fun(jnp.array(value))
            assert_quantity(result, expected)

            for unit, unit2scale in [(bu.ms, bu.second),
                                     (bu.mV, bu.volt),
                                     (bu.mV, bu.mV),
                                     (bu.nA, bu.amp)]:
                q = value * unit
                result = bulax_fun(q, unit_to_scale=unit2scale)
                expected = lax_fun(q.to_decimal(unit2scale))
                assert_quantity(result, expected)

                with pytest.raises(AssertionError):
                    result = bulax_fun(q)

                with pytest.raises(bu.UnitMismatchError):
                    result = bulax_fun(q, unit_to_scale=bu.nS)

    @parameterized.product(
        value=[[(1.0, 2.0), (3.0, 4.0), ],
               [(1.23, 2.34, 3.45), (4.56, 5.67, 6.78)]]
    )
    def test_lax_accept_unitless_binary(self, value):
        value1, value2 = value
        bulax_fun_list = [getattr(bulax, fun) for fun in lax_accept_unitless_binary]
        lax_fun_list = [getattr(lax, fun) for fun in lax_accept_unitless_binary]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')

            result = bulax_fun(jnp.array(value1), jnp.array(value2))
            expected = lax_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * meter
            q2 = value2 * meter
            result = bulax_fun(q1, q2, unit_to_scale=bu.dametre)
            expected = lax_fun(q1.to_decimal(bu.dametre), q2.to_decimal(bu.dametre))
            assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bulax_fun(q1, q2)

            with pytest.raises(bu.UnitMismatchError):
                result = bulax_fun(q1, q2, unit_to_scale=bu.second)

    @parameterized.product(
        value=[[(0, 1), (1, 1)]]
    )
    def test_lax_bit_operation_binary(self, value):
        value1, value2 = value
        bulax_fun_list = [getattr(bulax, fun) for fun in lax_bit_operation_binary]
        lax_fun_list = [getattr(lax, fun) for fun in lax_bit_operation_binary]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')

            result = bulax_fun(jnp.array(value1), jnp.array(value2))
            expected = lax_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * meter
            q2 = value2 * meter
            # result = bm_fun(q1.astype(jnp.bool_).to_value(), q2.astype(jnp.bool_).to_value())
            # expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            # assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bulax_fun(q1, q2)

    @parameterized.product(
        value=[[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
               [(1.23, 2.34, 3.45), (4.56, 5.67, 6.78), (7.89, 8.90, 9.01)]]
    )
    def test_lax_betainc(self, value):
        value1, value2, value3 = value
        fun_name = 'betainc'
        bulax_fun = getattr(bulax, fun_name)
        lax_fun = getattr(lax, fun_name)
        print(f'fun: {bulax_fun}')

        result = bulax_fun(jnp.array(value1), jnp.array(value2), jnp.array(value3))
        expected = lax_fun(jnp.array(value1), jnp.array(value2), jnp.array(value3))
        assert_quantity(result, expected)

        q1 = value1 * meter
        q2 = value2 * meter
        q3 = value3 * meter
        result = bulax_fun(q1, q2, q3, unit_to_scale=bu.dametre)
        expected = lax_fun(q1.to_decimal(bu.dametre), q2.to_decimal(bu.dametre), q3.to_decimal(bu.dametre))
        assert_quantity(result, expected)

        with pytest.raises(AssertionError):
            result = bulax_fun(q1, q2, q3)

        with pytest.raises(bu.UnitMismatchError):
            result = bulax_fun(q1, q2, q3, unit_to_scale=bu.second)

    @parameterized.product(
        value=[[(0, 1), 1]]
    )
    def test_lax_collapse(self, value):
        value1, value2 = value
        fun_name = 'collapse'
        bulax_fun = getattr(bulax, fun_name)
        lax_fun = getattr(lax, fun_name)
        print(f'fun: {bulax_fun}')

        result = bulax_fun(jnp.array(value1), value2)
        expected = lax_fun(jnp.array(value1), value2)
        assert_quantity(result, expected)

        q1 = value1 * meter
        result = bulax_fun(q1, value2, unit_to_scale=bu.dametre)
        expected = lax_fun(q1.to_decimal(bu.dametre), value2)
        assert_quantity(result, expected)

        with pytest.raises(AssertionError):
            result = bulax_fun(q1, value2)

        with pytest.raises(bu.UnitMismatchError):
            result = bulax_fun(q1, value2, unit_to_scale=bu.second)