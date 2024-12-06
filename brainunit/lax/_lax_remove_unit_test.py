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
import jax.lax as lax
import pytest
from absl.testing import parameterized

import brainunit as bu
import brainunit.lax as bulax
from brainunit import meter, second
from brainunit._base import assert_quantity

lax_remove_unit_unary = [
    'population_count', 'clz',
]

lax_logic_funcs_binary = [
    'eq', 'ne', 'ge', 'gt', 'le', 'lt',
]

class TestLaxRemoveUnit(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLaxRemoveUnit, self).__init__(*args, **kwargs)

        print()

    @parameterized.product(
        value=[(1, 2), (1, 2, 3)],
        unit=[meter, second],
    )
    def test_lax_remove_unit_unary(self, value, unit):
        bulax_fun_list = [getattr(bulax, fun) for fun in lax_remove_unit_unary]
        lax_fun_list = [getattr(lax, fun) for fun in lax_remove_unit_unary]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')

            result = bulax_fun(jnp.array(value))
            expected = lax_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bulax_fun(q)
            expected = lax_fun(jnp.array(value))
            assert_quantity(result, expected)

    @parameterized.product(
        value=[((1.0, 2.0), (3.0, 4.0)),
               ((1.23, 2.34, 3.45), (1.23, 2.34, 3.45))],
        unit=[meter, second],
    )
    def test_lax_remove_unit_logic_binary(self, value, unit):
        bulax_fun_list = [getattr(bulax, fun) for fun in lax_logic_funcs_binary]
        lax_fun_list = [getattr(lax, fun) for fun in lax_logic_funcs_binary]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')

            x1, x2 = value
            result = bulax_fun(jnp.array(x1), jnp.array(x2))
            expected = lax_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            q1 = x1 * unit
            q2 = x2 * unit
            result = bulax_fun(q1, q2)
            expected = lax_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bulax_fun(jnp.array(x1), q2)

            with pytest.raises(AssertionError):
                result = bulax_fun(q1, jnp.array(x2))