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

import brainunit as bu
import brainunit.linalg as bulinalg
from brainunit import meter, second, volt
from brainunit._base import assert_quantity

fun_change_unit_linear_algebra = [
    'dot', 'vdot', 'vecdot', 'inner', 'outer', 'kron', 'matmul',
]

fun_change_unit_linear_algebra_det = [
    'det',
]

fun_change_unit_linear_tensordot = [
    'tensordot',
]

class TestLinalgChangeUnit(parameterized.TestCase):
    @parameterized.product(
        value=[((1.123, 2.567, 3.891), (1.23, 2.34, 3.45)),
               ((1.0, 2.0), (3.0, 4.0),)],
        unit1=[meter, second],
        unit2=[meter, second]
    )
    def test_fun_change_unit_linear_algebra(self, value, unit1, unit2):
        bm_fun_list = [getattr(bulinalg, fun) for fun in fun_change_unit_linear_algebra]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_linear_algebra]

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
            assert_quantity(result, expected, unit=bm_fun._unit_change_fun(bu.get_unit(unit1), bu.get_unit(unit2)))

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
        unit=[meter, second],
    )
    def test_fun_change_unit_linear_algebra_det(self, value, unit):
        bm_fun_list = [getattr(bulinalg, fun) for fun in fun_change_unit_linear_algebra_det]
        jnp_fun_list = [getattr(jnp.linalg, fun) for fun in fun_change_unit_linear_algebra_det]
        value = jnp.array(value)
        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(value)
            expected = jnp_fun(value)
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(value)

            result_unit = unit ** value.shape[-1]

            assert_quantity(result, expected, unit=result_unit)

    @parameterized.product(
        value=[(((1, 2), (3, 4)), ((1, 2), (3, 4))), ],
        unit1=[meter, second],
        unit2=[meter, second]
    )
    def test_fun_change_unit_tensordot(self, value, unit1, unit2):
        bm_fun_list = [getattr(bulinalg, fun) for fun in fun_change_unit_linear_tensordot]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_linear_tensordot]

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
            assert_quantity(result, expected, unit=bm_fun._unit_change_fun(bu.get_unit(unit1), bu.get_unit(unit2)))

    def test_multi_dot(self):
        key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
        x = jax.random.normal(key1, shape=(200, 5)) * bu.mA
        y = jax.random.normal(key2, shape=(5, 100)) * bu.mV
        z = jax.random.normal(key3, shape=(100, 10)) * bu.ohm
        result1 = (x @ y) @ z
        result2 = x @ (y @ z)
        assert bu.math.allclose(result1, result2, atol=1E-4 * result1.unit)
        result3 = bu.linalg.multi_dot([x, y, z])
        assert bu.math.allclose(result1, result3, atol=1E-4 * result1.unit)
        assert jax.jit(lambda x, y, z: (x @ y) @ z).lower(x, y, z).cost_analysis()['flops'] == 600000.0
        assert jax.jit(lambda x, y, z: x @ (y @ z)).lower(x, y, z).cost_analysis()['flops'] == 30000.0
        assert jax.jit(bu.linalg.multi_dot).lower([x, y, z]).cost_analysis()['flops'] == 30000.0
