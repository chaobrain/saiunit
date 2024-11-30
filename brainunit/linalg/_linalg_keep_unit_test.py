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

import brainunit.linalg as bulinalg
from brainunit import second, meter
from brainunit._base import assert_quantity

fun_keep_unit_math_unary_linalg = [
    'norm',
]


class TestLinalgKeepUnit(parameterized.TestCase):
    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[second, meter]
    )
    def test_fun_keep_unit_math_unary_linalg(self, value, unit):
        bm_fun_list = [getattr(bulinalg, fun) for fun in fun_keep_unit_math_unary_linalg]
        jnp_fun_list = [getattr(jnp.linalg, fun) for fun in fun_keep_unit_math_unary_linalg]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected, unit=unit)
