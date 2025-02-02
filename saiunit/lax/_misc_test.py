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
from absl.testing import parameterized

import saiunit.lax as ulax
from saiunit._base import assert_quantity

lax_misc = [
    'after_all', 'reduce', 'reduce_precision',

    # getting attribute funcs
    'broadcast_shapes',
]


class TestLaxMisc(parameterized.TestCase):
    # def test_after_all(self):
    #     token1 = lax.create_token()
    #     token2 = lax.create_token()
    #
    #     result = ulax.after_all(token1, token2)
    #     expected = lax.after_all(token1, token2)
    #     assert_quantity(result, expected)

    def test_reduce(self):
        operands = jnp.array([1.0, 2.0, 3.0])
        init_values = jnp.array(0.0)
        computation = lax.add
        dimensions = [0]

        result = ulax.reduce(operands, init_values, computation, dimensions)
        expected = jnp.sum(operands)  # 使用 lax.add 进行 reduce 相当于求和
        assert_quantity(result, expected)

    def test_reduce_precision(self):
        operand = jnp.array([1.123456, 2.123456], dtype=jnp.float32)
        exponent_bits = 5
        mantissa_bits = 10

        result = ulax.reduce_precision(operand, exponent_bits, mantissa_bits)
        expected = lax.reduce_precision(operand, exponent_bits, mantissa_bits)
        assert_quantity(result, expected)

    def test_broadcast_shapes(self):
        shape1 = (2, 3)
        shape2 = (3,)
        results = ulax.broadcast_shapes(shape1, shape2)
        expecteds = lax.broadcast_shapes(shape1, shape2)

        for result, expected in zip(results, expecteds):
            self.assertTrue(result == expected)
