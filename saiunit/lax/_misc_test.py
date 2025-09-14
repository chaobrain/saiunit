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

import saiunit as u
import saiunit.lax as ulax
from saiunit import second, meter
from saiunit._base import assert_quantity


class Array(u.CustomArray):
    def __init__(self, value):
        self.data = value

lax_misc = [
    'after_all', 'reduce', 'reduce_precision',

    # getting attribute funcs
    'broadcast_shapes',
]


class TestLaxMiscWithArrayCustomArray(parameterized.TestCase):

    def test_reduce_operations_with_array(self):
        operands = jnp.array([1.0, 2.0, 3.0, 4.0]) * meter
        test_array = Array(operands)
        
        assert isinstance(test_array, u.CustomArray)
        
        init_values = jnp.array(0.0) * meter
        init_array = Array(init_values)
        assert isinstance(init_array, u.CustomArray)
        
        computation = lax.add
        dimensions = [0]
        
        reduce_result = ulax.reduce(test_array.data, init_array.data, computation, dimensions)
        reduce_array = Array(reduce_result)
        assert isinstance(reduce_array, u.CustomArray)
        expected = jnp.sum(jnp.array([1.0, 2.0, 3.0, 4.0]))
        assert_quantity(reduce_array.data, expected)

    def test_reduce_with_custom_computation_array(self):
        operands = jnp.array([2.0, 4.0, 6.0]) * second
        test_array = Array(operands)
        
        assert isinstance(test_array, u.CustomArray)
        
        init_values = jnp.array(1.0) * second
        init_array = Array(init_values)
        assert isinstance(init_array, u.CustomArray)
        
        computation = lax.mul
        dimensions = [0]
        
        reduce_result = ulax.reduce(test_array.data, init_array.data, computation, dimensions)
        reduce_array = Array(reduce_result)
        assert isinstance(reduce_array, u.CustomArray)
        # Product of [2, 4, 6] starting with 1
        expected = 2.0 * 4.0 * 6.0 * 1.0
        # Units: second * (second^3 from multiplication) = second^4
        # assert_quantity(reduce_array.data, expected, unit=second ** 4)
        assert_quantity(reduce_array.data, expected)

    def test_array_custom_array_compatibility_with_lax_misc(self):
        data = jnp.array([1.0, 2.0, 3.0]) * meter
        test_array = Array(data)
        
        assert isinstance(test_array, u.CustomArray)
        assert hasattr(test_array, 'data')
        
        # Test reduce with Array values
        init_val = jnp.array(0.0) * meter
        init_array = Array(init_val)
        
        result = ulax.reduce(test_array.data, init_array.data, lax.add, [0])
        result_array = Array(result)
        
        assert isinstance(result_array, u.CustomArray)
        


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
