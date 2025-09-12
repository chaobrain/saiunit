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
import saiunit.linalg as bulinalg
from saiunit import second, meter
from saiunit._base import assert_quantity


class Array(u.CustomArray):
    def __init__(self, value):
        self.value = value

fun_keep_unit_math_unary_linalg = [
    'norm',
]


class TestLinalgKeepUnitWithArrayCustomArray(parameterized.TestCase):

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[second, meter]
    )
    def test_fun_keep_unit_math_unary_linalg_with_array(self, value, unit):
        bm_fun_list = [getattr(bulinalg, fun) for fun in fun_keep_unit_math_unary_linalg]
        jnp_fun_list = [getattr(jnp.linalg, fun) for fun in fun_keep_unit_math_unary_linalg]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.value, expected)

            q = jnp.array(value) * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected, unit=unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.value, expected, unit=unit)

            array_input = Array(q)
            result = bm_fun(array_input.value)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.value, expected, unit=unit)

    def test_norm_with_different_vector_types_array(self):
        # Test 1D vector norm
        vector_1d = jnp.array([3.0, 4.0]) * meter
        test_array_1d = Array(vector_1d)
        
        assert isinstance(test_array_1d, u.CustomArray)
        
        norm_result = bulinalg.norm(test_array_1d.value)
        norm_array = Array(norm_result)
        assert isinstance(norm_array, u.CustomArray)
        expected = jnp.linalg.norm(jnp.array([3.0, 4.0]))
        assert_quantity(norm_array.value, expected, unit=meter)
        
        # Test 2D matrix Frobenius norm
        matrix_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * second
        test_array_2d = Array(matrix_2d)
        
        assert isinstance(test_array_2d, u.CustomArray)
        
        norm_result = bulinalg.norm(test_array_2d.value)
        norm_array = Array(norm_result)
        assert isinstance(norm_array, u.CustomArray)
        expected = jnp.linalg.norm(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        assert_quantity(norm_array.value, expected, unit=second)

    def test_norm_with_different_ord_parameters_array(self):
        vector = jnp.array([1.0, 2.0, 3.0]) * meter
        test_array = Array(vector)
        
        assert isinstance(test_array, u.CustomArray)
        
        # Test L1 norm
        norm_l1_result = bulinalg.norm(test_array.value, ord=1)
        norm_l1_array = Array(norm_l1_result)
        assert isinstance(norm_l1_array, u.CustomArray)
        expected_l1 = jnp.linalg.norm(jnp.array([1.0, 2.0, 3.0]), ord=1)
        assert_quantity(norm_l1_array.value, expected_l1, unit=meter)
        
        # Test L2 norm (default)
        norm_l2_result = bulinalg.norm(test_array.value, ord=2)
        norm_l2_array = Array(norm_l2_result)
        assert isinstance(norm_l2_array, u.CustomArray)
        expected_l2 = jnp.linalg.norm(jnp.array([1.0, 2.0, 3.0]), ord=2)
        assert_quantity(norm_l2_array.value, expected_l2, unit=meter)
        
        # Test infinity norm
        norm_inf_result = bulinalg.norm(test_array.value, ord=jnp.inf)
        norm_inf_array = Array(norm_inf_result)
        assert isinstance(norm_inf_array, u.CustomArray)
        expected_inf = jnp.linalg.norm(jnp.array([1.0, 2.0, 3.0]), ord=jnp.inf)
        assert_quantity(norm_inf_array.value, expected_inf, unit=meter)

    def test_norm_with_axis_parameter_array(self):
        matrix = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * second
        test_array = Array(matrix)
        
        assert isinstance(test_array, u.CustomArray)
        
        # Test norm along axis=0
        norm_axis0_result = bulinalg.norm(test_array.value, axis=0)
        norm_axis0_array = Array(norm_axis0_result)
        assert isinstance(norm_axis0_array, u.CustomArray)
        expected_axis0 = jnp.linalg.norm(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), axis=0)
        assert_quantity(norm_axis0_array.value, expected_axis0, unit=second)
        
        # Test norm along axis=1
        norm_axis1_result = bulinalg.norm(test_array.value, axis=1)
        norm_axis1_array = Array(norm_axis1_result)
        assert isinstance(norm_axis1_array, u.CustomArray)
        expected_axis1 = jnp.linalg.norm(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), axis=1)
        assert_quantity(norm_axis1_array.value, expected_axis1, unit=second)

    def test_norm_with_complex_values_array(self):
        complex_vector = jnp.array([1.0 + 2.0j, 3.0 + 4.0j]) * meter
        test_array = Array(complex_vector)
        
        assert isinstance(test_array, u.CustomArray)
        
        norm_result = bulinalg.norm(test_array.value)
        norm_array = Array(norm_result)
        assert isinstance(norm_array, u.CustomArray)
        expected = jnp.linalg.norm(jnp.array([1.0 + 2.0j, 3.0 + 4.0j]))
        assert_quantity(norm_array.value, expected, unit=meter)

    def test_array_custom_array_compatibility_with_norm(self):
        data = jnp.array([6.0, 8.0]) * meter
        test_array = Array(data)
        
        assert isinstance(test_array, u.CustomArray)
        assert hasattr(test_array, 'value')
        
        # Test that we can use the array value in norm function
        result = bulinalg.norm(test_array.value)
        result_array = Array(result)
        
        assert isinstance(result_array, u.CustomArray)
        
        # Compare with direct computation
        direct_result = bulinalg.norm(data)
        assert_quantity(result_array.value, direct_result.mantissa, unit=meter)
        
        # Verify the result is correct (6-8-10 triangle)
        expected_norm = 10.0
        assert_quantity(result_array.value, expected_norm, unit=meter)


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
