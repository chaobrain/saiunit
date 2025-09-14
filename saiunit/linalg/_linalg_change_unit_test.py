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

import saiunit as bu
import saiunit.linalg as bulinalg
from saiunit import meter, second
from saiunit._base import assert_quantity


class Array(bu.CustomArray):
    def __init__(self, value):
        self.data = value

fun_change_unit_linear_algebra = [
    'dot', 'vdot', 'vecdot', 'inner', 'outer', 'kron', 'matmul',
]

fun_change_unit_linear_algebra_det = [
    'det',
]

fun_change_unit_linear_tensordot = [
    'tensordot',
]


class TestLinalgChangeUnitWithArrayCustomArray(parameterized.TestCase):

    @parameterized.product(
        value=[((1.123, 2.567, 3.891), (1.23, 2.34, 3.45)),
               ((1.0, 2.0), (3.0, 4.0))],
        unit1=[meter, second],
        unit2=[meter, second]
    )
    def test_fun_change_unit_linear_algebra_with_array(self, value, unit1, unit2):
        bm_fun_list = [getattr(bulinalg, fun) for fun in fun_change_unit_linear_algebra]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_linear_algebra]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')
            value1, value2 = value

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.data, expected)

            q1 = jnp.array(value1) * unit1
            q2 = jnp.array(value2) * unit2
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            expected_unit = bm_fun._unit_change_fun(bu.get_unit(unit1), bu.get_unit(unit2))
            assert_quantity(result, expected, unit=expected_unit)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.data, expected, unit=expected_unit)

            array_input1 = Array(q1)
            array_input2 = Array(q2)
            result = bm_fun(array_input1.data, array_input2.data)
            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.data, expected, unit=expected_unit)

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
    def test_fun_change_unit_linear_algebra_det_with_array(self, value, unit):
        bm_fun_list = [getattr(bulinalg, fun) for fun in fun_change_unit_linear_algebra_det]
        jnp_fun_list = [getattr(jnp.linalg, fun) for fun in fun_change_unit_linear_algebra_det]
        value = jnp.array(value)
        
        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(value)
            expected = jnp_fun(value)
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.data, expected)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(value)
            result_unit = unit ** value.shape[-1]
            assert_quantity(result, expected, unit=result_unit)

            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.data, expected, unit=result_unit)

            array_input = Array(q)
            result = bm_fun(array_input.data)
            array_result = Array(result)
            assert isinstance(array_result, bu.CustomArray)
            assert_quantity(array_result.data, expected, unit=result_unit)

    def test_dot_operations_with_array(self):
        # Test dot product
        vec1 = jnp.array([1.0, 2.0, 3.0]) * meter
        vec2 = jnp.array([4.0, 5.0, 6.0]) * second
        
        array1 = Array(vec1)
        array2 = Array(vec2)
        
        assert isinstance(array1, bu.CustomArray)
        assert isinstance(array2, bu.CustomArray)
        
        dot_result = bulinalg.dot(array1.data, array2.data)
        dot_array = Array(dot_result)
        assert isinstance(dot_array, bu.CustomArray)
        expected = jnp.dot(jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))
        assert_quantity(dot_array.data, expected, unit=meter * second)

    def test_matmul_operations_with_array(self):
        # Test matrix multiplication
        mat1 = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * meter
        mat2 = jnp.array([[5.0, 6.0], [7.0, 8.0]]) * second
        
        array1 = Array(mat1)
        array2 = Array(mat2)
        
        assert isinstance(array1, bu.CustomArray)
        assert isinstance(array2, bu.CustomArray)
        
        matmul_result = bulinalg.matmul(array1.data, array2.data)
        matmul_array = Array(matmul_result)
        assert isinstance(matmul_array, bu.CustomArray)
        expected = jnp.matmul(jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([[5.0, 6.0], [7.0, 8.0]]))
        assert_quantity(matmul_array.data, expected, unit=meter * second)

    def test_outer_product_with_array(self):
        # Test outer product
        vec1 = jnp.array([1.0, 2.0]) * meter
        vec2 = jnp.array([3.0, 4.0, 5.0]) * second
        
        array1 = Array(vec1)
        array2 = Array(vec2)
        
        assert isinstance(array1, bu.CustomArray)
        assert isinstance(array2, bu.CustomArray)
        
        outer_result = bulinalg.outer(array1.data, array2.data)
        outer_array = Array(outer_result)
        assert isinstance(outer_array, bu.CustomArray)
        expected = jnp.outer(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0, 5.0]))
        assert_quantity(outer_array.data, expected, unit=meter * second)

    def test_kron_product_with_array(self):
        # Test Kronecker product
        mat1 = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * meter
        mat2 = jnp.array([[0.0, 5.0], [6.0, 7.0]]) * second
        
        array1 = Array(mat1)
        array2 = Array(mat2)
        
        assert isinstance(array1, bu.CustomArray)
        assert isinstance(array2, bu.CustomArray)
        
        kron_result = bulinalg.kron(array1.data, array2.data)
        kron_array = Array(kron_result)
        assert isinstance(kron_array, bu.CustomArray)
        expected = jnp.kron(jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([[0.0, 5.0], [6.0, 7.0]]))
        assert_quantity(kron_array.data, expected, unit=meter * second)

    def test_det_operations_with_array(self):
        # Test 2x2 determinant
        mat_2x2 = jnp.array([[2.0, 3.0], [1.0, 4.0]]) * meter
        array_2x2 = Array(mat_2x2)
        
        assert isinstance(array_2x2, bu.CustomArray)
        
        det_result = bulinalg.det(array_2x2.data)
        det_array = Array(det_result)
        assert isinstance(det_array, bu.CustomArray)
        expected = jnp.linalg.det(jnp.array([[2.0, 3.0], [1.0, 4.0]]))
        assert_quantity(det_array.data, expected, unit=meter ** 2)
        
        # Test 3x3 determinant
        mat_3x3 = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]) * second
        array_3x3 = Array(mat_3x3)
        
        assert isinstance(array_3x3, bu.CustomArray)
        
        det_result = bulinalg.det(array_3x3.data)
        det_array = Array(det_result)
        assert isinstance(det_array, bu.CustomArray)
        expected = jnp.linalg.det(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]))
        assert_quantity(det_array.data, expected, unit=second ** 3)

    def test_tensordot_operations_with_array(self):
        # Test tensordot
        tensor1 = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * meter
        tensor2 = jnp.array([[5.0, 6.0], [7.0, 8.0]]) * second
        
        array1 = Array(tensor1)
        array2 = Array(tensor2)
        
        assert isinstance(array1, bu.CustomArray)
        assert isinstance(array2, bu.CustomArray)
        
        tensordot_result = bulinalg.tensordot(array1.data, array2.data)
        tensordot_array = Array(tensordot_result)
        assert isinstance(tensordot_array, bu.CustomArray)
        expected = jnp.tensordot(jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([[5.0, 6.0], [7.0, 8.0]]))
        assert_quantity(tensordot_array.data, expected, unit=meter * second)

    def test_vdot_operations_with_array(self):
        # Test complex dot product (vdot)
        vec1 = jnp.array([1.0 + 2.0j, 3.0 + 4.0j]) * meter
        vec2 = jnp.array([5.0 + 6.0j, 7.0 + 8.0j]) * second
        
        array1 = Array(vec1)
        array2 = Array(vec2)
        
        assert isinstance(array1, bu.CustomArray)
        assert isinstance(array2, bu.CustomArray)
        
        vdot_result = bulinalg.vdot(array1.data, array2.data)
        vdot_array = Array(vdot_result)
        assert isinstance(vdot_array, bu.CustomArray)
        expected = jnp.vdot(jnp.array([1.0 + 2.0j, 3.0 + 4.0j]), jnp.array([5.0 + 6.0j, 7.0 + 8.0j]))
        assert_quantity(vdot_array.data, expected, unit=meter * second)

    def test_multi_dot_with_array(self):
        # Test multi_dot functionality with Array instances
        key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
        mat1 = jax.random.normal(key1, shape=(10, 5)) * bu.mA
        mat2 = jax.random.normal(key2, shape=(5, 8)) * bu.mV
        mat3 = jax.random.normal(key3, shape=(8, 3)) * bu.ohm
        
        array1 = Array(mat1)
        array2 = Array(mat2)
        array3 = Array(mat3)
        
        assert isinstance(array1, bu.CustomArray)
        assert isinstance(array2, bu.CustomArray)
        assert isinstance(array3, bu.CustomArray)
        
        # Test that multi_dot works with Array values
        result1 = (array1.data @ array2.data) @ array3.data
        result2 = array1.data @ (array2.data @ array3.data)
        result3 = bulinalg.multi_dot([array1.data, array2.data, array3.data])
        
        result1_array = Array(result1)
        result2_array = Array(result2)
        result3_array = Array(result3)
        
        assert isinstance(result1_array, bu.CustomArray)
        assert isinstance(result2_array, bu.CustomArray)
        assert isinstance(result3_array, bu.CustomArray)
        
        # Verify results are equivalent
        expected_unit = bu.mA * bu.mV * bu.ohm
        assert bu.math.allclose(result1_array.data, result3_array.data, atol=1E-4 * expected_unit)
        assert bu.math.allclose(result2_array.data, result3_array.data, atol=1E-4 * expected_unit)


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
