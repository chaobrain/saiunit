# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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
import saiunit.linalg as bulinalg
from saiunit import meter, second
from saiunit._base_getters import assert_quantity


class Array(u.CustomArray):
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
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            q1 = jnp.array(value1) * unit1
            q2 = jnp.array(value2) * unit2
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            expected_unit = bm_fun._unit_change_fun(u.get_unit(unit1), u.get_unit(unit2))
            assert_quantity(result, expected, unit=expected_unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=expected_unit)

            array_input1 = Array(q1)
            array_input2 = Array(q2)
            result = bm_fun(array_input1.data, array_input2.data)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
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
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(value)
            result_unit = unit ** value.shape[-1]
            assert_quantity(result, expected, unit=result_unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=result_unit)

            array_input = Array(q)
            result = bm_fun(array_input.data)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=result_unit)

    def test_dot_operations_with_array(self):
        # Test dot product
        vec1 = jnp.array([1.0, 2.0, 3.0]) * meter
        vec2 = jnp.array([4.0, 5.0, 6.0]) * second
        
        array1 = Array(vec1)
        array2 = Array(vec2)
        
        assert isinstance(array1, u.CustomArray)
        assert isinstance(array2, u.CustomArray)
        
        dot_result = bulinalg.dot(array1.data, array2.data)
        dot_array = Array(dot_result)
        assert isinstance(dot_array, u.CustomArray)
        expected = jnp.dot(jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))
        assert_quantity(dot_array.data, expected, unit=meter * second)

    def test_matmul_operations_with_array(self):
        # Test matrix multiplication
        mat1 = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * meter
        mat2 = jnp.array([[5.0, 6.0], [7.0, 8.0]]) * second
        
        array1 = Array(mat1)
        array2 = Array(mat2)
        
        assert isinstance(array1, u.CustomArray)
        assert isinstance(array2, u.CustomArray)
        
        matmul_result = bulinalg.matmul(array1.data, array2.data)
        matmul_array = Array(matmul_result)
        assert isinstance(matmul_array, u.CustomArray)
        expected = jnp.matmul(jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([[5.0, 6.0], [7.0, 8.0]]))
        assert_quantity(matmul_array.data, expected, unit=meter * second)

    def test_outer_product_with_array(self):
        # Test outer product
        vec1 = jnp.array([1.0, 2.0]) * meter
        vec2 = jnp.array([3.0, 4.0, 5.0]) * second
        
        array1 = Array(vec1)
        array2 = Array(vec2)
        
        assert isinstance(array1, u.CustomArray)
        assert isinstance(array2, u.CustomArray)
        
        outer_result = bulinalg.outer(array1.data, array2.data)
        outer_array = Array(outer_result)
        assert isinstance(outer_array, u.CustomArray)
        expected = jnp.outer(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0, 5.0]))
        assert_quantity(outer_array.data, expected, unit=meter * second)

    def test_kron_product_with_array(self):
        # Test Kronecker product
        mat1 = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * meter
        mat2 = jnp.array([[0.0, 5.0], [6.0, 7.0]]) * second
        
        array1 = Array(mat1)
        array2 = Array(mat2)
        
        assert isinstance(array1, u.CustomArray)
        assert isinstance(array2, u.CustomArray)
        
        kron_result = bulinalg.kron(array1.data, array2.data)
        kron_array = Array(kron_result)
        assert isinstance(kron_array, u.CustomArray)
        expected = jnp.kron(jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([[0.0, 5.0], [6.0, 7.0]]))
        assert_quantity(kron_array.data, expected, unit=meter * second)

    def test_det_operations_with_array(self):
        # Test 2x2 determinant
        mat_2x2 = jnp.array([[2.0, 3.0], [1.0, 4.0]]) * meter
        array_2x2 = Array(mat_2x2)
        
        assert isinstance(array_2x2, u.CustomArray)
        
        det_result = bulinalg.det(array_2x2.data)
        det_array = Array(det_result)
        assert isinstance(det_array, u.CustomArray)
        expected = jnp.linalg.det(jnp.array([[2.0, 3.0], [1.0, 4.0]]))
        assert_quantity(det_array.data, expected, unit=meter ** 2)
        
        # Test 3x3 determinant
        mat_3x3 = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]) * second
        array_3x3 = Array(mat_3x3)
        
        assert isinstance(array_3x3, u.CustomArray)
        
        det_result = bulinalg.det(array_3x3.data)
        det_array = Array(det_result)
        assert isinstance(det_array, u.CustomArray)
        expected = jnp.linalg.det(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]))
        assert_quantity(det_array.data, expected, unit=second ** 3)

    def test_tensordot_operations_with_array(self):
        # Test tensordot
        tensor1 = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * meter
        tensor2 = jnp.array([[5.0, 6.0], [7.0, 8.0]]) * second
        
        array1 = Array(tensor1)
        array2 = Array(tensor2)
        
        assert isinstance(array1, u.CustomArray)
        assert isinstance(array2, u.CustomArray)
        
        tensordot_result = bulinalg.tensordot(array1.data, array2.data)
        tensordot_array = Array(tensordot_result)
        assert isinstance(tensordot_array, u.CustomArray)
        expected = jnp.tensordot(jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([[5.0, 6.0], [7.0, 8.0]]))
        assert_quantity(tensordot_array.data, expected, unit=meter * second)

    def test_vdot_operations_with_array(self):
        # Test complex dot product (vdot)
        vec1 = jnp.array([1.0 + 2.0j, 3.0 + 4.0j]) * meter
        vec2 = jnp.array([5.0 + 6.0j, 7.0 + 8.0j]) * second
        
        array1 = Array(vec1)
        array2 = Array(vec2)
        
        assert isinstance(array1, u.CustomArray)
        assert isinstance(array2, u.CustomArray)
        
        vdot_result = bulinalg.vdot(array1.data, array2.data)
        vdot_array = Array(vdot_result)
        assert isinstance(vdot_array, u.CustomArray)
        expected = jnp.vdot(jnp.array([1.0 + 2.0j, 3.0 + 4.0j]), jnp.array([5.0 + 6.0j, 7.0 + 8.0j]))
        assert_quantity(vdot_array.data, expected, unit=meter * second)

    def test_multi_dot_with_array(self):
        # Test multi_dot functionality with Array instances
        key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
        mat1 = jax.random.normal(key1, shape=(10, 5)) * u.mA
        mat2 = jax.random.normal(key2, shape=(5, 8)) * u.mV
        mat3 = jax.random.normal(key3, shape=(8, 3)) * u.ohm
        
        array1 = Array(mat1)
        array2 = Array(mat2)
        array3 = Array(mat3)
        
        assert isinstance(array1, u.CustomArray)
        assert isinstance(array2, u.CustomArray)
        assert isinstance(array3, u.CustomArray)
        
        # Test that multi_dot works with Array values
        result1 = (array1.data @ array2.data) @ array3.data
        result2 = array1.data @ (array2.data @ array3.data)
        result3 = bulinalg.multi_dot([array1.data, array2.data, array3.data])
        
        result1_array = Array(result1)
        result2_array = Array(result2)
        result3_array = Array(result3)
        
        assert isinstance(result1_array, u.CustomArray)
        assert isinstance(result2_array, u.CustomArray)
        assert isinstance(result3_array, u.CustomArray)
        
        # Verify results are equivalent
        expected_unit = u.mA * u.mV * u.ohm
        assert u.math.allclose(result1_array.data, result3_array.data, atol=1E-4 * expected_unit)
        assert u.math.allclose(result2_array.data, result3_array.data, atol=1E-4 * expected_unit)


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
            assert_quantity(result, expected, unit=bm_fun._unit_change_fun(u.get_unit(unit1), u.get_unit(unit2)))

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
            assert_quantity(result, expected, unit=bm_fun._unit_change_fun(u.get_unit(unit1), u.get_unit(unit2)))

    def test_multi_dot(self):
        key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
        x = jax.random.normal(key1, shape=(200, 5)) * u.mA
        y = jax.random.normal(key2, shape=(5, 100)) * u.mV
        z = jax.random.normal(key3, shape=(100, 10)) * u.ohm
        result1 = (x @ y) @ z
        result2 = x @ (y @ z)
        assert u.math.allclose(result1, result2, atol=1E-4 * result1.unit)
        result3 = u.linalg.multi_dot([x, y, z])
        assert u.math.allclose(result1, result3, atol=1E-4 * result1.unit)
        assert jax.jit(lambda x, y, z: (x @ y) @ z).lower(x, y, z).cost_analysis()['flops'] == 600000.0
        assert jax.jit(lambda x, y, z: x @ (y @ z)).lower(x, y, z).cost_analysis()['flops'] == 30000.0
        assert jax.jit(u.linalg.multi_dot).lower([x, y, z]).cost_analysis()['flops'] == 30000.0

    def test_cholesky_supports_symmetrize_input(self):
        a = jnp.array([[2.0, 1.0], [1.0, 2.0]])

        result = bulinalg.cholesky(a, symmetrize_input=False)
        expected = jnp.linalg.cholesky(a, symmetrize_input=False)
        assert_quantity(result, expected)

        q = a * meter * meter
        result = bulinalg.cholesky(q, symmetrize_input=False)
        assert_quantity(result, expected, unit=meter)


# --- Docstring example tests ---


def test_docstring_example_cholesky():
    """Verify the cholesky docstring example."""
    import saiunit as u
    import jax.numpy as jnp

    x = jnp.array([[2., 1.],
                    [1., 2.]]) * u.meter2
    L = u.linalg.cholesky(x)
    assert L.unit == u.meter
    assert u.math.allclose(x, L @ L.T)


def test_docstring_example_solve():
    """Verify the solve docstring example."""
    import saiunit as u
    import jax.numpy as jnp

    A = jnp.array([[1., 2., 3.],
                    [2., 4., 2.],
                    [3., 2., 1.]]) * u.meter
    b = jnp.array([14., 16., 10.]) * u.second
    x = u.linalg.solve(A, b)
    assert x.unit == u.second / u.meter
    assert u.math.allclose(A @ x, b)


def test_docstring_example_tensorsolve():
    """Verify the tensorsolve docstring example."""
    import saiunit as u
    import jax

    key1, key2 = jax.random.split(jax.random.key(8675309))
    a = jax.random.normal(key1, shape=(2, 2, 4)) * u.meter
    b = jax.random.normal(key2, shape=(2, 2)) * u.second
    x = u.linalg.tensorsolve(a, b)
    assert x.shape == (4,)
    b_reconstructed = u.linalg.tensordot(a, x, axes=x.ndim)
    assert u.math.allclose(b, b_reconstructed)


def test_docstring_example_lstsq():
    """Verify the lstsq docstring example."""
    import saiunit as u
    import jax.numpy as jnp

    a = jnp.array([[1, 2],
                    [3, 4]]) * u.second
    b = jnp.array([5, 6]) * u.meter
    x, residuals, rank, s = u.linalg.lstsq(a, b)
    assert x.unit == u.meter / u.second


def test_docstring_example_inv():
    """Verify the inv docstring example."""
    import saiunit as u
    import jax.numpy as jnp

    a = jnp.array([[1., 2., 3.],
                    [2., 4., 2.],
                    [3., 2., 1.]]) * u.second
    a_inv = u.linalg.inv(a)
    assert u.math.allclose(a @ a_inv, jnp.eye(3), atol=1e-5)


def test_docstring_example_pinv():
    """Verify the pinv docstring example."""
    import saiunit as u
    import jax.numpy as jnp

    a = jnp.array([[1, 2],
                    [3, 4],
                    [5, 6]]) * u.second
    a_pinv = u.linalg.pinv(a)
    assert a_pinv.shape == (2, 3)
    assert u.math.allclose(a_pinv @ a, jnp.eye(2), atol=1e-4)


def test_docstring_example_tensorinv():
    """Verify the tensorinv docstring example."""
    import saiunit as u
    import jax
    import jax.numpy as jnp

    key = jax.random.key(1337)
    x = jax.random.normal(key, shape=(2, 2, 4)) * u.second
    xinv = u.linalg.tensorinv(x, 2)
    assert xinv.shape == (4, 2, 2)
    xinv_x = u.linalg.tensordot(xinv, x, axes=2)
    assert u.math.allclose(xinv_x, jnp.eye(4), atol=1e-4)


def test_docstring_example_dot():
    """Verify the dot docstring example (re-exported from math)."""
    import saiunit as u
    import jax.numpy as jnp

    a = jnp.array([1.0, 2.0, 3.0]) * u.meter
    b = jnp.array([4.0, 5.0, 6.0]) * u.second
    result = u.linalg.dot(a, b)
    assert result.unit == u.meter * u.second
    expected = jnp.dot(jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))
    assert jnp.allclose(result.mantissa, expected)


def test_docstring_example_multi_dot():
    """Verify the multi_dot docstring example (re-exported from math)."""
    import saiunit as u
    import jax

    k1, k2 = jax.random.split(jax.random.key(0))
    a = jax.random.normal(k1, shape=(3, 4)) * u.meter
    b = jax.random.normal(k2, shape=(4, 2)) * u.second
    result = u.linalg.multi_dot([a, b])
    assert result.shape == (3, 2)
    assert result.unit == u.meter * u.second


def test_docstring_example_vdot():
    """Verify the vdot docstring example (re-exported from math)."""
    import saiunit as u
    import jax.numpy as jnp

    a = jnp.array([1.0, 2.0, 3.0]) * u.meter
    b = jnp.array([4.0, 5.0, 6.0]) * u.second
    result = u.linalg.vdot(a, b)
    assert result.unit == u.meter * u.second


def test_docstring_example_vecdot():
    """Verify the vecdot docstring example (re-exported from math)."""
    import saiunit as u
    import jax.numpy as jnp

    a = jnp.array([1.0, 2.0, 3.0]) * u.meter
    b = jnp.array([4.0, 5.0, 6.0]) * u.second
    result = u.linalg.vecdot(a, b)
    assert result.unit == u.meter * u.second


def test_docstring_example_inner():
    """Verify the inner docstring example (re-exported from math)."""
    import saiunit as u
    import jax.numpy as jnp

    a = jnp.array([1.0, 2.0, 3.0]) * u.meter
    b = jnp.array([4.0, 5.0, 6.0]) * u.second
    result = u.linalg.inner(a, b)
    assert result.unit == u.meter * u.second


def test_docstring_example_outer():
    """Verify the outer docstring example (re-exported from math)."""
    import saiunit as u
    import jax.numpy as jnp

    a = jnp.array([1.0, 2.0]) * u.meter
    b = jnp.array([3.0, 4.0, 5.0]) * u.second
    result = u.linalg.outer(a, b)
    assert result.shape == (2, 3)
    assert result.unit == u.meter * u.second


def test_docstring_example_kron():
    """Verify the kron docstring example (re-exported from math)."""
    import saiunit as u
    import jax.numpy as jnp

    a = jnp.array([1.0, 2.0]) * u.meter
    b = jnp.array([3.0, 4.0]) * u.second
    result = u.linalg.kron(a, b)
    assert result.unit == u.meter * u.second


def test_docstring_example_matmul():
    """Verify the matmul docstring example (re-exported from math)."""
    import saiunit as u
    import jax.numpy as jnp

    a = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * u.meter
    b = jnp.array([[5.0, 6.0], [7.0, 8.0]]) * u.second
    result = u.linalg.matmul(a, b)
    assert result.shape == (2, 2)
    assert result.unit == u.meter * u.second


def test_docstring_example_tensordot():
    """Verify the tensordot docstring example (re-exported from math)."""
    import saiunit as u
    import jax.numpy as jnp

    a = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * u.meter
    b = jnp.array([[5.0, 6.0], [7.0, 8.0]]) * u.second
    result = u.linalg.tensordot(a, b, axes=1)
    assert result.unit == u.meter * u.second


def test_docstring_example_matrix_power():
    """Verify the matrix_power docstring example (re-exported from math)."""
    import saiunit as u
    import jax.numpy as jnp

    m = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * u.meter
    result = u.linalg.matrix_power(m, 2)
    assert result.unit == u.meter ** 2


def test_docstring_example_cross():
    """Verify the cross docstring example (re-exported from math)."""
    import saiunit as u
    import jax.numpy as jnp

    a = jnp.array([1.0, 0.0, 0.0]) * u.meter
    b = jnp.array([0.0, 1.0, 0.0]) * u.second
    result = u.linalg.cross(a, b)
    assert result.unit == u.meter * u.second


def test_docstring_example_det():
    """Verify the det docstring example (re-exported from math)."""
    import saiunit as u
    import jax.numpy as jnp

    a = jnp.array([[1., 2.],
                    [3., 4.]]) * u.meter
    result = u.linalg.det(a)
    assert result.unit == u.meter ** 2
    expected = jnp.linalg.det(jnp.array([[1., 2.], [3., 4.]]))
    assert jnp.allclose(result.mantissa, expected)


def test_lstsq_residuals_and_s_carry_units():
    """Regression: lstsq must wrap residuals (b.unit**2) and s (a.unit)."""
    a_v = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
    b_v = jnp.array([1., 2., 4.])
    x1, res1, rank1, s1 = bulinalg.lstsq(a_v * u.ms, b_v * u.mV)
    assert isinstance(res1, u.Quantity) and res1.unit.dim == (u.mV ** 2).dim
    assert isinstance(s1, u.Quantity) and s1.unit.dim == u.ms.dim
    assert not isinstance(rank1, u.Quantity)
    # the same physical system expressed in different display units
    x2, res2, rank2, s2 = bulinalg.lstsq(
        u.Quantity(a_v / 1000.0, unit=u.second),
        u.Quantity(b_v / 1000.0, unit=u.volt),
    )
    assert u.math.allclose(x1, x2, rtol=1e-3)
    assert u.math.allclose(res1, res2, rtol=1e-3)
    assert u.math.allclose(s1, s2, rtol=1e-3)
    assert int(rank1) == int(rank2)
    # one-sided branches
    x3, res3, rank3, s3 = bulinalg.lstsq(a_v * u.ms, b_v)
    assert isinstance(s3, u.Quantity) and s3.unit.dim == u.ms.dim
    assert not isinstance(res3, u.Quantity)
    x4, res4, rank4, s4 = bulinalg.lstsq(a_v, b_v * u.mV)
    assert isinstance(res4, u.Quantity) and res4.unit.dim == (u.mV ** 2).dim
    assert not isinstance(s4, u.Quantity)


def test_cholesky_symmetrize_input_consistent_across_backends():
    """Regression: numpy backend must symmetrize like jax instead of silently
    dropping ``symmetrize_input``."""
    import numpy as np
    a_v = np.array([[4.0, 2.0], [0.0, 3.0]])  # asymmetric input
    expected = jnp.linalg.cholesky((jnp.asarray(a_v) + jnp.asarray(a_v).T) / 2)
    r_jax = u.linalg.cholesky(jnp.asarray(a_v) * u.meter2)
    r_np = u.linalg.cholesky(u.Quantity(a_v, unit=u.meter2))
    assert r_jax.unit == meter
    assert r_np.unit == meter
    np.testing.assert_allclose(np.asarray(r_jax.mantissa), np.asarray(expected), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(r_np.mantissa), np.asarray(expected), rtol=1e-6)


def test_cholesky_symmetrize_input_false():
    """``symmetrize_input=False`` keeps jax semantics and must not crash numpy."""
    import numpy as np
    a_v = jnp.array([[4.0, 2.0], [0.0, 3.0]])
    expected = jnp.linalg.cholesky(a_v, symmetrize_input=False)
    r_jax = u.linalg.cholesky(a_v * u.meter2, symmetrize_input=False)
    np.testing.assert_allclose(np.asarray(r_jax.mantissa), np.asarray(expected),
                               rtol=1e-6, equal_nan=True)
    # numpy backend: the kwarg must not be forwarded (np.linalg.cholesky lacks it)
    sym = np.array([[4.0, 1.0], [1.0, 3.0]])
    r_np = u.linalg.cholesky(u.Quantity(sym, unit=u.meter2), symmetrize_input=False)
    np.testing.assert_allclose(np.asarray(r_np.mantissa),
                               np.linalg.cholesky(sym), rtol=1e-6)
