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


from __future__ import annotations

import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax.numpy as jnp
import pytest
import saiunit as u


class Array(u.CustomArray):
    def __init__(self, value):
        self.data = value


def test_vector_grad_simple():
    def simple_function(x):
        return x ** 2

    for unit in [None, u.ms, u.mvolt]:
        vector_grad_fn = u.autograd.vector_grad(simple_function)
        if unit is None:
            grad = vector_grad_fn(jnp.array([3.0, 4.0]))
            assert jnp.allclose(grad, jnp.array([6.0, 8.0]))
        else:
            grad = vector_grad_fn(jnp.array([3.0, 4.0]) * unit)
            assert u.math.allclose(grad, jnp.array([6.0, 8.0]) * unit)


def test_vector_grad_simple2():
    def simple_function(x):
        return x ** 3

    x = jnp.array([3.0, 4.0])
    for unit in [None, u.ms, u.mvolt]:
        vector_grad_fn = u.autograd.vector_grad(simple_function)
        if unit is None:
            grad = vector_grad_fn(x)
            assert jnp.allclose(grad, 3 * x ** 2)
        else:
            grad = vector_grad_fn(x * unit)
            assert u.math.allclose(grad, 3 * (x * unit) ** 2)


def test_vector_grad_multiple_args():
    def multi_arg_function(x, y):
        return x * y

    for ux, uy in ([u.ms, u.mV],
                   [u.ms, u.UNITLESS],
                   [u.UNITLESS, u.mV],
                   [u.UNITLESS, u.UNITLESS]):
        vector_grad_fn = u.autograd.vector_grad(multi_arg_function, argnums=(0, 1))
        grad = vector_grad_fn(jnp.array([3.0, 4.0]) * ux,
                              jnp.array([5.0, 6.0]) * uy)
        assert u.math.allclose(grad[0], jnp.array([5.0, 6.0]) * uy)
        assert u.math.allclose(grad[1], jnp.array([3.0, 4.0]) * ux)


def test_vector_grad_with_aux():
    def function_with_aux(x):
        return x ** 2, u.math.sum(x * 3)

    for unit in [u.UNITLESS, u.mV, u.ms, u.siemens]:
        vector_grad_fn = u.autograd.vector_grad(function_with_aux, has_aux=True, return_value=True)
        x = jnp.array([3.0, 4.0]) * unit
        grad, value, aux = vector_grad_fn(x)
        assert u.math.allclose(value, x ** 2)
        assert u.math.allclose(aux, jnp.array(21.0) * unit)
        assert u.math.allclose(grad, jnp.array([6.0, 8.0]) * unit)


def test_vector_grad_with_array_custom_array():
    def simple_function(x):
        return x ** 2

    vector_grad_fn = u.autograd.vector_grad(simple_function)
    
    # Test with Array containing unitless values
    x_array = Array(jnp.array([3.0, 4.0]))
    assert isinstance(x_array, u.CustomArray)
    grad = vector_grad_fn(x_array.data)
    grad_array = Array(grad)
    assert isinstance(grad_array, u.CustomArray)
    assert jnp.allclose(grad_array.data, jnp.array([6.0, 8.0]))
    
    # Test with Array containing unit values
    x_unit = jnp.array([3.0, 4.0]) * u.mvolt
    x_array_unit = Array(x_unit)
    assert isinstance(x_array_unit, u.CustomArray)
    grad = vector_grad_fn(x_array_unit.data)
    grad_array = Array(grad)
    assert isinstance(grad_array, u.CustomArray)
    assert u.math.allclose(grad_array.data, jnp.array([6.0, 8.0]) * u.mvolt)


def test_vector_grad_cubic_with_array():
    def cubic_function(x):
        return x ** 3

    x = jnp.array([3.0, 4.0])
    vector_grad_fn = u.autograd.vector_grad(cubic_function)
    
    # Test with Array containing unitless values
    x_array = Array(x)
    assert isinstance(x_array, u.CustomArray)
    grad = vector_grad_fn(x_array.data)
    grad_array = Array(grad)
    assert isinstance(grad_array, u.CustomArray)
    assert jnp.allclose(grad_array.data, 3 * x ** 2)
    
    # Test with Array containing unit values
    x_unit = x * u.ms
    x_array_unit = Array(x_unit)
    assert isinstance(x_array_unit, u.CustomArray)
    grad = vector_grad_fn(x_array_unit.data)
    grad_array = Array(grad)
    assert isinstance(grad_array, u.CustomArray)
    assert u.math.allclose(grad_array.data, 3 * (x * u.ms) ** 2)


def test_vector_grad_multiple_args_with_array():
    def multi_arg_function(x, y):
        return x * y

    vector_grad_fn = u.autograd.vector_grad(multi_arg_function, argnums=(0, 1))
    
    # Test with Array inputs
    x = jnp.array([3.0, 4.0]) * u.ms
    y = jnp.array([5.0, 6.0]) * u.mV
    x_array = Array(x)
    y_array = Array(y)
    
    assert isinstance(x_array, u.CustomArray)
    assert isinstance(y_array, u.CustomArray)
    
    grad = vector_grad_fn(x_array.data, y_array.data)
    grad0_array = Array(grad[0])
    grad1_array = Array(grad[1])
    
    assert isinstance(grad0_array, u.CustomArray)
    assert isinstance(grad1_array, u.CustomArray)
    assert u.math.allclose(grad0_array.data, jnp.array([5.0, 6.0]) * u.mV)
    assert u.math.allclose(grad1_array.data, jnp.array([3.0, 4.0]) * u.ms)


def test_vector_grad_with_aux_array():
    def function_with_aux(x):
        return x ** 2, u.math.sum(x * 3)

    vector_grad_fn = u.autograd.vector_grad(function_with_aux, has_aux=True, return_value=True)
    
    # Test with Array
    x = jnp.array([3.0, 4.0]) * u.mV
    x_array = Array(x)
    assert isinstance(x_array, u.CustomArray)
    
    grad, value, aux = vector_grad_fn(x_array.data)
    grad_array = Array(grad)
    value_array = Array(value)
    aux_array = Array(aux)
    
    assert isinstance(grad_array, u.CustomArray)
    assert isinstance(value_array, u.CustomArray)
    assert isinstance(aux_array, u.CustomArray)
    assert u.math.allclose(value_array.data, x ** 2)
    assert u.math.allclose(aux_array.data, jnp.array(21.0) * u.mV)
    assert u.math.allclose(grad_array.data, jnp.array([6.0, 8.0]) * u.mV)


def test_vector_grad_matrix_operations_with_array():
    def matrix_function(x):
        # Quadratic form: x^T * A * x where A is symmetric
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]]) * u.mA
        A_array = Array(A)
        assert isinstance(A_array, u.CustomArray)
        return x.T @ A_array.data @ x

    vector_grad_fn = u.autograd.vector_grad(matrix_function)
    
    # Test with vector Array
    x = jnp.array([1.0, 1.0]) * u.mV
    x_array = Array(x)
    assert isinstance(x_array, u.CustomArray)
    
    grad = vector_grad_fn(x_array.data)
    grad_array = Array(grad)
    assert isinstance(grad_array, u.CustomArray)
    
    # Gradient of x^T * A * x is 2 * A * x (since A is symmetric)
    A = jnp.array([[2.0, 1.0], [1.0, 3.0]]) * u.mA
    expected = 2 * A @ x
    assert u.math.allclose(grad_array.data, expected)


def test_array_custom_array_compatibility_with_vector_grad():
    data = jnp.array([1.5, 2.5, 3.5]) * u.second
    test_array = Array(data)
    
    assert isinstance(test_array, u.CustomArray)
    assert hasattr(test_array, 'data')
    
    def test_function(x):
        return u.math.sum(x ** 4)
    
    # Test vector_grad with Array
    vector_grad_fn = u.autograd.vector_grad(test_function)
    result = vector_grad_fn(test_array.data)
    result_array = Array(result)
    
    assert isinstance(result_array, u.CustomArray)
    
    # Compare with direct computation
    direct_result = vector_grad_fn(data)
    assert u.math.allclose(result_array.data, direct_result)


if __name__ == "__main__":
    pytest.main()
