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
        self.value = value


def test_value_and_grad_simple():
    def simple_function(x):
        return x ** 2

    for unit in [None, u.ms, u.mvolt]:
        value_and_grad_fn = u.autograd.value_and_grad(simple_function)
        if unit is None:
            value, grad = value_and_grad_fn(jnp.array(3.0))
            assert value == 9.0
            assert grad == 6.0
        else:
            value, grad = value_and_grad_fn(jnp.array(3.0) * unit)
            assert u.math.allclose(value, 9.0 * unit ** 2)
            assert u.math.allclose(grad, 6.0 * unit)


def test_value_and_grad_multiple_args():
    def multi_arg_function(x, y):
        return x * y

    for ux, uy in ([u.ms, u.mV],
                   [u.ms, u.UNITLESS],
                   [u.UNITLESS, u.mV],
                   [u.UNITLESS, u.UNITLESS]):
        value_and_grad_fn = u.autograd.value_and_grad(multi_arg_function, argnums=(0, 1))
        value, grad = value_and_grad_fn(jnp.array(3.0) * ux, jnp.array(4.0) * uy)
        assert u.math.allclose(value, 12.0 * ux * uy)
        assert u.math.allclose(grad[0], 4.0 * uy)
        assert u.math.allclose(grad[1], 3.0 * ux)


def test_value_and_grad_with_aux():
    def function_with_aux(x):
        return x ** 2, x * 3

    for unit in [u.UNITLESS, u.mV, u.ms, u.siemens]:
        value_and_grad_fn = u.autograd.value_and_grad(function_with_aux, has_aux=True)
        (value, aux), grad = value_and_grad_fn(jnp.array(3.0) * unit)
        assert u.math.allclose(value, 9.0 * unit ** 2)
        assert u.math.allclose(aux, 9.0 * unit)
        assert u.math.allclose(grad, 6.0 * unit)


def test_grad_simple():
    def simple_function(x):
        return x ** 2

    for unit in [None, u.ms, u.mvolt]:
        grad_fn = u.autograd.grad(simple_function)
        if unit is None:
            grad = grad_fn(jnp.array(3.0))
            assert grad == 6.0
        else:
            grad = grad_fn(jnp.array(3.0) * unit)
            assert u.math.allclose(grad, 6.0 * unit)


def test_grad_multiple_args():
    def multi_arg_function(x, y):
        return x * y

    for ux, uy in ([u.ms, u.mV],
                   [u.ms, u.UNITLESS],
                   [u.UNITLESS, u.mV],
                   [u.UNITLESS, u.UNITLESS]):
        grad_fn = u.autograd.grad(multi_arg_function, argnums=(0, 1))
        grad = grad_fn(jnp.array(3.0) * ux, jnp.array(4.0) * uy)
        assert u.math.allclose(grad[0], 4.0 * uy)
        assert u.math.allclose(grad[1], 3.0 * ux)


def test_grad_with_aux():
    def function_with_aux(x):
        return x ** 2, x * 3

    for unit in [u.UNITLESS, u.mV, u.ms, u.siemens]:
        grad_fn = u.autograd.grad(function_with_aux, has_aux=True)
        grad, aux = grad_fn(jnp.array(3.0) * unit)
        assert u.math.allclose(aux, 9.0 * unit)
        assert u.math.allclose(grad, 6.0 * unit)


def test_value_and_grad_with_array_custom_array():
    def simple_function(x):
        return x ** 2

    value_and_grad_fn = u.autograd.value_and_grad(simple_function)
    
    # Test with Array containing unitless values
    x_array = Array(jnp.array(3.0))
    assert isinstance(x_array, u.CustomArray)
    value, grad = value_and_grad_fn(x_array.value)
    
    value_array = Array(value)
    grad_array = Array(grad)
    assert isinstance(value_array, u.CustomArray)
    assert isinstance(grad_array, u.CustomArray)
    assert value_array.value == 9.0
    assert grad_array.value == 6.0
    
    # Test with Array containing unit values
    x_unit = jnp.array(3.0) * u.ms
    x_array_unit = Array(x_unit)
    assert isinstance(x_array_unit, u.CustomArray)
    value, grad = value_and_grad_fn(x_array_unit.value)
    
    value_array = Array(value)
    grad_array = Array(grad)
    assert isinstance(value_array, u.CustomArray)
    assert isinstance(grad_array, u.CustomArray)
    assert u.math.allclose(value_array.value, 9.0 * u.ms ** 2)
    assert u.math.allclose(grad_array.value, 6.0 * u.ms)


def test_grad_with_array_custom_array():
    def simple_function(x):
        return x ** 2

    grad_fn = u.autograd.grad(simple_function)
    
    # Test with Array containing unitless values
    x_array = Array(jnp.array(3.0))
    assert isinstance(x_array, u.CustomArray)
    grad = grad_fn(x_array.value)
    grad_array = Array(grad)
    assert isinstance(grad_array, u.CustomArray)
    assert grad_array.value == 6.0
    
    # Test with Array containing unit values
    x_unit = jnp.array(3.0) * u.mvolt
    x_array_unit = Array(x_unit)
    assert isinstance(x_array_unit, u.CustomArray)
    grad = grad_fn(x_array_unit.value)
    grad_array = Array(grad)
    assert isinstance(grad_array, u.CustomArray)
    assert u.math.allclose(grad_array.value, 6.0 * u.mvolt)


def test_value_and_grad_multiple_args_with_array():
    def multi_arg_function(x, y):
        return x * y

    value_and_grad_fn = u.autograd.value_and_grad(multi_arg_function, argnums=(0, 1))
    
    # Test with Array inputs
    x = jnp.array(3.0) * u.ms
    y = jnp.array(4.0) * u.mV
    x_array = Array(x)
    y_array = Array(y)
    
    assert isinstance(x_array, u.CustomArray)
    assert isinstance(y_array, u.CustomArray)
    
    value, grad = value_and_grad_fn(x_array.value, y_array.value)
    value_array = Array(value)
    grad0_array = Array(grad[0])
    grad1_array = Array(grad[1])
    
    assert isinstance(value_array, u.CustomArray)
    assert isinstance(grad0_array, u.CustomArray)
    assert isinstance(grad1_array, u.CustomArray)
    assert u.math.allclose(value_array.value, 12.0 * u.ms * u.mV)
    assert u.math.allclose(grad0_array.value, 4.0 * u.mV)
    assert u.math.allclose(grad1_array.value, 3.0 * u.ms)


def test_value_and_grad_with_aux_array():
    def function_with_aux(x):
        return x ** 2, x * 3

    value_and_grad_fn = u.autograd.value_and_grad(function_with_aux, has_aux=True)
    
    # Test with Array
    x_unit = jnp.array(3.0) * u.mV
    x_array = Array(x_unit)
    assert isinstance(x_array, u.CustomArray)
    
    (value, aux), grad = value_and_grad_fn(x_array.value)
    value_array = Array(value)
    aux_array = Array(aux)
    grad_array = Array(grad)
    
    assert isinstance(value_array, u.CustomArray)
    assert isinstance(aux_array, u.CustomArray)
    assert isinstance(grad_array, u.CustomArray)
    assert u.math.allclose(value_array.value, 9.0 * u.mV ** 2)
    assert u.math.allclose(aux_array.value, 9.0 * u.mV)
    assert u.math.allclose(grad_array.value, 6.0 * u.mV)


def test_grad_multiple_args_with_array():
    def multi_arg_function(x, y):
        return x * y

    grad_fn = u.autograd.grad(multi_arg_function, argnums=(0, 1))
    
    # Test with Array inputs
    x = jnp.array(3.0) * u.UNITLESS
    y = jnp.array(4.0) * u.mV
    x_array = Array(x)
    y_array = Array(y)
    
    assert isinstance(x_array, u.CustomArray)
    assert isinstance(y_array, u.CustomArray)
    
    grad = grad_fn(x_array.value, y_array.value)
    grad0_array = Array(grad[0])
    grad1_array = Array(grad[1])
    
    assert isinstance(grad0_array, u.CustomArray)
    assert isinstance(grad1_array, u.CustomArray)
    assert u.math.allclose(grad0_array.value, 4.0 * u.mV)
    assert u.math.allclose(grad1_array.value, 3.0 * u.UNITLESS)


def test_array_custom_array_compatibility_with_value_and_grad():
    data = jnp.array(2.5) * u.second
    test_array = Array(data)
    
    assert isinstance(test_array, u.CustomArray)
    assert hasattr(test_array, 'value')
    
    def test_function(x):
        return x ** 3
    
    # Test value_and_grad with Array
    value_and_grad_fn = u.autograd.value_and_grad(test_function)
    value, grad = value_and_grad_fn(test_array.value)
    value_array = Array(value)
    grad_array = Array(grad)
    
    assert isinstance(value_array, u.CustomArray)
    assert isinstance(grad_array, u.CustomArray)
    
    # Compare with direct computation
    direct_value, direct_grad = value_and_grad_fn(data)
    assert u.math.allclose(value_array.value, direct_value)
    assert u.math.allclose(grad_array.value, direct_grad)


if __name__ == "__main__":
    pytest.main()
