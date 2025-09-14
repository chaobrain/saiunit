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

import brainstate as bst
import jax.numpy as jnp
import pytest

import saiunit as u


class Array(u.CustomArray):
    def __init__(self, value):
        self.data = value


def test_jacrev_simple_function():
    def simple_function(x):
        return x ** 2

    jac_fn = u.autograd.jacrev(simple_function)

    x = jnp.array(3.0)
    jac = jac_fn(x)
    assert jnp.allclose(jac, jnp.array([6.0]))

    x = jnp.array(3.0) * u.ms
    jac = jac_fn(x)
    assert u.math.allclose(jac, jnp.array([6.0]) * u.ms)


def test_jacrev_function2():
    def simple_function(x, y):
        return x * y

    jac_fn = u.autograd.jacrev(simple_function, argnums=(0, 1))

    x = bst.random.rand(3) * u.ohm
    y = bst.random.rand(3) * u.mA
    jac = jac_fn(x, y)
    assert u.math.allclose(
        jac[0],
        u.math.diag(y)
    )
    assert u.math.allclose(
        jac[1],
        u.math.diag(x)
    )


def test_jacrev_function3():
    def simple_function(inputs):
        o1 = inputs['x'] * inputs['y']
        o2 = inputs['x'] * inputs['z']
        r = {'o1': o1, 'o2': o2}
        return r, r

    jac_fn = u.autograd.jacrev(simple_function, has_aux=True)

    x = bst.random.rand(3) * u.ohm
    y = bst.random.rand(3) * u.mA
    z = bst.random.rand(3) * u.siemens

    inp = {'x': x, 'y': y, 'z': z}
    jac, r = jac_fn(inp)

    assert u.math.allclose(
        jac['o1']['x'],
        u.math.diag(y)
    )
    assert u.math.allclose(
        jac['o1']['y'],
        u.math.diag(x)
    )
    assert u.math.allclose(
        jac['o1']['z'],
        u.math.diag(u.math.zeros(3) * u.get_unit(r['o1']) / u.get_unit(inp['z']))
    )

    assert u.math.allclose(
        jac['o2']['x'],
        u.math.diag(z)
    )
    assert u.math.allclose(
        jac['o2']['y'],
        u.math.diag(u.math.zeros(3) * u.get_unit(r['o2']) / u.get_unit(inp['y']))
    )
    assert u.math.allclose(
        jac['o2']['z'],
        u.math.diag(x)
    )


def test_jacrev_with_aux():
    def simple_function(x):
        return x ** 2, x

    jac_fn = u.autograd.jacrev(simple_function, has_aux=True)
    x = jnp.array(3.0)
    jac, aux = jac_fn(x)
    assert jnp.allclose(jac, jnp.array([6.0]))
    assert jnp.allclose(aux, jnp.array(3.0))

    x = jnp.array(3.0) * u.ms
    jac, aux = jac_fn(x)
    assert u.math.allclose(jac, jnp.array([6.0]) * u.ms)
    assert u.math.allclose(aux, jnp.array(3.0) * u.ms)


def test_jacfwd_simple_function():
    def simple_function(x):
        return x ** 2

    jac_fn = u.autograd.jacfwd(simple_function)
    x = jnp.array(3.0)
    jac = jac_fn(x)
    assert jnp.allclose(jac, jnp.array([6.0]))

    x = jnp.array(3.0) * u.ms
    jac = jac_fn(x)
    assert u.math.allclose(jac, jnp.array([6.0]) * u.ms)


def test_jacfwd_function2():
    def simple_function(x, y):
        return x * y

    jac_fn = u.autograd.jacfwd(simple_function, argnums=(0, 1))

    x = bst.random.rand(3) * u.ohm
    y = bst.random.rand(3) * u.mA
    jac = jac_fn(x, y)
    assert u.math.allclose(
        jac[0],
        u.math.diag(y)
    )
    assert u.math.allclose(
        jac[1],
        u.math.diag(x)
    )


def test_jacfwd_with_aux():
    def simple_function(x):
        return x ** 2, x

    jac_fn = u.autograd.jacfwd(simple_function, has_aux=True)

    x = jnp.array(3.0)
    jac, aux = jac_fn(x)
    assert jnp.allclose(jac, jnp.array([6.0]))
    assert jnp.allclose(aux, jnp.array(3.0))

    x = jnp.array(3.0) * u.ms
    jac, aux = jac_fn(x)
    assert u.math.allclose(jac, jnp.array([6.0]) * u.ms)
    assert u.math.allclose(aux, jnp.array(3.0) * u.ms)


def test_jacrev_with_array_custom_array():
    def simple_function(x):
        return x ** 2

    jac_fn = u.autograd.jacrev(simple_function)

    # Test with Array containing unitless values
    x_array = Array(jnp.array(3.0))
    assert isinstance(x_array, u.CustomArray)
    jac = jac_fn(x_array.data)
    jac_array = Array(jac)
    assert isinstance(jac_array, u.CustomArray)
    assert jnp.allclose(jac_array.data, jnp.array([6.0]))

    # Test with Array containing unit values
    x_unit = jnp.array(3.0) * u.ms
    x_array_unit = Array(x_unit)
    assert isinstance(x_array_unit, u.CustomArray)
    jac = jac_fn(x_array_unit.data)
    jac_array = Array(jac)
    assert isinstance(jac_array, u.CustomArray)
    assert u.math.allclose(jac_array.data, jnp.array([6.0]) * u.ms)


def test_jacfwd_with_array_custom_array():
    def simple_function(x):
        return x ** 2

    jac_fn = u.autograd.jacfwd(simple_function)

    # Test with Array containing unitless values
    x_array = Array(jnp.array(3.0))
    assert isinstance(x_array, u.CustomArray)
    jac = jac_fn(x_array.data)
    jac_array = Array(jac)
    assert isinstance(jac_array, u.CustomArray)
    assert jnp.allclose(jac_array.data, jnp.array([6.0]))

    # Test with Array containing unit values
    x_unit = jnp.array(3.0) * u.ms
    x_array_unit = Array(x_unit)
    assert isinstance(x_array_unit, u.CustomArray)
    jac = jac_fn(x_array_unit.data)
    jac_array = Array(jac)
    assert isinstance(jac_array, u.CustomArray)
    assert u.math.allclose(jac_array.data, jnp.array([6.0]) * u.ms)


def test_jacobian_multiple_args_with_array():
    def multi_arg_function(x, y):
        return x * y

    jac_fn = u.autograd.jacrev(multi_arg_function, argnums=(0, 1))

    # Test with Array inputs
    x = bst.random.rand(3) * u.ohm
    y = bst.random.rand(3) * u.mA
    x_array = Array(x)
    y_array = Array(y)
    
    assert isinstance(x_array, u.CustomArray)
    assert isinstance(y_array, u.CustomArray)
    
    jac = jac_fn(x_array.data, y_array.data)
    jac0_array = Array(jac[0])
    jac1_array = Array(jac[1])
    
    assert isinstance(jac0_array, u.CustomArray)
    assert isinstance(jac1_array, u.CustomArray)
    
    assert u.math.allclose(jac0_array.data, u.math.diag(y))
    assert u.math.allclose(jac1_array.data, u.math.diag(x))


def test_jacobian_with_aux_array():
    def function_with_aux(x):
        return x ** 2, x

    jac_fn = u.autograd.jacrev(function_with_aux, has_aux=True)
    
    # Test with Array
    x_unit = jnp.array(3.0) * u.ms
    x_array = Array(x_unit)
    assert isinstance(x_array, u.CustomArray)
    
    jac, aux = jac_fn(x_array.data)
    jac_array = Array(jac)
    aux_array = Array(aux)
    
    assert isinstance(jac_array, u.CustomArray)
    assert isinstance(aux_array, u.CustomArray)
    assert u.math.allclose(jac_array.data, jnp.array([6.0]) * u.ms)
    assert u.math.allclose(aux_array.data, jnp.array(3.0) * u.ms)


def test_jacobian_vector_inputs_with_array():
    def vector_function(x):
        return u.math.sum(x ** 2)

    jac_fn = u.autograd.jacrev(vector_function)
    
    # Test with vector Array
    x = jnp.array([1.0, 2.0, 3.0]) * u.mA
    x_array = Array(x)
    assert isinstance(x_array, u.CustomArray)
    
    jac = jac_fn(x_array.data)
    jac_array = Array(jac)
    assert isinstance(jac_array, u.CustomArray)
    
    expected = 2.0 * jnp.array([1.0, 2.0, 3.0]) * u.mA
    assert u.math.allclose(jac_array.data, expected)


def test_array_custom_array_compatibility_with_jacobian():
    data = jnp.array([1.0, 2.0, 3.0]) * u.second
    test_array = Array(data)
    
    assert isinstance(test_array, u.CustomArray)
    assert hasattr(test_array, 'data')
    
    def test_function(x):
        return u.math.sum(x ** 3)
    
    # Test jacrev with Array
    jac_fn = u.autograd.jacrev(test_function)
    result = jac_fn(test_array.data)
    result_array = Array(result)
    
    assert isinstance(result_array, u.CustomArray)
    
    # Compare with direct computation
    direct_result = jac_fn(data)
    assert u.math.allclose(result_array.data, direct_result)


if __name__ == "__main__":
    pytest.main()
