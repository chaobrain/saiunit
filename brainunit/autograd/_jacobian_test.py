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

import brainunit as u


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


if __name__ == "__main__":
    pytest.main()
