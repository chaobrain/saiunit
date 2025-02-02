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


if __name__ == "__main__":
    pytest.main()
