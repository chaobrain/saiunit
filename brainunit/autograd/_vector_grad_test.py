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
import brainunit as u


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


if __name__ == "__main__":
    pytest.main()
