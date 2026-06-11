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

"""Regression tests for edge cases in ``saiunit.autograd``.

These tests intentionally avoid the optional ``brainstate`` dependency so
they run in every CI configuration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import saiunit as u
import saiunit.autograd as suauto


# ---------------------------------------------------------------------------
# vector_grad: unit propagation bugs
# ---------------------------------------------------------------------------

def test_vector_grad_single_quantity_in_container_keeps_unit():
    # A single Quantity leaf wrapped in a dict must take its unit from the
    # leaf, not from the (unitless) container.
    def f(x):
        return {'o': x ** 2}

    grad = suauto.vector_grad(f)(jnp.array([3.0, 4.0]) * u.ms)
    assert u.get_unit(grad) == u.ms
    assert u.math.allclose(grad, jnp.array([6.0, 8.0]) * u.ms)


def test_vector_grad_single_quantity_in_nested_container_keeps_unit():
    def f(x):
        return {'a': {'b': x ** 2}}

    grad = suauto.vector_grad(f)(jnp.array([3.0, 4.0]) * u.ms)
    assert u.get_unit(grad) == u.ms


def test_vector_grad_unit_aware_false_returns_bare_mantissa():
    # With unit_aware=False the gradient must not carry a (wrong) unit.
    # d/dx x**3 = 3 x**2; the true unit would be ms**2, and stamping the
    # input unit ms would be physically wrong, so we return a plain array.
    def f(x):
        return x ** 3

    grad = suauto.vector_grad(f, unit_aware=False)(jnp.array([2.0]) * u.ms)
    assert not isinstance(grad, u.Quantity)
    assert jnp.allclose(grad, jnp.array([12.0]))


def test_vector_grad_rejects_integer_output():
    def f(x):
        return (x > 1.0).astype(jnp.int32)

    with pytest.raises(TypeError, match="real-valued outputs|inexact"):
        suauto.vector_grad(f)(jnp.array([3.0]))


def test_vector_grad_is_column_sum_not_diagonal():
    # vector_grad is a ones-cotangent VJP == column sums of the Jacobian,
    # which equals the diagonal only for element-wise functions.
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    def f(x):
        return A @ x

    grad = suauto.vector_grad(f)(jnp.array([1.0, 1.0]))
    assert jnp.allclose(grad, A.sum(axis=0))  # column sums: [4, 6]


# ---------------------------------------------------------------------------
# jacrev / jacfwd: dtype handling parity with jax
# ---------------------------------------------------------------------------

def test_jacrev_accepts_complex_input():
    # Reverse-mode C->R differentiation is allowed by jax.jacrev.
    def f(x):
        return jnp.abs(x) ** 2

    x = jnp.array(1.0 + 2.0j)
    got = suauto.jacrev(f)(x)
    ref = jax.jacrev(f)(x)
    assert np.allclose(np.asarray(u.get_mantissa(got)), np.asarray(ref), atol=1e-4)


def test_jacfwd_rejects_complex_input_without_holomorphic():
    # Forward-mode over a non-holomorphic complex input is meaningless;
    # jax.jacfwd raises and points at holomorphic=True.
    def f(x):
        return jnp.abs(x) ** 2

    with pytest.raises(TypeError, match="holomorphic"):
        suauto.jacfwd(f)(jnp.array(1.0 + 2.0j))


def test_jacfwd_allows_real_to_complex_output():
    # Non-holomorphic jacfwd imposes no constraint on the output dtype.
    def f(x):
        return jnp.exp(1j * x)

    x = jnp.array(0.5)
    got = suauto.jacfwd(f)(x)
    ref = jax.jacfwd(f)(x)
    assert np.allclose(np.asarray(u.get_mantissa(got)), np.asarray(ref), atol=1e-4)


def test_jacrev_accepts_bfloat16_input():
    def f(x):
        return x ** 2

    x = jnp.array(3.0, dtype=jnp.bfloat16)
    got = suauto.jacrev(f)(x)
    assert jnp.allclose(u.get_mantissa(got).astype(jnp.float32),
                        jnp.array([6.0], dtype=jnp.float32))


def test_jacfwd_accepts_bfloat16_input():
    def f(x):
        return x ** 2

    x = jnp.array(3.0, dtype=jnp.bfloat16)
    got = suauto.jacfwd(f)(x)
    assert jnp.allclose(u.get_mantissa(got).astype(jnp.float32),
                        jnp.array([6.0], dtype=jnp.float32))


def test_jacrev_complex_output_error_mentions_holomorphic():
    def f(x):
        return jnp.exp(1j * x)

    with pytest.raises(TypeError, match="holomorphic"):
        suauto.jacrev(f)(jnp.array(0.5))


def test_jacrev_integer_input_error_mentions_allow_int():
    def f(x):
        return x * 1.0

    with pytest.raises(TypeError, match="allow_int"):
        suauto.jacrev(f)(jnp.array(3))


# ---------------------------------------------------------------------------
# float0: integer-arg gradients must not become void-dtype Quantities
# ---------------------------------------------------------------------------

def test_value_and_grad_float0_not_wrapped_in_quantity():
    # Unit-ful loss + allow_int over an integer arg used to produce a
    # void-dtype Quantity that crashes on any arithmetic.
    def loss(i, x):
        return u.math.sum(x ** 2) * u.mV

    _, grads = suauto.value_and_grad(loss, argnums=(0, 1), allow_int=True)(
        jnp.array(3), jnp.array([2.0])
    )
    grad_i = grads[0]
    assert not isinstance(grad_i, u.Quantity)
    assert grad_i.dtype == jax.dtypes.float0


def test_jacrev_float0_not_wrapped_in_quantity():
    def f(i, x):
        return x ** 2

    jac = suauto.jacrev(f, argnums=(0, 1), allow_int=True)(
        jnp.array(2), jnp.array(3.0) * u.ms
    )
    assert not isinstance(jac[0], u.Quantity)
    assert jac[0].dtype == jax.dtypes.float0


# ---------------------------------------------------------------------------
# Eager validation of the differentiated callable
# ---------------------------------------------------------------------------

def test_grad_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        suauto.grad(42)


def test_value_and_grad_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        suauto.value_and_grad(42)


def test_grad_rejects_generator_function():
    def gen(x):
        yield x

    with pytest.raises(TypeError, match="generator"):
        suauto.grad(gen)


def test_value_and_grad_has_aux_non_tuple_return():
    # has_aux=True but the function returns a bare scalar.
    def f(x):
        return x ** 2

    with pytest.raises(TypeError, match="pair|two-element|aux"):
        suauto.value_and_grad(f, has_aux=True)(jnp.array(1.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
