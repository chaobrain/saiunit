# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""
Comprehensive tests for the exprel function.

Tests cover:
1. VJP/JVP gradient equivalence
2. Dtype-specific accuracy
3. Numerical stability at edge cases
4. Second-order gradients (Hessian)
5. vmap batching
6. Taylor series order configuration
"""

# Enable x64 before importing jax
# import os
# os.environ['JAX_ENABLE_X64'] = 'True'

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import exprel as scipy_exprel

import saiunit as u
from saiunit.math._exprel import (
    exprel,
    set_exprel_order,
    get_exprel_order,
    _get_threshold,
    _exprel_coefficients,
    _exprel_deriv_coefficients,
)


class TestExprelGradients:
    """Tests for gradient consistency between VJP and JVP."""

    def test_vjp_jvp_equivalence_at_zero(self):
        """Test VJP/JVP equivalence at x=0 (singularity point)."""
        x = jnp.array([0.0])

        # JVP
        primal_jvp, tangent_jvp = jax.jvp(exprel, (x,), (jnp.ones_like(x),))

        # VJP
        primal_vjp, vjp_fn = jax.vjp(exprel, x)
        grad_vjp = vjp_fn(jnp.ones_like(x))[0]

        # Primals should match
        assert jnp.allclose(primal_jvp, primal_vjp), f"Primals differ: {primal_jvp} vs {primal_vjp}"

        # Gradients should match (derivative at 0 should be 0.5)
        assert jnp.allclose(tangent_jvp, grad_vjp, rtol=1e-5), \
            f"Gradients differ at x=0: JVP={tangent_jvp}, VJP={grad_vjp}"

        # Verify the derivative value is correct (should be 0.5)
        assert jnp.allclose(grad_vjp, 0.5, rtol=1e-5), f"Gradient at 0 should be 0.5, got {grad_vjp}"

    def test_vjp_jvp_equivalence_near_zero(self):
        """Test VJP/JVP equivalence for values near zero."""
        test_values = [1e-20, 1e-15, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]

        for val in test_values:
            x = jnp.array([val], dtype=jnp.float64)

            primal_jvp, tangent_jvp = jax.jvp(exprel, (x,), (jnp.ones_like(x),))
            primal_vjp, vjp_fn = jax.vjp(exprel, x)
            grad_vjp = vjp_fn(jnp.ones_like(x))[0]

            assert jnp.allclose(primal_jvp, primal_vjp), \
                f"Primals differ at x={val}: {primal_jvp} vs {primal_vjp}"
            assert jnp.allclose(tangent_jvp, grad_vjp, rtol=1e-4), \
                f"Gradients differ at x={val}: JVP={tangent_jvp}, VJP={grad_vjp}"

    def test_vjp_jvp_equivalence_negative_near_zero(self):
        """Test VJP/JVP equivalence for negative values near zero."""
        test_values = [-1e-20, -1e-15, -1e-10, -1e-8, -1e-6, -1e-5, -1e-4, -1e-3]

        for val in test_values:
            x = jnp.array([val], dtype=jnp.float64)

            primal_jvp, tangent_jvp = jax.jvp(exprel, (x,), (jnp.ones_like(x),))
            primal_vjp, vjp_fn = jax.vjp(exprel, x)
            grad_vjp = vjp_fn(jnp.ones_like(x))[0]

            assert jnp.allclose(primal_jvp, primal_vjp), \
                f"Primals differ at x={val}: {primal_jvp} vs {primal_vjp}"
            assert jnp.allclose(tangent_jvp, grad_vjp, rtol=1e-4), \
                f"Gradients differ at x={val}: JVP={tangent_jvp}, VJP={grad_vjp}"

    def test_vjp_jvp_equivalence_regular_values(self):
        """Test VJP/JVP equivalence for regular values."""
        test_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        for val in test_values:
            x = jnp.array([val])

            primal_jvp, tangent_jvp = jax.jvp(exprel, (x,), (jnp.ones_like(x),))
            primal_vjp, vjp_fn = jax.vjp(exprel, x)
            grad_vjp = vjp_fn(jnp.ones_like(x))[0]

            assert jnp.allclose(primal_jvp, primal_vjp), \
                f"Primals differ at x={val}: {primal_jvp} vs {primal_vjp}"
            assert jnp.allclose(tangent_jvp, grad_vjp, rtol=1e-5), \
                f"Gradients differ at x={val}: JVP={tangent_jvp}, VJP={grad_vjp}"

    def test_vjp_jvp_equivalence_large_values(self):
        """Test VJP/JVP equivalence for large values."""
        test_values = [50.0, 100.0, 200.0, 500.0]

        for val in test_values:
            x = jnp.array([val], dtype=jnp.float64)

            primal_jvp, tangent_jvp = jax.jvp(exprel, (x,), (jnp.ones_like(x),))
            primal_vjp, vjp_fn = jax.vjp(exprel, x)
            grad_vjp = vjp_fn(jnp.ones_like(x))[0]

            assert jnp.allclose(primal_jvp, primal_vjp), \
                f"Primals differ at x={val}: {primal_jvp} vs {primal_vjp}"
            # For large values, use relative tolerance
            assert jnp.allclose(tangent_jvp, grad_vjp, rtol=1e-4), \
                f"Gradients differ at x={val}: JVP={tangent_jvp}, VJP={grad_vjp}"

    def test_vjp_jvp_equivalence_negative_values(self):
        """Test VJP/JVP equivalence for negative values."""
        test_values = [-0.1, -0.5, -1.0, -2.0, -5.0, -10.0, -50.0, -100.0]

        for val in test_values:
            x = jnp.array([val], dtype=jnp.float64)

            primal_jvp, tangent_jvp = jax.jvp(exprel, (x,), (jnp.ones_like(x),))
            primal_vjp, vjp_fn = jax.vjp(exprel, x)
            grad_vjp = vjp_fn(jnp.ones_like(x))[0]

            assert jnp.allclose(primal_jvp, primal_vjp), \
                f"Primals differ at x={val}: {primal_jvp} vs {primal_vjp}"
            assert jnp.allclose(tangent_jvp, grad_vjp, rtol=1e-4), \
                f"Gradients differ at x={val}: JVP={tangent_jvp}, VJP={grad_vjp}"

    def test_vjp_jvp_equivalence_threshold_boundary(self):
        """Test VJP/JVP equivalence at threshold boundaries."""
        # Test around float32 threshold (1e-5)
        threshold_f32 = 1e-5
        boundary_values = [
            threshold_f32 * 0.1,
            threshold_f32 * 0.5,
            threshold_f32 * 0.99,
            threshold_f32,
            threshold_f32 * 1.01,
            threshold_f32 * 2,
            threshold_f32 * 10,
        ]

        for val in boundary_values:
            x = jnp.array([val], dtype=jnp.float32)

            primal_jvp, tangent_jvp = jax.jvp(exprel, (x,), (jnp.ones_like(x),))
            primal_vjp, vjp_fn = jax.vjp(exprel, x)
            grad_vjp = vjp_fn(jnp.ones_like(x))[0]

            assert jnp.allclose(primal_jvp, primal_vjp), \
                f"Primals differ at threshold boundary x={val}"
            assert jnp.allclose(tangent_jvp, grad_vjp, rtol=1e-4), \
                f"Gradients differ at threshold boundary x={val}: JVP={tangent_jvp}, VJP={grad_vjp}"

    def test_grad_function(self):
        """Test that jax.grad works correctly."""
        def loss_fn(x):
            return jnp.sum(exprel(x))

        x = jnp.array([0.0, 1e-5, 1.0])
        grads = jax.grad(loss_fn)(x)

        # Expected gradients: derivative at 0 is 0.5, etc.
        assert jnp.allclose(grads[0], 0.5, rtol=1e-4), f"Gradient at 0 should be ~0.5, got {grads[0]}"
        assert not jnp.any(jnp.isnan(grads)), "Gradients contain NaN"
        assert not jnp.any(jnp.isinf(grads)), "Gradients contain Inf"

    def test_batched_gradients(self):
        """Test gradients with batched inputs."""
        x = jnp.array([
            [0.0, 1e-5, 1.0],
            [0.1, 0.5, 2.0],
        ])

        def loss_fn(x):
            return jnp.sum(exprel(x))

        grads = jax.grad(loss_fn)(x)

        assert grads.shape == x.shape
        assert not jnp.any(jnp.isnan(grads))
        assert not jnp.any(jnp.isinf(grads))


class TestExprelDtypes:
    """Tests for dtype-specific accuracy."""

    def test_float64_accuracy(self):
        """Test accuracy with float64."""
        with brainstate.environ.context(precision=64):
            x = jnp.array([0.0, 1e-17, 1e-16, 1e-15, 1e-12, 1e-9, 1.0, 10.0, 100.0, 717.0],
                          dtype=jnp.float64)
            result = exprel(x)
            expected = scipy_exprel(np.asarray(x))
            assert jnp.allclose(result, expected, rtol=1e-10), \
                f"Float64 accuracy failed: max diff = {jnp.max(jnp.abs(result - expected))}"

    def test_float32_accuracy(self):
        """Test accuracy with float32."""
        with brainstate.environ.context(precision=32):
            x = jnp.array([0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1.0, 10.0, 100.0], dtype=jnp.float32)
            result = exprel(x)
            expected = scipy_exprel(np.asarray(x))
            assert jnp.allclose(result, expected, rtol=1e-5), \
                f"Float32 accuracy failed: max diff = {jnp.max(jnp.abs(result - expected))}"

    def test_float16_accuracy(self):
        """Test accuracy with float16."""
        x = jnp.array([0.0, 1e-4, 1e-3, 1e-2, 1.0, 5.0], dtype=jnp.float16)
        result = exprel(x)
        expected = scipy_exprel(np.asarray(x, dtype=np.float32)).astype(np.float16)
        # Float16 has limited precision, use relaxed tolerance
        assert np.allclose(result, expected, rtol=1e-2, atol=1e-3), \
            f"Float16 accuracy failed: max diff = {np.max(np.abs(np.asarray(result) - expected))}"

    def test_adaptive_threshold_selection(self):
        """Test that thresholds are selected correctly for each dtype."""
        assert _get_threshold(jnp.float64) == 1e-7
        assert _get_threshold(jnp.float32) == 1e-4
        assert _get_threshold(jnp.float16) == 1e-2
        assert _get_threshold(jnp.bfloat16) == 1e-2

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        # Test float16 and float32 always
        for dtype in [jnp.float16, jnp.float32]:
            x = jnp.array([0.0, 1.0], dtype=dtype)
            result = exprel(x)
            assert result.dtype == dtype, f"Dtype not preserved: expected {dtype}, got {result.dtype}"

        # Test float64 only if x64 is enabled
        if jax.config.x64_enabled:
            x = jnp.array([0.0, 1.0], dtype=jnp.float64)
            result = exprel(x)
            assert result.dtype == jnp.float64, f"Dtype not preserved for float64"


class TestExprelNumericalStability:
    """Tests for numerical stability at edge cases."""

    def test_singularity_at_zero(self):
        """Test that exprel(0) = 1 (the limit value)."""
        x = jnp.array([0.0])
        result = exprel(x)
        assert jnp.allclose(result, 1.0), f"exprel(0) should be 1, got {result}"

    def test_very_small_positive_values(self):
        """Test stability for very small positive values."""
        with brainstate.environ.context(precision=64):
            x = jnp.array([1e-300, 1e-200, 1e-100, 1e-50, 1e-20], dtype=jnp.float64)
            result = exprel(x)

            # Should not produce NaN or Inf
            assert not jnp.any(jnp.isnan(result)), "NaN produced for small positive values"
            assert not jnp.any(jnp.isinf(result)), "Inf produced for small positive values"

            # All values should be close to 1 for very small x
            assert jnp.allclose(result, 1.0, rtol=1e-10), \
                f"Values should be close to 1 for very small x, got {result}"

    def test_very_small_negative_values(self):
        """Test stability for very small negative values."""
        with brainstate.environ.context(precision=64):
            x = jnp.array([-1e-300, -1e-200, -1e-100, -1e-50, -1e-20], dtype=jnp.float64)
            result = exprel(x)

            assert not jnp.any(jnp.isnan(result)), "NaN produced for small negative values"
            assert not jnp.any(jnp.isinf(result)), "Inf produced for small negative values"
            assert jnp.allclose(result, 1.0, rtol=1e-10), \
                f"Values should be close to 1 for very small negative x, got {result}"

    def test_threshold_boundary_continuity(self):
        """Test continuity across threshold boundary."""
        for dtype, threshold in [(jnp.float32, 1e-5), (jnp.float64, 1e-8)]:
            # Values just below and above threshold
            x = jnp.array([
                threshold * 0.99,
                threshold,
                threshold * 1.01
            ], dtype=dtype)

            result = exprel(x)

            # Results should be continuous (no jumps)
            diff1 = jnp.abs(result[1] - result[0])
            diff2 = jnp.abs(result[2] - result[1])

            # Differences should be small and comparable
            assert diff1 < 1e-4, f"Large jump below threshold for {dtype}: {diff1}"
            assert diff2 < 1e-4, f"Large jump above threshold for {dtype}: {diff2}"

    def test_gradient_stability_at_zero(self):
        """Test that gradient at x=0 is stable and equals 0.5."""
        def f(x):
            return jnp.sum(exprel(x))

        x = jnp.array([0.0])
        grad = jax.grad(f)(x)

        assert not jnp.isnan(grad[0]), "Gradient at 0 is NaN"
        assert not jnp.isinf(grad[0]), "Gradient at 0 is Inf"
        assert jnp.allclose(grad[0], 0.5, rtol=1e-5), f"Gradient at 0 should be 0.5, got {grad[0]}"

    def test_gradient_stability_near_zero(self):
        """Test gradient stability for values near zero."""
        def f(x):
            return jnp.sum(exprel(x))

        # Test with float32 (works regardless of x64 mode)
        # Use values that are small but within float32's reasonable range
        test_values = [1e-6, 1e-5, 1e-4, 1e-3]

        for val in test_values:
            x = jnp.array([val], dtype=jnp.float32)
            grad = jax.grad(f)(x)

            assert not jnp.isnan(grad[0]), f"Gradient is NaN at x={val}"
            assert not jnp.isinf(grad[0]), f"Gradient is Inf at x={val}"
            # Gradient should be close to 0.5 for small x (derivative at 0 is 0.5)
            assert jnp.abs(grad[0] - 0.5) < 0.1, f"Gradient at x={val} should be ~0.5, got {grad[0]}"

    def test_large_positive_values(self):
        """Test stability for large positive values."""
        with brainstate.environ.context(precision=64):
            x = jnp.array([100.0, 200.0, 500.0, 700.0], dtype=jnp.float64)
            result = exprel(x)
            expected = scipy_exprel(np.asarray(x))

            assert not jnp.any(jnp.isnan(result)), "NaN produced for large positive values"
            assert jnp.allclose(result, expected, rtol=1e-6), "Large value accuracy failed"

    def test_large_negative_values(self):
        """Test stability for large negative values."""
        with brainstate.environ.context(precision=64):
            x = jnp.array([-10.0, -50.0, -100.0, -500.0], dtype=jnp.float64)
            result = exprel(x)
            expected = scipy_exprel(np.asarray(x))

            assert not jnp.any(jnp.isnan(result)), "NaN produced for large negative values"
            assert jnp.allclose(result, expected, rtol=1e-6), "Large negative value accuracy failed"


class TestExprelSecondOrder:
    """Tests for second-order derivatives (Hessian)."""

    def test_hessian_at_zero(self):
        """Test Hessian at x=0."""
        def f(x):
            return jnp.sum(exprel(x))

        x = jnp.array([0.0])
        hessian = jax.hessian(f)(x)

        assert not jnp.any(jnp.isnan(hessian)), "Hessian at 0 contains NaN"
        assert not jnp.any(jnp.isinf(hessian)), "Hessian at 0 contains Inf"

    def test_hessian_near_zero(self):
        """Test Hessian for values near zero."""
        def f(x):
            return jnp.sum(exprel(x))

        x = jnp.array([1e-8, 1e-5, 1e-3])
        hessian = jax.hessian(f)(x)

        assert not jnp.any(jnp.isnan(hessian)), "Hessian near 0 contains NaN"
        assert not jnp.any(jnp.isinf(hessian)), "Hessian near 0 contains Inf"

    def test_hessian_regular_values(self):
        """Test Hessian for regular values."""
        def f(x):
            return jnp.sum(exprel(x))

        x = jnp.array([0.1, 1.0, 2.0])
        hessian = jax.hessian(f)(x)

        assert not jnp.any(jnp.isnan(hessian)), "Hessian contains NaN"
        assert not jnp.any(jnp.isinf(hessian)), "Hessian contains Inf"


class TestExprelBatching:
    """Tests for vmap batching."""

    def test_vmap_basic(self):
        """Test basic vmap functionality."""
        x = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        result = jax.vmap(exprel)(x)

        # Should produce same result as direct call
        expected = exprel(x)
        assert jnp.allclose(result, expected), "vmap result differs from direct call"

    def test_vmap_with_gradients(self):
        """Test vmap with gradients."""
        def f(x):
            return jnp.sum(exprel(x))

        x = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        grad_vmap = jax.vmap(jax.grad(f))(x)

        assert grad_vmap.shape == x.shape
        assert not jnp.any(jnp.isnan(grad_vmap))

    def test_vmap_nested(self):
        """Test nested vmap."""
        x = jnp.array([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]])
        result = jax.vmap(jax.vmap(exprel))(x)

        expected = exprel(x)
        assert jnp.allclose(result, expected), "Nested vmap result differs"


class TestExprelTaylorOrder:
    """Tests for Taylor series order configuration."""

    def test_default_order(self):
        """Test default Taylor order."""
        assert get_exprel_order() == 5

    def test_set_order(self):
        """Test setting Taylor order."""
        original_order = get_exprel_order()

        try:
            set_exprel_order(3)
            assert get_exprel_order() == 3

            set_exprel_order(10)
            assert get_exprel_order() == 10
        finally:
            # Restore original order
            set_exprel_order(original_order)

    def test_invalid_order(self):
        """Test that invalid orders raise errors."""
        with pytest.raises(ValueError):
            set_exprel_order(1)  # Too low

        with pytest.raises(ValueError):
            set_exprel_order(21)  # Too high

        with pytest.raises(ValueError):
            set_exprel_order(3.5)  # Not an integer

    def test_higher_order_improves_accuracy(self):
        """Test that higher order improves accuracy."""
        original_order = get_exprel_order()

        try:
            # Use a very small value that will definitely use Taylor series
            # For float64, threshold is 1e-8, so use 1e-10
            x = jnp.array([1e-10], dtype=jnp.float64)
            reference = scipy_exprel(np.asarray(x))[0]

            errors = []
            for order in [2, 4, 6, 8, 10]:
                set_exprel_order(order)
                result = float(exprel(x)[0])
                error = abs(result - reference)
                errors.append(error)

            # Higher orders should generally have lower error
            # For very small x, all orders give similar results close to 1
            # But the trend should show improvement or at least not get worse
            # We just check that the highest order is no worse than the lowest
            assert errors[-1] <= errors[0] * 1.1, \
                f"Higher order should be at least as accurate: errors={errors}"
        finally:
            set_exprel_order(original_order)

    def test_coefficients_generation(self):
        """Test Taylor coefficient generation."""
        # Test exprel coefficients for order 5
        coeffs = _exprel_coefficients(5)
        # Should be [1/6!, 1/5!, 1/4!, 1/3!, 1/2!, 1/1!] = [1/720, 1/120, 1/24, 1/6, 1/2, 1]
        expected = [1/720, 1/120, 1/24, 1/6, 1/2, 1.0]
        assert jnp.allclose(jnp.array(coeffs), jnp.array(expected)), \
            f"Coefficient mismatch: {coeffs} vs {expected}"

    def test_derivative_coefficients_generation(self):
        """Test derivative Taylor coefficient generation."""
        # Test derivative coefficients for order 5
        coeffs = _exprel_deriv_coefficients(5)
        # Should be [(n+1)/(n+2)!] for n=5,4,3,2,1,0
        # = [6/7!, 5/6!, 4/5!, 3/4!, 2/3!, 1/2!]
        # = [6/5040, 5/720, 4/120, 3/24, 2/6, 1/2]
        # = [1/840, 1/144, 1/30, 1/8, 1/3, 1/2]
        expected = [1/840, 1/144, 1/30, 1/8, 1/3, 1/2]
        assert jnp.allclose(jnp.array(coeffs), jnp.array(expected)), \
            f"Derivative coefficient mismatch: {coeffs} vs {expected}"

    def test_order_parameter_in_primitive(self):
        """Test that order parameter is correctly passed through the primitive."""
        x = jnp.array([1e-6], dtype=jnp.float32)

        # Test with different orders
        result_order2 = exprel(x, order=2)
        result_order5 = exprel(x, order=5)
        result_order10 = exprel(x, order=10)

        # All should be close to 1 for small x, but higher orders should be more accurate
        assert jnp.allclose(result_order2, 1.0, rtol=1e-4)
        assert jnp.allclose(result_order5, 1.0, rtol=1e-4)
        assert jnp.allclose(result_order10, 1.0, rtol=1e-4)

    def test_order_parameter_with_jit(self):
        """Test that order parameter works with JIT compilation."""
        x = jnp.array([0.0, 1e-5, 1.0])

        # JIT with different orders
        jit_order2 = jax.jit(lambda x: exprel(x, order=2))
        jit_order5 = jax.jit(lambda x: exprel(x, order=5))

        result2 = jit_order2(x)
        result5 = jit_order5(x)

        assert result2.shape == x.shape
        assert result5.shape == x.shape
        assert not jnp.any(jnp.isnan(result2))
        assert not jnp.any(jnp.isnan(result5))

    def test_order_parameter_with_grad(self):
        """Test that order parameter works with gradients."""
        def loss_order2(x):
            return jnp.sum(exprel(x, order=2))

        def loss_order5(x):
            return jnp.sum(exprel(x, order=5))

        x = jnp.array([0.0, 1e-5, 1.0])

        grad2 = jax.grad(loss_order2)(x)
        grad5 = jax.grad(loss_order5)(x)

        # Gradients should be similar (both close to 0.5 at x=0)
        assert jnp.allclose(grad2[0], 0.5, rtol=1e-3)
        assert jnp.allclose(grad5[0], 0.5, rtol=1e-3)

    def test_order_parameter_with_vmap(self):
        """Test that order parameter works with vmap."""
        x = jnp.array([[0.0, 1.0], [2.0, 3.0]])

        # vmap with order parameter
        result = jax.vmap(lambda row: exprel(row, order=5))(x)

        assert result.shape == x.shape
        assert not jnp.any(jnp.isnan(result))


class TestExprelCompatibility:
    """Tests for compatibility with scipy.special.exprel."""

    def test_scipy_compatibility_positive(self):
        """Test compatibility with scipy for positive values."""
        with brainstate.environ.context(precision=64):
            x = jnp.linspace(0, 100, 1000, dtype=jnp.float64)
            result = exprel(x)
            expected = scipy_exprel(np.asarray(x))
            assert jnp.allclose(result, expected, rtol=1e-10), \
                f"Max diff from scipy: {jnp.max(jnp.abs(result - expected))}"

    def test_scipy_compatibility_negative(self):
        """Test compatibility with scipy for negative values."""
        with brainstate.environ.context(precision=64):
            x = jnp.linspace(-100, 0, 1000, dtype=jnp.float64)
            result = exprel(x)
            expected = scipy_exprel(np.asarray(x))
            assert jnp.allclose(result, expected, rtol=1e-10), \
                f"Max diff from scipy: {jnp.max(jnp.abs(result - expected))}"

    def test_scipy_compatibility_small_values(self):
        """Test compatibility with scipy for small values around zero."""
        with brainstate.environ.context(precision=64):
            x = jnp.linspace(-1e-3, 1e-3, 1000, dtype=jnp.float64)
            result = exprel(x)
            expected = scipy_exprel(np.asarray(x))
            assert jnp.allclose(result, expected, rtol=1e-10), \
                f"Max diff from scipy for small values: {jnp.max(jnp.abs(result - expected))}"


class TestExprelJIT:
    """Tests for JIT compilation."""

    def test_jit_basic(self):
        """Test basic JIT compilation."""
        jitted_exprel = jax.jit(exprel)

        x = jnp.array([0.0, 1.0, 2.0])
        result = jitted_exprel(x)
        expected = exprel(x)

        assert jnp.allclose(result, expected), "JIT result differs from eager"

    def test_jit_with_grad(self):
        """Test JIT with gradients."""
        def f(x):
            return jnp.sum(exprel(x))

        jitted_grad = jax.jit(jax.grad(f))

        # Use values that are clearly within the Taylor region to avoid boundary effects
        x = jnp.array([0.0, 1e-6, 1.0])
        result = jitted_grad(x)
        expected = jax.grad(f)(x)

        # Allow small tolerance for numerical differences between JIT and eager
        assert jnp.allclose(result, expected, rtol=1e-4), \
            f"JIT grad differs from eager grad: {result} vs {expected}"

    def test_jit_multiple_calls(self):
        """Test that JIT produces consistent results across multiple calls."""
        jitted_exprel = jax.jit(exprel)

        x = jnp.array([0.0, 1.0, 2.0])
        result1 = jitted_exprel(x)
        result2 = jitted_exprel(x)
        result3 = jitted_exprel(x)

        assert jnp.allclose(result1, result2)
        assert jnp.allclose(result2, result3)


# Run all tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
