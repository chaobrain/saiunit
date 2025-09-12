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

import brainstate
import jax.numpy as jnp
import numpy as np
from scipy.special import exprel

import saiunit as u
from saiunit import math


class Array(u.CustomArray):
    def __init__(self, value):
        self.value = value


def test_exprel():
    np.printoptions(precision=30)

    print()
    with brainstate.environ.context(precision=64):
        # Test with float64 input
        x = jnp.array([0.0, 1e-17, 1e-16, 1e-15, 1e-12, 1e-9, 1.0, 10.0, 100.0, 717.0, 718.0], dtype=jnp.float64)
        print(math.exprel(x), '\n', exprel(np.asarray(x)))
        assert jnp.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-6)

    with brainstate.environ.context(precision=32):
        # Test with float32 input
        x = jnp.array([0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1.0, 10.0, 100.0], dtype=jnp.float32)
        print(math.exprel(x), '\n', exprel(np.asarray(x)))
        assert jnp.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-6)

    # Test with float16 input
    x = jnp.array([0.0, 1e-5, 1e-4, 1e-3, 1.0, 10.0], dtype=jnp.float16)
    print(math.exprel(x), '\n', exprel(np.asarray(x)))
    assert np.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-03, atol=1e-05)

    # # Test with float8 input
    # x = jnp.array([0.0, 1e-5, 1e-4, 1e-3, 1.0, ], dtype=jnp.float8_e5m2fnuz)
    # print(math.exprel(x), '\n', exprel(np.asarray(x)))
    # assert np.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-03, atol=1e-05)

    # Test with int input
    x = jnp.array([0., 1., 10.])
    print(math.exprel(x), '\n', exprel(np.asarray(x)))
    assert jnp.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-6)

    with brainstate.environ.context(precision=64):
        # Test with negative input
        x = jnp.array([-1.0, -10.0, -100.0], dtype=jnp.float64)
        print(math.exprel(x), '\n', exprel(np.asarray(x)))
        assert jnp.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-6)


class TestMiscWithArrayCustomArray:

    def test_exprel_with_array(self):
        x_values = jnp.array([0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1.0, 10.0])
        test_array = Array(x_values)

        assert isinstance(test_array, u.CustomArray)
        assert hasattr(test_array, 'value')

        exprel_result = math.exprel(test_array.value)
        exprel_array = Array(exprel_result)
        assert isinstance(exprel_array, u.CustomArray)

        expected = exprel(np.asarray(x_values))
        assert jnp.allclose(exprel_array.value, expected, rtol=1e-6)

    def test_exprel_with_unitless_array(self):
        x_values = jnp.array([0.0, 1e-15, 1e-12, 1e-9, 1.0, 10.0])
        test_array = Array(x_values)

        assert isinstance(test_array, u.CustomArray)

        exprel_result = math.exprel(test_array.value)
        exprel_array = Array(exprel_result)
        assert isinstance(exprel_array, u.CustomArray)

        expected = exprel(np.asarray(x_values))
        assert jnp.allclose(exprel_array.value, expected, rtol=1e-6)

    def test_exprel_with_different_dtypes_array(self):
        # Test with float64 Array
        with brainstate.environ.context(precision=64):
            x64 = jnp.array([0.0, 1e-17, 1e-16, 1.0, 10.0], dtype=jnp.float64)
            test_array_64 = Array(x64)

            assert isinstance(test_array_64, u.CustomArray)

            exprel_result_64 = math.exprel(test_array_64.value)
            exprel_array_64 = Array(exprel_result_64)
            assert isinstance(exprel_array_64, u.CustomArray)

            expected_64 = exprel(np.asarray(x64))
            assert jnp.allclose(exprel_array_64.value, expected_64, rtol=1e-6)

        # Test with float32 Array
        with brainstate.environ.context(precision=32):
            x32 = jnp.array([0.0, 1e-9, 1e-8, 1.0, 10.0], dtype=jnp.float32)
            test_array_32 = Array(x32)

            assert isinstance(test_array_32, u.CustomArray)

            exprel_result_32 = math.exprel(test_array_32.value)
            exprel_array_32 = Array(exprel_result_32)
            assert isinstance(exprel_array_32, u.CustomArray)

            expected_32 = exprel(np.asarray(x32))
            assert jnp.allclose(exprel_array_32.value, expected_32, rtol=1e-6)

    def test_exprel_with_negative_values_array(self):
        with brainstate.environ.context(precision=64):
            x_neg = jnp.array([-1.0, -10.0, -0.1, -0.01], dtype=jnp.float64)
            test_array_neg = Array(x_neg)

            assert isinstance(test_array_neg, u.CustomArray)

            exprel_result_neg = math.exprel(test_array_neg.value)
            exprel_array_neg = Array(exprel_result_neg)
            assert isinstance(exprel_array_neg, u.CustomArray)

            expected_neg = exprel(np.asarray(x_neg))
            assert jnp.allclose(exprel_array_neg.value, expected_neg, rtol=1e-6)

    def test_exprel_with_zero_array(self):
        x_zero = jnp.array([0.0, 0.0, 0.0])
        test_array_zero = Array(x_zero)

        assert isinstance(test_array_zero, u.CustomArray)

        exprel_result_zero = math.exprel(test_array_zero.value)
        exprel_array_zero = Array(exprel_result_zero)
        assert isinstance(exprel_array_zero, u.CustomArray)

        # exprel(0) should be 1.0
        expected_zero = jnp.ones_like(x_zero)
        assert jnp.allclose(exprel_array_zero.value, expected_zero)

    def test_exprel_with_large_values_array(self):
        with brainstate.environ.context(precision=64):
            x_large = jnp.array([100.0, 200.0, 500.0, 717.0], dtype=jnp.float64)
            test_array_large = Array(x_large)

            assert isinstance(test_array_large, u.CustomArray)

            exprel_result_large = math.exprel(test_array_large.value)
            exprel_array_large = Array(exprel_result_large)
            assert isinstance(exprel_array_large, u.CustomArray)

            expected_large = exprel(np.asarray(x_large))
            # For large values, exprel(x) ≈ exp(x) / x
            assert jnp.allclose(exprel_array_large.value, expected_large, rtol=1e-6)

    def test_exprel_with_small_values_array(self):
        with brainstate.environ.context(precision=64):
            x_small = jnp.array([1e-20, 1e-18, 1e-16, 1e-14], dtype=jnp.float64)
            test_array_small = Array(x_small)

            assert isinstance(test_array_small, u.CustomArray)

            exprel_result_small = math.exprel(test_array_small.value)
            exprel_array_small = Array(exprel_result_small)
            assert isinstance(exprel_array_small, u.CustomArray)

            expected_small = exprel(np.asarray(x_small))
            # For small values, exprel(x) ≈ 1 + x/2 + x²/6 + ...
            assert jnp.allclose(exprel_array_small.value, expected_small, rtol=1e-12)

    def test_exprel_array_properties(self):
        x_values = jnp.array([0.1, 0.5, 1.0, 2.0])
        test_array = Array(x_values)

        assert isinstance(test_array, u.CustomArray)

        exprel_result = math.exprel(test_array.value)
        exprel_array = Array(exprel_result)

        # Verify that exprel_array maintains CustomArray properties
        assert isinstance(exprel_array, u.CustomArray)
        assert hasattr(exprel_array, 'value')
        assert exprel_array.value.shape == x_values.shape
        assert exprel_array.value.dtype == x_values.dtype

        # Verify mathematical property: exprel(x) = (exp(x) - 1) / x for x != 0
        for i, x_val in enumerate(x_values):
            if x_val != 0:
                expected_val = (jnp.exp(x_val) - 1) / x_val
                assert jnp.allclose(exprel_array.value[i], expected_val, rtol=1e-6)

    def test_array_custom_array_compatibility_with_exprel(self):
        x_data = jnp.array([0.0, 0.1, 1.0, 10.0])
        test_array = Array(x_data)

        assert isinstance(test_array, u.CustomArray)
        assert hasattr(test_array, 'value')

        # Test that we can use the array value in exprel function
        result = math.exprel(test_array.value)
        result_array = Array(result)

        assert isinstance(result_array, u.CustomArray)

        # Compare with direct computation
        direct_result = math.exprel(x_data)
        assert jnp.allclose(result_array.value, direct_result)
