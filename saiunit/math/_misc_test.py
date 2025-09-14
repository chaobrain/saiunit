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
from saiunit import meter, second
from saiunit._base import assert_quantity


class Array(u.CustomArray):
    def __init__(self, value):
        self.data = value


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
        assert hasattr(test_array, 'data')

        exprel_result = math.exprel(test_array.data)
        exprel_array = Array(exprel_result)
        assert isinstance(exprel_array, u.CustomArray)

        expected = exprel(np.asarray(x_values))
        assert jnp.allclose(exprel_array.data, expected, rtol=1e-6)

    def test_exprel_with_unitless_array(self):
        x_values = jnp.array([0.0, 1e-15, 1e-12, 1e-9, 1.0, 10.0])
        test_array = Array(x_values)

        assert isinstance(test_array, u.CustomArray)

        exprel_result = math.exprel(test_array.data)
        exprel_array = Array(exprel_result)
        assert isinstance(exprel_array, u.CustomArray)

        expected = exprel(np.asarray(x_values))
        assert jnp.allclose(exprel_array.data, expected, rtol=1e-6)

    def test_exprel_with_different_dtypes_array(self):
        # Test with float64 Array
        with brainstate.environ.context(precision=64):
            x64 = jnp.array([0.0, 1e-17, 1e-16, 1.0, 10.0], dtype=jnp.float64)
            test_array_64 = Array(x64)

            assert isinstance(test_array_64, u.CustomArray)

            exprel_result_64 = math.exprel(test_array_64.data)
            exprel_array_64 = Array(exprel_result_64)
            assert isinstance(exprel_array_64, u.CustomArray)

            expected_64 = exprel(np.asarray(x64))
            assert jnp.allclose(exprel_array_64.data, expected_64, rtol=1e-6)

        # Test with float32 Array
        with brainstate.environ.context(precision=32):
            x32 = jnp.array([0.0, 1e-9, 1e-8, 1.0, 10.0], dtype=jnp.float32)
            test_array_32 = Array(x32)

            assert isinstance(test_array_32, u.CustomArray)

            exprel_result_32 = math.exprel(test_array_32.data)
            exprel_array_32 = Array(exprel_result_32)
            assert isinstance(exprel_array_32, u.CustomArray)

            expected_32 = exprel(np.asarray(x32))
            assert jnp.allclose(exprel_array_32.data, expected_32, rtol=1e-6)

    def test_exprel_with_negative_values_array(self):
        with brainstate.environ.context(precision=64):
            x_neg = jnp.array([-1.0, -10.0, -0.1, -0.01], dtype=jnp.float64)
            test_array_neg = Array(x_neg)

            assert isinstance(test_array_neg, u.CustomArray)

            exprel_result_neg = math.exprel(test_array_neg.data)
            exprel_array_neg = Array(exprel_result_neg)
            assert isinstance(exprel_array_neg, u.CustomArray)

            expected_neg = exprel(np.asarray(x_neg))
            assert jnp.allclose(exprel_array_neg.data, expected_neg, rtol=1e-6)

    def test_exprel_with_zero_array(self):
        x_zero = jnp.array([0.0, 0.0, 0.0])
        test_array_zero = Array(x_zero)

        assert isinstance(test_array_zero, u.CustomArray)

        exprel_result_zero = math.exprel(test_array_zero.data)
        exprel_array_zero = Array(exprel_result_zero)
        assert isinstance(exprel_array_zero, u.CustomArray)

        # exprel(0) should be 1.0
        expected_zero = jnp.ones_like(x_zero)
        assert jnp.allclose(exprel_array_zero.data, expected_zero)

    def test_exprel_with_large_values_array(self):
        with brainstate.environ.context(precision=64):
            x_large = jnp.array([100.0, 200.0, 500.0, 717.0], dtype=jnp.float64)
            test_array_large = Array(x_large)

            assert isinstance(test_array_large, u.CustomArray)

            exprel_result_large = math.exprel(test_array_large.data)
            exprel_array_large = Array(exprel_result_large)
            assert isinstance(exprel_array_large, u.CustomArray)

            expected_large = exprel(np.asarray(x_large))
            # For large values, exprel(x) ≈ exp(x) / x
            assert jnp.allclose(exprel_array_large.data, expected_large, rtol=1e-6)

    def test_exprel_with_small_values_array(self):
        with brainstate.environ.context(precision=64):
            x_small = jnp.array([1e-20, 1e-18, 1e-16, 1e-14], dtype=jnp.float64)
            test_array_small = Array(x_small)

            assert isinstance(test_array_small, u.CustomArray)

            exprel_result_small = math.exprel(test_array_small.data)
            exprel_array_small = Array(exprel_result_small)
            assert isinstance(exprel_array_small, u.CustomArray)

            expected_small = exprel(np.asarray(x_small))
            # For small values, exprel(x) ≈ 1 + x/2 + x²/6 + ...
            assert jnp.allclose(exprel_array_small.data, expected_small, rtol=1e-12)

    def test_exprel_array_properties(self):
        x_values = jnp.array([0.1, 0.5, 1.0, 2.0])
        test_array = Array(x_values)

        assert isinstance(test_array, u.CustomArray)

        exprel_result = math.exprel(test_array.data)
        exprel_array = Array(exprel_result)

        # Verify that exprel_array maintains CustomArray properties
        assert isinstance(exprel_array, u.CustomArray)
        assert hasattr(exprel_array, 'data')
        assert exprel_array.data.shape == x_values.shape
        assert exprel_array.data.dtype == x_values.dtype

        # Verify mathematical property: exprel(x) = (exp(x) - 1) / x for x != 0
        for i, x_val in enumerate(x_values):
            if x_val != 0:
                expected_val = (jnp.exp(x_val) - 1) / x_val
                assert jnp.allclose(exprel_array.data[i], expected_val, rtol=1e-6)

    def test_array_custom_array_compatibility_with_exprel(self):
        x_data = jnp.array([0.0, 0.1, 1.0, 10.0])
        test_array = Array(x_data)

        assert isinstance(test_array, u.CustomArray)
        assert hasattr(test_array, 'data')

        # Test that we can use the array data in exprel function
        result = math.exprel(test_array.data)
        result_array = Array(result)

        assert isinstance(result_array, u.CustomArray)

        # Compare with direct computation
        direct_result = math.exprel(x_data)
        assert jnp.allclose(result_array.data, direct_result)


def test_constants_and_dtype_aliases():
    assert u.math.e == np.e
    assert u.math.pi == np.pi
    assert u.math.inf == np.inf
    assert np.isnan(u.math.nan)
    assert u.math.euler_gamma == np.euler_gamma
    assert u.math.dtype(jnp.float32) == jnp.dtype(jnp.float32)
    arr = jnp.zeros((2, 1))[..., u.math.newaxis]
    assert arr.shape == (2, 1, 1)
    assert u.math.inexact is jnp.inexact


def test_is_quantity():
    q = jnp.array([1., 2.]) * meter
    a = jnp.array([1., 2.])
    assert u.math.is_quantity(q)
    assert not u.math.is_quantity(a)


def test_ndim_shape_size_on_arrays_and_quantities():
    a = jnp.arange(6).reshape(2, 3)
    q = a * second
    assert u.math.ndim(a) == 2
    assert u.math.ndim(q) == 2
    assert u.math.shape(a) == (2, 3)
    assert u.math.shape(q) == (2, 3)
    assert u.math.size(a) == 6
    assert u.math.size(q) == 6
    assert u.math.size(a, 1) == 3
    assert u.math.size(q, 0) == 2
    assert u.math.shape(0) == ()
    assert u.math.size(0) == 1


def test_predicates_isreal_isscalar_isfinite_isinf_isnan():
    a = jnp.array([1 + 0j, 1 + 1j])
    q = a * meter
    np.testing.assert_array_equal(u.math.isreal(a), jnp.isreal(a))
    np.testing.assert_array_equal(u.math.isreal(q), jnp.isreal(a))

    assert u.math.isscalar(3.0)
    assert u.math.isscalar(3 * meter)
    assert not u.math.isscalar(jnp.array([3.0]))

    a = jnp.array([0.0, jnp.inf, -jnp.inf, jnp.nan])
    q = a * meter
    np.testing.assert_array_equal(u.math.isfinite(a), jnp.isfinite(a))
    np.testing.assert_array_equal(u.math.isfinite(q), jnp.isfinite(a))
    np.testing.assert_array_equal(u.math.isinf(a), jnp.isinf(a))
    np.testing.assert_array_equal(u.math.isinf(q), jnp.isinf(a))
    np.testing.assert_array_equal(u.math.isnan(a), jnp.isnan(a))
    np.testing.assert_array_equal(u.math.isnan(q), jnp.isnan(a))


def test_finfo_iinfo_with_quantities_and_dtypes():
    qf = jnp.array([0.0], dtype=jnp.float32) * meter
    qi = jnp.array([0], dtype=jnp.int32) * meter
    assert u.math.finfo(qf).dtype == jnp.finfo(jnp.float32).dtype
    assert u.math.iinfo(qi).dtype == jnp.iinfo(jnp.int32).dtype
    assert u.math.finfo(jnp.float64).dtype == jnp.finfo(jnp.float64).dtype
    assert u.math.iinfo(jnp.int16).dtype == jnp.iinfo(jnp.int16).dtype


def test_broadcast_shapes_and_result_type_issubdtype():
    assert u.math.broadcast_shapes((2, 1), (1, 3)) == (2, 3)
    assert u.math.issubdtype(jnp.float32, jnp.floating)
    assert not u.math.issubdtype(jnp.int32, jnp.floating)
    x = jnp.array([1, 2], dtype=jnp.int32) * meter
    y = jnp.array([1.0], dtype=jnp.float64)
    assert u.math.result_type(x, y) == jnp.result_type(x.mantissa, y)


def test_get_dtype_and_is_float_is_int():
    a = jnp.array([1., 2., 3.])
    q = a * second
    assert u.math.get_dtype(a) == a.dtype
    assert u.math.get_dtype(q) == a.dtype
    assert u.math.is_float(a)
    assert u.math.is_float(q)

    ai = jnp.array([1, 2, 3])
    qi = ai * meter
    assert u.math.is_int(ai)
    assert u.math.is_int(qi)

    with brainstate.environ.context(precision=64):
        assert u.math.get_dtype(True) is bool
        assert u.math.get_dtype(3) == brainstate.environ.ditype()
        assert u.math.get_dtype(3.0) == brainstate.environ.dftype()
        # assert u.math.get_dtype(3 + 0j) == brainstate.environ.dctype()


def test_gradient_quantity_no_spacing_returns_quantity():
    f = jnp.array([0.0, 1.0, 4.0, 9.0], dtype=jnp.float32) * meter
    g = u.math.gradient(f)
    assert isinstance(g, u.Quantity)
    assert_quantity(g, jnp.gradient(f.mantissa), unit=meter)


def test_gradient_with_unit_spacing_and_multi_axis():
    f = jnp.arange(12.0, dtype=jnp.float64).reshape(3, 4) * meter
    dy = 0.5 * second
    dx = 2.0 * second
    gy, gx = u.math.gradient(f, dy, dx)
    assert isinstance(gy, u.Quantity) and isinstance(gx, u.Quantity)
    assert_quantity(gy, jnp.gradient(f.mantissa, dy.mantissa, axis=0), unit=meter / second)
    assert_quantity(gx, jnp.gradient(f.mantissa, dx.mantissa, axis=1), unit=meter / second)


def test_gradient_edge_order_not_supported():
    f = jnp.array([0.0, 1.0, 4.0]) * meter
    try:
        u.math.gradient(f, edge_order=2)
        assert False, "Expected NotImplementedError for edge_order"
    except NotImplementedError:
        pass


def test_window_functions_shapes():
    n = 8
    np.testing.assert_allclose(u.math.bartlett(n), jnp.bartlett(n))
    np.testing.assert_allclose(u.math.blackman(n), jnp.blackman(n))
    np.testing.assert_allclose(u.math.hamming(n), jnp.hamming(n))
    np.testing.assert_allclose(u.math.hanning(n), jnp.hanning(n))
    np.testing.assert_allclose(u.math.kaiser(n, 14.0), jnp.kaiser(n, 14.0))
