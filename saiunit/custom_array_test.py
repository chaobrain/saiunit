# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import saiunit as u


@jax.tree_util.register_pytree_node_class
class Array(brainstate.State, u.CustomArray):
    @property
    def data(self):
        return self.value

    @data.setter
    def data(self, value):
        self.value = value


class TestArray(unittest.TestCase):
    def setUp(self):
        # Basic arrays for testing
        self.np_array = np.array([1, 2, 3])
        self.jax_array = jnp.array([1, 2, 3])
        # Create ArrayImpl instances
        self.array_impl_np = Array(self.np_array)
        self.array_impl_np.value = self.np_array
        self.array_impl_jax = Array(self.jax_array)
        self.array_impl_jax.value = self.jax_array
        # More complex arrays for advanced testing
        self.ones_2d = np.ones((2, 3))
        self.array_impl_2d = Array(self.ones_2d)
        self.array_impl_2d.value = self.ones_2d
        # Scalar array
        self.scalar = np.array(5.0)
        self.array_impl_scalar = Array(self.scalar)
        self.array_impl_scalar.value = self.scalar

    def test_basic_properties(self):
        # Test dtype property
        self.assertEqual(self.array_impl_np.dtype, self.np_array.dtype)
        self.assertEqual(self.array_impl_jax.dtype, self.jax_array.dtype)

        # Test shape property
        self.assertEqual(self.array_impl_np.shape, self.np_array.shape)
        self.assertEqual(self.array_impl_jax.shape, self.jax_array.shape)

        # Test ndim property
        self.assertEqual(self.array_impl_np.ndim, self.np_array.ndim)
        self.assertEqual(self.array_impl_2d.ndim, self.ones_2d.ndim)

        # Test size property
        self.assertEqual(self.array_impl_np.size, self.np_array.size)
        self.assertEqual(self.array_impl_2d.size, self.ones_2d.size)

        # Test real property (should work for real arrays)
        np.testing.assert_array_equal(self.array_impl_np.real, self.np_array.real)

        # Test T property (transpose)
        np.testing.assert_array_equal(self.array_impl_2d.T, self.ones_2d.T)

    def test_unary_operations(self):
        # Test __neg__
        np.testing.assert_array_equal(-self.array_impl_np, -self.np_array)

        # Test __pos__
        np.testing.assert_array_equal(+self.array_impl_np, +self.np_array)

        # Test __abs__
        neg_array = Array(np.array([-1, 2, -3]))
        np.testing.assert_array_equal(abs(neg_array), np.abs(neg_array.value))

        # Test __invert__ (bitwise NOT)
        uint_array = Array(np.array([1, 2, 3], dtype=np.uint8))
        np.testing.assert_array_equal(~uint_array, ~uint_array.value)

    def test_binary_operations(self):
        # Test __add__ and __radd__
        np.testing.assert_array_equal(self.array_impl_np + 2, self.np_array + 2)
        np.testing.assert_array_equal(2 + self.array_impl_np, 2 + self.np_array)

        # Test __sub__ and __rsub__
        np.testing.assert_array_equal(self.array_impl_np - 1, self.np_array - 1)
        np.testing.assert_array_equal(5 - self.array_impl_np, 5 - self.np_array)

        # Test __mul__ and __rmul__
        np.testing.assert_array_equal(self.array_impl_np * 2, self.np_array * 2)
        np.testing.assert_array_equal(2 * self.array_impl_np, 2 * self.np_array)

        # Test __truediv__ and __rtruediv__
        np.testing.assert_array_almost_equal(self.array_impl_np / 2, self.np_array / 2)
        np.testing.assert_array_almost_equal(6 / self.array_impl_np, 6 / self.np_array)

        # Test __pow__ and __rpow__
        np.testing.assert_array_equal(self.array_impl_np ** 2, self.np_array ** 2)
        np.testing.assert_array_equal(2 ** self.array_impl_np, 2 ** self.np_array)

        # Test __matmul__
        a = Array(np.array([[1, 2], [3, 4]]))
        b = Array(np.array([[5, 6], [7, 8]]))
        np.testing.assert_array_equal(a @ b, a.value @ b.value)

    def test_inplace_operations(self):
        # Test __iadd__
        test_array = Array(np.array([1, 2, 3]))
        test_array += 1
        np.testing.assert_array_equal(test_array.value, np.array([2, 3, 4]))

        # Test __isub__
        test_array = Array(np.array([1, 2, 3]))
        test_array -= 1
        np.testing.assert_array_equal(test_array.value, np.array([0, 1, 2]))

        # Test __imul__
        test_array = Array(np.array([1, 2, 3]))
        test_array *= 2
        np.testing.assert_array_equal(test_array.value, np.array([2, 4, 6]))

        # Test __itruediv__
        test_array = Array(np.array([2, 4, 6]))
        test_array /= 2
        np.testing.assert_array_equal(test_array.value, np.array([1, 2, 3]))

    def test_iterator(self):
        # Test __iter__
        values = [x for x in self.array_impl_np]
        expected_values = [x for x in self.np_array]
        self.assertEqual(values, expected_values)

    def test_misc_methods(self):
        # Test fill method
        test_array = Array(np.array([1, 2, 3]))
        test_array.fill(5)
        np.testing.assert_array_equal(test_array.value, np.array([5, 5, 5]))

        # Test flatten method
        flattened = self.array_impl_2d.flatten()
        np.testing.assert_array_equal(flattened, self.ones_2d.flatten())

        # Test item method
        scalar_impl = Array(np.array(5.0))
        self.assertEqual(scalar_impl.item(), 5.0)

        # Test view method
        view_array = Array(np.array([1, 2, 3], dtype=np.int32))
        int_view = view_array.view(np.int32)
        self.assertEqual(int_view.dtype, np.int32)


class TestArray2(unittest.TestCase):
    def setUp(self):
        # Basic arrays for testing
        self.np_array = np.array([1, 2, 3])
        self.jax_array = jnp.array([1, 2, 3])
        # Create ArrayImpl instances
        self.array_impl_np = Array(self.np_array)
        self.array_impl_jax = Array(self.jax_array)
        # More complex arrays for advanced testing
        self.ones_2d = np.ones((2, 3))
        self.array_impl_2d = Array(self.ones_2d)
        # Scalar array
        self.scalar = np.array(5.0)
        self.array_impl_scalar = Array(self.scalar)

    def test_basic_properties(self):
        # Test dtype property
        self.assertEqual(self.array_impl_np.dtype, self.np_array.dtype)
        self.assertEqual(self.array_impl_jax.dtype, self.jax_array.dtype)

        # Test shape property
        self.assertEqual(self.array_impl_np.shape, self.np_array.shape)
        self.assertEqual(self.array_impl_jax.shape, self.jax_array.shape)

        # Test ndim property
        self.assertEqual(self.array_impl_np.ndim, self.np_array.ndim)
        self.assertEqual(self.array_impl_2d.ndim, self.ones_2d.ndim)

        # Test real property (should work for real arrays)
        np.testing.assert_array_equal(self.array_impl_np.real, self.np_array.real)

    def test_unary_operations(self):
        # Test __neg__
        np.testing.assert_array_equal(-self.array_impl_np, -self.np_array)

        # Test __pos__
        np.testing.assert_array_equal(+self.array_impl_np, +self.np_array)

        # Test __abs__
        neg_array = Array(np.array([-1, 2, -3]))
        np.testing.assert_array_equal(abs(neg_array), np.abs(neg_array.value))

        # Test __invert__ (bitwise NOT)
        uint_array = Array(np.array([1, 2, 3], dtype=np.uint8))
        np.testing.assert_array_equal(~uint_array, ~uint_array.value)

    def test_binary_operations(self):
        # Test __add__ and __radd__
        np.testing.assert_array_equal(self.array_impl_np + 2, self.np_array + 2)
        np.testing.assert_array_equal(2 + self.array_impl_np, 2 + self.np_array)

        # Test __sub__ and __rsub__
        np.testing.assert_array_equal(self.array_impl_np - 1, self.np_array - 1)
        np.testing.assert_array_equal(5 - self.array_impl_np, 5 - self.np_array)

        # Test __mul__ and __rmul__
        np.testing.assert_array_equal(self.array_impl_np * 2, self.np_array * 2)
        np.testing.assert_array_equal(2 * self.array_impl_np, 2 * self.np_array)

        # Test __truediv__ and __rtruediv__
        np.testing.assert_array_almost_equal(self.array_impl_np / 2, self.np_array / 2)
        np.testing.assert_array_almost_equal(6 / self.array_impl_np, 6 / self.np_array)

        # Test __pow__ and __rpow__
        np.testing.assert_array_equal(self.array_impl_np ** 2, self.np_array ** 2)
        np.testing.assert_array_equal(2 ** self.array_impl_np, 2 ** self.np_array)

        # Test __matmul__
        a = Array(np.array([[1, 2], [3, 4]]))
        b = Array(np.array([[5, 6], [7, 8]]))
        np.testing.assert_array_equal(a @ b, a.value @ b.value)

    def test_inplace_operations(self):
        # Test __iadd__
        test_array = Array(np.array([1, 2, 3]))
        test_array += 1
        np.testing.assert_array_equal(test_array.value, np.array([2, 3, 4]))

        # Test __isub__
        test_array = Array(np.array([1, 2, 3]))
        test_array -= 1
        np.testing.assert_array_equal(test_array.value, np.array([0, 1, 2]))

        # Test __imul__
        test_array = Array(np.array([1, 2, 3]))
        test_array *= 2
        np.testing.assert_array_equal(test_array.value, np.array([2, 4, 6]))

        # Test __itruediv__
        test_array = Array(np.array([2, 4, 6]))
        test_array /= 2
        np.testing.assert_array_equal(test_array.value, np.array([1, 2, 3]))

    def test_iterator(self):
        # Test __iter__
        values = [x for x in self.array_impl_np]
        expected_values = [x for x in self.np_array]
        self.assertEqual(values, expected_values)

    def test_misc_methods(self):
        # Test fill method
        test_array = Array(np.array([1, 2, 3]))
        test_array.fill(5)
        np.testing.assert_array_equal(test_array.value, np.array([5, 5, 5]))

        # Test flatten method
        flattened = self.array_impl_2d.flatten()
        np.testing.assert_array_equal(flattened, self.ones_2d.flatten())

        # Test item method
        scalar_impl = Array(np.array(5.0))
        self.assertEqual(scalar_impl.item(), 5.0)

        # Test view method
        view_array = Array(np.array([1, 2, 3], dtype=np.int32))
        int_view = view_array.view(np.int32)
        self.assertEqual(int_view.dtype, np.int32)

    def test_advanced_indexing(self):
        # Test basic indexing
        arr = Array(np.array([1, 2, 3, 4, 5]))
        self.assertEqual(arr[0], 1)
        self.assertEqual(arr[-1], 5)
        np.testing.assert_array_equal(arr[1:4], np.array([2, 3, 4]))

        # Test boolean indexing
        bool_mask = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(arr[bool_mask], np.array([1, 3, 5]))

        # Test integer array indexing
        idx = np.array([0, 2, 4])
        np.testing.assert_array_equal(arr[idx], np.array([1, 3, 5]))

        # Test assignment via indexing
        arr = Array(np.array([1, 2, 3, 4, 5]))
        arr[1:4] = 10
        np.testing.assert_array_equal(arr.value, np.array([1, 10, 10, 10, 5]))

    def test_comparison_operators(self):
        arr = Array(np.array([1, 2, 3]))

        # Test ==, !=, <, <=, >, >=
        # np.testing.assert_array_equal(arr == 2, np.array([False, True, False]))
        np.testing.assert_array_equal(arr != 2, np.array([True, False, True]))
        np.testing.assert_array_equal(arr < 2, np.array([True, False, False]))
        np.testing.assert_array_equal(arr <= 2, np.array([True, True, False]))
        np.testing.assert_array_equal(arr > 2, np.array([False, False, True]))
        np.testing.assert_array_equal(arr >= 2, np.array([False, True, True]))

        # # Test array vs array comparisons
        # arr2 = Array(np.array([2, 2, 2]))
        # np.testing.assert_array_equal(arr == arr2, np.array([False, True, False]))

    def test_jax_integration(self):
        # Test JAX transformations with Array
        arr = Array(jnp.array([1.0, 2.0, 3.0]))

        # Test jit compilation
        @jax.jit
        def square(x):
            return x * x

        result = square(arr)
        self.assertIsInstance(result, jax.Array)
        np.testing.assert_array_equal(result, jnp.array([1.0, 4.0, 9.0]))

        # Test grad with Array
        @jax.grad
        def sum_squares(x):
            return jnp.sum(x * x)

        grad_result = sum_squares(arr)
        self.assertIsInstance(grad_result, Array)
        np.testing.assert_array_equal(grad_result, jnp.array([2.0, 4.0, 6.0]))

    def test_edge_cases(self):
        # Test empty array
        empty_arr = Array(np.array([]))
        self.assertEqual(empty_arr.shape, (0,))
        self.assertEqual(empty_arr.ndim, 1)

        # Test very large array
        large_arr = Array(np.ones((1000,)))
        self.assertEqual(large_arr.shape, (1000,))

        # Test high-dimensional array
        high_dim = Array(np.zeros((2, 3, 4, 5)))
        self.assertEqual(high_dim.ndim, 4)
        self.assertEqual(high_dim.shape, (2, 3, 4, 5))

    def test_error_handling(self):
        # Test setting value with different tree structure
        arr = Array(np.array([1, 2, 3]))

        # Test invalid operations
        with self.assertRaises(TypeError):
            # Attempt to add incompatible types
            arr + "string"

    def test_copy_and_clone(self):
        # Test copy method
        arr = Array(np.array([1, 2, 3]))
        arr_copy = arr.copy()

        # Check they have the same value but are different objects
        np.testing.assert_array_equal(arr.value, arr_copy.value)
        self.assertIsNot(arr, arr_copy)

        # Modify the copy and check original is unchanged
        arr_copy.value = np.array([4, 5, 6])
        np.testing.assert_array_equal(arr.value, np.array([1, 2, 3]))

        # Test replace method
        arr_replaced = arr.replace(value=np.array([7, 8, 9]))
        np.testing.assert_array_equal(arr_replaced.value, np.array([7, 8, 9]))
        # Original should remain unchanged
        np.testing.assert_array_equal(arr.value, np.array([1, 2, 3]))

    def test_state_integration(self):
        # Test State methods
        arr = Array(np.array([1, 2, 3]), name="test_array")
        self.assertEqual(arr.name, "test_array")

        # Test numel method
        self.assertEqual(arr.numel(), 3)
        self.assertEqual(self.array_impl_2d.numel(), 6)  # 2x3 array

        # Test stack level operations
        original_level = arr.stack_level
        arr.increase_stack_level()
        self.assertEqual(arr.stack_level, original_level + 1)
        arr.decrease_stack_level()
        self.assertEqual(arr.stack_level, original_level)

    def test_brainstate_integration(self):
        # Test value_call method
        arr = Array(np.array([1, 2, 3]))
        result = arr.value_call(lambda x: x * 2)
        np.testing.assert_array_equal(result, np.array([2, 4, 6]))

        # Test with brainstate functions
        import brainstate as bs

        # Test check_state_value_tree context manager
        with bs.check_state_value_tree():
            # This should work since tree structure is the same
            arr.value = np.array([4, 5, 6])

            # This should fail
            with self.assertRaises(ValueError):
                arr.value = (np.array([1, 2, 3]),)


# =========================================================================
# Integration tests migrated from _base_test.py (TestArrayWithCustomArray)
# =========================================================================

from saiunit._unit_common import *
from saiunit._unit_shortcuts import kHz, ms, mV, nS


@jax.tree_util.register_pytree_node_class
class SimpleArray(u.CustomArray):
    """Standalone CustomArray subclass for integration tests (no brainstate dependency)."""

    def __init__(self, value):
        self.data = value


class TestArrayWithCustomArrayIntegration:
    """Integration tests for CustomArray subclass with physical units (pytest style)."""

    def test_array_properties(self):
        array_1d = SimpleArray(np.array([1.0, 2.0, 3.0]))
        array_2d = SimpleArray(np.array([[1, 2], [3, 4]]))

        assert array_1d.dtype == np.float64
        assert np.issubdtype(array_2d.dtype, np.integer)
        assert array_1d.shape == (3,)
        assert array_2d.shape == (2, 2)
        assert array_1d.ndim == 1
        assert array_2d.ndim == 2
        assert array_1d.size == 3
        assert array_2d.size == 4

    def test_array_arithmetic_operations(self):
        arr = SimpleArray(np.array([1.0, 2.0, 3.0]))

        np.testing.assert_array_equal(arr + 2.0, np.array([3.0, 4.0, 5.0]))
        np.testing.assert_array_equal(2.0 + arr, np.array([3.0, 4.0, 5.0]))
        np.testing.assert_array_equal(arr - 1.0, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(arr * 2.0, np.array([2.0, 4.0, 6.0]))
        np.testing.assert_array_equal(arr / 2.0, np.array([0.5, 1.0, 1.5]))
        np.testing.assert_array_equal(arr ** 2, np.array([1.0, 4.0, 9.0]))

    def test_array_inplace_operations(self):
        arr = SimpleArray(np.array([1.0, 2.0, 3.0]))
        arr += 1.0
        np.testing.assert_array_equal(arr.data, np.array([2.0, 3.0, 4.0]))

        arr -= 1.0
        np.testing.assert_array_equal(arr.data, np.array([1.0, 2.0, 3.0]))

        arr *= 2.0
        np.testing.assert_array_equal(arr.data, np.array([2.0, 4.0, 6.0]))

        arr /= 2.0
        np.testing.assert_array_equal(arr.data, np.array([1.0, 2.0, 3.0]))

    def test_array_comparison_operations(self):
        arr1 = SimpleArray(np.array([1.0, 2.0, 3.0]))
        arr2 = SimpleArray(np.array([2.0, 2.0, 2.0]))

        np.testing.assert_array_equal(arr1 == 2.0, np.array([False, True, False]))
        np.testing.assert_array_equal(arr1 != 2.0, np.array([True, False, True]))
        np.testing.assert_array_equal(arr1 < arr2, np.array([True, False, False]))
        np.testing.assert_array_equal(arr1 > arr2, np.array([False, False, True]))

    def test_array_with_units(self):
        voltage1 = SimpleArray(np.array([1.0, 2.0])) * mV
        voltage2 = SimpleArray(np.array([3.0, 4.0])) * mV

        voltage_sum = voltage1 + voltage2
        np.testing.assert_array_almost_equal(voltage_sum.mantissa, np.array([4.0, 6.0]))
        assert voltage_sum.unit == mV

        voltage_scaled = voltage1 * 2.0
        np.testing.assert_array_almost_equal(voltage_scaled.mantissa, np.array([2.0, 4.0]))
        assert voltage_scaled.unit == mV

        voltage_in_v = voltage1.to(volt)
        np.testing.assert_array_almost_equal(voltage_in_v.mantissa, np.array([0.001, 0.002]))

    def test_array_statistical_methods(self):
        arr = SimpleArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        assert float(arr.mean()) == 3.0
        assert float(arr.sum()) == 15.0
        assert float(arr.min()) == 1.0
        assert float(arr.max()) == 5.0
        assert abs(float(arr.std()) - np.std([1, 2, 3, 4, 5])) < 1e-6
        assert abs(float(arr.var()) - np.var([1, 2, 3, 4, 5])) < 1e-6

    def test_array_manipulation_methods(self):
        arr = SimpleArray(np.array([1, 2, 3, 4, 5, 6]))
        reshaped = arr.reshape(2, 3)
        assert reshaped.shape == (2, 3)
        np.testing.assert_array_equal(reshaped, np.array([[1, 2, 3], [4, 5, 6]]))

        arr_2d = SimpleArray(np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(arr_2d.T, np.array([[1, 3], [2, 4]]))

        np.testing.assert_array_equal(arr_2d.flatten(), np.array([1, 2, 3, 4]))

        arr_squeezable = SimpleArray(np.array([[[1, 2, 3]]]))
        np.testing.assert_array_equal(arr_squeezable.squeeze(), np.array([1, 2, 3]))

    def test_array_indexing_and_slicing(self):
        arr = SimpleArray(np.array([10, 20, 30, 40, 50]))

        assert arr[0] == 10
        assert arr[-1] == 50
        np.testing.assert_array_equal(arr[1:4], np.array([20, 30, 40]))

        mask = arr > 25
        np.testing.assert_array_equal(arr[mask], np.array([30, 40, 50]))

        arr_copy = SimpleArray(np.array([10, 20, 30, 40, 50]))
        arr_copy[1:3] = 99
        np.testing.assert_array_equal(arr_copy.data, np.array([10, 99, 99, 40, 50]))

    def test_jax_compatibility(self):
        jax_arr = SimpleArray(jnp.array([1.0, 2.0, 3.0]))

        result = jax_arr * 2.0
        np.testing.assert_array_equal(result, jnp.array([2.0, 4.0, 6.0]))

        @jax.jit
        def square_array(x):
            return x * x

        squared = square_array(jax_arr)
        np.testing.assert_array_equal(squared, jnp.array([1.0, 4.0, 9.0]))

        @jax.grad
        def sum_squares(x):
            return jnp.sum(x * x)

        grad_result = sum_squares(jax_arr)
        np.testing.assert_array_equal(grad_result, jnp.array([2.0, 4.0, 6.0]))

    def test_array_with_physical_quantities(self):
        position = SimpleArray(np.array([1.0, 2.0, 3.0])) * meter
        time_vals = SimpleArray(np.array([1.0, 2.0, 3.0])) * second

        velocity = position / time_vals
        expected_unit = meter / second
        assert velocity.unit.dim == expected_unit.dim

        voltage = SimpleArray(np.array([1.0, 2.0])) * mV
        current = SimpleArray(np.array([10.0, 20.0])) * nS
        resistance = voltage * current
        assert resistance.unit.dim == u.mA.dim

    def test_array_error_handling(self):
        arr = SimpleArray(np.array([1.0, 2.0, 3.0]))

        with pytest.raises(TypeError):
            arr + "string"

        voltage = SimpleArray(np.array([1.0])) * mV
        time_val = SimpleArray(np.array([1.0])) * ms

        with pytest.raises(u.UnitMismatchError):
            voltage + time_val

    def test_array_special_methods(self):
        arr = SimpleArray(np.array([1, 2, 3]))

        assert len(arr) == 3
        values = [x for x in arr]
        assert values == [1, 2, 3]

        non_empty_arr = SimpleArray(np.array([1]))
        assert bool(non_empty_arr)

        scalar_arr = SimpleArray(5)
        assert isinstance(hash(scalar_arr), int)

    def test_array_numpy_compatibility(self):
        arr = SimpleArray(np.array([1.0, 2.0, 3.0]))

        numpy_result = arr.to_numpy()
        assert isinstance(numpy_result, np.ndarray)
        np.testing.assert_array_equal(numpy_result, np.array([1.0, 2.0, 3.0]))

        numpy_converted = np.array(arr)
        np.testing.assert_array_equal(numpy_converted, np.array([1.0, 2.0, 3.0]))

        sin_result = np.sin(arr)
        np.testing.assert_array_almost_equal(sin_result, np.sin(np.array([1.0, 2.0, 3.0])))

    def test_array_pytorch_style_methods(self):
        arr = SimpleArray(np.array([1.0, 2.0, 3.0]))

        unsqueezed = arr.unsqueeze(0)
        assert unsqueezed.shape == (1, 3)

        clamped = arr.clamp(min_data=1.5, max_data=2.5)
        np.testing.assert_array_equal(clamped, np.array([1.5, 2.0, 2.5]))

        cloned = arr.clone()
        np.testing.assert_array_equal(cloned, arr.data)
        assert cloned is not arr.data

    def test_array_advanced_operations(self):
        arr = SimpleArray(np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(arr @ arr, np.array([[7, 10], [15, 22]]))

        vec1 = SimpleArray(np.array([1, 2, 3]))
        vec2 = SimpleArray(np.array([4, 5, 6]))
        assert float(vec1.dot(vec2)) == 32

        np.testing.assert_array_equal(vec1.cumsum(), np.array([1, 3, 6]))
        np.testing.assert_array_equal(vec1.cumprod(), np.array([1, 2, 6]))


# --- Docstring example tests ---


def test_docstring_example_custom_array_class():
    """Verify basic CustomArray usage described in the class docstring."""
    # -- Subclass with a plain ``data`` attribute (standalone, no brainstate) --
    arr = SimpleArray(np.array([1.0, 2.0, 3.0]))

    # Properties
    assert arr.shape == (3,)
    assert arr.ndim == 1
    assert arr.size == 3
    assert arr.dtype == np.float64

    # Arithmetic
    np.testing.assert_array_equal(arr + 10, np.array([11.0, 12.0, 13.0]))
    np.testing.assert_array_equal(arr ** 2, np.array([1.0, 4.0, 9.0]))
    np.testing.assert_array_equal(arr * 2, np.array([2.0, 4.0, 6.0]))

    # Statistical operations
    arr5 = SimpleArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert float(arr5.mean()) == 3.0
    assert float(arr5.sum()) == 15.0
    assert abs(float(arr5.std()) - np.std([1, 2, 3, 4, 5])) < 1e-6

    # Array manipulation
    matrix = SimpleArray(np.array([[1, 2, 3], [4, 5, 6]]))
    assert matrix.T.shape == (3, 2)
    np.testing.assert_array_equal(matrix.reshape(6), np.array([1, 2, 3, 4, 5, 6]))
    np.testing.assert_array_equal(matrix.flatten(), np.array([1, 2, 3, 4, 5, 6]))

    # Conversion methods
    numpy_arr = arr.to_numpy()
    assert isinstance(numpy_arr, np.ndarray)
    np.testing.assert_array_equal(numpy_arr, np.array([1.0, 2.0, 3.0]))

    # JAX compatibility
    jax_arr = SimpleArray(jnp.array([1.0, 2.0, 3.0]))

    @jax.jit
    def square(x):
        return x * x

    result = square(jax_arr)
    np.testing.assert_array_equal(result, jnp.array([1.0, 4.0, 9.0]))
