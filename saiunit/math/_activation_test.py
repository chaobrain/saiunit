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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

import saiunit as u
import saiunit.math as um
from saiunit import meter, second, UNITLESS
from saiunit._base import Quantity



class Array(u.CustomArray):
    def __init__(self, value):
        self.value = value



class TestActivationFunctions(parameterized.TestCase):
    """Test suite for activation functions in saiunit.math._activation module."""

    def setUp(self):
        """Set up test data for activation function tests."""
        # Standard test input arrays
        self.test_array = jnp.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
        self.positive_array = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
        self.negative_array = jnp.array([-5.0, -2.0, -1.0, -0.5, -0.1])

        # Arrays with units for testing unit handling
        self.dimensionless_quantity = self.test_array * UNITLESS
        self.meter_quantity = self.test_array * meter
        self.time_quantity = self.test_array * second

    def test_relu_basic_functionality(self):
        """Test ReLU activation function basic functionality."""
        # Test with JAX array
        result = um.relu(self.test_array)
        expected = jnp.maximum(self.test_array, 0)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

        # Test with dimensionless quantity - should preserve units
        result_quantity = um.relu(self.dimensionless_quantity)
        assert isinstance(result_quantity, Quantity)
        np.testing.assert_allclose(result_quantity.mantissa, expected, rtol=1e-6)

        # Test with units - should preserve units
        result_meter = um.relu(self.meter_quantity)
        assert isinstance(result_meter, Quantity)
        np.testing.assert_allclose(result_meter.mantissa, expected, rtol=1e-6)
        assert result_meter.unit == meter

    def test_relu6_basic_functionality(self):
        """Test ReLU6 activation function basic functionality."""
        # Test with JAX array
        result = um.relu6(self.test_array)
        expected = jnp.minimum(jnp.maximum(self.test_array, 0), 6)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

        # Test with dimensionless quantity
        result_quantity = um.relu6(self.dimensionless_quantity)
        np.testing.assert_allclose(result_quantity, expected, rtol=1e-6)

        # Test that it requires unitless input
        with pytest.raises(Exception):
            um.relu6(self.meter_quantity)

    def test_sigmoid_basic_functionality(self):
        """Test Sigmoid activation function basic functionality."""
        # Test with JAX array
        result = um.sigmoid(self.test_array)
        expected = 1 / (1 + jnp.exp(-self.test_array))
        np.testing.assert_allclose(result, expected, rtol=1e-6)

        # Test with dimensionless quantity
        result_quantity = um.sigmoid(self.dimensionless_quantity)
        np.testing.assert_allclose(result_quantity, expected, rtol=1e-6)

        # Test that it requires unitless input
        with pytest.raises(Exception):
            um.sigmoid(self.meter_quantity)

        # Test sigmoid output is bounded between 0 and 1
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 1)

    def test_softplus_basic_functionality(self):
        """Test Softplus activation function basic functionality."""
        # Test with JAX array
        result = um.softplus(self.test_array)
        expected = jnp.log(1 + jnp.exp(self.test_array))
        np.testing.assert_allclose(result, expected, rtol=1e-6)

        # Test with dimensionless quantity
        result_quantity = um.softplus(self.dimensionless_quantity)
        np.testing.assert_allclose(result_quantity, expected, rtol=1e-6)

        # Test that it requires unitless input
        with pytest.raises(Exception):
            um.softplus(self.meter_quantity)

        # Test softplus is always positive
        assert jnp.all(result >= 0)

    def test_sparse_plus_basic_functionality(self):
        """Test Sparse Plus activation function basic functionality."""
        # Test with JAX array
        result = um.sparse_plus(self.test_array)

        # Verify piecewise definition
        for i, x in enumerate(self.test_array):
            if x <= -1:
                assert result[i] == 0
            elif x >= 1:
                assert abs(result[i] - x) < 1e-6
            else:  # -1 < x < 1
                expected_val = 0.25 * (x + 1) ** 2
                assert abs(result[i] - expected_val) < 1e-6

        # Test with dimensionless quantity
        result_quantity = um.sparse_plus(self.dimensionless_quantity)
        np.testing.assert_allclose(result_quantity, result, rtol=1e-6)

        # Test that it requires unitless input
        with pytest.raises(Exception):
            um.sparse_plus(self.meter_quantity)

    def test_sparse_sigmoid_basic_functionality(self):
        """Test Sparse Sigmoid activation function basic functionality."""
        # Test with JAX array
        result = um.sparse_sigmoid(self.test_array)

        # Verify piecewise definition
        for i, x in enumerate(self.test_array):
            if x <= -1:
                assert result[i] == 0
            elif x >= 1:
                assert result[i] == 1
            else:  # -1 < x < 1
                expected_val = 0.5 * (x + 1)
                assert abs(result[i] - expected_val) < 1e-6

        # Test output is bounded between 0 and 1
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 1)

    def test_soft_sign_basic_functionality(self):
        """Test Soft-sign activation function basic functionality."""
        # Test with JAX array
        result = um.soft_sign(self.test_array)
        expected = self.test_array / (jnp.abs(self.test_array) + 1)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

        # Test output is bounded between -1 and 1
        assert jnp.all(result >= -1)
        assert jnp.all(result <= 1)

        # Test with dimensionless quantity
        result_quantity = um.soft_sign(self.dimensionless_quantity)
        np.testing.assert_allclose(result_quantity, expected, rtol=1e-6)

    def test_silu_swish_equivalence(self):
        """Test that SiLU and Swish are equivalent functions."""
        # Test with JAX array
        silu_result = um.silu(self.test_array)
        swish_result = um.swish(self.test_array)

        np.testing.assert_allclose(silu_result, swish_result, rtol=1e-6)

        # Verify mathematical definition: x * sigmoid(x)
        expected = self.test_array * (1 / (1 + jnp.exp(-self.test_array)))
        np.testing.assert_allclose(silu_result, expected, rtol=1e-6)

        # Test with dimensionless quantity
        silu_quantity = um.silu(self.dimensionless_quantity)
        swish_quantity = um.swish(self.dimensionless_quantity)
        np.testing.assert_allclose(silu_quantity, swish_quantity, rtol=1e-6)

    def test_log_sigmoid_basic_functionality(self):
        """Test Log-sigmoid activation function basic functionality."""
        # Test with JAX array
        result = um.log_sigmoid(self.test_array)
        expected = jnp.log(1 / (1 + jnp.exp(-self.test_array)))
        np.testing.assert_allclose(result, expected, rtol=1e-6)

        # Alternative formulation: -log(1 + exp(-x))
        expected_alt = -jnp.log(1 + jnp.exp(-self.test_array))
        np.testing.assert_allclose(result, expected_alt, rtol=1e-6)

        # Test output is always negative or zero
        assert jnp.all(result <= 0)

    def test_hard_sigmoid_basic_functionality(self):
        """Test Hard Sigmoid activation function basic functionality."""
        # Test with JAX array
        result = um.hard_sigmoid(self.test_array)

        # Hard sigmoid is relu6(x + 3) / 6
        expected = jnp.minimum(jnp.maximum(self.test_array + 3, 0), 6) / 6
        np.testing.assert_allclose(result, expected, rtol=1e-6)

        # Test output is bounded between 0 and 1
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 1)

    def test_hard_silu_hard_swish_equivalence(self):
        """Test that Hard SiLU and Hard Swish are equivalent."""
        # Test with JAX array
        hard_silu_result = um.hard_silu(self.test_array)
        hard_swish_result = um.hard_swish(self.test_array)

        np.testing.assert_allclose(hard_silu_result, hard_swish_result, rtol=1e-6)

        # Verify mathematical definition: x * hard_sigmoid(x)
        hard_sigmoid_vals = jnp.minimum(jnp.maximum(self.test_array + 3, 0), 6) / 6
        expected = self.test_array * hard_sigmoid_vals
        np.testing.assert_allclose(hard_silu_result, expected, rtol=1e-6)

    def test_hard_tanh_basic_functionality(self):
        """Test Hard Tanh activation function basic functionality."""
        # Test with JAX array
        result = um.hard_tanh(self.test_array)

        # Verify piecewise definition
        for i, x in enumerate(self.test_array):
            if x < -1:
                assert result[i] == -1
            elif x > 1:
                assert result[i] == 1
            else:  # -1 <= x <= 1
                assert abs(result[i] - x) < 1e-6

        # Test output is bounded between -1 and 1
        assert jnp.all(result >= -1)
        assert jnp.all(result <= 1)

    def test_elu_functionality(self):
        """Test ELU activation function functionality."""
        # Test with default alpha (1.0)
        result = um.elu(self.test_array)

        for i, x in enumerate(self.test_array):
            if x > 0:
                assert abs(result[i] - x) < 1e-6
            else:
                expected_val = 1.0 * (jnp.exp(x) - 1)
                assert abs(result[i] - expected_val) < 1e-6

        # Test with custom alpha
        custom_alpha = 2.0
        result_custom = um.elu(self.test_array, alpha=custom_alpha)

        for i, x in enumerate(self.test_array):
            if x > 0:
                assert abs(result_custom[i] - x) < 1e-6
            else:
                expected_val = custom_alpha * (jnp.exp(x) - 1)
                assert abs(result_custom[i] - expected_val) < 1e-6

    def test_celu_functionality(self):
        """Test CELU activation function functionality."""
        # Test with default alpha (1.0)
        result = um.celu(self.test_array)

        for i, x in enumerate(self.test_array):
            if x > 0:
                assert abs(result[i] - x) < 1e-6
            else:
                expected_val = 1.0 * (jnp.exp(x / 1.0) - 1)
                assert abs(result[i] - expected_val) < 1e-6

        # Test with custom alpha
        custom_alpha = 2.0
        result_custom = um.celu(self.test_array, alpha=custom_alpha)

        for i, x in enumerate(self.test_array):
            if x > 0:
                assert abs(result_custom[i] - x) < 1e-6
            else:
                expected_val = custom_alpha * (jnp.exp(x / custom_alpha) - 1)
                assert abs(result_custom[i] - expected_val) < 1e-6

    def test_selu_basic_functionality(self):
        """Test SELU activation function basic functionality."""
        # Test with JAX array
        result = um.selu(self.test_array)

        # SELU constants
        lambda_val = 1.0507009873554804934193349852946
        alpha_val = 1.6732632423543772848170429916717

        for i, x in enumerate(self.test_array):
            if x > 0:
                expected_val = lambda_val * x
                assert abs(result[i] - expected_val) < 1e-6
            else:
                expected_val = lambda_val * alpha_val * (jnp.exp(x) - 1)
                assert abs(result[i] - expected_val) < 1e-6

    def test_gelu_functionality(self):
        """Test GELU activation function functionality."""
        # Test with approximate=True (default)
        result_approx = um.gelu(self.test_array, approximate=True)

        # Test with approximate=False
        result_exact = um.gelu(self.test_array, approximate=False)

        # Results should be close but not identical
        np.testing.assert_allclose(result_approx, result_exact, rtol=1e-1, atol=1e-1)

        # Test that both versions are smooth and reasonable
        assert jnp.all(jnp.isfinite(result_approx))
        assert jnp.all(jnp.isfinite(result_exact))

    def test_glu_functionality(self):
        """Test GLU activation function functionality."""
        # Create input with even number of elements along last axis
        test_input = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8.]])

        # Test with default axis (-1)
        result = um.glu(test_input)

        # GLU splits input in half and applies: first_half * sigmoid(second_half)
        first_half = test_input[:, :2]  # [1,2] and [5,6]
        second_half = test_input[:, 2:]  # [3,4] and [7,8]
        expected = first_half * (1 / (1 + jnp.exp(-second_half)))

        np.testing.assert_allclose(result, expected, rtol=1e-1, atol=1e-1)

        # Test with custom axis
        test_input_3d = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8.]]])
        result_axis0 = um.glu(test_input_3d, axis=0)

        # Should split along axis 0
        first_half_axis0 = test_input_3d[0:1]
        second_half_axis0 = test_input_3d[1:2]
        expected_axis0 = first_half_axis0 * (1 / (1 + jnp.exp(-second_half_axis0)))

        np.testing.assert_allclose(result_axis0, expected_axis0, rtol=1e-1)

    def test_squareplus_functionality(self):
        """Test Squareplus activation function functionality."""
        # Test with default b=4
        result = um.squareplus(self.test_array)
        expected = (self.test_array + jnp.sqrt(self.test_array ** 2 + 4)) / 2
        np.testing.assert_allclose(result, expected, rtol=1e-6)

        # Test with custom b
        custom_b = 2.0
        result_custom = um.squareplus(self.test_array, b=custom_b)
        expected_custom = (self.test_array + jnp.sqrt(self.test_array ** 2 + custom_b)) / 2
        np.testing.assert_allclose(result_custom, expected_custom, rtol=1e-6)

        # Test that output is always >= 0
        assert jnp.all(result >= 0)

    def test_mish_basic_functionality(self):
        """Test Mish activation function basic functionality."""
        # Test with JAX array
        result = um.mish(self.test_array)

        # Mish: x * tanh(softplus(x))
        softplus_vals = jnp.log(1 + jnp.exp(self.test_array))
        expected = self.test_array * jnp.tanh(softplus_vals)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

        # Test with dimensionless quantity
        result_quantity = um.mish(self.dimensionless_quantity)
        np.testing.assert_allclose(result_quantity, expected, rtol=1e-6)

    @parameterized.named_parameters(
        ('relu', 'relu'),
        ('relu6', 'relu6'),
        ('sigmoid', 'sigmoid'),
        ('softplus', 'softplus'),
        ('sparse_plus', 'sparse_plus'),
        ('sparse_sigmoid', 'sparse_sigmoid'),
        ('soft_sign', 'soft_sign'),
        ('silu', 'silu'),
        ('swish', 'swish'),
        ('log_sigmoid', 'log_sigmoid'),
        ('hard_sigmoid', 'hard_sigmoid'),
        ('hard_silu', 'hard_silu'),
        ('hard_tanh', 'hard_tanh'),
        ('elu', 'elu'),
        ('celu', 'celu'),
        ('selu', 'selu'),
        ('gelu', 'gelu'),
        ('squareplus', 'squareplus'),
        ('mish', 'mish'),
    )
    def test_activation_function_shapes(self, func_name):
        """Test that activation functions preserve input shapes."""
        func = getattr(um, func_name)

        # Test with 1D array
        input_1d = jnp.array([1.0, 2.0, 3.0])
        result_1d = func(input_1d)
        assert result_1d.shape == input_1d.shape

        # Test with 2D array
        input_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result_2d = func(input_2d)
        assert result_2d.shape == input_2d.shape

        # Test with 3D array
        input_3d = jnp.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        result_3d = func(input_3d)
        assert result_3d.shape == input_3d.shape

    @parameterized.named_parameters(
        ('relu', 'relu'),
        ('sigmoid', 'sigmoid'),
        ('softplus', 'softplus'),
        ('sparse_plus', 'sparse_plus'),
        ('sparse_sigmoid', 'sparse_sigmoid'),
        ('soft_sign', 'soft_sign'),
        ('silu', 'silu'),
        ('swish', 'swish'),
        ('log_sigmoid', 'log_sigmoid'),
        ('hard_sigmoid', 'hard_sigmoid'),
        ('hard_silu', 'hard_silu'),
        ('hard_tanh', 'hard_tanh'),
        ('elu', 'elu'),
        ('celu', 'celu'),
        ('selu', 'selu'),
        ('gelu', 'gelu'),
        ('squareplus', 'squareplus'),
        ('mish', 'mish'),
    )
    def test_activation_functions_finite_outputs(self, func_name):
        """Test that activation functions produce finite outputs for reasonable inputs."""
        func = getattr(um, func_name)

        # Test with various input ranges
        test_ranges = [
            jnp.linspace(-10, 10, 21),
            jnp.array([-100., -10, -1, 0, 1, 10, 100]),
            jnp.array([1e-8, 1e-4, 1e-2, 1e2, 1e4, 1e8])
        ]

        for test_range in test_ranges:
            try:
                result = func(test_range)
                assert jnp.all(jnp.isfinite(result)), f"{func_name} produced non-finite values for input {test_range}"
            except Exception as e:
                # Some functions might have specific requirements (e.g., unitless)
                if "unit" not in str(e).lower():
                    raise

    def test_special_cases(self):
        """Test activation functions with special input cases."""
        # Test with zero
        zero_input = jnp.array([0.0])

        # Functions that should return 0 for input 0
        zero_output_funcs = ['relu', 'soft_sign', 'silu', 'swish', 'log_sigmoid',
                             'hard_silu', 'hard_tanh', 'mish']

        # Functions that should return 0.5 for input 0
        half_output_funcs = ['sigmoid', 'hard_sigmoid']
        for func_name in half_output_funcs:
            func = getattr(um, func_name)
            result = func(zero_input)
            assert abs(result[0] - 0.5) < 1e-6

        # Test with very large positive values
        large_pos = jnp.array([100.0])
        bounded_funcs = ['sigmoid', 'sparse_sigmoid', 'hard_sigmoid']
        for func_name in bounded_funcs:
            func = getattr(um, func_name)
            result = func(large_pos)
            assert result[0] <= 1.0 + 1e-6

        # Test with very large negative values  
        large_neg = jnp.array([-100.0])
        for func_name in bounded_funcs:
            func = getattr(um, func_name)
            result = func(large_neg)
            assert result[0] >= -1e-6

    def test_differentiability(self):
        """Test that activation functions are differentiable."""
        test_input = jnp.array([1.0])

        # Test functions that should be differentiable everywhere
        differentiable_funcs = ['sigmoid', 'softplus', 'soft_sign', 'silu', 'swish',
                                'log_sigmoid', 'elu', 'celu', 'selu', 'gelu', 'mish']

        for func_name in differentiable_funcs:
            func = getattr(um, func_name)

            # Compute gradient
            def test_func(x):
                return jnp.sum(func(x))

            grad_func = jax.grad(test_func)
            gradient = grad_func(test_input)

            assert jnp.isfinite(gradient[0]), f"{func_name} gradient is not finite"

    def test_edge_cases_glu(self):
        """Test GLU with edge cases."""
        # Test with odd dimension (should raise error)
        odd_input = jnp.array([1, 2, 3])
        with pytest.raises(Exception):
            um.glu(odd_input)

        # Test with minimum valid input (2 elements)
        min_input = jnp.array([1.0, 2.0])
        result = um.glu(min_input)
        expected = jnp.array([1.0]) * (1 / (1 + jnp.exp(-2.0)))
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestActivationFunctionsWithArrayCustomArray(parameterized.TestCase):
    """Test suite for activation functions with Array that inherits from saiunit.CustomArray."""

    def setUp(self):
        """Set up test data for Array with CustomArray activation tests."""
        # Test data using Array with CustomArray
        self.array_test_data = Array(jnp.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]))
        self.array_positive = Array(jnp.array([0.1, 0.5, 1.0, 2.0, 5.0]))
        self.array_negative = Array(jnp.array([-5.0, -2.0, -1.0, -0.5, -0.1]))
        
        # 2D Array for shape testing
        self.array_2d = Array(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        
        # Arrays with units
        self.dimensionless_array = Array(jnp.array([-1.0, 0.0, 1.0])) * UNITLESS
        self.meter_array = Array(jnp.array([-1.0, 0.0, 1.0])) * meter
        
        # Complex test cases
        self.array_extreme = Array(jnp.array([-100.0, -10.0, 0.0, 10.0, 100.0]))
        self.array_small = Array(jnp.array([-1e-6, 0.0, 1e-6]))

    def test_relu_with_array_custom_array(self):
        """Test ReLU activation function with Array using CustomArray."""
        # Test basic ReLU functionality
        result = um.relu(self.array_test_data)
        expected = jnp.maximum(self.array_test_data.value, 0)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        
        # Test with 2D Array
        result_2d = um.relu(self.array_2d)
        expected_2d = jnp.maximum(self.array_2d.value, 0)
        np.testing.assert_allclose(result_2d, expected_2d, rtol=1e-6)
        
        # Test with dimensionless Array quantity
        result_dimensionless = um.relu(self.dimensionless_array)
        assert isinstance(result_dimensionless, Quantity)
        expected_dimensionless = jnp.maximum(self.dimensionless_array.mantissa, 0)
        np.testing.assert_allclose(result_dimensionless.mantissa, expected_dimensionless, rtol=1e-6)
        
        # Test with meter Array quantity (should preserve units)
        result_meter = um.relu(self.meter_array)
        assert isinstance(result_meter, Quantity)
        assert result_meter.unit == meter
        expected_meter = jnp.maximum(self.meter_array.mantissa, 0)
        np.testing.assert_allclose(result_meter.mantissa, expected_meter, rtol=1e-6)

    def test_sigmoid_with_array_custom_array(self):
        """Test Sigmoid activation function with Array using CustomArray."""
        # Test basic sigmoid functionality
        result = um.sigmoid(self.array_test_data)
        expected = 1 / (1 + jnp.exp(-self.array_test_data.value))
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        
        # Verify sigmoid properties
        assert jnp.all(result >= 0)
        assert jnp.all(result <= 1)
        
        # Test with dimensionless Array
        result_dimensionless = um.sigmoid(self.dimensionless_array)
        expected_dimensionless = 1 / (1 + jnp.exp(-self.dimensionless_array.mantissa))
        np.testing.assert_allclose(result_dimensionless, expected_dimensionless, rtol=1e-6)
        
        # Test that it requires unitless input for Array with units
        with pytest.raises(Exception):
            um.sigmoid(self.meter_array)

    def test_softplus_with_array_custom_array(self):
        """Test Softplus activation function with Array using CustomArray."""
        # Test basic softplus functionality
        result = um.softplus(self.array_test_data)
        expected = jnp.log(1 + jnp.exp(self.array_test_data.value))
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        
        # Verify softplus is always positive
        assert jnp.all(result >= 0)
        
        # Test with extreme values to check numerical stability
        result_extreme = um.softplus(self.array_extreme)
        assert jnp.all(jnp.isfinite(result_extreme))
        
        # Test with dimensionless Array
        result_dimensionless = um.softplus(self.dimensionless_array)
        expected_dimensionless = jnp.log(1 + jnp.exp(self.dimensionless_array.mantissa))
        np.testing.assert_allclose(result_dimensionless, expected_dimensionless, rtol=1e-6)

    def test_silu_swish_equivalence_array(self):
        """Test SiLU and Swish equivalence with Array using CustomArray."""
        # Test equivalence
        silu_result = um.silu(self.array_test_data)
        swish_result = um.swish(self.array_test_data)
        np.testing.assert_allclose(silu_result, swish_result, rtol=1e-6)
        
        # Verify mathematical definition: x * sigmoid(x)
        sigmoid_vals = 1 / (1 + jnp.exp(-self.array_test_data.value))
        expected = self.array_test_data.value * sigmoid_vals
        np.testing.assert_allclose(silu_result, expected, rtol=1e-6)
        
        # Test with dimensionless Array quantity
        silu_dimensionless = um.silu(self.dimensionless_array)
        swish_dimensionless = um.swish(self.dimensionless_array)
        np.testing.assert_allclose(silu_dimensionless, swish_dimensionless, rtol=1e-6)

    def test_sparse_plus_with_array_custom_array(self):
        """Test Sparse Plus activation function with Array using CustomArray."""
        result = um.sparse_plus(self.array_test_data)
        
        # Verify piecewise definition for Array values
        for i, x in enumerate(self.array_test_data.value):
            if x <= -1:
                assert result[i] == 0
            elif x >= 1:
                assert abs(result[i] - x) < 1e-6
            else:  # -1 < x < 1
                expected_val = 0.25 * (x + 1) ** 2
                assert abs(result[i] - expected_val) < 1e-6
        
        # Test with dimensionless Array
        result_dimensionless = um.sparse_plus(self.dimensionless_array)
        np.testing.assert_allclose(result_dimensionless, um.sparse_plus(self.dimensionless_array.mantissa), rtol=1e-6)

    def test_array_shape_preservation(self):
        """Test that activation functions preserve Array shapes."""
        activation_funcs = ['relu', 'sigmoid', 'softplus', 'silu', 'swish', 'soft_sign']
        
        for func_name in activation_funcs:
            func = getattr(um, func_name)
            
            # Test with 1D Array
            result_1d = func(self.array_test_data)
            assert result_1d.shape == self.array_test_data.shape
            
            # Test with 2D Array
            result_2d = func(self.array_2d)
            assert result_2d.shape == self.array_2d.shape

    def test_array_mathematical_properties(self):
        """Test mathematical properties of activation functions with Array."""
        # Test ReLU non-negative output
        relu_result = um.relu(self.array_test_data)
        assert jnp.all(relu_result >= 0)
        
        # Test sigmoid bounded output [0, 1]
        sigmoid_result = um.sigmoid(self.array_test_data)
        assert jnp.all(sigmoid_result >= 0)
        assert jnp.all(sigmoid_result <= 1)
        
        # Test soft_sign bounded output [-1, 1]
        soft_sign_result = um.soft_sign(self.array_test_data)
        assert jnp.all(soft_sign_result >= -1)
        assert jnp.all(soft_sign_result <= 1)
        
        # Test log_sigmoid non-positive output
        log_sigmoid_result = um.log_sigmoid(self.array_test_data)
        assert jnp.all(log_sigmoid_result <= 0)

    def test_array_with_jax_transformations(self):
        """Test activation functions with Array under JAX transformations."""
        # Test with jit compilation
        @jax.jit
        def relu_jitted(x):
            return um.relu(x)
        
        jit_result = relu_jitted(self.array_test_data)
        direct_result = um.relu(self.array_test_data)
        np.testing.assert_allclose(jit_result, direct_result, rtol=1e-6)
        
        # Test with grad
        @jax.grad
        def sigmoid_sum(x):
            return jnp.sum(um.sigmoid(x))
        
        grad_result = sigmoid_sum(self.array_test_data.value)
        assert jnp.all(jnp.isfinite(grad_result))

    def test_elu_with_array_custom_array(self):
        """Test ELU activation function with Array using CustomArray."""
        # Test with default alpha
        result = um.elu(self.array_test_data)
        
        for i, x in enumerate(self.array_test_data.value):
            if x > 0:
                assert abs(result[i] - x) < 1e-6
            else:
                expected_val = 1.0 * (jnp.exp(x) - 1)
                assert abs(result[i] - expected_val) < 1e-6
        
        # Test with custom alpha
        custom_alpha = 2.0
        result_custom = um.elu(self.array_test_data, alpha=custom_alpha)
        
        for i, x in enumerate(self.array_test_data.value):
            if x > 0:
                assert abs(result_custom[i] - x) < 1e-6
            else:
                expected_val = custom_alpha * (jnp.exp(x) - 1)
                assert abs(result_custom[i] - expected_val) < 1e-6

    def test_gelu_with_array_custom_array(self):
        """Test GELU activation function with Array using CustomArray."""
        # Test with approximate=True
        result_approx = um.gelu(self.array_test_data, approximate=True)
        
        # Test with approximate=False
        result_exact = um.gelu(self.array_test_data, approximate=False)
        
        # Results should be close but not identical
        np.testing.assert_allclose(result_approx, result_exact, rtol=1e-1, atol=1e-1)
        
        # Both should be finite
        assert jnp.all(jnp.isfinite(result_approx))
        assert jnp.all(jnp.isfinite(result_exact))

    def test_hard_activations_with_array(self):
        """Test hard activation functions with Array using CustomArray."""
        # Test hard_sigmoid
        hard_sigmoid_result = um.hard_sigmoid(self.array_test_data)
        assert jnp.all(hard_sigmoid_result >= 0)
        assert jnp.all(hard_sigmoid_result <= 1)
        
        # Test hard_tanh
        hard_tanh_result = um.hard_tanh(self.array_test_data)
        assert jnp.all(hard_tanh_result >= -1)
        assert jnp.all(hard_tanh_result <= 1)
        
        # Test hard_silu and hard_swish equivalence
        hard_silu_result = um.hard_silu(self.array_test_data)
        hard_swish_result = um.hard_swish(self.array_test_data)
        np.testing.assert_allclose(hard_silu_result, hard_swish_result, rtol=1e-6)

    def test_mish_with_array_custom_array(self):
        """Test Mish activation function with Array using CustomArray."""
        result = um.mish(self.array_test_data)
        
        # Verify mathematical definition: x * tanh(softplus(x))
        softplus_vals = jnp.log(1 + jnp.exp(self.array_test_data.value))
        expected = self.array_test_data.value * jnp.tanh(softplus_vals)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        
        # Test with dimensionless Array
        result_dimensionless = um.mish(self.dimensionless_array)
        softplus_dimensionless = jnp.log(1 + jnp.exp(self.dimensionless_array.mantissa))
        expected_dimensionless = self.dimensionless_array.mantissa * jnp.tanh(softplus_dimensionless)
        np.testing.assert_allclose(result_dimensionless, expected_dimensionless, rtol=1e-6)

    def test_squareplus_with_array_custom_array(self):
        """Test Squareplus activation function with Array using CustomArray."""
        # Test with default b=4
        result = um.squareplus(self.array_test_data)
        expected = (self.array_test_data.value + jnp.sqrt(self.array_test_data.value ** 2 + 4)) / 2
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        
        # Test that output is always >= 0
        assert jnp.all(result >= 0)
        
        # Test with custom b
        custom_b = 2.0
        result_custom = um.squareplus(self.array_test_data, b=custom_b)
        expected_custom = (self.array_test_data.value + jnp.sqrt(self.array_test_data.value ** 2 + custom_b)) / 2
        np.testing.assert_allclose(result_custom, expected_custom, rtol=1e-6)

    def test_glu_with_array_custom_array(self):
        """Test GLU activation function with Array using CustomArray."""
        # Create Array with even number of elements
        test_array_even = Array(jnp.array([[1, 2, 3, 4], [5, 6, 7, 8.]]))
        
        # Test GLU
        result = um.glu(test_array_even)
        
        # Verify GLU computation: first_half * sigmoid(second_half)
        first_half = test_array_even.value[:, :2]
        second_half = test_array_even.value[:, 2:]
        expected = first_half * (1 / (1 + jnp.exp(-second_half)))
        
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        
        # Test with odd dimension (should raise error)
        odd_array = Array(jnp.array([1, 2, 3]))
        with pytest.raises(Exception):
            um.glu(odd_array)

    def test_array_numerical_stability(self):
        """Test numerical stability of activation functions with Array."""
        # Test with very large values
        large_array = Array(jnp.array([-1000.0, 1000.0]))
        
        # These functions should handle large values gracefully
        stable_funcs = ['sigmoid', 'hard_sigmoid', 'soft_sign', 'hard_tanh']
        
        for func_name in stable_funcs:
            func = getattr(um, func_name)
            result = func(large_array)
            assert jnp.all(jnp.isfinite(result)), f"{func_name} produced non-finite values"
        
        # Test with very small values
        small_array = Array(jnp.array([-1e-10, 0.0, 1e-10]))
        
        for func_name in stable_funcs:
            func = getattr(um, func_name)
            result = func(small_array)
            assert jnp.all(jnp.isfinite(result)), f"{func_name} produced non-finite values for small inputs"

    def test_array_gradient_computation(self):
        """Test gradient computation for activation functions with Array."""
        # Test functions that should be differentiable
        differentiable_funcs = ['sigmoid', 'softplus', 'silu', 'swish', 'soft_sign', 'mish']
        
        for func_name in differentiable_funcs:
            func = getattr(um, func_name)
            
            # Define a function that sums the activation output
            def sum_activation(x):
                arr = Array(x)
                return jnp.sum(func(arr))
            
            # Compute gradient
            grad_func = jax.grad(sum_activation)
            gradient = grad_func(jnp.array([1.0, 2.0, 3.0]))
            
            assert jnp.all(jnp.isfinite(gradient)), f"{func_name} gradient contains non-finite values"
            assert gradient.shape == (3,), f"{func_name} gradient shape mismatch"

    def test_array_unit_handling_comprehensive(self):
        """Comprehensive test of unit handling with Array using CustomArray."""
        # Functions that should work with any units (preserve units)
        unit_preserving_funcs = ['relu']
        
        for func_name in unit_preserving_funcs:
            func = getattr(um, func_name)
            
            # Test with meter Array
            result_meter = func(self.meter_array)
            assert isinstance(result_meter, Quantity)
            assert result_meter.unit == meter
            
            # Test with dimensionless Array
            result_dimensionless = func(self.dimensionless_array)
            assert isinstance(result_dimensionless, Quantity)
            assert result_dimensionless.unit == UNITLESS
        
        # Functions that require unitless input
        unitless_required_funcs = ['sigmoid', 'softplus', 'silu', 'swish']
        
        for func_name in unitless_required_funcs:
            func = getattr(um, func_name)
            
            # Should work with dimensionless
            result_dimensionless = func(self.dimensionless_array)
            # Result should be unitless or the same as input
            
            # Should raise exception with units
            with pytest.raises(Exception):
                func(self.meter_array)

    def test_array_custom_array_inheritance(self):
        """Test that Array properly inherits from CustomArray and works with activations."""
        # Verify Array is instance of CustomArray
        assert isinstance(self.array_test_data, u.CustomArray)
        
        # Test that CustomArray methods work
        assert hasattr(self.array_test_data, 'value')
        assert hasattr(self.array_test_data, 'shape')
        assert hasattr(self.array_test_data, 'dtype')
        
        # Test activation functions work with Array methods
        relu_result = um.relu(self.array_test_data)
        
        # Verify we can still access CustomArray properties after activation
        assert self.array_test_data.shape == (7,)
        assert self.array_test_data.dtype == jnp.float32
        
        # Test that Array can be used in arithmetic operations
        doubled_array = self.array_test_data * 2
        relu_doubled = um.relu(doubled_array)
        
        assert relu_doubled.shape == self.array_test_data.shape
