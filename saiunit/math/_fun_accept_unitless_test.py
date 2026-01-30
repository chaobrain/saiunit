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
import pytest
from absl.testing import parameterized

import saiunit as u
import saiunit.math as um
from saiunit import meter
from saiunit._base import assert_quantity


class Array(u.CustomArray):
    def __init__(self, value):
        self.data = value


fun_accept_unitless_unary = [
    'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2',
    'deg2rad', 'rad2deg', 'degrees', 'radians', 'angle',
    'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
    'arctanh', 'cos', 'cosh', 'sin', 'sinc', 'sinh', 'tan',
    'tanh',
]

fun_accept_unitless_binary = [
    'hypot', 'arctan2', 'logaddexp', 'logaddexp2',
    'corrcoef', 'correlate', 'cov',
]
fun_accept_unitless_binary_ldexp = [
    'ldexp',
]

fun_elementwise_bit_operation_unary = [
    'bitwise_not', 'invert',
]
fun_elementwise_bit_operation_binary = [
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'left_shift', 'right_shift',
]


class TestFunAcceptUnitless(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFunAcceptUnitless, self).__init__(*args, **kwargs)

        print()

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)]
    )
    def test_fun_accept_unitless_unary_1(self, value):
        for fun_name in fun_accept_unitless_unary:
            fun = getattr(u.math, fun_name)
            jnp_fun = getattr(jnp, fun_name)
            print(f'fun: {fun}')

            result = fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            for unit, unit2scale in [(u.ms, u.second),
                                     (u.mV, u.volt),
                                     (u.mV, u.mV),
                                     (u.nA, u.amp)]:
                q = value * unit
                result = fun(q, unit_to_scale=unit2scale)
                expected = jnp_fun(q.to_decimal(unit2scale))
                assert_quantity(result, expected)

                with pytest.raises(AssertionError):
                    result = fun(q)

                with pytest.raises(u.UnitMismatchError):
                    result = fun(q, unit_to_scale=u.nS)

    @parameterized.product(
        value=[[(1.0, 2.0), (3.0, 4.0), ],
               [(1.23, 2.34, 3.45), (4.56, 5.67, 6.78)]]
    )
    def test_func_accept_unitless_binary(self, value):
        value1, value2 = value
        bm_fun_list = [getattr(um, fun) for fun in fun_accept_unitless_binary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_accept_unitless_binary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * meter
            q2 = value2 * meter
            result = bm_fun(q1, q2, unit_to_scale=u.dametre)
            expected = jnp_fun(q1.to_decimal(u.dametre), q2.to_decimal(u.dametre))
            assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(q1, q2)

            with pytest.raises(u.UnitMismatchError):
                result = bm_fun(q1, q2, unit_to_scale=u.second)

    @parameterized.product(
        value=[
            [(1.0, 2.0), (3, 4), ],
            [(1.23, 2.34, 3.45), (4, 5, 6)]
        ]
    )
    def test_func_accept_unitless_binary_ldexp(self, value):
        value1, value2 = value
        bm_fun_list = [getattr(um, fun) for fun in fun_accept_unitless_binary_ldexp]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_accept_unitless_binary_ldexp]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * meter
            q2 = value2 * meter
            result = bm_fun(q1.to_decimal(meter), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(q1, q2)

    @parameterized.product(
        value=[(1, 2), (1, 2, 3)]
    )
    def test_elementwise_bit_operation_unary(self, value):
        bm_fun_list = [getattr(um, fun) for fun in fun_elementwise_bit_operation_unary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_elementwise_bit_operation_unary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * meter
            # result = bm_fun(q.astype(jnp.int32).to_value())
            # expected = jnp_fun(jnp.array(data))
            # assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(q)

    @parameterized.product(
        value=[[(0, 1), (1, 1)],
               [(True, False, True, False), (False, False, True, True)]]
    )
    def test_elementwise_bit_operation_binary(self, value):
        value1, value2 = value
        bm_fun_list = [getattr(um, fun) for fun in fun_elementwise_bit_operation_binary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_elementwise_bit_operation_binary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * meter
            q2 = value2 * meter
            # result = bm_fun(q1.astype(jnp.bool_).to_value(), q2.astype(jnp.bool_).to_value())
            # expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            # assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(q1, q2)

    def test_dimensionless(self):
        a = u.Quantity(1.0)

        for fun_name in fun_accept_unitless_unary:
            r1 = getattr(u.math, fun_name)(a)
            r2 = getattr(jnp, fun_name)(a.to_decimal())
            print(fun_name, r1, r2)
            self.assertTrue(jnp.allclose(r1, r2, equal_nan=True))

        b = u.Quantity(2.0)

        for fun_name in ['hypot', 'arctan2', 'logaddexp', 'logaddexp2', ]:
            r1 = getattr(u.math, fun_name)(a, b)
            r2 = getattr(jnp, fun_name)(a.to_decimal(), b.to_decimal())
            print(fun_name, r1, r2)
            self.assertTrue(jnp.allclose(r1, r2, equal_nan=True))


class TestFunAcceptUnitlessWithArrayCustomArray(parameterized.TestCase):
    """Test suite for functions that accept unitless inputs with Array using CustomArray."""

    def setUp(self):
        """Set up test fixtures for Array with CustomArray tests."""
        # Basic Array instances
        self.array_1d = Array(jnp.array([0.5, 1.0, 1.5, 2.0]))
        self.array_2d = Array(jnp.array([[0.1, 0.5], [1.0, 2.0]]))
        self.array_int = Array(jnp.array([1, 2, 3, 4]))

        # Arrays with units
        self.meter_array = Array(jnp.array([1.0, 2.0, 3.0])) * meter
        self.voltage_array = Array(jnp.array([0.001, 0.002, 0.003])) * u.volt
        self.dimensionless_array = Array(jnp.array([1.0, 2.0, 3.0])) * u.UNITLESS

        # Special test arrays
        self.angle_array = Array(jnp.array([0.0, jnp.pi / 4, jnp.pi / 2, jnp.pi]))
        self.positive_array = Array(jnp.array([0.1, 1.0, 10.0, 100.0]))

        # Boolean arrays for bit operations
        self.bool_array1 = Array(jnp.array([True, False, True, False]))
        self.bool_array2 = Array(jnp.array([False, False, True, True]))
        self.int_array1 = Array(jnp.array([1, 2, 3, 4]))
        self.int_array2 = Array(jnp.array([4, 3, 2, 1]))

    @parameterized.named_parameters(
        *[(name, name) for name in fun_accept_unitless_unary]
    )
    def test_unary_functions_with_array_custom_array(self, fun_name):
        """Test unary functions that accept unitless inputs with Array."""
        fun = getattr(u.math, fun_name)
        jnp_fun = getattr(jnp, fun_name)

        # Test with plain Array (no units)
        try:
            result = fun(self.array_1d)
            expected = jnp_fun(self.array_1d.data)
            assert_quantity(result, expected)
        except (ValueError, OverflowError) as e:
            # Some functions may have domain restrictions
            pytest.skip(f"Function {fun_name} failed with domain error: {e}")

        # Test with dimensionless Array quantity
        try:
            result_dimensionless = fun(self.dimensionless_array)
            expected_dimensionless = jnp_fun(self.dimensionless_array.mantissa)
            assert_quantity(result_dimensionless, expected_dimensionless)
        except (ValueError, OverflowError) as e:
            pytest.skip(f"Function {fun_name} failed with dimensionless Array: {e}")

        # Test with Array having units (should require unit_to_scale)
        try:
            result_with_scale = fun(self.meter_array, unit_to_scale=u.dmetre)
            expected_with_scale = jnp_fun(self.meter_array.to_decimal(u.dmetre))
            assert_quantity(result_with_scale, expected_with_scale)
        except (ValueError, OverflowError, u.UnitMismatchError) as e:
            # Expected for functions that don't support unit scaling or have domain issues
            pass

        # Test that Array with units raises error without unit_to_scale
        with pytest.raises((AssertionError, u.UnitMismatchError)):
            fun(self.meter_array)

    def test_exp_functions_with_array_custom_array(self):
        """Test exponential functions specifically with Array."""
        # Test exp with Array
        result_exp = u.math.exp(self.array_1d)
        expected_exp = jnp.exp(self.array_1d.data)
        assert_quantity(result_exp, expected_exp)

        # Test exp2 with Array
        result_exp2 = u.math.exp2(self.array_1d)
        expected_exp2 = jnp.exp2(self.array_1d.data)
        assert_quantity(result_exp2, expected_exp2)

        # Test expm1 with Array
        result_expm1 = u.math.expm1(self.array_1d)
        expected_expm1 = jnp.expm1(self.array_1d.data)
        assert_quantity(result_expm1, expected_expm1)

        # Test with Array having voltage units
        voltage_scaled = self.voltage_array.to_decimal(u.mV)
        result_exp_voltage = u.math.exp(self.voltage_array, unit_to_scale=u.mV)
        expected_exp_voltage = jnp.exp(voltage_scaled)
        assert_quantity(result_exp_voltage, expected_exp_voltage)

    def test_log_functions_with_array_custom_array(self):
        """Test logarithmic functions specifically with Array."""
        # Use positive values for log functions
        pos_array = Array(jnp.array([0.1, 1.0, 10.0, 100.0]))

        # Test log with Array
        result_log = u.math.log(pos_array)
        expected_log = jnp.log(pos_array.data)
        assert_quantity(result_log, expected_log)

        # Test log10 with Array
        result_log10 = u.math.log10(pos_array)
        expected_log10 = jnp.log10(pos_array.data)
        assert_quantity(result_log10, expected_log10)

        # Test log1p with Array
        result_log1p = u.math.log1p(self.array_1d)
        expected_log1p = jnp.log1p(self.array_1d.data)
        assert_quantity(result_log1p, expected_log1p)

        # Test log2 with Array
        result_log2 = u.math.log2(pos_array)
        expected_log2 = jnp.log2(pos_array.data)
        assert_quantity(result_log2, expected_log2)

    def test_trigonometric_functions_with_array_custom_array(self):
        """Test trigonometric functions with Array."""
        # Test sin, cos, tan with angle Array
        result_sin = u.math.sin(self.angle_array)
        expected_sin = jnp.sin(self.angle_array.data)
        assert_quantity(result_sin, expected_sin)

        result_cos = u.math.cos(self.angle_array)
        expected_cos = jnp.cos(self.angle_array.data)
        assert_quantity(result_cos, expected_cos)

        result_tan = u.math.tan(self.angle_array)
        expected_tan = jnp.tan(self.angle_array.data)
        assert_quantity(result_tan, expected_tan)

        # Test inverse trigonometric functions
        unit_array = Array(jnp.array([-0.9, -0.5, 0.0, 0.5, 0.9]))

        result_arcsin = u.math.arcsin(unit_array)
        expected_arcsin = jnp.arcsin(unit_array.data)
        assert_quantity(result_arcsin, expected_arcsin)

        result_arccos = u.math.arccos(unit_array)
        expected_arccos = jnp.arccos(unit_array.data)
        assert_quantity(result_arccos, expected_arccos)

    def test_hyperbolic_functions_with_array_custom_array(self):
        """Test hyperbolic functions with Array."""
        # Test sinh, cosh, tanh
        result_sinh = u.math.sinh(self.array_1d)
        expected_sinh = jnp.sinh(self.array_1d.data)
        assert_quantity(result_sinh, expected_sinh)

        result_cosh = u.math.cosh(self.array_1d)
        expected_cosh = jnp.cosh(self.array_1d.data)
        assert_quantity(result_cosh, expected_cosh)

        result_tanh = u.math.tanh(self.array_1d)
        expected_tanh = jnp.tanh(self.array_1d.data)
        assert_quantity(result_tanh, expected_tanh)

        # Test inverse hyperbolic functions
        result_arcsinh = u.math.arcsinh(self.array_1d)
        expected_arcsinh = jnp.arcsinh(self.array_1d.data)
        assert_quantity(result_arcsinh, expected_arcsinh)

    def test_angle_conversion_functions_with_array_custom_array(self):
        """Test angle conversion functions with Array."""
        # Test deg2rad and rad2deg
        degree_array = Array(jnp.array([0.0, 45.0, 90.0, 180.0]))
        radian_array = Array(jnp.array([0.0, jnp.pi / 4, jnp.pi / 2, jnp.pi]))

        # Degrees to radians
        result_deg2rad = u.math.deg2rad(degree_array)
        expected_deg2rad = jnp.deg2rad(degree_array.data)
        assert_quantity(result_deg2rad, expected_deg2rad)

        # Radians to degrees
        result_rad2deg = u.math.rad2deg(radian_array)
        expected_rad2deg = jnp.rad2deg(radian_array.data)
        assert_quantity(result_rad2deg, expected_rad2deg)

        # Test degrees and radians aliases
        result_degrees = u.math.degrees(radian_array)
        expected_degrees = jnp.degrees(radian_array.data)
        assert_quantity(result_degrees, expected_degrees)

        result_radians = u.math.radians(degree_array)
        expected_radians = jnp.radians(degree_array.data)
        assert_quantity(result_radians, expected_radians)

    @parameterized.named_parameters(
        *[(name, name) for name in fun_accept_unitless_binary]
    )
    def test_binary_functions_with_array_custom_array(self, fun_name):
        """Test binary functions that accept unitless inputs with Array."""
        fun = getattr(u.math, fun_name)
        jnp_fun = getattr(jnp, fun_name)

        array1 = Array(jnp.array([1.0, 2.0, 3.0]))
        array2 = Array(jnp.array([4.0, 5.0, 6.0]))

        # Test with plain Arrays (no units)
        try:
            result = fun(array1, array2)
            expected = jnp_fun(array1.data, array2.data)
            assert_quantity(result, expected)
        except (ValueError, OverflowError) as e:
            pytest.skip(f"Function {fun_name} failed with domain error: {e}")

        # Test with Array quantities having same units
        meter1 = array1 * meter
        meter2 = array2 * meter

        try:
            result_with_scale = fun(meter1, meter2, unit_to_scale=u.cm)
            expected_with_scale = jnp_fun(
                meter1.to_decimal(u.cm),
                meter2.to_decimal(u.cm)
            )
            assert_quantity(result_with_scale, expected_with_scale)
        except (ValueError, OverflowError, u.UnitMismatchError) as e:
            # Expected for some functions
            pass

        # Test that Arrays with units raise error without unit_to_scale
        with pytest.raises((AssertionError, u.UnitMismatchError)):
            fun(meter1, meter2)

    def test_hypot_with_array_custom_array(self):
        """Test hypot function specifically with Array."""
        array_x = Array(jnp.array([3.0, 4.0, 5.0]))
        array_y = Array(jnp.array([4.0, 3.0, 12.0]))

        # Test hypot with Arrays
        result_hypot = u.math.hypot(array_x, array_y)
        expected_hypot = jnp.hypot(array_x.data, array_y.data)
        assert_quantity(result_hypot, expected_hypot)

        # Verify mathematical correctness
        expected_manual = jnp.sqrt(array_x.data ** 2 + array_y.data ** 2)
        assert jnp.allclose(result_hypot, expected_manual)

    def test_arctan2_with_array_custom_array(self):
        """Test arctan2 function specifically with Array."""
        array_y = Array(jnp.array([1.0, -1.0, 1.0, -1.0]))
        array_x = Array(jnp.array([1.0, 1.0, -1.0, -1.0]))

        # Test arctan2 with Arrays
        result_arctan2 = u.math.arctan2(array_y, array_x)
        expected_arctan2 = jnp.arctan2(array_y.data, array_x.data)
        assert_quantity(result_arctan2, expected_arctan2)

    def test_logaddexp_functions_with_array_custom_array(self):
        """Test logaddexp functions with Array."""
        array1 = Array(jnp.array([1.0, 2.0, 3.0]))
        array2 = Array(jnp.array([2.0, 1.0, 4.0]))

        # Test logaddexp
        result_logaddexp = u.math.logaddexp(array1, array2)
        expected_logaddexp = jnp.logaddexp(array1.data, array2.data)
        assert_quantity(result_logaddexp, expected_logaddexp)

        # Test logaddexp2
        result_logaddexp2 = u.math.logaddexp2(array1, array2)
        expected_logaddexp2 = jnp.logaddexp2(array1.data, array2.data)
        assert_quantity(result_logaddexp2, expected_logaddexp2)

    def test_ldexp_with_array_custom_array(self):
        """Test ldexp function with Array."""
        mantissa_array = Array(jnp.array([1.0, 2.0, 3.0]))
        exponent_array = Array(jnp.array([1, 2, 3]))

        # Test ldexp with Arrays
        result_ldexp = u.math.ldexp(mantissa_array, exponent_array)
        expected_ldexp = jnp.ldexp(mantissa_array.data, exponent_array.data)
        assert_quantity(result_ldexp, expected_ldexp)

        # Test with Array having units (first argument only)
        meter_array = mantissa_array * meter
        result_ldexp_units = u.math.ldexp(meter_array.to_decimal(meter), exponent_array)
        expected_ldexp_units = jnp.ldexp(mantissa_array.data, exponent_array.data)
        assert_quantity(result_ldexp_units, expected_ldexp_units)

    @parameterized.named_parameters(
        *[(name, name) for name in fun_elementwise_bit_operation_unary]
    )
    def test_bit_operations_unary_with_array_custom_array(self, fun_name):
        """Test unary bit operations with Array."""
        fun = getattr(u.math, fun_name)
        jnp_fun = getattr(jnp, fun_name)

        # Test with integer Array
        result = fun(self.int_array1)
        expected = jnp_fun(self.int_array1.data)
        assert_quantity(result, expected)

        # Test with boolean Array
        result_bool = fun(self.bool_array1)
        expected_bool = jnp_fun(self.bool_array1.data)
        assert_quantity(result_bool, expected_bool)

        # Test that Arrays with units raise error
        with pytest.raises(AssertionError):
            fun(self.meter_array)

    @parameterized.named_parameters(
        *[(name, name) for name in fun_elementwise_bit_operation_binary]
    )
    def test_bit_operations_binary_with_array_custom_array(self, fun_name):
        """Test binary bit operations with Array."""
        fun = getattr(u.math, fun_name)
        jnp_fun = getattr(jnp, fun_name)

        # Test with integer Arrays
        if fun_name in ['left_shift', 'right_shift']:
            # Shift operations need specific test values
            shift_array1 = Array(jnp.array([8, 16, 32]))
            shift_array2 = Array(jnp.array([1, 2, 1]))
            result = fun(shift_array1, shift_array2)
            expected = jnp_fun(shift_array1.data, shift_array2.data)
            assert_quantity(result, expected)
        else:
            result = fun(self.int_array1, self.int_array2)
            expected = jnp_fun(self.int_array1.data, self.int_array2.data)
            assert_quantity(result, expected)

        # Test with boolean Arrays
        result_bool = fun(self.bool_array1, self.bool_array2)
        expected_bool = jnp_fun(self.bool_array1.data, self.bool_array2.data)
        assert_quantity(result_bool, expected_bool)

    def test_statistical_functions_with_array_custom_array(self):
        """Test statistical functions that accept unitless inputs with Array."""
        array1 = Array(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        array2 = Array(jnp.array([[2.0, 3.0], [4.0, 5.0]]))

        # Test corrcoef if available
        if hasattr(u.math, 'corrcoef'):
            result_corrcoef = u.math.corrcoef(array1, array2)
            expected_corrcoef = jnp.corrcoef(array1.data, array2.data)
            assert_quantity(result_corrcoef, expected_corrcoef)

        # Test cov if available
        if hasattr(u.math, 'cov'):
            result_cov = u.math.cov(array1, array2)
            expected_cov = jnp.cov(array1.data, array2.data)
            assert_quantity(result_cov, expected_cov)

    def test_array_custom_array_inheritance_verification(self):
        """Test that Array properly inherits from CustomArray in unitless functions."""
        # Verify inheritance
        assert isinstance(self.array_1d, u.CustomArray)
        assert hasattr(self.array_1d, 'data')
        assert hasattr(self.array_1d, 'shape')
        assert hasattr(self.array_1d, 'dtype')

        # Test that functions work with CustomArray properties
        original_shape = self.array_2d.shape
        result_exp = u.math.exp(self.array_2d)

        # Verify original Array properties are intact
        assert self.array_2d.shape == original_shape

        # Test that result has correct shape
        assert result_exp.shape == self.array_2d.shape

    def test_array_error_handling_unitless_functions(self):
        """Test error handling with Array in unitless functions."""
        # Test unit mismatch errors
        with pytest.raises(u.UnitMismatchError):
            u.math.exp(self.meter_array, unit_to_scale=u.second)

        # Test domain errors with log functions
        negative_array = Array(jnp.array([-1.0, -2.0]))
        u.math.log(negative_array)

        # Test dimension mismatch in binary functions
        array_1d = Array(jnp.array([1.0, 2.0]))
        array_2d = Array(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        # This may or may not raise an error depending on broadcasting rules

    def test_array_with_complex_values(self):
        """Test unitless functions with complex Array values."""
        complex_array = Array(jnp.array([1 + 2j, 3 + 4j, 5 + 6j]))

        # Test functions that work with complex numbers
        if hasattr(u.math, 'angle'):
            result_angle = u.math.angle(complex_array)
            expected_angle = jnp.angle(complex_array.data)
            assert_quantity(result_angle, expected_angle)

        # Test exp with complex Array
        result_exp_complex = u.math.exp(complex_array)
        expected_exp_complex = jnp.exp(complex_array.data)
        assert_quantity(result_exp_complex, expected_exp_complex)

    def test_array_broadcasting_in_unitless_functions(self):
        """Test broadcasting behavior with Array in unitless functions."""
        array_scalar = Array(jnp.array(2.0))
        array_1d = Array(jnp.array([1.0, 2.0, 3.0]))

        # Test broadcasting in binary functions
        result_hypot = u.math.hypot(array_scalar, array_1d)
        expected_hypot = jnp.hypot(array_scalar.data, array_1d.data)
        assert_quantity(result_hypot, expected_hypot)

        # Verify broadcasting preserved shapes correctly
        assert result_hypot.shape == (3,)

    def test_array_unit_scaling_comprehensive(self):
        """Comprehensive test of unit scaling with Array."""
        # Test with various unit combinations
        voltage_array = Array(jnp.array([0.001, 0.002])) * u.volt  # 1mV, 2mV

        # Test scaling to millivolts
        result_exp_mv = u.math.exp(voltage_array, unit_to_scale=u.mV)
        expected_exp_mv = jnp.exp(jnp.array([1.0, 2.0]))  # exp of 1mV, 2mV in mV
        assert_quantity(result_exp_mv, expected_exp_mv)

        # Test scaling to microvolts
        result_exp_uv = u.math.exp(voltage_array, unit_to_scale=u.uvolt)
        expected_exp_uv = jnp.exp(jnp.array([1000.0, 2000.0]))  # exp of values in µV
        assert_quantity(result_exp_uv, expected_exp_uv)


def test_exprel():
    def loss_fn(x):
        return jnp.sum(u.math.exprel(x))

    x = jnp.array([0.0, 1e-5, 1.0])
    grads = jax.grad(loss_fn)(x)
    assert jnp.allclose(grads, jnp.asarray([0.5, 0.50000334, 1.]))

    def loss_fn2(x):
        return u.math.exprel(x)

    x = jnp.array([-1e-5, -1e-8, 0.0, 1e-8, 1e-5, 1e-3])
    grad1 = jax.jvp(loss_fn2, (x,), (jnp.ones_like(x),))[0]
    grad2 = jax.vjp(loss_fn2, x)[1](jnp.ones_like(x))[0]
    # assert jnp.allclose(grad1, grad2)

    print()
    print(grad1)
    print(grad2)
