# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

import jax.numpy as jnp
import numpy as np

import saiunit as u


def test1():
    a = u.celsius2kelvin(0)
    assert a == 273.15 * u.kelvin

    b = u.celsius2kelvin(-100)
    assert u.math.allclose(b, 173.15 * u.kelvin)


def test2():
    a = u.kelvin2celsius(273.15 * u.kelvin)
    assert a == 0

    b = u.kelvin2celsius(173.15 * u.kelvin)
    assert np.isclose(b, -100)


class Array(u.CustomArray):
    def __init__(self, value):
        self.value = value


class TestArrayCelsius(unittest.TestCase):
    def setUp(self):
        # Temperature arrays in Celsius
        self.celsius_temps = np.array([0, 25, 100])  # Water freezing, room temp, boiling
        self.celsius_array = Array(self.celsius_temps)

        # 2D temperature array
        self.celsius_2d = np.array([[0, 25], [100, -40]])  # Including absolute zero
        self.celsius_2d_array = Array(self.celsius_2d)

        # JAX array
        self.celsius_jax = jnp.array([20, 30, 40])
        self.celsius_jax_array = Array(self.celsius_jax)

    def test_temperature_properties(self):
        # Test basic properties with temperature data
        self.assertEqual(self.celsius_array.dtype, self.celsius_temps.dtype)
        self.assertEqual(self.celsius_array.shape, (3,))
        self.assertEqual(self.celsius_array.ndim, 1)
        self.assertEqual(self.celsius_array.size, 3)

        # Test 2D temperature array
        self.assertEqual(self.celsius_2d_array.shape, (2, 2))
        self.assertEqual(self.celsius_2d_array.ndim, 2)
        self.assertEqual(self.celsius_2d_array.size, 4)

    def test_temperature_conversions_addition(self):
        # Test Celsius to Kelvin conversion (C + 273.15)
        kelvin_offset = 273.15
        kelvin_temps = self.celsius_array + kelvin_offset
        expected_kelvin = np.array([273.15, 298.15, 373.15])
        np.testing.assert_array_almost_equal(kelvin_temps, expected_kelvin, decimal=2)

        # Test reverse addition
        kelvin_temps_rev = kelvin_offset + self.celsius_array
        np.testing.assert_array_almost_equal(kelvin_temps_rev, expected_kelvin, decimal=2)

    def test_temperature_conversions_multiplication(self):
        # Test Celsius to Fahrenheit: F = C * 9/5 + 32
        celsius_to_f_factor = 9 / 5
        fahrenheit_partial = self.celsius_array * celsius_to_f_factor
        expected_partial = np.array([0, 45, 180])  # Before adding 32
        np.testing.assert_array_almost_equal(fahrenheit_partial, expected_partial, decimal=2)

        # Complete Fahrenheit conversion
        fahrenheit_temps = celsius_to_f_factor * self.celsius_array + 32
        expected_fahrenheit = np.array([32, 77, 212])
        np.testing.assert_array_almost_equal(fahrenheit_temps, expected_fahrenheit, decimal=2)

    def test_temperature_ranges(self):
        # Test temperature comparison operations
        freezing_point = 0
        room_temp = 25

        # Find temperatures below freezing
        below_freezing = self.celsius_2d_array < freezing_point
        expected_below = np.array([[False, False], [False, True]])
        np.testing.assert_array_equal(below_freezing, expected_below)

        # Find room temperature or above
        room_or_above = self.celsius_2d_array >= room_temp
        expected_room = np.array([[False, True], [True, False]])
        np.testing.assert_array_equal(room_or_above, expected_room)

    def test_temperature_statistics(self):
        # Test statistical operations on temperature data
        avg_temp = self.celsius_array.mean()
        expected_avg = (0 + 25 + 100) / 3
        self.assertAlmostEqual(float(avg_temp), expected_avg, places=2)

        # Test max and min temperatures
        max_temp = self.celsius_array.max()
        min_temp = self.celsius_array.min()
        self.assertEqual(float(max_temp), 100)
        self.assertEqual(float(min_temp), 0)

        # Test temperature sum
        total_temp = self.celsius_array.sum()
        self.assertEqual(float(total_temp), 125)

    def test_temperature_array_operations(self):
        # Test array operations with temperature data
        daily_temps = Array(np.array([20, 22, 24, 26, 25]))

        # Test slicing
        weekend_temps = daily_temps[0:2]
        np.testing.assert_array_equal(weekend_temps, np.array([20, 22]))

        # Test indexing
        hottest_day = daily_temps[3]
        self.assertEqual(float(hottest_day), 26)

        # Test boolean indexing for hot days (>23°C)
        hot_days = daily_temps > 23
        expected_hot = np.array([False, False, True, True, True])
        np.testing.assert_array_equal(hot_days, expected_hot)

    def test_temperature_modifications(self):
        # Test in-place operations with temperature adjustments
        temps = Array(np.array([20, 25, 30]))

        # Increase all temperatures by 5 degrees
        temps += 5
        np.testing.assert_array_equal(temps.value, np.array([25, 30, 35]))

        # Decrease by 10 degrees
        temps -= 10
        np.testing.assert_array_equal(temps.value, np.array([15, 20, 25]))

        # Apply cooling factor
        temps *= 0.9
        expected = np.array([13.5, 18.0, 22.5])
        np.testing.assert_array_almost_equal(temps.value, expected, decimal=1)

    def test_temperature_array_methods(self):
        # Test various array methods with temperature data
        temps = Array(np.array([15, 20, 25, 20, 30]))

        # Test reshape
        temp_matrix = temps.reshape(5, 1)
        self.assertEqual(temp_matrix.shape, (5, 1))

        # Test transpose for 2D data
        temp_2d = Array(np.array([[10, 20], [30, 40]]))
        temp_transposed = temp_2d.T
        expected_transposed = np.array([[10, 30], [20, 40]])
        np.testing.assert_array_equal(temp_transposed, expected_transposed)

        # Test copy
        temps_copy = temps.copy()
        np.testing.assert_array_equal(temps.value, temps_copy)

        # Test fill with constant temperature
        constant_temp = Array(np.zeros(5))
        constant_temp.fill(22)
        np.testing.assert_array_equal(constant_temp.value, np.array([22, 22, 22, 22, 22]))

    def test_jax_temperature_operations(self):
        # Test JAX operations with temperature data
        jax_temps = Array(jnp.array([15.0, 25.0, 35.0]))

        # Test JAX math operations
        temp_squared = jax_temps ** 2
        expected_squared = jnp.array([225.0, 625.0, 1225.0])
        np.testing.assert_array_almost_equal(temp_squared, expected_squared)

        # Test JAX trigonometric functions (for periodic temperature variations)
        temp_sin = jax_temps.sin()
        expected_sin = jnp.sin(jnp.array([15.0, 25.0, 35.0]))
        np.testing.assert_array_almost_equal(temp_sin, expected_sin)

    def test_extreme_temperatures(self):
        # Test with extreme temperature values
        extreme_temps = Array(np.array([-273.15, 5778]))  # Absolute zero, Sun's surface

        # Test basic operations with extreme values
        self.assertEqual(extreme_temps.shape, (2,))
        self.assertEqual(float(extreme_temps.min()), -273.15)
        self.assertEqual(float(extreme_temps.max()), 5778)

        # Test temperature difference
        temp_diff = extreme_temps.max() - extreme_temps.min()
        expected_diff = 5778 - (-273.15)
        self.assertAlmostEqual(float(temp_diff), expected_diff, places=2)

    def test_temperature_units_consistency(self):
        # Test that operations maintain temperature unit consistency
        celsius_1 = Array(np.array([20, 25]))
        celsius_2 = Array(np.array([5, 10]))

        # Addition (temperature difference)
        temp_sum = celsius_1 + celsius_2
        np.testing.assert_array_equal(temp_sum, np.array([25, 35]))

        # Subtraction (temperature difference)
        temp_diff = celsius_1 - celsius_2
        np.testing.assert_array_equal(temp_diff, np.array([15, 15]))

        # Matrix multiplication for temperature data processing
        temp_matrix_1 = Array(np.array([[20, 25], [30, 35]]))
        temp_matrix_2 = Array(np.array([[1, 0], [0, 1]]))  # Identity matrix
        result = temp_matrix_1 @ temp_matrix_2
        np.testing.assert_array_equal(result, temp_matrix_1.value)


if __name__ == '__main__':
    unittest.main()
