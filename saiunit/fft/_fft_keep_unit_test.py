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

import jax.numpy as jnp
import jax.numpy.fft as jnpfft
from absl.testing import parameterized

import saiunit as u
import saiunit.fft as ufft
from saiunit import meter, second
from saiunit._base import assert_quantity


class Array(u.CustomArray):
    def __init__(self, value):
        self.value = value

fft_keep_unit = [
    'fftshift', 'ifftshift',
]


class TestFftKeepUnitWithArrayCustomArray(parameterized.TestCase):

    @parameterized.product(
        value_axes=[
            ([[1, 2, 3], [4, 5, 6]], (0, 1)),
            ([[1, 2, 3], [4, 5, 6]], (1, 0)),
            ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (0, 1)),
            ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (1, 0)),
        ],
        unit=[meter, second],
    )
    def test_fft_keep_unit_with_array(self, value_axes, unit):
        value = value_axes[0]
        axes = value_axes[1]
        ufft_fun_list = [getattr(ufft, fun) for fun in fft_keep_unit]
        jnpfft_fun_list = [getattr(jnpfft, fun) for fun in fft_keep_unit]

        for ufft_fun, jnpfft_fun in zip(ufft_fun_list, jnpfft_fun_list):
            print(f'fun: {ufft_fun.__name__}')

            result = ufft_fun(jnp.array(value), axes=axes)
            expected = jnpfft_fun(jnp.array(value), axes=axes)
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.value, expected)

            q = jnp.array(value) * unit
            result = ufft_fun(q, axes=axes)
            expected = jnpfft_fun(jnp.array(value), axes=axes)
            assert_quantity(result, expected, unit=unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.value, expected, unit=unit)

            array_input = Array(q)
            result = ufft_fun(array_input.value, axes=axes)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.value, expected, unit=unit)

    def test_fftshift_operations_with_array(self):
        # Test 1D fftshift
        data_1d = jnp.array([1, 2, 3, 4, 5]) * meter
        test_array_1d = Array(data_1d)
        
        assert isinstance(test_array_1d, u.CustomArray)
        
        fftshift_result = ufft.fftshift(test_array_1d.value)
        fftshift_array = Array(fftshift_result)
        assert isinstance(fftshift_array, u.CustomArray)
        expected = jnpfft.fftshift(jnp.array([1, 2, 3, 4, 5]))
        assert_quantity(fftshift_array.value, expected, unit=meter)
        
        # Test 2D fftshift
        data_2d = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * second
        test_array_2d = Array(data_2d)
        
        assert isinstance(test_array_2d, u.CustomArray)
        
        fftshift_result = ufft.fftshift(test_array_2d.value, axes=(0, 1))
        fftshift_array = Array(fftshift_result)
        assert isinstance(fftshift_array, u.CustomArray)
        expected = jnpfft.fftshift(jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), axes=(0, 1))
        assert_quantity(fftshift_array.value, expected, unit=second)

    def test_ifftshift_operations_with_array(self):
        # Test 1D ifftshift
        data_1d = jnp.array([1, 2, 3, 4, 5]) * meter
        test_array_1d = Array(data_1d)
        
        assert isinstance(test_array_1d, u.CustomArray)
        
        ifftshift_result = ufft.ifftshift(test_array_1d.value)
        ifftshift_array = Array(ifftshift_result)
        assert isinstance(ifftshift_array, u.CustomArray)
        expected = jnpfft.ifftshift(jnp.array([1, 2, 3, 4, 5]))
        assert_quantity(ifftshift_array.value, expected, unit=meter)
        
        # Test 2D ifftshift
        data_2d = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * second
        test_array_2d = Array(data_2d)
        
        assert isinstance(test_array_2d, u.CustomArray)
        
        ifftshift_result = ufft.ifftshift(test_array_2d.value, axes=0)
        ifftshift_array = Array(ifftshift_result)
        assert isinstance(ifftshift_array, u.CustomArray)
        expected = jnpfft.ifftshift(jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), axes=0)
        assert_quantity(ifftshift_array.value, expected, unit=second)

    def test_fft_shift_inverse_operations_with_array(self):
        # Test that fftshift and ifftshift are inverse operations
        original_data = jnp.array([1, 2, 3, 4, 5, 6]) * meter
        test_array = Array(original_data)
        
        assert isinstance(test_array, u.CustomArray)
        
        # fftshift -> ifftshift should recover original
        shifted = ufft.fftshift(test_array.value)
        shifted_array = Array(shifted)
        assert isinstance(shifted_array, u.CustomArray)
        
        recovered = ufft.ifftshift(shifted_array.value)
        recovered_array = Array(recovered)
        assert isinstance(recovered_array, u.CustomArray)
        
        assert_quantity(recovered_array.value, jnp.array([1, 2, 3, 4, 5, 6]), unit=meter)

    def test_array_custom_array_compatibility_with_fft_keep_unit(self):
        data = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]]) * second
        test_array = Array(data)
        
        assert isinstance(test_array, u.CustomArray)
        assert hasattr(test_array, 'value')
        
        # Test fftshift with Array
        result = ufft.fftshift(test_array.value, axes=1)
        result_array = Array(result)
        
        assert isinstance(result_array, u.CustomArray)
        
        # Compare with direct computation
        direct_result = ufft.fftshift(data, axes=1)
        assert_quantity(result_array.value, direct_result.mantissa, unit=second)


class TestFftKeepUnit(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFftKeepUnit, self).__init__(*args, **kwargs)

        print()

    @parameterized.product(
        value_axes=[
            ([[1, 2, 3], [4, 5, 6]], (0, 1)),
            ([[1, 2, 3], [4, 5, 6]], (1, 0)),
            ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (0, 1)),
            ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (1, 0)),
        ],
        unit=[meter, second],
    )
    def test_fft_keep_unit(self, value_axes, unit):
        value = value_axes[0]
        axes = value_axes[1]
        ufft_fun_list = [getattr(ufft, fun) for fun in fft_keep_unit]
        jnpfft_fun_list = [getattr(jnpfft, fun) for fun in fft_keep_unit]

        for ufft_fun, jnpfft_fun in zip(ufft_fun_list, jnpfft_fun_list):
            print(f'fun: {ufft_fun.__name__}')

            result = ufft_fun(jnp.array(value), axes=axes)
            expected = ufft_fun(jnp.array(value), axes=axes)
            assert_quantity(result, expected)

            q = value * unit
            result = ufft_fun(q, axes=axes)
            expected = ufft_fun(jnp.array(value), axes=axes)
            assert_quantity(result, expected, unit=unit)
