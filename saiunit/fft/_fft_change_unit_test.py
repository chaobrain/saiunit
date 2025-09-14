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
import pytest
from absl.testing import parameterized

import saiunit as u
import saiunit.fft as ufft
from saiunit import meter, second
from saiunit._base import assert_quantity, Unit, get_or_create_dimension


class Array(u.CustomArray):
    def __init__(self, value):
        self.data = value

fft_change_1d = [
    'fft', 'ifft',
    'rfft', 'irfft',
]

fft_change_2d = [
    'fft2', 'ifft2',
    'rfft2', 'irfft2',
]

fft_change_nd = [
    'fftn', 'ifftn',
    'rfftn', 'irfftn',
]

fft_change_unit_freq = [
    'fftfreq', 'rfftfreq',
]


class TestFftChangeUnitWithArrayCustomArray(parameterized.TestCase):

    @parameterized.product(
        value_axis=[
            ([1, 2, 3], 0),
            ([1, 2, 3], -1),
            ([[1, 2, 3], [4, 5, 6]], 0),
            ([[1, 2, 3], [4, 5, 6]], -1),
        ],
        unit=[meter, second],
        norm=[None, 'ortho']
    )
    def test_fft_change_1d_with_array(self, value_axis, norm, unit):
        value = value_axis[0]
        axis = value_axis[1]
        ufft_fun_list = [getattr(ufft, fun) for fun in fft_change_1d]
        jnpfft_fun_list = [getattr(jnpfft, fun) for fun in fft_change_1d]

        for ufft_fun, jnpfft_fun in zip(ufft_fun_list, jnpfft_fun_list):
            print(f'fun: {ufft_fun.__name__}')

            result = ufft_fun(jnp.array(value), axis=axis, norm=norm)
            expected = jnpfft_fun(jnp.array(value), axis=axis, norm=norm)
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            q = jnp.array(value) * unit
            result = ufft_fun(q, axis=axis, norm=norm)
            expected = jnpfft_fun(jnp.array(value), axis=axis, norm=norm)
            assert_quantity(result, expected, unit=ufft_fun._unit_change_fun(unit))

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=ufft_fun._unit_change_fun(unit))

            array_input = Array(q)
            result = ufft_fun(array_input.data, axis=axis, norm=norm)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=ufft_fun._unit_change_fun(unit))

    @parameterized.product(
        value_axes_s=[
            ([[1, 2, 3], [4, 5, 6]], (0, 1), (3, 2)),
            ([[1, 2, 3], [4, 5, 6]], (1, 0), (3, 2)),
            ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (0, 1), (2, 3)),
            ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (1, 0), (2, 3)),
        ],
        norm=[None, 'ortho'],
        unit=[meter, second],
    )
    def test_fft_change_2d_with_array(self, value_axes_s, norm, unit):
        value = value_axes_s[0]
        axes = value_axes_s[1]
        s = value_axes_s[2]
        ufft_fun_list = [getattr(ufft, fun) for fun in fft_change_2d]
        jnpfft_fun_list = [getattr(jnpfft, fun) for fun in fft_change_2d]

        for ufft_fun, jnpfft_fun in zip(ufft_fun_list, jnpfft_fun_list):
            print(f'fun: {ufft_fun.__name__}')

            result = ufft_fun(jnp.array(value), s=s, axes=axes, norm=norm)
            expected = jnpfft_fun(jnp.array(value), s=s, axes=axes, norm=norm)
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            q = jnp.array(value) * unit
            result = ufft_fun(q, s=s, axes=axes, norm=norm)
            expected = jnpfft_fun(jnp.array(value), s=s, axes=axes, norm=norm)
            assert_quantity(result, expected, unit=ufft_fun._unit_change_fun(unit))

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=ufft_fun._unit_change_fun(unit))

            array_input = Array(q)
            result = ufft_fun(array_input.data, s=s, axes=axes, norm=norm)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=ufft_fun._unit_change_fun(unit))

    @parameterized.product(
        value_axes_s=[
            ([[1, 2, 3], [4, 5, 6]], (0, 1), (3, 2)),
            ([[1, 2, 3], [4, 5, 6]], (1, 0), (3, 2)),
            ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (0, 1), (2, 3)),
            ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (1, 0), (2, 3)),
        ],
        norm=[None, 'ortho'],
        unit=[meter, second],
    )
    def test_fft_change_nd_with_array(self, value_axes_s, norm, unit):
        value = value_axes_s[0]
        axes = value_axes_s[1]
        s = value_axes_s[2]
        ufft_fun_list = [getattr(ufft, fun) for fun in fft_change_nd]
        jnpfft_fun_list = [getattr(jnpfft, fun) for fun in fft_change_nd]

        for ufft_fun, jnpfft_fun in zip(ufft_fun_list, jnpfft_fun_list):
            print(f'fun: {ufft_fun.__name__}')

            result = ufft_fun(jnp.array(value), s=s, axes=axes, norm=norm)
            expected = jnpfft_fun(jnp.array(value), s=s, axes=axes, norm=norm)
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            q = jnp.array(value) * unit
            result = ufft_fun(q, s=s, axes=axes, norm=norm)
            expected = jnpfft_fun(jnp.array(value), s=s, axes=axes, norm=norm)
            assert_quantity(result, expected, unit=ufft_fun._unit_change_fun(unit))

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=ufft_fun._unit_change_fun(unit))

            array_input = Array(q)
            result = ufft_fun(array_input.data, s=s, axes=axes, norm=norm)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=ufft_fun._unit_change_fun(unit))

    @parameterized.product(
        size=[9, 10, 101, 102],
        d=[0.1, 2.],
    )
    def test_fft_change_unit_freq_with_array(self, size, d):
        bufft_fun_list = [getattr(ufft, fun) for fun in fft_change_unit_freq]
        jnpfft_fun_list = [getattr(jnpfft, fun) for fun in fft_change_unit_freq]

        d = jnp.array(d)

        for bufft_fun, jnpfft_fun in zip(bufft_fun_list, jnpfft_fun_list):
            print(f'fun: {bufft_fun.__name__}')

            result = bufft_fun(size, d)
            expected = jnpfft_fun(size, d)
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            q = d * second
            result = bufft_fun(size, q)
            expected = jnpfft_fun(size, d)
            assert_quantity(result, expected, unit=u.hertz)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=u.hertz)

            array_input = Array(q)
            result = bufft_fun(size, array_input.data)
            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=u.hertz)

    def test_fft_1d_operations_with_array(self):
        # Test FFT and IFFT operations with Array
        data_1d = jnp.array([1.0, 2.0, 3.0, 4.0]) * meter
        test_array_1d = Array(data_1d)
        
        assert isinstance(test_array_1d, u.CustomArray)
        
        # Test fft
        fft_result = ufft.fft(test_array_1d.data, axis=-1)
        fft_array = Array(fft_result)
        assert isinstance(fft_array, u.CustomArray)
        expected = jnpfft.fft(jnp.array([1.0, 2.0, 3.0, 4.0]), axis=-1)
        assert_quantity(fft_array.data, expected, unit=ufft.fft._unit_change_fun(meter))
        
        # Test ifft
        ifft_result = ufft.ifft(test_array_1d.data, axis=-1)
        ifft_array = Array(ifft_result)
        assert isinstance(ifft_array, u.CustomArray)
        expected = jnpfft.ifft(jnp.array([1.0, 2.0, 3.0, 4.0]), axis=-1)
        assert_quantity(ifft_array.data, expected, unit=ufft.ifft._unit_change_fun(meter))

    def test_fft_2d_operations_with_array(self):
        # Test 2D FFT operations with Array
        data_2d = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * second
        test_array_2d = Array(data_2d)
        
        assert isinstance(test_array_2d, u.CustomArray)
        
        # Test fft2
        fft2_result = ufft.fft2(test_array_2d.data, axes=(0, 1))
        fft2_array = Array(fft2_result)
        assert isinstance(fft2_array, u.CustomArray)
        expected = jnpfft.fft2(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), axes=(0, 1))
        assert_quantity(fft2_array.data, expected, unit=ufft.fft2._unit_change_fun(second))
        
        # Test ifft2
        ifft2_result = ufft.ifft2(test_array_2d.data, axes=(0, 1))
        ifft2_array = Array(ifft2_result)
        assert isinstance(ifft2_array, u.CustomArray)
        expected = jnpfft.ifft2(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), axes=(0, 1))
        assert_quantity(ifft2_array.data, expected, unit=ufft.ifft2._unit_change_fun(second))

    def test_rfft_operations_with_array(self):
        # Test real FFT operations with Array
        data_real = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]) * meter
        test_array = Array(data_real)
        
        assert isinstance(test_array, u.CustomArray)
        
        # Test rfft
        rfft_result = ufft.rfft(test_array.data, axis=-1)
        rfft_array = Array(rfft_result)
        assert isinstance(rfft_array, u.CustomArray)
        expected = jnpfft.rfft(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis=-1)
        assert_quantity(rfft_array.data, expected, unit=ufft.rfft._unit_change_fun(meter))
        
        # Test irfft
        irfft_result = ufft.irfft(test_array.data, axis=-1)
        irfft_array = Array(irfft_result)
        assert isinstance(irfft_array, u.CustomArray)
        expected = jnpfft.irfft(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis=-1)
        assert_quantity(irfft_array.data, expected, unit=ufft.irfft._unit_change_fun(meter))

    def test_fftfreq_operations_with_array(self):
        # Test frequency operations with Array
        n = 8
        d_val = 0.125
        d_time = d_val * second
        d_array = Array(d_time)
        
        assert isinstance(d_array, u.CustomArray)
        
        # Test fftfreq
        fftfreq_result = ufft.fftfreq(n, d_array.data)
        fftfreq_array = Array(fftfreq_result)
        assert isinstance(fftfreq_array, u.CustomArray)
        expected = jnpfft.fftfreq(n, d_val)
        assert_quantity(fftfreq_array.data, expected, unit=u.hertz)
        
        # Test rfftfreq
        rfftfreq_result = ufft.rfftfreq(n, d_array.data)
        rfftfreq_array = Array(rfftfreq_result)
        assert isinstance(rfftfreq_array, u.CustomArray)
        expected = jnpfft.rfftfreq(n, d_val)
        assert_quantity(rfftfreq_array.data, expected, unit=u.hertz)

    def test_array_custom_array_compatibility_with_fft_change_unit(self):
        data = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]) * second
        test_array = Array(data)
        
        assert isinstance(test_array, u.CustomArray)
        assert hasattr(test_array, 'data')
        
        # Test fftn with Array
        result = ufft.fftn(test_array.data, axes=(0, 1))
        result_array = Array(result)
        
        assert isinstance(result_array, u.CustomArray)
        
        # Compare with direct computation
        direct_result = ufft.fftn(data, axes=(0, 1))
        assert_quantity(result_array.data, direct_result.mantissa, unit=ufft.fftn._unit_change_fun(second))


class TestFftChangeUnit(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFftChangeUnit, self).__init__(*args, **kwargs)

        print()

    def test_time_freq_map(self):
        from saiunit.fft._fft_change_unit import _time_freq_map
        for v1, v2 in _time_freq_map.values():
            # print(key.scale, data.scale)
            assert v1.scale == -v2.scale

    @parameterized.product(
        value_axis=[
            ([1, 2, 3], 0),
            ([1, 2, 3], -1),
            ([[1, 2, 3], [4, 5, 6]], 0),
            ([[1, 2, 3], [4, 5, 6]], -1),
        ],
        unit=[meter, second],
        norm=[None, 'ortho']
    )
    def test_fft_change_1d(self, value_axis, norm, unit):
        value = value_axis[0]
        axis = value_axis[1]
        ufft_fun_list = [getattr(ufft, fun) for fun in fft_change_1d]
        jnpfft_fun_list = [getattr(jnpfft, fun) for fun in fft_change_1d]

        for ufft_fun, jnpfft_fun in zip(ufft_fun_list, jnpfft_fun_list):
            print(f'fun: {ufft_fun.__name__}')

            result = ufft_fun(jnp.array(value), axis=axis, norm=norm)
            expected = jnpfft_fun(jnp.array(value), axis=axis, norm=norm)
            assert_quantity(result, expected)

            q = value * unit
            result = ufft_fun(q, axis=axis, norm=norm)
            expected = ufft_fun(jnp.array(value), axis=axis, norm=norm)
            assert_quantity(result, expected, unit=ufft_fun._unit_change_fun(unit))

    @parameterized.product(
        value_axes_s=[
            ([[1, 2, 3], [4, 5, 6]], (0, 1), (3, 2)),
            ([[1, 2, 3], [4, 5, 6]], (1, 0), (3, 2)),
            ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (0, 1), (2, 3)),
            ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (1, 0), (2, 3)),
        ],
        norm=[None, 'ortho'],
        unit=[meter, second],
    )
    def test_fft_change_2d(self, value_axes_s, norm, unit):
        value = value_axes_s[0]
        axes = value_axes_s[1]
        s = value_axes_s[2]
        ufft_fun_list = [getattr(ufft, fun) for fun in fft_change_2d]
        jnpfft_fun_list = [getattr(jnpfft, fun) for fun in fft_change_2d]

        for ufft_fun, jnpfft_fun in zip(ufft_fun_list, jnpfft_fun_list):
            print(f'fun: {ufft_fun.__name__}')

            result = ufft_fun(jnp.array(value), s=s, axes=axes, norm=norm)
            expected = jnpfft_fun(jnp.array(value), s=s, axes=axes, norm=norm)
            assert_quantity(result, expected)

            q = value * unit
            result = ufft_fun(q, s=s, axes=axes, norm=norm)
            expected = ufft_fun(jnp.array(value), s=s, axes=axes, norm=norm)
            assert_quantity(result, expected, unit=ufft_fun._unit_change_fun(unit))

    @parameterized.product(
        value_axes_s=[
            ([[1, 2, 3], [4, 5, 6]], (0, 1), (3, 2)),
            ([[1, 2, 3], [4, 5, 6]], (1, 0), (3, 2)),
            ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (0, 1), (2, 3)),
            ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (1, 0), (2, 3)),
        ],
        norm=[None, 'ortho'],
        unit=[meter, second],
    )
    def test_fft_change_nd(self, value_axes_s, norm, unit):
        value = value_axes_s[0]
        axes = value_axes_s[1]
        s = value_axes_s[2]
        ufft_fun_list = [getattr(ufft, fun) for fun in fft_change_nd]
        jnpfft_fun_list = [getattr(jnpfft, fun) for fun in fft_change_nd]

        for ufft_fun, jnpfft_fun in zip(ufft_fun_list, jnpfft_fun_list):
            print(f'fun: {ufft_fun.__name__}')

            result = ufft_fun(jnp.array(value), s=s, axes=axes, norm=norm)
            expected = jnpfft_fun(jnp.array(value), s=s, axes=axes, norm=norm)
            assert_quantity(result, expected)

            q = value * unit
            result = ufft_fun(q, s=s, axes=axes, norm=norm)
            expected = ufft_fun(jnp.array(value), s=s, axes=axes, norm=norm)
            assert_quantity(result, expected, unit=ufft_fun._unit_change_fun(unit))

    @parameterized.product(
        size=[9, 10, 101, 102],
        d=[0.1, 2.],
    )
    def test_fft_change_unit_freq(self, size, d):

        bufft_fun_list = [getattr(ufft, fun) for fun in fft_change_unit_freq]
        jnpfft_fun_list = [getattr(jnpfft, fun) for fun in fft_change_unit_freq]

        d = jnp.array(d)

        for bufft_fun, jnpfft_fun in zip(bufft_fun_list, jnpfft_fun_list):
            print(f'fun: {bufft_fun.__name__}')

            result = bufft_fun(size, d)
            expected = jnpfft_fun(size, d)
            assert_quantity(result, expected)

            q = d * second
            result = bufft_fun(size, q)
            expected = jnpfft_fun(size, d)
            assert_quantity(result, expected, unit=u.hertz)

            with pytest.raises(AssertionError):
                q = d * meter
                result = bufft_fun(size, q)

            custom_time_unit = Unit.create(get_or_create_dimension(s=1), "custom_second", "cs", scale=100)
            custom_hertz_unit = Unit.create(get_or_create_dimension(s=-1), "custom_hertz", "ch", scale=-100)

            q = d * custom_time_unit
            result = bufft_fun(size, q)
            expected = jnpfft_fun(size, d)
            assert_quantity(result, expected, unit=custom_hertz_unit)
