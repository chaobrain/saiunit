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


class TestFftChangeUnit(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFftChangeUnit, self).__init__(*args, **kwargs)

        print()

    def test_time_freq_map(self):
        from saiunit.fft._fft_change_unit import _time_freq_map
        for v1, v2 in _time_freq_map.values():
            # print(key.scale, value.scale)
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
