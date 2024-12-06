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

import brainunit.fft as ufft
from brainunit import meter, second
from brainunit._base import assert_quantity

fft_keep_unit = [
    'fftshift', 'ifftshift',
]


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
