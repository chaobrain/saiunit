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
import jax.lax as lax
import pytest
from absl.testing import parameterized

import brainunit as bu
import brainunit.lax as bulax
from brainunit import meter, second
from brainunit._base import assert_quantity

lax_array_creation_given_array = [
    'zeros_like_array',
    ]

lax_array_creation_misc = [
    'iota', 'broadcasted_iota',
]

class TestLaxArrayCreation(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLaxArrayCreation, self).__init__(*args, **kwargs)

        print()

    @parameterized.product(
        array=[jnp.array([1.0, 2.0]), jnp.array([[1.0, 2.0], [3.0, 4.0]])],
        unit=[second, meter]
    )
    def test_lax_array_creation_given_array(self, array, unit):
        bulax_fun_list = [getattr(bulax, fun) for fun in lax_array_creation_given_array]
        lax_fun_list = [getattr(lax, fun) for fun in lax_array_creation_given_array]

        for bulax_fun, lax_fun in zip(bulax_fun_list, lax_fun_list):
            print(f'fun: {bulax_fun.__name__}')

            result = bulax_fun(array)
            expected = lax_fun(array)
            assert_quantity(result, expected)

            result = bulax_fun(array, unit=unit)
            expected = lax_fun(array)
            assert_quantity(result, expected, unit=unit)

    @parameterized.product(
        value=[1, 10, 100],
        unit=[second, meter]
    )
    def test_lax_array_creation_iota(self, value, unit):
        fun_name = 'iota'
        bulax_fun = getattr(bulax, fun_name)
        lax_fun = getattr(lax, fun_name)

        print(f'fun: {bulax_fun.__name__}')

        result = bulax_fun(float, value)
        expected = lax_fun(float, value)
        assert_quantity(result, expected)

        result = bulax_fun(float, value, unit=unit)
        expected = lax_fun(float, value)
        assert_quantity(result, expected, unit=unit)

    @parameterized.product(
        shape=[(2, 3), (4, 5, 6)],
        unit=[second, meter]
    )
    def test_lax_array_creation_broadcasted_iota(self, shape, unit):
        fun_name = 'broadcasted_iota'
        bulax_fun = getattr(bulax, fun_name)
        lax_fun = getattr(lax, fun_name)

        dimension = len(shape) - 1

        print(f'fun: {bulax_fun.__name__}')

        result = bulax_fun(float, shape, dimension)
        expected = lax_fun(float, shape, dimension)
        assert_quantity(result, expected)

        result = bulax_fun(float, shape, dimension, unit=unit)
        expected = lax_fun(float, shape, dimension)
        assert_quantity(result, expected, unit=unit)