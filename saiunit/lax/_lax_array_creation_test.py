# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


import jax.lax as lax
import jax.numpy as jnp
from absl.testing import parameterized

import saiunit.lax as bulax
from saiunit import meter, second
from saiunit._base_getters import assert_quantity

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


class TestLaxArrayCreationDocstringExamples:
    """Tests verifying the docstring examples for array-creation lax functions."""

    def test_zeros_like_array_with_unit(self):
        """Docstring example: zeros_like_array preserves unit."""
        q = jnp.array([3.0, 5.0]) * meter
        result = bulax.zeros_like_array(q)
        expected = jnp.array([0.0, 0.0])
        assert_quantity(result, expected, unit=meter)

    def test_iota_with_unit(self):
        """Docstring example: iota with unit."""
        result = bulax.iota(float, 5, unit=second)
        expected = jnp.array([0., 1., 2., 3., 4.])
        assert_quantity(result, expected, unit=second)

    def test_broadcasted_iota_with_unit(self):
        """Docstring example: broadcasted_iota with unit."""
        result = bulax.broadcasted_iota(float, (2, 3), 1, unit=meter)
        expected = lax.broadcasted_iota(float, (2, 3), 1)
        assert_quantity(result, expected, unit=meter)
