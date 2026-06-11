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


import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

import saiunit as u
import saiunit.math as um
from saiunit import second, meter
from saiunit._base_getters import assert_quantity


class Array(u.CustomArray):
    def __init__(self, value):
        self.data = value


fun_array_creation_given_shape = [
    'empty', 'ones', 'zeros',
]
fun_array_creation_given_shape_fill_value = [
    'full',
]
fun_array_creation_given_int = [
    'eye', 'identity', 'tri',
]
fun_array_creation_given_array = [
    'empty_like', 'ones_like', 'zeros_like', 'diag',
]
fun_array_creation_given_array_fill_value = [
    'full_like',
]
fun_array_creation_given_square_array = [
    'tril', 'triu',
]
fun_array_creation_given_square_array_fill_value = [
    'fill_diagonal',
]
fun_array_creation_misc1 = [
    'arange', 'linspace', 'logspace',
]
fun_array_creation_misc2 = [
    'meshgrid', 'vander',
]
fun_array_creation_asarray = [
    'array', 'asarray',
]
fun_array_creation_indices = [
    'tril_indices', 'triu_indices'
]
fun_array_creation_indices_from = [
    'tril_indices_from', 'triu_indices_from',
]
fun_array_creation_other = [
    'from_numpy',
    'as_numpy',
    'tree_ones_like',
    'tree_zeros_like',
]


class TestFunArrayCreationWithArrayCustomArray(parameterized.TestCase):

    @parameterized.product(
        shape=[(1,), (2, 3), (4, 5, 6)],
        unit=[second, meter]
    )
    def test_fun_array_creation_given_shape_with_array(self, shape, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_given_shape]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_shape]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(shape)
            expected = jnp_fun(shape)
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            result = bm_fun(shape, unit=unit)
            expected = jnp_fun(shape)
            assert_quantity(result, expected, unit=unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=unit)

    @parameterized.product(
        shape=[(1,), (2, 3), (4, 5, 6)],
        unit=[second, meter],
        fill_value=[-1., 1.]
    )
    def test_fun_array_creation_given_shape_fill_value_with_array(self, shape, unit, fill_value):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_given_shape_fill_value]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_shape_fill_value]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(shape, fill_value=fill_value)
            expected = jnp_fun(shape, fill_value=fill_value)
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            result = bm_fun(shape, fill_value=fill_value * unit)
            expected = jnp_fun(shape, fill_value=fill_value)
            assert_quantity(result, expected, unit=unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=unit)

    @parameterized.product(
        value=[1, 10, 100],
        unit=[second, meter]
    )
    def test_fun_array_creation_given_int_with_array(self, value, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_given_int]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_int]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(value)
            expected = jnp_fun(value)
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            result = bm_fun(value, unit=unit)
            expected = jnp_fun(value)
            assert_quantity(result, expected, unit=unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=unit)

    @parameterized.product(
        array=[jnp.array([1.0, 2.0]), jnp.array([[1.0, 2.0], [3.0, 4.0]])],
        unit=[second, meter]
    )
    def test_fun_array_creation_given_array_with_array(self, array, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_given_array]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_array]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(array)
            expected = jnp_fun(array)
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            result = bm_fun(array, unit=unit)
            expected = jnp_fun(array)
            assert_quantity(result, expected, unit=unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=unit)

    @parameterized.product(
        array=[jnp.array([1.0, 2.0]), jnp.array([[1.0, 2.0], [3.0, 4.0]])],
        unit=[second, meter],
        fill_value=[-1., 1.]
    )
    def test_fun_array_creation_given_array_fill_value_with_array(self, array, unit, fill_value):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_given_array_fill_value]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_array_fill_value]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(array, fill_value=fill_value)
            expected = jnp_fun(array, fill_value=fill_value)
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            result = bm_fun(array * unit, fill_value=fill_value * unit)
            expected = jnp_fun(array, fill_value=fill_value)
            assert_quantity(result, expected, unit=unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=unit)

            with pytest.raises(TypeError):
                result = bm_fun(array, fill_value=fill_value * unit)

    @parameterized.product(
        unit=[second, meter],
    )
    def test_fun_array_creation_asarray_with_array(self, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_asarray]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_asarray]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            result = bm_fun([1, 2, 3])
            expected = jnp_fun([1, 2, 3])
            assert_quantity(result, expected)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected)

            result = bm_fun([1, 2, 3] * unit)
            expected = jnp_fun([1, 2, 3])
            assert_quantity(result, expected, unit=unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=unit)

            result = bm_fun([1 * unit, 2 * unit, 3 * unit])
            expected = jnp_fun([1, 2, 3])
            assert_quantity(result, expected, unit=unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=unit)

            result = bm_fun(1 * unit)
            expected = jnp_fun(1)
            assert_quantity(result, expected, unit=unit)

            array_result = Array(result)
            assert isinstance(array_result, u.CustomArray)
            assert_quantity(array_result.data, expected, unit=unit)

            with pytest.raises(u.UnitMismatchError):
                result = bm_fun(1 * unit, unit=u.volt)

    def test_array_custom_array_compatibility(self):
        test_array = Array(jnp.array([1.0, 2.0, 3.0]) * meter)

        assert isinstance(test_array, u.CustomArray)
        assert hasattr(test_array, 'data')
        assert_quantity(test_array.data, jnp.array([1.0, 2.0, 3.0]), unit=meter)

        result = um.zeros_like(test_array.data)
        array_result = Array(result)
        assert isinstance(array_result, u.CustomArray)
        assert_quantity(array_result.data, jnp.zeros(3), unit=meter)

    def test_array_creation_with_custom_array_input(self):
        original_data = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * second
        test_array = Array(original_data)

        result = um.ones_like(test_array.data)
        expected = jnp.ones((2, 2))
        assert_quantity(result, expected, unit=second)

        array_result = Array(result)
        assert isinstance(array_result, u.CustomArray)
        assert_quantity(array_result.data, expected, unit=second)

        result = um.empty_like(test_array.data)
        array_result = Array(result)
        assert isinstance(array_result, u.CustomArray)
        assert array_result.data.shape == (2, 2)
        assert u.get_unit(array_result.data) == second


class TestFunArrayCreation(parameterized.TestCase):

    @parameterized.product(
        shape=[(1,), (2, 3), (4, 5, 6)],
        unit=[second, meter]
    )
    def test_fun_array_creation_given_shape(self, shape, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_given_shape]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_shape]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(shape)
            expected = jnp_fun(shape)
            assert_quantity(result, expected)

            result = bm_fun(shape, unit=unit)
            expected = jnp_fun(shape)
            assert_quantity(result, expected, unit=unit)

    @parameterized.product(
        shape=[(1,), (2, 3), (4, 5, 6)],
        unit=[second, meter],
        fill_value=[-1., 1.]
    )
    def test_fun_array_creation_given_shape_fill_value(self, shape, unit, fill_value):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_given_shape_fill_value]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_shape_fill_value]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(shape, fill_value=fill_value)
            expected = jnp_fun(shape, fill_value=fill_value)
            assert_quantity(result, expected)

            result = bm_fun(shape, fill_value=fill_value * unit)
            expected = jnp_fun(shape, fill_value=fill_value)
            assert_quantity(result, expected, unit=unit)

    @parameterized.product(
        value=[1, 10, 100],
        unit=[second, meter]
    )
    def test_fun_array_creation_given_int(self, value, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_given_shape]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_shape]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(value)
            expected = jnp_fun(value)
            assert_quantity(result, expected)

            result = bm_fun(value, unit=unit)
            expected = jnp_fun(value)
            assert_quantity(result, expected, unit=unit)

    @parameterized.product(
        array=[jnp.array([1.0, 2.0]), jnp.array([[1.0, 2.0], [3.0, 4.0]])],
        unit=[second, meter]
    )
    def test_fun_array_creation_given_array(self, array, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_given_array]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_array]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(array)
            expected = jnp_fun(array)
            assert_quantity(result, expected)

            result = bm_fun(array, unit=unit)
            expected = jnp_fun(array)
            assert_quantity(result, expected, unit=unit)

    @parameterized.product(
        array=[jnp.array([1.0, 2.0]), jnp.array([[1.0, 2.0], [3.0, 4.0]])],
        unit=[second, meter],
        fill_value=[-1., 1.]
    )
    def test_fun_array_creation_given_array_fill_value(self, array, unit, fill_value):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_given_array_fill_value]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_array_fill_value]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(array, fill_value=fill_value)
            expected = jnp_fun(array, fill_value=fill_value)
            assert_quantity(result, expected)

            result = bm_fun(array * unit, fill_value=fill_value * unit)
            expected = jnp_fun(array, fill_value=fill_value)
            assert_quantity(result, expected, unit=unit)

            with pytest.raises(TypeError):
                result = bm_fun(array, fill_value=fill_value * unit)

    @parameterized.product(
        shape=[(3, 3), (6, 6), (10, 10)],
        unit=[second, meter],
    )
    def test_fun_array_creation_given_square_array(self, shape, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_given_square_array]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_square_array]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            value = jnp.ones(shape)
            result = bm_fun(value)
            expected = jnp_fun(value)
            assert_quantity(result, expected)

            result = bm_fun(value, unit=unit)
            expected = jnp_fun(value)
            assert_quantity(result, expected, unit=unit)

    @parameterized.product(
        unit=[second, meter],
    )
    def test_fun_array_creation_misc1(self, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_misc1]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_misc1]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            if bm_fun.__name__ == 'arange':
                result = bm_fun(5)
                expected = jnp_fun(5)
                assert_quantity(result, expected)

                result = bm_fun(1, 5)
                expected = jnp_fun(1, 5)
                assert_quantity(result, expected)

                result = bm_fun(1, 5, 2)
                expected = jnp_fun(1, 5, 2)
                assert_quantity(result, expected)

                result = bm_fun(5 * unit, step=1 * unit)
                expected = jnp_fun(5, step=1)
                assert_quantity(result, expected, unit=unit)

                result = bm_fun(3 * unit, 9 * unit, 1 * unit)
                expected = jnp_fun(3, 9, 1)
                assert_quantity(result, expected, unit=unit)

                with pytest.raises(u.UnitMismatchError):
                    result = bm_fun(5 * unit, step=1)

                with pytest.raises(u.UnitMismatchError):
                    result = bm_fun(5, step=1 * unit)

                with pytest.raises(u.UnitMismatchError):
                    result = bm_fun(3 * unit, 9 * unit, 1)
                with pytest.raises(u.UnitMismatchError):
                    result = bm_fun(3 * unit, 9, 1)

                with pytest.raises(u.UnitMismatchError):
                    result = bm_fun(3, 9 * unit, 1)

                with pytest.raises(u.UnitMismatchError):
                    result = bm_fun(3, 9, 1 * unit)
            elif bm_fun.__name__ == 'logspace':
                # logspace rejects unit-bearing start/stop: the result lives in
                # multiplicative space (10**x), so inputs must be dimensionless.
                result = bm_fun(5, 15, 5)
                expected = jnp_fun(5, 15, 5)
                assert_quantity(result, expected)

                with pytest.raises(u.UnitMismatchError):
                    result = bm_fun(5 * unit, 15 * unit, 5)

                with pytest.raises(u.UnitMismatchError):
                    result = bm_fun(5, 15 * unit, 5)

                with pytest.raises(u.UnitMismatchError):
                    result = bm_fun(5 * unit, 15, 5)
            else:
                result = bm_fun(5, 15, 5)
                expected = jnp_fun(5, 15, 5)
                assert_quantity(result, expected)

                result = bm_fun(5 * unit, 15 * unit, 5)
                expected = jnp_fun(5, 15, 5)
                assert_quantity(result, expected, unit=unit)

                with pytest.raises(u.UnitMismatchError):
                    result = bm_fun(5, 15 * unit, 5)

                with pytest.raises(u.UnitMismatchError):
                    result = bm_fun(5 * unit, 15, 5)

    @parameterized.product(
        unit=[second, meter],
    )
    def test_fun_array_creation_misc2(self, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_misc2]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_misc2]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            if bm_fun.__name__ == 'meshgrid':
                result = bm_fun(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
                expected = jnp_fun(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
                for r, e in zip(result, expected):
                    assert_quantity(r, e)

                result = bm_fun(jnp.array([1, 2, 3]) * unit, jnp.array([4, 5, 6]) * unit)
                expected = jnp_fun(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
                for r, e in zip(result, expected):
                    assert_quantity(r, e, unit=unit)

            elif bm_fun.__name__ == 'vander':
                result = bm_fun(jnp.array([1, 2, 3]), 3)
                expected = jnp_fun(jnp.array([1, 2, 3]), 3)
                assert_quantity(result, expected)

                result = bm_fun(jnp.array([1, 2, 3]), 3, unit=unit)
                expected = jnp_fun(jnp.array([1, 2, 3]), 3)
                assert_quantity(result, expected, unit=unit)

                with pytest.raises(TypeError):
                    result = bm_fun(jnp.array([1, 2, 3]) * unit, 3)

    @parameterized.product(
        unit=[second, meter],
    )
    def test_fun_array_creation_asarray(self, unit):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_asarray]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_asarray]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            result = bm_fun([1, 2, 3])
            expected = jnp_fun([1, 2, 3])
            assert_quantity(result, expected)

            result = bm_fun([1, 2, 3] * unit)
            expected = jnp_fun([1, 2, 3])
            assert_quantity(result, expected, unit=unit)

            result = bm_fun([1 * unit, 2 * unit, 3 * unit])
            expected = jnp_fun([1, 2, 3])
            assert_quantity(result, expected, unit=unit)

            # list of list
            result = bm_fun([[1, 2], [3, 4]])
            expected = jnp_fun([[1, 2], [3, 4]])
            assert_quantity(result, expected)

            result = bm_fun([[1, 2], [3, 4]] * unit)
            expected = jnp_fun([[1, 2], [3, 4]])
            assert_quantity(result, expected, unit=unit)

            # scalar
            result = bm_fun(1)
            expected = jnp_fun(1)
            assert_quantity(result, expected)

            result = bm_fun(1 * unit)
            expected = jnp_fun(1)
            assert_quantity(result, expected, unit=unit)

            with pytest.raises(u.UnitMismatchError):
                result = bm_fun(1 * unit, unit=u.volt)

            with pytest.raises(u.UnitMismatchError):
                result = bm_fun([1 * unit, 2 * unit * unit, 3 * unit / unit])

    @parameterized.product(
        value=[1, 10, 100]
    )
    def test_fun_array_creation_indices(self, value):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_indices]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_indices]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(value)
            expected = jnp_fun(value)
            for r, e in zip(result, expected):
                assert_quantity(r, e)

    @parameterized.product(
        shape=[(3, 3), (6, 6), (10, 10)]
    )
    def test_fun_array_creation_indices_from(self, shape):
        bm_fun_list = [getattr(um, fun) for fun in fun_array_creation_indices_from]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_indices_from]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            value = jnp.ones(shape)
            result = bm_fun(value)
            expected = jnp_fun(value)
            for r, e in zip(result, expected):
                assert_quantity(r, e)

    def test_fun_array_creation_other(self):
        # TODO
        ...


# ---------------------------------------------------------------------------
# Docstring example tests for key array-creation functions.
# These verify the concrete examples shown in each function's docstring.
# ---------------------------------------------------------------------------


class TestDocstringExamplesArray:
    """Tests that mirror the docstring examples for ``saiunit.math.array``."""

    def test_array_plain_list(self):
        result = um.array([1, 2, 3])
        expected = jnp.array([1, 2, 3])
        assert_quantity(result, expected)

    def test_array_with_unit(self):
        result = um.array([1, 2, 3] * meter)
        expected = jnp.array([1, 2, 3])
        assert_quantity(result, expected, unit=meter)

    def test_asarray_list_of_quantities(self):
        result = um.asarray([1 * meter, 2 * meter])
        expected = jnp.array([1, 2])
        assert_quantity(result, expected, unit=meter)


class TestDocstringExamplesOnes:
    """Tests that mirror the docstring examples for ``saiunit.math.ones``."""

    def test_ones_plain(self):
        result = um.ones((3,))
        expected = jnp.ones((3,))
        assert_quantity(result, expected)

    def test_ones_with_unit(self):
        result = um.ones((2, 2), unit=meter)
        expected = jnp.ones((2, 2))
        assert_quantity(result, expected, unit=meter)


class TestDocstringExamplesZeros:
    """Tests that mirror the docstring examples for ``saiunit.math.zeros``."""

    def test_zeros_plain(self):
        result = um.zeros((3,))
        expected = jnp.zeros((3,))
        assert_quantity(result, expected)

    def test_zeros_with_unit(self):
        result = um.zeros((2,), unit=second)
        expected = jnp.zeros((2,))
        assert_quantity(result, expected, unit=second)


class TestDocstringExamplesEye:
    """Tests that mirror the docstring examples for ``saiunit.math.eye``."""

    def test_eye_plain(self):
        result = um.eye(2)
        expected = jnp.eye(2)
        assert_quantity(result, expected)

    def test_eye_with_unit(self):
        result = um.eye(2, unit=meter)
        expected = jnp.eye(2)
        assert_quantity(result, expected, unit=meter)


class TestDocstringExamplesArange:
    """Tests that mirror the docstring examples for ``saiunit.math.arange``."""

    def test_arange_plain(self):
        result = um.arange(5)
        expected = jnp.arange(5)
        assert_quantity(result, expected)

    def test_arange_with_unit(self):
        result = um.arange(0 * meter, 3 * meter, 1 * meter)
        expected = jnp.arange(0, 3, 1)
        assert_quantity(result, expected, unit=meter)


class TestDocstringExamplesLinspace:
    """Tests that mirror the docstring examples for ``saiunit.math.linspace``."""

    def test_linspace_plain(self):
        result = um.linspace(0, 10, 5)
        expected = jnp.linspace(0, 10, 5)
        assert_quantity(result, expected)

    def test_linspace_with_unit(self):
        result = um.linspace(0 * meter, 10 * meter, 5)
        expected = jnp.linspace(0, 10, 5)
        assert_quantity(result, expected, unit=meter)


class TestDocstringExamplesFull:
    """Tests that mirror the docstring examples for ``saiunit.math.full``."""

    def test_full_plain(self):
        result = um.full((2, 3), 7.0)
        expected = jnp.full((2, 3), 7.0)
        assert_quantity(result, expected)

    def test_full_with_unit(self):
        result = um.full((3,), 5.0 * meter)
        expected = jnp.full((3,), 5.0)
        assert_quantity(result, expected, unit=meter)


class TestDocstringExamplesDiag:
    """Tests that mirror the docstring examples for ``saiunit.math.diag``."""

    def test_diag_plain(self):
        result = um.diag(jnp.array([1.0, 2.0, 3.0]))
        expected = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        assert_quantity(result, expected)

    def test_diag_with_unit(self):
        result = um.diag(jnp.array([1.0, 2.0]), unit=meter)
        expected = jnp.diag(jnp.array([1.0, 2.0]))
        assert_quantity(result, expected, unit=meter)


def test_zeros_respects_default_backend():
    import saiunit as u
    from saiunit import meter
    with u.using_backend("numpy"):
        q = u.math.zeros((3,), unit=meter)
        assert q.backend == "numpy"
    with u.using_backend("jax"):
        q = u.math.zeros((3,), unit=meter)
        assert q.backend == "jax"


def test_ones_respects_default_backend():
    import saiunit as u
    from saiunit import meter
    with u.using_backend("numpy"):
        q = u.math.ones((3,), unit=meter)
        assert q.backend == "numpy"


# ---------------------------------------------------------------------------
# Regression tests for audit fixes.
# ---------------------------------------------------------------------------


class TestFillDiagonalNumpyBackend:
    """``fill_diagonal`` on numpy-backed arrays must be functional by default."""

    def test_fill_diagonal_numpy_returns_copy(self):
        a = np.zeros((3, 3))
        result = um.fill_diagonal(a, 5.0)
        expected = np.zeros((3, 3))
        np.fill_diagonal(expected, 5.0)
        assert result is not None
        assert_quantity(result, expected)
        # inplace=False must leave the input untouched
        assert (a == 0).all()

    def test_fill_diagonal_numpy_inplace_mutates_and_returns(self):
        a = np.zeros((3, 3))
        result = um.fill_diagonal(a, 5.0, inplace=True)
        expected = np.zeros((3, 3))
        np.fill_diagonal(expected, 5.0)
        assert_quantity(result, expected)
        assert (a == expected).all()

    def test_fill_diagonal_numpy_quantity(self):
        aq = np.zeros((3, 3)) * u.mV
        result = um.fill_diagonal(aq, 5.0 * u.mV)
        expected = np.zeros((3, 3))
        np.fill_diagonal(expected, 5.0)
        assert_quantity(result, expected, unit=u.mV)
        # inplace=False must leave the input untouched
        assert (aq.mantissa == 0).all()


def test_fill_diagonal_plain_array_quantity_val_raises():
    with pytest.raises(TypeError):
        um.fill_diagonal(jnp.zeros((3, 3)), 5.0 * u.mV)


def test_empty_like_quantity_forwards_shape():
    q = jnp.array([1.0, 2.0, 3.0]) * meter
    result = um.empty_like(q, shape=(5,))
    assert result.shape == (5,)
    assert u.get_unit(result) == meter


class TestAsarrayUnitStrictness:
    """``asarray(..., unit=...)`` must honour the requested unit."""

    def test_asarray_unit_on_dimensionless_input_raises(self):
        with pytest.raises(u.UnitMismatchError):
            um.asarray([1, 2, 3], unit=meter)

    def test_asarray_unit_converts_matching_dimension(self):
        result = um.asarray([1.0, 2.0] * u.mV, unit=u.volt)
        assert_quantity(result, jnp.asarray([0.001, 0.002]), unit=u.volt)


class TestAsarrayEmpty:
    """``asarray([])`` must build an empty array instead of crashing."""

    def test_asarray_empty_list(self):
        result = um.asarray([])
        expected = jnp.asarray([])
        assert_quantity(result, expected)

    def test_asarray_empty_list_with_unit(self):
        result = um.asarray([], unit=meter)
        assert isinstance(result, u.Quantity)
        assert result.shape == (0,)
        assert u.get_unit(result) == meter


class TestArangeNoneHandling:
    """``arange`` keyword-only forms must mirror numpy's semantics."""

    def test_arange_stop_keyword_only(self):
        result = um.arange(stop=5)
        assert_quantity(result, jnp.arange(5))

    def test_arange_stop_and_step_with_units(self):
        result = um.arange(stop=3 * meter, step=1 * meter)
        assert_quantity(result, jnp.arange(0, 3, 1), unit=meter)

    def test_arange_step_only_raises(self):
        with pytest.raises(TypeError):
            um.arange(step=2)


class TestLinspaceRetstep:
    """``linspace(..., retstep=True)`` must wrap samples and step separately."""

    def test_linspace_retstep_with_units(self):
        samples, step = um.linspace(0 * meter, 10 * meter, 5, retstep=True)
        assert_quantity(samples, jnp.linspace(0, 10, 5), unit=meter)
        assert_quantity(step, jnp.asarray(2.5), unit=meter)

    def test_linspace_retstep_plain(self):
        samples, step = um.linspace(0, 10, 5, retstep=True)
        assert_quantity(samples, jnp.linspace(0, 10, 5))
        assert_quantity(step, jnp.asarray(2.5))


def test_vander_unit_error_message_suggests_to_decimal():
    with pytest.raises(TypeError, match='to_decimal'):
        um.vander(jnp.array([1.0, 2.0, 3.0]) * meter, 3)


class TestAsarrayBackendDispatch:
    """asarray must keep the input's backend instead of forcing the default."""

    def test_numpy_array_stays_numpy(self):
        r = um.asarray(np.array([1.0, 2.0]))
        assert isinstance(r, np.ndarray)

    def test_numpy_array_with_dtype_stays_numpy(self):
        r = um.asarray(np.array([1.0, 2.0]), dtype=np.float16)
        assert isinstance(r, np.ndarray)
        assert r.dtype == np.float16

    def test_numpy_quantity_stays_numpy(self):
        r = um.asarray(np.array([1.0, 2.0]) * u.mV)
        assert isinstance(r, u.Quantity)
        assert isinstance(r.mantissa, np.ndarray)

    def test_jax_array_stays_jax(self):
        r = um.asarray(jnp.array([1.0, 2.0]))
        assert isinstance(r, jnp.ndarray)

    def test_list_uses_default_backend(self):
        r = um.asarray([1.0, 2.0])
        assert isinstance(r, jnp.ndarray)
