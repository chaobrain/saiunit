import jax.numpy as jnp
import pytest
from absl.testing import parameterized

import brainunit as bu
import brainunit.math as bm
from brainunit import second, meter
from brainunit._base import assert_quantity

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


class TestFunArrayCreation(parameterized.TestCase):

    @parameterized.product(
        shape=[(1,), (2, 3), (4, 5, 6)],
        unit=[second, meter]
    )
    def test_fun_array_creation_given_shape(self, shape, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_given_shape]
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
        bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_given_shape_fill_value]
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
        bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_given_shape]
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
        bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_given_array]
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
        bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_given_array_fill_value]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_array_creation_given_array_fill_value]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(array, fill_value=fill_value)
            expected = jnp_fun(array, fill_value=fill_value)
            assert_quantity(result, expected)

            result = bm_fun(array * unit, fill_value=fill_value * unit)
            expected = jnp_fun(array, fill_value=fill_value)
            assert_quantity(result, expected, unit=unit)

            with pytest.raises(AssertionError):
                result = bm_fun(array, fill_value=fill_value * unit)

    @parameterized.product(
        shape=[(3, 3), (6, 6), (10, 10)],
        unit=[second, meter],
    )
    def test_fun_array_creation_given_square_array(self, shape, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_given_square_array]
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
        bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_misc1]
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

                with pytest.raises(bu.UnitMismatchError):
                    result = bm_fun(5 * unit, step=1)

                with pytest.raises(bu.UnitMismatchError):
                    result = bm_fun(5, step=1 * unit)

                with pytest.raises(bu.UnitMismatchError):
                    result = bm_fun(3 * unit, 9 * unit, 1)
                with pytest.raises(bu.UnitMismatchError):
                    result = bm_fun(3 * unit, 9, 1)

                with pytest.raises(bu.UnitMismatchError):
                    result = bm_fun(3, 9 * unit, 1)

                with pytest.raises(bu.UnitMismatchError):
                    result = bm_fun(3, 9, 1 * unit)
            else:
                result = bm_fun(5, 15, 5)
                expected = jnp_fun(5, 15, 5)
                assert_quantity(result, expected)

                result = bm_fun(5 * unit, 15 * unit, 5)
                expected = jnp_fun(5, 15, 5)
                assert_quantity(result, expected, unit=unit)

                with pytest.raises(bu.UnitMismatchError):
                    result = bm_fun(5, 15 * unit, 5)

                with pytest.raises(bu.UnitMismatchError):
                    result = bm_fun(5 * unit, 15, 5)

    @parameterized.product(
        unit=[second, meter],
    )
    def test_fun_array_creation_misc2(self, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_misc2]
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

                with pytest.raises(AssertionError):
                    result = bm_fun(jnp.array([1, 2, 3]) * unit, 3)

    @parameterized.product(
        unit=[second, meter],
    )
    def test_fun_array_creation_asarray(self, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_asarray]
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

            with pytest.raises(bu.UnitMismatchError):
                result = bm_fun(1 * unit, unit=bu.volt)

            with pytest.raises(bu.UnitMismatchError):
                result = bm_fun([1 * unit, 2 * unit * unit, 3 * unit / unit])

    @parameterized.product(
        value=[1, 10, 100]
    )
    def test_fun_array_creation_indices(self, value):
        bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_indices]
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
        bm_fun_list = [getattr(bm, fun) for fun in fun_array_creation_indices_from]
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
