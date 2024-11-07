import unittest

import brainstate as bst
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

import brainunit as bu
import brainunit.math as bm
from brainunit._base import assert_quantity

fun_remove_unit_unary = [
    'signbit', 'sign',
]

fun_remove_unit_heaviside = [
    'heaviside',
]
fun_remove_unit_bincount = [
    'bincount',
]
fun_remove_unit_digitize = [
    'digitize',
]

fun_remove_unit_logic_unary = [
    'all', 'any', 'logical_not',
]

fun_remove_unit_logic_binary = [
    'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
    'array_equal', 'isclose', 'allclose', 'logical_and',
    'logical_or', 'logical_xor',
]

fun_remove_unit_indexing = [
    'argsort', 'argmax', 'argmin', 'nanargmax', 'nanargmin', 'argwhere',
    'count_nonzero',
]
fun_remove_unit_indexing_return_tuple = [
    'nonzero', 'flatnonzero',
]
fun_remove_unit_searchsorted = [
    'searchsorted',
]


class TestFunChangeUnit(parameterized.TestCase):

    @parameterized.product(
        value=[(-1.0, 2.0), (-1.23, 2.34, 3.45)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_logic_unary(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_unary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_unary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

    @parameterized.product(
        value=[((1.0, 2.0), (3.0, 4.0)),
               ((1.23, 2.34, 3.45), (4.56, 5.67, 6.78))],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_heaviside(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_heaviside]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_heaviside]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            x1, x2 = value

            result = bm_fun(jnp.array(x1), jnp.array(x2))
            expected = jnp_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            q1 = x1 * unit
            q2 = x2 * unit
            result = bm_fun(q1, jnp.array(x2))
            expected = jnp_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(jnp.array(x1), q2)
                expected = jnp_fun(jnp.array(x1), jnp.array(x2))
                assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(q1, q2)
                expected = jnp_fun(jnp.array(x1), jnp.array(x2))
                assert_quantity(result, expected)

    @parameterized.product(
        value=[(1, 2), (1, 2, 3)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_bincount(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_bincount]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_bincount]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q.astype(jnp.int32))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

    @parameterized.product(
        array=[(1, 2, 3), (1, 2, 3, 4, 5)],
        bins=[(0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_digitize(self, array, bins, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_digitize]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_digitize]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(array), jnp.array(bins))
            expected = jnp_fun(jnp.array(array), jnp.array(bins))
            assert_quantity(result, expected)

            q_array = array * unit
            q_bins = bins * unit
            result = bm_fun(q_array, q_bins)
            expected = jnp_fun(jnp.array(array), jnp.array(bins))
            assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(jnp.array(array), q_bins)

            with pytest.raises(AssertionError):
                result = bm_fun(q_array, jnp.array(bins))

    @parameterized.product(
        value=[(True, True), (False, True, False)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_logic_unary(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_logic_unary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_logic_unary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit

            with pytest.raises(AssertionError):
                result = bm_fun(q)

    @parameterized.product(
        value=[((1.0, 2.0), (3.0, 4.0)),
               ((1.23, 2.34, 3.45), (1.23, 2.34, 3.45))],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_logic_binary(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_logic_binary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_logic_binary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            x1, x2 = value
            result = bm_fun(jnp.array(x1), jnp.array(x2))
            expected = jnp_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            q1 = x1 * unit
            q2 = x2 * unit
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(x1), jnp.array(x2))
            assert_quantity(result, expected)

            with pytest.raises(AssertionError):
                result = bm_fun(jnp.array(x1), q2)

            with pytest.raises(AssertionError):
                result = bm_fun(q1, jnp.array(x2))

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_indexing(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_indexing]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_indexing]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_indexing_return_tuple(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_indexing_return_tuple]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_indexing_return_tuple]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            for r, e in zip(result, expected):
                assert_quantity(r, e)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            for r, e in zip(result, expected):
                assert_quantity(r, e)

    @parameterized.product(
        value=[((1.0, 2.0), (3.0, 4.0)),
               ((1.23, 2.34, 3.45), (1.23, 2.34, 3.45))],
        unit=[bu.meter, bu.second]
    )
    def test_fun_remove_unit_searchsorted(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_remove_unit_searchsorted]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_remove_unit_searchsorted]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            x, v = value
            result = bm_fun(jnp.array(x), jnp.array(v))
            expected = jnp_fun(jnp.array(x), jnp.array(v))
            assert_quantity(result, expected)

            q_x = x * unit
            q_v = v * unit
            result = bm_fun(q_x, q_v)
            expected = jnp_fun(jnp.array(x), jnp.array(v))
            assert_quantity(result, expected)

            with pytest.raises(bu.UnitMismatchError):
                result = bm_fun(jnp.array(x), q_v)

            with pytest.raises(bu.UnitMismatchError):
                result = bm_fun(q_x, jnp.array(v))


class Test_allclose(unittest.TestCase):
    def test1(self):
        a = bst.random.random((10, 10))
        b = a + 1e-4
        assert bu.math.allclose(a, b, atol=1e-3)

        a = a * bu.ms
        b = b * bu.ms
        with pytest.raises(bu.UnitMismatchError):
            assert bu.math.allclose(a, b, atol=1e-3)
        assert bu.math.allclose(a, b, atol=1e-3 * bu.ms)

        val = bst.random.random((10, 10))
        a = val * bu.mV
        b = val * bu.ms
        with pytest.raises(bu.UnitMismatchError):
            assert bu.math.allclose(a, b)

        b = val * bu.volt
        assert not bu.math.allclose(a, b)

        b = val
        with pytest.raises(AssertionError):
            assert bu.math.allclose(a, b)

        b = val * bu.mV
        a = val
        with pytest.raises(AssertionError):
            assert bu.math.allclose(a, b)

    def test_tol(self):
        val = bst.random.random((10, 10))

        a = val * bu.mV
        b = val * bu.mV
        with pytest.raises(bu.UnitMismatchError):
            assert bu.math.allclose(a, b, atol=1e-3)
        with pytest.raises(bu.UnitMismatchError):
            assert bu.math.allclose(a, b, atol=1e-3 * bu.ms)
        assert bu.math.allclose(a, b, atol=1e-3 * bu.mV)

        with pytest.raises(bu.UnitMismatchError):
            assert bu.math.allclose(a, b, rtol=1e-8)
        with pytest.raises(bu.UnitMismatchError):
            assert bu.math.allclose(a, b, rtol=1e-8 * bu.ms)
        assert bu.math.allclose(a, b, rtol=1e-8 * bu.mV)
