import jax
import jax.numpy as jnp
from absl.testing import parameterized

import brainunit as bu
import brainunit.math as bm
from brainunit import meter, second, volt
from brainunit._base import assert_quantity

fun_change_unit_unary = [
    'reciprocal', 'var', 'nanvar', 'cbrt', 'square', 'sqrt',
]
fun_change_unit_unary_prod_cumprod = [
    'prod', 'nanprod', 'cumprod', 'nancumprod',
]
fun_change_unit_power = [
    'power', 'float_power',
]
fun_change_unit_binary = [
    'multiply', 'divide', 'cross',
    'true_divide', 'floor_divide', 'convolve',
]
fun_change_unit_binary_divmod = [
    'divmod',
]
fun_change_unit_linear_algebra = [
    'dot', 'vdot', 'inner', 'outer', 'kron', 'matmul',
]
fun_change_unit_binary_tensordot = [
    'tensordot',
]


class TestFunChangeUnit(parameterized.TestCase):

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[meter, second]
    )
    def test_fun_change_unit_unary(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_change_unit_unary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_unary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected, unit=bm_fun._unit_change_fun(unit))

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        unit=[meter, second]
    )
    def test_fun_change_unit_unary_prod_cumprod(self, value, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_change_unit_unary_prod_cumprod]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_unary_prod_cumprod]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value))
            expected = jnp_fun(jnp.array(value))
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q)
            expected = jnp_fun(jnp.array(value))

            size = len(value)
            result_unit = unit ** size
            assert_quantity(result, expected, unit=result_unit)

    @parameterized.product(
        value=[(1.0, 2.0), (1.23, 2.34, 3.45)],
        power_exponents=[2, 3],
        unit=[meter, second]
    )
    def test_fun_change_unit_power(self, value, power_exponents, unit):
        bm_fun_list = [getattr(bm, fun) for fun in fun_change_unit_power]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_power]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')

            result = bm_fun(jnp.array(value), power_exponents)
            expected = jnp_fun(jnp.array(value), power_exponents)
            assert_quantity(result, expected)

            q = value * unit
            result = bm_fun(q, power_exponents)
            expected = jnp_fun(jnp.array(value), power_exponents)
            result_unit = unit ** power_exponents
            assert_quantity(result, expected, unit=result_unit)

    @parameterized.product(
        value=[((1.123, 2.567, 3.891), (1.23, 2.34, 3.45)),
               ((1.0, 2.0), (3.0, 4.0),)],
        unit1=[meter, second],
        unit2=[volt, second]
    )
    def test_fun_change_unit_binary(self, value, unit1, unit2):
        bm_fun_list = [getattr(bm, fun) for fun in fun_change_unit_binary]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_binary]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')
            value1, value2 = value

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * unit1
            q2 = value2 * unit2
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected, unit=bm_fun._unit_change_fun(bu.get_unit(unit1), bu.get_unit(unit2)))

    @parameterized.product(
        value=[((1.123, 2.567, 3.891), (1.23, 2.34, 3.45)),
               ((1.0, 2.0), (3.0, 4.0),)],
        unit1=[meter, second],
        unit2=[meter, second]
    )
    def test_fun_change_unit_binary_divmod(self, value, unit1, unit2):
        bm_fun_list = [getattr(bm, fun) for fun in fun_change_unit_binary_divmod]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_binary_divmod]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')
            value1, value2 = value

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            for r, e in zip(result, expected):
                assert_quantity(r, e)

            q1 = value1 * unit1
            q2 = value2 * unit2
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result[0], expected[0], unit=unit1 / unit2)
            assert_quantity(result[1], expected[1], unit=unit1)

    @parameterized.product(
        value=[((1.123, 2.567, 3.891), (1.23, 2.34, 3.45)),
               ((1.0, 2.0), (3.0, 4.0),)],
        unit1=[meter, second],
        unit2=[meter, second]
    )
    def test_fun_change_unit_linear_algebra(self, value, unit1, unit2):
        bm_fun_list = [getattr(bm, fun) for fun in fun_change_unit_linear_algebra]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_linear_algebra]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')
            value1, value2 = value

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * unit1
            q2 = value2 * unit2
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected, unit=bm_fun._unit_change_fun(bu.get_unit(unit1), bu.get_unit(unit2)))

    @parameterized.product(
        value=[(((1, 2), (3, 4)), ((1, 2), (3, 4))), ],
        unit1=[meter, second],
        unit2=[meter, second]
    )
    def test_fun_change_unit_tensordot(self, value, unit1, unit2):
        bm_fun_list = [getattr(bm, fun) for fun in fun_change_unit_binary_tensordot]
        jnp_fun_list = [getattr(jnp, fun) for fun in fun_change_unit_binary_tensordot]

        for bm_fun, jnp_fun in zip(bm_fun_list, jnp_fun_list):
            print(f'fun: {bm_fun.__name__}')
            value1, value2 = value

            result = bm_fun(jnp.array(value1), jnp.array(value2))
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected)

            q1 = value1 * unit1
            q2 = value2 * unit2
            result = bm_fun(q1, q2)
            expected = jnp_fun(jnp.array(value1), jnp.array(value2))
            assert_quantity(result, expected, unit=bm_fun._unit_change_fun(bu.get_unit(unit1), bu.get_unit(unit2)))

    def test_multi_dot(self):
        key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
        x = jax.random.normal(key1, shape=(200, 5)) * bu.mA
        y = jax.random.normal(key2, shape=(5, 100)) * bu.mV
        z = jax.random.normal(key3, shape=(100, 10)) * bu.ohm
        result1 = (x @ y) @ z
        result2 = x @ (y @ z)
        assert bu.math.allclose(result1, result2, atol=1E-4 * result1.unit)
        result3 = bu.math.multi_dot([x, y, z])
        assert bu.math.allclose(result1, result3, atol=1E-4 * result1.unit)
        assert jax.jit(lambda x, y, z: (x @ y) @ z).lower(x, y, z).cost_analysis()['flops'] == 600000.0
        assert jax.jit(lambda x, y, z: x @ (y @ z)).lower(x, y, z).cost_analysis()['flops'] == 30000.0
        assert jax.jit(bu.math.multi_dot).lower([x, y, z]).cost_analysis()['flops'] == 30000.0
