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


import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
import itertools
import unittest
import warnings
from copy import deepcopy

import brainstate as bst
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_equal

import brainunit as u
from brainunit._base import (
    DIMENSIONLESS,
    UNITLESS,
    DimensionMismatchError,
    Quantity,
    Unit,
    check_units,
    fail_for_dimension_mismatch,
    get_or_create_dimension,
    get_dim,
    have_same_dim,
    display_in_unit,
    is_scalar_type,
    assert_quantity,
)
from brainunit._unit_common import *
from brainunit._unit_shortcuts import kHz, ms, mV, nS


class TestDimension(unittest.TestCase):

    def test_inplace_operations(self):
        for inplace_op in [
            volt.dim.__imul__,
            volt.dim.__idiv__,
            volt.dim.__itruediv__,
            volt.dim.__ipow__,
        ]:
            with pytest.raises(NotImplementedError):
                inplace_op(volt.dim)


class TestUnit(unittest.TestCase):
    def test_div(self):
        print()

        a = 1. * u.second
        b = 1. * u.ms
        print(a / b)

        a = 1. * u.ms
        print(a / b)

        c = u.ms / u.ms
        assert c.is_unitless

        print(u.Unit((u.ms / u.ms).dim, scale=2))
        print(u.Unit(u.ms.dim, scale=2))

    def test_mul(self):
        a = u.Unit(base=2)
        b = u.Unit(base=10)
        with pytest.raises(AssertionError):
            a * b

    def test_inplace_operations(self):
        # make sure that inplace operations do not work on units/dimensions at all
        for inplace_op in [
            volt.__iadd__,
            volt.__isub__,
            volt.__imul__,
            volt.__idiv__,
            volt.__itruediv__,
            volt.__ifloordiv__,
            volt.__imod__,
            volt.__ipow__,
        ]:
            with pytest.raises(NotImplementedError):
                inplace_op(volt)

    def test_display(self):
        print(str(u.kmeter / u.meter))
        assert_equal(str(u.kmeter / u.meter), 'Unit(10.0^3)')

    def test_unit_with_factor(self):
        self.assertTrue(1. * u.eV / u.joule == 1.6021766e-19)
        self.assertTrue(1. * u.joule / u.eV == 6.241509074460762e18)


class TestQuantity(unittest.TestCase):
    def test_dim(self):
        a = [1, 2.] * u.ms

        with self.assertRaises(NotImplementedError):
            a.dim = u.mV.dim

    def test_clip(self):
        a = [1, 2.] * u.ms
        self.assertTrue(u.math.allclose(a.clip(1.5 * u.ms, 2.5 * u.ms), [1.5, 2.] * u.ms))

        b = u.Quantity([1, 2.])
        self.assertTrue(u.math.allclose(b.clip(1.5, 2.5), u.math.asarray([1.5, 2.])))

    def test_round(self):
        for unit in [u.ms, u.joule, u.mV]:
            a = [1.1, 2.2] * unit
            self.assertTrue(u.math.allclose(a.round(), [1, 2] * unit))

        b = u.Quantity([1.1, 2.2])
        self.assertTrue(u.math.allclose(b.round(), u.math.asarray([1, 2])))

    def test_astype(self):
        a = [1, 2.] * u.ms
        self.assertTrue(a.astype(jnp.float16).dtype == jnp.float16)

    def test___array__(self):
        a = u.Quantity([1, 2.])
        self.assertTrue(u.math.allclose(np.asarray(a), np.asarray([1, 2.])))

        with self.assertRaises(TypeError):
            a = [1, 2.] * u.ms
            self.assertTrue(u.math.allclose(np.asarray(a), np.asarray([1, 2.])))

    def test__float__(self):
        a = u.Quantity(1.)
        self.assertTrue(u.math.allclose(float(a), 1.))

        a = u.Quantity([1, 2.])
        with self.assertRaises(TypeError):
            self.assertTrue(u.math.allclose(float(a), 1.5))

        with self.assertRaises(TypeError):
            a = [1, 2.] * u.ms
            self.assertTrue(u.math.allclose(float(a), 1.5))

    def test_construction(self):
        """Test the construction of Array objects"""
        q = 500 * ms
        assert_quantity(q, 0.5, second)
        q = np.float64(500) * ms
        assert_quantity(q, 0.5, second)
        q = np.array(500) * ms
        assert_quantity(q, 0.5, second)
        q = np.array([500, 1000]) * ms
        assert_quantity(q, np.array([0.5, 1]), second)
        q = Quantity(500)
        assert_quantity(q, 500)
        q = Quantity(500, unit=second)
        assert_quantity(q, 500, second)
        q = Quantity([0.5, 1], unit=second)
        assert_quantity(q, np.array([0.5, 1]), second)
        q = Quantity(np.array([0.5, 1]), unit=second)
        assert_quantity(q, np.array([0.5, 1]), second)
        q = Quantity([500 * ms, 1 * second])
        assert_quantity(q, np.array([0.5, 1]), second)
        q = Quantity.with_unit(np.array([0.5, 1]), unit=second)
        assert_quantity(q, np.array([0.5, 1]), second)
        q = [0.5, 1] * second
        assert_quantity(q, np.array([0.5, 1]), second)

        # dimensionless quantities
        q = Quantity([1, 2, 3])
        assert_quantity(q, np.array([1, 2, 3]), Unit())
        q = Quantity(np.array([1, 2, 3]))
        assert_quantity(q, np.array([1, 2, 3]), Unit())
        q = Quantity([])
        assert_quantity(q, np.array([]), Unit())

        # Illegal constructor calls
        with pytest.raises(TypeError):
            Quantity([500 * ms, 1])
        with pytest.raises(TypeError):
            Quantity(["some", "nonsense"])
        with pytest.raises(TypeError):
            Quantity([500 * ms, 1 * volt])

    def test_construction2(self):
        a = np.array([1, 2, 3]) * u.mV
        b = u.Quantity(a)
        self.assertTrue(u.math.allclose(a, b))

        c = u.Quantity(a, unit=u.volt)
        self.assertTrue(u.math.allclose(c.mantissa, np.asarray([1, 2, 3]) * 1e-3))
        self.assertTrue(u.math.allclose(c, a))
        print(c)

    def test_get_dimensions(self):
        """
        Test various ways of getting/comparing the dimensions of a Array.
        """
        q = 500 * ms
        assert get_dim(q) == get_or_create_dimension(q.dim._dims)
        assert get_dim(q) is q.dim
        assert q.has_same_unit(3 * second)
        dims = q.dim
        assert_equal(dims.get_dimension("time"), 1.0)
        assert_equal(dims.get_dimension("length"), 0)

        assert get_dim(5) is DIMENSIONLESS
        assert get_dim(5.0) is DIMENSIONLESS
        assert get_dim(np.array(5, dtype=np.int32)) is DIMENSIONLESS
        assert get_dim(np.array(5.0)) is DIMENSIONLESS
        assert get_dim(np.float32(5.0)) is DIMENSIONLESS
        assert get_dim(np.float64(5.0)) is DIMENSIONLESS
        assert is_scalar_type(5)
        assert is_scalar_type(5.0)
        assert is_scalar_type(np.array(5, dtype=np.int32))
        assert is_scalar_type(np.array(5.0))
        assert is_scalar_type(np.float32(5.0))
        assert is_scalar_type(np.float64(5.0))
        # wrong number of indices
        with pytest.raises(TypeError):
            get_or_create_dimension([1, 2, 3, 4, 5, 6])
        # not a sequence
        with pytest.raises(TypeError):
            get_or_create_dimension(42)

    def test_display(self):
        """
        Test displaying a Array in different units
        """

        assert_equal(display_in_unit(3. * volt, mvolt), "3000. * mvolt")
        # assert_equal(display_in_unit(10. * mV, ohm * amp), "0.01 ohm * A")
        assert_equal(display_in_unit(10. * mV, ohm * amp), "0.01 * volt")
        with pytest.raises(u.UnitMismatchError):
            display_in_unit(10 * nS, ohm)
        with bst.environ.context(precision=32):
            assert_equal(display_in_unit(3. * volt, mvolt), "3000. * mvolt")
            assert_equal(display_in_unit(10. * mV, ohm * amp), "0.01 * volt")
            with pytest.raises(u.UnitMismatchError):
                display_in_unit(10 * nS, ohm)
        assert_equal(display_in_unit(10.0, Unit(scale=1)), "1. * Unit(10.0^1)")
        assert_equal(str(3 * u.kmeter / u.meter), '3000.0')
        assert_equal(str(u.mS / u.cm ** 2), 'mS/cmeter2')

        assert_equal(display_in_unit(10. * u.mV), '10. * mvolt')
        assert_equal(display_in_unit(10. * u.ohm * u.amp), '10. * volt')
        assert_equal(display_in_unit(120. * (u.mS / u.cm ** 2)), '120. * msiemens / cmeter2')
        assert_equal(display_in_unit(3.0 * u.kmeter / 130.51 * u.meter), '0.02298675 * 10.0^3 * meter2')
        assert_equal(display_in_unit(3.0 * u.kmeter / (130.51 * u.meter)), 'Quantity(22.986746)')
        assert_equal(display_in_unit(3.0 * u.kmeter / 130.51 * u.meter * u.cm ** -2), 'Quantity(229867.45)')
        assert_equal(display_in_unit(3.0 * u.kmeter / 130.51 * u.meter * u.cm ** -1), '0.02298675 * 10.0^5 * meter')
        assert_equal(display_in_unit(1. * u.joule / u.kelvin), '1. * joule / kelvin')

        assert_equal(str(1. * u.metre / ((3.0 * u.ms) / (1. * u.second))), '333.33334 * meter')
        assert_equal(str(1. * u.metre / ((3.0 * u.ms) / 1. * u.second)), '0.33333334 * 10.0^3 * metre * second ** -2')
        assert_equal(str((3.0 * u.ms) / 1. * u.second), '3. * 10.0^-3 * second2')

    # def test_display2(self):
    #
    #   @jax.jit
    #   def f(s):
    #     a = u.ms ** s
    #     print(a)
    #     return u.Quantity(1., unit=a)
    #
    #   f(2)

    def test_unary_operations(self):
        q = Quantity(5, unit=mV)
        assert_quantity(-q, -5, mV)
        assert_quantity(+q, 5, mV)
        assert_quantity(abs(Quantity(-5, unit=mV)), 5, mV)
        assert_quantity(~Quantity(0b101), -0b110, UNITLESS)

    def test_operations(self):
        q1 = 5 * second
        q2 = 10 * second
        assert_quantity(q1 + q2, 15, second)
        assert_quantity(q1 - q2, -5, second)
        assert_quantity(q1 * q2, 50, second * second)
        assert_quantity(q2 / q1, 2)
        assert_quantity(q2 // q1, 2)
        assert_quantity(q2 % q1, 0, second)
        assert_quantity(divmod(q2, q1)[0], 2)
        assert_quantity(divmod(q2, q1)[1], 0, second)
        assert_quantity(q1 ** 2, 25, second ** 2)
        assert_quantity(round(q1, 0), 5, second)

        # matmul
        q1 = [1, 2] * second
        q2 = [3, 4] * second
        assert_quantity(q1 @ q2, 11, second ** 2)
        q1 = Quantity([1, 2], unit=second)
        q2 = Quantity([3, 4], unit=second)
        assert_quantity(q1 @ q2, 11, second ** 2)

        # shift
        q1 = Quantity(0b1100, dtype=jnp.int32)
        assert_quantity(q1 << 1, 0b11000)
        assert_quantity(q1 >> 1, 0b110)

    def test_numpy_methods(self):
        q = [[1, 2], [3, 4]] * second
        assert q.all()
        assert q.any()
        assert q.nonzero()[0].tolist() == [0, 0, 1, 1]
        assert q.argmax() == 3
        assert q.argmin() == 0
        assert q.argsort(axis=None).tolist() == [0, 1, 2, 3]
        assert_quantity(q.var(), 1.25, second ** 2)
        assert_quantity(q.round(), [[1, 2], [3, 4]], second)
        assert_quantity(q.std(), 1.11803398875, second)
        assert_quantity(q.sum(), 10, second)
        assert_quantity(q.trace(), 5, second)
        assert_quantity(q.cumsum(), [1, 3, 6, 10], second)
        assert_quantity(q.cumprod(), [1, 2, 6, 24], second ** 4)
        assert_quantity(q.diagonal(), [1, 4], second)
        assert_quantity(q.max(), 4, second)
        assert_quantity(q.mean(), 2.5, second)
        assert_quantity(q.min(), 1, second)
        assert_quantity(q.ptp(), 3, second)
        assert_quantity(q.ravel(), [1, 2, 3, 4], second)

    def test_shape_manipulation(self):
        q = [[1, 2], [3, 4]] * volt

        # Test flatten
        assert_quantity(q.flatten(), [1, 2, 3, 4], volt)

        # Test swapaxes
        assert_quantity(q.swapaxes(0, 1), [[1, 3], [2, 4]], volt)

        # Test take
        assert_quantity(q.take(jnp.array([0, 2])), [1, 3], volt)

        # Test transpose
        assert_quantity(q.transpose(), [[1, 3], [2, 4]], volt)

        # Test tile
        assert_quantity(q.tile(2), [[1, 2, 1, 2], [3, 4, 3, 4]], volt)

        # Test unsqueeze
        assert_quantity(q.unsqueeze(0), [[[1, 2], [3, 4]]], volt)

        # Test expand_dims
        assert_quantity(q.expand_dims(0), [[[1, 2], [3, 4]]], volt)

        # Test expand_as
        expand_as_shape = (1, 2, 2)
        assert_quantity(q.expand_as(jnp.zeros(expand_as_shape).shape), [[[1, 2], [3, 4]]], volt)

        # Test put
        q_put = [[1, 2], [3, 4]] * volt
        q_put.put(((1, 0), (0, 1)), [10, 30] * volt)
        assert_quantity(q_put, [[1, 30], [10, 4]], volt)

        # Test squeeze (no axes to squeeze in this case, so the array remains the same)
        q_squeeze = [[1, 2], [3, 4]] * volt
        assert_quantity(q_squeeze.squeeze(), [[1, 2], [3, 4]], volt)

        # Test array_split
        q_spilt = [[10, 2], [30, 4]] * volt
        assert_quantity(np.array_split(q_spilt, 2)[0], [[10, 2]], volt)

    def test_misc_methods(self):
        q = [5, 10, 15] * volt

        # Test astype
        assert_quantity(q.astype(np.float32), [5, 10, 15], volt)

        # Test clip
        min_val = [6, 6, 6] * volt
        max_val = [14, 14, 14] * volt
        assert_quantity(q.clip(min_val, max_val), [6, 10, 14], volt)

        # Test conj
        assert_quantity(q.conj(), [5, 10, 15], volt)

        # Test conjugate
        assert_quantity(q.conjugate(), [5, 10, 15], volt)

        # Test copy
        assert_quantity(q.copy(), [5, 10, 15], volt)

        # Test dot
        assert_quantity(q.dot(Quantity([2, 2, 2])), 60, volt)

        # Test fill
        q_filled = [5, 10, 15] * volt
        q_filled.fill(2 * volt)
        assert_quantity(q_filled, [2, 2, 2], volt)

        # Test item
        assert_quantity(q.item(0), 5, volt)

        # Test prod
        assert_quantity(q.prod(), 750, volt ** 3)

        # Test repeat
        assert_quantity(q.repeat(2), [5, 5, 10, 10, 15, 15], volt)

        # Test clamp (same as clip, but using min and max values directly)
        assert_quantity(q.clip(6 * volt, 14 * volt), [6, 10, 14], volt)

        # Test sort
        q = [15, 5, 10] * volt
        assert_quantity(q.sort(), [5, 10, 15], volt)

    def test_slicing(self):
        # Slicing and indexing, setting items
        a = np.reshape(np.arange(6), (2, 3))
        q = a * mV
        assert u.math.allclose(q[:].mantissa, q.mantissa)
        assert u.math.allclose(q[0].mantissa, (a[0] * volt).mantissa)
        assert u.math.allclose(q[0:1].mantissa, (a[0:1] * volt).mantissa)
        assert u.math.allclose(q[0, 1].mantissa, (a[0, 1] * volt).mantissa)
        assert u.math.allclose(q[0:1, 1:].mantissa, (a[0:1, 1:] * volt).mantissa)
        bool_matrix = np.array([[True, False, False], [False, False, True]])
        assert u.math.allclose(q[bool_matrix].mantissa, (a[bool_matrix] * volt).mantissa)

    def test_setting(self):
        quantity = np.reshape(np.arange(6), (2, 3)) * mV
        quantity[0, 1] = 10 * mV
        assert quantity[0, 1] == 10 * mV
        quantity[:, 1] = 20 * mV
        assert np.all(quantity[:, 1] == 20 * mV)
        quantity[1, :] = np.ones((3,)) * volt
        assert np.all(quantity[1, :] == 1 * volt)

        quantity[1, 2] = 0 * mV
        assert quantity[1, 2] == 0 * mV

        def set_to_value(key, value):
            quantity[key] = value

        with pytest.raises(TypeError):
            set_to_value(0, 1)
        with pytest.raises(u.UnitMismatchError):
            set_to_value(0, 1 * second)
        with pytest.raises(TypeError):
            set_to_value((slice(2), slice(3)), np.ones((2, 3)))

        quantity = Quantity(bst.random.rand(10))
        quantity[0] = 1

    def test_multiplication_division(self):
        u = mV
        quantities = [3 * mV, np.array([1, 2]) * u, np.ones((3, 3)) * u]
        q2 = 5 * second

        for q in quantities:
            # Scalars and array scalars
            assert_quantity(q / 3, q.mantissa / 3, u)
            assert_quantity(3 / q, 3 / q.mantissa, 1 / u)
            assert_quantity(q * 3, q.mantissa * 3, u)
            assert_quantity(3 * q, 3 * q.mantissa, u)
            assert_quantity(q / np.float64(3), q.mantissa / 3, u)
            assert_quantity(np.float64(3) / q, 3 / q.mantissa, 1 / u)
            assert_quantity(q * np.float64(3), q.mantissa * 3, u)
            assert_quantity(np.float64(3) * q, 3 * q.mantissa, u)
            assert_quantity(q / jnp.array(3), q.mantissa / 3, u)
            assert_quantity(np.array(3) / q, 3 / q.mantissa, 1 / u)
            assert_quantity(q * jnp.array(3), q.mantissa * 3, u)
            assert_quantity(np.array(3) * q, 3 * q.mantissa, u)

            # (unitless) arrays
            assert_quantity(q / np.array([3]), q.mantissa / 3, u)
            assert_quantity(np.array([3]) / q, 3 / q.mantissa, 1 / u)
            assert_quantity(q * np.array([3]), q.mantissa * 3, u)
            assert_quantity(np.array([3]) * q, 3 * q.mantissa, u)

            # arrays with units
            assert_quantity(q / q, q.mantissa / q.mantissa)
            assert_quantity(q * q, q.mantissa ** 2, u ** 2)
            assert_quantity(q / q2, q.mantissa / q2.mantissa, u / second)
            assert_quantity(q2 / q, q2.mantissa / q.mantissa, second / u)
            assert_quantity(q * q2, q.mantissa * q2.mantissa, u * second)

            # # using unsupported objects should fail
            # with pytest.raises(TypeError):
            #   q / "string"
            # with pytest.raises(TypeError):
            #   "string" / q
            # with pytest.raises(TypeError):
            #   "string" * q
            # with pytest.raises(TypeError):
            #   q * "string"

    def test_addition_subtraction(self):
        unit = mV
        quantities = [3 * mV, np.array([1, 2]) * mV, np.ones((3, 3)) * mV]
        q2 = 5 * volt
        q2_mantissa = q2.in_unit(unit).mantissa

        for q in quantities:
            # arrays with units
            assert_quantity(q + q, q.mantissa + q.mantissa, unit)
            assert_quantity(q - q, 0, unit)
            assert_quantity(q + q2, q.mantissa + q2_mantissa, unit)
            assert_quantity(q2 + q, q2_mantissa + q.mantissa, unit)
            assert_quantity(q - q2, q.mantissa - q2_mantissa, unit)
            assert_quantity(q2 - q, q2_mantissa - q.mantissa, unit)

            # mismatching units
            with pytest.raises(u.UnitMismatchError):
                q + 5 * second
            with pytest.raises(u.UnitMismatchError):
                5 * second + q
            with pytest.raises(u.UnitMismatchError):
                q - 5 * second
            with pytest.raises(u.UnitMismatchError):
                5 * second - q

            # scalar
            with pytest.raises(u.UnitMismatchError):
                q + 5
            with pytest.raises(u.UnitMismatchError):
                5 + q
            with pytest.raises(u.UnitMismatchError):
                q + np.float64(5)
            with pytest.raises(u.UnitMismatchError):
                np.float64(5) + q
            with pytest.raises(u.UnitMismatchError):
                q - 5
            with pytest.raises(u.UnitMismatchError):
                5 - q
            with pytest.raises(u.UnitMismatchError):
                q - np.float64(5)
            with pytest.raises(u.UnitMismatchError):
                np.float64(5) - q

            # unitless array
            with pytest.raises(u.UnitMismatchError):
                q + np.array([5])
            with pytest.raises(u.UnitMismatchError):
                np.array([5]) + q
            with pytest.raises(u.UnitMismatchError):
                q + np.array([5], dtype=np.float64)
            with pytest.raises(u.UnitMismatchError):
                np.array([5], dtype=np.float64) + q
            with pytest.raises(u.UnitMismatchError):
                q - np.array([5])
            with pytest.raises(u.UnitMismatchError):
                np.array([5]) - q
            with pytest.raises(u.UnitMismatchError):
                q - np.array([5], dtype=np.float64)
            with pytest.raises(u.UnitMismatchError):
                np.array([5], dtype=np.float64) - q

            # Check that operations with 0 work
            with pytest.raises(u.UnitMismatchError):
                assert_quantity(q + 0, q.mantissa, unit)
            with pytest.raises(u.UnitMismatchError):
                assert_quantity(0 + q, q.mantissa, unit)
            with pytest.raises(u.UnitMismatchError):
                assert_quantity(q - 0, q.mantissa, unit)
            with pytest.raises(u.UnitMismatchError):
                # Doesn't support 0 - Quantity
                # assert_quantity(0 - q, -q.mantissa, volt)
                assert_quantity(q + np.float64(0), q.mantissa, unit)
            with pytest.raises(u.UnitMismatchError):
                assert_quantity(np.float64(0) + q, q.mantissa, unit)
            with pytest.raises(u.UnitMismatchError):
                assert_quantity(q - np.float64(0), q.mantissa, unit)

            # # using unsupported objects should fail
            # with pytest.raises(u.UnitMismatchError):
            #   "string" + q
            # with pytest.raises(u.UnitMismatchError):
            #   q + "string"
            # with pytest.raises(u.UnitMismatchError):
            #   q - "string"
            # with pytest.raises(u.UnitMismatchError):
            #   "string" - q

    def test_binary_operations(self):
        """Test whether binary operations work when they should and raise
        DimensionMismatchErrors when they should.
        Does not test for the actual result.
        """
        from operator import add, eq, ge, gt, le, lt, ne, sub

        def assert_operations_work(a, b):
            try:
                # Test python builtins
                tryops = [add, sub, lt, le, gt, ge, eq, ne]
                for op in tryops:
                    op(a, b)
                    op(b, a)

                # Test equivalent numpy functions
                numpy_funcs = [
                    u.math.add,
                    u.math.subtract,
                    u.math.less,
                    u.math.less_equal,
                    u.math.greater,
                    u.math.greater_equal,
                    u.math.equal,
                    u.math.not_equal,
                    u.math.maximum,
                    u.math.minimum,
                ]
                for numpy_func in numpy_funcs:
                    numpy_func(a, b)
                    numpy_func(b, a)
            except DimensionMismatchError as ex:
                raise AssertionError(f"Operation raised unexpected exception: {ex}")

        def assert_operations_do_not_work(a, b):
            # Test python builtins
            tryops = [add, sub, lt, le, gt, ge, eq, ne]
            for op in tryops:
                with pytest.raises(u.UnitMismatchError):
                    op(a, b)
                with pytest.raises(u.UnitMismatchError):
                    op(b, a)

        #
        # Check that consistent units work
        #

        # unit arrays
        a = 1 * kilogram
        for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
            assert_operations_work(a, b)

        # dimensionless units and scalars
        a = 1
        for b in [
            2 * kilogram / kilogram,
            np.array([2]) * kilogram / kilogram,
            np.array([1, 2]) * kilogram / kilogram,
        ]:
            assert_operations_work(a, b)

        # dimensionless units and unitless arrays
        a = np.array([1])
        for b in [
            2 * kilogram / kilogram,
            np.array([2]) * kilogram / kilogram,
            np.array([1, 2]) * kilogram / kilogram,
        ]:
            assert_operations_work(a, b)

        #
        # Check that inconsistent units do not work
        #

        # unit arrays
        a = np.array([1]) * second
        for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
            assert_operations_do_not_work(a, b)

        # unitless array
        a = np.array([1])
        for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
            assert_operations_do_not_work(a, b)

        # scalar
        a = 1
        for b in [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]:
            assert_operations_do_not_work(a, b)

        # Check that comparisons with inf/-inf always work
        values = [
            2 * kilogram / kilogram,
            2 * kilogram,
            np.array([2]) * kilogram,
            np.array([1, 2]) * kilogram,
        ]
        for value in values:
            assert u.math.all(value < np.inf * u.get_unit(value))
            assert u.math.all(np.inf * u.get_unit(value) > value)
            assert u.math.all(value <= np.inf * u.get_unit(value))
            assert u.math.all(np.inf * u.get_unit(value) >= value)
            assert u.math.all(value != np.inf * u.get_unit(value))
            assert u.math.all(np.inf * u.get_unit(value) != value)
            assert u.math.all(value >= -np.inf * u.get_unit(value))
            assert u.math.all(-np.inf * u.get_unit(value) <= value)
            assert u.math.all(value > -np.inf * u.get_unit(value))
            assert u.math.all(-np.inf * u.get_unit(value) < value)

    def test_power(self):
        """
        Test raising quantities to a power.
        """
        arrs = [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]
        for a in arrs:
            assert_quantity(a ** 3, a.mantissa ** 3, kilogram ** 3)
            # Test raising to a dimensionless Array
            assert_quantity(a ** (3 * volt / volt), a.mantissa ** 3, kilogram ** 3)
            with pytest.raises(AssertionError):
                a ** (2 * volt)
            with pytest.raises(TypeError):
                a ** np.array([2, 3])

    def test_inplace_operations(self):
        q = np.arange(10) * volt
        q_orig = q.copy()
        q_id = id(q)

        # Doesn't support in-place operations which change unit
        # q *= 2
        # assert np.all(q == 2 * q_orig) and id(q) == q_id
        # q /= 2
        # assert np.all(q == q_orig) and id(q) == q_id
        q += 1 * volt
        assert np.all(q == q_orig + 1 * volt) and id(q) == q_id
        q -= 1 * volt
        assert np.all(q == q_orig) and id(q) == q_id

        # q **= 2
        # assert np.all(q == q_orig ** 2) and id(q) == q_id
        # q **= 0.5
        # assert np.all(q == q_orig) and id(q) == q_id

        def illegal_add(q2):
            q = np.arange(10) * volt
            q += q2

        with pytest.raises(u.UnitMismatchError):
            illegal_add(1 * second)
        with pytest.raises(u.UnitMismatchError):
            illegal_add(1)

        def illegal_sub(q2):
            q = np.arange(10) * volt
            q -= q2

        with pytest.raises(u.UnitMismatchError):
            illegal_sub(1 * second)
        with pytest.raises(u.UnitMismatchError):
            illegal_sub(1)

        def illegal_pow(q2):
            q = np.arange(10) * volt
            q **= q2

        with pytest.raises(NotImplementedError):
            illegal_pow(1 * volt)
        with pytest.raises(NotImplementedError):
            illegal_pow(np.arange(10))

    def test_deepcopy(self):
        d = {"x": 1 * second}

        d_copy = deepcopy(d)
        assert d_copy["x"] == 1 * second
        d_copy["x"] += 1 * second
        assert d_copy["x"] == 2 * second
        assert d["x"] == 1 * second

    def test_numpy_functions_indices(self):
        """
        Check numpy functions that return indices.
        """
        values = [np.array([-4, 3, -2, 1, 0]), np.ones((3, 3)), np.array([17])]
        units = [volt, second, siemens, mV, kHz]

        # numpy functions
        keep_dim_funcs = [np.argmin, np.argmax, np.argsort, np.nonzero]

        for value, unit in itertools.product(values, units):
            q_ar = value * unit
            for func in keep_dim_funcs:
                test_ar = func(q_ar)
                # Compare it to the result on the same value without units
                comparison_ar = func(value)
                test_ar = np.asarray(test_ar)
                comparison_ar = np.asarray(comparison_ar)
                assert_equal(
                    test_ar,
                    comparison_ar,
                    (
                        "function %s returned an incorrect result when used on quantities "
                        % func.__name__
                    ),
                )

    def test_list(self):
        """
        Test converting to and from a list.
        """
        values = [3 * mV, np.array([1, 2]) * mV, np.arange(12).reshape(4, 3) * mV]
        for value in values:
            l = value.tolist()
            from_list = Quantity(l)
            assert have_same_dim(from_list, value)
            assert u.math.allclose(from_list.mantissa, value.mantissa)

    def test_units_vs_quantities(self):
        # Unit objects should stay Unit objects under certain operations
        # (important e.g. in the unit definition of Equations, where only units but
        # not quantities are allowed)
        assert isinstance(meter ** 2, Unit)
        assert isinstance(meter ** -1, Unit)
        assert isinstance(meter ** 0.5, Unit)
        assert isinstance(meter / second, Unit)
        assert isinstance(amp / meter ** 2, Unit)
        assert isinstance(1 / meter, Unit)
        assert isinstance(1.0 / meter, Unit)

        # Using the unconventional type(x) == y since we want to test that
        # e.g. meter**2 stays a Unit and does not become a Array however Unit
        # inherits from Array and therefore both would pass the isinstance test
        assert type(2 / meter) == Quantity
        assert type(2 * meter) == Quantity
        assert type(meter + meter) == Unit
        assert type(meter - meter) == Unit

    def test_jit_array(self):
        @jax.jit
        def f1(a):
            b = a * u.siemens / u.cm ** 2
            print(b)
            return b

        val = np.random.rand(3)
        r = f1(val)
        u.math.allclose(val * u.siemens / u.cm ** 2, r)

        @jax.jit
        def f2(a):
            a = a + 1. * u.siemens / u.cm ** 2
            return a

        val = np.random.rand(3) * u.siemens / u.cm ** 2
        r = f2(val)
        u.math.allclose(val + 1 * u.siemens / u.cm ** 2, r)

        @jax.jit
        def f3(a):
            b = a * u.siemens / u.cm ** 2
            print(display_in_unit(b, u.siemens / u.meter ** 2))
            return b

        val = np.random.rand(3)
        r = f3(val)
        u.math.allclose(val * u.siemens / u.cm ** 2, r)

    def test_jit_array2(self):
        a = 2.0 * (u.farad / u.metre ** 2)
        print(a)

        @jax.jit
        def f(b):
            print(b)
            return b

        f(a)

    def test_setiterm(self):
        unit = u.Quantity([0, 0, 0.])
        unit[jnp.asarray([0, 1, 1])] += jnp.asarray([1., 1., 1.])
        assert_quantity(unit, [1., 1., 0.])

        unit = u.Quantity([0, 0, 0.])
        unit = unit.scatter_add(jnp.asarray([0, 1, 1]), jnp.asarray([1., 1., 1.]))
        assert_quantity(unit, [1., 2., 0.])

        nu = np.asarray([0, 0, 0.])
        nu[np.asarray([0, 1, 1])] += np.asarray([1., 1., 1.])
        self.assertTrue(np.allclose(nu, np.asarray([1., 1., 0.])))

    def test_at(self):
        x = jnp.arange(5.0) * u.mV
        with self.assertRaises(u.UnitMismatchError):
            x.at[2].add(10)
        x.at[2].add(10 * u.mV)
        x.at[10].add(10 * u.mV)  # out-of-bounds indices are ignored
        x.at[20].add(10 * u.mV, mode='clip')
        x.at[2].get()
        x.at[20].get()  # out-of-bounds indices clipped
        x.at[20].get(mode='fill')  # out-of-bounds indices filled with NaN
        with self.assertRaises(u.UnitMismatchError):
            x.at[20].get(mode='fill', fill_value=-1)  # custom fill value
        x.at[20].get(mode='fill', fill_value=-1 * u.mV)  # custom fill value

    def test_to(self):
        x = jnp.arange(5.0) * u.mV
        with self.assertRaises(u.UnitMismatchError):
            x.to(u.mA)
        print(x.to(u.volt))
        print(x.to(u.uvolt))


class TestNumPyFunctions(unittest.TestCase):
    def test_special_case_numpy_functions(self):
        """
        Test a couple of functions/methods that need special treatment.
        """
        from brainunit.math import diagonal, ravel, trace, where
        from brainunit.linalg import dot

        quadratic_matrix = np.reshape(np.arange(9), (3, 3)) * mV

        # Temporarily suppress warnings related to the matplotlib 1.3 bug
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Check that function and method do the same
            assert u.math.allclose(ravel(quadratic_matrix).mantissa, quadratic_matrix.ravel().mantissa)
            # Check that function gives the same result as on unitless arrays
            assert u.math.allclose(
                np.asarray(ravel(quadratic_matrix).mantissa),
                ravel(np.asarray(quadratic_matrix.mantissa))
            )
            # Check that the function gives the same results as the original numpy
            # function
            assert u.math.allclose(
                np.ravel(np.asarray(quadratic_matrix.mantissa)),
                ravel(np.asarray(quadratic_matrix.mantissa))
            )

        # Do the same checks for diagonal, trace and dot
        assert u.math.allclose(diagonal(quadratic_matrix).mantissa, quadratic_matrix.diagonal().mantissa)
        assert u.math.allclose(
            np.asarray(diagonal(quadratic_matrix).mantissa),
            diagonal(np.asarray(quadratic_matrix.mantissa))
        )
        assert u.math.allclose(
            np.diagonal(np.asarray(quadratic_matrix.mantissa)),
            diagonal(np.asarray(quadratic_matrix.mantissa)),
        )

        assert u.math.allclose(
            trace(quadratic_matrix).mantissa,
            quadratic_matrix.trace().mantissa
        )
        assert u.math.allclose(
            np.asarray(trace(quadratic_matrix).mantissa),
            trace(np.asarray(quadratic_matrix.mantissa))
        )
        assert u.math.allclose(
            np.trace(np.asarray(quadratic_matrix.mantissa)),
            trace(np.asarray(quadratic_matrix.mantissa))
        )

        assert u.math.allclose(
            dot(quadratic_matrix, quadratic_matrix).mantissa,
            quadratic_matrix.dot(quadratic_matrix).mantissa
        )
        assert u.math.allclose(
            np.asarray(dot(quadratic_matrix, quadratic_matrix).mantissa),
            dot(np.asarray(quadratic_matrix.mantissa),
                np.asarray(quadratic_matrix.mantissa)),
        )
        assert u.math.allclose(
            np.dot(np.asarray(quadratic_matrix.mantissa),
                   np.asarray(quadratic_matrix.mantissa)),
            dot(np.asarray(quadratic_matrix.mantissa),
                np.asarray(quadratic_matrix.mantissa)),
        )
        assert u.math.allclose(
            np.asarray(quadratic_matrix.prod().mantissa),
            np.asarray(quadratic_matrix.mantissa).prod()
        )
        assert u.math.allclose(
            np.asarray(quadratic_matrix.prod(axis=0).mantissa),
            np.asarray(quadratic_matrix.mantissa).prod(axis=0),
        )

        # Check for correct units
        assert have_same_dim(quadratic_matrix, ravel(quadratic_matrix))
        assert have_same_dim(quadratic_matrix, trace(quadratic_matrix))
        assert have_same_dim(quadratic_matrix, diagonal(quadratic_matrix))
        assert have_same_dim(
            quadratic_matrix[0] ** 2,
            dot(quadratic_matrix, quadratic_matrix)
        )
        assert have_same_dim(
            quadratic_matrix.prod(axis=0),
            quadratic_matrix[0] ** quadratic_matrix.shape[0]
        )

        # check the where function
        # pure numpy array
        cond = np.array([True, False, False])
        ar1 = np.array([1, 2, 3])
        ar2 = np.array([4, 5, 6])
        assert_equal(np.where(cond), where(cond))
        assert_equal(np.where(cond, ar1, ar2), where(cond, ar1, ar2))

        # dimensionless Array
        assert u.math.allclose(
            np.where(cond, ar1, ar2),
            np.asarray(where(cond, ar1 * mV / mV, ar2 * mV / mV))
        )

        # Array with dimensions
        ar1 = ar1 * mV
        ar2 = ar2 * mV
        assert u.math.allclose(
            np.where(cond, ar1.mantissa, ar2.mantissa),
            np.asarray(where(cond, ar1, ar2).mantissa),
        )

        # Check some error cases
        with pytest.raises(AssertionError):
            where(cond, ar1)
        with pytest.raises(TypeError):
            where(cond, ar1, ar1, ar2)
        with pytest.raises(u.UnitMismatchError):
            where(cond, ar1, ar1 / ms)

        # Check setasflat (for numpy < 1.7)
        if hasattr(Quantity, "setasflat"):
            a = np.arange(10) * mV
            b = np.ones(10).reshape(5, 2) * volt
            c = np.ones(10).reshape(5, 2) * second
            with pytest.raises(DimensionMismatchError):
                a.setasflat(c)
            a.setasflat(b)
            assert_equal(a.flatten(), b.flatten())

        # Check cumprod
        a = np.arange(1, 10) * mV / mV
        assert u.math.allclose(a.cumprod(), np.asarray(a).cumprod())
        (np.arange(1, 5) * mV).cumprod()

    def test_unit_discarding_functions(self):
        """
        Test functions that discard units.
        """

        values = [3 * mV, np.array([1, 2]) * mV, np.arange(12).reshape(3, 4) * mV]
        for a in values:
            assert np.allclose(np.sign(a.mantissa), np.sign(np.asarray(a.mantissa)))
            assert np.allclose(u.math.zeros_like(a).mantissa, np.zeros_like(np.asarray(a.mantissa)))
            assert np.allclose(u.math.ones_like(a).mantissa, np.ones_like(np.asarray(a.mantissa)))
            if a.ndim > 0:
                # Calling non-zero on a 0d array is deprecated, don't test it:
                assert np.allclose(np.nonzero(a.mantissa), np.nonzero(np.asarray(a.mantissa)))

    def test_numpy_functions_same_dimensions(self):
        values = [np.array([1, 2]), np.ones((3, 3))]
        units = [volt, second, siemens, mV, kHz]

        # Do not suopport numpy functions
        keep_dim_funcs = [
            # 'abs',
            'cumsum',
            'max',
            'mean',
            'min',
            # 'negative',
            'ptp',
            'round',
            'squeeze',
            'std',
            'sum',
            'transpose',
        ]

        for value, unit in itertools.product(values, units):
            q_ar = value * unit
            for func in keep_dim_funcs:
                test_ar = getattr(q_ar, func)()
                if u.get_unit(test_ar) != q_ar.unit:
                    raise AssertionError(
                        f"'{func.__name__}' failed on {q_ar!r} -- unit was "
                        f"{q_ar.unit}, is now {u.get_unit(test_ar)}."
                    )

        # Python builtins should work on one-dimensional arrays
        value = np.arange(5)
        builtins = [abs, max, min]
        for unit in units:
            q_ar = value * unit
            for func in builtins:
                test_ar = func(q_ar)
                if u.get_unit(test_ar) != q_ar.unit:
                    raise AssertionError(
                        f"'{func.__name__}' failed on {q_ar!r} -- unit "
                        f"was {q_ar.unit}, is now "
                        f"{u.get_unit(test_ar)}"
                    )

    def test_unitsafe_functions(self):
        """
        Test the unitsafe functions wrapping their numpy counterparts.
        """
        # All functions with their numpy counterparts
        funcs = [
            (u.math.sin, np.sin),
            (u.math.sinh, np.sinh),
            (u.math.arcsin, np.arcsin),
            (u.math.arcsinh, np.arcsinh),
            (u.math.cos, np.cos),
            (u.math.cosh, np.cosh),
            (u.math.arccos, np.arccos),
            (u.math.arccosh, np.arccosh),
            (u.math.tan, np.tan),
            (u.math.tanh, np.tanh),
            (u.math.arctan, np.arctan),
            (u.math.arctanh, np.arctanh),
            (u.math.log, np.log),
            (u.math.exp, np.exp),
        ]

        unitless_values = [0.1 * mV / mV, np.array([0.1, 0.5]) * mV / mV, np.random.rand(3, 3) * mV / mV]
        numpy_values = [0.1, np.array([0.1, 0.5]), np.random.rand(3, 3)]
        unit_values = [0.1 * mV, np.array([0.1, 0.5]) * mV, np.random.rand(3, 3) * mV]

        for bu_fun, np_fun in funcs:
            # make sure these functions raise errors when run on values with dimensions
            for val in unit_values:
                with pytest.raises(AssertionError):
                    bu_fun(val)

            for val in unitless_values:
                if hasattr(val, "mantissa"):
                    assert u.math.allclose(bu_fun(val.mantissa), np_fun(val.mantissa), equal_nan=True)
                else:
                    assert u.math.allclose(bu_fun(val), np_fun(val), equal_nan=True)

            for val in numpy_values:
                assert u.math.allclose(bu_fun(val), np_fun(val), equal_nan=True)


class TestHelperFunctions(unittest.TestCase):

    def test_fail_for_dimension_mismatch(self):
        """
        Test the fail_for_dimension_mismatch function.
        """
        # examples that should not raise an error
        dim1, dim2 = fail_for_dimension_mismatch(3)
        assert dim1 is DIMENSIONLESS
        assert dim2 is DIMENSIONLESS
        dim1, dim2 = fail_for_dimension_mismatch(3 * volt / volt)
        assert dim1 is DIMENSIONLESS
        assert dim2 is DIMENSIONLESS
        dim1, dim2 = fail_for_dimension_mismatch(3 * volt / volt, 7)
        assert dim1 is DIMENSIONLESS
        assert dim2 is DIMENSIONLESS
        dim1, dim2 = fail_for_dimension_mismatch(3 * volt, 5 * volt)
        assert dim1 is volt.dim
        assert dim2 is volt.dim

        # examples that should raise an error
        with pytest.raises(DimensionMismatchError):
            fail_for_dimension_mismatch(6 * volt)
        with pytest.raises(DimensionMismatchError):
            fail_for_dimension_mismatch(6 * volt, 5 * second)

    def test_check_dims(self):
        """
        Test the check_dims decorator
        """

        @u.check_dims(v=volt.dim)
        def a_function(v, x):
            """
            v has to have units of volt, x can have any (or no) unit.
            """
            pass

        # Try correct units
        a_function(3 * mV, 5 * second)
        a_function(5 * volt, "something")
        a_function([1, 2, 3] * volt, None)
        # lists that can be converted should also work
        a_function([1 * volt, 2 * volt, 3 * volt], None)
        # Strings and None are also allowed to pass
        a_function("a string", None)
        a_function(None, None)

        # Try incorrect units
        with pytest.raises(DimensionMismatchError):
            a_function(5 * second, None)
        with pytest.raises(DimensionMismatchError):
            a_function(5, None)
        with pytest.raises(DimensionMismatchError):
            a_function(object(), None)
        with pytest.raises(TypeError):
            a_function([1, 2 * volt, 3], None)

        @u.check_dims(result=second.dim)
        def b_function(return_second):
            """
            Return a value in seconds if return_second is True, otherwise return
            a value in volt.
            """
            if return_second:
                return 5 * second
            else:
                return 3 * volt

        # Should work (returns second)
        b_function(True)
        # Should fail (returns volt)
        with pytest.raises(DimensionMismatchError):
            b_function(False)

        @u.check_dims(a=bool, b=1, result=bool)
        def c_function(a, b):
            if a:
                return b > 0
            else:
                return b

        assert c_function(True, 1)
        assert not c_function(True, -1)
        with pytest.raises(TypeError):
            c_function(1, 1)
        with pytest.raises(TypeError):
            c_function(1 * mV, 1)

        # with pytest.raises(TypeError):
        #     c_function(False, 1)

        # Multiple results
        @u.check_dims(result=(second.dim, volt.dim))
        def d_function(true_result):
            """
            Return a value in seconds if return_second is True, otherwise return
            a value in volt.
            """
            if true_result:
                return 5 * second, 3 * volt
            else:
                return 3 * volt, 5 * second

        # Should work (returns second)
        d_function(True)
        # Should fail (returns volt)
        with pytest.raises(u.DimensionMismatchError):
            d_function(False)

        # Multiple results
        @u.check_dims(result={'u': second.dim, 'v': (volt.dim, metre.dim)})
        def d_function2(true_result):
            """
            Return a value in seconds if return_second is True, otherwise return
            a value in volt.
            """
            if true_result == 0:
                return {'u': 5 * second, 'v': (3 * volt, 2 * metre)}
            elif true_result == 1:
                return 3 * volt, 5 * second
            else:
                return {'u': 5 * second, 'v': (3 * volt, 2 * volt)}

        d_function2(0)

        with pytest.raises(TypeError):
            d_function2(1)

        with pytest.raises(u.DimensionMismatchError):
            d_function2(2)

    def test_check_units(self):
        """
        Test the check_units decorator
        """

        @u.check_units(v=volt)
        def a_function(v, x):
            """
            v has to have units of volt, x can have any (or no) unit.
            """
            pass

        # Try correct units

        with pytest.raises(u.UnitMismatchError):
            a_function(3 * mV, 5 * second)
        a_function(3 * volt, 5 * second)
        a_function(5 * volt, "something")
        a_function([1, 2, 3] * volt, None)
        # lists that can be converted should also work
        a_function([1 * volt, 2 * volt, 3 * volt], None)
        # Strings and None are also allowed to pass
        a_function("a string", None)
        a_function(None, None)

        # Try incorrect units
        with pytest.raises(u.UnitMismatchError):
            a_function(5 * second, None)
        with pytest.raises(u.UnitMismatchError):
            a_function(5, None)
        with pytest.raises(u.UnitMismatchError):
            a_function(object(), None)
        with pytest.raises(TypeError):
            a_function([1, 2 * volt, 3], None)

        @check_units(result=second)
        def b_function(return_second):
            """
            Return a value in seconds if return_second is True, otherwise return
            a value in volt.
            """
            if return_second:
                return 5 * second
            else:
                return 3 * volt

        # Should work (returns second)
        b_function(True)
        # Should fail (returns volt)
        with pytest.raises(u.UnitMismatchError):
            b_function(False)

        @check_units(a=bool, b=1, result=bool)
        def c_function(a, b):
            if a:
                return b > 0
            else:
                return b

        assert c_function(True, 1)
        assert not c_function(True, -1)
        with pytest.raises(TypeError):
            c_function(1, 1)
        with pytest.raises(TypeError):
            c_function(1 * mV, 1)

        # Multiple results
        @check_units(result=(second, volt))
        def d_function(true_result):
            """
            Return a value in seconds if return_second is True, otherwise return
            a value in volt.
            """
            if true_result:
                return 5 * second, 3 * volt
            else:
                return 3 * volt, 5 * second

        # Should work (returns second)
        d_function(True)
        # Should fail (returns volt)
        with pytest.raises(u.UnitMismatchError):
            d_function(False)

        # Multiple results
        @check_units(result={'u': second, 'v': (volt, metre)})
        def d_function2(true_result):
            """
            Return a value in seconds if return_second is True, otherwise return
            a value in volt.
            """
            if true_result == 0:
                return {'u': 5 * second, 'v': (3 * volt, 2 * metre)}
            elif true_result == 1:
                return 3 * volt, 5 * second
            else:
                return {'u': 5 * second, 'v': (3 * volt, 2 * volt)}

        # Should work (returns second)
        d_function2(0)
        # Should fail (returns volt)
        with pytest.raises(TypeError):
            d_function2(1)

        with pytest.raises(u.UnitMismatchError):
            d_function2(2)

    def test_assign_units(self):
        """
        Test the assign_units decorator
        """

        @u.assign_units(v=volt)
        def a_function(v, x):
            """
            v has to have units of volt, x can have any (or no) unit.
            """
            return v

        # Try correct units
        assert a_function(3 * mV, 5 * second) == (3 * mV).to_decimal(volt)
        assert a_function(3 * volt, 5 * second) == (3 * volt).to_decimal(volt)
        assert a_function(5 * volt, "something") == (5 * volt).to_decimal(volt)
        assert_quantity(a_function([1, 2, 3] * volt, None), ([1, 2, 3] * volt).to_decimal(volt))

        # Try incorrect units
        with pytest.raises(u.UnitMismatchError):
            a_function(5 * second, None)
        with pytest.raises(TypeError):
            a_function(5, None)
        with pytest.raises(TypeError):
            a_function(object(), None)

        @u.assign_units(result=second)
        def b_function():
            """
            Return a value in seconds if return_second is True, otherwise return
            a value in volt.
            """
            return 5

        # Should work (returns second)
        assert b_function() == 5 * second

        @u.assign_units(a=bool, b=1, result=bool)
        def c_function(a, b):
            if a:
                return b > 0
            else:
                return b

        assert c_function(True, 1)
        assert not c_function(True, -1)
        with pytest.raises(TypeError):
            c_function(1, 1)
        with pytest.raises(TypeError):
            c_function(1 * mV, 1)

        # Multiple results
        @u.assign_units(result=(second, volt))
        def d_function():
            return 5, 3

        # Should work (returns second)
        assert d_function()[0] == 5 * second
        assert d_function()[1] == 3 * volt

        # Multiple results
        @u.assign_units(result={'u': second, 'v': (volt, metre)})
        def d_function2(true_result):
            """
            Return a value in seconds if return_second is True, otherwise return
            a value in volt.
            """
            if true_result == 0:
                return {'u': 5, 'v': (3, 2)}
            elif true_result == 1:
                return 3, 5
            else:
                return 3, 5

        # Should work (returns dict)
        d_function2(0)
        # Should fail (returns tuple)
        with pytest.raises(TypeError):
            d_function2(1)



def test_str_repr():
    """
    Test that str representations do not raise any errors and that repr
    fullfills eval(repr(x)) == x.
    """

    units_which_should_exist = [
        u.metre,
        u.meter,
        u.kilogram,
        u.kilogramme,
        u.second,
        u.amp,
        u.kelvin,
        u.mole,
        u.candle,
        u.radian,
        u.steradian,
        u.hertz,
        u.newton,
        u.pascal,
        u.joule,
        u.watt,
        u.coulomb,
        u.volt,
        u.farad,
        u.ohm,
        u.siemens,
        u.weber,
        u.tesla,
        u.henry,
        u.lumen,
        u.lux,
        u.becquerel,
        u.gray,
        u.sievert,
        u.katal,
        u.gram,
        u.gramme,
        u.molar,
        u.liter,
        u.litre,
    ]

    # scaled versions of all these units should exist (we just check farad as an example)
    some_scaled_units = [
        u.Yfarad,
        u.Zfarad,
        u.Efarad,
        u.Pfarad,
        u.Tfarad,
        u.Gfarad,
        u.Mfarad,
        u.kfarad,
        u.hfarad,
        u.dafarad,
        u.dfarad,
        u.cfarad,
        u.mfarad,
        u.ufarad,
        u.nfarad,
        u.pfarad,
        u.ffarad,
        u.afarad,
        u.zfarad,
        u.yfarad,
    ]

    # some powered units
    powered_units = [u.cmetre2, u.Yfarad3]

    # Combined units
    complex_units = [
        (u.kgram * u.metre2) / (u.amp * u.second3),
        5 * (u.kgram * u.metre2) / (u.amp * u.second3),
        u.metre * u.second ** -1,
        10 * u.metre * u.second ** -1,
        np.array([1, 2, 3]) * u.kmetre / u.second,
        np.ones(3) * u.nS / u.cm ** 2,
        # Made-up unit:
        Unit(
            dim=get_or_create_dimension(length=5, time=2),
            dispname="O",
        ),
        8000 * u.umetre ** 3,
        [0.0001, 10000] * u.umetre ** 3,
        1 / u.metre,
        1 / (u.coulomb * u.metre ** 2),
        Unit() / second,
        3.0 * u.mM,
        5 * u.mole / u.liter,
        7 * u.liter / u.meter3,
        1 / second ** 2,
        volt ** -2,
        (volt ** 2) ** -1,
        (1 / second) / meter,
        1 / (1 / second),
    ]

    unitless = [second / second, 5 * second / second, Unit()]
    #
    # for u in itertools.chain(
    #     units_which_should_exist,
    #     some_scaled_units,
    #     powered_units,
    #     complex_units,
    #     unitless,
    # ):
    #   assert len(str(u)) > 0
    #   print(u)
    #   v1 = u.display_in_unit(u, python_code=False)
    #   if isinstance(u, Unit):
    #     if 'Unit(1.0)' in v1:
    #       continue
    #     v2 = eval(v1)
    #     assert v2 == u
    #     assert isinstance(u, Unit)
    #     assert u.math.allclose(v2.value, u.value)

    # test the `DIMENSIONLESS` object
    assert str(DIMENSIONLESS) == "1"
    assert repr(DIMENSIONLESS) == "Dimension()"

    # test DimensionMismatchError (only that it works without raising an error
    for error in [
        DimensionMismatchError("A description"),
        DimensionMismatchError("A description", DIMENSIONLESS),
        DimensionMismatchError("A description", DIMENSIONLESS, second.dim),
    ]:
        assert len(str(error))
        assert len(repr(error))


class TestGetMethod(unittest.TestCase):
    def test_get_dim(self):
        assert u.get_dim(1) == u.DIMENSIONLESS
        assert u.get_dim(1.0) == u.DIMENSIONLESS
        assert u.get_dim(1 * u.mV) == u.volt.dim
        assert u.get_dim(1 * u.mV / u.mV) == u.DIMENSIONLESS
        assert u.get_dim(1 * u.mV / u.second) == u.volt.dim / u.second.dim
        assert u.get_dim(1 * u.mV / u.second ** 2) == u.volt.dim / u.second.dim ** 2
        assert u.get_dim(1 * u.mV ** 2 / u.second ** 2) == u.volt.dim ** 2 / u.second.dim ** 2

        assert u.get_dim(object()) == u.DIMENSIONLESS
        assert u.get_dim("string") == u.DIMENSIONLESS
        assert u.get_dim([1, 2, 3]) == u.DIMENSIONLESS
        assert u.get_dim(np.array([1, 2, 3])) == u.DIMENSIONLESS
        assert u.get_dim(np.array([1, 2, 3]) * u.mV) == u.volt.dim

        assert u.get_dim(u.mV) == u.volt.dim
        assert u.get_dim(u.mV / u.mV) == u.DIMENSIONLESS
        assert u.get_dim(u.mV / u.second) == u.volt.dim / u.second.dim
        assert u.get_dim(u.mV / u.second ** 2) == u.volt.dim / u.second.dim ** 2
        assert u.get_dim(u.mV ** 2 / u.second ** 2) == u.volt.dim ** 2 / u.second.dim ** 2

        assert u.get_dim(u.mV.dim) == u.volt.dim
        assert u.get_dim(u.mV.dim / u.mV.dim) == u.DIMENSIONLESS
        assert u.get_dim(u.mV.dim / u.second.dim) == u.volt.dim / u.second.dim
        assert u.get_dim(u.mV.dim / u.second.dim ** 2) == u.volt.dim / u.second.dim ** 2
        assert u.get_dim(u.mV.dim ** 2 / u.second.dim ** 2) == u.volt.dim ** 2 / u.second.dim ** 2

    def test_unit(self):
        assert u.get_unit(1) == u.UNITLESS
        assert u.get_unit(1.0) == u.UNITLESS
        assert u.get_unit(1 * u.mV) == u.mV
        assert u.get_unit(1 * u.mV / u.mV) == u.UNITLESS
        assert u.get_unit(1 * u.mV / u.second) == u.mV / u.second
        assert u.get_unit(1 * u.mV / u.second ** 2) == u.mV / u.second ** 2
        assert u.get_unit(1 * u.mV ** 2 / u.second ** 2) == u.mV ** 2 / u.second ** 2

        assert u.get_unit(object()) == u.UNITLESS
        assert u.get_unit("string") == u.UNITLESS
        assert u.get_unit([1, 2, 3]) == u.UNITLESS
        assert u.get_unit(np.array([1, 2, 3])) == u.UNITLESS
        assert u.get_unit(np.array([1, 2, 3]) * u.mV) == u.mV

        assert u.get_unit(u.mV) == u.mV
        assert u.get_unit(u.mV / u.mV) == u.UNITLESS
        assert u.get_unit(u.mV / u.second) == u.mV / u.second
        assert u.get_unit(u.mV / u.second ** 2) == u.mV / u.second ** 2
        assert u.get_unit(u.mV ** 2 / u.second ** 2) == u.mV ** 2 / u.second ** 2

        assert u.get_unit(u.mV.dim) == u.UNITLESS
        assert u.get_unit(u.mV.dim / u.mV.dim) == u.UNITLESS
        assert u.get_unit(u.mV.dim / u.second.dim) == u.UNITLESS
        assert u.get_unit(u.mV.dim / u.second.dim ** 2) == u.UNITLESS
        assert u.get_unit(u.mV.dim ** 2 / u.second.dim ** 2) == u.UNITLESS

    def test_get_mantissa(self):
        assert u.get_mantissa(1) == 1
        assert u.get_mantissa(1.0) == 1.0
        assert u.get_mantissa(1 * u.mV) == 1
        assert u.get_mantissa(1 * u.mV / u.mV) == 1
        assert u.get_mantissa(1 * u.mV / u.second) == 1
        assert u.get_mantissa(1 * u.mV / u.second ** 2) == 1
        assert u.get_mantissa(1 * u.mV ** 2 / u.second ** 2) == 1

        obj = object()
        assert u.get_mantissa(obj) == obj
        assert u.get_mantissa("string") == "string"
        assert u.get_mantissa([1, 2, 3]) == [1, 2, 3]
        assert np.allclose(u.get_mantissa(np.array([1, 2, 3])), np.array([1, 2, 3]))
        assert np.allclose(u.get_mantissa(np.array([1, 2, 3]) * u.mV), np.array([1, 2, 3]))

        assert u.get_mantissa(u.mV) == u.mV
        assert u.get_mantissa(u.mV / u.mV) == u.mV / u.mV
        assert u.get_mantissa(u.mV / u.second) == u.mV / u.second
        assert u.get_mantissa(u.mV / u.second ** 2) == u.mV / u.second ** 2
        assert u.get_mantissa(u.mV ** 2 / u.second ** 2) == u.mV ** 2 / u.second ** 2

        assert u.get_mantissa(u.mV.dim) == u.mV.dim
        assert u.get_mantissa(u.mV.dim / u.mV.dim) == u.mV.dim / u.mV.dim
        assert u.get_mantissa(u.mV.dim / u.second.dim) == u.mV.dim / u.second.dim
        assert u.get_mantissa(u.mV.dim / u.second.dim ** 2) == u.mV.dim / u.second.dim ** 2
        assert u.get_mantissa(u.mV.dim ** 2 / u.second.dim ** 2) == u.mV.dim ** 2 / u.second.dim ** 2

    def test_format(self):
        with bst.environ.context(precision=64):
            q1 = 1.23456789 * u.mV
            assert f"{q1:.2f}" == "1.23 * mvolt"
            assert f"{q1:.3f}" == "1.235 * mvolt"
            assert f"{q1:.4f}" == "1.2346 * mvolt"

            q2 = [1.23456789, 1.23456789] * u.mV
            assert f"{q2:.2f}" == "ArrayImpl([1.23, 1.23]) * mvolt"
            assert f"{q2:.3f}" == "ArrayImpl([1.235, 1.235]) * mvolt"
            assert f"{q2:.4f}" == "ArrayImpl([1.2346, 1.2346]) * mvolt"
