"""Tests for _base_quantity.py: Quantity class, wrapping functions, list processing, pickle."""

from __future__ import annotations

import itertools
import os
import pickle
import tempfile
import warnings
from copy import deepcopy
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_equal

import saiunit as u
from saiunit._base_dimension import (
    DIMENSIONLESS,
    DimensionMismatchError,
    UnitMismatchError,
    get_or_create_dimension,
)
from saiunit._base_quantity import (
    Quantity,
    StaticScalar,
    _all_slice,
    _check_units_and_collect_values,
    _element_not_quantity,
    _process_list_with_units,
    _quantity_with_unit,
    _replace_with_array,
    _wrap_function_change_unit,
    _wrap_function_keep_unit,
    _wrap_function_remove_unit,
    _zoom_values_with_units,
    compat_with_equinox,
    compatible_with_equinox,
)
from saiunit._base_unit import UNITLESS, Unit
from saiunit._base_getters import (
    assert_quantity,
    display_in_unit,
    have_same_dim,
    is_scalar_type,
)
from saiunit._unit_common import *
from saiunit._unit_shortcuts import kHz, ms, mV, nS


# helpers: build common units once
_length_dim = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
_time_dim = get_or_create_dimension([0, 0, 1, 0, 0, 0, 0])
_mass_dim = get_or_create_dimension([0, 1, 0, 0, 0, 0, 0])
_metre = Unit(_length_dim, name="metre", dispname="m", scale=0)
_kmetre = Unit(_length_dim, name="kmetre", dispname="km", scale=3)
_second = Unit(_time_dim, name="second", dispname="s", scale=0)
_kg = Unit(_mass_dim, name="kilogram", dispname="kg", scale=0)


# =========================================================================
# Quantity construction
# =========================================================================

class TestQuantityConstruction:
    def test_scalar_no_unit(self):
        q = Quantity(5.0)
        assert q.is_unitless
        assert jnp.allclose(q.mantissa, 5.0)

    def test_scalar_with_unit(self):
        q = Quantity(5.0, unit=_metre)
        assert q.unit == _metre
        assert jnp.allclose(q.mantissa, 5.0)

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        q = Quantity(arr, unit=_metre)
        assert q.shape == (3,)

    def test_jax_array(self):
        arr = jnp.array([1.0, 2.0])
        q = Quantity(arr, unit=_metre)
        assert q.shape == (2,)

    def test_from_unit_object(self):
        q = Quantity(_metre)
        assert q.unit == _metre
        assert jnp.allclose(q.mantissa, 1.0)

    def test_from_unit_with_extra_unit_raises(self):
        with pytest.raises(ValueError, match="Cannot create"):
            Quantity(_metre, unit=_second)

    def test_from_quantity(self):
        q1 = Quantity(5.0, unit=_metre)
        q2 = Quantity(q1)
        assert q2.unit == _metre
        assert jnp.allclose(q2.mantissa, 5.0)

    def test_from_quantity_different_dim_raises(self):
        q1 = Quantity(5.0, unit=_metre)
        with pytest.raises(ValueError, match="different unit"):
            Quantity(q1, unit=_second)

    def test_from_list(self):
        q = Quantity([1.0, 2.0, 3.0], unit=_metre)
        assert q.shape == (3,)

    def test_from_list_with_units(self):
        q = Quantity([Quantity(1.0, unit=_metre), Quantity(2.0, unit=_metre)])
        assert q.unit == _metre
        assert q.shape == (2,)

    def test_with_dtype(self):
        q = Quantity(5.0, unit=_metre, dtype=jnp.float32)
        assert q.dtype == jnp.float32


# =========================================================================
# Quantity properties
# =========================================================================

class TestQuantityProperties:
    def test_mantissa(self):
        q = Quantity(5.0, unit=_metre)
        assert jnp.allclose(q.mantissa, 5.0)

    def test_unit(self):
        q = Quantity(5.0, unit=_metre)
        assert q.unit == _metre

    def test_dim(self):
        q = Quantity(5.0, unit=_metre)
        assert q.dim == _length_dim

    def test_shape(self):
        q = Quantity(np.array([1.0, 2.0, 3.0]), unit=_metre)
        assert q.shape == (3,)

    def test_ndim(self):
        q = Quantity(np.array([[1.0, 2.0], [3.0, 4.0]]), unit=_metre)
        assert q.ndim == 2

    def test_size(self):
        q = Quantity(np.array([1.0, 2.0, 3.0]), unit=_metre)
        assert q.size == 3

    def test_dtype(self):
        q = Quantity(np.array([1.0], dtype=np.float32), unit=_metre)
        assert q.dtype == np.float32

    def test_is_unitless_false(self):
        q = Quantity(5.0, unit=_metre)
        assert not q.is_unitless

    def test_is_unitless_true(self):
        q = Quantity(5.0)
        assert q.is_unitless


# =========================================================================
# Quantity arithmetic
# =========================================================================

class TestQuantityArithmetic:
    def test_add_same_unit(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_metre)
        result = q1 + q2
        assert jnp.allclose(result.mantissa, 3.0)
        assert result.unit == _metre

    def test_add_different_dim_raises(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_second)
        with pytest.raises((DimensionMismatchError, UnitMismatchError)):
            q1 + q2

    def test_sub_same_unit(self):
        q1 = Quantity(3.0, unit=_metre)
        q2 = Quantity(1.0, unit=_metre)
        result = q1 - q2
        assert jnp.allclose(result.mantissa, 2.0)

    def test_mul_quantities(self):
        q1 = Quantity(2.0, unit=_metre)
        q2 = Quantity(3.0, unit=_second)
        result = q1 * q2
        assert result.dim == _length_dim * _time_dim

    def test_mul_scalar(self):
        q = Quantity(2.0, unit=_metre)
        result = q * 3.0
        assert jnp.allclose(result.mantissa, 6.0)
        assert result.unit == _metre

    def test_rmul_scalar(self):
        q = Quantity(2.0, unit=_metre)
        result = 3.0 * q
        assert jnp.allclose(result.mantissa, 6.0)

    def test_div_quantities(self):
        q1 = Quantity(6.0, unit=_metre)
        q2 = Quantity(2.0, unit=_second)
        result = q1 / q2
        assert result.dim == _length_dim / _time_dim

    def test_div_scalar(self):
        q = Quantity(6.0, unit=_metre)
        result = q / 2.0
        assert jnp.allclose(result.mantissa, 3.0)

    def test_pow(self):
        q = Quantity(3.0, unit=_metre)
        result = q ** 2
        assert jnp.allclose(result.mantissa, 9.0)
        assert result.dim == _length_dim ** 2

    def test_neg(self):
        q = Quantity(3.0, unit=_metre)
        result = -q
        assert jnp.allclose(result.mantissa, -3.0)
        assert result.unit == _metre

    def test_pos(self):
        q = Quantity(3.0, unit=_metre)
        result = +q
        assert jnp.allclose(result.mantissa, 3.0)

    def test_abs(self):
        q = Quantity(-3.0, unit=_metre)
        result = abs(q)
        assert jnp.allclose(result.mantissa, 3.0)


# =========================================================================
# Quantity comparison
# =========================================================================

class TestQuantityComparison:
    def test_eq(self):
        q1 = Quantity(3.0, unit=_metre)
        q2 = Quantity(3.0, unit=_metre)
        assert (q1 == q2)

    def test_ne(self):
        q1 = Quantity(3.0, unit=_metre)
        q2 = Quantity(4.0, unit=_metre)
        assert (q1 != q2)

    def test_lt(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_metre)
        assert (q1 < q2)

    def test_le(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(1.0, unit=_metre)
        assert (q1 <= q2)

    def test_gt(self):
        q1 = Quantity(2.0, unit=_metre)
        q2 = Quantity(1.0, unit=_metre)
        assert (q1 > q2)

    def test_ge(self):
        q1 = Quantity(2.0, unit=_metre)
        q2 = Quantity(2.0, unit=_metre)
        assert (q1 >= q2)

    def test_comparison_different_dim_raises(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(1.0, unit=_second)
        with pytest.raises((DimensionMismatchError, UnitMismatchError)):
            q1 < q2


# =========================================================================
# Quantity slicing / indexing
# =========================================================================

class TestQuantityIndexing:
    def test_getitem(self):
        q = Quantity(np.array([1.0, 2.0, 3.0]), unit=_metre)
        result = q[1]
        assert jnp.allclose(result.mantissa, 2.0)
        assert result.unit == _metre

    def test_getitem_slice(self):
        q = Quantity(np.array([1.0, 2.0, 3.0]), unit=_metre)
        result = q[1:]
        assert result.shape == (2,)
        assert result.unit == _metre

    def test_len(self):
        q = Quantity(np.array([1.0, 2.0, 3.0]), unit=_metre)
        assert len(q) == 3


# =========================================================================
# Quantity shape manipulation
# =========================================================================

class TestQuantityShapeManipulation:
    def test_reshape(self):
        q = Quantity(np.array([1.0, 2.0, 3.0, 4.0]), unit=_metre)
        r = q.reshape((2, 2))
        assert r.shape == (2, 2)
        assert r.unit == _metre

    def test_flatten(self):
        q = Quantity(np.array([[1.0, 2.0], [3.0, 4.0]]), unit=_metre)
        r = q.flatten()
        assert r.shape == (4,)

    def test_squeeze(self):
        q = Quantity(np.array([[[1.0, 2.0]]]), unit=_metre)
        r = q.squeeze()
        assert r.shape == (2,)

    def test_transpose(self):
        q = Quantity(np.array([[1.0, 2.0], [3.0, 4.0]]), unit=_metre)
        r = q.T
        assert r.shape == (2, 2)
        assert jnp.allclose(r[0, 1].mantissa, 3.0)


# =========================================================================
# Quantity conversion
# =========================================================================

class TestQuantityConversion:
    def test_to_decimal_dimensionless(self):
        q = Quantity(5.0, unit=UNITLESS)
        result = q.to_decimal()
        assert jnp.allclose(result, 5.0)

    def test_to_decimal_with_unit(self):
        q = Quantity(1.0, unit=_kmetre)
        result = q.to_decimal(_metre)
        assert jnp.allclose(result, 1000.0)

    def test_in_unit(self):
        q = Quantity(1.0, unit=_kmetre)
        result = q.in_unit(_metre)
        assert jnp.allclose(result.mantissa, 1000.0)
        assert result.unit == _metre

    def test_factorless(self):
        q = Quantity(5.0, unit=_metre)
        result = q.factorless()
        assert result.unit.factor == 1.0


# =========================================================================
# Quantity display
# =========================================================================

class TestQuantityDisplay:
    def test_str(self):
        q = Quantity(5.0, unit=_metre)
        s = str(q)
        assert "5." in s
        assert "m" in s

    def test_repr(self):
        q = Quantity(5.0, unit=_metre)
        r = repr(q)
        assert "Quantity" in r or "5." in r

    def test_repr_in_unit(self):
        q = Quantity(1.0, unit=_kmetre)
        s = q.repr_in_unit()
        assert "1." in s


# =========================================================================
# Quantity deepcopy
# =========================================================================

class TestQuantityDeepCopy:
    def test_deepcopy(self):
        q = Quantity(np.array([1.0, 2.0]), unit=_metre)
        q2 = deepcopy(q)
        assert jnp.allclose(q.mantissa, q2.mantissa)
        assert q.unit == q2.unit


# =========================================================================
# Quantity pickle
# =========================================================================

class TestQuantityPickle:
    def test_pickle_roundtrip(self):
        q = Quantity(5.0, unit=_metre)
        data = pickle.dumps(q)
        q2 = pickle.loads(data)
        assert jnp.allclose(q.mantissa, q2.mantissa)
        assert q.unit == q2.unit

    def test_pickle_file_roundtrip(self):
        q = Quantity(np.array([1.0, 2.0, 3.0]), unit=_metre)
        tmpdir = tempfile.gettempdir()
        filename = os.path.join(tmpdir, "test_quantity.pkl")
        with open(filename, "wb") as f:
            pickle.dump(q, f)
        with open(filename, "rb") as f:
            q2 = pickle.load(f)
        assert jnp.allclose(q.mantissa, q2.mantissa)
        assert q.unit == q2.unit

    def test_quantity_with_unit_helper(self):
        q = _quantity_with_unit(5.0, _metre)
        assert isinstance(q, Quantity)
        assert q.unit == _metre


# =========================================================================
# JAX pytree
# =========================================================================

class TestQuantityPytree:
    def test_tree_flatten_unflatten(self):
        q = Quantity(5.0, unit=_metre)
        leaves, treedef = jax.tree.flatten(q)
        q2 = treedef.unflatten(leaves)
        assert jnp.allclose(q.mantissa, q2.mantissa)
        assert q.unit == q2.unit

    def test_jit_passthrough(self):
        @jax.jit
        def f(q):
            return q * 2

        q = Quantity(jnp.array(3.0), unit=_metre)
        result = f(q)
        assert jnp.allclose(result.mantissa, 6.0)
        assert result.unit == _metre


# =========================================================================
# Wrapping functions
# =========================================================================

class TestWrappingFunctions:
    def test_wrap_keep_unit(self):
        wrapped = _wrap_function_keep_unit(jnp.abs)
        q = Quantity(jnp.array([-1.0, 2.0, -3.0]), unit=_metre)
        result = wrapped(q)
        assert jnp.allclose(result.mantissa, jnp.array([1.0, 2.0, 3.0]))
        assert result.unit == _metre

    def test_wrap_remove_unit(self):
        wrapped = _wrap_function_remove_unit(jnp.sign)
        q = Quantity(jnp.array([-1.0, 2.0, -3.0]), unit=_metre)
        result = wrapped(q)
        assert jnp.allclose(result, jnp.array([-1.0, 1.0, -1.0]))
        assert not isinstance(result, Quantity)

    def test_wrap_change_unit(self):
        def sqrt_unit(u1, u2):
            return u1 ** 0.5

        wrapped = _wrap_function_change_unit(jnp.sqrt, sqrt_unit)
        q = Quantity(jnp.array([4.0, 9.0]), unit=_metre ** 2)
        result = wrapped(q)
        assert jnp.allclose(result.mantissa, jnp.array([2.0, 3.0]))


# =========================================================================
# List processing helpers
# =========================================================================

class TestListProcessing:
    def test_zoom_values_same_scale(self):
        values = [1.0, 2.0]
        units = [_metre, _metre]
        result = _zoom_values_with_units(values, units)
        assert result == [1.0, 2.0]

    def test_check_units_and_collect_all_scalar(self):
        values, unit = _check_units_and_collect_values([1.0, 2.0, 3.0])
        assert unit == UNITLESS

    def test_check_units_and_collect_with_quantities(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_metre)
        values, unit = _check_units_and_collect_values([q1, q2])
        assert unit == _metre

    def test_check_units_mixed_raises(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_second)
        with pytest.raises(TypeError, match="same units"):
            _check_units_and_collect_values([q1, q2])

    def test_process_list_with_units(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_metre)
        values, unit = _process_list_with_units([q1, q2])
        assert unit == _metre

    def test_element_not_quantity_passes(self):
        assert _element_not_quantity(5.0) == 5.0

    def test_element_not_quantity_raises(self):
        q = Quantity(5.0, unit=_metre)
        with pytest.raises(AssertionError):
            _element_not_quantity(q)


# =========================================================================
# _replace_with_array
# =========================================================================

class TestReplaceWithArray:
    def test_replaces_scalar(self):
        result = _replace_with_array(5.0, _metre)
        assert isinstance(result, Quantity)
        assert result.unit == _metre

    def test_replaces_list(self):
        result = _replace_with_array([1.0, 2.0, 3.0], _metre)
        assert isinstance(result, list)
        assert all(isinstance(r, Quantity) for r in result)
        assert all(r.unit == _metre for r in result)


# =========================================================================
# compatible_with_equinox
# =========================================================================

class TestCompatibleWithEquinox:
    def test_default_false(self):
        # Reset to default
        compatible_with_equinox(False)
        from saiunit._base_quantity import compat_with_equinox
        assert not compat_with_equinox

    def test_set_true(self):
        compatible_with_equinox(True)
        from saiunit._base_quantity import compat_with_equinox
        assert compat_with_equinox
        # Reset
        compatible_with_equinox(False)


# =========================================================================
# Type aliases
# =========================================================================

class TestTypeAliases:
    def test_all_slice(self):
        assert _all_slice == slice(None, None, None)

    def test_static_scalar_includes_int(self):
        assert isinstance(5, StaticScalar)

    def test_static_scalar_includes_float(self):
        assert isinstance(5.0, StaticScalar)

    def test_static_scalar_includes_bool(self):
        assert isinstance(True, StaticScalar)

    def test_static_scalar_includes_numpy(self):
        assert isinstance(np.float64(5.0), StaticScalar)


# =========================================================================
# CustomArray helper for tests
# =========================================================================

@jax.tree_util.register_pytree_node_class
class Array(u.CustomArray):
    def __init__(self, value):
        self.data = value


# =========================================================================
# Quantity integration tests (from _base_test.py TestQuantity)
# =========================================================================

class TestQuantityIntegration:
    """Extended Quantity tests covering methods, display, operations, etc."""

    def test_dim_setter_raises(self):
        a = [1, 2.] * u.ms
        with pytest.raises(NotImplementedError):
            a.dim = u.mV.dim

    def test_clip(self):
        a = [1, 2.] * u.ms
        assert u.math.allclose(a.clip(1.5 * u.ms, 2.5 * u.ms), [1.5, 2.] * u.ms)

        b = Quantity([1, 2.])
        assert u.math.allclose(b.clip(1.5, 2.5), u.math.asarray([1.5, 2.]))

    def test_round(self):
        for unit in [u.ms, u.joule, u.mV]:
            a = [1.1, 2.2] * unit
            assert u.math.allclose(a.round(), [1, 2] * unit)

        b = Quantity([1.1, 2.2])
        assert u.math.allclose(b.round(), u.math.asarray([1, 2]))

    def test_astype(self):
        a = [1, 2.] * u.ms
        assert a.astype(jnp.float16).dtype == jnp.float16

    def test___array__(self):
        a = Quantity([1, 2.])
        assert u.math.allclose(np.asarray(a), np.asarray([1, 2.]))

        with pytest.raises(TypeError):
            a = [1, 2.] * u.ms
            np.asarray(a)

    def test__float__(self):
        a = Quantity(1.)
        assert u.math.allclose(float(a), 1.)

        a = Quantity([1, 2.])
        with pytest.raises(TypeError):
            float(a)

        with pytest.raises(TypeError):
            a = [1, 2.] * u.ms
            float(a)

    def test_construction_extended(self):
        """Test the construction of Quantity objects with unit conversions."""
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
        b = Quantity(a)
        assert u.math.allclose(a, b)

        c = Quantity(a, unit=u.volt)
        assert u.math.allclose(c.mantissa, np.asarray([1, 2, 3]) * 1e-3)
        assert u.math.allclose(c, a)

    def test_get_dimensions(self):
        """Test various ways of getting/comparing the dimensions of a Quantity."""
        q = 500 * ms
        assert get_or_create_dimension(q.dim._dims) == q.dim
        assert q.dim is q.dim
        assert q.has_same_unit(3 * second)
        dims = q.dim
        assert_equal(dims.get_dimension("time"), 1.0)
        assert_equal(dims.get_dimension("length"), 0)

        assert u.get_dim(5) is DIMENSIONLESS
        assert u.get_dim(5.0) is DIMENSIONLESS
        assert u.get_dim(np.array(5, dtype=np.int32)) is DIMENSIONLESS
        assert u.get_dim(np.array(5.0)) is DIMENSIONLESS
        assert u.get_dim(np.float32(5.0)) is DIMENSIONLESS
        assert u.get_dim(np.float64(5.0)) is DIMENSIONLESS
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
        """Test displaying a Quantity in different units."""
        assert_equal(display_in_unit(3. * volt, mvolt), "3000. mV")
        assert_equal(display_in_unit(10. * mV, ohm * amp), "0.01 A * ohm")
        with pytest.raises(u.UnitMismatchError):
            display_in_unit(10 * nS, ohm)

        brainstate = pytest.importorskip("brainstate")
        with brainstate.environ.context(precision=32):
            assert_equal(display_in_unit(3. * volt, mvolt), "3000. mV")
            assert_equal(display_in_unit(10. * mV, ohm * amp), "0.01 A * ohm")
            with pytest.raises(u.UnitMismatchError):
                display_in_unit(10 * nS, ohm)

        assert_equal(display_in_unit(10.0, Unit(scale=1)), "1. 10.0^1")
        assert_equal(str(3 * u.kmeter / u.meter), '3000.0')
        assert_equal(str(u.mS / u.cm ** 2), 'mS / cm^2')

        assert_equal(display_in_unit(10. * u.mV), '10. mV')
        assert_equal(display_in_unit(10. * u.ohm * u.amp), '10. A * ohm')
        assert_equal(display_in_unit(120. * (u.mS / u.cm ** 2)), '120. mS / cm^2')
        assert_equal(display_in_unit(3.0 * u.kmeter / 130.51 * u.meter), '0.02298674 km * m')
        assert_equal(display_in_unit(3.0 * u.kmeter / (130.51 * u.meter)), '22.986744')
        assert_equal(display_in_unit(3.0 * u.kmeter / 130.51 * u.meter * u.cm ** -2), '229867.44')
        assert_equal(display_in_unit(3.0 * u.kmeter / 130.51 * u.meter * u.cm ** -1), '0.02298674 km * m / cm')
        assert_equal(display_in_unit(1. * u.joule / u.kelvin), '1. J / K')

        # __str__ uses canonical format
        assert_equal(str(1. * u.metre / ((3.0 * u.ms) / (1. * u.second))), '333.33334 m')
        assert_equal(str(1. * u.metre / ((3.0 * u.ms) / 1. * u.second)), '0.33333334 m / (ms * s)')
        assert_equal(str((3.0 * u.ms) / 1. * u.second), '3. ms * s')

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
        with pytest.raises(TypeError):
            q.cumprod()  # cumprod not supported for non-dimensionless quantities
        assert_quantity(q.diagonal(), [1, 4], second)
        assert_quantity(q.max(), 4, second)
        assert_quantity(q.mean(), 2.5, second)
        assert_quantity(q.min(), 1, second)
        assert_quantity(q.ptp(), 3, second)
        assert_quantity(q.ravel(), [1, 2, 3, 4], second)

    def test_shape_manipulation_extended(self):
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
        q_split = [[10, 2], [30, 4]] * volt
        assert_quantity(np.array_split(q_split, 2)[0], [[10, 2]], volt)

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

    def test_slicing_extended(self):
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

        brainstate = pytest.importorskip("brainstate")
        quantity = Quantity(brainstate.random.rand(10))
        quantity[0] = 1.0

    def test_multiplication_division(self):
        _u = mV
        quantities = [3 * mV, np.array([1, 2]) * _u, np.ones((3, 3)) * _u]
        q2 = 5 * second

        for q in quantities:
            # Scalars and array scalars
            assert_quantity(q / 3, q.mantissa / 3, _u)
            assert_quantity(3 / q, 3 / q.mantissa, _u.reverse())
            assert_quantity(q * 3, q.mantissa * 3, _u)
            assert_quantity(3 * q, 3 * q.mantissa, _u)
            assert_quantity(q / np.float64(3), q.mantissa / 3, _u)
            assert_quantity(np.float64(3) / q, 3 / q.mantissa, _u.reverse())
            assert_quantity(q * np.float64(3), q.mantissa * 3, _u)
            assert_quantity(np.float64(3) * q, 3 * q.mantissa, _u)
            assert_quantity(q / jnp.array(3), q.mantissa / 3, _u)
            assert_quantity(np.array(3) / q, 3 / q.mantissa, _u.reverse())
            assert_quantity(q * jnp.array(3), q.mantissa * 3, _u)
            assert_quantity(np.array(3) * q, 3 * q.mantissa, _u)

            # (unitless) arrays
            assert_quantity(q / np.array([3]), q.mantissa / 3, _u)
            assert_quantity(np.array([3]) / q, 3 / q.mantissa, _u.reverse())
            assert_quantity(q * np.array([3]), q.mantissa * 3, _u)
            assert_quantity(np.array([3]) * q, 3 * q.mantissa, _u)

            # arrays with units
            assert_quantity(q / q, q.mantissa / q.mantissa)
            assert_quantity(q * q, q.mantissa ** 2, _u ** 2)
            assert_quantity(q / q2, q.mantissa / q2.mantissa, _u / second)
            assert_quantity(q2 / q, q2.mantissa / q.mantissa, second / _u)
            assert_quantity(q * q2, q.mantissa * q2.mantissa, _u * second)

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

            # Check that operations with 0 raise
            with pytest.raises(u.UnitMismatchError):
                q + 0
            with pytest.raises(u.UnitMismatchError):
                0 + q
            with pytest.raises(u.UnitMismatchError):
                q - 0
            with pytest.raises(u.UnitMismatchError):
                q + np.float64(0)
            with pytest.raises(u.UnitMismatchError):
                np.float64(0) + q
            with pytest.raises(u.UnitMismatchError):
                q - np.float64(0)

    def test_binary_operations(self):
        """Test whether binary operations work when they should and raise
        DimensionMismatchErrors when they should.
        """
        from operator import add, eq, ge, gt, le, lt, ne, sub

        def assert_operations_work(a, b):
            try:
                tryops = [add, sub, lt, le, gt, ge, eq, ne]
                for op in tryops:
                    op(a, b)
                    op(b, a)

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
            tryops = [add, sub, lt, le, gt, ge, eq, ne]
            for op in tryops:
                with pytest.raises(u.UnitMismatchError):
                    op(a, b)
                with pytest.raises(u.UnitMismatchError):
                    op(b, a)

        # Check that consistent units work
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

        # Check that inconsistent units do not work
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
        """Test raising quantities to a power."""
        arrs = [2 * kilogram, np.array([2]) * kilogram, np.array([1, 2]) * kilogram]
        for a in arrs:
            assert_quantity(a ** 3, a.mantissa ** 3, kilogram ** 3)
            # Test raising to a dimensionless Quantity
            assert_quantity(a ** (3 * volt / volt), a.mantissa ** 3, kilogram ** 3)
            with pytest.raises(ValueError):
                a ** (2 * volt)
            with pytest.raises(TypeError):
                a ** np.array([2, 3])

    def test_inplace_operations(self):
        q = np.arange(10) * volt
        q_orig = q.copy()
        q_id = id(q)

        q += 1 * volt
        assert np.all(q == q_orig + 1 * volt) and id(q) == q_id
        q -= 1 * volt
        assert np.all(q == q_orig) and id(q) == q_id

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

    def test_indices_functions(self):
        """Check numpy functions that return indices."""
        values = [np.array([-4, 3, -2, 1, 0]), np.ones((3, 3)), np.array([17])]
        units = [volt, second, siemens, mV, kHz]

        indice_funcs = [u.math.argmin, u.math.argmax, u.math.argsort, u.math.nonzero]

        for value, unit in itertools.product(values, units):
            q_ar = value * unit
            for func in indice_funcs:
                test_ar = func(q_ar)
                comparison_ar = func(value)
                test_ar = u.math.asarray(test_ar)
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
        """Test converting to and from a list."""
        values = [3 * mV, np.array([1, 2]) * mV, np.arange(12).reshape(4, 3) * mV]
        for value in values:
            l = value.tolist()
            from_list = Quantity(l)
            assert have_same_dim(from_list, value)
            assert u.math.allclose(from_list.mantissa, value.mantissa)

    def test_units_vs_quantities(self):
        # Unit objects should stay Unit objects under certain operations
        assert isinstance(meter ** 2, Unit)
        assert isinstance(meter ** -1, Unit)
        assert isinstance(meter ** 0.5, Unit)
        assert isinstance(meter / second, Unit)
        assert isinstance(amp / meter ** 2, Unit)

        assert type(2 / meter) == Quantity
        assert type(2 * meter) == Quantity
        assert type(meter + meter) == Unit
        assert type(meter - meter) == Unit

    def test_jit_array(self):
        @jax.jit
        def f1(a):
            b = a * u.siemens / u.cm ** 2
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
            return b

        val = np.random.rand(3)
        r = f3(val)
        u.math.allclose(val * u.siemens / u.cm ** 2, r)

    def test_jit_array2(self):
        a = 2.0 * (u.farad / u.metre ** 2)

        @jax.jit
        def f(b):
            return b

        f(a)

    def test_setiterm(self):
        unit = Quantity([0, 0, 0.])
        unit[jnp.asarray([0, 1, 1])] += jnp.asarray([1., 1., 1.])
        assert_quantity(unit, [1., 1., 0.])

        unit = Quantity([0, 0, 0.])
        unit = unit.scatter_add(jnp.asarray([0, 1, 1]), jnp.asarray([1., 1., 1.]))
        assert_quantity(unit, [1., 2., 0.])

        nu = np.asarray([0, 0, 0.])
        nu[np.asarray([0, 1, 1])] += np.asarray([1., 1., 1.])
        assert np.allclose(nu, np.asarray([1., 1., 0.]))

    def test_at(self):
        x = jnp.arange(5.0) * u.mV
        with pytest.raises(u.UnitMismatchError):
            x.at[2].add(10)
        x.at[2].add(10 * u.mV)
        x.at[10].add(10 * u.mV)  # out-of-bounds indices are ignored
        x.at[20].add(10 * u.mV, mode='clip')
        x.at[2].get()
        x.at[20].get()  # out-of-bounds indices clipped
        x.at[20].get(mode='fill')  # out-of-bounds indices filled with NaN
        with pytest.raises(u.UnitMismatchError):
            x.at[20].get(mode='fill', fill_value=-1)  # custom fill data
        x.at[20].get(mode='fill', fill_value=-1 * u.mV)  # custom fill data

    def test_to(self):
        x = jnp.arange(5.0) * u.mV
        with pytest.raises(u.UnitMismatchError):
            x.to(u.mA)
        x.to(u.volt)
        x.to(u.uvolt)

    def test_quantity_type(self):
        def f1(a: u.Quantity[u.ms]) -> u.Quantity[u.mV]:
            return a

        def f2(a: u.Quantity[Union[u.ms, u.mA]]) -> u.Quantity[u.mV]:
            return a

        def f3(a: u.Quantity[Union[u.ms, u.mA]]) -> u.Quantity[Union[u.mV, u.ms]]:
            return a


# =========================================================================
# NumPy function tests (from _base_test.py TestNumPyFunctions)
# =========================================================================

class TestNumPyFunctions:
    """Test numpy functions and methods on Quantity objects."""

    def test_special_case_numpy_functions(self):
        """Test a couple of functions/methods that need special treatment."""
        from saiunit.math import diagonal, ravel, trace, where
        from saiunit.linalg import dot

        quadratic_matrix = np.reshape(np.arange(9), (3, 3)) * mV

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert u.math.allclose(ravel(quadratic_matrix).mantissa, quadratic_matrix.ravel().mantissa)
            assert u.math.allclose(
                np.asarray(ravel(quadratic_matrix).mantissa),
                ravel(np.asarray(quadratic_matrix.mantissa))
            )
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
        cond = np.array([True, False, False])
        ar1 = np.array([1, 2, 3])
        ar2 = np.array([4, 5, 6])
        assert_equal(np.where(cond), where(cond))
        assert_equal(np.where(cond, ar1, ar2), where(cond, ar1, ar2))

        # dimensionless Quantity
        assert u.math.allclose(
            np.where(cond, ar1, ar2),
            np.asarray(where(cond, ar1 * mV / mV, ar2 * mV / mV))
        )

        # Quantity with dimensions
        ar1 = ar1 * mV
        ar2 = ar2 * mV
        assert u.math.allclose(
            np.where(cond, ar1.mantissa, ar2.mantissa),
            np.asarray(where(cond, ar1, ar2).mantissa),
        )

        # Check some error cases
        with pytest.raises(TypeError):
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
        with pytest.raises(TypeError):
            (np.arange(1, 5) * mV).cumprod()  # non-dimensionless raises

    def test_unit_discarding_functions(self):
        """Test functions that discard units."""
        values = [3 * mV, np.array([1, 2]) * mV, np.arange(12).reshape(3, 4) * mV]
        for a in values:
            assert np.allclose(np.sign(a.mantissa), np.sign(np.asarray(a.mantissa)))
            assert np.allclose(u.math.zeros_like(a).mantissa, np.zeros_like(np.asarray(a.mantissa)))
            assert np.allclose(u.math.ones_like(a).mantissa, np.ones_like(np.asarray(a.mantissa)))
            if a.ndim > 0:
                assert np.allclose(np.nonzero(a.mantissa), np.nonzero(np.asarray(a.mantissa)))

    def test_numpy_functions_same_dimensions(self):
        values = [np.array([1, 2]), np.ones((3, 3))]
        units = [volt, second, siemens, mV, kHz]

        keep_dim_funcs = [
            'cumsum',
            'max',
            'mean',
            'min',
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
                        f"'{func}' failed on {q_ar!r} -- unit was "
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
        """Test the unitsafe functions wrapping their numpy counterparts."""
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
                with pytest.raises(TypeError):
                    bu_fun(val)

            for val in unitless_values:
                if hasattr(val, "mantissa"):
                    assert u.math.allclose(bu_fun(val.mantissa), np_fun(val.mantissa),
                                           equal_nan=True, atol=1e-3, rtol=1e-3)
                else:
                    assert u.math.allclose(bu_fun(val), np_fun(val),
                                           equal_nan=True, atol=1e-3, rtol=1e-3)

            for val in numpy_values:
                assert u.math.allclose(bu_fun(val), np_fun(val),
                                       equal_nan=True, atol=1e-3, rtol=1e-3)


# =========================================================================
# JIT tests (from _base_test.py TestJit)
# =========================================================================

class TestJitQuantity:
    """Test jax.jit with Quantity objects."""

    def test_jit_multiply(self):
        @jax.jit
        def f(a):
            return a * 2

        f(3 * u.mV)
        f(jnp.ones(10) * u.mV)

    def test_jit_kwargs(self):
        @jax.jit
        def f(**kwargs):
            return kwargs['a'] * 2

        f(a=3 * u.mV)
        f(a=jnp.ones(10) * u.mV)


# =========================================================================
# Pickle standalone test (from _base_test.py test_pickle)
# =========================================================================

def test_pickle_with_units():
    tmpdir = tempfile.gettempdir()
    filename = os.path.join(tmpdir, "test_quantity_with_units.pkl")
    a = 3 * mV
    with open(filename, "wb") as f:
        pickle.dump(a, f)

    with open(filename, "rb") as f:
        b = pickle.load(f)
    assert u.math.allclose(a, b)


# --- Docstring example tests ---


def test_docstring_example_quantity_construction():
    """Test Quantity construction patterns from the class docstring."""
    import saiunit as u

    # Scalar with unit
    q = u.Quantity(3.0, unit=u.mV)
    assert jnp.allclose(q.mantissa, 3.0)
    assert q.unit == u.mV

    # Array with unit via multiplication shorthand
    arr = jnp.array([1.0, 2.0, 3.0]) * u.mV
    assert arr.shape == (3,)

    # From a Unit object directly
    q_unit = u.Quantity(u.metre)
    assert jnp.allclose(q_unit.mantissa, 1.0)
    assert q_unit.unit == u.metre


def test_docstring_example_to():
    """Test Quantity.to() from the docstring."""
    import saiunit as u

    q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
    converted = q.to(u.volt)
    assert converted.unit == u.volt
    assert jnp.allclose(converted.mantissa, jnp.array([0.001, 0.002, 0.003]))


def test_docstring_example_to_decimal():
    """Test Quantity.to_decimal() from the docstring."""
    import saiunit as u

    q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
    result = q.to_decimal(u.volt)
    assert jnp.allclose(result, jnp.array([0.001, 0.002, 0.003]))
    # result should be a plain array, not a Quantity
    assert not isinstance(result, u.Quantity)


def test_docstring_example_repr_in_unit():
    """Test Quantity.repr_in_unit() from the docstring."""
    import saiunit as u

    x = u.Quantity(25.123456, unit=u.mV)
    s = x.repr_in_unit()
    # Float32 may truncate the last digit; check the significant prefix
    assert s.startswith("25.12345")
    assert "mV" in s

    # With precision after unit conversion
    s2 = x.to(u.volt).repr_in_unit(3)
    assert "0.025" in s2
    assert "V" in s2


def test_docstring_example_has_same_unit():
    """Test Quantity.has_same_unit() from the docstring."""
    import saiunit as u

    a = u.Quantity(1.0, unit=u.mV)
    b = u.Quantity(2.0, unit=u.volt)
    assert a.has_same_unit(b) is True

    c = u.Quantity(1.0, unit=u.second)
    assert a.has_same_unit(c) is False


def test_docstring_example_with_unit():
    """Test Quantity.with_unit() static method from the docstring."""
    import saiunit as u

    q = u.Quantity.with_unit(2.0, unit=u.metre)
    assert jnp.allclose(q.mantissa, 2.0)
    assert q.unit == u.metre


def test_docstring_example_compatible_with_equinox():
    """Test compatible_with_equinox() function from the docstring."""
    import saiunit as u
    import saiunit._base_quantity as bq

    u.compatible_with_equinox(True)
    assert bq.compat_with_equinox is True

    u.compatible_with_equinox(False)
    assert bq.compat_with_equinox is False


def test_docstring_example_mantissa():
    """Test Quantity.mantissa property from the docstring."""
    import saiunit as u

    q = u.Quantity(3.0, unit=u.mV)
    assert q.mantissa == 3.0


def test_docstring_example_is_unitless():
    """Test Quantity.is_unitless property from the docstring."""
    import saiunit as u

    assert u.Quantity(5.0).is_unitless is True
    assert u.Quantity(5.0, unit=u.mV).is_unitless is False


def test_docstring_example_dim():
    """Test Quantity.dim property from the docstring."""
    import saiunit as u

    q = u.Quantity(5.0, unit=u.metre)
    assert q.dim is not None


def test_docstring_example_shape_ndim_size():
    """Test Quantity.shape, .ndim, .size from the docstrings."""
    import saiunit as u

    q = u.Quantity(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.mV)
    assert q.shape == (2, 2)
    assert q.ndim == 2
    assert q.size == 4


def test_docstring_example_dtype():
    """Test Quantity.dtype property from the docstring."""
    import saiunit as u

    q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
    assert q.dtype == jnp.float32


def test_docstring_example_real_imag():
    """Test Quantity.real and .imag from the docstrings."""
    import saiunit as u

    q = u.Quantity(1.0 + 2.0j, unit=u.mV)
    assert jnp.allclose(q.real.mantissa, 1.0)
    assert jnp.allclose(q.imag.mantissa, 2.0)
    assert q.real.unit == u.mV
    assert q.imag.unit == u.mV


def test_docstring_example_mT():
    """Test Quantity.mT from the docstring."""
    import saiunit as u

    q = u.Quantity(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.mV)
    result = q.mT
    assert result.shape == (2, 2)
    assert result.unit == u.mV


def test_docstring_example_scatter_add():
    """Test Quantity.scatter_add() from the docstring."""
    import saiunit as u

    q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
    result = q.scatter_add(0, u.Quantity(10.0, unit=u.mV))
    assert jnp.allclose(result.mantissa, jnp.array([11.0, 2.0, 3.0]))
    assert result.unit == u.mV


def test_docstring_example_dot():
    """Test Quantity.dot() from the docstring."""
    import saiunit as u

    a = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
    b = u.Quantity(jnp.array([1.0, 1.0, 1.0]), unit=u.mV)
    result = a.dot(b)
    assert jnp.allclose(result.mantissa, 6.0)


def test_docstring_example_flatten():
    """Test Quantity.flatten() from the docstring."""
    import saiunit as u

    q = u.Quantity(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.mV)
    flat = q.flatten()
    assert flat.shape == (4,)
    assert jnp.allclose(flat.mantissa, jnp.array([1.0, 2.0, 3.0, 4.0]))


def test_docstring_example_pow():
    """Test Quantity.pow() from the docstring."""
    import saiunit as u

    q = u.Quantity(2.0, unit=u.mV)
    result = q.pow(2)
    assert jnp.allclose(result.mantissa, 4.0)


# ---------------------------------------------------------------------------
# Quantity with string unit tests
# ---------------------------------------------------------------------------


class TestQuantityStringUnit:
    """Tests for Quantity(value, 'unit_string')."""

    def test_simple_string_unit(self):
        q = Quantity(1.0, "mV")
        assert q.unit == mvolt
        assert q.mantissa == 1.0

    def test_fullname_string_unit(self):
        q = Quantity(1.0, "mvolt")
        assert q.unit == mvolt

    def test_compound_string_unit(self):
        q = Quantity(1.0, "J / kg")
        assert q.unit == joule / kilogram

    def test_unitless_string(self):
        q = Quantity(1.0, "1")
        assert q.unit == UNITLESS

    def test_array_with_string_unit(self):
        q = Quantity(jnp.array([1.0, 2.0, 3.0]), "mV")
        assert q.unit == mvolt
        assert q.shape == (3,)

    def test_invalid_string_unit_raises(self):
        with pytest.raises(ValueError):
            Quantity(1.0, "nonexistent_xyz")
