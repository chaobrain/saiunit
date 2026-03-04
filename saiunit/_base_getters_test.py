"""Tests for _base_getters.py: getter functions, validation, type checking."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from saiunit._base_dimension import (
    DIMENSIONLESS,
    DimensionMismatchError,
    UnitMismatchError,
    get_or_create_dimension,
)
from saiunit._base_getters import (
    _assert_not_quantity,
    _short_str,
    _to_quantity,
    array_with_unit,
    assert_quantity,
    change_printoption,
    display_in_unit,
    fail_for_dimension_mismatch,
    fail_for_unit_mismatch,
    get_dim,
    get_magnitude,
    get_mantissa,
    get_unit,
    has_same_unit,
    have_same_dim,
    is_dimensionless,
    is_scalar_type,
    is_unitless,
    maybe_decimal,
    split_mantissa_unit,
    unit_scale_align_to_first,
)
from saiunit._base_quantity import Quantity
from saiunit._base_unit import UNITLESS, Unit


# helpers: build common units/dims once
_length_dim = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
_time_dim = get_or_create_dimension([0, 0, 1, 0, 0, 0, 0])
_metre = Unit(_length_dim, name="metre", dispname="m", scale=0)
_second = Unit(_time_dim, name="second", dispname="s", scale=0)
_kmetre = Unit(_length_dim, name="kmetre", dispname="km", scale=3)


# =========================================================================
# _to_quantity / _assert_not_quantity
# =========================================================================

class TestToQuantity:
    def test_quantity_passthrough(self):
        q = Quantity(3.0, unit=_metre)
        assert _to_quantity(q) is q

    def test_scalar_becomes_quantity(self):
        q = _to_quantity(5.0)
        assert isinstance(q, Quantity)
        assert q.is_unitless

    def test_array_becomes_quantity(self):
        q = _to_quantity(np.array([1, 2, 3]))
        assert isinstance(q, Quantity)

    def test_assert_not_quantity_passes_for_scalar(self):
        _assert_not_quantity(5.0)  # should not raise

    def test_assert_not_quantity_raises_for_quantity(self):
        q = Quantity(3.0, unit=_metre)
        with pytest.raises(ValueError, match="should not be"):
            _assert_not_quantity(q)


# =========================================================================
# change_printoption / _short_str
# =========================================================================

class TestPrintHelpers:
    def test_change_printoption_restores(self):
        old = np.get_printoptions()["threshold"]
        with change_printoption(threshold=10):
            assert np.get_printoptions()["threshold"] == 10
        assert np.get_printoptions()["threshold"] == old

    def test_short_str_scalar(self):
        s = _short_str(np.array(3.14))
        assert "3.14" in s

    def test_short_str_quantity(self):
        q = Quantity(np.array([1.0, 2.0, 3.0]), unit=_metre)
        s = _short_str(q)
        assert "1." in s


# =========================================================================
# get_dim
# =========================================================================

class TestGetDim:
    def test_from_dimension(self):
        assert get_dim(_length_dim) is _length_dim

    def test_from_unit(self):
        assert get_dim(_metre) is _length_dim

    def test_from_quantity(self):
        q = Quantity(5.0, unit=_metre)
        assert get_dim(q) is _length_dim

    def test_from_scalar(self):
        assert get_dim(5.0) is DIMENSIONLESS

    def test_from_numpy_array(self):
        assert get_dim(np.array(5.0)) is DIMENSIONLESS

    def test_from_jax_array(self):
        assert get_dim(jnp.array(5.0)) is DIMENSIONLESS


# =========================================================================
# get_unit
# =========================================================================

class TestGetUnit:
    def test_from_unit(self):
        assert get_unit(_metre) is _metre

    def test_from_quantity(self):
        q = Quantity(5.0, unit=_metre)
        assert get_unit(q) == _metre

    def test_from_scalar(self):
        assert get_unit(5.0) == UNITLESS


# =========================================================================
# get_mantissa / get_magnitude
# =========================================================================

class TestGetMantissa:
    def test_from_quantity(self):
        q = Quantity(5.0, unit=_metre)
        m = get_mantissa(q)
        assert jnp.allclose(m, 5.0)

    def test_from_scalar(self):
        assert get_mantissa(5.0) == 5.0

    def test_from_array(self):
        arr = np.array([1, 2, 3])
        assert np.array_equal(get_mantissa(arr), arr)

    def test_get_magnitude_is_alias(self):
        assert get_magnitude is get_mantissa


# =========================================================================
# split_mantissa_unit
# =========================================================================

class TestSplitMantissaUnit:
    def test_quantity(self):
        q = Quantity(5.0, unit=_metre)
        m, u = split_mantissa_unit(q)
        assert jnp.allclose(m, 5.0)
        assert u == _metre

    def test_scalar(self):
        m, u = split_mantissa_unit(5.0)
        assert jnp.allclose(m, 5.0)
        assert u == UNITLESS


# =========================================================================
# have_same_dim / has_same_unit
# =========================================================================

class TestSameDim:
    def test_same_dim_quantities(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_kmetre)
        assert have_same_dim(q1, q2)

    def test_different_dim_quantities(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_second)
        assert not have_same_dim(q1, q2)

    def test_scalar_vs_scalar(self):
        assert have_same_dim(1.0, 2.0)

    def test_scalar_vs_quantity(self):
        q = Quantity(1.0, unit=_metre)
        assert not have_same_dim(1.0, q)

    def test_quantity_with_same_dim_int(self):
        assert have_same_dim(1.0, 1)


class TestSameUnit:
    def test_same_unit(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_metre)
        assert has_same_unit(q1, q2)

    def test_different_scale(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_kmetre)
        assert not has_same_unit(q1, q2)

    def test_scalar_vs_scalar(self):
        assert has_same_unit(1.0, 2.0)


# =========================================================================
# fail_for_dimension_mismatch
# =========================================================================

class TestFailForDimensionMismatch:
    def test_same_dim_returns_dims(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_kmetre)
        d1, d2 = fail_for_dimension_mismatch(q1, q2)
        assert d1 == _length_dim
        assert d2 == _length_dim

    def test_different_dim_raises(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_second)
        with pytest.raises(DimensionMismatchError):
            fail_for_dimension_mismatch(q1, q2)

    def test_against_none_means_dimensionless(self):
        # scalar is dimensionless — should not raise
        d1, d2 = fail_for_dimension_mismatch(5.0)
        assert d1 == DIMENSIONLESS
        assert d2 == DIMENSIONLESS

    def test_non_dimensionless_against_none_raises(self):
        q = Quantity(1.0, unit=_metre)
        with pytest.raises(DimensionMismatchError):
            fail_for_dimension_mismatch(q)

    def test_custom_error_message(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_second)
        with pytest.raises(DimensionMismatchError, match="custom"):
            fail_for_dimension_mismatch(q1, q2, error_message="custom {x}", x=q1)


# =========================================================================
# fail_for_unit_mismatch
# =========================================================================

class TestFailForUnitMismatch:
    def test_same_dim_returns_units(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_kmetre)
        u1, u2 = fail_for_unit_mismatch(q1, q2)
        assert u1.has_same_dim(u2)

    def test_different_dim_raises(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_second)
        with pytest.raises(UnitMismatchError):
            fail_for_unit_mismatch(q1, q2)

    def test_against_none_means_unitless(self):
        u1, u2 = fail_for_unit_mismatch(5.0)
        assert u1 == UNITLESS
        assert u2 == UNITLESS


# =========================================================================
# display_in_unit
# =========================================================================

class TestDisplayInUnit:
    def test_basic_display(self):
        q = Quantity(5.0, unit=_metre)
        s = display_in_unit(q)
        assert "5." in s

    def test_display_with_target_unit(self):
        q = Quantity(1.0, unit=_kmetre)
        s = display_in_unit(q, _metre)
        assert "1000." in s

    def test_display_scalar(self):
        s = display_in_unit(5.0)
        assert "5." in s


# =========================================================================
# maybe_decimal
# =========================================================================

class TestMaybeDecimal:
    def test_dimensionless_returns_scalar(self):
        q = Quantity(5.0, unit=UNITLESS)
        result = maybe_decimal(q)
        assert not isinstance(result, Quantity)
        assert jnp.allclose(result, 5.0)

    def test_with_unit_returns_quantity(self):
        q = Quantity(5.0, unit=_metre)
        result = maybe_decimal(q)
        assert isinstance(result, Quantity)

    def test_with_target_unit(self):
        q = Quantity(1.0, unit=_kmetre)
        result = maybe_decimal(q, _metre)
        assert not isinstance(result, Quantity)
        assert jnp.allclose(result, 1000.)

    def test_scalar_returns_scalar(self):
        result = maybe_decimal(5.0)
        assert jnp.allclose(result, 5.0)


# =========================================================================
# unit_scale_align_to_first
# =========================================================================

class TestUnitScaleAlignToFirst:
    def test_align_same_units(self):
        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_metre)
        aligned = unit_scale_align_to_first(q1, q2)
        assert len(aligned) == 2
        assert aligned[0].unit == aligned[1].unit

    def test_empty_args(self):
        result = unit_scale_align_to_first()
        assert len(result) == 0


# =========================================================================
# array_with_unit
# =========================================================================

class TestArrayWithUnit:
    def test_basic(self):
        q = array_with_unit(5.0, _metre)
        assert isinstance(q, Quantity)
        assert q.unit == _metre

    def test_with_dtype(self):
        q = array_with_unit(5.0, _metre, dtype=jnp.float32)
        assert q.dtype == jnp.float32

    def test_non_unit_raises(self):
        with pytest.raises(TypeError, match="Expected instance of Unit"):
            array_with_unit(5.0, "not a unit")


# =========================================================================
# is_dimensionless / is_unitless / is_scalar_type
# =========================================================================

class TestTypeChecking:
    def test_is_dimensionless_scalar(self):
        assert is_dimensionless(5.0)

    def test_is_dimensionless_dimensionless_quantity(self):
        q = Quantity(5.0, unit=UNITLESS)
        assert is_dimensionless(q)

    def test_is_dimensionless_physical_quantity(self):
        q = Quantity(5.0, unit=_metre)
        assert not is_dimensionless(q)

    def test_is_dimensionless_dimension_object(self):
        assert is_dimensionless(DIMENSIONLESS)
        assert not is_dimensionless(_length_dim)

    def test_is_unitless_scalar(self):
        assert is_unitless(5.0)

    def test_is_unitless_unitless_quantity(self):
        q = Quantity(5.0, unit=UNITLESS)
        assert is_unitless(q)

    def test_is_unitless_physical_quantity(self):
        q = Quantity(5.0, unit=_metre)
        assert not is_unitless(q)

    def test_is_unitless_dimension_raises(self):
        with pytest.raises(TypeError, match="Dimension objects"):
            is_unitless(DIMENSIONLESS)

    def test_is_scalar_type_int(self):
        assert is_scalar_type(5)

    def test_is_scalar_type_float(self):
        assert is_scalar_type(5.0)

    def test_is_scalar_type_string_false(self):
        assert not is_scalar_type("hello")

    def test_is_scalar_type_array_false(self):
        assert not is_scalar_type(np.array([1, 2]))

    def test_is_scalar_type_quantity_scalar_unitless(self):
        q = Quantity(5.0, unit=UNITLESS)
        assert is_scalar_type(q)

    def test_is_scalar_type_quantity_with_unit_false(self):
        q = Quantity(5.0, unit=_metre)
        assert not is_scalar_type(q)


# =========================================================================
# assert_quantity
# =========================================================================

class TestAssertQuantity:
    def test_unitless_quantity_no_unit_given(self):
        q = Quantity(5.0, unit=UNITLESS)
        assert_quantity(q, 5.0)  # should not raise

    def test_quantity_with_unit(self):
        q = Quantity(5.0, unit=_metre)
        assert_quantity(q, 5.0, _metre)  # should not raise

    def test_scalar_no_unit(self):
        assert_quantity(5.0, 5.0)  # should not raise

    def test_wrong_value_raises(self):
        q = Quantity(5.0, unit=UNITLESS)
        with pytest.raises(AssertionError):
            assert_quantity(q, 10.0)

    def test_wrong_unit_raises(self):
        q = Quantity(5.0, unit=_metre)
        with pytest.raises((AssertionError, DimensionMismatchError)):
            assert_quantity(q, 5.0, _second)

    def test_non_unit_raises(self):
        q = Quantity(5.0, unit=_metre)
        with pytest.raises(AssertionError, match="Expected a Unit"):
            assert_quantity(q, 5.0, "not a unit")

    def test_unitless_quantity_value_mismatch(self):
        q = Quantity(5.0, unit=_metre)
        with pytest.raises(AssertionError):
            assert_quantity(q, 5.0)


# =========================================================================
# Integration tests migrated from _base_test.py
# =========================================================================

import saiunit as u
from saiunit._unit_common import *
from saiunit._unit_shortcuts import kHz, ms, mV, nS


class TestFailForDimensionMismatchIntegration:
    """Integration tests for fail_for_dimension_mismatch using public unit objects."""

    def test_scalar_returns_dimensionless(self):
        dim1, dim2 = fail_for_dimension_mismatch(3)
        assert dim1 is DIMENSIONLESS
        assert dim2 is DIMENSIONLESS

    def test_unitless_quantity_returns_dimensionless(self):
        dim1, dim2 = fail_for_dimension_mismatch(3 * volt / volt)
        assert dim1 is DIMENSIONLESS
        assert dim2 is DIMENSIONLESS

    def test_unitless_quantity_and_scalar(self):
        dim1, dim2 = fail_for_dimension_mismatch(3 * volt / volt, 7)
        assert dim1 is DIMENSIONLESS
        assert dim2 is DIMENSIONLESS

    def test_same_unit_quantities(self):
        dim1, dim2 = fail_for_dimension_mismatch(3 * volt, 5 * volt)
        assert dim1 is volt.dim
        assert dim2 is volt.dim

    def test_single_non_dimensionless_raises(self):
        with pytest.raises(DimensionMismatchError):
            fail_for_dimension_mismatch(6 * volt)

    def test_different_units_raises(self):
        with pytest.raises(DimensionMismatchError):
            fail_for_dimension_mismatch(6 * volt, 5 * second)


class TestGetMethodIntegration:
    """Integration tests for get_dim, get_unit, get_mantissa using public unit objects."""

    def test_get_dim_scalars(self):
        assert u.get_dim(1) == u.DIMENSIONLESS
        assert u.get_dim(1.0) == u.DIMENSIONLESS

    def test_get_dim_quantities(self):
        assert u.get_dim(1 * u.mV) == u.volt.dim
        assert u.get_dim(1 * u.mV / u.mV) == u.DIMENSIONLESS
        assert u.get_dim(1 * u.mV / u.second) == u.volt.dim / u.second.dim
        assert u.get_dim(1 * u.mV / u.second ** 2) == u.volt.dim / u.second.dim ** 2
        assert u.get_dim(1 * u.mV ** 2 / u.second ** 2) == u.volt.dim ** 2 / u.second.dim ** 2

    def test_get_dim_non_numeric(self):
        assert u.get_dim(object()) == u.DIMENSIONLESS
        assert u.get_dim("string") == u.DIMENSIONLESS
        assert u.get_dim([1, 2, 3]) == u.DIMENSIONLESS

    def test_get_dim_arrays(self):
        assert u.get_dim(np.array([1, 2, 3])) == u.DIMENSIONLESS
        assert u.get_dim(np.array([1, 2, 3]) * u.mV) == u.volt.dim

    def test_get_dim_units(self):
        assert u.get_dim(u.mV) == u.volt.dim
        assert u.get_dim(u.mV / u.mV) == u.DIMENSIONLESS
        assert u.get_dim(u.mV / u.second) == u.volt.dim / u.second.dim
        assert u.get_dim(u.mV / u.second ** 2) == u.volt.dim / u.second.dim ** 2
        assert u.get_dim(u.mV ** 2 / u.second ** 2) == u.volt.dim ** 2 / u.second.dim ** 2

    def test_get_dim_dimensions(self):
        assert u.get_dim(u.mV.dim) == u.volt.dim
        assert u.get_dim(u.mV.dim / u.mV.dim) == u.DIMENSIONLESS
        assert u.get_dim(u.mV.dim / u.second.dim) == u.volt.dim / u.second.dim
        assert u.get_dim(u.mV.dim / u.second.dim ** 2) == u.volt.dim / u.second.dim ** 2
        assert u.get_dim(u.mV.dim ** 2 / u.second.dim ** 2) == u.volt.dim ** 2 / u.second.dim ** 2

    def test_get_unit_scalars(self):
        assert u.get_unit(1) == u.UNITLESS
        assert u.get_unit(1.0) == u.UNITLESS

    def test_get_unit_quantities(self):
        assert u.get_unit(1 * u.mV) == u.mV
        assert u.get_unit(1 * u.mV / u.mV) == u.UNITLESS
        assert u.get_unit(1 * u.mV / u.second) == u.mV / u.second
        assert u.get_unit(1 * u.mV / u.second ** 2) == u.mV / u.second ** 2
        assert u.get_unit(1 * u.mV ** 2 / u.second ** 2) == u.mV ** 2 / u.second ** 2

    def test_get_unit_non_numeric(self):
        assert u.get_unit(object()) == u.UNITLESS
        assert u.get_unit("string") == u.UNITLESS
        assert u.get_unit([1, 2, 3]) == u.UNITLESS

    def test_get_unit_arrays(self):
        assert u.get_unit(np.array([1, 2, 3])) == u.UNITLESS
        assert u.get_unit(np.array([1, 2, 3]) * u.mV) == u.mV

    def test_get_unit_units(self):
        assert u.get_unit(u.mV) == u.mV
        assert u.get_unit(u.mV / u.mV) == u.UNITLESS
        assert u.get_unit(u.mV / u.second) == u.mV / u.second
        assert u.get_unit(u.mV / u.second ** 2) == u.mV / u.second ** 2
        assert u.get_unit(u.mV ** 2 / u.second ** 2) == u.mV ** 2 / u.second ** 2

    def test_get_unit_dimensions_return_unitless(self):
        assert u.get_unit(u.mV.dim) == u.UNITLESS
        assert u.get_unit(u.mV.dim / u.mV.dim) == u.UNITLESS
        assert u.get_unit(u.mV.dim / u.second.dim) == u.UNITLESS
        assert u.get_unit(u.mV.dim / u.second.dim ** 2) == u.UNITLESS
        assert u.get_unit(u.mV.dim ** 2 / u.second.dim ** 2) == u.UNITLESS

    def test_get_mantissa_scalars(self):
        assert u.get_mantissa(1) == 1
        assert u.get_mantissa(1.0) == 1.0

    def test_get_mantissa_quantities(self):
        assert u.get_mantissa(1 * u.mV) == 1
        assert u.get_mantissa(1 * u.mV / u.mV) == 1
        assert u.get_mantissa(1 * u.mV / u.second) == 1
        assert u.get_mantissa(1 * u.mV / u.second ** 2) == 1
        assert u.get_mantissa(1 * u.mV ** 2 / u.second ** 2) == 1

    def test_get_mantissa_non_numeric(self):
        obj = object()
        assert u.get_mantissa(obj) == obj
        assert u.get_mantissa("string") == "string"
        assert u.get_mantissa([1, 2, 3]) == [1, 2, 3]

    def test_get_mantissa_arrays(self):
        assert np.allclose(u.get_mantissa(np.array([1, 2, 3])), np.array([1, 2, 3]))
        assert np.allclose(u.get_mantissa(np.array([1, 2, 3]) * u.mV), np.array([1, 2, 3]))

    def test_get_mantissa_units_passthrough(self):
        assert u.get_mantissa(u.mV) == u.mV
        assert u.get_mantissa(u.mV / u.mV) == u.mV / u.mV
        assert u.get_mantissa(u.mV / u.second) == u.mV / u.second
        assert u.get_mantissa(u.mV / u.second ** 2) == u.mV / u.second ** 2
        assert u.get_mantissa(u.mV ** 2 / u.second ** 2) == u.mV ** 2 / u.second ** 2

    def test_get_mantissa_dimensions_passthrough(self):
        assert u.get_mantissa(u.mV.dim) == u.mV.dim
        assert u.get_mantissa(u.mV.dim / u.mV.dim) == u.mV.dim / u.mV.dim
        assert u.get_mantissa(u.mV.dim / u.second.dim) == u.mV.dim / u.second.dim
        assert u.get_mantissa(u.mV.dim / u.second.dim ** 2) == u.mV.dim / u.second.dim ** 2
        assert u.get_mantissa(u.mV.dim ** 2 / u.second.dim ** 2) == u.mV.dim ** 2 / u.second.dim ** 2

    def test_format_scalar(self):
        import brainstate
        with brainstate.environ.context(precision=64):
            q1 = 1.23456789 * u.mV
            assert f"{q1:.2f}" == "1.23 mV"
            assert f"{q1:.3f}" == "1.235 mV"
            assert f"{q1:.4f}" == "1.2346 mV"

    def test_format_array(self):
        import brainstate
        with brainstate.environ.context(precision=64):
            q2 = [1.23456789, 1.23456789] * u.mV
            assert f"{q2:.2f}" == "[1.23 1.23] mV"
            assert f"{q2:.3f}" == "[1.235 1.235] mV"
            assert f"{q2:.4f}" == "[1.2346 1.2346] mV"
