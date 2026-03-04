"""Tests for _base_unit.py: Unit, UNITLESS, display helpers, pickle."""

from __future__ import annotations

import pickle
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_equal

import saiunit as u
from saiunit import Quantity
from saiunit._base_dimension import (
    DIMENSIONLESS,
    Dimension,
    DimensionMismatchError,
    get_or_create_dimension,
)
from saiunit._base_getters import display_in_unit
from saiunit._base_unit import (
    UNITLESS,
    Unit,
    _assert_same_base,
    _find_a_name,
    _find_standard_unit,
    _fmt_exp,
    _format_display_parts,
    _get_display_parts,
    _merge_display_parts,
    _normalise_display_parts,
    _siprefixes,
    _standard_units,
    _unit_name_registry,
    add_standard_unit,
    parse_unit,
)
from saiunit._unit_common import *
from saiunit._unit_shortcuts import kHz, ms, mV, nS


# =========================================================================
# _siprefixes
# =========================================================================

class TestSIPrefixes:
    def test_kilo(self):
        assert _siprefixes["k"] == 3

    def test_milli(self):
        assert _siprefixes["m"] == -3

    def test_micro(self):
        assert _siprefixes["u"] == -6

    def test_empty_is_zero(self):
        assert _siprefixes[""] == 0

    def test_all_prefixes_are_integers(self):
        for v in _siprefixes.values():
            assert isinstance(v, int)


# =========================================================================
# UNITLESS
# =========================================================================

class TestUnitless:
    def test_unitless_is_dimensionless(self):
        assert UNITLESS.dim == DIMENSIONLESS

    def test_unitless_scale_zero(self):
        assert UNITLESS.scale == 0

    def test_unitless_factor_one(self):
        assert UNITLESS.factor == 1.0

    def test_unitless_is_unitless_property(self):
        assert UNITLESS.is_unitless


# =========================================================================
# Unit construction
# =========================================================================

class TestUnitConstruction:
    def test_default_construction(self):
        u = Unit()
        assert u.dim == DIMENSIONLESS
        assert u.scale == 0
        assert u.base == 10.0
        assert u.factor == 1.0

    def test_construction_with_dim(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d)
        assert u.dim is d

    def test_construction_non_dimension_raises(self):
        with pytest.raises(TypeError, match="Expected instance of Dimension"):
            Unit(dim=42)

    def test_name_default_from_dim(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d)
        assert u.name is not None
        assert not u.is_fullname

    def test_name_explicit(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d, name="metre", dispname="m")
        assert u.name == "metre"
        assert u.dispname == "m"

    def test_dispname_defaults_to_name(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d, name="metre")
        assert u.dispname == "metre"


# =========================================================================
# Unit properties
# =========================================================================

class TestUnitProperties:
    def test_factor_property(self):
        u = Unit(DIMENSIONLESS, factor=2.5)
        assert u.factor == 2.5

    def test_factor_setter_raises(self):
        u = Unit()
        with pytest.raises(NotImplementedError):
            u.factor = 1.0

    def test_base_property(self):
        u = Unit(DIMENSIONLESS, base=2.0)
        assert u.base == 2.0

    def test_base_setter_raises(self):
        u = Unit()
        with pytest.raises(NotImplementedError):
            u.base = 2.0

    def test_scale_property(self):
        u = Unit(DIMENSIONLESS, scale=3)
        assert u.scale == 3

    def test_scale_setter_raises(self):
        u = Unit()
        with pytest.raises(NotImplementedError):
            u.scale = 3

    def test_magnitude_property(self):
        u = Unit(DIMENSIONLESS, scale=3, base=10., factor=2.)
        assert u.magnitude == 2. * 10. ** 3

    def test_magnitude_setter_raises(self):
        u = Unit()
        with pytest.raises(NotImplementedError):
            u.magnitude = 1.0

    def test_dim_setter_raises(self):
        u = Unit()
        with pytest.raises(NotImplementedError):
            u.dim = DIMENSIONLESS

    def test_name_setter_raises(self):
        u = Unit()
        with pytest.raises(NotImplementedError):
            u.name = "foo"

    def test_dispname_setter_raises(self):
        u = Unit()
        with pytest.raises(NotImplementedError):
            u.dispname = "foo"

    def test_is_unitless_true(self):
        assert UNITLESS.is_unitless

    def test_is_unitless_false_nonzero_scale(self):
        u = Unit(DIMENSIONLESS, scale=3)
        assert not u.is_unitless

    def test_is_unitless_false_nonunit_dim(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d)
        assert not u.is_unitless


# =========================================================================
# Unit.create / Unit.create_scaled_unit
# =========================================================================

class TestUnitCreate:
    def test_create_registers_standard_unit(self):
        d = get_or_create_dimension([0, 0, 0, 0, 0, 0, 1])  # luminosity
        u = Unit.create(d, name="testcandle", dispname="tcd", scale=0)
        key = (d, 0, 10., 1.)
        assert key in _standard_units

    def test_create_scaled_unit(self):
        d = get_or_create_dimension([0, 0, 0, 0, 0, 0, 1])
        base = Unit.create(d, name="testlum", dispname="tlm", scale=0)
        scaled = Unit.create_scaled_unit(base, "m")
        assert scaled.name == "mtestlum"
        assert scaled.dispname == "mtlm"
        assert scaled.scale == -3

    def test_create_scaled_unit_invalid_prefix(self):
        base = Unit(DIMENSIONLESS, name="x", dispname="x")
        with pytest.raises(ValueError, match="Unknown SI prefix"):
            Unit.create_scaled_unit(base, "INVALID")


# =========================================================================
# Unit comparison / has_same_*
# =========================================================================

class TestUnitComparison:
    def test_eq_same(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, name="m", dispname="m", scale=0)
        u2 = Unit(d, name="m", dispname="m", scale=0)
        assert u1 == u2

    def test_eq_different_dim(self):
        d1 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = get_or_create_dimension([0, 1, 0, 0, 0, 0, 0])
        assert Unit(d1) != Unit(d2)

    def test_eq_different_scale(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, scale=0)
        u2 = Unit(d, scale=3)
        assert u1 != u2

    def test_eq_non_unit(self):
        assert Unit() != 42
        assert Unit() != "string"

    def test_ne(self):
        d1 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = get_or_create_dimension([0, 1, 0, 0, 0, 0, 0])
        assert Unit(d1) != Unit(d2)

    def test_has_same_magnitude(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, scale=3, base=10., factor=1.)
        u2 = Unit(d, scale=3, base=10., factor=1.)
        assert u1.has_same_magnitude(u2)

    def test_has_same_magnitude_different(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, scale=3)
        u2 = Unit(d, scale=6)
        assert not u1.has_same_magnitude(u2)

    def test_has_same_base(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, base=10.)
        u2 = Unit(d, base=10.)
        assert u1.has_same_base(u2)

    def test_has_same_base_different(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, base=10.)
        u2 = Unit(d, base=2.)
        assert not u1.has_same_base(u2)

    def test_has_same_dim(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, scale=0)
        u2 = Unit(d, scale=3)
        assert u1.has_same_dim(u2)

    def test_has_same_dim_different(self):
        d1 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = get_or_create_dimension([0, 1, 0, 0, 0, 0, 0])
        assert not Unit(d1).has_same_dim(Unit(d2))


# =========================================================================
# Unit arithmetic
# =========================================================================

class TestUnitArithmetic:
    def test_mul_units(self):
        d1 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = get_or_create_dimension([0, 0, -1, 0, 0, 0, 0])
        u1 = Unit(d1, name="metre", dispname="m", scale=0)
        u2 = Unit(d2, name="hertz", dispname="Hz", scale=0)
        result = u1 * u2
        assert result.dim == d1 * d2
        assert result.scale == 0

    def test_mul_different_base_raises(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, base=10.)
        u2 = Unit(d, base=2.)
        with pytest.raises(TypeError, match="different bases"):
            u1 * u2

    def test_mul_by_dimension_raises(self):
        u = Unit()
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        with pytest.raises(TypeError, match="cannot multiply"):
            u * d

    def test_mul_by_scalar_creates_quantity(self):
        from saiunit._base_quantity import Quantity
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d, name="metre", dispname="m")
        result = u * 3.0
        assert isinstance(result, Quantity)

    def test_rmul_scalar(self):
        from saiunit._base_quantity import Quantity
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d, name="metre", dispname="m")
        result = 3.0 * u
        assert isinstance(result, Quantity)

    def test_div_units(self):
        d1 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = get_or_create_dimension([0, 0, 1, 0, 0, 0, 0])
        u1 = Unit(d1, name="metre", dispname="m", scale=0)
        u2 = Unit(d2, name="second", dispname="s", scale=0)
        result = u1 / u2
        assert result.dim == d1 / d2

    def test_div_by_non_unit_raises(self):
        u = Unit()
        with pytest.raises(TypeError, match="cannot divide"):
            u / 3

    def test_rdiv_scalar(self):
        from saiunit._base_quantity import Quantity
        d = get_or_create_dimension([0, 0, 1, 0, 0, 0, 0])
        u = Unit(d, name="second", dispname="s")
        result = 1.0 / u
        assert isinstance(result, Quantity)

    def test_pow_scalar(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d, name="metre", dispname="m", scale=0)
        result = u ** 2
        assert result.dim == d ** 2

    def test_pow_non_scalar_raises(self):
        u = Unit(get_or_create_dimension([1, 0, 0, 0, 0, 0, 0]), name="m", dispname="m")
        with pytest.raises(TypeError, match="non-scalar"):
            u ** np.array([2, 3])

    def test_add_same_unit(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, name="m", dispname="m", scale=0)
        u2 = Unit(d, name="m", dispname="m", scale=0)
        result = u1 + u2
        assert result == u1

    def test_add_different_dim_raises(self):
        d1 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = get_or_create_dimension([0, 1, 0, 0, 0, 0, 0])
        with pytest.raises(TypeError, match="different dimensions"):
            Unit(d1, name="m", dispname="m") + Unit(d2, name="kg", dispname="kg")

    def test_add_different_scale_raises(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, name="m", dispname="m", scale=0)
        u2 = Unit(d, name="km", dispname="km", scale=3)
        with pytest.raises(TypeError, match="different units"):
            u1 + u2

    def test_add_non_unit_raises(self):
        with pytest.raises(TypeError, match="Expected a Unit"):
            Unit() + 42

    def test_sub_same_unit(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, name="m", dispname="m", scale=0)
        u2 = Unit(d, name="m", dispname="m", scale=0)
        result = u1 - u2
        assert result == u1

    def test_sub_non_unit_raises(self):
        with pytest.raises(TypeError, match="Expected a Unit"):
            Unit() - 42

    # ---- In-place raises ----

    def test_imul_raises(self):
        with pytest.raises(NotImplementedError):
            u = Unit()
            u *= u

    def test_idiv_raises(self):
        with pytest.raises(NotImplementedError):
            Unit().__idiv__(Unit())

    def test_itruediv_raises(self):
        with pytest.raises(NotImplementedError):
            u = Unit()
            u /= u

    def test_ipow_raises(self):
        with pytest.raises(NotImplementedError):
            u = Unit()
            u **= 2

    def test_iadd_raises(self):
        with pytest.raises(NotImplementedError):
            u = Unit()
            u += u

    def test_isub_raises(self):
        with pytest.raises(NotImplementedError):
            u = Unit()
            u -= u

    def test_floordiv_raises(self):
        with pytest.raises(NotImplementedError):
            Unit() // Unit()

    def test_rfloordiv_raises(self):
        with pytest.raises(NotImplementedError):
            Unit().__rfloordiv__(Unit())

    def test_ifloordiv_raises(self):
        with pytest.raises(NotImplementedError):
            Unit().__ifloordiv__(Unit())

    def test_mod_raises(self):
        with pytest.raises(NotImplementedError):
            Unit() % Unit()

    def test_rmod_raises(self):
        with pytest.raises(NotImplementedError):
            Unit().__rmod__(Unit())

    def test_imod_raises(self):
        with pytest.raises(NotImplementedError):
            Unit().__imod__(Unit())

    def test_abs(self):
        u = Unit()
        assert abs(u) is u


# =========================================================================
# Unit display
# =========================================================================

class TestUnitDisplay:
    def test_str_unitless(self):
        assert str(UNITLESS) == "1"

    def test_repr_unitless(self):
        assert repr(UNITLESS) == 'Unit("1")'

    def test_str_named(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d, name="metre", dispname="m", is_fullname=True)
        assert str(u) == "m"

    def test_repr_named(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d, name="metre", dispname="m", is_fullname=True)
        assert repr(u) == 'Unit("m")'

    def test_should_display_unit_for_physical(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d, name="metre", dispname="m")
        assert u.should_display_unit

    def test_should_display_unit_unitless(self):
        assert not UNITLESS.should_display_unit


# =========================================================================
# Unit copy / deepcopy / hash
# =========================================================================

class TestUnitCopyHash:
    def test_copy(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d, name="metre", dispname="m", scale=0)
        c = u.copy()
        assert c == u
        assert c is not u

    def test_deepcopy(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d, name="metre", dispname="m", scale=0)
        c = deepcopy(u)
        assert c == u

    def test_hash_same_units(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, name="metre", dispname="m", scale=0)
        u2 = Unit(d, name="metre", dispname="m", scale=0)
        assert hash(u1) == hash(u2)

    def test_hash_cached(self):
        u = Unit()
        h1 = hash(u)
        h2 = hash(u)
        assert h1 == h2

    def test_factorless(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d, name="metre", dispname="m", scale=0, factor=2.)
        fl = u.factorless()
        assert fl.factor == 1.
        assert fl.dim == u.dim
        assert fl.scale == u.scale

    def test_reverse(self):
        d = get_or_create_dimension([0, 0, 1, 0, 0, 0, 0])
        u = Unit(d, name="second", dispname="s", scale=0)
        r = u.reverse()
        assert r.dim == d ** -1
        assert r.scale == 0


# =========================================================================
# Pickle
# =========================================================================

class TestUnitPickle:
    def test_pickle_roundtrip(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d, name="metre", dispname="m", scale=0)
        data = pickle.dumps(u)
        u2 = pickle.loads(data)
        assert u == u2

    def test_pickle_unitless(self):
        data = pickle.dumps(UNITLESS)
        u2 = pickle.loads(data)
        assert u2 == UNITLESS


# =========================================================================
# Display-parts helpers
# =========================================================================

class TestDisplayParts:
    def test_get_display_parts_simple(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u = Unit(d, name="metre", dispname="m")
        parts = _get_display_parts(u)
        assert parts == [("metre", "m", 1)]

    def test_get_display_parts_with_cached(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        cached = [("metre", "m", 2)]
        u = Unit(d, name="m2", dispname="m^2", display_parts=cached)
        parts = _get_display_parts(u)
        assert parts == [("metre", "m", 2)]

    def test_merge_display_parts(self):
        parts_a = [("metre", "m", 1)]
        parts_b = [("second", "s", -1)]
        merged = _merge_display_parts(parts_a, parts_b)
        assert len(merged) == 2
        # positive exponent first
        assert merged[0][2] > 0
        assert merged[1][2] < 0

    def test_merge_display_parts_cancel(self):
        parts_a = [("metre", "m", 1)]
        parts_b = [("metre", "m", -1)]
        merged = _merge_display_parts(parts_a, parts_b)
        assert len(merged) == 0

    def test_normalise_display_parts_stacked_exp(self):
        parts = [("metre2", "m^2", 3)]
        normed = _normalise_display_parts(parts)
        assert len(normed) == 1
        assert normed[0][1] == "m"
        assert normed[0][2] == 6

    def test_normalise_drops_zero(self):
        parts = [("metre", "m", 0)]
        normed = _normalise_display_parts(parts)
        assert len(normed) == 0

    def test_fmt_exp_int(self):
        assert _fmt_exp(3.0) == "3"
        assert _fmt_exp(3) == "3"

    def test_fmt_exp_float(self):
        assert _fmt_exp(0.5) == "0.5"

    def test_format_display_parts_empty(self):
        assert _format_display_parts([]) == "1"

    def test_format_display_parts_single_positive(self):
        parts = [("metre", "m", 1)]
        assert _format_display_parts(parts) == "m"

    def test_format_display_parts_with_exponent(self):
        parts = [("metre", "m", 2)]
        assert _format_display_parts(parts) == "m^2"

    def test_format_display_parts_division(self):
        parts = [("metre", "m", 1), ("second", "s", -1)]
        s = _format_display_parts(parts)
        assert "m" in s
        assert "/" in s
        assert "s" in s

    def test_format_display_parts_numerator_only(self):
        parts = [("metre", "m", 1), ("second", "s", 1)]
        s = _format_display_parts(parts)
        assert "/" not in s
        assert "*" in s

    def test_assert_same_base_ok(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, base=10.)
        u2 = Unit(d, base=10.)
        _assert_same_base(u1, u2)  # should not raise

    def test_assert_same_base_raises(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        u1 = Unit(d, base=10.)
        u2 = Unit(d, base=2.)
        with pytest.raises(TypeError, match="different bases"):
            _assert_same_base(u1, u2)


# =========================================================================
# _find_standard_unit / _find_a_name / add_standard_unit
# =========================================================================

class TestStandardUnitLookup:
    def test_find_standard_unit_dimensionless(self):
        name, disp, is_full, is_dimless = _find_standard_unit(DIMENSIONLESS, 10., 0, 1.)
        assert is_dimless
        assert name is None

    def test_find_standard_unit_non_numeric_base(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        name, disp, is_full, is_dimless = _find_standard_unit(d, "non-numeric", 0, 1.)
        assert name is None
        assert not is_dimless

    def test_find_a_name_dimensionless(self):
        name, is_full = _find_a_name(DIMENSIONLESS, 10., 0, 1.)
        assert "Unit" in name
        assert not is_full

    def test_add_standard_unit_registers(self):
        from saiunit._base_unit import _standard_unit_aliases, _unit_name_registry, _ambiguous_keys
        d = get_or_create_dimension([0, 0, 0, 1, 0, 0, 0])
        key = (d, 0, 10., 1.)
        prev_standard = _standard_units.get(key)
        prev_aliases = list(_standard_unit_aliases.get(key, []))
        prev_tA = _unit_name_registry.get('tA')
        prev_teestamp = _unit_name_registry.get('teestamp')
        was_ambiguous = key in _ambiguous_keys
        try:
            u = Unit(d, name="teestamp", dispname="tA", scale=0, base=10., factor=1.)
            add_standard_unit(u)
            assert key in _standard_units
        finally:
            if prev_standard is None:
                _standard_units.pop(key, None)
            else:
                _standard_units[key] = prev_standard
            _standard_unit_aliases[key] = prev_aliases
            if prev_tA is None:
                _unit_name_registry.pop('tA', None)
            if prev_teestamp is None:
                _unit_name_registry.pop('teestamp', None)
            if not was_ambiguous:
                _ambiguous_keys.discard(key)


# =========================================================================
# Integration tests: Unit operations with real saiunit units
# (moved from _base_test.py TestUnit)
# =========================================================================

class TestUnitIntegration:
    def test_div(self):
        a = 1. * u.second
        b = 1. * u.ms
        _ = a / b

        a = 1. * u.ms
        _ = a / b

        c = u.ms / u.ms
        assert c.is_unitless

        _ = u.Unit((u.ms / u.ms).dim, scale=2)
        _ = u.Unit(u.ms.dim, scale=2)

    def test_mul_different_base_raises(self):
        a = u.Unit(base=2)
        b = u.Unit(base=10)
        with pytest.raises(TypeError):
            a * b

    def test_inplace_operations(self):
        # make sure that inplace operations do not work on units/dimensions
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
        assert_equal(str(u.kmeter / u.meter), '10.0^3')

    def test_inverse_second_prefers_hertz_alias(self):
        q = 1 / u.second
        assert_equal(str(q.unit), "Hz")
        assert_equal(repr(q.unit), 'Unit("Hz")')

        q_ms = 1 / u.ms
        assert_equal(str(q_ms.unit), "kHz")
        assert_equal(repr(q_ms.unit), 'Unit("kHz")')

    def test_compound_unit_multiplication_keeps_grouping(self):
        unit = (u.nA / (u.cm ** 2)) * u.mS
        assert_equal(repr(unit), 'Unit("mS * nA / cm^2")')
        assert_equal(str(unit), "mS * nA / cm^2")

        # Reversed operand order produces identical display (canonical sort)
        unit_rhs = u.mS * (u.nA / (u.cm ** 2))
        assert_equal(repr(unit_rhs), 'Unit("mS * nA / cm^2")')
        assert_equal(str(unit_rhs), "mS * nA / cm^2")

    def test_unit_with_factor(self):
        assert u.math.isclose(1. * u.eV / u.joule, 1.6021766e-19)
        assert u.math.isclose(1. * u.joule / u.eV, 6.241509074460762e18)


# =========================================================================
# str / repr smoke tests
# (moved from _base_test.py test_str_repr)
# =========================================================================

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

    # test the `DIMENSIONLESS` object
    assert str(DIMENSIONLESS) == "1"
    assert repr(DIMENSIONLESS) == "Dimension()"

    # test DimensionMismatchError (only that it works without raising an error)
    for error in [
        DimensionMismatchError("A description"),
        DimensionMismatchError("A description", DIMENSIONLESS),
        DimensionMismatchError("A description", DIMENSIONLESS, second.dim),
    ]:
        assert len(str(error))
        assert len(repr(error))


# =========================================================================
# Display redesign regression tests
# (moved from _base_test.py TestDisplayRedesign)
# =========================================================================

class TestDisplayRedesign:
    """Regression tests for the unified display convention.

    Convention: One canonical format using dispname symbols (``mV``,
    ``Hz``, ``kg``), ``^`` for exponentiation, `` * `` for
    multiplication, and `` / `` for division.

    - ``str(unit)``  -> canonical: ``"J / kg"``
    - ``repr(unit)`` -> ``Unit("J / kg")``
    - ``str(qty)``   -> ``"3.0 mV"``
    - ``repr(qty)``  -> ``Quantity(3.0, "mV")``
    """

    # --- Issue 1: Semantic-safe aliasing ---

    def test_joule_per_kg_not_sievert(self):
        """joule/kg must NOT auto-relabel as sievert."""
        unit = u.joule / u.kilogram
        assert "sievert" not in str(unit).lower()
        assert "gray" not in str(unit).lower()
        assert_equal(str(unit), "J / kg")
        assert_equal(repr(unit), 'Unit("J / kg")')

    def test_meter2_per_second2_not_sievert(self):
        """m^2/s^2 must NOT auto-relabel as sievert or gray."""
        unit = u.meter ** 2 / u.second ** 2
        assert "sievert" not in str(unit).lower()
        assert "gray" not in str(unit).lower()
        assert_equal(str(unit), "m^2 / s^2")
        assert_equal(repr(unit), 'Unit("m^2 / s^2")')

    def test_inverse_second_stays_hertz(self):
        """1/s^1 should still resolve to hertz (non-contextual)."""
        q = 1 / u.second
        assert_equal(str(q.unit), "Hz")
        assert_equal(repr(q.unit), 'Unit("Hz")')

    def test_becquerel_not_used_in_reverse(self):
        """Becquerel (contextual) must not appear via reverse()."""
        unit = u.second.reverse()
        assert "becquerel" not in str(unit).lower()
        assert_equal(str(unit), "Hz")

    # --- Issue 2: Deterministic composed-unit display ---

    def test_permutation_invariance_mul(self):
        """Operand order must not affect display for multiplication."""
        a = u.meter * u.second * u.amp
        b = u.amp * u.second * u.meter
        assert_equal(str(a), str(b))
        assert_equal(repr(a), repr(b))
        assert_equal(str(a), "A * m * s")

    def test_permutation_invariance_compound(self):
        """Compound units from different orderings must match."""
        a = (u.nA / u.cm ** 2) * u.mS
        b = u.mS * (u.nA / u.cm ** 2)
        assert_equal(str(a), str(b))
        assert_equal(repr(a), repr(b))

    def test_no_intermediate_simplification(self):
        """amp*second must NOT collapse to coulomb during composition."""
        unit = u.amp * u.second * u.meter
        s = str(unit)
        assert "coulomb" not in s.lower()
        assert_equal(s, "A * m * s")

    # --- Issue 3: Large-array repr validity ---

    def test_large_array_repr_balanced(self):
        """Large array repr must have balanced brackets."""
        big = jnp.arange(200) * u.mV
        r = repr(big)
        assert r.count("[") == r.count("]"), f"Unbalanced brackets: {r}"
        assert r.count("(") == r.count(")"), f"Unbalanced parens: {r}"
        assert "..." in r  # must use summarization
        assert "mV" in r  # unit still present (dispname)

    def test_large_array_str_balanced(self):
        """Large array str must have balanced brackets."""
        big = jnp.arange(200) * u.mV
        s = str(big)
        assert s.count("[") == s.count("]"), f"Unbalanced brackets: {s}"
        assert "..." in s
        assert "mV" in s

    # --- Issue 4: __format__ consistency ---

    def test_format_scalar_uses_dispname(self):
        """Scalar __format__ must use display symbol, not code name."""
        q = 1.5 * u.mV
        formatted = f"{q:.1f}"
        assert "mV" in formatted
        assert "mvolt" not in formatted

    def test_format_empty_spec_returns_str(self):
        """Empty format spec returns str(self)."""
        q = 1.5 * u.mV
        assert format(q, "") == str(q)

    # --- Issue 5: Unified display convention ---

    def test_str_human_oriented(self):
        """__str__ must use dispname (human-readable), not code name."""
        q = 10.0 * u.mV
        s = str(q)
        assert "mV" in s
        assert "mvolt" not in s

    def test_repr_shows_type(self):
        """__repr__ wraps canonical format in Quantity(...)."""
        q = 10.0 * u.mV
        r = repr(q)
        assert r.startswith("Quantity(")
        assert "mV" in r

    # --- Issue 6: display_in_unit is unified ---

    def test_display_in_unit_default_human(self):
        """display_in_unit returns canonical format."""
        s = display_in_unit(3.0 * u.volt, u.mvolt)
        assert "mV" in s
        assert "mvolt" not in s

    def test_display_in_unit_canonical(self):
        """display_in_unit always returns canonical format (no python_code param)."""
        s = display_in_unit(3.0 * u.volt, u.mvolt)
        assert_equal(s, "3000. mV")

    # --- Issue 7: Alias preference determinism ---

    def test_hertz_preferred_over_becquerel(self):
        """For s^-1, hertz must always win over becquerel."""
        unit = u.second.reverse()
        assert_equal(str(unit), "Hz")
        assert_equal(repr(unit), 'Unit("Hz")')

    def test_scaled_hertz_preferred(self):
        """kHz must be preferred over kBq for ms^-1."""
        unit = (1 / u.ms).unit
        assert_equal(str(unit), "kHz")
        assert_equal(repr(unit), 'Unit("kHz")')

    # --- Round-trip consistency ---

    def test_repr_str_consistency(self):
        """repr wraps str in Unit('...')."""
        for unit in [u.mV, u.joule / u.kilogram, u.nA / u.cm ** 2]:
            assert_equal(repr(unit), f'Unit("{str(unit)}")')

    def test_quantity_repr_str_consistency(self):
        """Quantity repr wraps value and unit string."""
        q = 3.0 * u.mV
        r = repr(q)
        assert r.startswith("Quantity(")
        assert 'mV' in r


# =========================================================================
# Display bug-fix tests
# (moved from _base_test.py TestDisplayBugFixes)
# =========================================================================

class TestDisplayBugFixes:
    """Tests for display bug fixes (2026-03)."""

    # --- Bug 1: __pow__ must not auto-alias to ambiguous units ---

    def test_pow_compound_not_gray(self):
        """(m/s)**2 must NOT become 'Gy' (gray)."""
        unit = (u.meter / u.second) ** 2
        assert "gray" not in str(unit).lower()
        assert "sievert" not in str(unit).lower()
        assert "Gy" not in str(unit)
        assert "Sv" not in str(unit)
        assert_equal(str(unit), "m^2 / s^2")

    def test_pow_compound_not_gray_repr(self):
        """repr((m/s)**2) must not contain gray."""
        unit = (u.meter / u.second) ** 2
        assert_equal(repr(unit), 'Unit("m^2 / s^2")')

    # --- Bug 2: __pow__ must not alias m^3 to kiloliter ---

    def test_meter_cubed_not_kliter(self):
        """m**3 must NOT auto-relabel as kl (kiloliter)."""
        unit = u.meter ** 3
        assert "kl" not in str(unit)
        assert "liter" not in str(unit).lower()
        assert_equal(str(unit), "m^3")

    def test_meter_cubed_consistent_with_mul(self):
        """m**3 and m*m*m must display identically."""
        assert_equal(str(u.meter ** 3), str(u.meter * u.meter * u.meter))

    def test_meter_squared_consistent_with_mul(self):
        """m**2 and m*m must display identically."""
        assert_equal(str(u.meter ** 2), str(u.meter * u.meter))

    def test_cm_cubed_via_pow(self):
        """cm**3 must show as 'cm^3', not some alias."""
        assert_equal(str(u.cmeter ** 3), "cm^3")

    # --- Bug 3: Exponent stacking ---

    def test_no_exponent_stacking(self):
        """(m^2/s)**3 must be 'm^6 / s^3', NOT 'm^2^3 / s^3'."""
        unit = (u.meter ** 2 / u.second) ** 3
        assert_equal(str(unit), "m^6 / s^3")

    def test_no_exponent_stacking_squared(self):
        """(m^2)**2 must be 'm^4', NOT 'm^2^2'."""
        assert_equal(str((u.meter ** 2) ** 2), "m^4")

    def test_no_exponent_stacking_cm(self):
        """(cm^2)**3 must be 'cm^6', NOT 'cm^2^3'."""
        assert_equal(str((u.cmeter ** 2) ** 3), "cm^6")

    def test_compound_pow_preserves_parts(self):
        """(m*s/A)^2 must show 'm^2 * s^2 / A^2'."""
        unit = (u.meter * u.second / u.amp) ** 2
        assert_equal(str(unit), "m^2 * s^2 / A^2")

    # --- Bug 4: __format__ for arrays ---

    def test_format_array_2f(self):
        """Array format '.2f' applies precision."""
        q = jnp.array([1.23456, 2.34567]) * u.mV
        result = format(q, ".2f")
        assert "mV" in result
        assert "1.23" in result

    def test_format_array_2e(self):
        """Array format '.2e' should apply precision."""
        q = jnp.array([1.23456, 2.34567]) * u.mV
        result = format(q, ".2e")
        assert "mV" in result

    def test_format_array_2g(self):
        """Array format '.2g' should apply precision."""
        q = jnp.array([1.23456, 2.34567]) * u.mV
        result = format(q, ".2g")
        assert "mV" in result

    def test_format_array_width_precision(self):
        """Array format '10.2f' should apply precision (2 digits)."""
        q = jnp.array([1.23456, 2.34567]) * u.mV
        result = format(q, "10.2f")
        assert "mV" in result

    def test_format_array_sign_precision(self):
        """Array format '+.2f' should apply precision."""
        q = jnp.array([1.23456, 2.34567]) * u.mV
        result = format(q, "+.2f")
        assert "mV" in result

    def test_format_array_bad_spec_fallback(self):
        """Array format with non-numeric spec falls through to str."""
        q = jnp.array([1.23, 2.34]) * u.mV
        result = format(q, "d")
        assert "mV" in result
        assert_equal(result, str(q))

    # --- Bug 5: Dimensionless unit display ---

    def test_dimensionless_unit_str(self):
        """str of dimensionless unit should be '1', not 'Unit(10.0^0)'."""
        unit = u.meter / u.meter
        assert_equal(str(unit), "1")

    def test_dimensionless_unit_repr(self):
        """repr of dimensionless unit should not have double 'Unit'."""
        unit = u.meter / u.meter
        r = repr(unit)
        assert_equal(r, 'Unit("1")')
        # No double 'Unit'
        assert r.count("Unit") == 1

    def test_dimensionless_scaled_str(self):
        """Dimensionless with scale should show base^scale."""
        unit = u.kmeter / u.meter
        assert_equal(str(unit), "10.0^3")

    def test_meter_pow_zero_str(self):
        """m**0 should show as '1'."""
        assert_equal(str(u.meter ** 0), "1")

    # --- Bug 6: Radian/steradian display ---

    def test_radian_str_shows_unit(self):
        """str(3.14 * radian) should show 'rad'."""
        q = 3.14 * u.radian
        s = str(q)
        assert "rad" in s
        assert_equal(s, "3.14 rad")

    def test_radian_repr_shows_unit(self):
        """repr(3.14 * radian) should show 'rad'."""
        q = 3.14 * u.radian
        r = repr(q)
        assert "rad" in r
        assert_equal(r, 'Quantity(3.14, "rad")')

    def test_steradian_str_shows_unit(self):
        """str(3.14 * steradian) should show 'sr'."""
        q = 3.14 * u.steradian
        assert_equal(str(q), "3.14 sr")

    def test_steradian_repr_shows_unit(self):
        """repr should include steradian display."""
        q = 3.14 * u.steradian
        assert_equal(repr(q), 'Quantity(3.14, "sr")')

    def test_radian_format_shows_unit(self):
        """__format__ on radian quantity shows 'rad'."""
        q = 3.14 * u.radian
        assert_equal(f"{q:.1f}", "3.1 rad")

    def test_plain_unitless_no_unit_shown(self):
        """Plain unitless quantity should NOT show any unit."""
        q = Quantity(3.14)
        assert_equal(str(q), "3.14")
        assert_equal(repr(q), "Quantity(3.14)")

    def test_division_dimensionless_no_unit(self):
        """m/m quantity should not show unit."""
        q = 3.0 * (u.meter / u.meter)
        assert_equal(str(q), "3.")
        assert_equal(repr(q), "Quantity(3.)")

    # --- Bug 7: % format blocked for physical units ---

    def test_percent_format_raises_for_physical_unit(self):
        """'%' format on mV quantity must raise ValueError."""
        q = 0.5 * u.mV
        with pytest.raises(ValueError, match="not supported"):
            f"{q:%}"

    def test_percent_format_ok_for_unitless(self):
        """'%' format on unitless quantity should work."""
        q = Quantity(0.5)
        assert_equal(f"{q:%}", "50.000000%")


# --- Docstring example tests ---


def test_docstring_example_unit_class():
    """Test the example from Unit class docstring."""
    import saiunit as u
    Nm = u.newton * u.metre
    # newton * metre simplifies to joule
    assert str(Nm) == 'J'
    q = 1.0 * Nm
    s = q.repr_in_unit(Nm)
    assert 'J' in s


def test_docstring_example_unit_is_unitless():
    """Test the example from Unit.is_unitless docstring."""
    import saiunit as u
    assert u.UNITLESS.is_unitless is True
    assert u.meter.is_unitless is False


def test_docstring_example_unit_dim():
    """Test the example from Unit.dim docstring."""
    import saiunit as u
    assert u.meter.dim == u.get_or_create_dimension(length=1)


def test_docstring_example_unit_has_same_dim():
    """Test the example from Unit.has_same_dim docstring."""
    import saiunit as u
    assert u.meter.has_same_dim(u.kmeter) is True
    assert u.meter.has_same_dim(u.second) is False


def test_docstring_example_unit_create():
    """Test the example from Unit.create docstring."""
    import saiunit as u
    dim = u.get_or_create_dimension(length=1)
    my_unit = u.Unit.create(dim, name='myunit', dispname='mu')
    assert my_unit.name == 'myunit'
    assert my_unit.dispname == 'mu'


def test_docstring_example_unit_create_scaled():
    """Test the example from Unit.create_scaled_unit docstring."""
    import saiunit as u
    assert u.mvolt.name == 'mvolt'
    assert u.mvolt.scale == -3 + u.volt.scale


def test_docstring_example_unitless():
    """Test the example from UNITLESS docstring."""
    import saiunit as u
    assert u.UNITLESS.is_unitless is True
    assert str(u.UNITLESS) == '1'


def test_docstring_example_add_standard_unit():
    """Test the example from add_standard_unit docstring."""
    import saiunit as u
    from saiunit._base_unit import (
        _standard_units, _standard_unit_aliases, _unit_name_registry, _ambiguous_keys
    )
    dim = u.get_or_create_dimension(length=1, time=-1)
    key = (dim, 0, 10., 1.)
    # Save state before registration
    prev_standard = _standard_units.get(key)
    prev_aliases = list(_standard_unit_aliases.get(key, []))
    prev_tv = _unit_name_registry.get('tv')
    prev_testvel = _unit_name_registry.get('testvel')
    was_ambiguous = key in _ambiguous_keys
    try:
        vel_unit = u.Unit(dim, name='testvel', dispname='tv', scale=0, base=10., factor=1.)
        u.add_standard_unit(vel_unit)
        assert key in _standard_units
    finally:
        # Restore global state
        if prev_standard is None:
            _standard_units.pop(key, None)
        else:
            _standard_units[key] = prev_standard
        _standard_unit_aliases[key] = prev_aliases
        if prev_tv is None:
            _unit_name_registry.pop('tv', None)
        if prev_testvel is None:
            _unit_name_registry.pop('testvel', None)
        if not was_ambiguous:
            _ambiguous_keys.discard(key)


# ---------------------------------------------------------------------------
# parse_unit / Unit(str) tests
# ---------------------------------------------------------------------------


class TestParseUnit:
    """Tests for string-based Unit construction."""

    # --- Simple dispname lookup ---
    def test_simple_dispname(self):
        assert Unit("mV") == mvolt

    def test_base_unit_dispname(self):
        assert Unit("V") == volt

    def test_si_prefixed(self):
        assert Unit("kHz") == khertz

    # --- Full name lookup ---
    def test_fullname(self):
        assert Unit("mvolt") == mvolt

    def test_fullname_base(self):
        assert Unit("volt") == volt

    # --- Exponents ---
    def test_positive_exponent(self):
        assert Unit("m^2") == metre ** 2

    def test_negative_exponent(self):
        assert Unit("s^-1") == second ** -1

    def test_exponent_3(self):
        assert Unit("m^3") == metre ** 3

    # --- Division ---
    def test_simple_division(self):
        assert Unit("J / kg") == joule / kilogram

    def test_division_with_exponent(self):
        assert Unit("nA / cm^2") == namp / cmetre ** 2

    # --- Multiplication + division ---
    def test_product_and_division(self):
        parsed = Unit("mS * nA / cm^2")
        expected = msiemens * namp / cmetre ** 2
        assert parsed == expected

    # --- Parenthesized denominator ---
    def test_parenthesized_denominator(self):
        parsed = Unit("m / (kg * s^2)")
        expected = metre / (kilogram * second ** 2)
        assert parsed == expected

    # --- Dimensionless ---
    def test_unitless(self):
        assert Unit("1") == UNITLESS

    def test_dimensionless_scaled(self):
        u = Unit("10^3")
        assert u.dim == DIMENSIONLESS
        assert u.base == 10.0
        assert u.scale == 3

    # --- Repr wrapper ---
    def test_from_repr_double_quote(self):
        assert Unit('Unit("mV")') == mvolt

    def test_from_repr_single_quote(self):
        assert Unit("Unit('mV')") == mvolt

    # --- Roundtrip for common units ---
    @pytest.mark.parametrize("unit", [
        volt, mvolt, hertz, metre, kilogram, second, amp,
        joule, watt, newton, pascal, ohm, siemens, farad,
        coulomb, weber, tesla, henry, lumen, lux, katal,
    ])
    def test_roundtrip_named(self, unit):
        assert Unit(str(unit)) == unit

    def test_roundtrip_compound(self):
        compound = mvolt / (kilogram * second)
        assert Unit(str(compound)) == compound

    def test_roundtrip_all_standard_units(self):
        """Every registered standard unit should roundtrip.

        Units whose dispname collides with another unit (e.g. survey_foot
        and foot both display as 'ft') are skipped because the parser
        always resolves to the first-registered unit for a given dispname.
        """
        for key, unit in _standard_units.items():
            s = str(unit)
            parsed = Unit(s)
            if parsed != unit:
                # Verify the mismatch is due to a dispname collision,
                # not a genuine parser bug.
                assert s in _unit_name_registry
                assert _unit_name_registry[s] == parsed

    # --- Error handling ---
    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            Unit("nonexistent_unit_xyz")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            Unit("")

    # --- parse_unit function directly ---
    def test_parse_unit_function(self):
        assert parse_unit("mV") == mvolt

    def test_registry_populated(self):
        assert "mV" in _unit_name_registry
        assert "volt" in _unit_name_registry
        assert "V" in _unit_name_registry
