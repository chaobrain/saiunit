"""Tests for _base_dimension.py: Dimension, DIMENSIONLESS, error classes, helpers."""

from __future__ import annotations

import pickle
from copy import deepcopy

import jax
import numpy as np
import pytest

from saiunit._base_dimension import (
    DIMENSIONLESS,
    Dimension,
    DimensionMismatchError,
    UnitMismatchError,
    _dim2index,
    _iclass_label,
    _ilabel,
    _is_tracer,
    _dimension_cache,
    get_dim_for_display,
    get_or_create_dimension,
)


# =========================================================================
# _dim2index / _ilabel / _iclass_label constants
# =========================================================================

class TestConstants:
    def test_dim2index_length_aliases(self):
        for alias in ("Length", "length", "metre", "metres", "meter", "meters", "m"):
            assert _dim2index[alias] == 0

    def test_dim2index_mass_aliases(self):
        for alias in ("Mass", "mass", "kilogram", "kilograms", "kg"):
            assert _dim2index[alias] == 1

    def test_dim2index_time_aliases(self):
        for alias in ("Time", "time", "second", "seconds", "s"):
            assert _dim2index[alias] == 2

    def test_dim2index_current_aliases(self):
        for alias in ("Electric Current", "electric current", "Current", "current", "ampere", "amperes", "A"):
            assert _dim2index[alias] == 3

    def test_dim2index_temperature_aliases(self):
        for alias in ("Temperature", "temperature", "kelvin", "kelvins", "K"):
            assert _dim2index[alias] == 4

    def test_dim2index_substance_aliases(self):
        for alias in ("Substance", "substance", "mole", "moles", "mol"):
            assert _dim2index[alias] == 5

    def test_dim2index_luminosity_aliases(self):
        for alias in ("Luminosity", "luminosity", "candle", "candles", "cd"):
            assert _dim2index[alias] == 6

    def test_ilabel_order(self):
        assert _ilabel == ["m", "kg", "s", "A", "K", "mol", "cd"]

    def test_iclass_label_order(self):
        assert _iclass_label == ["metre", "kilogram", "second", "amp", "kelvin", "mole", "candle"]

    def test_seven_dimensions(self):
        assert len(_ilabel) == 7
        assert len(_iclass_label) == 7


# =========================================================================
# _is_tracer
# =========================================================================

class TestIsTracer:
    def test_plain_values_not_tracers(self):
        assert not _is_tracer(1)
        assert not _is_tracer(1.0)
        assert not _is_tracer(np.array(1))

    def test_shape_dtype_struct_is_tracer(self):
        assert _is_tracer(jax.ShapeDtypeStruct((2, 3), np.float32))

    def test_shaped_array_is_tracer(self):
        assert _is_tracer(jax.core.ShapedArray((2,), np.float32))


# =========================================================================
# Dimension class
# =========================================================================

class TestDimension:
    def test_init_and_dims(self):
        d = Dimension([1, 0, -2, 0, 0, 0, 0])
        assert d._dims[0] == 1
        assert d._dims[2] == -2

    def test_get_dimension(self):
        d = get_or_create_dimension(m=1, kg=1, s=-2)
        assert d.get_dimension("m") == 1
        assert d.get_dimension("kg") == 1
        assert d.get_dimension("s") == -2
        assert d.get_dimension("A") == 0

    def test_is_dimensionless(self):
        assert DIMENSIONLESS.is_dimensionless
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        assert not d.is_dimensionless

    def test_dim_property_returns_self(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        assert d.dim is d

    def test_str_dimensionless(self):
        assert str(DIMENSIONLESS) == "1"

    def test_str_single_dimension(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        assert str(d) == "m"

    def test_str_compound_dimension(self):
        d = get_or_create_dimension([1, 1, -2, 0, 0, 0, 0])
        assert "m" in str(d)
        assert "kg" in str(d)
        assert "s^-2" in str(d)

    def test_repr_dimensionless(self):
        assert repr(DIMENSIONLESS) == "Dimension()"

    def test_repr_single_dimension(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        assert repr(d) == "metre"

    def test_repr_compound_dimension(self):
        d = get_or_create_dimension([1, 1, -2, 0, 0, 0, 0])
        r = repr(d)
        assert "metre" in r
        assert "kilogram" in r
        assert "second ** -2" in r

    # ---- Arithmetic ----

    def test_mul(self):
        length = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        time = get_or_create_dimension([0, 0, 1, 0, 0, 0, 0])
        result = length * time
        assert result == get_or_create_dimension([1, 0, 1, 0, 0, 0, 0])

    def test_mul_type_error(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        with pytest.raises(TypeError, match="Can only multiply"):
            d * 2

    def test_div(self):
        length = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        time = get_or_create_dimension([0, 0, 1, 0, 0, 0, 0])
        result = length / time
        assert result == get_or_create_dimension([1, 0, -1, 0, 0, 0, 0])

    def test_div_type_error(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        with pytest.raises(TypeError, match="Can only divide"):
            d / 2

    def test_truediv_same_as_div(self):
        length = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        time = get_or_create_dimension([0, 0, 1, 0, 0, 0, 0])
        assert length.__truediv__(time) == length.__div__(time)

    def test_pow(self):
        length = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        area = length ** 2
        assert area == get_or_create_dimension([2, 0, 0, 0, 0, 0, 0])

    def test_pow_with_tracer_raises(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        tracer = jax.ShapeDtypeStruct((), np.float32)
        with pytest.raises(TypeError, match="Cannot use a tracer"):
            d ** tracer

    def test_pow_too_many_exponents(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        with pytest.raises(TypeError, match="Too many exponents"):
            d ** np.array([2, 3])

    # ---- In-place operations raise ----

    def test_imul_raises(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        with pytest.raises(NotImplementedError):
            d *= d

    def test_idiv_raises(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        with pytest.raises(NotImplementedError):
            d.__idiv__(d)

    def test_itruediv_raises(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        with pytest.raises(NotImplementedError):
            d.__itruediv__(d)

    def test_ipow_raises(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        with pytest.raises(NotImplementedError):
            d **= 2

    # ---- Comparison ----

    def test_eq_same(self):
        d1 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        assert d1 == d2

    def test_eq_different(self):
        d1 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = get_or_create_dimension([0, 1, 0, 0, 0, 0, 0])
        assert not (d1 == d2)

    def test_eq_non_dimension(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        assert d != "not a dimension"
        assert d != 42

    def test_ne(self):
        d1 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = get_or_create_dimension([0, 1, 0, 0, 0, 0, 0])
        assert d1 != d2

    # ---- Hash ----

    def test_hash_cached(self):
        d = Dimension([1, 0, 0, 0, 0, 0, 0])
        h1 = hash(d)
        h2 = hash(d)
        assert h1 == h2

    def test_hash_setter_raises(self):
        d = Dimension([1, 0, 0, 0, 0, 0, 0])
        with pytest.raises(ValueError, match="Cannot set hash"):
            d.hash = 42

    def test_hash_equal_dimensions_same_hash(self):
        d1 = Dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = Dimension([1, 0, 0, 0, 0, 0, 0])
        assert hash(d1) == hash(d2)

    # ---- Pickling ----

    def test_pickle_roundtrip(self):
        d = get_or_create_dimension([1, 0, -2, 0, 0, 0, 0])
        data = pickle.dumps(d)
        d2 = pickle.loads(data)
        assert d == d2

    def test_pickle_preserves_singleton(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        data = pickle.dumps(d)
        d2 = pickle.loads(data)
        assert d is d2

    def test_deepcopy_returns_self(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = deepcopy(d)
        assert d is d2


# =========================================================================
# get_or_create_dimension
# =========================================================================

class TestGetOrCreateDimension:
    def test_by_list(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        assert d.get_dimension("m") == 1

    def test_by_keywords(self):
        d = get_or_create_dimension(m=1, kg=1, s=-2)
        assert d.get_dimension("m") == 1
        assert d.get_dimension("kg") == 1
        assert d.get_dimension("s") == -2

    def test_singleton_by_list(self):
        d1 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        assert d1 is d2

    def test_singleton_by_keywords(self):
        d1 = get_or_create_dimension(m=1)
        d2 = get_or_create_dimension(length=1)
        assert d1 is d2

    def test_wrong_number_of_items(self):
        with pytest.raises(TypeError, match="exactly 7"):
            get_or_create_dimension([1, 0, 0])

    def test_too_many_positional_args(self):
        with pytest.raises(TypeError, match="at most 1 positional"):
            get_or_create_dimension([1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0])

    def test_non_sequence_raises(self):
        with pytest.raises(TypeError, match="exactly 7"):
            get_or_create_dimension(42)

    def test_cache_populated(self):
        d = get_or_create_dimension([0, 0, 0, 0, 1, 0, 0])
        key = tuple(np.asarray([0, 0, 0, 0, 1, 0, 0]))
        assert key in _dimension_cache
        assert _dimension_cache[key] is d


# =========================================================================
# DIMENSIONLESS
# =========================================================================

class TestDimensionless:
    def test_is_dimensionless(self):
        assert DIMENSIONLESS.is_dimensionless

    def test_str(self):
        assert str(DIMENSIONLESS) == "1"

    def test_repr(self):
        assert repr(DIMENSIONLESS) == "Dimension()"


# =========================================================================
# get_dim_for_display
# =========================================================================

class TestGetDimForDisplay:
    def test_dimensionless_constant(self):
        assert get_dim_for_display(DIMENSIONLESS) == "1"

    def test_int_one(self):
        assert get_dim_for_display(1) == "1"

    def test_dimension_object(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        assert get_dim_for_display(d) == "m"

    def test_compound_dimension(self):
        d = get_or_create_dimension([1, 0, -1, 0, 0, 0, 0])
        s = get_dim_for_display(d)
        assert "m" in s
        assert "s" in s


# =========================================================================
# DimensionMismatchError
# =========================================================================

class TestDimensionMismatchError:
    def test_no_dims(self):
        e = DimensionMismatchError("mismatch")
        assert "mismatch" in str(e)
        assert e.dims == ()

    def test_one_dim(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        e = DimensionMismatchError("mismatch", d)
        s = str(e)
        assert "mismatch" in s
        assert "m" in s

    def test_two_dims(self):
        d1 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = get_or_create_dimension([0, 1, 0, 0, 0, 0, 0])
        e = DimensionMismatchError("mismatch", d1, d2)
        s = str(e)
        assert "mismatch" in s
        assert "m" in s
        assert "kg" in s

    def test_three_dims(self):
        d1 = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        d2 = get_or_create_dimension([0, 1, 0, 0, 0, 0, 0])
        d3 = get_or_create_dimension([0, 0, 1, 0, 0, 0, 0])
        e = DimensionMismatchError("mismatch", d1, d2, d3)
        s = str(e)
        assert "m" in s
        assert "kg" in s
        assert "s" in s

    def test_repr(self):
        d = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
        e = DimensionMismatchError("test", d)
        r = repr(e)
        assert "DimensionMismatchError" in r

    def test_pickle(self):
        e = DimensionMismatchError("test", DIMENSIONLESS)
        data = pickle.dumps(e)
        e2 = pickle.loads(data)
        assert str(e) == str(e2)


# =========================================================================
# UnitMismatchError
# =========================================================================

class TestUnitMismatchError:
    def test_no_units(self):
        e = UnitMismatchError("mismatch")
        assert "mismatch" in str(e)
        assert e.units == ()

    def test_one_unit(self):
        from saiunit._base_unit import Unit
        u = Unit(get_or_create_dimension([1, 0, 0, 0, 0, 0, 0]), name="metre", dispname="m")
        e = UnitMismatchError("mismatch", u)
        s = str(e)
        assert "mismatch" in s

    def test_two_units(self):
        from saiunit._base_unit import Unit
        u1 = Unit(get_or_create_dimension([1, 0, 0, 0, 0, 0, 0]), name="metre", dispname="m")
        u2 = Unit(get_or_create_dimension([0, 1, 0, 0, 0, 0, 0]), name="kilogram", dispname="kg")
        e = UnitMismatchError("mismatch", u1, u2)
        s = str(e)
        assert "mismatch" in s

    def test_repr(self):
        e = UnitMismatchError("test")
        r = repr(e)
        assert "UnitMismatchError" in r

    def test_pickle(self):
        e = UnitMismatchError("test")
        data = pickle.dumps(e)
        e2 = pickle.loads(data)
        assert str(e) == str(e2)


# --- Docstring example tests ---


def test_docstring_example_dimension_class():
    """Test the example from Dimension class docstring."""
    import saiunit as u
    length_dim = u.meter.dim
    assert length_dim.get_dimension('m') == 1.0
    assert not length_dim.is_dimensionless
    assert u.DIMENSIONLESS.is_dimensionless


def test_docstring_example_is_dimensionless():
    """Test the example from Dimension.is_dimensionless docstring."""
    import saiunit as u
    assert u.DIMENSIONLESS.is_dimensionless is True
    assert u.meter.dim.is_dimensionless is False


def test_docstring_example_dim_property():
    """Test the example from Dimension.dim docstring."""
    import saiunit as u
    d = u.meter.dim
    assert d.dim is d


def test_docstring_example_get_or_create_dimension():
    """Test the example from get_or_create_dimension docstring."""
    import saiunit as u
    d1 = u.get_or_create_dimension(length=1, mass=1, time=-2)
    d2 = u.get_or_create_dimension(m=1, kg=1, s=-2)
    d3 = u.get_or_create_dimension([1, 1, -2, 0, 0, 0, 0])
    assert d1 is d2
    assert d2 is d3
    assert repr(d1) == "metre * kilogram * second ** -2"


def test_docstring_example_dimensionless():
    """Test the example from DIMENSIONLESS docstring."""
    import saiunit as u
    assert u.DIMENSIONLESS.is_dimensionless is True
    assert str(u.DIMENSIONLESS) == '1'


def test_docstring_example_get_dim_for_display():
    """Test the example from get_dim_for_display docstring."""
    import saiunit as u
    assert u.get_dim_for_display(u.DIMENSIONLESS) == '1'
    assert u.get_dim_for_display(u.meter.dim) == 'm'


def test_docstring_example_dimension_mismatch_error():
    """Test the example from DimensionMismatchError docstring."""
    import saiunit as u
    e = u.DimensionMismatchError("Addition", u.meter.dim, u.second.dim)
    assert 'Addition' in str(e)
    assert 'm' in str(e)


def test_docstring_example_unit_mismatch_error():
    """Test the example from UnitMismatchError docstring."""
    import saiunit as u
    e = u.UnitMismatchError("Addition", u.mvolt, u.volt)
    assert 'Addition' in str(e)
