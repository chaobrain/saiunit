"""Tests for _base_decorators.py: check_dims, check_units, assign_units, helpers."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from saiunit._base_decorators import (
    CallableAssignUnit,
    Missing,
    _assign_unit,
    _check_dim,
    _check_unit,
    _is_quantity,
    _remove_unit,
    assign_units,
    check_dims,
    check_units,
    missing,
)
from saiunit._base_dimension import (
    DIMENSIONLESS,
    DimensionMismatchError,
    UnitMismatchError,
    get_or_create_dimension,
)
from saiunit._base_quantity import Quantity
from saiunit._base_unit import UNITLESS, Unit


# helpers: build common units/dims once
_length_dim = get_or_create_dimension([1, 0, 0, 0, 0, 0, 0])
_time_dim = get_or_create_dimension([0, 0, 1, 0, 0, 0, 0])
_current_dim = get_or_create_dimension([0, 0, 0, 1, 0, 0, 0])
_voltage_dim = get_or_create_dimension([2, 1, -3, -1, 0, 0, 0])
_metre = Unit(_length_dim, name="metre", dispname="m", scale=0)
_second = Unit(_time_dim, name="second", dispname="s", scale=0)
_amp = Unit(_current_dim, name="amp", dispname="A", scale=0)
_volt = Unit(_voltage_dim, name="volt", dispname="V", scale=0)
_msecond = Unit(_time_dim, name="msecond", dispname="ms", scale=-3)


# =========================================================================
# _is_quantity
# =========================================================================

class TestIsQuantity:
    def test_quantity_true(self):
        q = Quantity(5.0, unit=_metre)
        assert _is_quantity(q)

    def test_scalar_false(self):
        assert not _is_quantity(5.0)

    def test_unit_false(self):
        assert not _is_quantity(_metre)

    def test_none_false(self):
        assert not _is_quantity(None)


# =========================================================================
# check_dims
# =========================================================================

class TestCheckDims:
    def test_basic_check_passes(self):
        @check_dims(x=_length_dim)
        def f(x):
            return x

        q = Quantity(5.0, unit=_metre)
        result = f(q)
        assert result.unit == _metre

    def test_wrong_dim_raises(self):
        @check_dims(x=_length_dim)
        def f(x):
            return x

        q = Quantity(5.0, unit=_second)
        with pytest.raises(DimensionMismatchError):
            f(q)

    def test_none_arg_skipped(self):
        @check_dims(x=_length_dim)
        def f(x=None):
            return x

        result = f()
        assert result is None

    def test_string_arg_skipped(self):
        @check_dims(x=_length_dim)
        def f(x="hello"):
            return x

        result = f()
        assert result == "hello"

    def test_bool_check(self):
        @check_dims(x=bool)
        def f(x):
            return x

        assert f(True) is True
        with pytest.raises(TypeError, match="boolean"):
            f(5)

    def test_result_check_passes(self):
        @check_dims(result=_length_dim)
        def f():
            return Quantity(5.0, unit=_metre)

        result = f()
        assert result.unit == _metre

    def test_result_check_fails(self):
        @check_dims(result=_length_dim)
        def f():
            return Quantity(5.0, unit=_second)

        with pytest.raises(DimensionMismatchError):
            f()

    def test_result_callable(self):
        @check_dims(result=lambda d: d ** 2)
        def square(x):
            return x ** 2

        q = Quantity(3.0, unit=_metre)
        result = square(q)
        assert result.dim == _length_dim ** 2

    def test_same_dim_string_reference(self):
        @check_dims(a=None, b='a')
        def f(a, b):
            return a + b

        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_metre)
        result = f(q1, q2)
        assert jnp.allclose(result.mantissa, 3.0)

    def test_same_dim_string_reference_mismatch(self):
        @check_dims(a=None, b='a')
        def f(a, b):
            return a + b

        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_second)
        with pytest.raises(DimensionMismatchError):
            f(q1, q2)

    def test_same_dim_string_reference_missing(self):
        @check_dims(a=None, b='nonexistent')
        def f(a, b):
            return a

        with pytest.raises(TypeError, match="no argument of that name"):
            f(Quantity(1.0, unit=_metre), Quantity(1.0, unit=_metre))

    def test_stores_metadata(self):
        @check_dims(x=_length_dim, result=_length_dim)
        def f(x):
            return x

        assert hasattr(f, '_arg_names')
        assert hasattr(f, '_arg_units')
        assert hasattr(f, '_return_unit')
        assert hasattr(f, '_orig_func')
        assert f._return_unit == _length_dim

    def test_bool_return_flag(self):
        @check_dims(result=bool)
        def f():
            return True

        assert f._returns_bool is True

    def test_list_arg_converted_to_quantity(self):
        @check_dims(x=_length_dim)
        def f(x):
            return x

        # List of scalars with matching dim constraint should raise
        # because a plain list becomes dimensionless Quantity
        with pytest.raises(DimensionMismatchError):
            f([1.0, 2.0])

    def test_unchecked_args_pass_through(self):
        @check_dims(x=_length_dim)
        def f(x, y):
            return x

        q = Quantity(5.0, unit=_metre)
        result = f(q, "unchecked")
        assert result.unit == _metre


# =========================================================================
# check_units
# =========================================================================

class TestCheckUnits:
    def test_basic_check_passes(self):
        @check_units(x=_metre)
        def f(x):
            return x

        q = Quantity(5.0, unit=_metre)
        result = f(q)
        assert result.unit == _metre

    def test_wrong_unit_raises(self):
        @check_units(x=_metre)
        def f(x):
            return x

        q = Quantity(5.0, unit=_second)
        with pytest.raises(UnitMismatchError):
            f(q)

    def test_bool_check(self):
        @check_units(x=bool)
        def f(x):
            return x

        assert f(True) is True
        with pytest.raises(TypeError, match="boolean"):
            f(5)

    def test_string_reference_same_unit(self):
        @check_units(a=None, b='a')
        def f(a, b):
            return a + b

        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_metre)
        result = f(q1, q2)
        assert jnp.allclose(result.mantissa, 3.0)

    def test_string_reference_different_unit_raises(self):
        @check_units(a=None, b='a')
        def f(a, b):
            return a

        q1 = Quantity(1.0, unit=_metre)
        q2 = Quantity(2.0, unit=_second)
        with pytest.raises(UnitMismatchError):
            f(q1, q2)

    def test_result_check_passes(self):
        @check_units(result=_metre)
        def f():
            return Quantity(5.0, unit=_metre)

        result = f()
        assert result.unit == _metre

    def test_result_check_fails(self):
        @check_units(result=_metre)
        def f():
            return Quantity(5.0, unit=_second)

        with pytest.raises(UnitMismatchError):
            f()

    def test_stores_metadata(self):
        @check_units(x=_metre, result=_metre)
        def f(x):
            return x

        assert hasattr(f, '_arg_names')
        assert hasattr(f, '_return_unit')


# =========================================================================
# assign_units
# =========================================================================

class TestAssignUnits:
    def test_basic_strip_units(self):
        @assign_units(x=_metre)
        def f(x):
            return x * 2

        q = Quantity(5.0, unit=_metre)
        result = f(q)
        # x should be stripped to scalar, result is scalar
        assert jnp.allclose(result, 10.0)

    def test_strip_and_assign_result(self):
        @assign_units(x=_metre, result=_metre)
        def f(x):
            return x * 2

        q = Quantity(5.0, unit=_metre)
        result = f(q)
        assert isinstance(result, Quantity)
        assert result.unit == _metre

    def test_partial_application(self):
        decorator = assign_units(x=_metre)
        assert callable(decorator)

        @decorator
        def f(x):
            return x

        result = f(Quantity(5.0, unit=_metre))
        assert jnp.allclose(result, 5.0)

    def test_without_result_units(self):
        @assign_units(x=_metre, result=_metre)
        def f(x):
            return x * 2

        q = Quantity(5.0, unit=_metre)
        result = f.without_result_units(q)
        # without_result_units skips the result unit assignment
        assert not isinstance(result, Quantity) or result.is_unitless

    def test_bool_arg(self):
        @assign_units(flag=bool)
        def f(flag):
            return flag

        assert f(True) is True
        with pytest.raises(TypeError, match="boolean"):
            f(5)

    def test_unitless_arg(self):
        @assign_units(x=1)
        def f(x):
            return x

        assert f(5.0) == 5.0
        with pytest.raises(TypeError, match="Number"):
            f(Quantity(5.0, unit=_metre))

    def test_non_quantity_for_unit_arg_raises(self):
        @assign_units(x=_metre)
        def f(x):
            return x

        with pytest.raises(TypeError, match="Quantity"):
            f(5.0)

    def test_none_arg_passthrough(self):
        @assign_units(x=_metre)
        def f(x=None):
            return x

        assert f() is None

    def test_multiple_results(self):
        @assign_units(result=(_second, _volt))
        def f():
            return 5, 3

        r = f()
        assert isinstance(r[0], Quantity)
        assert isinstance(r[1], Quantity)
        assert r[0].unit == _second
        assert r[1].unit == _volt

    def test_dict_results(self):
        @assign_units(result={'a': _second, 'b': _volt})
        def f():
            return {'a': 5, 'b': 3}

        r = f()
        assert isinstance(r['a'], Quantity)
        assert isinstance(r['b'], Quantity)

    def test_result_type_mismatch_raises(self):
        @assign_units(result=(_second, _volt))
        def f():
            return 5  # not a tuple

        with pytest.raises(TypeError, match="pytree"):
            f()

    def test_result_callable_dynamic(self):
        @assign_units(result=lambda u: u ** 2)
        def f(x):
            return x.mantissa ** 2

        q = Quantity(3.0, unit=_metre)
        result = f(q)
        assert isinstance(result, Quantity)


# =========================================================================
# _remove_unit
# =========================================================================

class TestRemoveUnit:
    def test_none_passthrough(self):
        assert _remove_unit("f", "x", None, 5.0) == 5.0

    def test_bool_passes(self):
        assert _remove_unit("f", "x", bool, True) is True

    def test_bool_wrong_type_raises(self):
        with pytest.raises(TypeError, match="boolean"):
            _remove_unit("f", "x", bool, 5)

    def test_unit_strips_quantity(self):
        q = Quantity(5.0, unit=_metre)
        result = _remove_unit("f", "x", _metre, q)
        assert jnp.allclose(result, 5.0)

    def test_unit_non_quantity_raises(self):
        with pytest.raises(TypeError, match="Quantity"):
            _remove_unit("f", "x", _metre, 5.0)

    def test_one_passes_scalar(self):
        assert _remove_unit("f", "x", 1, 5.0) == 5.0

    def test_one_rejects_quantity(self):
        q = Quantity(5.0, unit=_metre)
        with pytest.raises(TypeError, match="Number"):
            _remove_unit("f", "x", 1, q)

    def test_invalid_unit_raises(self):
        with pytest.raises(TypeError, match="target unit"):
            _remove_unit("f", "x", "invalid", 5.0)


# =========================================================================
# _check_dim / _check_unit
# =========================================================================

class TestCheckDimHelper:
    def test_matching_dim(self):
        q = Quantity(5.0, unit=_metre)
        _check_dim(lambda: None, q, _length_dim)  # should not raise

    def test_mismatching_dim_raises(self):
        q = Quantity(5.0, unit=_metre)
        func = lambda: None
        func.__name__ = "test_func"
        with pytest.raises(DimensionMismatchError, match="test_func"):
            _check_dim(func, q, _time_dim)

    def test_none_means_dimensionless(self):
        q = Quantity(5.0, unit=UNITLESS)
        _check_dim(lambda: None, q, None)  # should not raise


class TestCheckUnitHelper:
    def test_matching_unit(self):
        q = Quantity(5.0, unit=_metre)
        func = lambda: None
        func.__name__ = "test_func"
        _check_unit(func, q, _metre)  # should not raise

    def test_mismatching_unit_raises(self):
        q = Quantity(5.0, unit=_metre)
        func = lambda: None
        func.__name__ = "test_func"
        with pytest.raises(UnitMismatchError, match="test_func"):
            _check_unit(func, q, _second)

    def test_none_means_unitless(self):
        q = Quantity(5.0, unit=UNITLESS)
        func = lambda: None
        func.__name__ = "test_func"
        _check_unit(func, q, None)  # should not raise


# =========================================================================
# _assign_unit
# =========================================================================

class TestAssignUnitHelper:
    def test_none_passthrough(self):
        result = _assign_unit(lambda: None, 5.0, None)
        assert result == 5.0

    def test_bool_passthrough(self):
        result = _assign_unit(lambda: None, True, bool)
        assert result is True

    def test_assigns_unit(self):
        func = lambda: None
        result = _assign_unit(func, 5.0, _metre)
        assert isinstance(result, Quantity)
        assert result.unit == _metre


# =========================================================================
# Missing / missing sentinel
# =========================================================================

class TestMissingSentinel:
    def test_missing_is_instance(self):
        assert isinstance(missing, Missing)

    def test_missing_triggers_partial(self):
        # When f=missing, assign_units returns a partial decorator
        decorator = assign_units(x=_metre)
        assert callable(decorator)


# =========================================================================
# CallableAssignUnit
# =========================================================================

class TestCallableAssignUnit:
    def test_is_callable_subclass(self):
        from collections.abc import Callable
        assert issubclass(CallableAssignUnit, Callable)


# =========================================================================
# Integration tests migrated from _base_test.py
# =========================================================================

import saiunit as u
from saiunit._unit_common import *
from saiunit._unit_shortcuts import kHz, ms, mV, nS
from saiunit._base_getters import assert_quantity, fail_for_dimension_mismatch


class TestCheckDimsIntegration:
    """Integration tests for check_dims using public unit objects."""

    def test_correct_units_pass(self):
        @u.check_dims(v=volt.dim)
        def a_function(v, x):
            pass

        a_function(3 * mV, 5 * second)
        a_function(5 * volt, "something")
        a_function([1, 2, 3] * volt, None)
        a_function([1 * volt, 2 * volt, 3 * volt], None)
        a_function("a string", None)
        a_function(None, None)

    def test_incorrect_units_raise(self):
        @u.check_dims(v=volt.dim)
        def a_function(v, x):
            pass

        with pytest.raises(DimensionMismatchError):
            a_function(5 * second, None)
        with pytest.raises(DimensionMismatchError):
            a_function(5, None)
        with pytest.raises(DimensionMismatchError):
            a_function(object(), None)
        with pytest.raises(TypeError):
            a_function([1, 2 * volt, 3], None)

    def test_result_dim_check(self):
        @u.check_dims(result=second.dim)
        def b_function(return_second):
            if return_second:
                return 5 * second
            else:
                return 3 * volt

        b_function(True)
        with pytest.raises(DimensionMismatchError):
            b_function(False)

    def test_bool_and_unitless_args(self):
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

    def test_multiple_results_tuple(self):
        @u.check_dims(result=(second.dim, volt.dim))
        def d_function(true_result):
            if true_result:
                return 5 * second, 3 * volt
            else:
                return 3 * volt, 5 * second

        d_function(True)
        with pytest.raises(u.DimensionMismatchError):
            d_function(False)

    def test_multiple_results_dict(self):
        @u.check_dims(result={'u': second.dim, 'v': (volt.dim, metre.dim)})
        def d_function2(true_result):
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


class TestCheckUnitsIntegration:
    """Integration tests for check_units using public unit objects."""

    def test_correct_units_pass(self):
        @u.check_units(v=volt)
        def a_function(v, x):
            pass

        with pytest.raises(u.UnitMismatchError):
            a_function(3 * mV, 5 * second)
        a_function(3 * volt, 5 * second)
        a_function(5 * volt, "something")
        a_function([1, 2, 3] * volt, None)
        a_function([1 * volt, 2 * volt, 3 * volt], None)
        a_function("a string", None)
        a_function(None, None)

    def test_incorrect_units_raise(self):
        @u.check_units(v=volt)
        def a_function(v, x):
            pass

        with pytest.raises(u.UnitMismatchError):
            a_function(5 * second, None)
        with pytest.raises(u.UnitMismatchError):
            a_function(5, None)
        with pytest.raises(u.UnitMismatchError):
            a_function(object(), None)
        with pytest.raises(TypeError):
            a_function([1, 2 * volt, 3], None)

    def test_result_unit_check(self):
        @check_units(result=second)
        def b_function(return_second):
            if return_second:
                return 5 * second
            else:
                return 3 * volt

        b_function(True)
        with pytest.raises(u.UnitMismatchError):
            b_function(False)

    def test_bool_and_unitless_args(self):
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

    def test_multiple_results_tuple(self):
        @check_units(result=(second, volt))
        def d_function(true_result):
            if true_result:
                return 5 * second, 3 * volt
            else:
                return 3 * volt, 5 * second

        d_function(True)
        with pytest.raises(u.UnitMismatchError):
            d_function(False)

    def test_multiple_results_dict(self):
        @check_units(result={'u': second, 'v': (volt, metre)})
        def d_function2(true_result):
            if true_result == 0:
                return {'u': 5 * second, 'v': (3 * volt, 2 * metre)}
            elif true_result == 1:
                return 3 * volt, 5 * second
            else:
                return {'u': 5 * second, 'v': (3 * volt, 2 * volt)}

        d_function2(0)
        with pytest.raises(TypeError):
            d_function2(1)
        with pytest.raises(u.UnitMismatchError):
            d_function2(2)


class TestAssignUnitsIntegration:
    """Integration tests for assign_units using public unit objects."""

    def test_correct_units_converted(self):
        @u.assign_units(v=volt)
        def a_function(v, x):
            return v

        assert a_function(3 * mV, 5 * second) == (3 * mV).to_decimal(volt)
        assert a_function(3 * volt, 5 * second) == (3 * volt).to_decimal(volt)
        assert a_function(5 * volt, "something") == (5 * volt).to_decimal(volt)
        assert_quantity(a_function([1, 2, 3] * volt, None), ([1, 2, 3] * volt).to_decimal(volt))

    def test_incorrect_units_raise(self):
        @u.assign_units(v=volt)
        def a_function(v, x):
            return v

        with pytest.raises(u.UnitMismatchError):
            a_function(5 * second, None)
        with pytest.raises(TypeError):
            a_function(5, None)
        with pytest.raises(TypeError):
            a_function(object(), None)

    def test_result_unit_assigned(self):
        @u.assign_units(result=second)
        def b_function():
            return 5

        assert b_function() == 5 * second

    def test_bool_and_unitless_args(self):
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

    def test_multiple_results_tuple(self):
        @u.assign_units(result=(second, volt))
        def d_function():
            return 5, 3

        assert d_function()[0] == 5 * second
        assert d_function()[1] == 3 * volt

    def test_multiple_results_dict(self):
        @u.assign_units(result={'u': second, 'v': (volt, metre)})
        def d_function2(true_result):
            if true_result == 0:
                return {'u': 5, 'v': (3, 2)}
            elif true_result == 1:
                return 3, 5
            else:
                return 3, 5

        d_function2(0)
        with pytest.raises(TypeError):
            d_function2(1)


# --- Docstring example tests ---


import saiunit as su


class TestDocstringExampleCheckDims:
    """Tests that exercise the examples shown in check_dims's docstring."""

    def test_docstring_example_check_dims_basic(self):
        """Basic dimension checking: I in amps, R in ohms, result in volts."""
        @su.check_dims(I=su.amp.dim, R=su.ohm.dim, result=su.volt.dim)
        def get_voltage(I, R):
            return I * R

        result = get_voltage(1 * su.amp, 1 * su.ohm)
        assert isinstance(result, Quantity)
        assert result.dim == su.volt.dim
        assert jnp.allclose(result.mantissa, 1.0)

    def test_docstring_example_check_dims_dimensionless_and_bool(self):
        """Use ``1`` for dimensionless and ``bool`` for boolean arguments."""
        @su.check_dims(value=1, flag=bool)
        def scale(value, flag):
            return value * 2 if flag else value

        assert scale(5, True) == 10
        assert scale(5, False) == 5
        # Non-boolean value (int) should raise TypeError
        with pytest.raises(TypeError, match="boolean"):
            scale(5, flag=42)

    def test_docstring_example_check_dims_callable_result(self):
        """Callable ``result`` derives return dimension from input dimensions."""
        @su.check_dims(result=lambda d: d ** 2)
        def square(x):
            return x ** 2

        result = square(3.0 * su.metre)
        assert isinstance(result, Quantity)
        assert result.dim == su.metre.dim ** 2
        assert jnp.allclose(result.mantissa, 9.0)

    def test_docstring_example_check_dims_string_reference(self):
        """String reference forces two arguments to share the same dimension."""
        @su.check_dims(a=None, b='a')
        def add(a, b):
            return a + b

        result = add(1.0 * su.metre, 2.0 * su.metre)
        assert isinstance(result, Quantity)
        assert jnp.allclose(result.mantissa, 3.0)

        # Mismatched dimensions must raise
        with pytest.raises(DimensionMismatchError):
            add(1.0 * su.metre, 2.0 * su.second)


class TestDocstringExampleCheckUnits:
    """Tests that exercise the examples shown in check_units's docstring."""

    def test_docstring_example_check_units_basic(self):
        """Basic unit checking: I in amp, R in ohm, result in volt."""
        @su.check_units(I=su.amp, R=su.ohm, result=su.volt)
        def get_voltage(I, R):
            return I * R

        result = get_voltage(1 * su.amp, 1 * su.ohm)
        assert isinstance(result, Quantity)
        assert result.unit == su.volt
        assert jnp.allclose(result.mantissa, 1.0)

    def test_docstring_example_check_units_bool(self):
        """Use ``bool`` to require a boolean argument."""
        @su.check_units(flag=bool)
        def toggle(flag):
            return not flag

        assert toggle(True) is False
        assert toggle(False) is True
        with pytest.raises(TypeError):
            toggle(5)

    def test_docstring_example_check_units_string_reference(self):
        """String reference forces two arguments to share the same unit."""
        @su.check_units(a=None, b='a')
        def add(a, b):
            return a + b

        result = add(1.0 * su.volt, 2.0 * su.volt)
        assert isinstance(result, Quantity)
        assert result.unit == su.volt
        assert jnp.allclose(result.mantissa, 3.0)

        # Different units must raise
        with pytest.raises(UnitMismatchError):
            add(1.0 * su.volt, 2.0 * su.second)


class TestDocstringExampleAssignUnits:
    """Tests that exercise the examples shown in assign_units's docstring."""

    def test_docstring_example_assign_units_strip(self):
        """Strip units from inputs so the function body works with plain numbers."""
        @su.assign_units(v=su.volt)
        def double_voltage(v):
            return v * 2

        result = double_voltage(3.0 * su.volt)
        assert jnp.allclose(result, 6.0)

    def test_docstring_example_assign_units_result(self):
        """Assign a unit to the return value."""
        @su.assign_units(result=su.second)
        def make_time():
            return 5

        result = make_time()
        assert isinstance(result, Quantity)
        assert result.unit == su.second
        assert jnp.allclose(result.mantissa, 5.0)

    def test_docstring_example_assign_units_without_result_units(self):
        """``without_result_units`` returns the raw numeric result."""
        @su.assign_units(x=su.volt, result=su.volt)
        def identity(x):
            return x

        raw = identity.without_result_units(3.0 * su.volt)
        assert jnp.allclose(raw, 3.0)
        assert not isinstance(raw, Quantity)

    def test_docstring_example_assign_units_tuple_result(self):
        """Strip inputs and assign units to a tuple of return values."""
        @su.assign_units(result=(su.second, su.volt))
        def make_pair():
            return 5, 3

        t, v = make_pair()
        assert isinstance(t, Quantity)
        assert isinstance(v, Quantity)
        assert t.unit == su.second
        assert v.unit == su.volt
        assert jnp.allclose(t.mantissa, 5.0)
        assert jnp.allclose(v.mantissa, 3.0)
