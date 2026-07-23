# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Regression tests for Quantity API edge cases and bug fixes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import saiunit as u
from saiunit import Quantity, UnitMismatchError
from saiunit._base_unit import Unit


class TestScatterFactorUnits:
    """at[].multiply / at[].divide must not re-label untouched elements."""

    def test_multiply_unit_bearing_factor_raises(self):
        q = jnp.array([1.0, 2.0, 3.0]) * u.mV
        with pytest.raises(TypeError, match="dimensionless"):
            q.at[0].multiply(2.0 * u.ms)

    def test_divide_unit_bearing_factor_raises(self):
        q = jnp.array([1.0, 2.0, 3.0]) * u.mV
        with pytest.raises(TypeError, match="dimensionless"):
            q.at[0].divide(2.0 * u.ms)

    def test_multiply_dimensionless_keeps_unit(self):
        q = jnp.array([1.0, 2.0, 3.0]) * u.mV
        r = q.at[0].multiply(2.0)
        assert r.unit == u.mV
        np.testing.assert_allclose(np.asarray(r.mantissa), [2.0, 2.0, 3.0])


class TestSearchsorted:
    def test_plain_number_against_unit_bearing_raises(self):
        q = jnp.array([1.0, 2.0, 3.0]) * u.mV
        with pytest.raises(UnitMismatchError):
            q.searchsorted(2.5)

    def test_quantity_value_is_scaled(self):
        q = jnp.array([1.0, 2.0, 3.0]) * u.volt
        idx = q.searchsorted(Quantity(1500.0, unit=u.mV))
        assert int(idx) == 1

    def test_unitless_plain_number_ok(self):
        q = Quantity(jnp.array([1.0, 2.0, 3.0]))
        assert int(q.searchsorted(2.5)) == 2


class TestResize:
    def test_resize_grows(self):
        q = Quantity(jnp.arange(4.0), unit=u.mV).resize((8,))
        assert q.shape == (8,)
        assert q.unit == u.mV

    def test_resize_reshapes_numpy(self):
        q = Quantity(np.arange(4.0), unit=u.mV).resize((2, 2))
        assert q.shape == (2, 2)


class TestExpandAs:
    def test_expand_as_quantity(self):
        q = Quantity(jnp.ones(3), unit=u.mV)
        r = q.expand_as(Quantity(jnp.zeros((2, 3)), unit=u.mV))
        assert r.shape == (2, 3)
        assert r.unit == u.mV

    def test_expand_as_raw_array(self):
        q = Quantity(jnp.ones(3), unit=u.mV)
        r = q.expand_as(jnp.zeros((2, 3)))
        assert r.shape == (2, 3)

    def test_expand_as_shape_tuple_still_works(self):
        q = Quantity(jnp.ones(3), unit=u.mV)
        r = q.expand_as((2, 3))
        assert r.shape == (2, 3)


class TestInplaceShift:
    def test_ilshift_unitless_array(self):
        q = Quantity(jnp.array([4, 8]))
        q <<= 1
        np.testing.assert_array_equal(np.asarray(q.mantissa), [8, 16])

    def test_irshift_unitless_array(self):
        q = Quantity(jnp.array([4, 8]))
        q >>= 1
        np.testing.assert_array_equal(np.asarray(q.mantissa), [2, 4])

    def test_ilshift_scalar_mantissa(self):
        q = Quantity(4)
        q <<= 1
        assert int(q.mantissa) == 8


class TestFlat:
    def test_flat_jax_backend(self):
        q = Quantity(jnp.arange(3.0), unit=u.mV)
        elements = list(q.flat)
        assert len(elements) == 3
        assert all(e.unit == u.mV for e in elements)

    def test_flat_numpy_backend(self):
        q = Quantity(np.arange(3.0), unit=u.mV)
        assert len(list(q.flat)) == 3


class TestBuiltinRound:
    def test_round_numpy_backed(self):
        q = Quantity(np.array([1.234, 2.567]), unit=u.mV)
        r = round(q, 1)
        np.testing.assert_allclose(np.asarray(r.mantissa), [1.2, 2.6])
        assert r.unit == u.mV

    def test_round_jax_backed(self):
        q = Quantity(jnp.array([1.234, 2.567]), unit=u.mV)
        r = round(q, 1)
        np.testing.assert_allclose(np.asarray(r.mantissa), [1.2, 2.6], rtol=1e-6)

    def test_round_scalar_mantissa(self):
        r = round(Quantity(1.234, unit=u.mV), 1)
        assert r.mantissa == pytest.approx(1.2)


class TestScalarMantissaInplace:
    def test_iadd_scalar_mantissa(self):
        q = Quantity(1.0, unit=u.mV)
        q += 2.0 * u.mV
        assert q.mantissa == pytest.approx(3.0)
        assert q.unit == u.mV

    def test_iadd_zero_compat(self):
        q = Quantity(0)
        q += 3 * u.ms
        assert q.unit == u.ms
        assert float(q.mantissa) == pytest.approx(3.0)

    def test_isub_scalar_mantissa(self):
        q = Quantity(5.0, unit=u.mV)
        q -= 2.0 * u.mV
        assert q.mantissa == pytest.approx(3.0)


class TestScalarMantissaMethods:
    def test_item(self):
        r = Quantity(3.0, unit=u.mV).item()
        assert isinstance(r, Quantity)
        assert r.mantissa == pytest.approx(3.0)

    def test_prod(self):
        r = Quantity(3.0, unit=u.mV).prod()
        assert r.unit == u.mV
        assert float(r.mantissa) == pytest.approx(3.0)

    def test_nanprod(self):
        r = Quantity(3.0, unit=u.mV).nanprod()
        assert r.unit == u.mV


class TestClipBounds:
    def test_plain_bound_against_unit_bearing_raises_unit_mismatch(self):
        q = jnp.array([1.0, 5.0]) * u.mV
        with pytest.raises(UnitMismatchError):
            q.clip(min=2.0)

    def test_quantity_bounds_scale(self):
        q = jnp.array([1.0, 5.0]) * u.mV
        r = q.clip(max=Quantity(0.002, unit=u.volt))
        np.testing.assert_allclose(np.asarray(r.mantissa), [1.0, 2.0])


class TestListConstructor:
    def test_mixed_unitless_sublist_raises_type_error(self):
        with pytest.raises(TypeError, match="same units"):
            Quantity([[1 * u.mV, 2 * u.mV], [3, 4]])

    def test_mixed_order_independent(self):
        with pytest.raises(TypeError, match="same units"):
            Quantity([[1, 2], jnp.array([3.0, 4.0]) * u.mV])

    def test_all_unitless_nested_ok(self):
        q = Quantity([[1, 2], [3, 4]])
        assert q.shape == (2, 2)

    def test_all_united_nested_ok(self):
        q = Quantity([[1 * u.mV, 2 * u.mV], [3 * u.mV, 4 * u.mV]])
        assert q.unit == u.mV
        assert q.shape == (2, 2)


class TestZeroComparison:
    def test_gt_zero(self):
        assert bool((3.0 * u.mV) > 0)

    def test_eq_zero(self):
        assert not bool((3.0 * u.mV) == 0)
        assert bool((0.0 * u.mV) == 0)

    def test_le_zero(self):
        assert bool((-1.0 * u.mV) <= 0)

    def test_nonzero_plain_still_raises(self):
        with pytest.raises(UnitMismatchError):
            (3.0 * u.mV) > 1.0

    def test_different_dim_nonzero_still_raises(self):
        with pytest.raises(UnitMismatchError):
            (3.0 * u.mV) > (1.0 * u.ms)


class TestComparisonContract:
    def test_eq_none_is_false(self):
        assert ((3.0 * u.mV) == None) is False  # noqa: E711

    def test_ne_none_is_true(self):
        assert ((3.0 * u.mV) != None) is True  # noqa: E711

    def test_eq_string_is_false(self):
        assert ((3.0 * u.mV) == "foo") is False

    def test_ordering_with_none_raises_type_error(self):
        with pytest.raises(TypeError):
            (3.0 * u.mV) < None

    def test_eq_same_dim_still_elementwise(self):
        q = jnp.array([1.0, 2.0]) * u.mV
        r = q == jnp.array([1.0, 3.0]) * u.mV
        np.testing.assert_array_equal(np.asarray(r), [True, False])


class TestTakeNumpyModes:
    def test_mode_fill_with_fill_value(self):
        q = Quantity(np.array([1.0, 2.0]), unit=u.mV)
        r = q.take(np.array([0, 5]), mode='fill', fill_value=Quantity(0.0, unit=u.mV))
        np.testing.assert_allclose(np.asarray(r.mantissa), [1.0, 0.0])
        assert r.unit == u.mV

    def test_mode_fill_default_nan(self):
        q = Quantity(np.array([1.0, 2.0]), unit=u.mV)
        r = q.take(np.array([0, 5]), mode='fill')
        assert np.isnan(np.asarray(r.mantissa)[1])

    def test_mode_clip(self):
        q = Quantity(np.array([1.0, 2.0]), unit=u.mV)
        r = q.take(np.array([0, 5]), mode='clip')
        np.testing.assert_allclose(np.asarray(r.mantissa), [1.0, 2.0])

    def test_mode_fill_negative_valid_index_wraps(self):
        q = Quantity(np.array([1.0, 2.0, 3.0]), unit=u.mV)
        r = q.take(np.array([-1, 0]), mode='fill', fill_value=Quantity(9.0, unit=u.mV))
        np.testing.assert_allclose(np.asarray(r.mantissa), [3.0, 1.0])

    def test_mode_fill_with_axis(self):
        q = Quantity(np.arange(6.0).reshape(2, 3), unit=u.mV)
        r = q.take(np.array([0, 5]), axis=1, mode='fill', fill_value=Quantity(-1.0, unit=u.mV))
        np.testing.assert_allclose(np.asarray(r.mantissa), [[0.0, -1.0], [3.0, -1.0]])

    def test_default_mode_raises_on_oob(self):
        q = Quantity(np.array([1.0, 2.0]), unit=u.mV)
        with pytest.raises(IndexError):
            q.take(np.array([5]))


class TestConstructorDtype:
    def test_scalar_dtype(self):
        assert Quantity(3, dtype=jnp.float32).dtype == jnp.float32

    def test_quantity_input_dtype(self):
        q = Quantity(3.0, unit=u.mV)
        assert Quantity(q, dtype=jnp.float16).dtype == jnp.float16

    def test_unit_mantissa_dtype(self):
        assert Quantity(u.mV, dtype=jnp.float16).dtype == jnp.float16

    def test_numpy_scalar_dtype(self):
        q = Quantity(np.float32(3.0), dtype=np.float64)
        assert q.dtype == np.float64


class TestUnitlessPowArrayExponent:
    def test_unitless_base_array_exponent(self):
        r = Quantity(2.0) ** jnp.array([1.0, 2.0])
        np.testing.assert_allclose(np.asarray(r), [2.0, 4.0])

    def test_unit_bearing_base_array_exponent_still_raises(self):
        with pytest.raises(TypeError):
            (2.0 * u.mV) ** jnp.array([1.0, 2.0])

    def test_unit_bearing_scalar_exponent(self):
        r = (2.0 * u.mV) ** 2
        assert r.unit == u.mV ** 2


class TestConstructorGarbage:
    def test_str_mantissa_rejected(self):
        with pytest.raises(TypeError, match="str"):
            Quantity("foo")

    def test_bytes_mantissa_rejected(self):
        with pytest.raises(TypeError, match="bytes"):
            Quantity(b"foo")


class TestInvert:
    def test_invert_unit_bearing_raises(self):
        with pytest.raises(NotImplementedError):
            ~Quantity(jnp.array([1, 2]), unit=u.mV)

    def test_invert_unitless_ok(self):
        r = ~Quantity(jnp.array([1, 2]))
        np.testing.assert_array_equal(np.asarray(r.mantissa), [-2, -3])


class TestSortOrder:
    def test_order_param_rejected(self):
        q = Quantity(jnp.array([3.0, 1.0]), unit=u.mV)
        with pytest.raises(NotImplementedError):
            q.sort(order='field')

    def test_sort_still_works(self):
        q = Quantity(jnp.array([3.0, 1.0, 2.0]), unit=u.mV)
        q.sort()
        np.testing.assert_allclose(np.asarray(q.mantissa), [1.0, 2.0, 3.0])


class TestBoolTruthiness:
    """bool() must follow the mantissa's NumPy/JAX value semantics, not len()."""

    def test_single_element_zero_is_false(self):
        assert bool(Quantity(jnp.array([0.0]), unit=u.mV)) is False

    def test_single_element_nonzero_is_true(self):
        assert bool(Quantity(jnp.array([1.0]), unit=u.mV)) is True

    def test_zero_dim(self):
        assert bool(Quantity(2.5, unit=u.mV)) is True
        assert bool(Quantity(0.0, unit=u.mV)) is False

    def test_multi_element_is_ambiguous(self):
        with pytest.raises((ValueError, TypeError)):
            bool(Quantity(jnp.array([1.0, 2.0]), unit=u.mV))

    def test_len_zero_dim_raises_cleanly(self):
        with pytest.raises(TypeError, match="0-d"):
            len(Quantity(2.5, unit=u.mV))


class TestIsubBareUnit:
    """Q1: `q -= unit` must be rejected the same way `+`, `-`, `+=` are."""

    def test_isub_bare_unit_raises(self):
        q = Quantity(3.0, unit=u.mV)
        with pytest.raises(TypeError, match="bare Unit"):
            q -= u.mV


class TestUfuncDispatchExtras:
    """Q2: ufuncs missing from `_UFUNC_DISPATCH` should route through saiunit.math."""

    def test_np_maximum(self):
        q1 = Quantity(jnp.array([1.0, 5.0]), unit=u.mV)
        q2 = Quantity(jnp.array([3.0, 2.0]), unit=u.mV)
        r = np.maximum(q1, q2)
        np.testing.assert_allclose(np.asarray(r.mantissa), [3.0, 5.0])
        assert r.unit == u.mV

    def test_np_floor(self):
        q = Quantity(jnp.array([1.7, 2.3]), unit=u.mV)
        r = np.floor(q)
        np.testing.assert_allclose(np.asarray(r.mantissa), [1.0, 2.0])
        assert r.unit == u.mV

    def test_np_tanh_dimensionless(self):
        q = Quantity(jnp.array([0.0, 1.0]))
        r = np.tanh(q)
        np.testing.assert_allclose(np.asarray(r), np.tanh([0.0, 1.0]))

    def test_np_hypot(self):
        # saiunit.math.hypot only accepts dimensionless operands (unless
        # `unit_to_scale` is given), so dispatch it through dimensionless
        # Quantities rather than unit-bearing ones.
        q1 = Quantity(jnp.array([3.0]))
        q2 = Quantity(jnp.array([4.0]))
        r = np.hypot(q1, q2)
        np.testing.assert_allclose(np.asarray(r), [5.0])


class TestBinaryOpNotImplementedFallback:
    """Q3: an operand `_binary_operation` can't handle must defer via
    `NotImplemented` so Python tries the operand's own reflected dunder."""

    class _RMulSentinel:
        def __rmul__(self, other):
            return "handled-by-other"

    def test_mul_defers_to_other_rmul(self):
        q = Quantity(3.0, unit=u.mV)
        assert (q * self._RMulSentinel()) == "handled-by-other"

    def test_unsupported_operand_raises_type_error_not_internal_crash(self):
        q = Quantity(3.0, unit=u.mV)
        with pytest.raises(TypeError):
            q * object()


class TestAssignmentZeroCompat:
    """Q4/Q6: plain `0` is compatible with any unit for assignment-style ops,
    and a mismatched non-zero value raises UnitMismatchError uniformly."""

    def test_setitem_zero(self):
        q = Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
        q[0] = 0
        assert float(q.mantissa[0]) == pytest.approx(0.0)

    def test_setitem_mismatch_raises_unit_mismatch(self):
        q = Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
        with pytest.raises(UnitMismatchError):
            q[0] = 5.0

    def test_scatter_add_zero(self):
        q = Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
        r = q.scatter_add(0, 0)
        assert float(r.mantissa[0]) == pytest.approx(1.0)

    def test_scatter_add_mismatch_raises_unit_mismatch(self):
        q = Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
        with pytest.raises(UnitMismatchError):
            q.scatter_add(0, 5.0)

    def test_fill_zero(self):
        q = Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
        q.fill(0)
        np.testing.assert_allclose(np.asarray(q.mantissa), [0.0, 0.0])

    def test_at_set_zero(self):
        q = Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
        r = q.at[0].set(0)
        assert float(r.mantissa[0]) == pytest.approx(0.0)


class TestEqualityMismatchDoesNotRaise:
    """Q5: `==`/`!=` return elementwise False/True on a dimension mismatch
    instead of raising; ordering comparisons keep raising."""

    def test_eq_different_dim_is_false(self):
        r = (3.0 * u.mV) == (3.0 * u.second)
        assert bool(r) is False

    def test_ne_different_dim_is_true(self):
        r = (3.0 * u.mV) != (3.0 * u.second)
        assert bool(r) is True

    def test_eq_different_dim_array_shape(self):
        q1 = Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
        q2 = Quantity(jnp.array([1.0, 2.0]), unit=u.second)
        r = q1 == q2
        np.testing.assert_array_equal(np.asarray(r), [False, False])

    def test_ordering_different_dim_still_raises(self):
        with pytest.raises(UnitMismatchError):
            (3.0 * u.mV) < (3.0 * u.second)

    def test_membership_works_with_mismatched_dim(self):
        # `in` relies on `__eq__` not raising for incompatible elements.
        assert (3.0 * u.second) not in [1.0 * u.mV, 2.0 * u.mV]


class TestPowScaledDimensionlessExponent:
    """Q7: a dimensionless-but-scaled Quantity exponent (e.g. percent) must
    be accepted, not just a strictly unitless one."""

    def test_percent_exponent(self):
        percent = Unit(scale=-2)
        r = 2.0 ** Quantity(50.0, unit=percent)
        assert float(r) == pytest.approx(2.0 ** 0.5)

    def test_rpow_percent_exponent(self):
        percent = Unit(scale=-2)
        base = Quantity(50.0, unit=percent)  # dimensionless base too
        r = 2.0 ** base
        assert float(r) == pytest.approx(2.0 ** 0.5)

    def test_non_dimensionless_exponent_still_raises(self):
        with pytest.raises(ValueError):
            2.0 ** Quantity(2.0, unit=u.mV)


class TestDivmodUpfrontValidation:
    """Q8: divmod must raise on a dimension mismatch before doing any
    partial computation."""

    def test_divmod_mismatch_raises(self):
        with pytest.raises(UnitMismatchError):
            divmod(3.0 * u.mV, 2.0 * u.second)

    def test_divmod_same_dim_ok(self):
        q, r = divmod(7.0 * u.mV, 2.0 * u.mV)
        assert float(q) == pytest.approx(3.0)
        assert float(r.mantissa) == pytest.approx(1.0)


class TestTracerMismatchHint:
    """Q9: a unit mismatch involving a traced unitless zero should hint at
    the eager-vs-jit divergence instead of leaving a bare mismatch error."""

    def test_hint_present_under_jit(self):
        @jax.jit
        def f(x):
            return (x - x) + 3 * u.mV

        with pytest.raises(UnitMismatchError, match="JAX tracer"):
            f(jnp.array(1.0))


class TestDtypeBoolMantissa:
    """Q10: `Quantity.dtype` for a Python bool mantissa must be a real
    dtype object, not the Python `bool` type."""

    def test_bool_mantissa_dtype_is_np_dtype(self):
        q = Quantity(True)
        assert q.dtype == np.dtype(bool)
        assert isinstance(q.dtype, np.dtype)


class TestResizeTracerGuard:
    """Q11: resize() must reject a traced mantissa like other mutators do."""

    def test_resize_under_jit_raises(self):
        @jax.jit
        def f(x):
            q = Quantity(x, unit=u.mV)
            q.resize((4,))
            return q.mantissa

        with pytest.raises(RuntimeError, match="tracer"):
            f(jnp.arange(4.0))


class TestTakeEmptyAxis:
    """Q12: take(mode='clip'/'fill') on an empty NumPy-backed axis must not
    raise IndexError."""

    def test_take_fill_empty_array(self):
        q = Quantity(np.zeros((0,)), unit=u.mV)
        r = q.take(np.array([0, 1]), mode='fill', fill_value=Quantity(9.0, unit=u.mV))
        np.testing.assert_allclose(np.asarray(r.mantissa), [9.0, 9.0])
        assert r.unit == u.mV

    def test_take_clip_empty_array_default_fill(self):
        q = Quantity(np.zeros((0,)), unit=u.mV)
        r = q.take(np.array([0]), mode='clip')
        assert np.isnan(np.asarray(r.mantissa)[0])
