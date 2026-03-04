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

"""
Comprehensive display tests for Unit and Quantity.

Covers: str/repr/format, unit composition simplification, permutation
invariance, ambiguous-unit avoidance, SI prefixes, exponent handling,
dimensionless display, Quantity formatting with scalars/arrays/2-D arrays,
display_in_unit, repr_in_unit, edge cases, deeply nested compositions,
cross-prefix compositions, round-trip consistency, and unexpected usages.
"""

import re

import jax.numpy as jnp
import numpy as np
import pytest

import saiunit as u
from saiunit._base_unit import Unit, UNITLESS, parse_unit
from saiunit._base_quantity import Quantity
from saiunit._base_getters import display_in_unit


# ===================================================================
# Section 1: Unit — basic str / repr
# ===================================================================


class TestUnitBasicDisplay:
    """str() and repr() for simple named units."""

    # -- base SI units --

    @pytest.mark.parametrize("unit, expected", [
        (u.metre, "m"),
        (u.meter, "m"),
        (u.kilogram, "kg"),
        (u.second, "s"),
        (u.amp, "A"),
        (u.ampere, "A"),
        (u.kelvin, "K"),
        (u.mole, "mol"),
        (u.candle, "cd"),
    ])
    def test_str_base_units(self, unit, expected):
        assert str(unit) == expected

    @pytest.mark.parametrize("unit, expected", [
        (u.metre, 'Unit("m")'),
        (u.kilogram, 'Unit("kg")'),
        (u.second, 'Unit("s")'),
        (u.amp, 'Unit("A")'),
        (u.kelvin, 'Unit("K")'),
    ])
    def test_repr_base_units(self, unit, expected):
        assert repr(unit) == expected

    # -- named derived SI units --

    @pytest.mark.parametrize("unit, expected", [
        (u.hertz, "Hz"),
        (u.newton, "N"),
        (u.pascal, "Pa"),
        (u.joule, "J"),
        (u.watt, "W"),
        (u.coulomb, "C"),
        (u.volt, "V"),
        (u.farad, "F"),
        (u.ohm, "ohm"),
        (u.siemens, "S"),
        (u.weber, "Wb"),
        (u.tesla, "T"),
        (u.henry, "H"),
        (u.lumen, "lm"),
        (u.lux, "lx"),
        (u.becquerel, "Bq"),
        (u.gray, "Gy"),
        (u.sievert, "Sv"),
        (u.katal, "kat"),
    ])
    def test_str_derived_units(self, unit, expected):
        assert str(unit) == expected

    # -- dimensionless named units --

    def test_str_radian(self):
        assert str(u.radian) == "rad"

    def test_str_steradian(self):
        assert str(u.steradian) == "sr"

    # -- UNITLESS --

    def test_str_unitless(self):
        assert str(UNITLESS) == "1"

    def test_repr_unitless(self):
        assert repr(UNITLESS) == 'Unit("1")'


# ===================================================================
# Section 2: Unit — SI-prefixed display
# ===================================================================


class TestUnitPrefixedDisplay:
    """SI-prefixed units display the correct symbol."""

    @pytest.mark.parametrize("unit, expected", [
        (u.mV, "mV"),
        (u.mA, "mA"),
        (u.uA, "uA"),
        (u.nA, "nA"),
        (u.pA, "pA"),
        (u.pF, "pF"),
        (u.uF, "uF"),
        (u.nF, "nF"),
        (u.nS, "nS"),
        (u.uS, "uS"),
        (u.mS, "mS"),
        (u.ms, "ms"),
        (u.us, "us"),
        (u.Hz, "Hz"),
        (u.kHz, "kHz"),
        (u.MHz, "MHz"),
        (u.cm, "cm"),
        (u.mm, "mm"),
        (u.um, "um"),
        (u.cm2, "cm^2"),
        (u.mm2, "mm^2"),
        (u.um2, "um^2"),
        (u.cm3, "cm^3"),
        (u.mm3, "mm^3"),
        (u.um3, "um^3"),
        (u.mM, "mM"),
        (u.uM, "uM"),
        (u.nM, "nM"),
    ])
    def test_str_shortcut_units(self, unit, expected):
        assert str(unit) == expected

    @pytest.mark.parametrize("unit, expected", [
        (u.kvolt, "kV"),
        (u.mvolt, "mV"),
        (u.uvolt, "uV"),
        (u.nvolt, "nV"),
        (u.pvolt, "pV"),
        (u.Mvolt, "MV"),
        (u.Gvolt, "GV"),
    ])
    def test_str_voltage_prefixes(self, unit, expected):
        assert str(unit) == expected

    @pytest.mark.parametrize("unit, expected", [
        (u.kmetre, "km"),
        (u.mmetre, "mm"),
        (u.umetre, "um"),
        (u.nmetre, "nm"),
        (u.pmetre, "pm"),
        (u.fmetre, "fm"),
    ])
    def test_str_length_prefixes(self, unit, expected):
        assert str(unit) == expected

    @pytest.mark.parametrize("unit, expected", [
        (u.ksecond, "ks"),
        (u.msecond, "ms"),
        (u.usecond, "us"),
        (u.nsecond, "ns"),
        (u.psecond, "ps"),
        (u.fsecond, "fs"),
    ])
    def test_str_time_prefixes(self, unit, expected):
        assert str(unit) == expected


# ===================================================================
# Section 3: Unit — composition simplification (the core new feature)
# ===================================================================


class TestUnitCompositionSimplification:
    """Composed units that match a known derived unit display the derived name."""

    # -- multiplication --

    @pytest.mark.parametrize("a, b, expected", [
        # Ohm's law: V = I * R
        (u.mA, u.ohm, "mV"),
        (u.amp, u.ohm, "V"),
        (u.uA, u.ohm, "uV"),
        (u.nA, u.ohm, "nV"),
        # Power: W = V * A
        (u.volt, u.amp, "W"),
        (u.mV, u.amp, "mW"),
        (u.volt, u.mA, "mW"),
        (u.kvolt, u.mA, "W"),
        # Force: N = kg * m / s^2  (via pascal * m^2)
        (u.pascal, u.metre2, "N"),
        # Energy: J = N * m
        (u.newton, u.metre, "J"),
        # Energy: J = Pa * m^3
        (u.pascal, u.metre3, "J"),
        # Charge: C = A * s
        (u.amp, u.second, "C"),
        # Magnetic flux: Wb = T * m^2
        (u.tesla, u.metre2, "Wb"),
        # Charge: C = F * V
        (u.farad, u.volt, "C"),
        # Current: A = S * V
        (u.siemens, u.volt, "A"),
    ])
    def test_mul_simplifies(self, a, b, expected):
        assert str(a * b) == expected

    # -- commutativity of simplification --

    @pytest.mark.parametrize("a, b, expected", [
        (u.mA, u.ohm, "mV"),
        (u.volt, u.amp, "W"),
        (u.newton, u.metre, "J"),
        (u.amp, u.second, "C"),
        (u.farad, u.volt, "C"),
    ])
    def test_mul_commutative(self, a, b, expected):
        assert str(a * b) == expected
        assert str(b * a) == expected

    # -- division --

    @pytest.mark.parametrize("a, b, expected", [
        # Resistance: ohm = V / A
        (u.volt, u.amp, "ohm"),
        # Voltage: V = W / A
        (u.watt, u.amp, "V"),
        # Current: A = W / V
        (u.watt, u.volt, "A"),
        # Voltage: V = J / C
        (u.joule, u.coulomb, "V"),
        # Pressure: Pa = N / m^2
        (u.newton, u.metre2, "Pa"),
        # Inductance: H = Wb / A
        (u.weber, u.amp, "H"),
        # Resistance: ohm = H / s
        (u.henry, u.second, "ohm"),
        # Time: s = Wb / V
        (u.weber, u.volt, "s"),
    ])
    def test_div_simplifies(self, a, b, expected):
        assert str(a / b) == expected

    # -- cross-prefix compositions --

    @pytest.mark.parametrize("a, b, expected", [
        (u.kvolt, u.mA, "W"),
        (u.uamp, u.kohm, "mV"),
        (u.namp, u.Mohm, "mV"),
    ])
    def test_cross_prefix_simplifies(self, a, b, expected):
        assert str(a * b) == expected


# ===================================================================
# Section 4: Unit — compositions that should NOT simplify
# ===================================================================


class TestUnitCompositionNoSimplify:
    """Composed units that have no matching derived unit stay as compound."""

    @pytest.mark.parametrize("expr, expected", [
        (u.metre / u.second, "m / s"),
        (u.amp / u.metre, "A / m"),
        (u.kilogram / u.metre3, "kg / m^3"),
        (u.newton / u.amp, "N / A"),
    ])
    def test_no_standard_unit_stays_compound(self, expr, expected):
        assert str(expr) == expected


# ===================================================================
# Section 5: Unit — ambiguous-unit avoidance
# ===================================================================


class TestUnitAmbiguousAvoidance:
    """Ambiguous dimension keys (Hz/Bq, Gy/Sv) are not auto-substituted."""

    def test_joule_per_kg_not_sievert(self):
        """J / kg has same dim as Gy and Sv — must NOT simplify."""
        assert str(u.joule / u.kilogram) == "J / kg"

    def test_m2_per_s2_not_gray(self):
        """m^2 / s^2 has same dim as Gy and Sv — must NOT simplify."""
        result = u.metre ** 2 / u.second ** 2
        assert str(result) == "m^2 / s^2"

    def test_watt_per_kg_not_sievert(self):
        """W / kg has dim m^2 * s^-3 — not ambiguous, no derived unit."""
        result = u.watt / u.kilogram
        assert str(result) == "W / kg"

    def test_pow_compound_not_gray(self):
        """(m/s)^2 should NOT become Gy."""
        result = (u.metre / u.second) ** 2
        assert str(result) == "m^2 / s^2"

    def test_direct_units_still_display(self):
        """Direct ambiguous units display their own names."""
        assert str(u.hertz) == "Hz"
        assert str(u.becquerel) == "Bq"
        assert str(u.gray) == "Gy"
        assert str(u.sievert) == "Sv"


# ===================================================================
# Section 6: Unit — permutation invariance
# ===================================================================


class TestUnitPermutationInvariance:
    """Operand order must not affect display for multi-operand compositions."""

    def test_three_way_mul_invariant(self):
        a = u.meter * u.second * u.amp
        b = u.amp * u.second * u.meter
        c = u.second * u.amp * u.meter
        assert str(a) == str(b) == str(c)

    def test_compound_mul_invariant(self):
        a = (u.nA / u.cm2) * u.mS
        b = u.mS * (u.nA / u.cm2)
        assert str(a) == str(b)

    def test_four_way_invariant(self):
        a = u.kilogram * u.metre * u.amp * u.kelvin
        b = u.kelvin * u.amp * u.metre * u.kilogram
        c = u.amp * u.kilogram * u.kelvin * u.metre
        assert str(a) == str(b) == str(c)

    def test_no_intermediate_collapse(self):
        """amp * second * meter should not collapse amp*second into coulomb."""
        result = u.amp * u.second * u.meter
        s = str(result)
        # Must be the alphabetically-sorted base parts, not "C * m"
        assert s == "A * m * s"


# ===================================================================
# Section 7: Unit — exponent handling
# ===================================================================


class TestUnitExponentDisplay:
    """Powers, fractional exponents, stacking avoidance."""

    def test_squared(self):
        assert str(u.metre ** 2) == "m^2"

    def test_cubed(self):
        assert str(u.metre ** 3) == "m^3"

    def test_negative_exponent(self):
        assert str(u.second ** -1) == "1 / s"

    def test_negative_exponent_two(self):
        assert str(u.second ** -2) == "1 / s^2"

    def test_fractional_exponent(self):
        assert str(u.ohm ** 0.5) == "ohm^0.5"

    def test_fractional_exponent_metre(self):
        assert str(u.metre ** 0.5) == "m^0.5"

    def test_fractional_negative_exponent(self):
        assert str(u.second ** -0.5) == "1 / s^0.5"

    def test_compound_fractional_exponent(self):
        result = (u.metre * u.second) ** 0.5
        assert str(result) == "m^0.5 * s^0.5"

    def test_no_exponent_stacking(self):
        """(m^2)^3 must display m^6, not m^2^3."""
        assert str((u.metre ** 2) ** 3) == "m^6"

    def test_no_exponent_stacking_compound(self):
        """(m^2 / s)^3 must display m^6 / s^3."""
        assert str((u.metre ** 2 / u.second) ** 3) == "m^6 / s^3"

    def test_compound_pow_preserves_parts(self):
        """(m * s / A)^2 should keep all parts."""
        assert str((u.metre * u.second / u.amp) ** 2) == "m^2 * s^2 / A^2"

    def test_pow_zero_is_dimensionless(self):
        assert str(u.metre ** 0) == "1"

    def test_pow_one_identity(self):
        assert str(u.metre ** 1) == "m"

    def test_registered_squared_unit(self):
        """Pre-registered m^2 displays correctly."""
        assert str(u.metre2) == "m^2"

    def test_registered_cubed_unit(self):
        assert str(u.metre3) == "m^3"

    def test_cm_squared(self):
        assert str(u.cm ** 2) == "cm^2"

    def test_cm_cubed(self):
        assert str(u.cm ** 3) == "cm^3"

    def test_squared_consistent_with_mul(self):
        assert str(u.metre ** 2) == str(u.metre * u.metre)

    def test_cubed_consistent_with_mul(self):
        assert str(u.metre ** 3) == str(u.metre * u.metre * u.metre)


# ===================================================================
# Section 8: Unit — reverse / inverse
# ===================================================================


class TestUnitReverseDisplay:
    """reverse() and reciprocal display."""

    def test_second_reverse_is_hertz(self):
        assert str(u.second.reverse()) == "Hz"

    def test_millisecond_reverse_is_khertz(self):
        assert str(u.msecond.reverse()) == "kHz"

    def test_metre_reverse(self):
        assert str(u.metre.reverse()) == "1 / m"

    def test_repr_reverse(self):
        assert repr(u.second.reverse()) == 'Unit("Hz")'


# ===================================================================
# Section 9: Unit — dimensionless display
# ===================================================================


class TestUnitDimensionlessDisplay:
    """Dimensionless results from composition."""

    def test_same_unit_division(self):
        assert str(u.metre / u.metre) == "1"

    def test_same_derived_division(self):
        assert str(u.volt / u.volt) == "1"

    def test_different_scale_division(self):
        """km / m is dimensionless with scale."""
        result = u.kmetre / u.metre
        s = str(result)
        assert "10" in s and "3" in s  # "10.0^3"

    def test_pow_zero(self):
        assert str(u.volt ** 0) == "1"

    def test_repr_dimensionless_from_division(self):
        assert repr(u.metre / u.metre) == 'Unit("1")'


# ===================================================================
# Section 10: Unit — multi-step / deeply nested compositions
# ===================================================================


class TestUnitDeepCompositions:
    """Complex chains of arithmetic that should simplify."""

    def test_kg_m_per_s2_is_newton(self):
        """kg * m / s^2 → N."""
        result = u.kilogram * u.metre / u.second ** 2
        assert str(result) == "N"

    def test_kg_m2_per_s2_is_joule(self):
        """kg * m^2 / s^2 → J."""
        result = u.kilogram * u.metre ** 2 / u.second ** 2
        assert str(result) == "J"

    def test_kg_m2_per_s3_is_watt(self):
        """kg * m^2 / s^3 → W."""
        result = u.kilogram * u.metre ** 2 / u.second ** 3
        assert str(result) == "W"

    def test_kg_m2_per_s3_per_A_is_volt(self):
        """kg * m^2 / (s^3 * A) → V."""
        result = u.kilogram * u.metre ** 2 / u.second ** 3 / u.amp
        assert str(result) == "V"

    def test_volt_amp_second_is_joule(self):
        """V * A * s → J."""
        result = u.volt * u.amp * u.second
        assert str(result) == "J"

    def test_newton_metre_per_coulomb_is_volt(self):
        """(N * m) / (A * s) → V."""
        result = (u.newton * u.metre) / (u.amp * u.second)
        assert str(result) == "V"

    def test_pascal_m2_is_newton(self):
        """Pa * m^2 → N."""
        result = u.pascal * u.metre2
        assert str(result) == "N"

    def test_pascal_m3_is_joule(self):
        """Pa * m^3 → J."""
        result = u.pascal * u.metre3
        assert str(result) == "J"

    def test_tesla_m2_is_weber(self):
        """T * m^2 → Wb."""
        result = u.tesla * u.metre2
        assert str(result) == "Wb"

    def test_compound_div_chain(self):
        """(V / A) is ohm, then (ohm * A / s) → V / s."""
        va = u.volt / u.amp  # ohm
        result = va * u.amp / u.second
        assert str(result) == "V / s"

    def test_nested_simplification(self):
        """(kg * m / s^2) * m → J."""
        force = u.kilogram * u.metre / u.second ** 2  # N
        energy = force * u.metre
        assert str(energy) == "J"


# ===================================================================
# Section 11: Unit — compound display formatting
# ===================================================================


class TestUnitCompoundFormatting:
    """Formatting of compound units (numerator/denominator, parentheses)."""

    def test_single_denominator(self):
        assert str(u.metre / u.second) == "m / s"

    def test_multiple_denominators(self):
        result = u.metre / (u.kilogram * u.second ** 2)
        assert str(result) == "m / (kg * s^2)"

    def test_numerator_product(self):
        """Multiple positive-exponent terms."""
        result = u.kilogram * u.metre * u.amp * u.kelvin
        s = str(result)
        # All four should be in the numerator, sorted alphabetically
        assert " * " in s
        parts = s.split(" * ")
        assert len(parts) == 4

    def test_mixed_numerator_denominator(self):
        result = u.mS * u.nA / u.cm2
        assert str(result) == "mS * nA / cm^2"

    def test_repr_compound(self):
        result = u.newton / u.metre2
        assert repr(result) == 'Unit("Pa")'

    def test_repr_non_simplifiable_compound(self):
        result = u.metre / u.second
        assert repr(result) == 'Unit("m / s")'


# ===================================================================
# Section 12: Unit — repr / str consistency
# ===================================================================


class TestUnitReprStrConsistency:
    """repr(unit) must wrap str(unit)."""

    @pytest.mark.parametrize("unit", [
        u.mV, u.volt, u.ohm, u.newton, u.hertz,
        u.metre / u.second,
        u.joule / u.kilogram,
        u.nA / u.cm2,
        u.mS * u.nA / u.cm2,
    ])
    def test_repr_wraps_str(self, unit):
        assert repr(unit) == f'Unit("{str(unit)}")'


# ===================================================================
# Section 13: Unit — parse round-trip
# ===================================================================


class TestUnitParseRoundTrip:
    """Unit(str(unit)).has_same_dim(unit) for composed units."""

    @pytest.mark.parametrize("unit", [
        u.mV,
        u.volt,
        u.ohm,
        u.hertz,
        u.newton,
        u.joule,
        u.watt,
        u.pascal,
        u.coulomb,
        u.farad,
        u.siemens,
        u.weber,
        u.tesla,
        u.henry,
    ])
    def test_roundtrip_named_units(self, unit):
        parsed = Unit(str(unit))
        assert parsed.has_same_dim(unit)

    @pytest.mark.parametrize("expr, expected_str", [
        (u.metre / u.second, "m / s"),
        (u.joule / u.kilogram, "J / kg"),
        (u.nA / u.cm2, "nA / cm^2"),
    ])
    def test_roundtrip_compound_units(self, expr, expected_str):
        assert str(expr) == expected_str
        parsed = Unit(expected_str)
        assert parsed.has_same_dim(expr)

    def test_roundtrip_simplified(self):
        """Composed unit that simplifies can round-trip via its simplified name."""
        composed = u.mA * u.ohm  # → mV
        parsed = Unit(str(composed))
        assert parsed.has_same_dim(composed)
        assert str(parsed) == "mV"


# ===================================================================
# Section 14: Quantity — basic str / repr
# ===================================================================


class TestQuantityBasicDisplay:
    """str() and repr() for Quantity with various mantissa types."""

    def test_str_scalar(self):
        q = 3.14 * u.mV
        assert str(q) == "3.14 mV"

    def test_repr_scalar(self):
        q = 3.14 * u.mV
        assert repr(q) == 'Quantity(3.14, "mV")'

    def test_str_integer_mantissa(self):
        q = 5 * u.mV
        assert str(q) == "5 mV"

    def test_repr_integer_mantissa(self):
        q = 5 * u.mV
        assert repr(q) == 'Quantity(5, "mV")'

    def test_str_negative(self):
        q = -5 * u.mV
        assert str(q) == "-5 mV"

    def test_str_zero(self):
        q = 0.0 * u.mV
        assert str(q) == "0. mV"

    def test_str_very_small(self):
        q = 1e-15 * u.mV
        s = str(q)
        assert "mV" in s
        assert "e-15" in s or "e-015" in s

    def test_str_very_large(self):
        q = 1e15 * u.mV
        s = str(q)
        assert "mV" in s
        assert "e+15" in s or "e+015" in s

    def test_str_boolean_mantissa(self):
        q = True * u.mV
        assert "mV" in str(q)

    def test_str_jnp_scalar(self):
        q = jnp.float32(5.0) * u.mV
        s = str(q)
        assert "mV" in s
        assert "5." in s


# ===================================================================
# Section 15: Quantity — array display
# ===================================================================


class TestQuantityArrayDisplay:
    """Display for 1-D, 2-D, and large arrays."""

    def test_str_1d_array(self):
        q = jnp.array([1.0, 2.0, 3.0]) * u.mV
        assert str(q) == "[1. 2. 3.] mV"

    def test_repr_1d_array(self):
        q = jnp.array([1.0, 2.0, 3.0]) * u.mV
        assert repr(q) == 'Quantity([1. 2. 3.], "mV")'

    def test_str_2d_array(self):
        q = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * u.mV
        s = str(q)
        assert "mV" in s
        assert "1." in s and "4." in s

    def test_repr_2d_array_indentation(self):
        """Multi-line repr has continuation lines indented under 'Quantity('."""
        q = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * u.mV
        r = repr(q)
        assert r.startswith("Quantity(")
        assert '"mV"' in r
        lines = r.split("\n")
        if len(lines) > 1:
            indent = " " * len("Quantity(")
            for line in lines[1:]:
                assert line.startswith(indent) or line.strip().endswith(")")

    def test_str_large_array_summarised(self):
        """Arrays > 100 elements are summarised with '...'."""
        q = jnp.arange(200.0) * u.mV
        s = str(q)
        assert "..." in s
        assert "mV" in s

    def test_repr_large_array_balanced_brackets(self):
        q = jnp.arange(200.0) * u.mV
        r = repr(q)
        assert r.count("[") == r.count("]")

    def test_str_single_element_array(self):
        q = jnp.array([5.0]) * u.mV
        assert "5." in str(q)
        assert "mV" in str(q)

    def test_str_0d_array(self):
        q = jnp.array(5.0) * u.mV
        s = str(q)
        assert "5." in s
        assert "mV" in s


# ===================================================================
# Section 16: Quantity — __format__
# ===================================================================


class TestQuantityFormat:
    """Format-string display for Quantity."""

    # -- scalar --

    def test_format_f(self):
        q = 3.14 * u.mV
        assert f"{q:.2f}" == "3.14 mV"

    def test_format_e(self):
        q = 3.14 * u.mV
        assert f"{q:.3e}" == "3.140e+00 mV"

    def test_format_g(self):
        q = 3.14 * u.mV
        assert f"{q:.2g}" == "3.1 mV"

    def test_format_width_precision(self):
        q = 3.14 * u.mV
        result = f"{q:10.2f}"
        assert "3.14" in result
        assert "mV" in result

    def test_format_sign(self):
        q = 3.14 * u.mV
        result = f"{q:+.2f}"
        assert result.startswith("+3.14")
        assert "mV" in result

    def test_format_empty_returns_str(self):
        q = 3.14 * u.mV
        assert format(q, "") == str(q)

    # -- array --

    def test_format_array_f(self):
        q = jnp.array([1.234, 2.345, 3.456]) * u.mV
        result = f"{q:.2f}"
        assert "1.23" in result
        assert "mV" in result

    def test_format_array_e(self):
        q = jnp.array([1.234, 2.345]) * u.mV
        result = f"{q:.2e}"
        assert "mV" in result

    # -- percent format blocked for physical units --

    def test_percent_format_raises(self):
        q = 0.5 * u.mV
        with pytest.raises(ValueError, match="%"):
            f"{q:%}"

    def test_percent_format_ok_for_unitless(self):
        q = Quantity(0.5)
        result = f"{q:%}"
        assert "%" in result


# ===================================================================
# Section 17: Quantity — unitless display
# ===================================================================


class TestQuantityUnitlessDisplay:
    """Unitless Quantity should not show a unit symbol."""

    def test_str_unitless_scalar(self):
        q = Quantity(3.14)
        assert str(q) == "3.14"

    def test_repr_unitless_scalar(self):
        q = Quantity(3.14)
        assert repr(q) == "Quantity(3.14)"

    def test_str_division_dimensionless(self):
        q = 3.0 * (u.metre / u.metre)
        assert str(q) == "3."
        assert "m" not in str(q)

    def test_repr_division_dimensionless(self):
        q = 3.0 * (u.metre / u.metre)
        r = repr(q)
        assert "Quantity(3.)" == r


# ===================================================================
# Section 18: Quantity — radian / steradian display
# ===================================================================


class TestQuantityDimensionlessNamedDisplay:
    """Radian and steradian are dimensionless but carry names."""

    def test_str_radian(self):
        q = 3.14 * u.radian
        assert str(q) == "3.14 rad"

    def test_repr_radian(self):
        q = 3.14 * u.radian
        assert repr(q) == 'Quantity(3.14, "rad")'

    def test_str_steradian(self):
        q = 3.14 * u.steradian
        assert str(q) == "3.14 sr"

    def test_repr_steradian(self):
        q = 3.14 * u.steradian
        assert repr(q) == 'Quantity(3.14, "sr")'

    def test_format_radian(self):
        q = 3.14 * u.radian
        assert f"{q:.1f}" == "3.1 rad"

    def test_should_display_radian(self):
        assert u.radian.should_display_unit is True

    def test_should_display_steradian(self):
        assert u.steradian.should_display_unit is True

    def test_should_not_display_unitless(self):
        assert UNITLESS.should_display_unit is False


# ===================================================================
# Section 19: Quantity — display with composed/simplified units
# ===================================================================


class TestQuantityComposedUnitDisplay:
    """Quantity display when unit comes from composition."""

    def test_str_quantity_from_mul(self):
        """5.0 * (N * m) displays as joule."""
        q = 5.0 * (u.newton * u.metre)
        assert str(q) == "5. J"

    def test_repr_quantity_from_mul(self):
        q = 5.0 * (u.newton * u.metre)
        assert repr(q) == 'Quantity(5., "J")'

    def test_quantity_mul_simplifies_unit(self):
        """Quantity arithmetic: mV * mA = uW."""
        q1 = 3.0 * u.mV
        q2 = 2.0 * u.mA
        result = q1 * q2
        assert str(result) == "6. uW"
        assert repr(result) == 'Quantity(6., "uW")'

    def test_quantity_div_simplifies_unit(self):
        """Quantity arithmetic: mV / mA = ohm."""
        q1 = 3.0 * u.mV
        q2 = 2.0 * u.mA
        result = q1 / q2
        assert "ohm" in str(result)

    def test_quantity_compound_unit_not_simplified(self):
        """Quantity with compound unit that has no derived equivalent."""
        q = 5.0 * (u.metre / u.second)
        assert str(q) == "5. m / s"


# ===================================================================
# Section 20: Quantity — repr_in_unit and display_in_unit
# ===================================================================


class TestQuantityReprInUnit:
    """repr_in_unit and display_in_unit functionality."""

    def test_repr_in_unit_default(self):
        q = 25.0 * u.mV
        assert q.repr_in_unit() == "25. mV"

    def test_repr_in_unit_precision(self):
        q = 123.456789 * u.mV
        result = q.repr_in_unit(2)
        assert "123.46" in result
        assert "mV" in result

    def test_repr_in_unit_high_precision(self):
        q = 123.456789 * u.mV
        result = q.repr_in_unit(5)
        assert "mV" in result

    def test_display_in_unit_conversion(self):
        assert display_in_unit(3 * u.volt, u.mV) == "3000. mV"

    def test_display_in_unit_precision(self):
        result = display_in_unit(123123 * u.msecond, u.second, 2)
        assert "s" in result
        assert "123.12" in result

    def test_display_in_unit_compound(self):
        """ohm * amp simplifies to V, so display uses V."""
        result = display_in_unit(10 * u.mV, u.ohm * u.amp)
        assert "V" in result

    def test_display_in_unit_no_target_uses_own(self):
        q = 5.0 * u.mV
        assert display_in_unit(q) == "5. mV"


# ===================================================================
# Section 21: Quantity — unit conversion display
# ===================================================================


class TestQuantityConversionDisplay:
    """Display after .to() / .in_unit() conversions."""

    def test_to_changes_display(self):
        q = (1000.0 * u.mV).to(u.volt)
        assert str(q) == "1. V"
        assert repr(q) == 'Quantity(1., "V")'

    def test_to_millivolt(self):
        q = (1.0 * u.volt).to(u.mV)
        assert str(q) == "1000. mV"

    def test_in_unit_changes_display(self):
        q = (5000.0 * u.metre).in_unit(u.kmetre)
        assert "km" in str(q)


# ===================================================================
# Section 22: Quantity — string-constructed units
# ===================================================================


class TestQuantityStringUnit:
    """Quantity created with a string unit argument."""

    def test_string_unit_mV(self):
        q = Quantity(1.0, "mV")
        assert str(q) == "1. mV"
        assert repr(q) == 'Quantity(1., "mV")'

    def test_string_unit_compound(self):
        q = Quantity(1.0, "J / kg")
        assert str(q) == "1. J / kg"

    def test_string_unit_invalid_raises(self):
        with pytest.raises((ValueError, KeyError)):
            Quantity(1.0, "invalid_unit_xyz")

    def test_string_unit_array(self):
        q = Quantity(jnp.array([1.0, 2.0]), "mV")
        assert "mV" in str(q)


# ===================================================================
# Section 23: Unit — should_display_unit
# ===================================================================


class TestShouldDisplayUnit:
    """should_display_unit property for various unit types."""

    def test_physical_unit(self):
        assert u.volt.should_display_unit is True
        assert u.metre.should_display_unit is True
        assert u.mV.should_display_unit is True

    def test_unitless(self):
        assert UNITLESS.should_display_unit is False

    def test_dimensionless_from_division(self):
        result = u.metre / u.metre
        assert result.should_display_unit is False

    def test_radian_is_displayed(self):
        assert u.radian.should_display_unit is True

    def test_steradian_is_displayed(self):
        assert u.steradian.should_display_unit is True

    def test_composed_physical_unit(self):
        result = u.metre / u.second
        assert result.should_display_unit is True


# ===================================================================
# Section 24: Unit — alias preference
# ===================================================================


class TestUnitAliasPreference:
    """Preferred aliases (hertz over becquerel, etc.)."""

    def test_hertz_preferred_for_inverse_second(self):
        assert str(u.second.reverse()) == "Hz"

    def test_khertz_preferred_for_inverse_ms(self):
        assert str(u.msecond.reverse()) == "kHz"

    def test_meter_metre_same_display(self):
        """Spelling variants display identically."""
        assert str(u.meter) == str(u.metre)

    def test_kilogram_kilogramme_same_display(self):
        assert str(u.kilogram) == str(u.kilogramme)

    def test_amp_ampere_same_display(self):
        assert str(u.amp) == str(u.ampere)


# ===================================================================
# Section 25: Edge cases and unexpected usages
# ===================================================================


class TestEdgeCases:
    """Unusual, degenerate, and boundary-condition display scenarios."""

    # -- self-composition --

    def test_unit_mul_itself(self):
        result = u.metre * u.metre
        assert str(result) == "m^2"

    def test_unit_mul_itself_three(self):
        result = u.metre * u.metre * u.metre
        assert str(result) == "m^3"

    def test_unit_div_itself(self):
        result = u.volt / u.volt
        assert str(result) == "1"

    # -- identity operations --

    def test_mul_by_unitless_quantity(self):
        """Multiplying unit by scalar 1 gives a Quantity."""
        q = 1 * u.mV
        assert "mV" in str(q)

    # -- very high / low SI prefixes --

    def test_yocto_display(self):
        assert str(u.yvolt) == "yV"

    def test_yotta_display(self):
        assert str(u.Yvolt) == "YV"

    def test_zepto_display(self):
        assert str(u.zvolt) == "zV"

    def test_zetta_display(self):
        assert str(u.Zvolt) == "ZV"

    # -- gram vs kilogram edge case --

    def test_gram_display(self):
        assert str(u.gram) == "g"

    def test_kilogram_display(self):
        assert str(u.kilogram) == "kg"

    # -- liter/litre --

    def test_liter_display(self):
        assert str(u.liter) == "l"

    def test_litre_display(self):
        assert str(u.litre) == "l"

    # -- molar --

    def test_molar_display(self):
        assert str(u.molar) == "M"

    # -- compound with pre-registered squared/cubed units --

    def test_registered_cm2_in_compound(self):
        result = u.nA / u.cm2
        assert str(result) == "nA / cm^2"

    # -- chain that passes through simplification and then back --

    def test_simplify_then_compose_further(self):
        """mA * ohm → mV (simplified), then mV / mA → ohm again."""
        step1 = u.mA * u.ohm  # simplified to mV at display
        step2 = step1 / u.mA
        assert str(step2) == "ohm"

    # -- double reverse --

    def test_double_reverse_is_identity(self):
        """unit.reverse().reverse() should display the same as unit."""
        original = u.second
        double = original.reverse().reverse()
        assert str(double) == str(original)

    # -- Quantity with unit from pow --

    def test_quantity_with_pow_unit(self):
        q = 5.0 * u.metre ** 2
        assert str(q) == "5. m^2"

    # -- Quantity negative array --

    def test_quantity_negative_array(self):
        q = jnp.array([-1.0, -2.0, -3.0]) * u.mV
        s = str(q)
        assert "mV" in s
        assert "-1." in s

    # -- Quantity with zero-dimensional unit --

    def test_quantity_unitless_not_show_unit(self):
        q = Quantity(42.0)
        assert str(q) == "42."
        assert "Unit" not in repr(q)

    # -- NaN and Inf --

    def test_quantity_nan(self):
        q = jnp.nan * u.mV
        s = str(q)
        assert "nan" in s.lower()
        assert "mV" in s

    def test_quantity_inf(self):
        q = jnp.inf * u.mV
        s = str(q)
        assert "inf" in s.lower()
        assert "mV" in s

    def test_quantity_negative_inf(self):
        q = -jnp.inf * u.mV
        s = str(q)
        assert "inf" in s.lower()
        assert "mV" in s


# ===================================================================
# Section 26: Complex real-world compositions
# ===================================================================


class TestRealWorldCompositions:
    """Physically meaningful complex compositions."""

    def test_ohms_law_voltage(self):
        """V = I * R."""
        I = 10.0 * u.mA
        R = 100.0 * u.ohm
        V = I * R
        assert "V" in str(V)

    def test_power_dissipation(self):
        """P = V * I."""
        V = 5.0 * u.volt
        I = 2.0 * u.amp
        P = V * I
        assert str(P) == "10. W"

    def test_energy_computation(self):
        """E = P * t."""
        P = 1.0 * u.watt
        t = 1.0 * u.second
        E = P * t
        assert str(E) == "1. J"

    def test_capacitor_charge(self):
        """Q = C * V."""
        C = 10.0 * u.uF
        V = 5.0 * u.volt
        Q = C * V
        assert "C" in str(Q)  # coulomb

    def test_conductance_current(self):
        """I = G * V."""
        G = 1.0 * u.siemens
        V = 5.0 * u.volt
        I = G * V
        assert str(I) == "5. A"

    def test_magnetic_flux(self):
        """Phi = B * A."""
        B = 1.0 * u.tesla
        A = 2.0 * u.metre2
        Phi = B * A
        assert str(Phi) == "2. Wb"

    def test_neuroscience_current_density(self):
        """I/A in nA/cm^2 — common in neuroscience."""
        I = 5.0 * u.nA
        A = u.cm2
        q = I / (1.0 * A)
        assert "nA / cm^2" in str(q)

    def test_neuroscience_conductance_density(self):
        """g in mS/cm^2 — membrane conductance density."""
        g = 36.0 * (u.mS / u.cm2)
        assert "mS / cm^2" in str(g)


# ===================================================================
# Section 27: Exhaustive simplification coverage
# ===================================================================


class TestExhaustiveSimplification:
    """Verify that ALL standard SI derived units can be reached by composition."""

    def test_hertz_from_composition(self):
        """1/s → Hz (via reverse, since ambiguous in composition)."""
        assert str(u.second.reverse()) == "Hz"

    def test_newton_from_composition(self):
        assert str(u.kilogram * u.metre / u.second ** 2) == "N"

    def test_pascal_from_composition(self):
        assert str(u.newton / u.metre2) == "Pa"

    def test_joule_from_composition(self):
        assert str(u.newton * u.metre) == "J"

    def test_watt_from_composition(self):
        assert str(u.joule / u.second) == "W"

    def test_coulomb_from_composition(self):
        assert str(u.amp * u.second) == "C"

    def test_volt_from_composition(self):
        assert str(u.watt / u.amp) == "V"

    def test_farad_from_composition(self):
        assert str(u.coulomb / u.volt) == "F"

    def test_ohm_from_composition(self):
        assert str(u.volt / u.amp) == "ohm"

    def test_siemens_from_composition(self):
        assert str(u.amp / u.volt) == "S"

    def test_weber_from_composition(self):
        assert str(u.volt * u.second) == "Wb"

    def test_tesla_from_composition(self):
        assert str(u.weber / u.metre2) == "T"

    def test_henry_from_composition(self):
        assert str(u.weber / u.amp) == "H"

    def test_katal_from_composition(self):
        assert str(u.mole / u.second) == "kat"


# ===================================================================
# Section 28: Multi-step compositions with mixed prefixes
# ===================================================================


class TestMixedPrefixCompositions:
    """Compositions mixing different SI prefixes."""

    def test_kV_mA_is_W(self):
        assert str(u.kvolt * u.mA) == "W"

    def test_uA_kohm_is_mV(self):
        assert str(u.uamp * u.kohm) == "mV"

    def test_nA_Mohm_is_mV(self):
        assert str(u.namp * u.Mohm) == "mV"

    def test_mV_mA_is_uW(self):
        assert str(u.mV * u.mA) == "uW"

    def test_kN_mm_is_J(self):
        """kN * mm → J (10^3 * 10^-3 = 10^0)."""
        assert str(u.knewton * u.mmetre) == "J"

    def test_MV_uA_is_W(self):
        """MV * uA → W (10^6 * 10^-6 = 10^0)."""
        assert str(u.Mvolt * u.uamp) == "W"


# ===================================================================
# Section 29: Quantity — format edge cases
# ===================================================================


class TestQuantityFormatEdgeCases:
    """Edge cases for Quantity __format__."""

    def test_format_zero(self):
        q = 0.0 * u.mV
        assert f"{q:.2f}" == "0.00 mV"

    def test_format_negative(self):
        q = -3.14 * u.mV
        result = f"{q:.2f}"
        assert "-3.14" in result
        assert "mV" in result

    def test_format_nan(self):
        q = jnp.nan * u.mV
        result = f"{q:.2f}"
        assert "nan" in result.lower()

    def test_format_inf(self):
        q = jnp.inf * u.mV
        result = f"{q:.2f}"
        assert "inf" in result.lower()

    def test_format_bad_spec_raises_for_scalar(self):
        """Invalid format spec on scalar Quantity raises ValueError."""
        q = 3.14 * u.mV
        with pytest.raises(ValueError):
            format(q, "xyz")

    def test_format_bad_spec_fallback_for_array(self):
        """Non-parseable format spec on array falls back to str()."""
        q = jnp.array([1.0, 2.0]) * u.mV
        result = format(q, "xyz")
        assert result == str(q)

    def test_format_unitless_no_unit(self):
        q = Quantity(3.14)
        result = f"{q:.2f}"
        assert result == "3.14"
        assert "Unit" not in result


# ===================================================================
# Section 30: Deeply nested / pathological compositions
# ===================================================================


class TestPathologicalCompositions:
    """Stress-test: deeply nested and unusual compositions."""

    def test_chain_simplify_to_base(self):
        """V * A * s / (m^2 * kg) should simplify or display sensibly."""
        result = u.volt * u.amp * u.second / (u.metre ** 2 * u.kilogram)
        # This is J / (m^2 * kg) = (m^2 * kg * s^-2) / (m^2 * kg) = s^-2
        # s^-2 has no named derived unit, so compound display
        s = str(result)
        assert s  # at minimum does not crash

    def test_repeated_mul_div(self):
        """Multiplying and dividing by the same unit cancels out."""
        result = u.volt * u.amp / u.amp
        # volt * amp = watt (display), then / amp
        # Actually the display_parts: volt*amp gives display_parts,
        # then /amp merges, leaving just volt parts
        assert str(result) == "V"

    def test_many_cancellations(self):
        """Multiple cancellations in a chain."""
        result = (u.kilogram * u.metre * u.second *
                  u.amp / u.kilogram / u.metre / u.second)
        assert str(result) == "A"

    def test_six_way_composition(self):
        """Six units multiplied together — should not crash."""
        result = (u.kilogram * u.metre * u.metre *
                  u.amp / u.second / u.second / u.second)
        # kg * m^2 * A / s^3  — no standard unit
        s = str(result)
        assert s  # no crash
        assert "kg" in s or "A" in s or "m" in s

    def test_composed_then_pow(self):
        """Compose, then raise to a power."""
        ms = u.metre / u.second
        result = ms ** 3
        assert str(result) == "m^3 / s^3"

    def test_pow_then_compose(self):
        """Raise to power, then compose with another unit."""
        m2 = u.metre ** 2
        result = m2 * u.kilogram / u.second ** 2
        # m^2 * kg / s^2 → J
        assert str(result) == "J"

    def test_reverse_composed(self):
        """Reverse of a compound unit."""
        ms = u.metre / u.second
        result = ms.reverse()
        assert str(result) == "s / m"

    def test_double_reverse_compound(self):
        ms = u.metre / u.second
        result = ms.reverse().reverse()
        assert str(result) == "m / s"

    def test_triple_product_simplification(self):
        """Three units whose product is a derived unit."""
        # kg * m * (1/s^2) = N
        result = u.kilogram * u.metre * (u.second ** -2)
        assert str(result) == "N"

    def test_mixed_compound_and_derived(self):
        """Compose a derived unit with a compound unit."""
        # Pa * m = (kg / (m * s^2)) * m = kg / s^2
        # No standard unit for kg / s^2
        result = u.pascal * u.metre
        s = str(result)
        assert s  # does not crash

    def test_long_chain_all_cancels(self):
        """Everything cancels to dimensionless."""
        result = (u.volt * u.amp * u.second /
                  u.joule)
        # V * A * s / J = W * s / J = J / J = 1
        assert str(result) == "1"
