# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import unittest

import numpy as np
import pytest

try:
    import scipy.constants as sc

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import saiunit as u
from saiunit._unit_common import *

constants_list = [
    # Mass
    'metric_ton', 'grain', 'pound', 'slinch', 'slug', 'ounce', 'stone', 'long_ton', 'short_ton', 'troy_ounce',
    'troy_pound', 'carat', 'atomic_mass',
    # Angle
    'degree', 'arcmin', 'arcsec',
    # Time
    'minute', 'hour', 'day', 'week', 'month', 'year', 'julian_year',
    # Length
    'inch', 'foot', 'yard', 'mile', 'mil', 'point', 'pica', 'survey_foot', 'survey_mile', 'nautical_mile', 'fermi',
    'angstrom', 'micron', 'au', 'light_year',
    # Pressure
    'atm', 'bar', 'mmHg', 'psi',
    # Area
    'hectare', 'acre',
    # Volume
    'gallon', 'gallon_imp', 'fluid_ounce', 'fluid_ounce_imp', 'bbl',
    # Speed
    'kmh', 'mph', 'knot', 'mach',
    # Energy
    'eV', 'calorie', 'calorie_IT', 'erg', 'Btu', 'Btu_IT', 'ton_TNT',
    # Power
    'hp',
    # Force
    'dyn', 'lbf', 'kgf', 'IMF',
    # --- added coverage: dual-defined Quantity constants in constants + _unit_constants ---
    'arcminute', 'arcsecond', 'astronomical_unit', 'atmosphere', 'calorie_th',
    'fluid_ounce_US', 'gallon_US', 'horsepower', 'kilogram_force', 'lb',
]


class TestConstant(unittest.TestCase):

    def test_constants(self):
        import saiunit.constants as constants

        # Check that the expected names exist and have the correct dimensions
        assert constants.avogadro.dim == (1 / mole).dim
        assert constants.boltzmann.dim == (joule / kelvin).dim
        assert constants.electric.dim == (farad / meter).dim
        assert constants.electron_mass.dim == kilogram.dim
        assert constants.elementary_charge.dim == coulomb.dim
        assert constants.faraday.dim == (coulomb / mole).dim
        assert constants.gas.dim == (joule / mole / kelvin).dim
        assert constants.magnetic.dim == (newton / amp2).dim
        assert constants.molar_mass.dim == (kilogram / mole).dim
        assert constants.zero_celsius.dim == kelvin.dim

        # Check the consistency between a few constants
        assert u.math.allclose(
            constants.gas.mantissa,
            (constants.avogadro * constants.boltzmann).mantissa,
        )
        assert u.math.allclose(
            constants.faraday.mantissa,
            (constants.avogadro * constants.elementary_charge).mantissa,
        )

    def test_quantity_constants_and_unit_constants(self):
        import saiunit.constants as quantity_constants
        import saiunit._unit_constants as unit_constants
        for c in constants_list:
            print(c)
            q_c = getattr(quantity_constants, c)
            u_c = getattr(unit_constants, c)
            assert u.math.isclose(
                q_c.to_decimal(q_c.unit), (1. * u_c).to_decimal(q_c.unit)
            ), f"Mismatch between {c} in quantity_constants and unit_constants"


# --- Docstring example tests ---


def test_docstring_example_constants_module():
    """Verify representative constants exist and carry the expected units."""
    import saiunit.constants as constants

    # -- Fundamental constants carry correct dimensions --
    assert constants.avogadro.dim == (1 / mole).dim
    assert constants.boltzmann.dim == (joule / kelvin).dim
    assert constants.electron_mass.dim == kilogram.dim
    assert constants.elementary_charge.dim == coulomb.dim

    # -- Mass constants have kilogram dimension --
    assert constants.pound.dim == kilogram.dim
    assert constants.atomic_mass.dim == kilogram.dim
    assert constants.carat.dim == kilogram.dim

    # -- Time constants have second dimension --
    assert constants.minute.dim == second.dim
    assert constants.hour.dim == second.dim
    assert constants.day.dim == second.dim

    # -- Length constants have meter dimension --
    assert constants.mile.dim == meter.dim
    assert constants.light_year.dim == meter.dim
    assert constants.angstrom.dim == meter.dim

    # -- Pressure constants have correct dimension (N / m^2) --
    expected_pressure_dim = (newton / meter2).dim
    assert constants.atm.dim == expected_pressure_dim
    assert constants.bar.dim == expected_pressure_dim
    assert constants.psi.dim == expected_pressure_dim

    # -- Energy constants have joule dimension --
    assert constants.eV.dim == joule.dim
    assert constants.calorie.dim == joule.dim
    assert constants.erg.dim == joule.dim

    # -- Force constants have newton dimension --
    assert constants.dyne.dim == newton.dim
    assert constants.pound_force.dim == newton.dim

    # -- Verify a computed relationship: 1 hour == 3600 seconds --
    assert u.math.allclose(
        constants.hour.mantissa,
        (3600.0 * second).mantissa,
    )


# ===========================================================================
# Fundamental constants — CODATA 2018 values
# ===========================================================================

class TestFundamentalConstantValues:
    """Verify fundamental constants match CODATA 2018."""

    def test_avogadro(self):
        import saiunit.constants as c
        np.testing.assert_allclose(float(c.avogadro.mantissa), 6.02214076e23, rtol=1e-8)

    def test_boltzmann(self):
        import saiunit.constants as c
        np.testing.assert_allclose(float(c.boltzmann.mantissa), 1.380649e-23, rtol=1e-8)

    def test_electric(self):
        import saiunit.constants as c
        np.testing.assert_allclose(float(c.electric.mantissa), 8.8541878188e-12, rtol=1e-8)

    def test_electron_mass(self):
        import saiunit.constants as c
        np.testing.assert_allclose(float(c.electron_mass.mantissa), 9.1093837139e-31, rtol=1e-8)

    def test_elementary_charge(self):
        import saiunit.constants as c
        np.testing.assert_allclose(float(c.elementary_charge.mantissa), 1.602176634e-19, rtol=1e-8)

    def test_faraday(self):
        import saiunit.constants as c
        np.testing.assert_allclose(float(c.faraday.mantissa), 96485.33212331, rtol=1e-8)

    def test_gas_constant(self):
        import saiunit.constants as c
        np.testing.assert_allclose(float(c.gas.mantissa), 8.314462618153240, rtol=1e-8)

    def test_magnetic(self):
        import saiunit.constants as c
        np.testing.assert_allclose(float(c.magnetic.mantissa), 1.25663706127e-6, rtol=1e-8)

    def test_molar_mass(self):
        import saiunit.constants as c
        np.testing.assert_allclose(float(c.molar_mass.mantissa), 1e-3, rtol=1e-12)

    def test_zero_celsius(self):
        import saiunit.constants as c
        np.testing.assert_allclose(float(c.zero_celsius.mantissa), 273.15, rtol=1e-12)


# ===========================================================================
# Quantity constant values — cross-check against known SI equivalents
# ===========================================================================

class TestQuantityConstantValues:
    """Verify each quantity constant's mantissa equals the known SI value."""

    def _check(self, name, expected, rtol=1e-8):
        import saiunit.constants as c
        val = float(getattr(c, name).mantissa)
        np.testing.assert_allclose(val, expected, rtol=rtol, err_msg=f"{name}")

    # Mass
    def test_metric_ton(self):
        self._check("metric_ton", 1e3)

    def test_grain(self):
        self._check("grain", 6.479891e-5)

    def test_pound(self):
        self._check("pound", 0.45359237)

    def test_blob(self):
        self._check("blob", 175.126836, rtol=1e-6)

    def test_slug(self):
        self._check("slug", 14.59390294)

    def test_ounce(self):
        self._check("ounce", 2.8349523125e-2)

    def test_stone(self):
        self._check("stone", 6.35029318)

    def test_long_ton(self):
        self._check("long_ton", 1016.0469088)

    def test_short_ton(self):
        self._check("short_ton", 907.18474)

    def test_troy_ounce(self):
        self._check("troy_ounce", 3.11034768e-2)

    def test_troy_pound(self):
        self._check("troy_pound", 0.3732417216)

    def test_carat(self):
        self._check("carat", 2e-4)

    def test_atomic_mass(self):
        self._check("atomic_mass", 1.66053906892e-27)

    # Angle
    def test_degree(self):
        self._check("degree", np.pi / 180)

    def test_arcmin(self):
        self._check("arcmin", np.pi / 10800)

    def test_arcsec(self):
        self._check("arcsec", np.pi / 648000)

    # Time
    def test_minute(self):
        self._check("minute", 60.0)

    def test_hour(self):
        self._check("hour", 3600.0)

    def test_day(self):
        self._check("day", 86400.0)

    def test_week(self):
        self._check("week", 604800.0)

    def test_year(self):
        self._check("year", 365.2425 * 86400)

    def test_julian_year(self):
        self._check("julian_year", 365.25 * 86400)

    # Length
    def test_inch(self):
        self._check("inch", 0.0254)

    def test_foot(self):
        self._check("foot", 0.3048)

    def test_yard(self):
        self._check("yard", 0.9144)

    def test_mile(self):
        self._check("mile", 1609.344)

    def test_mil(self):
        self._check("mil", 2.54e-5)

    def test_nautical_mile(self):
        self._check("nautical_mile", 1852.0)

    def test_angstrom(self):
        self._check("angstrom", 1e-10)

    def test_au(self):
        self._check("au", 1.495978707e11)

    def test_light_year(self):
        self._check("light_year", 9.46073047258080e15)

    def test_parsec(self):
        self._check("parsec", 3.085677581491367e16)

    # Pressure
    def test_atm(self):
        self._check("atm", 101325.0)

    def test_bar(self):
        self._check("bar", 1e5)

    def test_torr(self):
        self._check("torr", 101325.0 / 760)

    def test_psi(self):
        self._check("psi", 6894.757293168361)

    # Area
    def test_hectare(self):
        self._check("hectare", 1e4)

    def test_acre(self):
        self._check("acre", 4046.8564224)

    # Volume
    def test_gallon(self):
        self._check("gallon", 3.785411784e-3)

    def test_gallon_imp(self):
        self._check("gallon_imp", 4.54609e-3)

    def test_fluid_ounce(self):
        self._check("fluid_ounce", 2.95735295625e-5)

    def test_fluid_ounce_imp(self):
        self._check("fluid_ounce_imp", 2.84130625e-5)

    def test_barrel(self):
        self._check("bbl", 0.158987294928)

    # Speed
    def test_kmh(self):
        self._check("kmh", 1000.0 / 3600, rtol=1e-6)

    def test_mph(self):
        self._check("mph", 0.44704)

    def test_knot(self):
        self._check("knot", 1852.0 / 3600, rtol=1e-6)

    def test_mach(self):
        self._check("mach", 340.5)

    # Energy
    def test_ev(self):
        self._check("eV", 1.602176634e-19)

    def test_calorie(self):
        self._check("calorie", 4.184)

    def test_calorie_it(self):
        self._check("calorie_IT", 4.1868)

    def test_erg(self):
        self._check("erg", 1e-7)

    def test_btu(self):
        self._check("Btu", 1055.05585262)

    def test_btu_th(self):
        self._check("Btu_th", 1054.350264488889)

    def test_ton_tnt(self):
        self._check("ton_TNT", 4.184e9)

    # Power
    def test_hp(self):
        self._check("hp", 745.69987158227022)

    # Force
    def test_dyn(self):
        self._check("dyn", 1e-5)

    def test_lbf(self):
        self._check("lbf", 4.4482216152605)

    def test_kgf(self):
        self._check("kgf", 9.80665)

    def test_imf(self):
        self._check("IMF", 1.602176634e-9)


# ===========================================================================
# Cross-check quantity constants against scipy.constants
# ===========================================================================

# ``parametrize`` evaluates its argument list at class-body time, so
# referencing ``sc.<name>`` directly would crash collection on installs
# without scipy (e.g., the per-backend CI jobs). Build the list under the
# HAS_SCIPY guard and fall back to an empty list — the class-level skipif
# then has nothing to skip, which is the intended outcome.
_SCIPY_QUANTITY_PAIRS = [
    ("grain", sc.grain),
    ("pound", sc.pound),
    ("ounce", sc.oz),
    ("stone", sc.stone),
    ("long_ton", sc.long_ton),
    ("short_ton", sc.short_ton),
    ("troy_ounce", sc.troy_ounce),
    ("troy_pound", sc.troy_pound),
    ("carat", sc.carat),
    ("atomic_mass", sc.atomic_mass),
    ("minute", sc.minute),
    ("hour", sc.hour),
    ("day", sc.day),
    ("week", sc.week),
    ("julian_year", sc.Julian_year),
    ("inch", sc.inch),
    ("foot", sc.foot),
    ("yard", sc.yard),
    ("mile", sc.mile),
    ("mil", sc.mil),
    ("nautical_mile", sc.nautical_mile),
    ("angstrom", sc.angstrom),
    ("au", sc.au),
    ("light_year", sc.light_year),
    ("parsec", sc.parsec),
    ("atm", sc.atm),
    ("bar", sc.bar),
    ("torr", sc.torr),
    ("psi", sc.psi),
    ("hectare", sc.hectare),
    ("acre", sc.acre),
    ("gallon", sc.gallon),
    ("gallon_imp", sc.gallon_imp),
    ("fluid_ounce", sc.fluid_ounce),
    ("fluid_ounce_imp", sc.fluid_ounce_imp),
    ("bbl", sc.bbl),
    ("kmh", sc.kmh),
    ("mph", sc.mph),
    ("mach", sc.mach),
    ("knot", sc.knot),
    ("eV", sc.eV),
    ("calorie", sc.calorie),
    ("calorie_IT", sc.calorie_IT),
    ("erg", sc.erg),
    ("Btu", sc.Btu),
    ("Btu_th", sc.Btu_th),
    ("ton_TNT", sc.ton_TNT),
    ("hp", sc.hp),
    ("dyn", sc.dyn),
    ("lbf", sc.lbf),
    ("kgf", sc.kgf),
] if HAS_SCIPY else []


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestQuantityConstantsAgainstScipy:
    """Verify quantity constants match scipy.constants values."""

    def _check(self, name, scipy_val, rtol=1e-8):
        import saiunit.constants as c
        val = float(getattr(c, name).mantissa)
        np.testing.assert_allclose(val, scipy_val, rtol=rtol, err_msg=f"{name}")

    @pytest.mark.parametrize("name, scipy_val", _SCIPY_QUANTITY_PAIRS)
    def test_matches_scipy(self, name, scipy_val):
        self._check(name, scipy_val)


# ===========================================================================
# Consistency checks between related constants
# ===========================================================================

class TestQuantityConstantRelationships:
    """Test known physical relationships between constants."""

    def test_gas_equals_avogadro_times_boltzmann(self):
        import saiunit.constants as c
        computed = float((c.avogadro * c.boltzmann).mantissa)
        expected = float(c.gas.mantissa)
        np.testing.assert_allclose(computed, expected, rtol=1e-6)

    def test_faraday_equals_avogadro_times_charge(self):
        import saiunit.constants as c
        computed = float((c.avogadro * c.elementary_charge).mantissa)
        expected = float(c.faraday.mantissa)
        np.testing.assert_allclose(computed, expected, rtol=1e-6)

    def test_barrel_equals_42_gallons(self):
        import saiunit.constants as c
        np.testing.assert_allclose(
            float(c.barrel.mantissa),
            42 * float(c.gallon.mantissa),
            rtol=1e-8,
        )

    def test_mile_equals_5280_feet(self):
        import saiunit.constants as c
        np.testing.assert_allclose(
            float(c.mile.mantissa),
            5280 * float(c.foot.mantissa),
            rtol=1e-10,
        )

    def test_yard_equals_3_feet(self):
        import saiunit.constants as c
        np.testing.assert_allclose(
            float(c.yard.mantissa),
            3 * float(c.foot.mantissa),
            rtol=1e-10,
        )

    def test_foot_equals_12_inches(self):
        import saiunit.constants as c
        np.testing.assert_allclose(
            float(c.foot.mantissa),
            12 * float(c.inch.mantissa),
            rtol=1e-10,
        )

    def test_day_equals_24_hours(self):
        import saiunit.constants as c
        np.testing.assert_allclose(
            float(c.day.mantissa),
            24 * float(c.hour.mantissa),
            rtol=1e-10,
        )

    def test_week_equals_7_days(self):
        import saiunit.constants as c
        np.testing.assert_allclose(
            float(c.week.mantissa),
            7 * float(c.day.mantissa),
            rtol=1e-10,
        )

    def test_imp_fluid_ounce_equals_imp_gallon_over_160(self):
        import saiunit.constants as c
        np.testing.assert_allclose(
            float(c.fluid_ounce_imp.mantissa),
            float(c.gallon_imp.mantissa) / 160,
            rtol=1e-8,
        )

    def test_us_gallon_equals_128_fluid_ounces(self):
        import saiunit.constants as c
        np.testing.assert_allclose(
            float(c.gallon.mantissa),
            128 * float(c.fluid_ounce.mantissa),
            rtol=1e-8,
        )


def test_added_constant_dimensions():
    import saiunit.constants as constants

    # 10 dual-defined Quantity names (also exercised by the cross-check loop).
    assert constants.arcminute.dim == u.radian.dim
    assert constants.arcsecond.dim == u.radian.dim
    assert constants.astronomical_unit.dim == meter.dim
    assert constants.atmosphere.dim == (newton / meter2).dim
    assert constants.calorie_th.dim == joule.dim
    assert constants.fluid_ounce_US.dim == (meter ** 3).dim
    assert constants.gallon_US.dim == (meter ** 3).dim
    assert constants.horsepower.dim == watt.dim
    assert constants.kilogram_force.dim == newton.dim
    assert constants.lb.dim == kilogram.dim

    # 3 names exposed as Unit objects in saiunit.constants.
    assert constants.radian.dim == u.radian.dim
    assert constants.speed_unit.dim == (meter / second).dim
    assert constants.watt.dim == watt.dim

    # 2 names defined only in saiunit.constants.
    assert constants.electronvolt.dim == joule.dim
    assert constants.gram.dim == kilogram.dim
    # gram is one-thousandth of a kilogram. Use a plain-float comparison so the
    # check stays backend-agnostic (torch/ndonnx have no float ``isclose`` op).
    assert abs(float((1.0 * constants.gram).to_decimal(kilogram)) - 0.001) < 1e-9


class TestNewPhysicalConstants:
    """Regression tests for newly added constants: c, h, hbar, G, g_n, sigma."""

    def test_speed_of_light(self):
        c = u.constants.speed_of_light
        assert c.unit.has_same_dim(u.meter / u.second)
        assert float(c.to_decimal(u.meter / u.second)) == 2.99792458e8

    def test_planck(self):
        h = u.constants.planck
        assert h.unit.has_same_dim(u.joule * u.second)
        assert float(h.to_decimal(u.joule * u.second)) == 6.62607015e-34

    def test_hbar(self):
        hbar = u.constants.hbar
        assert hbar.unit.has_same_dim(u.joule * u.second)
        np.testing.assert_allclose(
            float(hbar.to_decimal(u.joule * u.second)), 1.054571817e-34, rtol=1e-12
        )

    def test_hbar_is_h_over_2pi(self):
        h = float(u.constants.planck.to_decimal(u.joule * u.second))
        hbar = float(u.constants.hbar.to_decimal(u.joule * u.second))
        np.testing.assert_allclose(hbar, h / (2 * np.pi), rtol=1e-9)

    def test_gravitational_constant(self):
        G = u.constants.gravitational_constant
        assert G.unit.has_same_dim(u.meter ** 3 / u.kilogram / u.second ** 2)
        assert float(G.to_decimal(u.meter ** 3 / u.kilogram / u.second ** 2)) == 6.67430e-11

    def test_standard_gravity(self):
        g = u.constants.standard_gravity
        assert g.unit.has_same_dim(u.meter / u.second ** 2)
        assert float(g.to_decimal(u.meter / u.second ** 2)) == 9.80665

    def test_stefan_boltzmann(self):
        sigma = u.constants.stefan_boltzmann
        assert sigma.unit.has_same_dim(u.watt / u.meter2 / u.kelvin ** 4)
        assert float(sigma.to_decimal(u.watt / u.meter2 / u.kelvin ** 4)) == 5.670374419e-8

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_against_scipy(self):
        assert float(u.constants.speed_of_light.to_decimal(u.meter / u.second)) == sc.c
        assert float(u.constants.planck.to_decimal(u.joule * u.second)) == sc.h
        assert float(u.constants.gravitational_constant.to_decimal(
            u.meter ** 3 / u.kilogram / u.second ** 2)) == sc.G
        assert float(u.constants.standard_gravity.to_decimal(u.meter / u.second ** 2)) == sc.g
