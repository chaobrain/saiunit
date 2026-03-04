# -*- coding: utf-8 -*-
"""
Comprehensive tests for non-SI unit constants.

Each unit is verified against its known SI conversion value
(cross-referenced with scipy.constants and NIST CODATA 2018).
"""

import pytest
import numpy as np

try:
    import scipy.constants as sc

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import saiunit as u
from saiunit._unit_constants import (
    metric_ton, grain, lb, pound, slinch, blob, slug, oz, ounce, stone,
    long_ton, short_ton, troy_ounce, troy_pound, carat, atomic_mass, amu,
    u as unit_u, um_u,
    degree, arcmin, arcminute, arcsec, arcsecond,
    minute, hour, day, week, month, year, julian_year,
    inch, foot, yard, mile, mil, point, pica, survey_foot, survey_mile,
    nautical_mile, fermi, angstrom, micron, astronomical_unit, au, light_year, parsec,
    atm, atmosphere, bar, mmHg, torr, psi,
    hectare, acre,
    gallon, gallon_US, gallon_imp, fluid_ounce, fluid_ounce_US, fluid_ounce_imp, bbl, barrel,
    speed_unit, kmh, mph, mach, speed_of_sound, knot,
    degree_Fahrenheit,
    eV, electron_volt, calorie, calorie_th, calorie_IT, erg, Btu, Btu_IT, Btu_th, ton_TNT,
    hp, horsepower, kcal_per_h,
    dyn, dyne, lbf, pound_force, kgf, kilogram_force, IMF,
)
from saiunit._base_unit import Unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_magnitude(unit_obj, expected, rtol=1e-9, msg=""):
    """Assert that a unit's magnitude matches the expected SI value."""
    got = unit_obj.magnitude
    np.testing.assert_allclose(got, expected, rtol=rtol, err_msg=msg or f"{unit_obj}")


# ===========================================================================
# Test aliases point to the same Unit object
# ===========================================================================

class TestAliases:
    """Verify that aliases reference the identical Unit instance."""

    def test_mass_aliases(self):
        assert lb is pound
        assert oz is ounce
        assert slinch is blob
        assert unit_u is atomic_mass
        assert um_u is atomic_mass
        assert amu is atomic_mass

    def test_angle_aliases(self):
        assert arcmin is arcminute
        assert arcsec is arcsecond

    def test_length_aliases(self):
        assert au is astronomical_unit

    def test_pressure_aliases(self):
        assert atm is atmosphere

    def test_volume_aliases(self):
        assert gallon is gallon_US
        assert fluid_ounce is fluid_ounce_US
        assert bbl is barrel

    def test_speed_aliases(self):
        assert mach is speed_of_sound

    def test_energy_aliases(self):
        assert eV is electron_volt
        assert calorie is calorie_th
        assert Btu is Btu_IT

    def test_power_aliases(self):
        assert hp is horsepower

    def test_force_aliases(self):
        assert dyn is dyne
        assert lbf is pound_force
        assert kgf is kilogram_force


# ===========================================================================
# Test that all objects are Unit instances
# ===========================================================================

class TestUnitType:
    """All exports must be Unit instances (except speed_unit which is derived)."""

    @pytest.mark.parametrize("obj", [
        metric_ton, grain, pound, slug, ounce, stone, long_ton, short_ton,
        troy_ounce, troy_pound, carat, atomic_mass, blob,
        degree, arcminute, arcsecond,
        minute, hour, day, week, month, year, julian_year,
        inch, foot, yard, mile, mil, point, pica, survey_foot, survey_mile,
        nautical_mile, fermi, angstrom, micron, astronomical_unit, light_year, parsec,
        atmosphere, bar, torr, psi,
        hectare, acre,
        gallon, gallon_imp, fluid_ounce, fluid_ounce_imp, barrel,
        speed_unit, kmh, mph, mach, knot,
        degree_Fahrenheit,
        electron_volt, calorie, calorie_IT, erg, Btu, Btu_th, ton_TNT,
        horsepower, kcal_per_h,
        dyne, pound_force, kilogram_force, IMF,
    ])
    def test_is_unit(self, obj):
        assert isinstance(obj, Unit), f"{obj} is not a Unit"


# ===========================================================================
# Magnitude tests — exact known SI values
# ===========================================================================

class TestMassMagnitudes:
    def test_metric_ton(self):
        assert_magnitude(metric_ton, 1000.0)

    def test_grain(self):
        assert_magnitude(grain, 6.479891e-5)

    def test_pound(self):
        assert_magnitude(pound, 0.45359237)

    def test_blob(self):
        # 1 blob = 1 lbf·s²/in
        assert_magnitude(blob, 175.126836)

    def test_slug(self):
        assert_magnitude(slug, 14.59390294)

    def test_ounce(self):
        assert_magnitude(ounce, 0.028349523125)

    def test_stone(self):
        assert_magnitude(stone, 6.35029318)

    def test_long_ton(self):
        assert_magnitude(long_ton, 1016.0469088)

    def test_short_ton(self):
        assert_magnitude(short_ton, 907.18474)

    def test_troy_ounce(self):
        assert_magnitude(troy_ounce, 0.0311034768)

    def test_troy_pound(self):
        assert_magnitude(troy_pound, 0.3732417216)

    def test_carat(self):
        assert_magnitude(carat, 2e-4)

    def test_atomic_mass(self):
        # CODATA 2018
        assert_magnitude(atomic_mass, 1.66053906892e-27)


class TestAngleMagnitudes:
    def test_degree(self):
        assert_magnitude(degree, np.pi / 180)

    def test_arcminute(self):
        assert_magnitude(arcminute, np.pi / 10800)

    def test_arcsecond(self):
        assert_magnitude(arcsecond, np.pi / 648000)


class TestTimeMagnitudes:
    def test_minute(self):
        assert_magnitude(minute, 60.0)

    def test_hour(self):
        assert_magnitude(hour, 3600.0)

    def test_day(self):
        assert_magnitude(day, 86400.0)

    def test_week(self):
        assert_magnitude(week, 604800.0)

    def test_month(self):
        # Gregorian month: 365.2425/12 days
        assert_magnitude(month, 365.2425 / 12 * 86400)

    def test_year(self):
        # Gregorian year: 365.2425 days
        assert_magnitude(year, 365.2425 * 86400)

    def test_julian_year(self):
        assert_magnitude(julian_year, 365.25 * 86400)


class TestLengthMagnitudes:
    def test_inch(self):
        assert_magnitude(inch, 0.0254)

    def test_foot(self):
        assert_magnitude(foot, 0.3048)

    def test_yard(self):
        assert_magnitude(yard, 0.9144)

    def test_mile(self):
        assert_magnitude(mile, 1609.344)

    def test_mil(self):
        assert_magnitude(mil, 2.54e-5)

    def test_point(self):
        # 1 point = 1/72 inch
        assert_magnitude(point, 0.0254 / 72)

    def test_pica(self):
        # 1 pica = 12 points = 1/6 inch
        assert_magnitude(pica, 0.0254 / 6)

    def test_survey_foot(self):
        assert_magnitude(survey_foot, 0.3048006096012192)

    def test_survey_mile(self):
        assert_magnitude(survey_mile, 1609.3472186944374)

    def test_nautical_mile(self):
        assert_magnitude(nautical_mile, 1852.0)

    def test_fermi(self):
        assert_magnitude(fermi, 1e-15)

    def test_angstrom(self):
        assert_magnitude(angstrom, 1e-10)

    def test_micron(self):
        assert_magnitude(micron, 1e-6)

    def test_astronomical_unit(self):
        assert_magnitude(astronomical_unit, 1.495978707e11)

    def test_light_year(self):
        # c × Julian year = 299792458 × 31557600
        assert_magnitude(light_year, 9.46073047258080e15)

    def test_parsec(self):
        assert_magnitude(parsec, 3.085677581491367e16)


class TestPressureMagnitudes:
    def test_atmosphere(self):
        assert_magnitude(atmosphere, 101325.0)

    def test_bar(self):
        assert_magnitude(bar, 1e5)

    def test_torr(self):
        # Exact: 101325/760
        assert_magnitude(torr, 101325.0 / 760)

    def test_psi(self):
        assert_magnitude(psi, 6894.757293168361)


class TestAreaMagnitudes:
    def test_hectare(self):
        assert_magnitude(hectare, 1e4)

    def test_acre(self):
        # 43560 ft² (international)
        assert_magnitude(acre, 4046.8564224)


class TestVolumeMagnitudes:
    def test_gallon_us(self):
        assert_magnitude(gallon, 3.785411784e-3)

    def test_gallon_imp(self):
        assert_magnitude(gallon_imp, 4.54609e-3)

    def test_fluid_ounce_us(self):
        assert_magnitude(fluid_ounce, 2.95735295625e-5)

    def test_fluid_ounce_imp(self):
        # Exact: gallon_imp / 160
        assert_magnitude(fluid_ounce_imp, 4.54609e-3 / 160)

    def test_barrel(self):
        # 42 US gallons
        assert_magnitude(barrel, 0.158987294928)


class TestSpeedMagnitudes:
    def test_kmh(self):
        assert_magnitude(kmh, 1000.0 / 3600)

    def test_mph(self):
        assert_magnitude(mph, 0.44704)

    def test_mach(self):
        assert_magnitude(mach, 340.5)

    def test_knot(self):
        assert_magnitude(knot, 1852.0 / 3600)


class TestTemperatureMagnitudes:
    def test_degree_fahrenheit(self):
        # Scale factor only (no offset)
        assert_magnitude(degree_Fahrenheit, 5.0 / 9)


class TestEnergyMagnitudes:
    def test_electronvolt(self):
        # CODATA 2018 exact
        assert_magnitude(electron_volt, 1.602176634e-19)

    def test_calorie_th(self):
        assert_magnitude(calorie, 4.184)

    def test_calorie_it(self):
        assert_magnitude(calorie_IT, 4.1868)

    def test_erg(self):
        assert_magnitude(erg, 1e-7)

    def test_btu_it(self):
        assert_magnitude(Btu, 1055.05585262)

    def test_btu_th(self):
        assert_magnitude(Btu_th, 1054.350264488889)

    def test_ton_tnt(self):
        assert_magnitude(ton_TNT, 4.184e9)


class TestPowerMagnitudes:
    def test_horsepower(self):
        assert_magnitude(horsepower, 745.69987158227022)

    def test_kcal_per_h(self):
        assert_magnitude(kcal_per_h, 4184.0 / 3600, rtol=1e-5)


class TestForceMagnitudes:
    def test_dyne(self):
        assert_magnitude(dyne, 1e-5)

    def test_pound_force(self):
        assert_magnitude(pound_force, 4.4482216152605)

    def test_kilogram_force(self):
        assert_magnitude(kilogram_force, 9.80665)

    def test_imf(self):
        assert_magnitude(IMF, 1.602176634e-9)


# ===========================================================================
# Cross-check against scipy.constants
# ===========================================================================

@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestAgainstScipy:
    """Verify every unit matches scipy.constants to within 1e-8 relative."""

    @pytest.mark.parametrize("unit_obj, scipy_val", [
        (grain, sc.grain),
        (pound, sc.pound),
        (ounce, sc.oz),
        (stone, sc.stone),
        (long_ton, sc.long_ton),
        (short_ton, sc.short_ton),
        (troy_ounce, sc.troy_ounce),
        (troy_pound, sc.troy_pound),
        (carat, sc.carat),
        (atomic_mass, sc.atomic_mass),
        (degree, sc.degree),
        (arcminute, sc.arcmin),
        (arcsecond, sc.arcsec),
        (minute, sc.minute),
        (hour, sc.hour),
        (day, sc.day),
        (week, sc.week),
        (julian_year, sc.Julian_year),
        (inch, sc.inch),
        (foot, sc.foot),
        (yard, sc.yard),
        (mile, sc.mile),
        (mil, sc.mil),
        (point, sc.point),
        (survey_foot, sc.survey_foot),
        (survey_mile, sc.survey_mile),
        (nautical_mile, sc.nautical_mile),
        (fermi, sc.fermi),
        (angstrom, sc.angstrom),
        (micron, sc.micron),
        (astronomical_unit, sc.au),
        (light_year, sc.light_year),
        (parsec, sc.parsec),
        (atmosphere, sc.atm),
        (bar, sc.bar),
        (torr, sc.torr),
        (psi, sc.psi),
        (hectare, sc.hectare),
        (acre, sc.acre),
        (gallon, sc.gallon),
        (gallon_imp, sc.gallon_imp),
        (fluid_ounce, sc.fluid_ounce),
        (fluid_ounce_imp, sc.fluid_ounce_imp),
        (barrel, sc.bbl),
        (kmh, sc.kmh),
        (mph, sc.mph),
        (mach, sc.mach),
        (knot, sc.knot),
        (degree_Fahrenheit, sc.degree_Fahrenheit),
        (electron_volt, sc.eV),
        (calorie, sc.calorie),
        (calorie_IT, sc.calorie_IT),
        (erg, sc.erg),
        (Btu, sc.Btu),
        (Btu_th, sc.Btu_th),
        (ton_TNT, sc.ton_TNT),
        (horsepower, sc.hp),
        (dyne, sc.dyn),
        (pound_force, sc.lbf),
        (kilogram_force, sc.kgf),
    ], ids=lambda x: getattr(x, 'name', str(x)))
    def test_matches_scipy(self, unit_obj, scipy_val):
        np.testing.assert_allclose(
            unit_obj.magnitude, scipy_val, rtol=1e-8,
            err_msg=f"{unit_obj.name}: magnitude={unit_obj.magnitude}, scipy={scipy_val}"
        )


# ===========================================================================
# Display name (dispname) tests
# ===========================================================================

class TestDispnames:
    """Verify display names are correct and unambiguous."""

    def test_mass_dispnames(self):
        assert str(metric_ton) == "t"
        assert str(grain) == "gr"
        assert str(pound) == "lb"
        assert str(ounce) == "oz"
        assert str(stone) == "st"
        assert str(troy_ounce) == "oz t"
        assert str(troy_pound) == "lb t"
        assert str(carat) == "ct"
        assert str(atomic_mass) == "u"
        assert str(blob) == "blob"
        assert str(slug) == "slug"

    def test_angle_dispnames(self):
        assert str(degree) == "°"
        assert str(arcminute) == "′"
        assert str(arcsecond) == "″"

    def test_time_dispnames(self):
        assert str(minute) == "min"
        assert str(hour) == "h"
        assert str(day) == "d"
        assert str(week) == "wk"
        assert str(month) == "mon"
        assert str(year) == "yr"
        assert str(julian_year) == "julian yr"

    def test_length_dispnames(self):
        assert str(inch) == "in"
        assert str(foot) == "ft"
        assert str(yard) == "yd"
        assert str(mile) == "mi"
        assert str(nautical_mile) == "nmi"
        assert str(fermi) == "fm"
        assert str(angstrom) == "Å"
        assert str(micron) == "µm"
        assert str(astronomical_unit) == "AU"
        assert str(light_year) == "ly"
        assert str(parsec) == "pc"

    def test_survey_dispnames_distinct(self):
        """survey_foot/survey_mile must NOT have same dispname as foot/mile."""
        assert str(survey_foot) != str(foot)
        assert str(survey_mile) != str(mile)

    def test_imperial_gallon_distinct(self):
        """Imperial gallon must NOT have same dispname as US gallon."""
        assert str(gallon_imp) != str(gallon)

    def test_pressure_dispnames(self):
        assert str(atmosphere) == "atm"
        assert str(bar) == "bar"
        assert str(torr) == "torr"
        assert str(psi) == "psi"

    def test_volume_dispnames(self):
        assert str(gallon) == "gal"
        assert str(fluid_ounce) == "fl oz"
        assert str(barrel) == "bbl"

    def test_speed_dispnames(self):
        assert str(kmh) == "km/h"
        assert str(mph) == "mph"
        assert str(mach) == "mach"
        assert str(knot) == "kn"

    def test_energy_dispnames(self):
        assert str(electron_volt) == "eV"
        assert str(calorie) == "cal"
        assert str(erg) == "erg"
        assert str(Btu) == "Btu"

    def test_power_dispnames(self):
        assert str(horsepower) == "hp"

    def test_force_dispnames(self):
        assert str(dyne) == "dyn"
        assert str(pound_force) == "lbf"
        assert str(kilogram_force) == "kgf"


# ===========================================================================
# Dimension correctness tests
# ===========================================================================

class TestDimensions:
    """Each unit must have the correct physical dimension."""

    def test_mass_dimensions(self):
        from saiunit._unit_common import kilogram
        for unit_obj in [metric_ton, grain, pound, slug, ounce, stone,
                         long_ton, short_ton, troy_ounce, troy_pound,
                         carat, atomic_mass, blob]:
            assert unit_obj.dim == kilogram.dim, f"{unit_obj.name} dim mismatch"

    def test_angle_dimensions(self):
        from saiunit._unit_common import radian
        for unit_obj in [degree, arcminute, arcsecond]:
            assert unit_obj.dim == radian.dim, f"{unit_obj.name} dim mismatch"

    def test_time_dimensions(self):
        from saiunit._unit_common import second
        for unit_obj in [minute, hour, day, week, month, year, julian_year]:
            assert unit_obj.dim == second.dim, f"{unit_obj.name} dim mismatch"

    def test_length_dimensions(self):
        from saiunit._unit_common import meter
        for unit_obj in [inch, foot, yard, mile, mil, point, pica,
                         survey_foot, survey_mile, nautical_mile,
                         fermi, angstrom, micron, astronomical_unit,
                         light_year, parsec]:
            assert unit_obj.dim == meter.dim, f"{unit_obj.name} dim mismatch"

    def test_pressure_dimensions(self):
        from saiunit._unit_common import pascal
        for unit_obj in [atmosphere, bar, torr, psi]:
            assert unit_obj.dim == pascal.dim, f"{unit_obj.name} dim mismatch"

    def test_area_dimensions(self):
        from saiunit._unit_common import meter2
        for unit_obj in [hectare, acre]:
            assert unit_obj.dim == meter2.dim, f"{unit_obj.name} dim mismatch"

    def test_volume_dimensions(self):
        from saiunit._unit_common import meter3
        for unit_obj in [gallon, gallon_imp, fluid_ounce,
                         fluid_ounce_imp, barrel]:
            assert unit_obj.dim == meter3.dim, f"{unit_obj.name} dim mismatch"

    def test_energy_dimensions(self):
        from saiunit._unit_common import joule
        for unit_obj in [electron_volt, calorie, calorie_IT, erg,
                         Btu, Btu_th, ton_TNT]:
            assert unit_obj.dim == joule.dim, f"{unit_obj.name} dim mismatch"

    def test_power_dimensions(self):
        from saiunit._unit_common import watt
        for unit_obj in [horsepower, kcal_per_h]:
            assert unit_obj.dim == watt.dim, f"{unit_obj.name} dim mismatch"

    def test_force_dimensions(self):
        from saiunit._unit_common import newton
        for unit_obj in [dyne, pound_force, kilogram_force, IMF]:
            assert unit_obj.dim == newton.dim, f"{unit_obj.name} dim mismatch"

    def test_temperature_dimensions(self):
        from saiunit._unit_common import kelvin
        assert degree_Fahrenheit.dim == kelvin.dim


# ===========================================================================
# Conversion round-trip tests
# ===========================================================================

class TestConversions:
    """Test that quantities convert correctly between unit systems."""

    def test_pound_to_kg(self):
        q = 1.0 * pound
        result = q.to_decimal(u.kilogram)
        np.testing.assert_allclose(float(result), 0.45359237, rtol=1e-9)

    def test_mile_to_km(self):
        q = 1.0 * mile
        result = q.to_decimal(u.kmeter)
        np.testing.assert_allclose(float(result), 1.609344, rtol=1e-9)

    def test_atm_to_pascal(self):
        q = 1.0 * atmosphere
        result = q.to_decimal(u.pascal)
        np.testing.assert_allclose(float(result), 101325.0, rtol=1e-9)

    def test_gallon_to_liter(self):
        q = 1.0 * gallon
        result = q.to_decimal(u.liter)
        np.testing.assert_allclose(float(result), 3.785411784, rtol=1e-9)

    def test_barrel_to_gallon(self):
        q = 1.0 * barrel
        result = q.to_decimal(gallon)
        np.testing.assert_allclose(float(result), 42.0, rtol=1e-6)

    def test_horsepower_to_watt(self):
        q = 1.0 * horsepower
        result = q.to_decimal(u.watt)
        np.testing.assert_allclose(float(result), 745.69987158227022, rtol=1e-9)

    def test_ev_to_joule(self):
        q = 1.0 * electron_volt
        result = q.to_decimal(u.joule)
        np.testing.assert_allclose(float(result), 1.602176634e-19, rtol=1e-9)

    def test_nautical_mile_to_meter(self):
        q = 1.0 * nautical_mile
        result = q.to_decimal(u.meter)
        np.testing.assert_allclose(float(result), 1852.0, rtol=1e-9)

    def test_knot_to_mps(self):
        q = 1.0 * knot
        result = q.to_decimal(u.meter / u.second)
        np.testing.assert_allclose(float(result), 1852.0 / 3600, rtol=1e-6)

    def test_acre_to_hectare(self):
        q = 1.0 * acre
        result = q.to_decimal(hectare)
        np.testing.assert_allclose(float(result), 0.40468564224, rtol=1e-8)


# ===========================================================================
# Consistency tests (relationships between units)
# ===========================================================================

class TestInternalConsistency:
    """Test known relationships between units."""

    def test_foot_equals_12_inches(self):
        np.testing.assert_allclose(foot.magnitude, 12 * inch.magnitude, rtol=1e-12)

    def test_yard_equals_3_feet(self):
        np.testing.assert_allclose(yard.magnitude, 3 * foot.magnitude, rtol=1e-12)

    def test_mile_equals_5280_feet(self):
        np.testing.assert_allclose(mile.magnitude, 5280 * foot.magnitude, rtol=1e-12)

    def test_stone_equals_14_pounds(self):
        np.testing.assert_allclose(stone.magnitude, 14 * pound.magnitude, rtol=1e-9)

    def test_short_ton_equals_2000_pounds(self):
        np.testing.assert_allclose(short_ton.magnitude, 2000 * pound.magnitude, rtol=1e-9)

    def test_long_ton_equals_2240_pounds(self):
        np.testing.assert_allclose(long_ton.magnitude, 2240 * pound.magnitude, rtol=1e-9)

    def test_troy_pound_equals_12_troy_ounces(self):
        np.testing.assert_allclose(troy_pound.magnitude, 12 * troy_ounce.magnitude, rtol=1e-9)

    def test_ounce_equals_pound_over_16(self):
        np.testing.assert_allclose(ounce.magnitude, pound.magnitude / 16, rtol=1e-9)

    def test_mil_equals_inch_over_1000(self):
        np.testing.assert_allclose(mil.magnitude, inch.magnitude / 1000, rtol=1e-12)

    def test_pica_equals_12_points(self):
        np.testing.assert_allclose(pica.magnitude, 12 * point.magnitude, rtol=1e-9)

    def test_arcmin_equals_degree_over_60(self):
        np.testing.assert_allclose(arcminute.magnitude, degree.magnitude / 60, rtol=1e-12)

    def test_arcsec_equals_arcmin_over_60(self):
        np.testing.assert_allclose(arcsecond.magnitude, arcminute.magnitude / 60, rtol=1e-12)

    def test_hour_equals_60_minutes(self):
        np.testing.assert_allclose(hour.magnitude, 60 * minute.magnitude, rtol=1e-12)

    def test_day_equals_24_hours(self):
        np.testing.assert_allclose(day.magnitude, 24 * hour.magnitude, rtol=1e-12)

    def test_week_equals_7_days(self):
        np.testing.assert_allclose(week.magnitude, 7 * day.magnitude, rtol=1e-12)

    def test_gallon_equals_128_fluid_ounces(self):
        np.testing.assert_allclose(gallon.magnitude, 128 * fluid_ounce.magnitude, rtol=1e-9)

    def test_barrel_equals_42_gallons(self):
        np.testing.assert_allclose(barrel.magnitude, 42 * gallon.magnitude, rtol=1e-9)

    def test_imperial_fluid_ounce_equals_gallon_imp_over_160(self):
        np.testing.assert_allclose(fluid_ounce_imp.magnitude, gallon_imp.magnitude / 160, rtol=1e-9)
