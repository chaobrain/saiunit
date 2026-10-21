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

"""Tests for _unit_parse.py and the display/parse round-trip guarantees."""

import itertools
from fractions import Fraction

import pytest

import saiunit as u
from saiunit import _unit_common, _unit_constants
from saiunit._base_unit import Unit, UNITLESS, parse_unit
from saiunit._unit_constants import kmh, troy_ounce
from saiunit._unit_terms import derive_dim, derive_magnitude


class TestGrammar:
    def test_division_binds_looser_than_product(self):
        assert parse_unit("m / s * A") == u.metre / (u.second * u.amp)

    def test_chained_division_left_associative(self):
        assert parse_unit("m / s / A") == u.metre / u.second / u.amp

    @pytest.mark.parametrize("s", ["m^(1/3)", "m^0.3333333333333333", "m^(+1/3)"])
    def test_rational_exponent(self, s):
        assert parse_unit(s)._terms == (u.metre ** Fraction(1, 3))._terms

    def test_negative_exponents(self):
        assert parse_unit("mV^-1") == u.mvolt.reverse()
        assert parse_unit("m^(-1/2)") == u.metre ** -0.5

    @pytest.mark.parametrize("s", ["µF", "μF", "uF"])
    def test_micro_spellings(self, s):
        assert parse_unit(s) == u.ufarad

    def test_unicode_symbols(self):
        assert parse_unit("kΩ") == u.kohm
        assert parse_unit("kΩ") == u.kohm  # U+2126 OHM SIGN
        assert parse_unit("°") == u.degree
        assert parse_unit("Å") == u.angstrom

    def test_superscripts_and_operators(self):
        assert parse_unit("mS·nA/cm²") == u.mS * u.nA / u.cmetre ** 2
        assert parse_unit("mS·nA / cm²") == u.mS * u.nA / u.cmetre ** 2
        assert parse_unit("m⋅s⁻¹") == u.metre / u.second
        assert parse_unit("kg×m") == u.kilogram * u.metre

    def test_special_symbols_longest_match(self):
        assert parse_unit("km/h * s") == kmh * u.second
        assert parse_unit("km / h") == u.kmetre / u.hour
        assert parse_unit("(oz t)^2") == troy_ounce ** 2
        assert parse_unit("oz t^2") == troy_ounce ** 2

    def test_numbers(self):
        assert parse_unit("10^3 * m") == Unit(u.metre.dim, scale=3)
        assert parse_unit("10.0^-3") == Unit(scale=-3)  # legacy form
        assert parse_unit("10^1") == Unit(scale=1)
        assert parse_unit("10") == Unit(factor=10.)
        assert parse_unit("4.184 * 10^3 * m") == Unit(u.metre.dim, scale=3, factor=4.184)
        assert parse_unit("1") is UNITLESS

    @pytest.mark.parametrize("s", ["", "m *", "(m", "m)", "()", "m ^", "m^(1/0)", "nonexistent_xyz", "m $ s"])
    def test_errors(self, s):
        with pytest.raises(ValueError):
            parse_unit(s)


def _exported_units():
    seen = {}
    for mod in (_unit_common, _unit_constants):
        for name in mod.__all__:
            unit = getattr(mod, name)
            if isinstance(unit, Unit):
                seen[id(unit)] = (name, unit)
    return sorted(seen.values(), key=lambda t: t[0])


@pytest.mark.parametrize("style", ["ascii", "unicode"])
def test_every_exported_unit_round_trips(style):
    bad = []
    for name, unit in _exported_units():
        s = unit.to_string(style)
        parsed = parse_unit(s)
        if parsed != unit or str(parsed) != str(unit):
            bad.append((name, s, str(parsed)))
    assert not bad


SAMPLE = [
    u.metre, u.kilogram, u.gram, u.second, u.amp, u.kelvin, u.mole, u.candle,
    u.hertz, u.newton, u.pascal, u.joule, u.watt, u.coulomb, u.volt, u.farad, u.ohm, u.siemens,
    u.weber, u.tesla, u.henry, u.becquerel, u.gray, u.katal, u.lumen, u.lux,
    u.mvolt, u.kvolt, u.uamp, u.nA, u.mS, u.uF, u.kohm, u.ms, u.khertz, u.cmetre, u.kmetre, u.mgram,
    u.metre2, u.cmetre3, u.mvolt2, u.liter, u.molar,
    u.calorie, u.inch, u.hour, u.minute, u.eV, u.atm, kmh, troy_ounce,
    u.radian, u.steradian, u.degree, u.arcmin, u.percent if hasattr(u, 'percent') else u.radian,
    UNITLESS, Unit(u.volt.dim), Unit(scale=3), Unit(u.metre.dim, factor=2.5), Unit(u.second.dim, scale=-3),
]


def _check_round_trip(x):
    s = str(x)
    parsed = parse_unit(s)
    assert parsed == x, (s, str(parsed))
    assert str(parsed) == s
    assert parse_unit(x.to_string('unicode')) == x, x.to_string('unicode')
    assert derive_magnitude(x._terms, x._pre) == (x.scale, x.factor)
    assert derive_dim(x._terms) == x.dim


def test_pairwise_products_and_quotients():
    for a, b in itertools.product(SAMPLE, repeat=2):
        ab, ba = a * b, b * a
        assert ab == ba
        assert str(ab) == str(ba)
        _check_round_trip(ab)
        _check_round_trip(a / b)


@pytest.mark.parametrize("k", [-2, -1, 0.5, 1 / 3, 2])
def test_powers(k):
    for a in SAMPLE:
        _check_round_trip(a ** k)


def test_associativity():
    sample = SAMPLE[::4]
    for a, b, c in itertools.product(sample, repeat=3):
        left, right = (a * b) * c, a * (b * c)
        assert left == right
        assert str(left) == str(right)
        assert str(c * (b * a)) == str(left)


def test_parse_is_idempotent():
    for s in ["1 / mV", "mS * nA / cm^2", "kg * m / s^2", "A * das^3 / (kg * m^2)", "rad * m", "h / ms"]:
        once = str(parse_unit(s))
        assert str(parse_unit(once)) == once
