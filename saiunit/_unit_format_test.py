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

"""Tests for _unit_format.py and the Unit/Quantity string APIs built on it."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import pytest

import saiunit as u
from saiunit._unit_format import format_exp

STYLES = ('ascii', 'ascii_inline', 'unicode', 'unicode_inline', 'latex', 'latex_inline')

GOLDEN = [
    (lambda: u.mS * u.nA / u.cmetre ** 2,
     ['mS * nA / cm^2', 'mS * nA * cm^-2', 'mS·nA / cm²', 'mS·nA·cm⁻²',
      r'\frac{\mathrm{mS}\,\mathrm{nA}}{\mathrm{cm}^{2}}', r'\mathrm{mS}\,\mathrm{nA}\,\mathrm{cm}^{-2}']),
    (lambda: u.uF / u.cmetre2,
     ['uF / cm^2', 'uF * cm^-2', 'µF / cm²', 'µF·cm⁻²',
      r'\frac{\mu\mathrm{F}}{\mathrm{cm}^{2}}', r'\mu\mathrm{F}\,\mathrm{cm}^{-2}']),
    (lambda: u.mvolt.reverse(),
     ['1 / mV', 'mV^-1', '1 / mV', 'mV⁻¹', r'\frac{1}{\mathrm{mV}}', r'\mathrm{mV}^{-1}']),
    (lambda: u.Unit("ms * mV").reverse(),
     ['1 / (ms * mV)', 'ms^-1 * mV^-1', '1 / (ms·mV)', 'ms⁻¹·mV⁻¹',
      r'\frac{1}{\mathrm{ms}\,\mathrm{mV}}', r'\mathrm{ms}^{-1}\,\mathrm{mV}^{-1}']),
    (lambda: u.kohm,
     ['kohm', 'kohm', 'kΩ', 'kΩ', r'\mathrm{k}\Omega', r'\mathrm{k}\Omega']),
    (lambda: u.metre ** (1 / 3),
     ['m^(1/3)', 'm^(1/3)', 'm^(1/3)', 'm^(1/3)', r'\mathrm{m}^{1/3}', r'\mathrm{m}^{1/3}']),
    (lambda: u.radian / u.second,
     ['rad / s', 'rad * s^-1', 'rad / s', 'rad·s⁻¹',
      r'\frac{\mathrm{rad}}{\mathrm{s}}', r'\mathrm{rad}\,\mathrm{s}^{-1}']),
    (lambda: u.degree ** 2,
     ['deg^2', 'deg^2', '°²', '°²', r'{^{\circ}}^{2}', r'{^{\circ}}^{2}']),
    (lambda: u.Unit(u.metre.dim, scale=3, factor=4.184),
     ['4.184 * 10^3 * m', '4.184 * 10^3 * m', '4.184·10³·m', '4.184·10³·m',
      r'4.184 \times 10^{3}\,\mathrm{m}', r'4.184 \times 10^{3}\,\mathrm{m}']),
    (lambda: u.Unit(scale=-3),
     ['10^-3', '10^-3', '10⁻³', '10⁻³', '10^{-3}', '10^{-3}']),
    (lambda: u.UNITLESS, ['1'] * 6),
    (lambda: u.angstrom,
     ['Angstrom', 'Angstrom', 'Å', 'Å', r'\mathring{A}', r'\mathring{A}']),
]


@pytest.mark.parametrize("make, expected", GOLDEN)
def test_golden_strings(make, expected):
    unit = make()
    assert [unit.to_string(style) for style in STYLES] == expected


@pytest.mark.parametrize("e, expected", [
    (2, '2'), (-3, '-3'), (Fraction(1, 2), '0.5'), (Fraction(5, 4), '1.25'),
    (Fraction(1, 3), '(1/3)'), (Fraction(-2, 3), '(-2/3)'), (2 ** 0.5, repr(2 ** 0.5)),
])
def test_format_exp(e, expected):
    assert format_exp(e) == expected


def test_unknown_style_rejected():
    with pytest.raises(ValueError, match="Unknown unit format"):
        u.mV.to_string('html')


def test_unit_format_protocol():
    unit = u.mS / u.cmetre2
    assert f"{unit}" == "mS / cm^2"
    assert f"{unit:unicode}" == "mS / cm²"
    assert f"{unit:latex}" == unit.to_string('latex')


def test_unit_repr_latex():
    assert u.uF._repr_latex_() == r'$\mu\mathrm{F}$'


def test_special_symbols_are_wrapped_in_composites():
    from saiunit._unit_constants import kmh, troy_ounce
    assert str(kmh) == "km/h"
    assert str(kmh * u.second) == "(km/h) * s"
    assert str(troy_ounce ** 2) == "(oz t)^2"


class TestQuantityDisplay:
    def test_numerator_less_unit_uses_negative_exponents(self):
        q = 1 / (2.5 * u.mV)
        assert str(q) == "0.4 mV^-1"
        assert repr(q) == 'Quantity(0.4, "1 / mV")'
        assert f"{q:.1f}" == "0.4 mV^-1"

    def test_numeric_unit_joined_with_star(self):
        q = u.Quantity(2.5, unit=u.Unit(scale=3))
        assert str(q) == "2.5 * 10^3"
        assert f"{q:.1f}" == "2.5 * 10^3"

    def test_named_dimensionless_shown(self):
        assert str(2. * u.radian) == "2. rad"
        assert str(u.Quantity(2., unit=u.UNITLESS)) == "2."

    def test_repr_latex(self):
        assert (2.5 * u.mV)._repr_latex_() == r'$2.5\;\mathrm{mV}$'
        assert (jnp.array([1., 2.]) * u.uF)._repr_latex_() == r'$\text{[1. 2.]}\;\mu\mathrm{F}$'

    def test_repr_latex_tracer_is_none(self):
        out = []
        jax.jit(lambda x: out.append((x * u.mV)._repr_latex_()) or x)(1.)
        assert out == [None]

    def test_simplify(self):
        q = 3. * u.volt * u.second
        assert str(q) == "3. s * V"
        assert str(q.simplify()) == "3. Wb"
        assert q.simplify().unit == q.unit
