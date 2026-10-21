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

"""Tests for _unit_terms.py: atoms, exponent normalisation, term merging."""

import pickle
from fractions import Fraction

import numpy as np
import pytest

import saiunit as u
from saiunit._base_dimension import get_or_create_dimension
from saiunit._unit_terms import (
    PRE_ONE,
    combine_terms,
    derive_dim,
    derive_magnitude,
    make_atom,
    norm_exp,
)


def _atom(unit):
    return unit._single_atom()


class TestNormExp:
    @pytest.mark.parametrize("e, expected", [
        (2, 2), (2.0, 2), (np.int64(3), 3), (np.float64(-1.0), -1), (True, 1),
        (0.5, Fraction(1, 2)), (1 / 3, Fraction(1, 3)), (-0.25, Fraction(-1, 4)),
        (Fraction(4, 2), 2), (Fraction(2, 6), Fraction(1, 3)),
    ])
    def test_values(self, e, expected):
        got = norm_exp(e)
        assert got == expected
        assert type(got) is type(expected)

    def test_irrational_stays_float(self):
        e = 2 ** 0.5
        assert norm_exp(e) == e
        assert type(norm_exp(e)) is float

    @pytest.mark.parametrize("e", [float('inf'), float('-inf'), float('nan')])
    def test_non_finite_rejected(self, e):
        with pytest.raises(ValueError, match="finite"):
            norm_exp(e)

    def test_non_real_rejected(self):
        with pytest.raises(TypeError):
            norm_exp(1j)


class TestAtom:
    def test_interned_by_symbol_and_value(self):
        assert _atom(u.metre) is _atom(u.meter)
        assert _atom(u.kilogram) is _atom(u.kgram)
        assert _atom(u.micron) is _atom(u.umetre)

    def test_immutable(self):
        with pytest.raises(AttributeError):
            _atom(u.metre).symbol = 'x'

    def test_pickle_reinterns(self):
        a = _atom(u.mvolt)
        assert pickle.loads(pickle.dumps(a)) is a

    def test_default_unicode_and_latex(self):
        assert _atom(u.uvolt).unicode == 'µV'
        assert _atom(u.uvolt).latex == r'\mu\mathrm{V}'
        assert _atom(u.mvolt).latex == r'\mathrm{mV}'
        # amu's symbol 'u' is not a micro prefix
        assert _atom(u.amu).unicode == 'u'

    def test_explicit_symbols_propagate_to_prefixes(self):
        assert _atom(u.kohm).unicode == 'kΩ'
        assert _atom(u.uohm).unicode == 'µΩ'
        assert _atom(u.kohm).latex == r'\mathrm{k}\Omega'

    def test_latex_escaping(self):
        d = get_or_create_dimension(m=1)
        a = make_atom('a b%', 'test_escape', d, 0, 7.5)
        assert a.latex == r'\mathrm{a\ b\%}'


class TestCombineTerms:
    def test_merge_and_cancel(self):
        m, s = u.metre._terms, u.second._terms
        assert combine_terms(m, 1, m, -1) == ()
        assert combine_terms(m, 2, s, -1) == ((_atom(u.metre), 2), (_atom(u.second), -1))

    def test_total_order(self):
        # numerator before denominator, then case-insensitive symbol
        terms = combine_terms((u.second * u.amp)._terms, 1, u.mol._terms, -1)
        symbols = [(a.symbol, e) for a, e in terms]
        assert symbols == [('A', 1), ('s', 1), ('mol', -1)]

    def test_order_independent(self):
        a, b, c = u.mvolt._terms, u.nA._terms, u.cmetre._terms
        assert combine_terms(combine_terms(a, 1, b, 1), 1, c, -2) == \
            combine_terms(combine_terms(c, -2, b, 1), 1, a, 1)

    def test_fraction_exponents(self):
        terms = combine_terms(u.metre._terms, Fraction(1, 3))
        terms = combine_terms(terms, 1, terms, 2)
        assert terms == ((_atom(u.metre), 1),)


class TestDerive:
    def test_magnitude(self):
        terms = combine_terms(u.calorie._terms, 2, u.kmetre._terms, -1)
        scale, factor = derive_magnitude(terms, PRE_ONE)
        assert scale == u.calorie.scale * 2 - 3
        assert factor == u.calorie.factor ** 2

    def test_permutations_are_bit_identical(self):
        units = [u.calorie, u.inch, u.hour, u.mile]
        a = units[0] * units[1] * units[2] / units[3]
        b = units[2] / units[3] * units[1] * units[0]
        assert a.factor == b.factor
        assert a == b

    def test_dim_exact_for_rational_exponents(self):
        terms = combine_terms(u.joule._terms, Fraction(1, 3))
        assert derive_dim(combine_terms(terms, 3)) == u.joule.dim
