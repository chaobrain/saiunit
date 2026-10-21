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

"""Tokenizer and recursive-descent parser for unit strings.

Grammar (``/`` binds looser than ``*``, so ``a / b * c`` is ``a / (b * c)``)::

    expr     := product ('/' product)*
    product  := power (('*' | '·' | '⋅' | '×') power)*
    power    := primary ('^' exponent | SUPERSCRIPTS)?
    primary  := NUMBER | '-' NUMBER | NAME | SYMBOL | '(' expr ')'
    exponent := ['+' | '-'] NUMBER | '(' ['+' | '-'] NUMBER ['/' NUMBER] ')'

``NAME`` is a registered unit name or symbol (Unicode letters allowed);
``SYMBOL`` is a registered symbol that is not an identifier (``km/h``,
``oz t``, ``°``), matched longest-first.  ``10^n`` is a pure scale.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import TYPE_CHECKING, Callable, Iterable

from ._unit_terms import Exp, norm_exp

if TYPE_CHECKING:
    from ._base_unit import Unit

__all__ = ['parse']

_SUPERSCRIPT_DIGITS = '⁰¹²³⁴⁵⁶⁷⁸⁹'
_FROM_SUPERSCRIPT = str.maketrans(_SUPERSCRIPT_DIGITS + '⁻⁺', '0123456789-+')
_TOKEN_RE = re.compile(
    r'(?P<ws>\s+)'
    r'|(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'
    r'|(?P<sup>[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)'
    rf'|(?P<name>[^\W\d{_SUPERSCRIPT_DIGITS}][^\W{_SUPERSCRIPT_DIGITS}]*)'
    r'|(?P<op>[*·⋅×/^()+-])'
)
_NAME_RE = re.compile(rf'[^\W\d{_SUPERSCRIPT_DIGITS}][^\W{_SUPERSCRIPT_DIGITS}]*')
_MUL_OPS = frozenset('*·⋅×')
_OPERATOR_CHARS = frozenset('^*()')


def is_special_symbol(key: str) -> bool:
    """Whether *key* must be matched as a whole token (``km/h``, ``oz t``, ``°``)."""
    return (
        _NAME_RE.fullmatch(key) is None
        and _OPERATOR_CHARS.isdisjoint(key)
        and ' / ' not in key
        and not key[:1].isdigit()
    )


def _tokenize(s: str, specials: Iterable[str]) -> list[tuple[str, str]]:
    tokens = []
    pos = 0
    n = len(s)
    while pos < n:
        for sym in specials:
            end = pos + len(sym)
            if s.startswith(sym, pos) and (end == n or _NAME_RE.match(s[end]) is None):
                tokens.append(('name', sym))
                pos = end
                break
        else:
            m = _TOKEN_RE.match(s, pos)
            if m is None:
                raise ValueError(f"Unexpected character {s[pos]!r} in unit string {s!r}.")
            kind = m.lastgroup
            assert kind is not None
            if kind != 'ws':
                tokens.append((kind, m.group()))
            pos = m.end()
    return tokens


class _Parser:
    def __init__(self, s: str, tokens: list[tuple[str, str]], lookup: Callable[[str], 'Unit | None']):
        self.s = s
        self.tokens = tokens
        self.i = 0
        self.lookup = lookup

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise ValueError(f"Unexpected end of unit string {self.s!r}.")
        self.i += 1
        return tok

    def expect(self, value: str) -> None:
        tok = self.take()
        if tok[1] != value:
            raise ValueError(f"Expected {value!r} but found {tok[1]!r} in unit string {self.s!r}.")

    def at_op(self, values) -> bool:
        tok = self.peek()
        return tok is not None and tok[0] == 'op' and tok[1] in values

    def parse(self) -> 'Unit':
        result = self.expr()
        tok = self.peek()
        if tok is not None:
            raise ValueError(f"Unexpected token {tok[1]!r} in unit string {self.s!r}.")
        return result

    def expr(self) -> 'Unit':
        from ._base_unit import _compose

        result = self.product()
        while self.at_op('/'):
            self.take()
            result = _compose(result, 1, self.product(), -1)
        return result

    def product(self) -> 'Unit':
        from ._base_unit import _compose

        result = self.power()
        while self.at_op(_MUL_OPS):
            self.take()
            result = _compose(result, 1, self.power(), 1)
        return result

    def power(self) -> 'Unit':
        from ._base_dimension import DIMENSIONLESS
        from ._base_unit import Unit

        value = self.primary()
        exp: Exp = 1
        has_exp = True
        tok = self.peek()
        if tok is not None and tok[0] == 'sup':
            self.take()
            exp = self._number_exp(tok[1].translate(_FROM_SUPERSCRIPT))
        elif self.at_op('^'):
            self.take()
            exp = self.exponent()
        else:
            has_exp = False
        if not isinstance(value, str):
            return value if exp == 1 else value ** exp
        number = float(value)
        if number == 10. and has_exp:
            scale = exp if type(exp) is int else float(exp)
            return Unit(DIMENSIONLESS, scale=scale)
        try:
            factor = number if exp == 1 else number ** (exp if type(exp) is int else float(exp))
        except OverflowError:
            raise ValueError(
                f"unit factor {number} ** {exp} overflows the float range; reduce the exponent."
            ) from None
        return Unit(DIMENSIONLESS, factor=factor)

    def primary(self) -> 'Unit | str':
        """Return a Unit, or the text of a numeric literal."""
        kind, value = self.take()
        if kind == 'num':
            return value
        if kind == 'op' and value in '+-':
            sign = value
            kind, value = self.take()
            if kind != 'num':
                raise ValueError(f"Expected a number after {sign!r} in unit string {self.s!r}.")
            return sign + value
        if kind == 'name':
            unit = self.lookup(value)
            if unit is None:
                raise ValueError(
                    f"Unknown unit: {value!r} in {self.s!r}. "
                    f"Use a registered unit name or display name."
                )
            return unit
        if kind == 'op' and value == '(':
            if self.at_op(')'):
                raise ValueError(f"Empty parenthesised group in {self.s!r}.")
            inner = self.expr()
            self.expect(')')
            return inner
        raise ValueError(f"Unexpected token {value!r} in unit string {self.s!r}.")

    @staticmethod
    def _number_exp(text: str) -> Exp:
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return norm_exp(float(text))
        except ValueError:
            raise ValueError(f"Invalid exponent {text!r} in unit string.") from None

    def _signed_number(self) -> str:
        sign = ''
        if self.at_op('+-'):
            sign = self.take()[1]
        kind, value = self.take()
        if kind != 'num':
            raise ValueError(f"Invalid exponent {value!r} in unit string {self.s!r}.")
        return sign + value

    def exponent(self) -> Exp:
        if not self.at_op('('):
            return self._number_exp(self._signed_number())
        self.take()
        num = self._signed_number()
        if self.at_op('/'):
            self.take()
            den = self._signed_number()
            self.expect(')')
            try:
                return norm_exp(Fraction(int(num), int(den)))
            except (ValueError, ZeroDivisionError):
                raise ValueError(f"Invalid exponent ({num}/{den}) in unit string {self.s!r}.") from None
        self.expect(')')
        return self._number_exp(num)


def parse(s: str, lookup: Callable[[str], 'Unit | None'], specials: Iterable[str]) -> 'Unit':
    """Parse the unit expression *s*.

    *lookup* maps a name or symbol to a registered :class:`Unit` (or None);
    *specials* are the non-identifier symbols, longest first.
    """
    s = s.replace('Ω', 'Ω')  # OHM SIGN -> GREEK CAPITAL OMEGA

    def _lookup(name: str) -> 'Unit | None':
        unit = lookup(name)
        if unit is None and name[:1] in ('µ', 'μ'):
            unit = lookup('u' + name[1:])
        return unit

    return _Parser(s, _tokenize(s, specials), _lookup).parse()
