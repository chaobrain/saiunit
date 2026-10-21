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

"""Render structured unit terms as ASCII, Unicode or LaTeX strings.

Styles
------
``ascii``           canonical, parseable: ``mS * nA / cm^2``
``ascii_inline``    signed exponents: ``mS * nA * cm^-2``
``unicode``         ``mS·nA / cm²``
``unicode_inline``  ``mS·nA·cm⁻²``
``latex``           ``\\frac{\\mathrm{mS}\\,\\mathrm{nA}}{\\mathrm{cm}^{2}}``
``latex_inline``    ``\\mathrm{mS}\\,\\mathrm{nA}\\,\\mathrm{cm}^{-2}``
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from ._unit_terms import Exp, Pre, Terms

__all__ = ['STYLES', 'render', 'format_exp']

STYLES = ('ascii', 'ascii_inline', 'unicode', 'unicode_inline', 'latex', 'latex_inline')

_SUPERSCRIPT = str.maketrans('0123456789-+', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺')
_WRAP_CHARS = frozenset(' /*^()')


def format_exp(e: Exp) -> str:
    """Format an exponent: ``2``, ``0.5`` or ``(1/3)``."""
    if type(e) is int:
        return str(e)
    if isinstance(e, Fraction):
        if 10 ** 6 % e.denominator == 0:
            return repr(float(e))
        return f"({e.numerator}/{e.denominator})"
    f = float(e)
    return str(int(f)) if f.is_integer() else repr(f)


def _format_number(x: Any) -> str:
    if isinstance(x, float):
        f = float(x)
        return str(int(f)) if f.is_integer() and abs(f) < 1e16 else repr(f)
    return str(x)


def _ascii_symbol(symbol: str) -> str:
    return f"({symbol})" if not _WRAP_CHARS.isdisjoint(symbol) else symbol


def _ascii_item(symbol: str, e: Exp, wrap: bool) -> str:
    if e == 1:
        return _ascii_symbol(symbol) if wrap else symbol
    return f"{_ascii_symbol(symbol)}^{format_exp(e)}"


def _unicode_item(symbol: str, e: Exp, wrap: bool) -> str:
    if e == 1:
        return _ascii_symbol(symbol) if wrap else symbol
    if type(e) is int:
        return _ascii_symbol(symbol) + str(e).translate(_SUPERSCRIPT)
    return f"{_ascii_symbol(symbol)}^{format_exp(e)}"


def _latex_exp(e: Exp) -> str:
    if isinstance(e, Fraction) and 10 ** 6 % e.denominator != 0:
        return f"{e.numerator}/{e.denominator}"
    return format_exp(e)


def _latex_item(latex: str, e: Exp) -> str:
    if e == 1:
        return latex
    base = '{' + latex + '}' if '^' in latex else latex
    return f"{base}^{{{_latex_exp(e)}}}"


def _pre_parts(pre: Pre, style: str) -> list[str]:
    scale, factor = pre
    parts = []
    if factor != 1:
        parts.append(_format_number(factor))
    if scale != 0:
        if style.startswith('unicode') and (type(scale) is int or float(scale).is_integer()):
            parts.append('10' + str(int(scale)).translate(_SUPERSCRIPT))
        elif style.startswith('latex'):
            parts.append(f"10^{{{_format_number(scale)}}}")
        else:
            parts.append(f"10^{_format_number(scale)}")
    if style.startswith('latex') and len(parts) == 2:
        return [f"{parts[0]} \\times {parts[1]}"]
    return parts


def render(terms: Terms, pre: Pre, style: str = 'ascii') -> str:
    """Render ``pre * prod(atom ** e)`` in the given *style*."""
    pre_parts = _pre_parts(pre, style)
    if style == 'ascii':
        wrap = len(terms) + len(pre_parts) > 1
        num = pre_parts + [_ascii_item(a.symbol, e, wrap) for a, e in terms if e > 0]
        den = [_ascii_item(a.symbol, -e, wrap) for a, e in terms if e < 0]
        num_str = ' * '.join(num) if num else '1'
        if not den:
            return num_str
        den_str = den[0] if len(den) == 1 else f"({' * '.join(den)})"
        return f"{num_str} / {den_str}"
    if style == 'ascii_inline':
        items = pre_parts + [_ascii_item(a.symbol, e, True) for a, e in terms]
        return ' * '.join(items) if items else '1'
    if style == 'unicode':
        # `` / `` keeps a quotient such as ``km / h`` distinct from the
        # registered symbol ``km/h``.
        wrap = len(terms) + len(pre_parts) > 1
        num = pre_parts + [_unicode_item(a.unicode, e, wrap) for a, e in terms if e > 0]
        den = [_unicode_item(a.unicode, -e, wrap) for a, e in terms if e < 0]
        num_str = '·'.join(num) if num else '1'
        if not den:
            return num_str
        den_str = den[0] if len(den) == 1 else f"({'·'.join(den)})"
        return f"{num_str} / {den_str}"
    if style == 'unicode_inline':
        items = pre_parts + [_unicode_item(a.unicode, e, True) for a, e in terms]
        return '·'.join(items) if items else '1'
    if style == 'latex':
        num = pre_parts + [_latex_item(a.latex, e) for a, e in terms if e > 0]
        den = [_latex_item(a.latex, -e) for a, e in terms if e < 0]
        num_str = r'\,'.join(num) if num else '1'
        if not den:
            return num_str
        return r'\frac{' + num_str + '}{' + r'\,'.join(den) + '}'
    if style == 'latex_inline':
        items = pre_parts + [_latex_item(a.latex, e) for a, e in terms]
        return r'\,'.join(items) if items else '1'
    raise ValueError(f"Unknown unit format {style!r}; expected one of {STYLES}.")
