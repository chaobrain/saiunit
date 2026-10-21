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

"""Structured unit terms.

A composed unit is displayed from a canonical tuple of ``(Atom, exponent)``
pairs plus a numeric prefactor ``(scale, factor)``.  An :class:`Atom` is an
interned, immutable named unit symbol (``m``, ``mV``, ``Hz``, ...).
Exponents are ``int`` when integral, :class:`fractions.Fraction` when
rational and ``float`` only as a last resort, so ``m ** (1/3)`` stays exact.
"""

from __future__ import annotations

import math
import numbers
import threading
from fractions import Fraction
from typing import Any, Union

from ._base_dimension import Dimension, get_or_create_dimension

__all__ = [
    'Atom',
    'Exp',
    'Terms',
    'Pre',
    'PRE_ONE',
    'make_atom',
    'norm_exp',
    'combine_terms',
    'derive_magnitude',
    'derive_dim',
    'norm_scale',
    'latex_mathrm',
]

Exp = Union[int, Fraction, float]
Pre = tuple  # (scale, factor)
PRE_ONE: Pre = (0, 1.)


class Atom:
    """An interned named unit symbol, e.g. ``mV`` or ``Hz``.

    Identity is ``(symbol, dim, scale, factor)``: spelling variants with the
    same symbol (``metre``/``meter``) share one atom.  Use :func:`make_atom`
    to construct; atoms are never created directly.
    """

    __slots__ = ('symbol', 'name', 'dim', 'scale', 'factor', 'unicode', 'latex', 'prefix',
                 'sort_key', '_hash')

    symbol: str
    name: str
    dim: Dimension
    scale: Any
    factor: Any
    unicode: str
    latex: str
    prefix: str | None
    sort_key: tuple
    _hash: int

    def __setattr__(self, key, value):
        raise AttributeError("Atom is immutable")

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"Atom({self.symbol!r})"

    def __reduce__(self):
        return (make_atom, (self.symbol, self.name, self.dim, self.scale, self.factor,
                            self.unicode, self.latex, self.prefix))


Terms = tuple  # tuple[tuple[Atom, Exp], ...]

_atoms: dict[tuple, Atom] = {}
_atoms_lock = threading.Lock()


def latex_mathrm(symbol: str) -> str:
    """Wrap *symbol* in ``\\mathrm{}``, escaping LaTeX special characters."""
    s = symbol.replace('\\', r'\backslash ').replace('%', r'\%').replace('_', r'\_')
    s = s.replace('#', r'\#').replace('&', r'\&').replace('$', r'\$').replace(' ', r'\ ')
    return r'\mathrm{' + s + '}'


def make_atom(
    symbol: str,
    name: str,
    dim: Dimension,
    scale: Any,
    factor: Any,
    unicode: str | None = None,
    latex: str | None = None,
    prefix: str | None = None,
) -> Atom:
    """Return the interned atom for ``(symbol, dim, scale, factor)``.

    The first registration wins: later calls with the same identity return
    the existing atom, whatever ``name``/``unicode``/``latex`` they pass.
    """
    key = (symbol, dim, scale, factor)
    atom = _atoms.get(key)
    if atom is not None:
        return atom
    with _atoms_lock:
        atom = _atoms.get(key)
        if atom is not None:
            return atom
        if unicode is None:
            unicode = ('µ' + symbol[1:]) if prefix == 'u' else symbol
        if latex is None:
            latex = (r'\mu' + latex_mathrm(symbol[1:])) if prefix == 'u' else latex_mathrm(symbol)
        atom = object.__new__(Atom)
        setter = object.__setattr__
        setter(atom, 'symbol', symbol)
        setter(atom, 'name', name)
        setter(atom, 'dim', dim)
        setter(atom, 'scale', scale)
        setter(atom, 'factor', factor)
        setter(atom, 'unicode', unicode)
        setter(atom, 'latex', latex)
        setter(atom, 'prefix', prefix)
        setter(atom, 'sort_key', (symbol.casefold(), symbol, scale, factor, tuple(dim._dims.tolist())))
        setter(atom, '_hash', hash(key))
        _atoms[key] = atom
        return atom


def norm_exp(e: Any) -> Exp:
    """Normalise an exponent to ``int``, ``Fraction`` or (last resort) ``float``.

    Floats that are exactly a fraction with denominator <= 10**6 (``0.5``,
    ``1/3``) become :class:`Fraction`.  Non-finite exponents are rejected.
    """
    if type(e) is int:
        return e
    if isinstance(e, Fraction):
        return e.numerator if e.denominator == 1 else e
    if isinstance(e, numbers.Integral):
        return int(e)
    if isinstance(e, numbers.Rational):
        return norm_exp(Fraction(int(e.numerator), int(e.denominator)))
    if isinstance(e, numbers.Real):
        f = float(e)
        if not math.isfinite(f):
            raise ValueError(f"Unit exponent must be finite; got {e!r}.")
        if f.is_integer():
            return int(f)
        fr = Fraction(f).limit_denominator(10 ** 6)
        if float(fr) == f:
            return fr
        return f
    if getattr(e, 'ndim', None) == 0:
        return norm_exp(e.item())
    raise TypeError(f"Unit exponent must be a real number; got {e!r}.")


def _mul_exp(a: Exp, b: Exp) -> Exp:
    if type(a) is int and type(b) is int:
        return a * b
    return norm_exp(a * b)


def _add_exp(a: Exp, b: Exp) -> Exp:
    if type(a) is int and type(b) is int:
        return a + b
    return norm_exp(a + b)


def _term_key(term):
    return (term[1] < 0, term[0].sort_key)


def combine_terms(ta: Terms, ea: Exp, tb: Terms = (), eb: Exp = 0) -> Terms:
    """Return the canonical terms of ``ta ** ea * tb ** eb``.

    Exponents of the same atom are summed, zero exponents are dropped and the
    result is sorted by the total order ``(exponent < 0, atom.sort_key)``.
    """
    acc: dict[Atom, Exp] = {}
    for atom, e in ta:
        acc[atom] = e if ea == 1 else _mul_exp(e, ea)
    if eb != 0:
        for atom, e in tb:
            e2 = e if eb == 1 else (-e if eb == -1 else _mul_exp(e, eb))
            prev = acc.get(atom)
            acc[atom] = e2 if prev is None else _add_exp(prev, e2)
    terms = [(a, e) for a, e in acc.items() if e != 0]
    terms.sort(key=_term_key)
    return tuple(terms)


def norm_scale(scale: Any) -> Any:
    if isinstance(scale, Fraction):
        return scale.numerator if scale.denominator == 1 else float(scale)
    return scale


def _factor_pow(factor: Any, e: Exp) -> Any:
    try:
        return factor ** (e if type(e) is int else float(e))
    except OverflowError:
        raise ValueError(
            f"unit factor {factor} ** {e} overflows the float range; strip the "
            f"factor first (factorless()) or reduce the exponent."
        ) from None


def derive_magnitude(terms: Terms, pre: Pre) -> tuple:
    """Return ``(scale, factor)`` of ``pre * prod(atom ** e)``.

    Atoms are folded in canonical order, so every permutation of the same
    product yields bit-identical results.
    """
    scale, factor = pre
    for atom, e in terms:
        if atom.scale != 0:
            scale = scale + atom.scale * e
        if atom.factor != 1:
            factor = factor * _factor_pow(atom.factor, e)
    return norm_scale(scale), factor


def derive_dim(terms: Terms) -> Dimension:
    """Return the dimension of ``prod(atom ** e)``, exact for rational exponents."""
    acc = [Fraction(0)] * 7
    for atom, e in terms:
        dims = atom.dim._dims.tolist()
        for i in range(7):
            if dims[i]:
                acc[i] += Fraction(dims[i]) * Fraction(e)
    out = [int(v) if v.denominator == 1 else float(v) for v in acc]
    return get_or_create_dimension(out)
