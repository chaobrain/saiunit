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

from __future__ import annotations

import math
import numbers
import threading
from typing import TYPE_CHECKING, Any

from . import _unit_parse
from ._base_dimension import (
    Dimension,
    DIMENSIONLESS,
    _is_tracer,
    get_or_create_dimension,
)
from ._typing import ArrayLike
from ._unit_format import STYLES, render
from ._unit_terms import (
    PRE_ONE,
    Atom,
    Exp,
    Pre,
    Terms,
    combine_terms,
    derive_dim,
    derive_magnitude,
    latex_mathrm,
    make_atom,
    norm_exp,
    norm_scale,
)

if TYPE_CHECKING:
    from ._base_quantity import Quantity

__all__ = [
    'Unit',
    'UNITLESS',
    'add_standard_unit',
    'parse_unit',
]

# SI unit _prefixes as integer exponents of 10, see table at end of file.
_siprefixes = {
    "y": -24,
    "z": -21,
    "a": -18,
    "f": -15,
    "p": -12,
    "n": -9,
    "u": -6,
    "m": -3,
    "c": -2,
    "d": -1,
    "": 0,
    "da": 1,
    "h": 2,
    "k": 3,
    "M": 6,
    "G": 9,
    "T": 12,
    "P": 15,
    "E": 18,
    "Z": 21,
    "Y": 24,
}


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

def _assert_same_base(u1, u2):
    if not u1.has_same_base(u2):
        raise TypeError(f"Cannot operate on units with different bases. Got {u1.base} != {u2.base}.")


_standard_units: 'dict[tuple, Unit]' = {}
_standard_unit_aliases: 'dict[tuple, list[Unit]]' = {}
_unit_name_registry: 'dict[str, Unit]' = {}
# Alternative spellings accepted by the parser but never rendered by
# ``str()``: Unicode symbols (``µm``, ``Ω``, ``°``) of registered units.
_symbol_aliases: 'dict[str, Unit]' = {}
# Curated targets for automatic simplification during unit arithmetic,
# keyed like ``_standard_units``.  Only units registered with
# ``auto_simplify=True`` (SI base units, N, Pa, J, W, C, V, F, ohm, S, Hz
# and their prefixed forms) appear here, so ``mV * ms`` stays ``mV * ms``
# rather than collapsing onto ``uWb``.
_auto_targets: 'dict[tuple, list[Atom]]' = {}
_auto_atoms: 'set[Atom]' = set()
# Monotonically-increasing registration index for each registered Unit
# identity.  Used by :func:`_select_preferred_standard_unit` to prefer
# units that were registered earlier (i.e. library built-ins) over
# user-added aliases for the same physical key.
# Entries are never evicted, but this is safe: the key is ``id(u)``, and
# every registered ``u`` stays alive for the process lifetime via the
# strong reference held in ``_standard_unit_aliases``, so an id can never
# be recycled and silently collide with a later, unrelated Unit.
_unit_registration_index: 'dict[int, int]' = {}
_next_registration_index: 'list[int]' = [0]
# Guards all registry mutation in ``add_standard_unit`` so concurrent
# registration (e.g. user threads importing unit-defining modules) cannot
# interleave alias-list updates with preferred-unit selection.
_registry_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Ambiguous-key detection
#
# A dimension key is "ambiguous" when >=2 registered aliases have
# **different display names** (dispname).  Spelling variants like
# meter/metre share the same dispname ("m") so they are NOT flagged.
# Genuine semantic collisions like hertz/becquerel ("Hz" vs "Bq") ARE
# flagged automatically—no hardcoded list required.
#
# Ambiguous keys are never auto-substituted during unit composition, so
# that e.g. joule/kg never silently becomes sievert.
# ---------------------------------------------------------------------------
_ambiguous_keys: set = set()


def _standard_unit_preference_score(unit: 'Unit') -> int:
    """
    Return a preference score for choosing canonical display aliases.

    Lower is better.  Deterministic: on ties the name that sorts first
    alphabetically wins (via ``_select_preferred_standard_unit``).
    """
    name = unit.name.lower() if isinstance(unit.name, str) else ""
    score = 0
    # Prefer frequency over radioactivity for s^-1
    if "hertz" in name:
        score -= 10
    return score


def _select_preferred_standard_unit(units: 'list[Unit]') -> 'Unit':
    """Pick the preferred alias.

    Order of preference (lower is better):

    1. ``_standard_unit_preference_score`` (e.g. prefer hertz over
       becquerel for s^-1).
    2. Registration index — library built-ins win over user
       additions made via :func:`add_standard_unit` after import,
       so user aliases cannot hijack canonical display.
    3. Alphabetical, as a final deterministic tie-breaker for units
       registered in the same call.
    """
    def _key(u):
        idx = _unit_registration_index.get(id(u), float('inf'))
        return (
            _standard_unit_preference_score(u),
            idx,
            u.name.lower() if isinstance(u.name, str) else "",
        )
    return min(units, key=_key)


def add_standard_unit(u: 'Unit', *, auto_simplify: bool = False):
    """
    Register a unit as a standard unit.

    Registered units are recognised by :func:`parse_unit` (by name and
    display name) and are the targets of :meth:`Unit.simplify`.  If
    multiple units are registered for the same key, the preferred alias is
    selected automatically.  Keys with two or more distinct display names
    are flagged as *ambiguous*.

    Parameters
    ----------
    u : Unit
        The unit to register.  Its ``base``, ``scale``, and ``factor``
        must all be plain Python ``int`` or ``float`` values (not JAX
        tracers) for registration to take effect.
    auto_simplify : bool, optional
        If True, unit arithmetic whose result has exactly this unit's
        dimension and magnitude is displayed as this unit (e.g.
        ``kg * m / s^2`` displays as ``N``).  Only meaningful for a named
        (non-composite) unit.  Defaults to False.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> my_unit = u.Unit(
        ...     dim=u.joule.dim,
        ...     name='my_energy',
        ...     dispname='myE',
        ...     is_fullname=True,
        ... )
        >>> u.add_standard_unit(my_unit)
    """
    if (
        isinstance(u.base, (int, float)) and
        isinstance(u.scale, (int, float)) and
        isinstance(u.factor, (int, float))
    ):
        with _registry_lock:
            key = (u.dim, u.scale, u.base, u.factor)
            aliases = _standard_unit_aliases.setdefault(key, [])
            # Dedup by identity *and* by (name, dispname): re-running the
            # same ``Unit.create(...)`` call produces a new instance each
            # time; without value-based dedup, repeated registration grows
            # the alias list unboundedly and poisons the ambiguity
            # heuristic below.
            if not any(
                existing is u
                or (existing.name == u.name and existing.dispname == u.dispname)
                for existing in aliases
            ):
                aliases.append(u)
                # Stamp a monotonic registration index so that later
                # additions cannot hijack the canonical display.
                _unit_registration_index[id(u)] = _next_registration_index[0]
                _next_registration_index[0] += 1
            _standard_units[key] = _select_preferred_standard_unit(aliases)

            # Auto-detect ambiguity: >=2 distinct display names → ambiguous
            dispnames = {a.dispname for a in aliases if isinstance(a.dispname, str)}
            if len(dispnames) >= 2:
                _ambiguous_keys.add(key)

            # Register by dispname and name for string-based lookup
            if isinstance(u.dispname, str) and u.dispname:
                _unit_name_registry.setdefault(u.dispname, u)
            if isinstance(u.name, str) and u.name:
                _unit_name_registry.setdefault(u.name, u)

            atom = u._single_atom()
            if atom is not None:
                if atom.unicode != atom.symbol:
                    _symbol_aliases.setdefault(atom.unicode, u)
                if auto_simplify:
                    targets = _auto_targets.setdefault(key, [])
                    if atom not in targets:
                        targets.append(atom)
                    _auto_atoms.add(atom)


def _lookup_symbol(s: str) -> 'Unit | None':
    """Return the registered unit spelled *s* (name, symbol or alias)."""
    unit = _unit_name_registry.get(s)
    return unit if unit is not None else _symbol_aliases.get(s)


_special_symbol_cache: 'list' = [None, ()]


def _special_symbols() -> 'tuple[str, ...]':
    """Registered symbols that are not identifiers (``km/h``, ``oz t``, ``°``), longest first."""
    stamp = (len(_unit_name_registry), len(_symbol_aliases))
    if _special_symbol_cache[0] != stamp:
        keys = set(_unit_name_registry) | set(_symbol_aliases)
        specials = sorted((k for k in keys if _unit_parse.is_special_symbol(k)), key=lambda k: (-len(k), k))
        _special_symbol_cache[:] = [stamp, tuple(specials)]
    return _special_symbol_cache[1]


def parse_unit(s: str) -> 'Unit':
    """Parse a unit string into a :class:`Unit`.

    Accepts the strings produced by ``str(unit)``, ``repr(unit)`` and
    ``unit.to_string('unicode')``, e.g. ``"mV"``, ``"J / kg"``,
    ``"nA / cm^2"``, ``"mS·nA/cm²"``, ``"m^(1/3)"``, ``"10^3 * m"``.

    Both display names (``"mV"``) and full names (``"mvolt"``) are
    recognised, as are the Unicode symbols ``µ``/``μ``, ``Ω`` and ``°``.
    ``/`` binds looser than ``*``: ``"a / b * c"`` is ``a / (b * c)``.

    Parameters
    ----------
    s : str
        The unit string to parse.

    Returns
    -------
    Unit

    Raises
    ------
    ValueError
        If the string cannot be parsed into a known unit.

    Examples
    --------
    >>> parse_unit("mV")
    Unit("mV")
    >>> parse_unit("J / kg")
    Unit("J / kg")
    """
    s = s.strip()

    # Strip the Unit("...") repr wrapper if present
    if s.startswith('Unit("') and s.endswith('")'):
        s = s[6:-2]
    elif s.startswith("Unit('") and s.endswith("')"):
        s = s[6:-2]

    if not s:
        raise ValueError("Cannot parse an empty unit string.")

    # Dimensionless
    if s == '1':
        return UNITLESS

    # Fast path: direct registry lookup (also covers registered names
    # containing spaces or parentheses, e.g. "troy pound")
    unit = _lookup_symbol(s)
    if unit is None:
        s = ' '.join(s.split())
        unit = _lookup_symbol(s)
    if unit is not None:
        return unit
    return _unit_parse.parse(s, _lookup_symbol, _special_symbols())


def _normalise_scalar(x: Any) -> Any:
    """Coerce numpy/generic real scalars to plain ``int``/``float``.

    Standard-unit registration and lookup require plain Python numbers,
    so e.g. ``np.int64(3)`` must behave the same as ``3``.  Non-Real
    objects (JAX tracers, arrays) pass through untouched, except for
    concrete 0-d arrays, which are unwrapped to their scalar value (a
    concrete 0-d array would otherwise construct a ``Unit`` whose
    ``hash()`` crashes).  JAX tracers are left untouched so tracing
    under ``jit``/``vmap``/``grad`` keeps working.
    """
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, numbers.Integral):
        return int(x)
    if isinstance(x, numbers.Real):
        return float(x)
    if getattr(x, 'ndim', None) == 0 and not _is_tracer(x):
        try:
            return _normalise_scalar(x.item())
        except (AttributeError, TypeError):
            pass
    return x


def _validate_scale_and_factor(scale: Any, factor: Any) -> None:
    """Reject scale/factor values that poison Unit arithmetic.

    Non-finite scales break ``__eq__``/``__hash__`` (NaN is unequal to
    itself) and display; non-positive factors cannot represent a physical
    conversion and crash ``reverse()``/division (factor 0) or produce
    complex factors under fractional powers (negative factor).  Values
    that are not plain real numbers (e.g. JAX tracers) are not checked.
    """
    if isinstance(scale, complex) or isinstance(factor, complex):
        raise TypeError(
            f"Unit scale and factor must be real numbers; "
            f"got scale={scale!r}, factor={factor!r}."
        )
    if isinstance(scale, int) and not isinstance(scale, bool):
        try:
            float(scale)
        except OverflowError:
            raise ValueError(
                f"Unit scale overflows the float range: {scale!r}."
            ) from None
    if isinstance(scale, (int, float)) and not math.isfinite(scale):
        raise ValueError(
            f"Unit scale must be a finite real number; got scale={scale!r}."
        )
    if isinstance(factor, int) and not isinstance(factor, bool):
        try:
            float(factor)
        except OverflowError:
            raise ValueError(
                f"Unit factor overflows the float range: {factor!r}."
            ) from None
    if isinstance(factor, (int, float)):
        if not math.isfinite(factor):
            raise ValueError(
                f"Unit factor must be a finite real number; got factor={factor!r}."
            )
        if factor <= 0:
            raise ValueError(
                f"Unit factor must be positive; got factor={factor!r}."
            )


class Unit:
    r"""
    A physical unit.

    Basically, a unit is just a number with given dimensions, e.g.
    mvolt = 0.001 with the dimensions of voltage. The units module
    defines a large number of standard units, and you can also define
    your own (see below).

    Mathematically, a unit represents:

    .. math::

        \text{{factor}} \times \text{{base}}^{\text{{scale}}} \times \text{{dimension}}

    where the ``factor`` is the conversion factor of the unit (e.g.
    ``1 calorie = 4.18400 Joule``, so the factor is 4.18400), the ``base``
    is the base of the exponent (e.g. 10 for the kilo prefix), the ``scale``
    is the exponent of the base (e.g. 3 for the kilo prefix), and the
    ``dimension`` is the physical dimensions of the unit (e.g. ``joule`` for
    energy).

    The unit class also keeps track of the named units it was composed
    from, to generate a readable string representation.  See below.

    Parameters
    ----------
    dim : Dimension, optional
        The physical dimensions of the unit. Defaults to ``DIMENSIONLESS``.
    scale : array_like, optional
        The scale exponent, e.g. 3 for a "k" (kilo) prefix. Defaults to 0.
    base : array_like, optional
        The base of the exponent. Must be 10 (the only supported base);
        other values raise ``ValueError``.
    factor : array_like, optional
        The conversion factor of the unit. Must be positive. Defaults to 1.
    name : str, optional
        The full name of the unit, e.g. ``'volt'``.
    dispname : str, optional
        The display name, e.g. ``'V'``.
    is_fullname : bool, optional
        Whether ``name`` is the canonical full name. Defaults to ``True``.
    display_parts : list of tuple, optional
        Legacy ``(name, dispname, exponent)`` display components, accepted
        for backward compatibility (old pickles).

    Notes
    -----
    When creating scaled units, you can use the following prefixes:

    ======     ======  ==============
    Factor     Name    Prefix
    ======     ======  ==============
    10^24      yotta   Y
    10^21      zetta   Z
    10^18      exa     E
    10^15      peta    P
    10^12      tera    T
    10^9       giga    G
    10^6       mega    M
    10^3       kilo    k
    10^2       hecto   h
    10^1       deka    da
    1
    10^-1      deci    d
    10^-2      centi   c
    10^-3      milli   m
    10^-6      micro   u (\mu in SI)
    10^-9      nano    n
    10^-12     pico    p
    10^-15     femto   f
    10^-18     atto    a
    10^-21     zepto   z
    10^-24     yocto   y
    ======     ======  ==============

    **Display**

    A unit keeps the named units it was composed from, so composites
    display the way they were written: ``u.mS * u.nA / u.cm ** 2`` is
    ``mS * nA / cm^2`` and ``u.mV * u.ms`` is ``ms * mV``.  Terms are
    ordered canonically (numerator first, then by symbol), so ``a * b`` and
    ``b * a`` display identically.  A composite collapses automatically
    only onto a curated set of units (SI base units, N, Pa, J, W, C, V, F,
    ohm, S, Hz and their prefixed forms):

    .. code-block:: python

        >>> import saiunit as u
        >>> u.kilogram * u.metre / u.second ** 2
        Unit("N")
        >>> u.volt * u.second
        Unit("s * V")
        >>> (u.volt * u.second).simplify()
        Unit("Wb")

    ``str(unit)`` is canonical ASCII that :func:`parse_unit` reads back;
    :meth:`to_string` also renders Unicode (``µF / cm²``) and LaTeX.

    **Defining your own**

    .. code-block:: python

        >>> import saiunit as u
        >>> cal = u.Unit.create(u.joule.dim, 'calorie', 'cal', factor=4.184)
        >>> cal / u.gram
        Unit("cal / g")

    Comparison and arithmetic treat a bare ``Unit`` operand differently.
    Comparing a ``Quantity`` against a bare ``Unit`` implicitly reads the
    unit as "one of that unit" (``1 * unit``), so ``Quantity(1, mV) == mV``
    is ``True``. Addition/subtraction, by contrast, reject a bare ``Unit``
    operand outright (``Quantity(1, mV) + mV`` raises ``TypeError``),
    since there is no natural mantissa to add. Both behaviours are
    intentional; see ``__eq__`` for the comparison side.

    Examples
    --------
    Create a simple unit:

    .. code-block:: python

        >>> import saiunit as u
        >>> u.volt
        Unit("V")
        >>> u.mvolt
        Unit("mV")

    Combine units:

    .. code-block:: python

        >>> import saiunit as u
        >>> u.volt / u.amp
        Unit("V / A")

    """

    __module__ = "saiunit"
    __slots__ = ["_dim", "_base", "_scale", "_factor", "_terms", "_pre", "_name", "_hash", "_disp", "_str"]
    __array_priority__ = 1000

    _dim: 'Dimension'
    _base: int | float
    _scale: int | float
    _factor: int | float
    # Canonical ``((Atom, exponent), ...)`` display terms and the numeric
    # ``(scale, factor)`` prefactor; ``scale``/``factor`` are derived from
    # them.  ``_terms`` is None only when scale or factor is a JAX tracer.
    _terms: 'Terms | None'
    _pre: 'Pre'
    # Explicit full name of a named unit or power alias; None for composites.
    _name: str | None
    _hash: int | None
    _disp: 'tuple[Terms, Pre] | None'
    _str: str | None

    def __init__(
        self,
        dim: 'Dimension | str | None' = None,
        scale: int | float = 0,
        base: int | float = 10.,
        factor: int | float = 1.,
        name: str | None = None,
        dispname: str | None = None,
        is_fullname: bool = True,
        display_parts=None,
    ):
        # String-based construction: Unit("mV"), Unit("J / kg"), etc.
        if isinstance(dim, str):
            # The string form ignores every other constructor argument —
            # silently dropping ``Unit("mV", scale=99, factor=99)`` was a
            # source of confusing bugs.  Reject any non-default secondary
            # argument explicitly so the caller knows.
            extras = []
            if scale != 0:
                extras.append("scale")
            if base != 10.:
                extras.append("base")
            if factor != 1.:
                extras.append("factor")
            if name is not None:
                extras.append("name")
            if dispname is not None:
                extras.append("dispname")
            if is_fullname is not True:
                extras.append("is_fullname")
            if display_parts is not None:
                extras.append("display_parts")
            if extras:
                raise TypeError(
                    "Unit(str, ...) does not accept additional arguments: "
                    + ", ".join(extras)
                    + ". Use parse_unit() and modify the result, or construct "
                    "the Unit from a Dimension instead."
                )
            self._assign(parse_unit(dim))
            return

        # ``base`` is fixed at 10 for now — Units canonicalize to base=10
        # internally, and accepting other bases silently rewrote them,
        # losing information.  Raise so callers can not be surprised.
        if base != 10.:
            raise ValueError(
                f"Unit currently only supports base=10; got base={base!r}. "
                "Encode non-decimal scales in ``factor`` instead."
            )

        scale = _normalise_scalar(scale)
        factor = _normalise_scalar(factor)
        _validate_scale_and_factor(scale, factor)

        # The physical unit dimensions of this unit
        if dim is None:
            dim = DIMENSIONLESS
        if not isinstance(dim, Dimension):
            raise TypeError(f'Expected instance of Dimension, but got {dim}')

        if not (isinstance(scale, (int, float)) and isinstance(factor, (int, float))):
            # JAX tracer scale/factor: no symbolic display terms.
            self._assign(Unit._make(dim, scale, factor, None, PRE_ONE))
        elif display_parts is not None:
            self._assign(_from_legacy(dim, scale, factor, name, dispname, is_fullname, display_parts))
        elif name is not None and is_fullname:
            self._assign(_named_unit(dim, scale, factor, name, dispname if dispname is not None else name))
        else:
            self._assign(_anonymous_unit(dim, scale, factor))

    @classmethod
    def _make(
        cls,
        dim: Dimension,
        scale: Any,
        factor: Any,
        terms: 'Terms | None',
        pre: 'Pre',
        name: str | None = None,
    ) -> 'Unit':
        """Construct a Unit from already-validated fields (fast path)."""
        u = object.__new__(cls)
        u._dim = dim
        u._base = 10.
        u._scale = scale
        u._factor = factor
        u._terms = terms
        u._pre = pre
        u._name = name
        u._hash = None
        u._disp = None
        u._str = None
        return u

    def _assign(self, other: 'Unit') -> None:
        self._dim = other._dim
        self._base = other._base
        self._scale = other._scale
        self._factor = other._factor
        self._terms = other._terms
        self._pre = other._pre
        self._name = other._name
        self._hash = None
        self._disp = other._disp
        self._str = other._str

    def _single_atom(self) -> 'Atom | None':
        """The atom of a plain named unit (one atom, exponent 1, no prefactor)."""
        terms = self._terms
        if terms is not None and len(terms) == 1 and terms[0][1] == 1 and self._pre == PRE_ONE:
            return terms[0][0]
        return None


    @property
    def factor(self) -> float:
        """
        Return the conversion factor of the unit.

        The factor represents a multiplicative constant that converts a
        quantity expressed in this unit to its base-unit equivalent.  For
        example, 1 calorie = 4.184 joule, so ``calorie.factor == 4.184``.

        Returns
        -------
        float
            The conversion factor.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.volt.factor
            1.0
        """
        return self._factor

    @factor.setter
    def factor(self, factor):
        raise NotImplementedError(
            "Cannot set the factor of a Unit object directly,"
            "Please create a new Unit object with the factor you want."
        )

    @property
    def base(self) -> float:
        """
        Return the base of the unit's scale exponent.

        The base is the number that is raised to the ``scale`` power to
        produce the unit's magnitude.  For SI-prefixed units this is 10
        (e.g. ``kilo`` means ``10 ** 3``).

        Returns
        -------
        float
            The base of the exponent.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.kvolt.base
            10.0
        """
        return self._base

    @base.setter
    def base(self, base):
        raise NotImplementedError(
            "Cannot set the base of a Unit object directly,"
            "Please create a new Unit object with the base you want."
        )

    @property
    def scale(self) -> float | int:
        """
        Return the scale exponent of the unit.

        The scale is the integer exponent applied to :attr:`base` to
        produce the unit's magnitude relative to the base unit.  For
        example, ``mvolt`` has ``scale == -3`` (i.e. ``10 ** -3``).

        Returns
        -------
        float or int
            The scale exponent.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.mvolt.scale
            -3
        """
        return self._scale

    @scale.setter
    def scale(self, scale):
        raise NotImplementedError(
            "Cannot set the scale of a Unit object directly,"
            "Please create a new Unit object with the scale you want."
        )

    @property
    def magnitude(self) -> float:
        """
        Return the absolute magnitude of the unit.

        The magnitude is computed as ``factor * base ** scale`` and
        represents the overall multiplicative factor that converts a
        value in this unit to the corresponding base-unit value.

        Returns
        -------
        float
            The absolute magnitude of the unit.  Extreme scales whose
            magnitude exceeds the float range yield ``inf`` (IEEE-754
            overflow semantics) rather than raising.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.mvolt.magnitude
            0.001
            >>> u.kvolt.magnitude
            1000.0
        """
        # magnitude = factor * base ** scale
        try:
            return self.factor * self.base ** self.scale
        except OverflowError:
            # Python float pow raises on overflow instead of following
            # IEEE-754; treat extreme scales as infinity like every other
            # float operation in the library.
            return float('inf')

    @magnitude.setter
    def magnitude(self, scale):
        raise NotImplementedError(
            "Cannot set the magnitude of a Unit object."
        )

    @property
    def dim(self) -> Dimension:
        """
        Return the physical unit dimensions of this unit.

        Returns
        -------
        Dimension
            The :class:`~saiunit.Dimension` instance describing the
            physical dimensions (e.g. length, mass, time, ...).

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.volt.dim
            metre ** 2 * kilogram * second ** -3 * amp ** -1
        """
        return self._dim

    @dim.setter
    def dim(self, value):
        # Do not support setting the unit directly
        raise NotImplementedError(
            "Cannot set the dimension of a Quantity object directly,"
            "Please create a new Quantity object with the dimension you want."
        )

    @property
    def is_unitless(self) -> bool:
        """
        Whether the unit is dimensionless with no scaling.

        A unit is considered unitless when its dimension is dimensionless,
        its scale exponent is 0, and its factor is 1.0.

        Returns
        -------
        bool
            ``True`` if the unit is unitless, ``False`` otherwise.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.UNITLESS.is_unitless
            True
            >>> u.volt.is_unitless
            False
        """
        return self.dim.is_dimensionless and self.scale == 0 and self.factor == 1.0

    @property
    def should_display_unit(self) -> bool:
        """
        Whether the unit should be shown in formatted output.

        Returns ``True`` unless the unit renders as ``"1"``: dimensionless
        units that carry a meaningful name (e.g. radian, steradian) or a
        scale (``mV / V``) are shown.

        Returns
        -------
        bool
            ``True`` if the unit should be displayed, ``False`` otherwise.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.volt.should_display_unit
            True
            >>> u.UNITLESS.should_display_unit
            False
        """
        return str(self) != '1'

    @property
    def is_fullname(self) -> bool:
        """Whether this is a named unit (or power alias) rather than a composite."""
        return self._name is not None

    @property
    def name(self):
        """
        Return the full name of the unit.

        For composite units this is the canonical display string.

        Returns
        -------
        str
            The full name of the unit (e.g. ``'volt'``, ``'mvolt'``).

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.volt.name
            'volt'
            >>> u.mvolt.name
            'mvolt'
        """
        return self._name if self._name is not None else str(self)

    @name.setter
    def name(self, name):
        raise NotImplementedError(
            "Cannot set the name of a Unit object directly,"
            "Please create a new Unit object with the name you want."
        )

    @property
    def dispname(self):
        """
        Return the display name of the unit.

        The display name is the canonical string used when rendering the
        unit (e.g. ``'V'`` for volt, ``'mV'`` for millivolt), identical to
        ``str(unit)``.

        Returns
        -------
        str
            The display name of the unit.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.volt.dispname
            'V'
            >>> u.mvolt.dispname
            'mV'
        """
        return str(self)

    @dispname.setter
    def dispname(self, dispname):
        raise NotImplementedError(
            "Cannot set the dispname of a Unit object directly,"
            "Please create a new Unit object with the dispname you want."
        )

    def factorless(self) -> 'Unit':
        """
        Return an equivalent Unit with the factor set to 1.

        If this unit already has ``factor == 1``, the unit itself is
        returned unchanged.  (Substituting a registry-preferred alias
        here would silently rename units, e.g. becquerel → hertz or
        steradian → radian.)

        Returns
        -------
        Unit
            This unit if its factor is already 1, otherwise a Unit with
            the factor set to 1 (the registered standard unit for the
            same dimension/scale when one exists).

        Notes
        -----
        For a dimensionless unit, the result may carry an angular label
        (e.g. ``rad``) even though the input was a plain ratio: saiunit
        does not distinguish angle from ratio in the dimensionless
        registry keyspace.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> pound = u.Unit.create(
            ...     u.get_or_create_dimension(kg=1), 'pound', 'lb', factor=0.453592)
            >>> pound.factor
            0.453592
            >>> pound.factorless().factor
            1.0
        """
        # Already factorless — nothing to strip.
        if isinstance(self.factor, (int, float)) and self.factor == 1.:
            return self

        # using standard units
        key = (self.dim, self.scale, self.base, 1.)
        if key in _standard_units:
            candidate = _standard_units[key]
            # Dimensionless keys collapse radian powers (rad^2, rad^3, ...)
            # onto plain ratio scales; substituting a power alias would
            # mislabel a first-power unit (arcmin -> "crad^2").  Only
            # accept plain named units for dimensionless dims.
            if not self.dim.is_dimensionless or candidate._single_atom() is not None:
                return candidate

        # using temporary units
        return Unit(dim=self.dim, scale=self.scale, base=self.base)

    def simplify(self) -> 'Unit':
        """
        Return the registered unit with exactly this unit's value, if any.

        Unit arithmetic only collapses onto a curated set of units (``kg * m
        / s^2`` displays as ``N``, but ``V * s`` stays ``V * s``).
        ``simplify`` collapses onto any registered unit, e.g. ``V * s`` →
        ``Wb`` and ``mol / l`` → ``M``.  Dimensionless units, and units with
        no registered match, become an anonymous unit (``mV / V`` →
        ``10^-3``).

        Returns
        -------
        Unit
            A unit equal to this one.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> (u.volt * u.second).simplify()
            Unit("Wb")
            >>> (u.mvolt / u.volt).simplify()
            Unit("10^-3")
        """
        if self._terms is None:
            return self
        if not self.dim.is_dimensionless:
            unit = _standard_units.get((self.dim, self.scale, self.base, self.factor))
            if unit is not None:
                return unit
        return _anonymous_unit(self.dim, self.scale, self.factor)

    def copy(self):
        """
        Return a copy of this Unit.

        Returns
        -------
        Unit
            A new Unit object with the same attributes.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> v = u.volt.copy()
            >>> v == u.volt
            True
            >>> v is u.volt
            False
        """
        return Unit._make(self._dim, self._scale, self._factor, self._terms, self._pre, self._name)

    def __deepcopy__(self, memodict):
        return self.copy()

    def __hash__(self):
        if self._hash is None:
            # Equality is *physical*: two units that resolve to the same
            # ``(dim, factor, base, scale)`` must hash equal regardless
            # of name spelling (e.g. ``metre`` vs ``meter``).
            self._hash = hash(
                (
                    self.dim,
                    self.factor,
                    self.base,
                    self.scale,
                )
            )
        return self._hash

    def has_same_magnitude(self, other: 'Unit') -> bool:
        """
        Whether this Unit has the same magnitude as another Unit.

        Two units have the same magnitude when they share the same
        ``scale``, ``base``, and ``factor``.  Note that the comparison is
        component-wise, not on the computed product ``factor * base**scale``:
        a unit encoded as ``scale=3`` and one encoded as ``factor=1000.``
        have equal magnitude products but compare unequal here.

        Note also that comparing quantities across factor units after
        conversion is exact-float equality: e.g. ``1*u.inch == 25.4*u.mm``
        is ``False`` (the conversion ratio is off by one ulp). For factor
        units, prefer ``u.math.isclose`` over ``==``.

        Parameters
        ----------
        other : Unit
            The other Unit to compare with.

        Returns
        -------
        bool
            Whether the two Units have the same magnitude.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.mvolt.has_same_magnitude(u.mamp)
            True
            >>> u.mvolt.has_same_magnitude(u.volt)
            False
        """
        return self.scale == other.scale and self.base == other.base and self.factor == other.factor

    def has_same_base(self, other: 'Unit') -> bool:
        """
        Whether this Unit has the same ``base`` as another Unit.

        Parameters
        ----------
        other : Unit
            The other Unit to compare with.

        Returns
        -------
        bool
            Whether the two Units have the same base.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.volt.has_same_base(u.amp)
            True
            >>> u.volt.has_same_base(u.mvolt)
            True
        """
        return self.base == other.base

    def has_same_dim(self, other: 'Unit') -> bool:
        """
        Whether this Unit has the same unit dimensions as another Unit.

        Parameters
        ----------
        other : Unit
            The other Unit to compare with.

        Returns
        -------
        bool
            Whether the two Units have the same unit dimensions.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.volt.has_same_dim(u.mvolt)
            True
            >>> u.volt.has_same_dim(u.amp)
            False
        """
        from ._base_getters import get_dim
        other_dim = get_dim(other)
        return get_dim(self) == other_dim

    @staticmethod
    def create(
        dim: Dimension,
        name: str,
        dispname: str,
        scale: int | float = 0,
        base: float = 10.,
        factor: float = 1.,
        *,
        auto_simplify: bool = False,
        unicode: str | None = None,
        latex: str | None = None,
    ) -> 'Unit':
        """
        Create and register a new named unit.

        Parameters
        ----------
        dim : Dimension
            The dimensions of the unit.
        name : `str`
            The full name of the unit, e.g. ``'volt'``
        dispname : `str`
            The ASCII display symbol, e.g. ``'V'``.  A symbol of the form
            ``'<registered symbol>^<int>'`` (e.g. ``'cm^2'``) creates a
            power alias equivalent to :meth:`create_power`.
        scale : int, optional
            The scale of this unit as an exponent of 10, e.g. -3 for a unit that
            is 1/1000 of the base scale. Defaults to 0 (i.e. a base unit).
        base: float, optional
            The base for this unit (as the base of the exponent). Must be 10,
            the only supported base; other values raise ``ValueError``.
        factor: float, optional
            The factor for this unit (as the conversion factor), e.g.
            a factor of 1 cal = 4.18400 J, where 4.18400 is the factor.
            Defaults to 1.
        auto_simplify : bool, optional
            Display arithmetic results with exactly this unit's dimension
            and magnitude as this unit.  Defaults to False.
        unicode : str, optional
            Unicode symbol (e.g. ``'Ω'``), used by ``to_string('unicode')``
            and accepted by the parser.  Defaults to ``dispname``.
        latex : str, optional
            LaTeX symbol (e.g. ``r'\\Omega'``).  Defaults to
            ``\\mathrm{dispname}``.

        Returns
        -------
        u : Unit
            The new unit.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> from saiunit import Dimension
            >>> energy_dim = u.joule.dim
            >>> cal = u.Unit.create(energy_dim, 'calorie', 'cal', factor=4.184)
            >>> cal
            Unit("cal")
        """
        if base != 10.:
            raise ValueError(
                f"Unit currently only supports base=10; got base={base!r}. "
                "Encode non-decimal scales in ``factor`` instead."
            )
        scale = _normalise_scalar(scale)
        factor = _normalise_scalar(factor)
        _validate_scale_and_factor(scale, factor)
        u = _power_alias(dim, scale, factor, name, dispname)
        if u is None:
            u = _named_unit(dim, scale, factor, name, dispname, unicode=unicode, latex=latex)
        add_standard_unit(u, auto_simplify=auto_simplify)
        return u

    @staticmethod
    def create_power(baseunit: 'Unit', n: int, name: str | None = None) -> 'Unit':
        """
        Create and register a power alias such as ``cmetre2 = cm^2``.

        Parameters
        ----------
        baseunit : Unit
            The unit to raise to a power.
        n : int
            The exponent.
        name : str, optional
            The full name; defaults to ``baseunit.name + str(n)``.

        Returns
        -------
        Unit
            A unit equal to ``baseunit ** n`` that displays as ``'cm^2'``
            and is registered under *name*.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> cm2 = u.Unit.create_power(u.cmetre, 2)
            >>> cm2.name, str(cm2)
            ('cmetre2', 'cm^2')
        """
        p = baseunit ** n
        u = Unit._make(p._dim, p._scale, p._factor, p._terms, p._pre,
                       name if name is not None else f"{baseunit.name}{n}")
        add_standard_unit(u)
        return u

    @staticmethod
    def create_scaled_unit(baseunit: 'Unit', scalefactor: str) -> 'Unit':
        """
        Create a scaled unit from a base unit.

        The scaled unit inherits the Unicode/LaTeX symbols of the base unit
        (``kohm`` renders ``kΩ``) and, except for the ``da``, ``h``, ``d``
        and ``c`` prefixes, its ``auto_simplify`` flag.

        Parameters
        ----------
        baseunit : `Unit`
            The unit of which to create a scaled version, e.g. ``volt``,
            ``amp``.
        scalefactor : `str`
            The scaling factor, e.g. ``"m"`` for mvolt, mamp

        Returns
        -------
        u : Unit
            The new unit.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> uvolt = u.Unit.create_scaled_unit(u.volt, 'u')
            >>> uvolt.name
            'uvolt'
            >>> uvolt.scale
            -6
        """
        if scalefactor not in _siprefixes:
            raise ValueError(
                f"Unknown SI prefix {scalefactor!r}. "
                f"Valid prefixes are: {list(_siprefixes.keys())}"
            )
        name = scalefactor + baseunit.name
        scale = _siprefixes[scalefactor] + baseunit.scale
        base_atom = baseunit._single_atom()
        if base_atom is None:
            u = _named_unit(baseunit.dim, scale, baseunit.factor, name, scalefactor + baseunit.dispname)
            auto = False
        else:
            if scalefactor == '':
                unicode, latex = base_atom.unicode, base_atom.latex
            else:
                micro = scalefactor == 'u'
                unicode = ('µ' if micro else scalefactor) + base_atom.unicode
                if base_atom.latex == latex_mathrm(base_atom.symbol):
                    latex = (r'\mu' if micro else '') + latex_mathrm(
                        base_atom.symbol if micro else scalefactor + base_atom.symbol)
                else:
                    latex = (r'\mu ' if micro else latex_mathrm(scalefactor)) + base_atom.latex
            u = _named_unit(baseunit.dim, scale, baseunit.factor, name, scalefactor + base_atom.symbol,
                            unicode=unicode, latex=latex, prefix=scalefactor or None)
            auto = base_atom in _auto_atoms and scalefactor not in ('da', 'h', 'd', 'c')
        add_standard_unit(u, auto_simplify=auto)
        return u

    def _display(self) -> 'tuple[Terms, Pre]':
        """Return the ``(terms, prefactor)`` to render (cached).

        * Dimensionless units of magnitude 1 show only their named
          dimensionless atoms (``rad``, ``sr``) — or ``1``.  Other
          dimensionless units show every term (``mV / V``, ``h / ms``).
        * Dimensioned units containing a dimensionless atom keep their
          terms (``rad / s``, ``m * rad``).
        * A single plain atom shows itself.
        * Otherwise collapse onto the unique ``auto_simplify`` target with
          the same dimension and magnitude (``kg * m / s^2`` → ``N``), or
          with the same magnitude apart from the prefactor
          (``2 * V``); failing that, show the terms.
        """
        disp = self._disp
        if disp is not None:
            return disp
        terms, pre = self._terms, self._pre
        assert terms is not None
        if self._dim.is_dimensionless:
            if self._scale == 0 and self._factor == 1:
                named = tuple(t for t in terms if t[0].dim.is_dimensionless)
                disp = (named, PRE_ONE) if named and derive_magnitude(named, PRE_ONE) == (0, 1) else ((), PRE_ONE)
            else:
                disp = (terms, pre)
        elif (
            any(t[0].dim.is_dimensionless for t in terms)
            or (len(terms) == 1 and terms[0][1] == 1 and pre == PRE_ONE)
        ):
            disp = (terms, pre)
        else:
            disp = (terms, pre)
            targets = _auto_targets.get((self._dim, self._scale, self._base, self._factor))
            if targets is not None and len(targets) == 1:
                disp = (((targets[0], 1),), PRE_ONE)
            elif pre != PRE_ONE:
                scale, factor = derive_magnitude(terms, PRE_ONE)
                targets = _auto_targets.get((self._dim, scale, self._base, factor))
                if targets is not None and len(targets) == 1:
                    disp = (((targets[0], 1),), pre)
        self._disp = disp
        return disp

    def _traced_str(self) -> str:
        """Display string of a unit whose scale or factor is a JAX tracer."""
        terms = _si_terms(self.dim)
        dim_str = render(terms, PRE_ONE) if terms else ''
        pre_str = f'{self.factor} * {self.base}^{self.scale}'
        return f'{pre_str} * {dim_str}' if dim_str else pre_str

    def to_string(self, format: str = 'ascii') -> str:
        """
        Render the unit as a string.

        Parameters
        ----------
        format : str
            One of ``'ascii'`` (canonical, parseable: ``mS * nA / cm^2``),
            ``'ascii_inline'`` (``mS * nA * cm^-2``), ``'unicode'``
            (``mS·nA / cm²``), ``'unicode_inline'`` (``mS·nA·cm⁻²``),
            ``'latex'`` (``\\frac{\\mathrm{mS}\\,\\mathrm{nA}}{\\mathrm{cm}^{2}}``)
            or ``'latex_inline'``.

        Returns
        -------
        str

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> (u.uF / u.cm ** 2).to_string('unicode')
            'µF / cm²'
            >>> u.kohm.to_string('latex')
            '\\\\mathrm{k}\\\\Omega'
        """
        if format == 'ascii':
            return str(self)
        if format not in STYLES:
            raise ValueError(f"Unknown unit format {format!r}; expected one of {STYLES}.")
        if self._terms is None:
            return self._traced_str()
        return render(*self._display(), format)

    def __format__(self, format_spec: str) -> str:
        return self.to_string(format_spec or 'ascii')

    def _repr_latex_(self) -> str:
        return f"${self.to_string('latex')}$"

    def __repr__(self) -> str:
        return f"Unit(\"{self}\")"

    def __str__(self) -> str:
        s = self._str
        if s is None:
            s = self._traced_str() if self._terms is None else render(*self._display())
            self._str = s
        return s

    def __mul__(self, other) -> 'Unit | Quantity':
        # self * other
        if isinstance(other, Unit):
            _assert_same_base(self, other)
            return _compose(self, 1, other, 1)

        elif isinstance(other, Dimension):
            raise TypeError(f"unit {self} cannot multiply by a Dimension {other}.")

        else:
            from ._base_quantity import Quantity
            if isinstance(other, Quantity):
                return Quantity(other.mantissa, unit=(self * other.unit))  # type: ignore[arg-type]
            return Quantity(other, unit=self)

    def __rmul__(self, other) -> 'Unit | Quantity':
        # other * self
        if isinstance(other, Unit):
            return other.__mul__(self)

        from ._base_quantity import Quantity
        if isinstance(other, Quantity):
            return Quantity(other.mantissa, unit=(other.unit * self))  # type: ignore[arg-type]
        return Quantity(other, unit=self)

    def __imul__(self, other):
        raise NotImplementedError("Units cannot be modified in-place")

    def __div__(self, other) -> 'Unit | Quantity':
        # self / other
        if isinstance(other, Unit):
            _assert_same_base(self, other)
            return _compose(self, 1, other, -1)

        elif isinstance(other, Dimension):
            raise TypeError(f"unit {self} cannot divide by a Dimension {other}.")

        else:
            # Mirror ``__mul__``: dividing a unit by a number or quantity
            # yields a Quantity (e.g. ``mV / 2`` == 0.5 mV), keeping the
            # operator surface consistent with ``mV * 2`` and ``2 / mV``.
            from ._base_quantity import Quantity
            if isinstance(other, Quantity):
                return Quantity(1.0 / other.mantissa, unit=(self / other.unit))
            return Quantity(1.0 / other, unit=self)

    def __rdiv__(self, other) -> 'Unit | Quantity':
        # other / self
        if isinstance(other, Unit):
            return other.__div__(self)

        from ._base_quantity import Quantity
        if isinstance(other, Quantity):
            return Quantity(other.mantissa, unit=(other.unit / self))
        return Quantity(other, unit=self.reverse())

    def reverse(self):
        """
        Return the multiplicative inverse of this unit.

        Computes ``1 / self``, producing a new unit with negated scale,
        inverted factor, and reciprocal dimensions.  ``u.reverse().reverse()``
        equals ``u`` exactly.

        Returns
        -------
        Unit
            A new Unit representing the reciprocal of this unit.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.second.reverse()
            Unit("Hz")
            >>> u.metre.reverse()
            Unit("1 / m")
        """
        return _compose(self, -1)

    def __idiv__(self, other):
        raise NotImplementedError("Units cannot be modified in-place")

    def __truediv__(self, oc):
        # self / oc
        return self.__div__(oc)

    def __rtruediv__(self, oc):
        # oc / self
        return self.__rdiv__(oc)

    def __itruediv__(self, other):
        raise NotImplementedError("Units cannot be modified in-place")

    def __floordiv__(self, oc):
        raise NotImplementedError("Units cannot be performed floor division")

    def __rfloordiv__(self, oc):
        raise NotImplementedError("Units cannot be performed floor division")

    def __ifloordiv__(self, other):
        raise NotImplementedError("Units cannot be modified in-place")

    def __pow__(self, other):
        # self ** other
        from ._base_getters import is_scalar_type
        if is_scalar_type(other):
            return _compose(self, norm_exp(_normalise_scalar(other)))
        else:
            raise TypeError(
                f"unit cannot perform an exponentiation (unit ** other) with a non-scalar, "
                f"since one unit cannot contain multiple units. \n"
                f"But we got unit={self}, other={other}"
            )

    def __ipow__(self, other, modulo=None):
        raise NotImplementedError("Units cannot be modified in-place")

    def __add__(self, other):
        raise TypeError(
            "Units cannot be added: addition is defined on quantities, not units. "
            "To add quantities, attach mantissas first (e.g. 1*ms + 2*ms)."
        )

    def __radd__(self, other):
        raise TypeError(
            "Units cannot be added: addition is defined on quantities, not units. "
            "To add quantities, attach mantissas first (e.g. 1*ms + 2*ms)."
        )

    def __iadd__(self, other):
        raise NotImplementedError("Units cannot be modified in-place")

    def __sub__(self, other):
        raise TypeError(
            "Units cannot be subtracted: subtraction is defined on quantities, not "
            "units. To subtract quantities, attach mantissas first (e.g. 2*ms - 1*ms)."
        )

    def __rsub__(self, other):
        raise TypeError(
            "Units cannot be subtracted: subtraction is defined on quantities, not "
            "units. To subtract quantities, attach mantissas first (e.g. 2*ms - 1*ms)."
        )

    def __isub__(self, other):
        raise NotImplementedError("Units cannot be modified in-place")

    def __mod__(self, oc):
        raise NotImplementedError("Units cannot be performed modulo")

    def __rmod__(self, oc):
        raise NotImplementedError("Units cannot be performed modulo")

    def __imod__(self, other):
        raise NotImplementedError("Units cannot be modified in-place")

    def __eq__(self, other) -> bool:
        # Two Units are equal when they have the same dim/scale/base/factor
        # *components*.  Names and display strings are not part of equality:
        # spelling aliases such as ``metre`` vs ``meter`` and
        # registry-canonical compounds such as ``A * ohm`` vs ``V`` compare
        # equal, and the corresponding hashes collide as required by the
        # ``__hash__`` contract.  Note the comparison is component-wise,
        # not magnitude-based: ``Unit(scale=3)`` and ``Unit(factor=1000.)``
        # have the same magnitude but compare unequal.  Factor-1
        # dimensionless units (radian, steradian) compare equal to
        # ``UNITLESS`` — physically they are 1.
        #
        # Non-Unit operands return ``NotImplemented`` so that the other
        # operand's reflected comparison runs (e.g. Quantity.__eq__),
        # keeping ``unit == quantity`` and ``quantity == unit`` symmetric.
        if not isinstance(other, Unit):
            return NotImplemented
        return (
            (other.dim == self.dim)
            and (other.scale == self.scale)
            and (other.base == self.base)
            and (other.factor == self.factor)
        )

    def __ne__(self, other) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __abs__(self) -> 'Unit':
        """Return the unit itself — units are always non-negative."""
        return self

    def __reduce__(self):
        # For pickling.  The display terms are pickled with the unit so
        # composites (e.g. ``mS * nA / cm^2``) keep their rendering.
        return (
            _to_unit_v2,
            (self._dim, self._scale, self._factor, self._terms, self._pre, self._name),
        )


def _to_unit_v2(dim, scale, factor, terms, pre, name):
    """Private pickle reconstruction shim for Unit."""
    return Unit._make(dim, scale, factor, terms, pre, name)


def _to_unit(dim, scale, base, factor, name, dispname, is_fullname,
             display_parts=None):
    """Reconstruct a Unit pickled by older saiunit versions (string display parts)."""
    return Unit(
        dim=dim, scale=scale, base=base, factor=factor,
        name=name, dispname=dispname, is_fullname=is_fullname,
        display_parts=display_parts if display_parts is not None else [],
    )


_to_unit_v2.__module__ = 'saiunit._base_unit'
_to_unit.__module__ = 'saiunit._base_unit'

# The seven SI base atoms; anonymous units (``Unit(dim)``) are displayed in
# terms of them.  They are interned, so ``metre``/``kilogram``/... defined
# in ``_unit_common`` share these atoms.
_SI_BASE_ATOMS = tuple(
    make_atom(symbol, name, get_or_create_dimension(**{symbol: 1}), 0, 1.)
    for symbol, name in (
        ('m', 'metre'), ('kg', 'kilogram'), ('s', 'second'), ('A', 'amp'),
        ('K', 'kelvin'), ('mol', 'mole'), ('cd', 'candle'),
    )
)


def _si_terms(dim: Dimension) -> 'Terms':
    return combine_terms(
        tuple((atom, norm_exp(d)) for atom, d in zip(_SI_BASE_ATOMS, dim._dims.tolist()) if d), 1
    )


def _anonymous_unit(dim: Dimension, scale: Any, factor: Any) -> Unit:
    """A unit with no name, displayed via SI base atoms (or an auto target)."""
    return Unit._make(dim, scale, factor, _si_terms(dim), (scale, factor))


def _named_unit(
    dim: Dimension,
    scale: Any,
    factor: Any,
    name: str,
    symbol: str,
    unicode: str | None = None,
    latex: str | None = None,
    prefix: str | None = None,
) -> Unit:
    atom = make_atom(symbol, name, dim, scale, factor, unicode=unicode, latex=latex, prefix=prefix)
    return Unit._make(dim, scale, factor, ((atom, 1),), PRE_ONE, name)


def _power_alias(dim: Dimension, scale: Any, factor: Any, name: str, dispname: str) -> 'Unit | None':
    """Interpret a legacy ``'<symbol>^<int>'`` display name as a power alias."""
    base_symbol, sep, exp = dispname.rpartition('^')
    if not sep or not exp.lstrip('-').isdigit():
        return None
    base = _lookup_symbol(base_symbol)
    if base is None or base._single_atom() is None:
        return None
    p = base ** int(exp)
    if not (p.dim == dim and p.scale == scale and p.factor == factor):
        return None
    return Unit._make(p._dim, p._scale, p._factor, p._terms, p._pre, name)


def _compose(a: Unit, ea: 'Exp', b: 'Unit | None' = None, eb: 'Exp' = 0) -> Unit:
    """Return ``a ** ea * b ** eb``: the single composition primitive."""
    tb: 'Terms | None' = () if b is None else b._terms
    if a._terms is None or tb is None:
        return _compose_traced(a, ea, b, eb)
    terms = combine_terms(a._terms, ea, tb, eb)
    pre = _pre_pow(a._pre, ea)
    if b is not None and b._pre != PRE_ONE:
        pb = _pre_pow(b._pre, eb)
        pre = (pre[0] + pb[0], pre[1] * pb[1])
    scale, factor = derive_magnitude(terms, pre)
    if type(ea) is int and type(eb) is int:
        dim = a._dim if ea == 1 else _dim_pow(a._dim, ea)
        if b is not None:
            dim = dim * b._dim if eb == 1 else (dim / b._dim if eb == -1 else dim * _dim_pow(b._dim, eb))
    else:
        dim = derive_dim(terms)
    return Unit._make(dim, scale, factor, terms, pre)


def _dim_pow(dim: Dimension, e: Any) -> Dimension:
    return dim ** e


def _pre_pow(pre: 'Pre', e: 'Exp') -> 'Pre':
    if e == 1 or pre == PRE_ONE:
        return pre
    scale, factor = norm_scale(pre[0] * e), pre[1]
    if factor != 1:
        try:
            factor = factor ** (e if type(e) is int else float(e))
        except OverflowError:
            raise ValueError(
                f"unit factor {factor} ** {e} overflows the float range; reduce the exponent."
            ) from None
    return (scale, factor)


def _compose_traced(a: Unit, ea: 'Exp', b: 'Unit | None', eb: 'Exp') -> Unit:
    """Composition when a scale or factor is a JAX tracer (no display terms)."""
    fa = ea if type(ea) is int else float(ea)
    dim, scale, factor = _dim_pow(a.dim, fa), a.scale * fa, a.factor ** fa
    if b is not None:
        fb = eb if type(eb) is int else float(eb)
        dim, scale, factor = dim * _dim_pow(b.dim, fb), scale + b.scale * fb, factor * b.factor ** fb
    return Unit._make(dim, scale, factor, None, PRE_ONE)


def _same_value(u: Unit, dim: Dimension, scale: Any, factor: Any) -> bool:
    return u.dim == dim and u.scale == scale and u.factor == factor


def _from_legacy(dim, scale, factor, name, dispname, is_fullname, display_parts) -> Unit:
    """Build a Unit from the pre-0.4 ``(name, dispname, display_parts)`` fields.

    The display is recovered from the registry (by name, then dispname),
    then from the legacy display parts, then by parsing the display name.
    The given ``dim``/``scale``/``factor`` are authoritative: a candidate
    whose value differs is rejected.
    """
    if is_fullname:
        for s in (name, dispname):
            if isinstance(s, str) and s:
                unit = _lookup_symbol(s)
                if unit is not None and _same_value(unit, dim, scale, factor):
                    return unit
    candidates = []
    if display_parts:
        try:
            unit = UNITLESS
            for part_name, part_disp, exp in display_parts:
                part = _lookup_symbol(part_name) or _lookup_symbol(part_disp) or parse_unit(part_disp)
                unit = unit * part ** exp
            candidates.append(unit)
        except (ValueError, TypeError):
            pass
    if isinstance(dispname, str) and dispname:
        try:
            candidates.append(parse_unit(dispname))
        except (ValueError, TypeError):
            pass
    for unit in candidates:
        if _same_value(unit, dim, scale, factor):
            return unit
    if is_fullname and isinstance(name, str) and name:
        return _named_unit(dim, scale, factor, name, dispname if isinstance(dispname, str) and dispname else name)
    return _anonymous_unit(dim, scale, factor)


UNITLESS = Unit()
"""
The canonical unitless (dimensionless) unit.

``UNITLESS`` is a singleton-like :class:`Unit` with no physical dimensions,
a scale of 0, a base of 10, and a factor of 1.  It is returned by default
when a :class:`Unit` is constructed with no arguments, and is used
internally as the neutral element of unit arithmetic.

.. code-block:: python

    >>> import saiunit as u
    >>> u.UNITLESS.is_unitless
    True
    >>> u.UNITLESS.dim.is_dimensionless
    True
"""
