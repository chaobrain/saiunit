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

import re
from copy import deepcopy

import jax

from ._base_dimension import (
    Dimension,
    DIMENSIONLESS,
    _is_tracer,
)

__all__ = [
    'Unit',
    'UNITLESS',
    'add_standard_unit',
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
# Display-parts helpers – canonical, sorted factored-unit representation
# ---------------------------------------------------------------------------

def _assert_same_base(u1, u2):
    if not u1.has_same_base(u2):
        raise TypeError(f"Cannot operate on units with different bases. Got {u1.base} != {u2.base}.")


def _find_standard_unit(
    dim: Dimension,
    base,
    scale,
    factor,
    for_composition: bool = False,
) -> tuple[str | None, str | None, bool, bool]:
    """
    Find a standard unit for the given dimension, base, scale, and factor.

    Parameters
    ----------
    for_composition : bool
        When True, keys that are *ambiguous* (i.e. have >=2 registered
        aliases with distinct display names, e.g. hertz vs becquerel)
        are skipped so that they are never auto-substituted during
        unit arithmetic.  This is detected automatically at
        registration time—no hardcoded list.

    Returns
    -------
    (name, dispname, is_fullname, is_dimensionless)
    """
    if dim == DIMENSIONLESS:
        return None, None, False, True
    if isinstance(base, (int, float)):
        if isinstance(scale, (int, float)):
            if isinstance(factor, (int, float)):
                key = (dim, scale, base, factor)
                if key in _standard_units:
                    if for_composition and key in _ambiguous_keys:
                        pass  # skip – ambiguous, fall through
                    else:
                        u = _standard_units[key]
                        return u.name, u.dispname, True, False

        key = (dim, 0, base, 1.0)
        if key in _standard_units:
            if for_composition and key in _ambiguous_keys:
                return None, None, False, False
            u = _standard_units[key]
            return u.name, u.dispname, False, False
    return None, None, False, False


def _find_a_name(dim: Dimension, base, scale, factor) -> tuple[str | None, bool]:
    if dim == DIMENSIONLESS:
        u_name = f"Unit({base}^{scale})"
        return u_name, False

    if isinstance(base, (int, float)):
        if isinstance(scale, (int, float)):
            if isinstance(factor, (int, float)):
                key = (dim, scale, base, factor)
                if key in _standard_units:
                    u_name = _standard_units[key].name
                    return u_name, True

        if isinstance(factor, (int, float)):
            key = (dim, 0, base, factor)
            if key in _standard_units:
                u_name = _standard_units[key].name
                if factor == 1.:
                    return f"{base}^{scale} * {u_name}", False
                else:
                    return f"{factor} * {base}^{scale} * {u_name}", False

        key = (dim, 0, base, 1.)
        if key in _standard_units:
            u_name = _standard_units[key].name
            if _is_tracer(scale):
                return u_name, False
            else:
                return f"{base}^{scale} * {u_name}", False
    return None, True


_standard_units: 'dict[tuple, Unit]' = {}
_standard_unit_aliases: 'dict[tuple, list[Unit]]' = {}

# ---------------------------------------------------------------------------
# Ambiguous-key detection
#
# A dimension key is "ambiguous" when >=2 registered aliases have
# **different display names** (dispname).  Spelling variants like
# meter/metre share the same dispname ("m") so they are NOT flagged.
# Genuine semantic collisions like hertz/becquerel ("Hz" vs "Bq") ARE
# flagged automatically—no hardcoded list required.
#
# Ambiguous keys are never auto-substituted during unit composition
# (mul / div / pow / reverse) so that e.g. joule/kg never silently
# becomes sievert.
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
    """Pick the preferred alias – deterministic (score, then alpha)."""
    return min(
        units,
        key=lambda u: (
            _standard_unit_preference_score(u),
            u.name.lower() if isinstance(u.name, str) else "",
        ),
    )


def add_standard_unit(u: 'Unit'):
    """
    Register a unit as a standard unit for display purposes.

    Once registered, this unit will be used when formatting quantities
    whose dimensions, scale, base, and factor match.  If multiple units
    are registered for the same key, the preferred alias is selected
    automatically.  Keys with two or more distinct display names are
    flagged as *ambiguous* and will not be auto-substituted during unit
    composition.

    Parameters
    ----------
    u : Unit
        The unit to register.  Its ``base``, ``scale``, and ``factor``
        must all be plain Python ``int`` or ``float`` values (not JAX
        tracers) for registration to take effect.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> my_unit = su.Unit(
        ...     dim=su.joule.dim,
        ...     name='my_energy',
        ...     dispname='myE',
        ...     is_fullname=True,
        ... )
        >>> su.add_standard_unit(my_unit)
    """
    if (
        isinstance(u.base, (int, float)) and
        isinstance(u.scale, (int, float)) and
        isinstance(u.factor, (int, float))
    ):
        key = (u.dim, u.scale, u.base, u.factor)
        aliases = _standard_unit_aliases.setdefault(key, [])
        aliases.append(u)
        _standard_units[key] = _select_preferred_standard_unit(aliases)

        # Auto-detect ambiguity: >=2 distinct display names → ambiguous
        dispnames = {a.dispname for a in aliases if isinstance(a.dispname, str)}
        if len(dispnames) >= 2:
            _ambiguous_keys.add(key)


def _get_display_parts(unit: 'Unit'):
    """Return the display-parts list for *unit*.

    Each element is ``(name, dispname, exponent)``.
    """
    if getattr(unit, '_display_parts', None) is not None:
        return list(unit._display_parts)
    return [(unit.name, unit.dispname, 1)]


def _merge_display_parts(parts_a, parts_b):
    """Merge two part-lists, combine same-name entries, drop zeros, sort."""
    merged: dict[str, tuple] = {}
    for name, disp, exp in list(parts_a) + list(parts_b):
        if name in merged:
            _, old_disp, old_exp = merged[name]
            merged[name] = (name, disp, old_exp + exp)
        else:
            merged[name] = (name, disp, exp)
    result = [(n, d, e) for n, d, e in merged.values() if e != 0]
    # positive exponents first (alphabetical), then negative (alphabetical)
    result.sort(key=lambda x: (0 if x[2] > 0 else 1, x[0].lower()))
    return result


_RE_DISPNAME_EXP = re.compile(r'^(.+)\^(-?\d+(?:\.\d+)?)$')


def _normalise_display_parts(parts):
    """Normalise display parts: decompose stacked exponents, drop zeros, sort.

    If a dispname already contains an exponent (e.g. ``'m^2'``), fold that
    exponent into the part's own exponent so that ``('meter2', 'm^2', 3)``
    becomes ``('meter2', 'm', 6)`` instead of rendering as ``m^2^3``.
    """
    result = []
    for name, disp, exp in parts:
        if exp == 0:
            continue
        m = _RE_DISPNAME_EXP.match(disp)
        if m:
            base_disp = m.group(1)
            inner_exp = float(m.group(2))
            disp = base_disp
            exp = inner_exp * exp
        result.append((name, disp, exp))
    # Merge entries that now share the same base dispname
    merged: dict[str, tuple] = {}
    for name, disp, exp in result:
        if disp in merged:
            _, old_disp, old_exp = merged[disp]
            merged[disp] = (name, disp, old_exp + exp)
        else:
            merged[disp] = (name, disp, exp)
    result = [(n, d, e) for n, d, e in merged.values() if e != 0]
    result.sort(key=lambda x: (0 if x[2] > 0 else 1, x[0].lower()))
    return result


def _fmt_exp(exp):
    """Format an exponent value, using int form when possible."""
    return str(int(exp)) if exp == int(exp) else str(exp)


def _format_display_parts(parts) -> str:
    """Render a parts-list as a canonical unit string.

    The canonical format uses dispname symbols (e.g. ``mV``, ``Hz``),
    ``^`` for exponentiation, `` * `` for multiplication, and `` / ``
    for division.  This single format is both human-readable and
    machine-parseable:

        mV
        J / kg
        nA / cm^2
        mS * nA / cm^2
        m / (kg * s^2)
    """
    if not parts:
        return "1"

    numerator = [(n, d, e) for n, d, e in parts if e > 0]
    denominator = [(n, d, -e) for n, d, e in parts if e < 0]

    def _fmt_term(name, dispname, exp):
        if exp == 1:
            return dispname
        return f"{dispname}^{_fmt_exp(exp)}"

    num_str = " * ".join(_fmt_term(n, d, e) for n, d, e in numerator) if numerator else "1"

    if not denominator:
        return num_str

    if len(denominator) == 1:
        den_str = _fmt_term(*denominator[0])
    else:
        inner = " * ".join(_fmt_term(n, d, e) for n, d, e in denominator)
        den_str = f"({inner})"
    return f"{num_str} / {den_str}"


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

    The unit class also keeps track of various things that were used
    to define it so as to generate a nice string representation of it.
    See below.

    Parameters
    ----------
    dim : Dimension, optional
        The physical dimensions of the unit. Defaults to ``DIMENSIONLESS``.
    scale : array_like, optional
        The scale exponent, e.g. 3 for a "k" (kilo) prefix. Defaults to 0.
    base : array_like, optional
        The base of the exponent, e.g. 10 for SI prefixes. Defaults to 10.
    factor : array_like, optional
        The conversion factor of the unit. Defaults to 1.
    name : str, optional
        The full name of the unit, e.g. ``'volt'``.
    dispname : str, optional
        The display name, e.g. ``'V'``.
    is_fullname : bool, optional
        Whether ``name`` is the canonical full name. Defaults to ``True``.
    display_parts : list of tuple, optional
        Canonical display components for compound units.

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

    **Defining your own**

    It can be useful to define your own units for printing
    purposes. So for example, to define the newton metre, you
    write:

    .. code-block:: python

        >>> import saiunit as su
        >>> Nm = su.newton * su.metre

    You can then do:

    .. code-block:: python

        >>> (1 * Nm).in_unit(Nm)
        '1. N m'

    New "compound units", i.e. units that are composed of other units will be
    automatically registered and from then on used for display. For example,
    imagine you define total conductance for a membrane, and the total area of
    that membrane:

    .. code-block:: python

        >>> import saiunit as su
        >>> conductance = 10. * su.nS
        >>> area = 20000 * su.um ** 2

    If you now ask for the conductance density, you will get an "ugly" display
    in basic SI dimensions, as saiunit does not know of a corresponding unit:

    .. code-block:: python

        >>> conductance / area
        0.5 * metre ** -4 * kilogram ** -1 * second ** 3 * amp ** 2

    By using an appropriate unit once, it will be registered and from then on
    used for display when appropriate:

    .. code-block:: python

        >>> su.usiemens / su.cm ** 2
        usiemens / (cmetre ** 2)
        >>> conductance / area  # same as before, but now knows about uS/cm^2
        50. * usiemens / (cmetre ** 2)

    Note that user-defined units cannot override the standard units (``volt``,
    ``second``, etc.) that are predefined. For example, the unit ``Nm`` has the
    dimensions "length^2 * mass / time^2", and therefore the same dimensions as
    the standard unit ``joule``. The latter will be used for display purposes:

    .. code-block:: python

        >>> 3 * su.joule
        3. * joule
        >>> 3 * Nm
        3. * joule

    Examples
    --------
    Create a simple unit:

    .. code-block:: python

        >>> import saiunit as su
        >>> su.volt
        Unit("V")
        >>> su.mvolt
        Unit("mV")

    Combine units:

    .. code-block:: python

        >>> import saiunit as su
        >>> su.volt / su.amp
        Unit("V / A")

    """

    __module__ = "saiunit"
    __slots__ = ["_dim", "_base", "_scale", "_factor", "_dispname", "_name", "is_fullname", "_hash", "_display_parts"]
    __array_priority__ = 1000

    def __init__(
        self,
        dim: Dimension = None,
        scale: jax.typing.ArrayLike = 0,
        base: jax.typing.ArrayLike = 10.,
        factor: jax.typing.ArrayLike = 1.,
        name: str = None,
        dispname: str = None,
        is_fullname: bool = True,
        display_parts=None,
    ):
        # The base for this unit (as the base of the exponent), i.e.
        # a base of 10 means 10^3, for a "k" prefix.
        self._base = base

        # The scale for this unit (as the integer exponent of 10), i.e.
        # a scale of 3 means base^3, for a "k" prefix.
        self._scale = scale

        # The factor for this unit (as the conversion factor), i.e.
        # a factor of cal = 4.18400 means 1 cal = 4.18400 J,
        # where 4.18400 is the factor.
        self._factor = factor

        # The physical unit dimensions of this unit
        if dim is None:
            dim = DIMENSIONLESS
        if not isinstance(dim, Dimension):
            raise TypeError(f'Expected instance of Dimension, but got {dim}')
        self._dim = dim

        # The name of this unit
        if name is None:
            is_fullname = False
            if dim == DIMENSIONLESS:
                name = f"Unit({base}^{scale})"
            else:
                name = dim.__repr__()
                dispname = dim.__str__()
        self._name = name

        # The display name of this unit
        self._dispname = (name if dispname is None else dispname)

        # whether the name is the full name
        self.is_fullname = is_fullname

        # cached hash (computed lazily)
        self._hash = None

        # Canonical display components: list of (name, dispname, exponent).
        # None for simple (non-compound) units.
        self._display_parts = display_parts

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

            >>> import saiunit as su
            >>> su.volt.factor
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

            >>> import saiunit as su
            >>> su.kvolt.base
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

            >>> import saiunit as su
            >>> su.mvolt.scale
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
            The absolute magnitude of the unit.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as su
            >>> su.mvolt.magnitude
            0.001
            >>> su.kvolt.magnitude
            1000.0
        """
        # magnitude = factor * base ** scale
        return self.factor * self.base ** self.scale

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

            >>> import saiunit as su
            >>> su.volt.dim
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

            >>> import saiunit as su
            >>> su.UNITLESS.is_unitless
            True
            >>> su.volt.is_unitless
            False
        """
        return self.dim.is_dimensionless and self.scale == 0 and self.factor == 1.0

    @property
    def should_display_unit(self) -> bool:
        """
        Whether the unit should be shown in formatted output.

        Returns ``True`` for all non-unitless units, and also for
        dimensionless units that carry a meaningful registered name
        (e.g. radian, steradian).

        Returns
        -------
        bool
            ``True`` if the unit should be displayed, ``False`` otherwise.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as su
            >>> su.volt.should_display_unit
            True
            >>> su.UNITLESS.should_display_unit
            False
        """
        if not self.is_unitless:
            return True
        # Dimensionless but with a registered display name (e.g. rad, sr)
        return self.is_fullname and self._canonical_str() != '1'

    @property
    def name(self):
        """
        Return the full name of the unit.

        Returns
        -------
        str or None
            The full name of the unit (e.g. ``'volt'``, ``'mvolt'``),
            or ``None`` if no name was assigned.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as su
            >>> su.volt.name
            'volt'
            >>> su.mvolt.name
            'mvolt'
        """
        return self._name

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

        The display name is the short symbol used when rendering the
        unit in string output (e.g. ``'V'`` for volt, ``'mV'`` for
        millivolt).

        Returns
        -------
        str or None
            The display name of the unit.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as su
            >>> su.volt.dispname
            'V'
            >>> su.mvolt.dispname
            'mV'
        """
        return self._dispname

    @dispname.setter
    def dispname(self, dispname):
        raise NotImplementedError(
            "Cannot set the dispname of a Unit object directly,"
            "Please create a new Unit object with the dispname you want."
        )

    def factorless(self) -> 'Unit':
        """
        Return a copy of this Unit with the factor set to 1.

        Returns
        -------
        Unit
            A new Unit object with the factor set to 1.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as su
            >>> u = su.Unit.create(su.Dimension(kg=1), 'pound', 'lb', factor=0.453592)
            >>> u.factor
            0.453592
            >>> u.factorless().factor
            1.0
        """
        # using standard units
        key = (self.dim, self.scale, self.base, 1.)
        if key in _standard_units:
            return _standard_units[key]

        # using temporary units
        name, dispname, is_fullname, dimless = _find_standard_unit(self.dim, self.base, self.scale, 1.0)
        return Unit(
            dim=self.dim,
            scale=self.scale,
            base=self.base,
            factor=1.,
            name=name,
            dispname=dispname,
            is_fullname=is_fullname,
        )

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

            >>> import saiunit as su
            >>> u = su.volt.copy()
            >>> u == su.volt
            True
            >>> u is su.volt
            False
        """
        return Unit(
            dim=self.dim,
            scale=self.scale,
            base=self.base,
            factor=self.factor,
            name=self.name,
            dispname=self.dispname,
            is_fullname=self.is_fullname,
        )

    def __deepcopy__(self, memodict):
        return Unit(
            dim=self.dim.__deepcopy__(memodict),
            scale=deepcopy(self.scale),
            base=deepcopy(self.base),
            factor=deepcopy(self.factor),
            name=deepcopy(self.name),
            dispname=deepcopy(self.dispname),
            is_fullname=deepcopy(self.is_fullname),
        )

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(
                (
                    self.dim,
                    self.factor,
                    self.base,
                    self.scale,
                    self.name,
                    self.dispname,
                )
            )
        return self._hash

    def has_same_magnitude(self, other: 'Unit') -> bool:
        """
        Whether this Unit has the same magnitude as another Unit.

        Two units have the same magnitude when they share the same
        ``scale``, ``base``, and ``factor``.

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

            >>> import saiunit as su
            >>> su.mvolt.has_same_magnitude(su.mamp)
            True
            >>> su.mvolt.has_same_magnitude(su.volt)
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

            >>> import saiunit as su
            >>> su.volt.has_same_base(su.amp)
            True
            >>> su.volt.has_same_base(su.mvolt)
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

            >>> import saiunit as su
            >>> su.volt.has_same_dim(su.mvolt)
            True
            >>> su.volt.has_same_dim(su.amp)
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
        scale: int = 0,
        base: float = 10.,
        factor: float = 1.,
    ) -> 'Unit':
        """
        Create a new named unit.

        Parameters
        ----------
        dim : Dimension
            The dimensions of the unit.
        name : `str`
            The full name of the unit, e.g. ``'volt'``
        dispname : `str`
            The display name, e.g. ``'V'``
        scale : int, optional
            The scale of this unit as an exponent of 10, e.g. -3 for a unit that
            is 1/1000 of the base scale. Defaults to 0 (i.e. a base unit).
        base: float, optional
            The base for this unit (as the base of the exponent), i.e.
            a base of 10 means 10^3, for a "k" prefix. Defaults to 10.
        factor: float, optional
            The factor for this unit (as the conversion factor), e.g.
            a factor of 1 cal = 4.18400 J, where 4.18400 is the factor.
            Defaults to 1.

        Returns
        -------
        u : Unit
            The new unit.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as su
            >>> from saiunit import Dimension
            >>> energy_dim = su.joule.dim
            >>> cal = su.Unit.create(energy_dim, 'calorie', 'cal', factor=4.184)
            >>> cal
            Unit("cal")
        """
        u = Unit(
            dim=dim,
            scale=scale,
            base=base,
            factor=factor,
            name=name,
            dispname=dispname,
            is_fullname=True,
        )
        add_standard_unit(u)
        return u

    @staticmethod
    def create_scaled_unit(baseunit: 'Unit', scalefactor: str) -> 'Unit':
        """
        Create a scaled unit from a base unit.

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

            >>> import saiunit as su
            >>> uvolt = su.Unit.create_scaled_unit(su.volt, 'u')
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
        dispname = scalefactor + baseunit.dispname
        scale = _siprefixes[scalefactor] + baseunit.scale
        u = Unit(
            dim=baseunit.dim,
            name=name,
            dispname=dispname,
            scale=scale,
            base=baseunit.base,
            is_fullname=True,
        )
        add_standard_unit(u)
        return u

    def _canonical_str(self) -> str:
        """Return the canonical display string for this unit.

        Uses dispname symbols (``mV``, ``Hz``, ``kg``), ``^`` for
        exponentiation, `` * `` for multiplication, and `` / `` for
        division.  The result is both human-readable and
        machine-parseable.
        """
        if self._display_parts is not None:
            return _format_display_parts(self._display_parts)
        if self.is_fullname:
            return self.dispname
        if self.dim.is_dimensionless:
            if self.scale == 0 and self.factor == 1.:
                return '1'
            elif self.factor == 1.:
                return f'{self.base}^{_fmt_exp(self.scale)}'
            elif self.scale == 0:
                return str(self.factor)
            else:
                return f'{self.factor} * {self.base}^{_fmt_exp(self.scale)}'
        # Anonymous unit — build a descriptive string from components
        if self.factor == 1.:
            if self.scale == 0:
                return f'{self.dispname}'
            else:
                return f'{self.base}^{self.scale} * {self.dispname}'
        else:
            if self.scale == 0:
                return f'{self.factor} * {self.dispname}'
            else:
                return f'{self.factor} * {self.base}^{self.scale} * {self.dispname}'

    def __repr__(self) -> str:
        s = self._canonical_str()
        return f"Unit(\"{s}\")"

    def __str__(self) -> str:
        return self._canonical_str()

    def __mul__(self, other) -> 'Unit | Quantity':
        # self * other
        if isinstance(other, Unit):
            _assert_same_base(self, other)
            scale = self.scale + other.scale
            dim = self.dim * other.dim
            factor = self.factor * other.factor

            # Dimensionless → no compound display
            if dim == DIMENSIONLESS:
                return Unit(dim, scale=scale, base=self.base, factor=factor)

            # Both named → deterministic compound via display_parts
            if self.is_fullname and other.is_fullname:
                parts = _merge_display_parts(
                    _get_display_parts(self),
                    _get_display_parts(other),
                )
                canonical = _format_display_parts(parts)
                return Unit(
                    dim, scale=scale, base=self.base, factor=factor,
                    name=canonical, dispname=canonical,
                    is_fullname=True,
                    display_parts=parts,
                )

            # Fallback: standard-unit lookup
            name, dispname, is_fullname, _ = _find_standard_unit(
                dim, self.base, scale, factor
            )
            return Unit(
                dim, scale=scale, base=self.base, factor=factor,
                name=name, dispname=dispname,
                is_fullname=is_fullname,
            )

        elif isinstance(other, Dimension):
            raise TypeError(f"unit {self} cannot multiply by a Dimension {other}.")

        else:
            from ._base_quantity import Quantity
            if isinstance(other, Quantity):
                return Quantity(other.mantissa, unit=(self * other.unit))
            return Quantity(other, unit=self)

    def __rmul__(self, other) -> 'Unit | Quantity':
        # other * self
        if isinstance(other, Unit):
            return other.__mul__(self)

        from ._base_quantity import Quantity
        if isinstance(other, Quantity):
            return Quantity(other.mantissa, unit=(other.unit * self))
        return Quantity(other, unit=self)

    def __imul__(self, other):
        raise NotImplementedError("Units cannot be modified in-place")

    def __div__(self, other) -> 'Unit':
        # self / other
        if isinstance(other, Unit):
            _assert_same_base(self, other)
            scale = self.scale - other.scale
            dim = self.dim / other.dim
            factor = self.factor / other.factor

            # Dimensionless → no compound display
            if dim == DIMENSIONLESS:
                return Unit(dim, scale=scale, base=self.base, factor=factor)

            # Both named → deterministic compound via display_parts
            if self.is_fullname and other.is_fullname:
                other_parts = [(n, d, -e) for n, d, e in _get_display_parts(other)]
                parts = _merge_display_parts(
                    _get_display_parts(self), other_parts,
                )
                canonical = _format_display_parts(parts)
                return Unit(
                    dim, base=self.base, scale=scale, factor=factor,
                    name=canonical, dispname=canonical,
                    is_fullname=True,
                    display_parts=parts,
                )

            # Fallback: standard-unit lookup
            name, dispname, is_fullname, _ = _find_standard_unit(
                dim, self.base, scale, factor
            )
            return Unit(
                dim, base=self.base, scale=scale, factor=factor,
                name=name, dispname=dispname,
                is_fullname=is_fullname,
            )

        else:
            raise TypeError(f"unit {self} cannot divide by a non-unit {other}")

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
        inverted factor, and reciprocal dimensions.

        Returns
        -------
        Unit
            A new Unit representing the reciprocal of this unit.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as su
            >>> su.second.reverse()
            Unit("Hz")
            >>> su.metre.reverse()
            Unit("1 / m")
        """
        dim = self.dim ** -1
        scale = -self.scale
        factor = 1. / self.factor

        # Standard-unit lookup — allowed for reverse() because it is a
        # single-operand transform where the preference system correctly
        # picks hertz over becquerel, etc.
        name, dispname, is_fullname, dimless = _find_standard_unit(
            dim, self.base, scale, factor
        )
        if is_fullname:
            return Unit(
                dim, base=self.base, scale=scale, factor=factor,
                name=name, dispname=dispname,
                is_fullname=True,
            )

        # Build from display_parts (negate exponents)
        if self.is_fullname:
            parts = [(n, d, -e) for n, d, e in _get_display_parts(self)]
            parts = _normalise_display_parts(parts)
            canonical = _format_display_parts(parts)
            return Unit(
                dim, base=self.base, scale=scale, factor=factor,
                name=canonical, dispname=canonical,
                is_fullname=True,
                display_parts=parts,
            )

        return Unit(
            dim, base=self.base, scale=scale, factor=factor,
            name=name, dispname=dispname,
            is_fullname=is_fullname,
        )

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
            dim = self.dim ** other
            scale = self.scale * other
            factor = self.factor ** other

            if dim == DIMENSIONLESS:
                return Unit(dim, scale=scale, base=self.base, factor=factor)

            # Named source → build from display_parts (multiply exponents).
            # This avoids ambiguous standard-unit aliases (e.g. m^3→kl,
            # (m/s)^2→Gy) and keeps display consistent with __mul__/__div__.
            if self.is_fullname:
                src_parts = _get_display_parts(self)
                parts = [(n, d, e * other) for n, d, e in src_parts]
                parts = _normalise_display_parts(parts)
                canonical = _format_display_parts(parts)
                return Unit(
                    dim, base=self.base, scale=scale, factor=factor,
                    name=canonical, dispname=canonical,
                    is_fullname=True,
                    display_parts=parts,
                )

            # Fallback: standard-unit lookup (for anonymous units)
            name, dispname, is_fullname, dimless = _find_standard_unit(
                dim, self.base, scale, factor
            )
            return Unit(
                dim, base=self.base, scale=scale, factor=factor,
                name=name, dispname=dispname,
                is_fullname=is_fullname,
            )
        else:
            raise TypeError(
                f"unit cannot perform an exponentiation (unit ** other) with a non-scalar, "
                f"since one unit cannot contain multiple units. \n"
                f"But we got unit={self}, other={other}"
            )

    def __ipow__(self, other, modulo=None):
        raise NotImplementedError("Units cannot be modified in-place")

    def __add__(self, other: 'Unit') -> 'Unit':
        # self + other
        if not isinstance(other, Unit):
            raise TypeError(f"Expected a Unit, but got {other}")
        if self.has_same_dim(other):
            if self.has_same_magnitude(other):
                return self.copy()
            else:
                raise TypeError(f"Units {self} and {other} have different units.")
        else:
            raise TypeError(f"Units {self} and {other} have different dimensions.")

    def __radd__(self, oc: 'Unit') -> 'Unit':
        return self.__add__(oc)

    def __iadd__(self, other):
        raise NotImplementedError("Units cannot be modified in-place")

    def __sub__(self, other: 'Unit') -> 'Unit':
        # self - other
        if not isinstance(other, Unit):
            raise TypeError(f"Expected a Unit, but got {other}")
        if self.has_same_dim(other):
            if self.has_same_magnitude(other):
                return self.copy()
            else:
                raise TypeError(f"Units {self} and {other} have different units.")
        else:
            raise TypeError(f"Units {self} and {other} have different dimensions.")

    def __rsub__(self, oc: 'Unit') -> 'Unit':
        return self.__sub__(oc)

    def __isub__(self, other):
        raise NotImplementedError("Units cannot be modified in-place")

    def __mod__(self, oc):
        raise NotImplementedError("Units cannot be performed modulo")

    def __rmod__(self, oc):
        raise NotImplementedError("Units cannot be performed modulo")

    def __imod__(self, other):
        raise NotImplementedError("Units cannot be modified in-place")

    def __eq__(self, other) -> bool:
        if isinstance(other, Unit):
            return (
                (other.dim == self.dim) and
                (other.scale == self.scale) and
                (other.base == self.base) and
                (other.factor == self.factor)
            )
        else:
            return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __abs__(self) -> 'Unit':
        """Return the unit itself — units are always non-negative."""
        return self

    def __reduce__(self):
        # For pickling
        return (
            _to_unit,
            (
                self.dim,
                self.scale,
                self.base,
                self.factor,
                self.name,
                self.dispname,
                self.is_fullname
            )
        )


def _to_unit(*args):
    """Private pickle reconstruction shim for Unit.
    """
    return Unit(*args)


_to_unit.__module__ = 'saiunit._base_unit'

UNITLESS = Unit()
"""
The canonical unitless (dimensionless) unit.

``UNITLESS`` is a singleton-like :class:`Unit` with no physical dimensions,
a scale of 0, a base of 10, and a factor of 1.  It is returned by default
when a :class:`Unit` is constructed with no arguments, and is used
internally as the neutral element of unit arithmetic.

.. code-block:: python

    >>> import saiunit as su
    >>> su.UNITLESS.is_unitless
    True
    >>> su.UNITLESS.dim.is_dimensionless
    True
"""
