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

"""
Unit-aware type annotations for saiunit.

This module provides type aliases and annotation helpers that let you
express physical-unit constraints directly in Python type hints, using
:pep:`593` (``typing.Annotated``).

Quick start
-----------

.. code-block:: python

    import saiunit as u
    from saiunit.typing import QuantityLike, UnitLike

    # Annotate with a specific unit
    def kinetic_energy(m: u.Quantity[u.kilogram], v: u.Quantity[u.meter / u.second]) -> u.Quantity[u.joule]:
        return 0.5 * m * v ** 2

    # Annotate with a physical type (dimension) string
    def travel_time(distance: u.Quantity["length"], speed: u.Quantity["speed"]) -> u.Quantity["time"]:
        return distance / speed

Subscript syntax
~~~~~~~~~~~~~~~~

``Quantity[x]`` is a shorthand that produces a type usable with both
``isinstance`` and ``typing.Annotated``:

* ``Quantity[u.meter]``   — matches any ``Quantity`` with length dimension
* ``Quantity["length"]``  — same, using a physical type string

isinstance support
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    x = 2.0 * u.km
    isinstance(x, u.Quantity[u.meter])    # True  (same dimension)
    isinstance(x, u.Quantity["length"])   # True
    isinstance(x, u.Quantity["mass"])     # False

    from saiunit.typing import PhysicalType
    isinstance(x, PhysicalType("length"))  # True

Pre-built aliases
~~~~~~~~~~~~~~~~~

Common physical types are available as ready-made aliases:

.. code-block:: python

    from saiunit.typing import LENGTH, SPEED, VOLTAGE

    def displacement(v: SPEED, t: saiunit.typing.TIME) -> LENGTH:
        return v * t

Runtime validation
~~~~~~~~~~~~~~~~~~

The :func:`validate_units` decorator checks that ``Quantity`` arguments
match the annotated units/dimensions at call time.

.. code-block:: python

    from saiunit.typing import validate_units

    @validate_units
    def ohms_law(V: u.Quantity[u.volt], R: u.Quantity[u.ohm]) -> u.Quantity[u.amp]:
        return V / R
"""

from __future__ import annotations

import functools
import inspect
from typing import Annotated, Any, Union, get_type_hints

import jax
import numpy as np

from ._base_dimension import Dimension, DIMENSIONLESS

__all__ = [
    # Marker classes
    'PhysicalType',
    'is_physical_type',

    # Core type aliases
    'QuantityLike',
    'UnitLike',
    'DimensionLike',

    # Physical-type annotations (also usable with isinstance)
    'HAS_UNIT',
    'LENGTH',
    'MASS',
    'TIME',
    'CURRENT',
    'TEMPERATURE',
    'SUBSTANCE',
    'LUMINOSITY',
    'FREQUENCY',
    'FORCE',
    'ENERGY',
    'POWER',
    'PRESSURE',
    'CHARGE',
    'VOLTAGE',
    'RESISTANCE',
    'CAPACITANCE',
    'CONDUCTANCE',
    'MAGNETIC_FLUX',
    'MAGNETIC_FIELD',
    'INDUCTANCE',
    'SPEED',
    'ACCELERATION',
    'AREA',
    'VOLUME',
    'DENSITY',
    'DIMENSIONLESS_TYPE',

    # Runtime validation decorator
    'validate_units',
]


# ---------------------------------------------------------------------------
# Mapping from human-readable physical type names to SI dimension exponents
# ---------------------------------------------------------------------------
# Order: [length, mass, time, current, temperature, substance, luminosity]

_PHYSICAL_TYPE_DIMS: dict[str, tuple[float, ...]] = {
    # Base dimensions
    "length": (1, 0, 0, 0, 0, 0, 0),
    "mass": (0, 1, 0, 0, 0, 0, 0),
    "time": (0, 0, 1, 0, 0, 0, 0),
    "current": (0, 0, 0, 1, 0, 0, 0),
    "electric current": (0, 0, 0, 1, 0, 0, 0),
    "temperature": (0, 0, 0, 0, 1, 0, 0),
    "substance": (0, 0, 0, 0, 0, 1, 0),
    "amount of substance": (0, 0, 0, 0, 0, 1, 0),
    "luminosity": (0, 0, 0, 0, 0, 0, 1),
    "luminous intensity": (0, 0, 0, 0, 0, 0, 1),
    "dimensionless": (0, 0, 0, 0, 0, 0, 0),

    # Derived dimensions
    "frequency": (0, 0, -1, 0, 0, 0, 0),
    "force": (1, 1, -2, 0, 0, 0, 0),
    "energy": (2, 1, -2, 0, 0, 0, 0),
    "power": (2, 1, -3, 0, 0, 0, 0),
    "pressure": (-1, 1, -2, 0, 0, 0, 0),
    "charge": (0, 0, 1, 1, 0, 0, 0),
    "voltage": (2, 1, -3, -1, 0, 0, 0),
    "electric potential": (2, 1, -3, -1, 0, 0, 0),
    "resistance": (2, 1, -3, -2, 0, 0, 0),
    "capacitance": (-2, -1, 4, 2, 0, 0, 0),
    "conductance": (-2, -1, 3, 2, 0, 0, 0),
    "magnetic flux": (2, 1, -2, -1, 0, 0, 0),
    "magnetic field": (0, 1, -2, -1, 0, 0, 0),
    "inductance": (2, 1, -2, -2, 0, 0, 0),

    # Common compound dimensions
    "speed": (1, 0, -1, 0, 0, 0, 0),
    "velocity": (1, 0, -1, 0, 0, 0, 0),
    "acceleration": (1, 0, -2, 0, 0, 0, 0),
    "area": (2, 0, 0, 0, 0, 0, 0),
    "volume": (3, 0, 0, 0, 0, 0, 0),
    "density": (-3, 1, 0, 0, 0, 0, 0),
    "momentum": (1, 1, -1, 0, 0, 0, 0),
    "angular velocity": (0, 0, -1, 0, 0, 0, 0),
    "torque": (2, 1, -2, 0, 0, 0, 0),
}


# ---------------------------------------------------------------------------
# PhysicalType — a callable that returns isinstance-compatible type objects
# ---------------------------------------------------------------------------

class _PhysicalTypeMeta(type):
    """Metaclass enabling ``isinstance(quantity, PhysicalType("length"))``."""

    def __instancecheck__(cls, instance):
        from ._base_quantity import Quantity
        if not type.__instancecheck__(Quantity, instance):
            return False
        return instance.dim == cls.dimension

    def __repr__(cls):
        return f"PhysicalType({cls.physical_type!r})"

    def __eq__(cls, other):
        if isinstance(other, _PhysicalTypeMeta):
            return cls.physical_type == other.physical_type
        return NotImplemented

    def __hash__(cls):
        return hash(('PhysicalType', cls.physical_type))


# Cache so that PhysicalType("length") is PhysicalType("length")
_physical_type_cache: dict[str, type] = {}


class PhysicalType:
    """Create a physical type that works with both type annotations and ``isinstance``.

    ``PhysicalType("length")`` returns a class (type) that can be used:

    1. As ``isinstance`` second argument::

        isinstance(5.0 * u.meter, PhysicalType("length"))  # True
        isinstance(5.0 * u.second, PhysicalType("length"))  # False

    2. Inside ``Annotated`` metadata (via ``Quantity["length"]``)::

        def f(x: Quantity["length"]) -> Quantity["time"]: ...

    Parameters
    ----------
    physical_type : str
        A human-readable physical type name such as ``"length"``,
        ``"speed"``, ``"voltage"``, etc.

    Returns
    -------
    type
        A class whose metaclass implements ``__instancecheck__`` to verify
        that a ``Quantity`` has the correct dimension.

    Examples
    --------
    >>> from saiunit.typing import PhysicalType
    >>> import saiunit as u
    >>> pt = PhysicalType("speed")
    >>> pt.physical_type
    'speed'
    >>> isinstance(3.0 * u.meter / u.second, pt)
    True
    >>> isinstance(3.0 * u.meter, pt)
    False
    """

    def __new__(cls, physical_type: str):
        key = physical_type.lower().strip()
        if key not in _PHYSICAL_TYPE_DIMS:
            raise ValueError(
                f"Unknown physical type {physical_type!r}. "
                f"Known types: {', '.join(sorted(_PHYSICAL_TYPE_DIMS))}"
            )

        if key in _physical_type_cache:
            return _physical_type_cache[key]

        from ._base_dimension import get_or_create_dimension
        dim = get_or_create_dimension(_PHYSICAL_TYPE_DIMS[key])

        new_cls = _PhysicalTypeMeta(
            f'PhysicalType_{key}',
            (),
            {
                'physical_type': key,
                'dimension': dim,
            },
        )
        _physical_type_cache[key] = new_cls
        return new_cls


def is_physical_type(obj) -> bool:
    """Check whether *obj* is a ``PhysicalType``-created class.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        ``True`` if *obj* was created by ``PhysicalType(...)``.
    """
    return isinstance(obj, _PhysicalTypeMeta)


# ---------------------------------------------------------------------------
# _AnnotatedQuantityMeta — metaclass for Quantity[unit] isinstance support
# ---------------------------------------------------------------------------

class _AnnotatedQuantityMeta(type):
    """Metaclass enabling ``isinstance(quantity, Quantity[u.meter])``."""

    def __instancecheck__(cls, instance):
        from ._base_quantity import Quantity
        if not type.__instancecheck__(Quantity, instance):
            return False
        return cls._unit_check(instance)

    def __repr__(cls):
        return cls._repr

    def __eq__(cls, other):
        if isinstance(other, _AnnotatedQuantityMeta):
            return cls._cache_key == other._cache_key
        return NotImplemented

    def __hash__(cls):
        return hash(('AnnotatedQuantity', cls._cache_key))


# Cache for Quantity[x] types
_annotated_quantity_cache: dict = {}


def _make_annotated_quantity_type(item):
    """Create a class for ``Quantity[item]`` that supports isinstance.

    *item* is a ``Unit`` or a physical-type string.
    """
    from ._base_unit import Unit

    # Determine cache key
    if isinstance(item, Unit):
        cache_key = ('unit', id(item), str(item.dim))
    elif isinstance(item, str):
        cache_key = ('str', item.lower().strip())
    else:
        raise TypeError(
            f"Quantity[...] expects a Unit or a physical-type string, "
            f"got {type(item).__name__}: {item!r}"
        )

    if cache_key in _annotated_quantity_cache:
        return _annotated_quantity_cache[cache_key]

    if isinstance(item, Unit):
        target_dim = item.dim

        def _check(q):
            return q.dim == target_dim

        repr_str = f"Quantity[{item!s}]"
        metadata = item  # stored as Annotated metadata
    else:
        pt = PhysicalType(item)  # validates the string
        target_dim = pt.dimension

        def _check(q):
            return q.dim == target_dim

        repr_str = f"Quantity[{item!r}]"
        metadata = pt

    new_cls = _AnnotatedQuantityMeta(
        repr_str,
        (),
        {
            '_unit_check': staticmethod(_check),
            '_repr': repr_str,
            '_cache_key': cache_key,
            '_metadata': metadata,
        },
    )
    _annotated_quantity_cache[cache_key] = new_cls
    return new_cls


# ---------------------------------------------------------------------------
# Core type aliases
# ---------------------------------------------------------------------------

#: Type alias for objects that can be converted to a :class:`Quantity`.
#: Includes plain numbers, NumPy arrays, JAX arrays, and existing Quantity objects.
QuantityLike = Union[
    int,
    float,
    complex,
    np.number,
    np.ndarray,
    jax.Array,
    "Quantity",
]

#: Type alias for objects that can be interpreted as a :class:`Unit`.
#: Includes Unit objects, strings (e.g. ``"mV"``), and ``None`` (meaning unitless).
UnitLike = Union["Unit", str, None]

#: Type alias for objects that can be interpreted as a :class:`Dimension`.
#: Includes Dimension objects and strings (e.g. ``"length"``).
DimensionLike = Union["Dimension", str]


# ---------------------------------------------------------------------------
# Pre-built physical-type aliases
# ---------------------------------------------------------------------------

class _LazyAliases:
    """Lazy alias container to avoid circular imports."""

    _cache: dict[str, Any] = {}

    _MAP = {
        'HAS_UNIT': None,  # special: any Quantity
        'DIMENSIONLESS_TYPE': 'dimensionless',
        'LENGTH': 'length',
        'MASS': 'mass',
        'TIME': 'time',
        'CURRENT': 'current',
        'TEMPERATURE': 'temperature',
        'SUBSTANCE': 'substance',
        'LUMINOSITY': 'luminosity',
        'FREQUENCY': 'frequency',
        'FORCE': 'force',
        'ENERGY': 'energy',
        'POWER': 'power',
        'PRESSURE': 'pressure',
        'CHARGE': 'charge',
        'VOLTAGE': 'voltage',
        'RESISTANCE': 'resistance',
        'CAPACITANCE': 'capacitance',
        'CONDUCTANCE': 'conductance',
        'MAGNETIC_FLUX': 'magnetic flux',
        'MAGNETIC_FIELD': 'magnetic field',
        'INDUCTANCE': 'inductance',
        'SPEED': 'speed',
        'ACCELERATION': 'acceleration',
        'AREA': 'area',
        'VOLUME': 'volume',
        'DENSITY': 'density',
    }

    @classmethod
    def get(cls, attr: str) -> Any:
        if attr not in cls._MAP:
            raise AttributeError(attr)
        if attr not in cls._cache:
            from ._base_quantity import Quantity
            phys = cls._MAP[attr]
            if phys is None:
                # HAS_UNIT: any Quantity with a unit (just Quantity itself)
                cls._cache[attr] = Quantity
            else:
                cls._cache[attr] = Quantity[phys]
        return cls._cache[attr]


# Eagerly populate module-level names so that ``from saiunit.typing import LENGTH``
# works and static type checkers can see the names.
def __getattr__(name: str):
    try:
        return _LazyAliases.get(name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


# ---------------------------------------------------------------------------
# Runtime validation decorator
# ---------------------------------------------------------------------------

def validate_units(func=None, *, strict: bool = False):
    """Decorator that validates ``Quantity`` argument units at call time.

    Inspects the function's type annotations for ``Quantity[...]`` types
    (produced by ``Quantity[u.meter]`` or ``Quantity["length"]``) and
    checks every annotated argument on each call.

    Parameters
    ----------
    func : callable, optional
        The function to decorate. If ``None``, returns a partial decorator
        so that ``@validate_units(strict=True)`` works.
    strict : bool, optional
        If ``True``, require exact unit match (same scale). If ``False``
        (default), only require dimensional compatibility.

    Returns
    -------
    callable
        The decorated function with unit validation.

    Raises
    ------
    saiunit.UnitMismatchError
        If an argument's unit/dimension does not match the annotation.
    TypeError
        If an annotated parameter is not a ``Quantity``.

    Examples
    --------
    >>> import saiunit as u
    >>> from saiunit.typing import validate_units
    >>>
    >>> @validate_units
    ... def ohms_law(V: u.Quantity[u.volt], R: u.Quantity[u.ohm]) -> u.Quantity[u.amp]:
    ...     return V / R
    ...
    >>> ohms_law(5.0 * u.volt, 100.0 * u.ohm)
    Quantity(0.05, "A")
    """
    if func is None:
        return functools.partial(validate_units, strict=strict)

    from ._base_quantity import Quantity
    from ._base_unit import Unit
    from ._base_dimension import UnitMismatchError, DimensionMismatchError

    # Resolve annotations (handles string forward refs).
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        hints = {}

    # Pre-compute which parameters have unit constraints.
    sig = inspect.signature(func)
    _constraints: dict[str, tuple[str, Any]] = {}  # param_name -> (check_kind, ref)

    for param_name, hint in hints.items():
        if param_name == "return":
            continue
        meta = _extract_unit_metadata(hint)
        if meta is not None:
            _constraints[param_name] = meta

    if not _constraints:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        for param_name, (check_kind, ref) in _constraints.items():
            if param_name not in bound.arguments:
                continue
            value = bound.arguments[param_name]
            if value is None:
                continue

            if not isinstance(value, Quantity):
                raise TypeError(
                    f"Argument {param_name!r} of {func.__name__!r} "
                    f"expected a Quantity, got {type(value).__name__}."
                )

            if check_kind == "unit" and strict:
                if not value.unit.has_same_magnitude(ref):
                    raise UnitMismatchError(
                        f"Argument {param_name!r} of {func.__name__!r} "
                        f"expected unit {ref}, got {value.unit}."
                    )
            elif check_kind == "unit":
                if not value.unit.has_same_dim(ref):
                    raise DimensionMismatchError(
                        f"Argument {param_name!r} of {func.__name__!r} "
                        f"expected dimension compatible with {ref}, "
                        f"got {value.unit}."
                    )
            elif check_kind == "physical_type":
                pt = ref
                if value.dim != pt.dimension:
                    raise DimensionMismatchError(
                        f"Argument {param_name!r} of {func.__name__!r} "
                        f"expected physical type {pt.physical_type!r}, "
                        f"got dimension {value.dim}."
                    )

        return func(*args, **kwargs)

    return wrapper


def _extract_unit_metadata(hint) -> tuple[str, Any] | None:
    """Extract unit/physical-type metadata from an annotated type hint.

    Handles both:
    - ``_AnnotatedQuantityMeta`` types (from ``Quantity[u.meter]``)
    - ``typing.Annotated[Quantity, ...]`` types (legacy)
    """
    import typing as _typing
    from ._base_unit import Unit

    # Check for _AnnotatedQuantityMeta (new-style Quantity[...])
    if isinstance(hint, _AnnotatedQuantityMeta):
        meta = hint._metadata
        if isinstance(meta, Unit):
            return ("unit", meta)
        if is_physical_type(meta):
            return ("physical_type", meta)
        return None

    # Check for typing.Annotated (fallback)
    if _typing.get_origin(hint) is Annotated:
        metadata = hint.__metadata__
        for meta in metadata:
            if isinstance(meta, Unit):
                return ("unit", meta)
            if is_physical_type(meta):
                return ("physical_type", meta)
        return None

    return None
