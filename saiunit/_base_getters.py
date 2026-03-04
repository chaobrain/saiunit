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

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np

from ._base_dimension import (
    Dimension,
    DIMENSIONLESS,
    DimensionMismatchError,
    UnitMismatchError,
    _is_tracer,
)
from ._base_unit import Unit, UNITLESS
from ._misc import set_module_as, maybe_custom_array

__all__ = [
    'is_dimensionless',
    'is_unitless',
    'get_dim',
    'get_unit',
    'get_mantissa',
    'get_magnitude',
    'display_in_unit',
    'split_mantissa_unit',
    'maybe_decimal',
    'fail_for_dimension_mismatch',
    'fail_for_unit_mismatch',
    'assert_quantity',
    'have_same_dim',
    'has_same_unit',
    'unit_scale_align_to_first',
    'array_with_unit',
    'is_scalar_type',
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_quantity(array):
    from ._base_quantity import Quantity
    array = maybe_custom_array(array)
    if isinstance(array, Quantity):
        return array
    else:
        return Quantity(array)


def _assert_not_quantity(array):
    from ._base_quantity import Quantity
    if isinstance(array, Quantity):
        raise ValueError('Input array should not be an instance of Quantity.')


@contextmanager
def change_printoption(**kwargs):
    """
    Temporarily change the numpy print options.

    :param kwargs: The new print options.
    """
    old_printoptions = np.get_printoptions()
    try:
        np.set_printoptions(**kwargs)
        yield
    finally:
        np.set_printoptions(**old_printoptions)


def _short_str(arr):
    """
    Return a short string representation of an array, suitable for use in
    error messages.
    """
    from ._base_quantity import Quantity
    arr = arr.mantissa if isinstance(arr, Quantity) else arr
    if not isinstance(arr, (jax.core.Tracer, jax.core.ShapedArray, jax.ShapeDtypeStruct)):
        arr = np.asanyarray(arr)
    with change_printoption(edgeitems=2, threshold=5):
        arr_string = str(arr)
    return arr_string


# ---------------------------------------------------------------------------
# Getter functions
# ---------------------------------------------------------------------------

@set_module_as('saiunit')
def get_dim(obj) -> Dimension:
    """
    Return the dimension of any object that has them.

    Slightly more general than `Array.dimensions` because it will
    return `DIMENSIONLESS` if the object is of number type but not a `Quantity`
    (e.g. a ``float`` or ``int``).

    Parameters
    ----------
    obj : object
        The object to check.  Can be a `Quantity`, `Unit`, `Dimension`,
        or a plain numeric type.

    Returns
    -------
    dim : Dimension
        The physical dimensions of `obj`.

    See Also
    --------
    get_unit : Return the unit of an object.
    get_mantissa : Return the mantissa (numeric value) of an object.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.get_dim(1.0 * u.mV)
        metre ** 2 * kilogram * second ** -3 * amp ** -1
        >>> u.get_dim(5.0)
        1
        >>> u.get_dim(u.volt)
        metre ** 2 * kilogram * second ** -3 * amp ** -1
    """
    from ._base_quantity import Quantity
    obj = maybe_custom_array(obj)
    if isinstance(obj, Unit):
        return obj.dim
    if isinstance(obj, Dimension):
        return obj
    if isinstance(obj, Quantity):
        return obj.dim
    try:
        return Quantity(obj).dim
    except TypeError:
        raise TypeError(f"Object of type {type(obj)} does not have a dim")


@set_module_as('saiunit')
def get_unit(obj) -> 'Unit':
    """
    Return the unit of any object that has them.

    Parameters
    ----------
    obj : object
        The object to check.  Can be a `Quantity`, `Unit`, or a plain
        numeric type.

    Returns
    -------
    unit : Unit
        The physical unit of `obj`.

    See Also
    --------
    get_dim : Return the dimension of an object.
    get_mantissa : Return the mantissa (numeric value) of an object.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.get_unit(3.0 * u.mV)
        mV
        >>> u.get_unit(5.0)
        1
        >>> u.get_unit(u.volt)
        V
    """
    from ._base_quantity import Quantity
    obj = maybe_custom_array(obj)
    if isinstance(obj, Unit):
        return obj
    if isinstance(obj, Quantity):
        return obj.unit
    try:
        return Quantity(obj).unit
    except TypeError:
        raise TypeError(f"Object of type {type(obj)} does not have a unit")


@set_module_as('saiunit')
def get_mantissa(obj):
    """
    Return the mantissa of a `Quantity` or a number.

    For a `Quantity` the numeric value (mantissa) is returned, stripping
    the unit.  For plain numbers or arrays the input is returned unchanged.

    Parameters
    ----------
    obj : object
        The object to check.  Can be a `Quantity` or any numeric type.

    Returns
    -------
    mantissa : float or array_like
        The mantissa of `obj`.

    See Also
    --------
    get_dim : Return the dimension of an object.
    get_unit : Return the unit of an object.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.get_mantissa(3.0 * u.mV)
        3.0
        >>> u.get_mantissa(5.0)
        5.0
    """
    obj = maybe_custom_array(obj)
    try:
        return obj.mantissa
    except AttributeError:
        return obj


get_magnitude = get_mantissa


def split_mantissa_unit(obj):
    """
    Split a `Quantity` into its mantissa and unit.

    Plain numeric values are treated as unitless quantities.

    Parameters
    ----------
    obj : object
        The object to split.  Can be a `Quantity` or a plain numeric type.

    Returns
    -------
    mantissa : float or array_like
        The mantissa of `obj`.
    unit : Unit
        The physical unit of `obj`.

    See Also
    --------
    get_mantissa : Return only the mantissa.
    get_unit : Return only the unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> m, u = u.split_mantissa_unit(3.0 * u.mV)
        >>> float(m)
        3.0
        >>> u
        mV
        >>> m, u = u.split_mantissa_unit(5.0)
        >>> float(m)
        5.0
        >>> u == u.UNITLESS
        True
    """
    obj = _to_quantity(obj)
    return obj.mantissa, obj.unit


# ---------------------------------------------------------------------------
# Comparison / validation
# ---------------------------------------------------------------------------

def have_same_dim(obj1, obj2) -> bool:
    """
    Test if two values have the same dimensions.

    Parameters
    ----------
    obj1 : {Quantity, Unit, array-like, number}
        The first value.
    obj2 : {Quantity, Unit, array-like, number}
        The second value.

    Returns
    -------
    same : bool
        ``True`` if `obj1` and `obj2` have the same dimensions.

    See Also
    --------
    has_same_unit : Check whether two objects share the same unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.have_same_dim(1.0 * u.mV, 2.0 * u.volt)
        True
        >>> u.have_same_dim(1.0 * u.mV, 2.0 * u.second)
        False
        >>> u.have_same_dim(1.0, 2.0)
        True
    """
    # If dimensions are consistently created using get_or_create_dimensions,
    #   the fast "is" comparison should always return the correct result.
    #   To be safe, we also do an equals comparison in case it fails. This
    #   should only add a small amount of unnecessary computation for cases in
    #   which this function returns False which very likely leads to a
    #   DimensionMismatchError anyway.
    obj1 = maybe_custom_array(obj1)
    obj2 = maybe_custom_array(obj2)
    dim1 = get_dim(obj1)
    dim2 = get_dim(obj2)
    return (dim1 is dim2) or (dim1 == dim2)


@set_module_as('saiunit')
def has_same_unit(obj1, obj2) -> bool:
    """
    Check whether two objects have the same unit.

    Unlike `have_same_dim`, this function also checks that the *scale*
    matches (e.g. ``mV`` and ``V`` have the same dimension but different
    units).

    Parameters
    ----------
    obj1 : {Quantity, Unit, array-like, number}
        The first value.
    obj2 : {Quantity, Unit, array-like, number}
        The second value.

    Returns
    -------
    same : bool
        ``True`` if `obj1` and `obj2` have the same unit.

    See Also
    --------
    have_same_dim : Check whether two objects share the same dimension.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.has_same_unit(1.0 * u.mV, 2.0 * u.mV)
        True
        >>> u.has_same_unit(1.0 * u.mV, 2.0 * u.volt)
        False
        >>> u.has_same_unit(1.0, 2.0)
        True
    """
    obj1 = maybe_custom_array(obj1)
    obj2 = maybe_custom_array(obj2)
    unit1 = get_unit(obj1)
    unit2 = get_unit(obj2)
    return unit1 == unit2


@set_module_as('saiunit')
def fail_for_dimension_mismatch(
    obj1, obj2=None, error_message=None, **error_arrays
):
    """
    Compare the dimensions of two objects.

    If the dimensions do not match a `DimensionMismatchError` is raised.

    Parameters
    ----------
    obj1 : {array-like, Quantity}
        The first object to compare.
    obj2 : {array-like, Quantity}, optional
        The second object to compare.  If ``None``, assume it to be
        dimensionless.
    error_message : str, optional
        An error message that is used in the `DimensionMismatchError`.
        May contain ``{name}`` placeholders that will be filled from
        *error_arrays*.
    **error_arrays : dict mapping str to Quantity
        Arrays in this dictionary will be converted using the ``_short_str``
        helper and inserted into *error_message*.

    Returns
    -------
    dim1 : Dimension
        The physical dimension of `obj1`.
    dim2 : Dimension
        The physical dimension of `obj2` (or ``DIMENSIONLESS``).

    Raises
    ------
    DimensionMismatchError
        If the dimensions of `obj1` and `obj2` do not match (or, if `obj2`
        is ``None``, when `obj1` is not dimensionless).

    Notes
    -----
    Implements special checking for ``0``, treating it as having "any
    dimensions".

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> d1, d2 = u.fail_for_dimension_mismatch(3.0 * u.volt, 5.0 * u.volt)
        >>> d1 == d2
        True
    """
    dim1 = get_dim(obj1)
    if obj2 is None:
        dim2 = DIMENSIONLESS
    else:
        dim2 = get_dim(obj2)

    if dim1 is not dim2 and not (dim1 is None or dim2 is None):
        if dim1 == dim2:
            return dim1, dim2

        if error_message is None:
            error_message = "Dimension mismatch"
        else:
            error_arrays = {
                name: _short_str(q) for name, q in error_arrays.items()
            }
            error_message = error_message.format(**error_arrays)
        # If we are comparing an object to a specific unit, we don't want to
        # restate this unit (it is probably mentioned in the text already)
        if obj2 is None or isinstance(obj2, (Dimension, Unit)):
            raise DimensionMismatchError(error_message, dim1)
        else:
            raise DimensionMismatchError(error_message, dim1, dim2)
    else:
        return dim1, dim2


@set_module_as('saiunit')
def fail_for_unit_mismatch(
    obj1, obj2=None, error_message=None, **error_arrays
) -> 'tuple[Unit, Unit]':
    """
    Compare the units of two objects.

    If the units do not share the same dimension a `UnitMismatchError` is
    raised.

    Parameters
    ----------
    obj1 : {array-like, Quantity}
        The first object to compare.
    obj2 : {array-like, Quantity}, optional
        The second object to compare.  If ``None``, assume it to be
        unitless.
    error_message : str, optional
        An error message used in the `UnitMismatchError`.  May contain
        ``{name}`` placeholders filled from *error_arrays*.
    **error_arrays : dict mapping str to Quantity
        Arrays in this dictionary will be converted using the ``_short_str``
        helper and inserted into *error_message*.

    Returns
    -------
    unit1 : Unit
        The physical unit of `obj1`.
    unit2 : Unit
        The physical unit of `obj2` (or ``UNITLESS``).

    Raises
    ------
    UnitMismatchError
        If the dimensions of `obj1` and `obj2` do not match (or, if `obj2`
        is ``None``, when `obj1` is not unitless).

    Notes
    -----
    Implements special checking for ``0``, treating it as having "any
    dimensions".

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u1, u2 = u.fail_for_unit_mismatch(3.0 * u.mV, 5.0 * u.volt)
        >>> u1.has_same_dim(u2)
        True
    """
    unit1 = get_unit(obj1)
    if obj2 is None:
        unit2 = UNITLESS
    else:
        unit2 = get_unit(obj2)

    if unit1.has_same_dim(unit2):
        return unit1, unit2

    if error_message is None:
        error_message = "Unit mismatch"
    else:
        error_arrays = {
            name: _short_str(q) for name, q in error_arrays.items()
        }
        error_message = error_message.format(**error_arrays)
    # If we are comparing an object to a specific unit, we don't want to
    # restate this unit (it is probably mentioned in the text already)
    if obj2 is None or isinstance(obj2, (Dimension, Unit)):
        raise UnitMismatchError(error_message, unit1)
    else:
        raise UnitMismatchError(error_message, unit1, unit2)


# ---------------------------------------------------------------------------
# Display / conversion
# ---------------------------------------------------------------------------

@set_module_as('saiunit')
def display_in_unit(
    x: 'jax.typing.ArrayLike | Quantity',
    u: 'Unit' = None,
    precision: int | None = None,
) -> str:
    """
    Display a value in a certain unit with a given precision.

    Returns the canonical ``"value unit"`` format.

    Parameters
    ----------
    x : {Quantity, array-like, number}
        The value to display.
    u : Unit, optional
        The unit to display the value `x` in.  If ``None``, the value's
        own unit is used.
    precision : int, optional
        The number of digits of precision (in the given unit, see Examples).
        If no value is given, numpy's ``get_printoptions`` value is used.

    Returns
    -------
    s : str
        A string representation of `x` in units of `u`.

    See Also
    --------
    Quantity.repr_in_unit

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.display_in_unit(3 * u.volt, u.mvolt)
        '3000. mV'
        >>> u.display_in_unit(123123 * u.msecond, u.second, 2)
        '123.12 s'
    """
    x = _to_quantity(x)
    if u is not None:
        x = x.in_unit(u)
    return x.repr_in_unit(precision=precision)


@set_module_as('saiunit')
def maybe_decimal(
    val: 'Quantity | jax.typing.ArrayLike',
    unit: 'Unit | None' = None
) -> 'jax.Array | Quantity':
    """
    Convert a quantity to a plain number if it is dimensionless.

    If `val` has physical dimensions and no *unit* is provided, the
    original `Quantity` is returned unchanged.

    Parameters
    ----------
    val : {Quantity, array-like, number}
        The value to convert.
    unit : Unit, optional
        If provided, convert `val` to this unit before stripping the unit.

    Returns
    -------
    decimal : float or Quantity
        A plain number when `val` is dimensionless (or convertible via
        *unit*), otherwise the original `Quantity`.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.maybe_decimal(u.Quantity(5.0))
        5.0
        >>> q = 1.0 * u.metre
        >>> u.maybe_decimal(q) == q       # not dimensionless, returned as-is
        True
    """
    valq = _to_quantity(val)
    if valq.dim.is_dimensionless:
        return valq.to_decimal()
    if unit is not None:
        return valq.to_decimal(unit)
    else:
        return val


@set_module_as('saiunit')
def unit_scale_align_to_first(*args) -> 'list[Quantity]':
    """
    Align the units of all arguments to the unit of the first argument.

    All values are re-expressed in the unit of ``args[0]``.

    Parameters
    ----------
    *args : Quantity or array-like
        The values to align.  All values must share the same dimension.

    Returns
    -------
    aligned : list of Quantity
        The values with units aligned to the first one.

    Raises
    ------
    DimensionMismatchError
        If any argument has a different dimension than the first.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> aligned = u.unit_scale_align_to_first(1 * u.mV, 2 * u.volt)
        >>> aligned[0].unit == aligned[1].unit
        True
    """
    from ._base_quantity import Quantity
    if len(args) == 0:
        return args
    args = list(args)
    first_unit = get_unit(args[0])
    if first_unit.is_unitless:
        if not isinstance(args[0], Quantity):
            args[0] = Quantity(args[0])
        for i in range(1, len(args)):
            fail_for_unit_mismatch(args[i], args[0], 'Non-matching unit for function "unit_scale_align_to_first"')
            if not isinstance(args[i], Quantity):
                args[i] = Quantity(args[i])
    else:
        for i in range(1, len(args)):
            args[i] = args[i].in_unit(first_unit)
    return args


def array_with_unit(
    mantissa,
    unit: 'Unit',
    dtype: jax.typing.DTypeLike | None = None
) -> 'Quantity':
    """
    Create a new `Quantity` with the given unit.

    Parameters
    ----------
    mantissa : float or array-like
        The numeric value of the quantity.
    unit : Unit
        The physical unit to attach.
    dtype : dtype, optional
        The data type of the underlying array.

    Returns
    -------
    array : Quantity
        The new `Quantity` object.

    Raises
    ------
    TypeError
        If *unit* is not a `Unit` instance.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.array_with_unit(5.0, u.volt)
        5. * volt
        >>> u.array_with_unit(0.001, u.volt)
        1. * mvolt
    """
    from ._base_quantity import Quantity
    if not isinstance(unit, Unit):
        raise TypeError(f'Expected instance of Unit, but got {unit}')
    return Quantity(mantissa, unit=unit, dtype=dtype)


# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------

@set_module_as('saiunit')
def is_dimensionless(obj: 'Quantity | Unit | Dimension | jax.typing.ArrayLike') -> bool:
    """
    Test if a value is dimensionless or not.

    Parameters
    ----------
    obj : {Quantity, Unit, Dimension, array-like, number}
        The object to check.

    Returns
    -------
    dimensionless : bool
        ``True`` if `obj` is dimensionless.

    See Also
    --------
    is_unitless : Check whether the object is unitless.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.is_dimensionless(5.0)
        True
        >>> u.is_dimensionless(5.0 * u.volt)
        False
        >>> u.is_dimensionless(u.DIMENSIONLESS)
        True
    """
    obj = maybe_custom_array(obj)
    if isinstance(obj, Dimension):
        return obj.is_dimensionless
    return _to_quantity(obj).dim.is_dimensionless


@set_module_as('saiunit')
def is_unitless(obj: 'Quantity | Unit | jax.typing.ArrayLike') -> bool:
    """
    Test if a value is unitless or not.

    Parameters
    ----------
    obj : {Quantity, Unit, array-like, number}
        The object to check.  Must *not* be a `Dimension` instance.

    Returns
    -------
    unitless : bool
        ``True`` if `obj` is unitless.

    Raises
    ------
    TypeError
        If `obj` is a `Dimension` instance (dimensions do not carry a unit).

    See Also
    --------
    is_dimensionless : Check whether the object is dimensionless.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.is_unitless(5.0)
        True
        >>> u.is_unitless(5.0 * u.volt)
        False
    """
    obj = maybe_custom_array(obj)
    if isinstance(obj, Dimension):
        raise TypeError(f"Dimension objects are not unitless or not, but got {obj}")
    return _to_quantity(obj).is_unitless


@set_module_as('saiunit')
def is_scalar_type(obj) -> bool:
    """
    Test whether *obj* is a scalar (0-d) numeric type.

    Returns ``True`` for plain Python scalars (``int``, ``float``) and
    0-d unitless `Quantity` values.  Strings and arrays with ``ndim > 0``
    return ``False``.

    Parameters
    ----------
    obj : object
        The object to check.

    Returns
    -------
    scalar : bool
        ``True`` if `obj` is a scalar that can be interpreted as a
        dimensionless `Quantity`.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.is_scalar_type(5)
        True
        >>> u.is_scalar_type(5.0 * u.volt)
        False
        >>> u.is_scalar_type("hello")
        False
    """
    try:
        return obj.ndim == 0 and is_unitless(obj) and not _is_tracer(obj)
    except AttributeError:
        return jnp.isscalar(obj) and not isinstance(obj, str)


# ---------------------------------------------------------------------------
# Assertion
# ---------------------------------------------------------------------------

@set_module_as('saiunit')
def assert_quantity(
    q: 'Quantity | jax.typing.ArrayLike',
    mantissa: jax.typing.ArrayLike,
    unit: 'Unit' = None
):
    """
    Assert that a `Quantity` has a certain mantissa and unit.

    When *unit* is ``None`` the function checks that `q` is unitless and
    that its numeric value matches *mantissa*.  When *unit* is given the
    function additionally checks that the dimensions agree and that the
    value expressed in *unit* matches *mantissa*.

    Parameters
    ----------
    q : Quantity or array-like
        The quantity to check.
    mantissa : array-like
        The expected numeric value.
    unit : Unit, optional
        The expected unit.

    Raises
    ------
    AssertionError
        If the value or the unit does not match.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.assert_quantity(u.Quantity(1, u.mV), 1, u.mV)
    """
    from ._base_quantity import Quantity
    mantissa = jnp.asarray(mantissa)
    if unit is None:
        if isinstance(q, Quantity):
            assert q.is_unitless, f"Expected a unitless quantity when 'unit' is not given, but got {q}"
            q = q.mantissa
        assert jnp.allclose(q, mantissa, equal_nan=True), f"Values do not match: {q} != {mantissa}"
    else:
        assert isinstance(unit, Unit), f"Expected a Unit, but got {unit}."
        q = _to_quantity(q)
        assert have_same_dim(get_dim(q), unit), f"Dimension mismatch: ({get_dim(q)}) ({get_dim(unit)})"
        if not jnp.allclose(q.to_decimal(unit), mantissa, equal_nan=True):
            raise AssertionError(f"Values do not match: {q.to_decimal(unit)} != {mantissa}")
