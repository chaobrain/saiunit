# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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
    return `DIMENSIONLESS` if the object is of number type but not a `Array`
    (e.g. a `float` or `int`).

    Parameters
    ----------
    obj : `object`
        The object to check.

    Returns
    -------
    dim : Dimension
        The physical dimensions of the `obj`.
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
    obj : `object`
        The object to check.

    Returns
    -------
    unit : Unit
        The physical unit of the `obj`.
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
    Return the mantissa of a Quantity or a number.

    Parameters
    ----------
    obj : `object`
        The object to check.

    Returns
    -------
    mantissa : `float` or `array_like`
        The mantissa of the `obj`.


    See Also
    --------
    get_dim
    get_unit
    """
    obj = maybe_custom_array(obj)
    try:
        return obj.mantissa
    except AttributeError:
        return obj


get_magnitude = get_mantissa


def split_mantissa_unit(obj):
    """
    Split a Quantity into its mantissa and unit.

    Parameters
    ----------
    obj : `object`
        The object to check.

    Returns
    -------
    mantissa : `float` or `array_like`
        The mantissa of the `obj`.
    unit : Unit
        The physical unit of the `obj`.
    """
    obj = _to_quantity(obj)
    return obj.mantissa, obj.unit


# ---------------------------------------------------------------------------
# Comparison / validation
# ---------------------------------------------------------------------------

def have_same_dim(obj1, obj2) -> bool:
    """Test if two values have the same dimensions.

    Parameters
    ----------
    obj1, obj2 : {`Array`, array-like, number}
        The values of which to compare the dimensions.

    Returns
    -------
    same : `bool`
        ``True`` if `obj1` and `obj2` have the same dimensions.
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

    Parameters
    ----------
    obj1, obj2 : {`Array`, array-like, number}
        The values of which to compare the units.

    Returns
    -------
    same : `bool`
        ``True`` if `obj1` and `obj2` have the same unit.
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

    Parameters
    ----------
    obj1, obj2 : {array-like, `Array`}
        The object to compare. If `obj2` is ``None``, assume it to be
        dimensionless
    error_message : str, optional
        An error message that is used in the UnitMismatchError
    error_arrays : dict mapping str to `Array`, optional
        Arrays in this dictionary will be converted using the `_short_str`
        helper method and inserted into the ``error_message`` (which should
        have placeholders with the corresponding names). The reason for doing
        this in a somewhat complicated way instead of directly including all the
        details in ``error_messsage`` is that converting large arrays
        to strings can be rather costly and we don't want to do it if no error
        occured.

    Returns
    -------
    dim1, dim2 : Dimension, `Dimension`
        The physical dimensions of the two arguments (so that later code does
        not need to get the dimensions again).

    Raises
    ------
    UnitMismatchError
        If the dimensions of `obj1` and `obj2` do not match (or, if `obj2` is
        ``None``, in case `obj1` is not dimensionsless).

    Notes
    -----
    Implements special checking for ``0``, treating it as having "any
    dimensions".
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
    Compare the dimensions of two objects.

    Parameters
    ----------
    obj1, obj2 : {array-like, `Array`}
        The object to compare. If `obj2` is ``None``, assume it to be
        dimensionless
    error_message : str, optional
        An error message that is used in the UnitMismatchError
    error_arrays : dict mapping str to `Array`, optional
        Arrays in this dictionary will be converted using the `_short_str`
        helper method and inserted into the ``error_message`` (which should
        have placeholders with the corresponding names). The reason for doing
        this in a somewhat complicated way instead of directly including all the
        details in ``error_messsage`` is that converting large arrays
        to strings can be rather costly and we don't want to do it if no error
        occured.

    Returns
    -------
    unit1, unit2 : Unit, Unit
        The physical units of the two arguments (so that later code does
        not need to get the dimensions again).

    Raises
    ------
    UnitMismatchError
        If the dimensions of `obj1` and `obj2` do not match (or, if `obj2` is
        ``None``, in case `obj1` is not dimensionsless).

    Notes
    -----
    Implements special checking for ``0``, treating it as having "any
    dimensions".
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
    x : {`Array`, array-like, number}
        The value to display
    u : {`Array`, `Unit`}
        The unit to display the value `x` in.
    precision : `int`, optional
        The number of digits of precision (in the given unit, see Examples).
        If no value is given, numpy's `get_printoptions` value is used.

    Returns
    -------
    s : `str`
        A string representation of `x` in units of `u`.

    Examples
    --------
    >>> from saiunit import *
    >>> display_in_unit(3 * volt, mvolt)
    '3000. mV'
    >>> display_in_unit(123123 * msecond, second, 2)
    '123.12 s'
    >>> display_in_unit(10 * nS, ohm) # doctest: +NORMALIZE_WHITESPACE
    ...                       # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    DimensionMismatchError: Non-matching unit for method "in_unit",
    dimensions were (m^-2 kg^-1 s^3 A^2) (m^2 kg s^-3 A^-2)

    See Also
    --------
    Array.in_unit
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
    Convert a quantity to a decimal number if it is a dimensionless quantity.

    Parameters
    ----------
    val : {`Array`, array-like, number}
        The value to convert.
    unit: `Unit`, optional
        The base unit maybe used to convert the value to.

    Returns
    -------
    decimal : `float`
        The value as a decimal number.
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
    Align the unit units of all arguments to the first one.

    Parameters
    ----------
    args : sequence of {`Array`, array-like, number}
        The values to align.

    Returns
    -------
    aligned : sequence of {`Array`, array-like, number}
        The values with units aligned to the first one.

    Examples
    --------
    >>> from saiunit import *
    >>> unit_scale_align_to_first(1 * mV, 2 * volt, 3 * uV)
    (1. mV, 2. mV, 3. mV)
    >>> unit_scale_align_to_first(1 * mV, 2 * volt, 3 * uA)
    Traceback (most recent call last):
        ...
    DimensionMismatchError: Non-matching unit for function "align_to_first_unit",
    dimensions were (mV) (V) (A)

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
    Create a new `Array` with the given dimensions. Calls
    `get_or_create_dimension` with the dimension tuple of the `dims`
    argument to make sure that unpickling (which calls this function) does not
    accidentally create new Dimension objects which should instead refer to
    existing ones.

    Parameters
    ----------
    mantissa : `float`
        The floating point value of the array.
    unit: Unit
        The dim dimensions of the array.
    dtype: `dtype`, optional
        The data type of the array.

    Returns
    -------
    array : `Quantity`
        The new `Array` object.

    Examples
    --------
    >>> from saiunit import *
    >>> array_with_unit(0.001, volt)
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
    obj : `object`
        The object to check.

    Returns
    -------
    dimensionless : `bool`
        ``True`` if `obj` is dimensionless.
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
    obj : `object`
        The object to check.

    Returns
    -------
    unitless : `bool`
        ``True`` if `obj` is unitless.
    """
    obj = maybe_custom_array(obj)
    if isinstance(obj, Dimension):
        raise TypeError(f"Dimension objects are not unitless or not, but got {obj}")
    return _to_quantity(obj).is_unitless


@set_module_as('saiunit')
def is_scalar_type(obj) -> bool:
    """
    Tells you if the object is a 1d number type.

    Parameters
    ----------
    obj : `object`
        The object to check.

    Returns
    -------
    scalar : `bool`
        ``True`` if `obj` is a scalar that can be interpreted as a
        dimensionless `Array`.
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
    Assert that a Quantity has a certain mantissa and unit.

    Parameters
    ----------
    q : Quantity
        The Quantity to check.
    mantissa : array-like
        The mantissa to check.
    unit : Unit, optional
        The unit to check.

    Raises
    ------
    AssertionError

    Examples
    --------
    >>> from saiunit import *
    >>> assert_quantity(Quantity(1, mV), 1, mV)
    >>> assert_quantity(Quantity(1, mV), 1)
    Traceback (most recent call last):
      ...
    >>> assert_quantity(Quantity(1, mV), 1, V)
    Traceback (most recent call last):
        ...
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
