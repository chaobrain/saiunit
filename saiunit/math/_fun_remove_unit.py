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
from __future__ import annotations

from typing import (Union, Optional, Sequence)

import jax
import jax.numpy as jnp

from saiunit._base_getters import get_unit
from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as, maybe_custom_array, maybe_custom_array_tree

__all__ = [
    # math funcs remove unit (unary)
    'iscomplexobj', 'heaviside', 'signbit', 'sign', 'bincount', 'digitize', 'get_promote_dtypes',

    # logic funcs (unary)
    'all', 'any', 'logical_not',

    # logic funcs (binary)
    'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
    'array_equal', 'isclose', 'allclose', 'logical_and',
    'logical_or', 'logical_xor', "alltrue", 'sometrue',

    # indexing
    'argsort', 'argmax', 'argmin', 'nanargmax', 'nanargmin', 'argwhere',
    'nonzero', 'flatnonzero', 'searchsorted', 'count_nonzero', 'diag_indices_from',
]


# math funcs remove unit (unary)
# ------------------------------


@set_module_as('saiunit.math')
def get_promote_dtypes(
    *args: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity | jax.Array | Sequence[jax.Array | Quantity]]:
    """
    Promote the data types of the inputs to a common type.

    Parameters
    ----------
    *args : array_like or Quantity
        The arrays whose dtypes should be promoted.

    Returns
    -------
    promoted : dtype
        The promoted common dtype.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.get_promote_dtypes(jnp.float32, jnp.int32)
        dtype('float32')
    """
    args = maybe_custom_array_tree(args)
    return jnp.promote_types(*jax.tree.leaves(args))


def _fun_remove_unit_unary(func, x, *args, **kwargs):
    x = maybe_custom_array(x)
    args, kwargs = maybe_custom_array_tree((args, kwargs))
    if isinstance(x, Quantity):
        # x = x.factorless()
        return func(x.mantissa, *args, **kwargs)
    else:
        return func(x, *args, **kwargs)


@set_module_as('saiunit.math')
def iscomplexobj(
    x: Union[jax.typing.ArrayLike, Quantity],
) -> bool:
    """
    Return True if x is a complex type or an array of complex numbers.

    Units are stripped before the check is performed.

    Parameters
    ----------
    x : array_like or Quantity
        Input array or Quantity.

    Returns
    -------
    out : bool
        ``True`` if ``x`` is a complex type or an array of complex numbers.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.iscomplexobj(jnp.array([1.0, 2.0]))
        False
        >>> u.math.iscomplexobj(jnp.array([1.0 + 2.0j]))
        True
    """
    return _fun_remove_unit_unary(jnp.iscomplexobj, x)


@set_module_as('saiunit.math')
def heaviside(
    x1: Union[Quantity, jax.Array],
    x2: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.Array]:
    """
    Compute the Heaviside step function.

    The unit is stripped from ``x1`` before evaluation. ``x2`` must be
    dimensionless (it is the value returned where ``x1 == 0``).

    Parameters
    ----------
    x1 : array_like or Quantity
        Input values.
    x2 : array_like or Quantity
        The value of the function when ``x1`` is zero. Must be
        dimensionless if given as a Quantity.

    Returns
    -------
    out : jax.Array
        The Heaviside step function applied to ``x1`` with half-value
        ``x2``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.heaviside(jnp.array([-1.0, 0.0, 1.0]),
        ...                   jnp.array([0.5, 0.5, 0.5]))
        Array([0. , 0.5, 1. ], dtype=float32)
    """
    x1 = maybe_custom_array(x1)
    x2 = maybe_custom_array(x2)
    x1 = x1.mantissa if isinstance(x1, Quantity) else x1
    if isinstance(x2, Quantity):
        if not x2.is_unitless:
            raise TypeError(
                f'heaviside requires "x2" (the step value) to be dimensionless, '
                f'but got x2 with unit={x2.unit}. Strip the unit from x2 before calling heaviside.'
            )
        x2 = x2.mantissa
    return _fun_remove_unit_unary(jnp.heaviside, x1, x2)


@set_module_as('saiunit.math')
def signbit(x: Union[jax.typing.ArrayLike, Quantity]) -> jax.Array:
    """
    Return element-wise True where the sign bit is set (less than zero).

    Units are stripped before the check is performed.

    Parameters
    ----------
    x : array_like or Quantity
        The input value(s).

    Returns
    -------
    result : jax.Array of bool
        Boolean array indicating where the sign bit is set.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.signbit(jnp.array([-2.0, 0.0, 3.0]))
        Array([ True, False, False], dtype=bool)
        >>> q = jnp.array([-1.0, 1.0]) * u.meter
        >>> u.math.signbit(q)
        Array([ True, False], dtype=bool)
    """
    return _fun_remove_unit_unary(jnp.signbit, x)


@set_module_as('saiunit.math')
def sign(x: Union[jax.typing.ArrayLike, Quantity]) -> jax.Array:
    """
    Return the sign of each element in the input array.

    Units are stripped before the sign is computed. Returns -1, 0, or +1.

    Parameters
    ----------
    x : array_like or Quantity
        Input values.

    Returns
    -------
    y : jax.Array
        The sign of ``x``. Contains -1 for negative, 0 for zero, and
        +1 for positive elements.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.sign(jnp.array([-5.0, 0.0, 3.0]))
        Array([-1.,  0.,  1.], dtype=float32)
        >>> q = jnp.array([-2.0, 0.0, 4.0]) * u.second
        >>> u.math.sign(q)
        Array([-1.,  0.,  1.], dtype=float32)
    """
    return _fun_remove_unit_unary(jnp.sign, x)


@set_module_as('saiunit.math')
def bincount(
    x: Union[jax.typing.ArrayLike, Quantity],
    weights: Optional[jax.typing.ArrayLike] = None,
    minlength: int = 0,
    *,
    length: Optional[int] = None
) -> jax.Array:
    """
    Count number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest value in
    `x`. If `minlength` is specified, there will be at least this number
    of bins in the output array (though it will be longer if necessary,
    depending on the contents of `x`).
    Each bin gives the number of occurrences of its index value in `x`.
    If `weights` is specified the input array is weighted by it, i.e. if a
    value ``n`` is found at position ``i``, ``out[n] += weight[i]`` instead
    of ``out[n] += 1``.

    Parameters
    ----------
    x : array_like, Quantity, 1 dimension, nonnegative ints
      Input array.
    weights : array_like, optional
      Weights, array of the same shape as `x`.
    minlength : int, optional
      A minimum number of bins for the output array.

    Returns
    -------
    out : jax.Array of int
        The result of binning the input array.
        The length of ``out`` is equal to ``max(x) + 1``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.bincount(jnp.array([0, 1, 1, 2, 2, 2]))
        Array([1, 2, 3], dtype=int32)
    """
    return _fun_remove_unit_unary(jnp.bincount, x, weights=weights, minlength=minlength, length=length)


@set_module_as('saiunit.math')
def digitize(
    x: Union[jax.typing.ArrayLike, Quantity],
    bins: Union[jax.typing.ArrayLike, Quantity],
    right: bool = False
) -> jax.Array:
    """
    Return the indices of the bins to which each value in input array belongs.

    =========  =============  ============================
    `right`    order of bins  returned index `i` satisfies
    =========  =============  ============================
    ``False``  increasing     ``bins[i-1] <= x < bins[i]``
    ``True``   increasing     ``bins[i-1] < x <= bins[i]``
    ``False``  decreasing     ``bins[i-1] > x >= bins[i]``
    ``True``   decreasing     ``bins[i-1] >= x > bins[i]``
    =========  =============  ============================

    If values in `x` are beyond the bounds of `bins`, 0 or ``len(bins)`` is
    returned as appropriate.

    Parameters
    ----------
    x : array_like, Quantity
      Input array to be binned. Prior to NumPy 1.10.0, this array had to
      be 1-dimensional, but can now have any shape.
    bins : array_like, Quantity
      Array of bins. It has to be 1-dimensional and monotonic.
    right : bool, optional
      Indicating whether the intervals include the right or the left bin
      edge. Default behavior is (right==False) indicating that the interval
      does not include the right edge. The left bin end is open in this
      case, i.e., bins[i-1] <= x < bins[i] is the default behavior for
      monotonically increasing bins.

    Returns
    -------
    indices : jax.Array of int
        Output array of bin indices, same shape as ``x``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> x = jnp.array([0.5, 1.5, 2.5])
        >>> bins = jnp.array([0.0, 1.0, 2.0, 3.0])
        >>> u.math.digitize(x, bins)
        Array([1, 2, 3], dtype=int32)
    """
    x = maybe_custom_array(x)
    bins = maybe_custom_array(bins)
    if isinstance(x, Quantity) and isinstance(bins, Quantity):
        bins = bins.in_unit(x.unit).mantissa
        x = x.mantissa
    elif isinstance(x, Quantity):
        if not x.is_unitless:
            raise TypeError(
                f'digitize requires "x" to be dimensionless when "bins" is a plain array, '
                f'but got x with unit={x.unit}. '
                f'Either pass a Quantity for bins with matching units, or strip the unit from x.'
            )
        x = x.mantissa
    elif isinstance(bins, Quantity):
        if not bins.is_unitless:
            raise TypeError(
                f'digitize requires "bins" to be dimensionless when "x" is a plain array, '
                f'but got bins with unit={bins.unit}. '
                f'Either pass a Quantity for x with matching units, or strip the unit from bins.'
            )
        bins = bins.mantissa
    return jnp.digitize(x, bins, right=right)


def _fun_logic_unary(func, x, *args, **kwargs):
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        if not x.is_unitless:
            raise TypeError(
                f'{func.__name__} requires a dimensionless input, '
                f'but got x with unit={x.unit}. Strip the unit from x before calling {func.__name__}.'
            )
    return func(x, *args, **kwargs)


@set_module_as('saiunit.math')
def all(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    keepdims: bool = False,
    where: Optional[jax.Array] = None
) -> Union[bool, jax.Array]:
    """
    Test whether all array elements along a given axis evaluate to True.

    The input must be dimensionless; a ``TypeError`` is raised if ``x``
    carries physical units.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be dimensionless if it is a Quantity.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed.
        The default (``axis=None``) reduces over all dimensions.
    keepdims : bool, optional
        If True, reduced axes are kept as dimensions with size one.
    where : array_like of bool, optional
        Elements to include in the check.

    Returns
    -------
    all : jax.Array or bool
        Boolean result of the AND reduction.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.all(jnp.array([True, True, True]))
        Array(True, dtype=bool)
        >>> u.math.all(jnp.array([True, False, True]))
        Array(False, dtype=bool)
        >>> u.math.all(jnp.array([[True, False], [True, True]]), axis=1)
        Array([False,  True], dtype=bool)
    """
    return _fun_logic_unary(jnp.all, x, axis=axis, keepdims=keepdims, where=where)


@set_module_as('saiunit.math')
def any(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    keepdims: bool = False,
    where: Optional[jax.Array] = None
) -> Union[bool, jax.Array]:
    """
    Test whether any array element along a given axis evaluates to True.

    The input must be dimensionless; a ``TypeError`` is raised if ``x``
    carries physical units.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be dimensionless if it is a Quantity.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical OR reduction is performed.
        The default (``axis=None``) reduces over all dimensions.
    keepdims : bool, optional
        If True, reduced axes are kept as dimensions with size one.
    where : array_like of bool, optional
        Elements to include in the check.

    Returns
    -------
    any : jax.Array or bool
        Boolean result of the OR reduction.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.any(jnp.array([False, False, True]))
        Array(True, dtype=bool)
        >>> u.math.any(jnp.array([False, False, False]))
        Array(False, dtype=bool)
    """
    return _fun_logic_unary(jnp.any, x, axis=axis, keepdims=keepdims, where=where)


@set_module_as('saiunit.math')
def logical_not(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    """
    Compute the truth value of NOT x element-wise.

    The input must be dimensionless; a ``TypeError`` is raised if ``x``
    carries physical units.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be dimensionless if it is a Quantity.

    Returns
    -------
    out : jax.Array or bool
        Boolean result of the NOT operation applied element-wise.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.logical_not(jnp.array([True, False, True]))
        Array([False,  True, False], dtype=bool)
    """
    return _fun_logic_unary(jnp.logical_not, x)


alltrue = all
sometrue = any


# logic funcs (binary)
# --------------------


def _fun_logic_binary(func, x, y, *args, **kwargs):
    x = maybe_custom_array(x)
    y = maybe_custom_array(y)
    args, kwargs = maybe_custom_array_tree((args, kwargs))
    if isinstance(x, Quantity) and isinstance(y, Quantity):
        # x = x.factorless()
        # y = y.factorless()
        return func(x.mantissa, y.in_unit(x.unit).mantissa, *args, **kwargs)
    elif isinstance(x, Quantity):
        # x = x.factorless()
        if not x.is_unitless:
            raise TypeError(
                f'{func.__name__} requires "x" to be dimensionless when "y" is a plain array, '
                f'but got x with unit={x.unit}. '
                f'Either pass a Quantity for y with matching units, or strip the unit from x.'
            )
        return func(x.mantissa, y, *args, **kwargs)
    elif isinstance(y, Quantity):
        # y = y.factorless()
        if not y.is_unitless:
            raise TypeError(
                f'{func.__name__} requires "y" to be dimensionless when "x" is a plain array, '
                f'but got y with unit={y.unit}. '
                f'Either pass a Quantity for x with matching units, or strip the unit from y.'
            )
        return func(x, y.mantissa, *args, **kwargs)
    else:
        return func(x, y, *args, **kwargs)


@set_module_as('saiunit.math')
def equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    Return ``(x == y)`` element-wise.

    When both ``x`` and ``y`` are Quantities, ``y`` is converted to the
    unit of ``x`` before comparison. A ``TypeError`` is raised if only
    one operand has units.

    Parameters
    ----------
    x : array_like or Quantity
        First input array.
    y : array_like or Quantity
        Second input array. Must be broadcastable with ``x``.

    Returns
    -------
    out : jax.Array of bool
        Element-wise equality comparison.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.equal(jnp.array([1, 2, 3]), jnp.array([1, 0, 3]))
        Array([ True, False,  True], dtype=bool)
        >>> a = jnp.array([1.0, 2.0]) * u.meter
        >>> b = jnp.array([1.0, 2.0]) * u.meter
        >>> u.math.equal(a, b)
        Array([ True,  True], dtype=bool)
    """
    return _fun_logic_binary(jnp.equal, x, y, *args, **kwargs)


@set_module_as('saiunit.math')
def not_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    Return ``(x != y)`` element-wise.

    When both ``x`` and ``y`` are Quantities, ``y`` is converted to the
    unit of ``x`` before comparison.

    Parameters
    ----------
    x : array_like or Quantity
        First input array.
    y : array_like or Quantity
        Second input array. Must be broadcastable with ``x``.

    Returns
    -------
    out : jax.Array of bool
        Element-wise inequality comparison.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.not_equal(jnp.array([1, 2, 3]), jnp.array([1, 0, 3]))
        Array([False,  True, False], dtype=bool)
    """
    return _fun_logic_binary(jnp.not_equal, x, y, *args, **kwargs)


@set_module_as('saiunit.math')
def greater(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    Return ``(x > y)`` element-wise.

    When both ``x`` and ``y`` are Quantities, ``y`` is converted to the
    unit of ``x`` before comparison.

    Parameters
    ----------
    x : array_like or Quantity
        First input array.
    y : array_like or Quantity
        Second input array. Must be broadcastable with ``x``.

    Returns
    -------
    out : jax.Array of bool
        Element-wise greater-than comparison.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.greater(jnp.array([3, 2, 1]), jnp.array([1, 2, 3]))
        Array([ True, False, False], dtype=bool)
        >>> a = jnp.array([2.0, 1.0]) * u.meter
        >>> b = jnp.array([1.0, 2.0]) * u.meter
        >>> u.math.greater(a, b)
        Array([ True, False], dtype=bool)
    """
    return _fun_logic_binary(jnp.greater, x, y, *args, **kwargs)


@set_module_as('saiunit.math')
def greater_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[
    bool, jax.Array]:
    """
    Return ``(x >= y)`` element-wise.

    When both ``x`` and ``y`` are Quantities, ``y`` is converted to the
    unit of ``x`` before comparison.

    Parameters
    ----------
    x : array_like or Quantity
        First input array.
    y : array_like or Quantity
        Second input array. Must be broadcastable with ``x``.

    Returns
    -------
    out : jax.Array of bool
        Element-wise greater-than-or-equal comparison.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.greater_equal(jnp.array([3, 2, 1]), jnp.array([1, 2, 3]))
        Array([ True,  True, False], dtype=bool)
    """
    return _fun_logic_binary(jnp.greater_equal, x, y, *args, **kwargs)


@set_module_as('saiunit.math')
def less(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    Return ``(x < y)`` element-wise.

    When both ``x`` and ``y`` are Quantities, ``y`` is converted to the
    unit of ``x`` before comparison.

    Parameters
    ----------
    x : array_like or Quantity
        First input array.
    y : array_like or Quantity
        Second input array. Must be broadcastable with ``x``.

    Returns
    -------
    out : jax.Array of bool
        Element-wise less-than comparison.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.less(jnp.array([1, 2, 3]), jnp.array([3, 2, 1]))
        Array([ True, False, False], dtype=bool)
    """
    return _fun_logic_binary(jnp.less, x, y, *args, **kwargs)


@set_module_as('saiunit.math')
def less_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[
    bool, jax.Array]:
    """
    Return ``(x <= y)`` element-wise.

    When both ``x`` and ``y`` are Quantities, ``y`` is converted to the
    unit of ``x`` before comparison.

    Parameters
    ----------
    x : array_like or Quantity
        First input array.
    y : array_like or Quantity
        Second input array. Must be broadcastable with ``x``.

    Returns
    -------
    out : jax.Array of bool
        Element-wise less-than-or-equal comparison.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.less_equal(jnp.array([1, 2, 3]), jnp.array([3, 2, 1]))
        Array([ True,  True, False], dtype=bool)
    """
    return _fun_logic_binary(jnp.less_equal, x, y, *args, **kwargs)


@set_module_as('saiunit.math')
def array_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[
    bool, jax.Array]:
    """
    Return True if two arrays have the same shape and elements.

    When both inputs are Quantities, ``y`` is converted to the unit of
    ``x`` before comparison.

    Parameters
    ----------
    x : array_like or Quantity
        First input array.
    y : array_like or Quantity
        Second input array.

    Returns
    -------
    out : bool
        True if the arrays are equal.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.array_equal(jnp.array([1, 2]), jnp.array([1, 2]))
        Array(True, dtype=bool)
        >>> u.math.array_equal(jnp.array([1, 2]), jnp.array([1, 3]))
        Array(False, dtype=bool)
    """
    return _fun_logic_binary(jnp.array_equal, x, y, *args, **kwargs)


@set_module_as('saiunit.math')
def isclose(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    rtol: float | Quantity = None,
    atol: float | Quantity = None,
    equal_nan: bool = False
) -> Union[bool, jax.Array]:
    """
    Returns a boolean array where two arrays are element-wise equal within a
    tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    Parameters
    ----------
    x, y : array_like, Quantity
      Input arrays to compare.
    rtol : float, Quantity
      The relative tolerance parameter (see Notes).
    atol : float, Quantity
      The absolute tolerance parameter (see Notes).
    equal_nan : bool
      Whether to compare NaN's as equal.  If True, NaN's in `a` will be
      considered equal to NaN's in `b` in the output array.

    Returns
    -------
    out : jax.Array of bool
        Boolean array where ``x`` and ``y`` are equal within tolerance.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.isclose(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0001]))
        Array([ True,  True], dtype=bool)
    """
    x = maybe_custom_array(x)
    y = maybe_custom_array(y)
    rtol = maybe_custom_array(rtol)
    atol = maybe_custom_array(atol)
    unit = get_unit(x)
    if isinstance(x, Quantity) and isinstance(y, Quantity):
        y = y.in_unit(x.unit).mantissa
        x = x.mantissa
    elif isinstance(x, Quantity):
        if not x.is_unitless:
            raise TypeError(
                f'isclose requires "x" to be dimensionless when "y" is a plain array, '
                f'but got x with unit={x.unit}. '
                f'Either pass a Quantity for y with matching units, or strip the unit from x.'
            )
        x = x.mantissa
    elif isinstance(y, Quantity):
        if not y.is_unitless:
            raise TypeError(
                f'isclose requires "y" to be dimensionless when "x" is a plain array, '
                f'but got y with unit={y.unit}. '
                f'Either pass a Quantity for x with matching units, or strip the unit from y.'
            )
        y = y.mantissa
    if rtol is None:
        rtol = 1e-5 * unit
    if atol is None:
        atol = 1e-8 * unit
    atol = Quantity(atol).in_unit(unit).mantissa
    rtol = Quantity(rtol).in_unit(unit).mantissa
    return _fun_logic_binary(jnp.isclose, x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)


@set_module_as('saiunit.math')
def allclose(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    rtol: float | Quantity = None,
    atol: float | Quantity = None,
    equal_nan: bool = False
) -> Union[bool, jax.Array]:
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    NaNs are treated as equal if they are in the same place and if
    ``equal_nan=True``.  Infs are treated as equal if they are in the same
    place and of the same sign in both arrays.

    Parameters
    ----------
    x, y : array_like, Quantity
      Input arrays to compare.
    rtol : float
      The relative tolerance parameter (see Notes).
    atol : float
      The absolute tolerance parameter (see Notes).
    equal_nan : bool
      Whether to compare NaN's as equal.  If True, NaN's in `a` will be
      considered equal to NaN's in `b` in the output array.

    Returns
    -------
    allclose : bool
        True if the two arrays are element-wise equal within tolerance.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.allclose(jnp.array([1.0, 2.0]),
        ...                  jnp.array([1.0, 2.0]))
        Array(True, dtype=bool)
    """
    x = maybe_custom_array(x)
    y = maybe_custom_array(y)
    rtol = maybe_custom_array(rtol)
    atol = maybe_custom_array(atol)
    unit = get_unit(x)
    if isinstance(x, Quantity) and isinstance(y, Quantity):
        y = y.in_unit(x.unit)
        x_val = x.mantissa
        y_val = y.mantissa
    elif isinstance(x, Quantity):
        if not x.is_unitless:
            raise TypeError(
                f'allclose requires "x" to be dimensionless when "y" is a plain array, '
                f'but got x with unit={x.unit}. '
                f'Either pass a Quantity for y with matching units, or strip the unit from x.'
            )
        x_val = x.mantissa
        y_val = y
    elif isinstance(y, Quantity):
        if not y.is_unitless:
            raise TypeError(
                f'allclose requires "y" to be dimensionless when "x" is a plain array, '
                f'but got y with unit={y.unit}. '
                f'Either pass a Quantity for x with matching units, or strip the unit from y.'
            )
        y_val = y.mantissa
        x_val = x
    else:
        x_val = x
        y_val = y
    if rtol is None:
        rtol = 1e-5 * unit
    if atol is None:
        atol = 1e-8 * unit
    rtol = Quantity(rtol).in_unit(unit).mantissa
    atol = Quantity(atol).in_unit(unit).mantissa
    return jnp.allclose(x_val, y_val, rtol=rtol, atol=atol, equal_nan=equal_nan)


@set_module_as('saiunit.math')
def logical_and(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    Compute the truth value of ``x AND y`` element-wise.

    When both inputs are Quantities, ``y`` is converted to the unit of
    ``x`` before the operation.

    Parameters
    ----------
    x : array_like or Quantity
        First input array.
    y : array_like or Quantity
        Second input array. Must be broadcastable with ``x``.

    Returns
    -------
    out : jax.Array of bool
        Boolean AND result.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.logical_and(jnp.array([True, False]),
        ...                     jnp.array([True, True]))
        Array([ True, False], dtype=bool)
    """
    return _fun_logic_binary(jnp.logical_and, x, y, *args, **kwargs)


@set_module_as('saiunit.math')
def logical_or(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    Compute the truth value of ``x OR y`` element-wise.

    When both inputs are Quantities, ``y`` is converted to the unit of
    ``x`` before the operation.

    Parameters
    ----------
    x : array_like or Quantity
        First input array.
    y : array_like or Quantity
        Second input array. Must be broadcastable with ``x``.

    Returns
    -------
    out : jax.Array of bool
        Boolean OR result.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.logical_or(jnp.array([True, False]),
        ...                    jnp.array([False, False]))
        Array([ True, False], dtype=bool)
    """
    return _fun_logic_binary(jnp.logical_or, x, y, *args, **kwargs)


@set_module_as('saiunit.math')
def logical_xor(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    Compute the truth value of ``x XOR y`` element-wise.

    When both inputs are Quantities, ``y`` is converted to the unit of
    ``x`` before the operation.

    Parameters
    ----------
    x : array_like or Quantity
        First input array.
    y : array_like or Quantity
        Second input array. Must be broadcastable with ``x``.

    Returns
    -------
    out : jax.Array of bool
        Boolean XOR result.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.logical_xor(jnp.array([True, False]),
        ...                     jnp.array([True, True]))
        Array([False,  True], dtype=bool)
    """
    return _fun_logic_binary(jnp.logical_xor, x, y, *args, **kwargs)


# ----------------------
# Indexing functions
# ----------------------


@set_module_as('saiunit.math')
def argsort(
    a: Union[jax.typing.ArrayLike, Quantity],
    axis: Optional[int] = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> jax.Array:
    """
    Return the indices that would sort an array or Quantity.

    Units are stripped before sorting.

    Parameters
    ----------
    a : array_like or Quantity
        Array or Quantity to be sorted.
    axis : int or None, optional
        Axis along which to sort. Default is -1 (last axis).
        If None, the array is flattened first.
    kind : None, optional
        Sorting algorithm. Unused in JAX.
    order : None, optional
        Unused in JAX.
    stable : bool, optional
        Whether to use a stable sort. Default is True.
    descending : bool, optional
        Whether to sort in descending order. Default is False.

    Returns
    -------
    indices : jax.Array
        Array of indices that would sort the input.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.argsort(jnp.array([3.0, 1.0, 2.0]))
        Array([1, 2, 0], dtype=int32)
        >>> q = jnp.array([3.0, 1.0, 2.0]) * u.meter
        >>> u.math.argsort(q)
        Array([1, 2, 0], dtype=int32)
    """
    return _fun_remove_unit_unary(jnp.argsort,
                                  a,
                                  axis=axis,
                                  kind=kind,
                                  order=order,
                                  stable=stable,
                                  descending=descending)


@set_module_as('saiunit.math')
def argmax(
    a: Union[jax.typing.ArrayLike, Quantity],
    axis: Optional[int] = None,
) -> jax.Array:
    """
    Return the index of the maximum value along an axis.

    Units are stripped before finding the maximum.

    Parameters
    ----------
    a : array_like or Quantity
        Input data.
    axis : int or None, optional
        Axis along which to operate. By default the flattened input is
        used.

    Returns
    -------
    index : jax.Array
        Index of the maximum value.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.argmax(jnp.array([1.0, 3.0, 2.0]))
        Array(1, dtype=int32)
        >>> q = jnp.array([1.0, 3.0, 2.0]) * u.meter
        >>> u.math.argmax(q)
        Array(1, dtype=int32)
    """
    return _fun_remove_unit_unary(jnp.argmax, a, axis=axis)


@set_module_as('saiunit.math')
def argmin(
    a: Union[jax.typing.ArrayLike, Quantity],
    axis: Optional[int] = None,
    keepdims: Optional[bool] = None
) -> jax.Array:
    """
    Return the index of the minimum value along an axis.

    Units are stripped before finding the minimum.

    Parameters
    ----------
    a : array_like or Quantity
        Input data.
    axis : int or None, optional
        Axis along which to operate. By default the flattened input is
        used.
    keepdims : bool or None, optional
        If True, reduced axes are kept with size one.

    Returns
    -------
    index : jax.Array
        Index of the minimum value.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.argmin(jnp.array([3.0, 1.0, 2.0]))
        Array(1, dtype=int32)
    """
    return _fun_remove_unit_unary(jnp.argmin, a, axis=axis, keepdims=keepdims)


@set_module_as('saiunit.math')
def nanargmax(
    a: Union[jax.typing.ArrayLike, Quantity],
    axis: int = None,
    keepdims: bool = False
) -> jax.Array:
    """
    Return the index of the maximum value, ignoring NaNs.

    Units are stripped before finding the maximum.

    Parameters
    ----------
    a : array_like or Quantity
        Input data.
    axis : int or None, optional
        Axis along which to operate. By default the flattened input is
        used.
    keepdims : bool, optional
        If True, reduced axes are kept with size one.

    Returns
    -------
    index : jax.Array
        Index of the maximum value (NaNs ignored).

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.nanargmax(jnp.array([1.0, jnp.nan, 3.0]))
        Array(2, dtype=int32)
    """
    return _fun_remove_unit_unary(jnp.nanargmax,
                                  a,
                                  axis=axis,
                                  keepdims=keepdims)


@set_module_as('saiunit.math')
def nanargmin(
    a: Union[jax.typing.ArrayLike, Quantity],
    axis: int = None,
    keepdims: bool = False
) -> jax.Array:
    """
    Return the index of the minimum value, ignoring NaNs.

    Units are stripped before finding the minimum.

    Parameters
    ----------
    a : array_like or Quantity
        Input data.
    axis : int or None, optional
        Axis along which to operate. By default the flattened input is
        used.
    keepdims : bool, optional
        If True, reduced axes are kept with size one.

    Returns
    -------
    index : jax.Array
        Index of the minimum value (NaNs ignored).

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.nanargmin(jnp.array([3.0, jnp.nan, 1.0]))
        Array(2, dtype=int32)
    """
    return _fun_remove_unit_unary(jnp.nanargmin,
                                  a,
                                  axis=axis,
                                  keepdims=keepdims)


@set_module_as('saiunit.math')
def argwhere(
    a: Union[jax.typing.ArrayLike, Quantity],
    *,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike] = None,
) -> jax.Array:
    """
    Find the indices of array elements that are non-zero.

    Units are stripped before the search.

    Parameters
    ----------
    a : array_like or Quantity
        Input data.
    size : int or None, optional
        Fixed output size (for use inside ``jax.jit``).
    fill_value : scalar or None, optional
        Fill value for padding when ``size`` is given.

    Returns
    -------
    indices : jax.Array
        Array of shape ``(N, a.ndim)`` containing the indices of
        non-zero elements.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.argwhere(jnp.array([0, 1, 0, 2]), size=2)
        Array([[1],
               [3]], dtype=int32)
    """
    return _fun_remove_unit_unary(jnp.argwhere, a, size=size, fill_value=fill_value)


@set_module_as('saiunit.math')
def nonzero(
    a: Union[jax.typing.ArrayLike, Quantity],
    *,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike] = None,
) -> Sequence[jax.Array]:
    """
    Return the indices of non-zero elements.

    Units are stripped before the search.

    Parameters
    ----------
    a : array_like or Quantity
        Input data.
    size : int or None, optional
        Fixed output size (for use inside ``jax.jit``).
    fill_value : scalar or None, optional
        Fill value for padding when ``size`` is given.

    Returns
    -------
    indices : tuple of jax.Array
        Tuple of arrays, one per dimension, containing the indices of
        non-zero elements.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.nonzero(jnp.array([0, 1, 0, 2]), size=2)
        (Array([1, 3], dtype=int32),)
    """
    return _fun_remove_unit_unary(jnp.nonzero, a, size=size, fill_value=fill_value)


@set_module_as('saiunit.math')
def flatnonzero(
    a: Union[jax.typing.ArrayLike, Quantity],
    *,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike] = None,
) -> jax.Array:
    """
    Return indices that are non-zero in the flattened input.

    Units are stripped before the search.

    Parameters
    ----------
    a : array_like or Quantity
        Input data.
    size : int or None, optional
        Fixed output size (for use inside ``jax.jit``).
    fill_value : scalar or None, optional
        Fill value for padding when ``size`` is given.

    Returns
    -------
    indices : jax.Array
        Indices of non-zero elements in the flattened array.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.flatnonzero(jnp.array([0, 1, 0, 2]), size=2)
        Array([1, 3], dtype=int32)
    """
    a = maybe_custom_array(a)
    fill_value = maybe_custom_array(fill_value)
    a_unit = get_unit(a)
    if fill_value is not None:
        fill_value = Quantity(fill_value).in_unit(a_unit).mantissa
    return _fun_remove_unit_unary(jnp.flatnonzero, a, size=size, fill_value=fill_value)


@set_module_as('saiunit.math')
def count_nonzero(
    a: Union[jax.typing.ArrayLike, Quantity],
    axis: Optional[int] = None,
    keepdims: Optional[bool] = None
) -> jax.Array:
    """
    Count the number of non-zero values in the input.

    Units are stripped before counting.

    Parameters
    ----------
    a : array_like or Quantity
        Input data.
    axis : int or None, optional
        Axis along which to count. Default counts over the whole array.
    keepdims : bool or None, optional
        If True, reduced axes are kept with size one.

    Returns
    -------
    count : jax.Array
        Number of non-zero values along the given axis.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.count_nonzero(jnp.array([0, 1, 0, 2, 3]))
        Array(3, dtype=int32)
    """
    return _fun_remove_unit_unary(jnp.count_nonzero, a, axis=axis, keepdims=keepdims)


@set_module_as('saiunit.math')
def searchsorted(
    a: Union[jax.typing.ArrayLike, Quantity],
    v: Union[jax.typing.ArrayLike, Quantity],
    side: str = 'left',
    sorter: Optional[jax.Array] = None,
    *,
    method: Optional[str] = 'scan'
) -> jax.Array | Quantity:
    """
    Find indices where elements should be inserted to maintain order.

    When both ``a`` and ``v`` are Quantities, ``v`` is converted to the
    unit of ``a`` before searching.

    Parameters
    ----------
    a : array_like or Quantity
        Sorted input array.
    v : array_like or Quantity
        Values to insert into ``a``.
    side : {'left', 'right'}, optional
        If ``'left'``, the first suitable index is returned. Default
        is ``'left'``.
    sorter : array_like of int or None, optional
        Indices that sort ``a`` into ascending order.
    method : str, optional
        Algorithm selection. Default is ``'scan'``.

    Returns
    -------
    indices : jax.Array
        Insertion points with the same shape as ``v``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> a = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> u.math.searchsorted(a, jnp.array([2.5]))
        Array([2], dtype=int32)
    """
    a = maybe_custom_array(a)
    a_unit = get_unit(a)
    v = Quantity(v).in_unit(a_unit).mantissa
    a = Quantity(a).mantissa
    r = jnp.searchsorted(a, v, side=side, sorter=sorter, method=method)
    return r


@set_module_as('saiunit.math')
def diag_indices_from(
    arr: Union[jax.typing.ArrayLike, Quantity],
) -> tuple[jax.Array, ...]:
    """
    Return indices for accessing the main diagonal of a given array.

    Units are stripped before computing the indices.

    Parameters
    ----------
    arr : array_like or Quantity
        Input array. Must be at least 2-D with equal-length dimensions.

    Returns
    -------
    indices : tuple of jax.Array
        Index arrays to access the main diagonal.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> arr = jnp.array([[1, 2], [3, 4]])
        >>> u.math.diag_indices_from(arr)
        (Array([0, 1], dtype=int32), Array([0, 1], dtype=int32))
    """
    return _fun_remove_unit_unary(jnp.diag_indices_from, arr)
