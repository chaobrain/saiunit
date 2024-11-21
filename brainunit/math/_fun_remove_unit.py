# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from .._base import Quantity, get_unit
from .._misc import set_module_as

__all__ = [
    # math funcs remove unit (unary)
    'iscomplexobj', 'heaviside', 'signbit', 'sign', 'bincount', 'digitize',

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

def _fun_remove_unit_unary(func, x, *args, **kwargs):
    if isinstance(x, Quantity):
        return func(x.mantissa, *args, **kwargs)
    else:
        return func(x, *args, **kwargs)

@set_module_as('brainunit.math')
def iscomplexobj(
    x: Union[jax.typing.ArrayLike, Quantity],
) -> bool:
    """
    Return True if x is a complex type or an array of complex numbers.

    Parameters
    ----------
    x : array_like, Quantity
        Input array.

    Returns
    -------
    out : bool
        True if `x` is  a complex type or an array of complex numbers.

    """
    return _fun_remove_unit_unary(jnp.iscomplexobj, x)



@set_module_as('brainunit.math')
def heaviside(
    x1: Union[Quantity, jax.Array],
    x2: jax.typing.ArrayLike
) -> Union[Quantity, jax.Array]:
    """
    Compute the Heaviside step function.

    Parameters
    ----------
    x1: array_like, Quantity
      Input array.
    x2: array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
    """
    x1 = x1.mantissa if isinstance(x1, Quantity) else x1
    if isinstance(x2, Quantity):
        assert x2.is_unitless, f'Expected unitless array for x2, while got {x2}'
        x2 = x2.mantissa
    return _fun_remove_unit_unary(jnp.heaviside, x1, x2)


@set_module_as('brainunit.math')
def signbit(x: Union[jax.typing.ArrayLike, Quantity]) -> jax.Array:
    """
    Returns element-wise True where signbit is set (less than zero).

    Parameters
    ----------
    x : array_like, Quantity
        The input value(s).

    Returns
    -------
    result : ndarray of bool
        Output array, or reference to `out` if that was supplied.
        This is a scalar if `x` is a scalar.
    """
    return _fun_remove_unit_unary(jnp.signbit, x)


@set_module_as('brainunit.math')
def sign(x: Union[jax.typing.ArrayLike, Quantity]) -> jax.Array:
    """
    Returns the sign of each element in the input array.

    Parameters
    ----------
    x : array_like, Quantity
      Input values.

    Returns
    -------
    y : ndarray
      The sign of `x`.
      This is a scalar if `x` is a scalar.
    """
    return _fun_remove_unit_unary(jnp.sign, x)


@set_module_as('brainunit.math')
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
    out : ndarray of ints
      The result of binning the input array.
      The length of `out` is equal to ``bu.amax(x)+1``.
    """
    return _fun_remove_unit_unary(jnp.bincount, x, weights=weights, minlength=minlength, length=length)


@set_module_as('brainunit.math')
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
    indices : ndarray of ints
        Output array of indices, of same shape as `x`.
    """
    if isinstance(x, Quantity) and isinstance(bins, Quantity):
        bins = bins.in_unit(x.unit).mantissa
        x = x.mantissa
    elif isinstance(x, Quantity):
        assert x.is_unitless, f'Expected unitless Quantity when bins is not a Quantity, got {x}'
        x = x.mantissa
    elif isinstance(bins, Quantity):
        assert bins.is_unitless, f'Expected unitless Quantity when x is not a Quantity, got {bins}'
        bins = bins.mantissa
    return jnp.digitize(x, bins, right=right)


def _fun_logic_unary(func, x, *args, **kwargs):
    if isinstance(x, Quantity):
        assert x.is_unitless, f'Expected unitless array for {func.__name__}, while got {x}'
    return func(x, *args, **kwargs)


@set_module_as('brainunit.math')
def all(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    keepdims: bool = False,
    where: Optional[jax.Array] = None
) -> Union[bool, jax.Array]:
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
      Axis or axes along which a logical AND reduction is performed.
      The default (``axis=None``) is to perform a logical AND over all
      the dimensions of the input array. `axis` may be negative, in
      which case it counts from the last to the first axis.

      If this is a tuple of ints, a reduction is performed on multiple
      axes, instead of a single axis or all the axes as before.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the input array.

      If the default value is passed, then `keepdims` will not be
      passed through to the `all` method of sub-classes of
      `ndarray`, however any non-default value will be.  If the
      sub-class' method does not implement `keepdims` any
      exceptions will be raised.
    where : array_like of bool, optional
      Elements to include in checking for all `True` values.

    Returns
    -------
    all : ndarray, bool
      A new boolean or array is returned unless `out` is specified,
      in which case a reference to `out` is returned.
    """
    return _fun_logic_unary(jnp.all, x, axis=axis, keepdims=keepdims, where=where)


@set_module_as('brainunit.math')
def any(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    keepdims: bool = False,
    where: Optional[jax.Array] = None
) -> Union[bool, jax.Array]:
    """
    Test whether any array element along a given axis evaluates to True.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
      Axis or axes along which a logical AND reduction is performed.
      The default (``axis=None``) is to perform a logical AND over all
      the dimensions of the input array. `axis` may be negative, in
      which case it counts from the last to the first axis.

      If this is a tuple of ints, a reduction is performed on multiple
      axes, instead of a single axis or all the axes as before.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the input array.

      If the default value is passed, then `keepdims` will not be
      passed through to the `all` method of sub-classes of
      `ndarray`, however any non-default value will be.  If the
      sub-class' method does not implement `keepdims` any
      exceptions will be raised.
    where : array_like of bool, optional
      Elements to include in checking for all `True` values.

    Returns
    -------
    any : ndarray, bool
      A new boolean or array is returned unless `out` is specified,
      in which case a reference to `out` is returned.
    """
    return _fun_logic_unary(jnp.any, x, axis=axis, keepdims=keepdims, where=where)


@set_module_as('brainunit.math')
def logical_not(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    """
    Compute the truth value of NOT x element-wise.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or object that can be converted to an array.

    Returns
    -------
    logical_not : ndarray, bool
      A new boolean or array is returned unless `out` is specified,
      in which case a reference to `out` is returned.
    """
    return _fun_logic_unary(jnp.logical_not, x)


alltrue = all
sometrue = any


# logic funcs (binary)
# --------------------


def _fun_logic_binary(func, x, y, *args, **kwargs):
    if isinstance(x, Quantity) and isinstance(y, Quantity):
        return func(x.mantissa, y.in_unit(x.unit).mantissa, *args, **kwargs)
    elif isinstance(x, Quantity):
        assert x.is_unitless, f'Expected unitless array when y is not Quantity, while got {x}'
        return func(x.mantissa, y, *args, **kwargs)
    elif isinstance(y, Quantity):
        assert y.is_unitless, f'Expected unitless array when x is not Quantity, while got {y}'
        return func(x, y.mantissa, *args, **kwargs)
    else:
        return func(x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    equal(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

    Return (x == y) element-wise.

    Parameters
    ----------
    x, y : array_like, Quantity
      Input arrays.
      If ``x.shape != y.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
      A location into which the result is stored. If provided, it must have
      a shape that the inputs broadcast to. If not provided or None,
      a freshly-allocated array is returned. A tuple (possible only as a
      keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
      This condition is broadcast over the input. At locations where the
      condition is True, the `out` array will be set to the ufunc result.
      Elsewhere, the `out` array will retain its original value.
      Note that if an uninitialized `out` array is created via the default
      ``out=None``, locations within it where the condition is False will
      remain uninitialized.
    **kwargs
      For other keyword-only arguments, see the
      :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
      Output array, element-wise comparison of `x` and `y`.
      Typically of type bool, unless ``dtype=object`` is passed.
      This is a scalar if both `x` and `y` are scalars.
    """
    return _fun_logic_binary(jnp.equal, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def not_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    not_equal(x, y, /, out=None, *, where=True, casting='same_kind',
    order='K', dtype=None, subok=True[, signature, extobj])

    Return (x != y) element-wise.

    Parameters
    ----------
    x, y : array_like, Quantity
      Input arrays.
      If ``x.shape != y.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
      A location into which the result is stored. If provided, it must have
      a shape that the inputs broadcast to. If not provided or None,
      a freshly-allocated array is returned. A tuple (possible only as a
      keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
      This condition is broadcast over the input. At locations where the
      condition is True, the `out` array will be set to the ufunc result.
      Elsewhere, the `out` array will retain its original value.
      Note that if an uninitialized `out` array is created via the default
      ``out=None``, locations within it where the condition is False will
      remain uninitialized.
    **kwargs
      For other keyword-only arguments, see the
      :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
      Output array, element-wise comparison of `x` and `y`.
      Typically of type bool, unless ``dtype=object`` is passed.
      This is a scalar if both `x` and `y` are scalars.
    """
    return _fun_logic_binary(jnp.not_equal, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def greater(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    greater(x, y, /, out=None, *, where=True, casting='same_kind',
    order='K', dtype=None, subok=True[, signature, extobj])

    Return the truth value of (x > y) element-wise.

    Parameters
    ----------
    x, y : array_like, Quantity
      Input arrays.
      If ``x.shape != y.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
      A location into which the result is stored. If provided, it must have
      a shape that the inputs broadcast to. If not provided or None,
      a freshly-allocated array is returned. A tuple (possible only as a
      keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
      This condition is broadcast over the input. At locations where the
      condition is True, the `out` array will be set to the ufunc result.
      Elsewhere, the `out` array will retain its original value.
      Note that if an uninitialized `out` array is created via the default
      ``out=None``, locations within it where the condition is False will
      remain uninitialized.
    **kwargs
      For other keyword-only arguments, see the
      :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
      Output array, element-wise comparison of `x` and `y`.
      Typically of type bool, unless ``dtype=object`` is passed.
      This is a scalar if both `x` and `y` are scalars.
    """
    return _fun_logic_binary(jnp.greater, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def greater_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[
    bool, jax.Array]:
    """
    greater_equal(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

    Return the truth value of (x >= y) element-wise.

    Parameters
    ----------
    x, y : array_like, Quantity
      Input arrays.
      If ``x.shape != y.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
      A location into which the result is stored. If provided, it must have
      a shape that the inputs broadcast to. If not provided or None,
      a freshly-allocated array is returned. A tuple (possible only as a
      keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
      This condition is broadcast over the input. At locations where the
      condition is True, the `out` array will be set to the ufunc result.
      Elsewhere, the `out` array will retain its original value.
      Note that if an uninitialized `out` array is created via the default
      ``out=None``, locations within it where the condition is False will
      remain uninitialized.
    **kwargs
      For other keyword-only arguments, see the
      :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : bool or ndarray of bool
      Output array, element-wise comparison of `x` and `y`.
      Typically of type bool, unless ``dtype=object`` is passed.
      This is a scalar if both `x` and `y` are scalars.
    """
    return _fun_logic_binary(jnp.greater_equal, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def less(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    less(x, y, /, out=None, *, where=True, casting='same_kind',
    order='K', dtype=None, subok=True[, signature, extobj])

    Return the truth value of (x < y) element-wise.

    Parameters
    ----------
    x, y : array_like, Quantity
        Input arrays.
        If ``x1.shape != y.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise comparison of `x` and `y`.
        Typically of type bool, unless ``dtype=object`` is passed.
        This is a scalar if both `x` and `y` are scalars.
    """
    return _fun_logic_binary(jnp.less, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def less_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[
    bool, jax.Array]:
    """
    less_equal(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

    Return the truth value of (x <= y) element-wise.

    Parameters
    ----------
    x, y : array_like, Quantity
      Input arrays.
      If ``x.shape != y.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
      A location into which the result is stored. If provided, it must have
      a shape that the inputs broadcast to. If not provided or None,
      a freshly-allocated array is returned. A tuple (possible only as a
      keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
      This condition is broadcast over the input. At locations where the
      condition is True, the `out` array will be set to the ufunc result.
      Elsewhere, the `out` array will retain its original value.
      Note that if an uninitialized `out` array is created via the default
      ``out=None``, locations within it where the condition is False will
      remain uninitialized.
    **kwargs
      For other keyword-only arguments, see the
      :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
      Output array, element-wise comparison of `x` and `y`.
      Typically of type bool, unless ``dtype=object`` is passed.
      This is a scalar if both `x` and `y` are scalars.
    """
    return _fun_logic_binary(jnp.less_equal, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def array_equal(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[
    bool, jax.Array]:
    """
    True if two arrays have the same shape and elements, False otherwise.

    Parameters
    ----------
    x, y : array_like, Quantity
      Input arrays.
    equal_nan : bool
      Whether to compare NaN's as equal. If the dtype of a1 and a2 is
      complex, values will be considered equal if either the real or the
      imaginary component of a given value is ``nan``.

    Returns
    -------
    b : bool
      Returns True if the arrays are equal.
    """
    return _fun_logic_binary(jnp.array_equal, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
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
    out : array_like
      Returns a boolean array of where `a` and `b` are equal within the
      given tolerance. If both `a` and `b` are scalars, returns a single
      boolean value.
    """
    unit = get_unit(x)
    if isinstance(x, Quantity) and isinstance(y, Quantity):
        y = y.in_unit(x.unit).mantissa
        x = x.mantissa
    elif isinstance(x, Quantity):
        assert x.is_unitless, f'Expected unitless array when y is not Quantity, while got {x}'
        x = x.mantissa
    elif isinstance(y, Quantity):
        assert y.is_unitless, f'Expected unitless array when x is not Quantity, while got {y}'
        y = y.mantissa
    if rtol is None:
        rtol = 1e-5 * unit
    if atol is None:
        atol = 1e-8 * unit
    atol = Quantity(atol).in_unit(unit).mantissa
    rtol = Quantity(rtol).in_unit(unit).mantissa
    return _fun_logic_binary(jnp.isclose, x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)


@set_module_as('brainunit.math')
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
      Returns True if the two arrays are equal within the given
      tolerance; False otherwise.
    """
    unit = get_unit(x)
    if isinstance(x, Quantity) and isinstance(y, Quantity):
        y = y.in_unit(x.unit)
        x_val = x.mantissa
        y_val = y.mantissa
    elif isinstance(x, Quantity):
        assert x.is_unitless, f'Expected unitless array when y is not Quantity, while got {x}'
        x_val = x.mantissa
        y_val = y
    elif isinstance(y, Quantity):
        assert y.is_unitless, f'Expected unitless array when x is not Quantity, while got {y}'
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


@set_module_as('brainunit.math')
def logical_and(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    logical_and(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

    Compute the truth value of x AND y element-wise.

    Parameters
    ----------
    x, y : array_like, Quantity
      Input arrays.
      If ``x.shape != y.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
      A location into which the result is stored. If provided, it must have
      a shape that the inputs broadcast to. If not provided or None,
      a freshly-allocated array is returned. A tuple (possible only as a
      keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
      This condition is broadcast over the input. At locations where the
      condition is True, the `out` array will be set to the ufunc result.
      Elsewhere, the `out` array will retain its original value.
      Note that if an uninitialized `out` array is created via the default
      ``out=None``, locations within it where the condition is False will
      remain uninitialized.
    **kwargs
      For other keyword-only arguments, see the
      :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or bool
      Boolean result of the logical AND operation applied to the elements
      of `x` and `y`; the shape is determined by broadcasting.
      This is a scalar if both `x` and `y` are scalars.
    """
    return _fun_logic_binary(jnp.logical_and, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def logical_or(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    logical_or(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

    Compute the truth value of x OR y element-wise.

    Parameters
    ----------
    x, y : array_like, Quantity
      Logical OR is applied to the elements of `x` and `y`.
      If ``x.shape != y.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
      A location into which the result is stored. If provided, it must have
      a shape that the inputs broadcast to. If not provided or None,
      a freshly-allocated array is returned. A tuple (possible only as a
      keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
      This condition is broadcast over the input. At locations where the
      condition is True, the `out` array will be set to the ufunc result.
      Elsewhere, the `out` array will retain its original value.
      Note that if an uninitialized `out` array is created via the default
      ``out=None``, locations within it where the condition is False will
      remain uninitialized.
    **kwargs
      For other keyword-only arguments, see the
      :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or bool
      Boolean result of the logical OR operation applied to the elements
      of `x` and `y`; the shape is determined by broadcasting.
      This is a scalar if both `x` and `y` are scalars.
    """
    return _fun_logic_binary(jnp.logical_or, x, y, *args, **kwargs)


@set_module_as('brainunit.math')
def logical_xor(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    *args,
    **kwargs
) -> Union[bool, jax.Array]:
    """
    logical_xor(x, y, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

    Compute the truth value of x XOR y, element-wise.

    Parameters
    ----------
    x, y : array_like, Quantity
      Logical XOR is applied to the elements of `x` and `y`.
      If ``x.shape != y.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
      A location into which the result is stored. If provided, it must have
      a shape that the inputs broadcast to. If not provided or None,
      a freshly-allocated array is returned. A tuple (possible only as a
      keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
      This condition is broadcast over the input. At locations where the
      condition is True, the `out` array will be set to the ufunc result.
      Elsewhere, the `out` array will retain its original value.
      Note that if an uninitialized `out` array is created via the default
      ``out=None``, locations within it where the condition is False will
      remain uninitialized.
    **kwargs
      For other keyword-only arguments, see the
      :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : bool or ndarray of bool
      Boolean result of the logical XOR operation applied to the elements
      of `x` and `y`; the shape is determined by broadcasting.
      This is a scalar if both `x` and `y` are scalars.
    """
    return _fun_logic_binary(jnp.logical_xor, x, y, *args, **kwargs)


# ----------------------
# Indexing functions
# ----------------------


@set_module_as('brainunit.math')
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
    Returns the indices that would sort an array or a quantity.

    Parameters
    ----------
    a : array_like, Quantity
      Array or quantity to be sorted.
    axis : int or None, optional
      Axis along which to sort. If None, the array is flattened before sorting. The default is -1, which sorts along
      the last axis.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
      Sorting algorithm. The default is 'None'.
    order : str or list of str, optional
      When `a` is a quantity, it can be a string or a sequence of strings, which is interpreted as an order the quantity
      should be sorted. The default is None.
    stable : bool, optional
      Whether to use a stable sorting algorithm. The default is True.
    descending : bool, optional
      Whether to sort in descending order. The default is False.

    Returns
    -------
    res : ndarray
      Array of indices that sort the array.
    """
    return _fun_remove_unit_unary(jnp.argsort,
                                  a,
                                  axis=axis,
                                  kind=kind,
                                  order=order,
                                  stable=stable,
                                  descending=descending)


@set_module_as('brainunit.math')
def argmax(
    a: Union[jax.typing.ArrayLike, Quantity],
    axis: Optional[int] = None,
) -> jax.Array:
    """
    Returns indices of the max value along an axis.

    Parameters
    ----------
    a : array_like, Quantity
      Input data.
    axis : int, optional
      By default, the index is into the flattened array, otherwise along the specified axis.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
      option, the result will broadcast correctly against the input array.

    Returns
    -------
    res : ndarray
      Array of indices into the array. It has the same shape as `a.shape` with the dimension along `axis` removed.
    """
    return _fun_remove_unit_unary(jnp.argmax, a, axis=axis)


@set_module_as('brainunit.math')
def argmin(
    a: Union[jax.typing.ArrayLike, Quantity],
    axis: Optional[int] = None,
    keepdims: Optional[bool] = None
) -> jax.Array:
    """
    Returns indices of the min value along an axis.

    Parameters
    ----------
    a : array_like, Quantity
      Input data.
    axis : int, optional
      By default, the index is into the flattened array, otherwise along the specified axis.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
      option, the result will broadcast correctly against the input array.

    Returns
    -------
    res : ndarray
      Array of indices into the array. It has the same shape as `a.shape` with the dimension along `axis` removed.
    """
    return _fun_remove_unit_unary(jnp.argmin, a, axis=axis, keepdims=keepdims)


@set_module_as('brainunit.math')
def nanargmax(
    a: Union[jax.typing.ArrayLike, Quantity],
    axis: int = None,
    keepdims: bool = False
) -> jax.Array:
    """
    Return the indices of the maximum values in the specified axis ignoring
    NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the
    results cannot be trusted if a slice contains only NaNs and -Infs.


    Parameters
    ----------
    a : array_like, Quantity
      Input data.
    axis : int, optional
      Axis along which to operate.  By default flattened input is used.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the array.

    Returns
    -------
    index_array : ndarray
      An array of indices or a single index value.
    """
    return _fun_remove_unit_unary(jnp.nanargmax,
                                  a,
                                  axis=axis,
                                  keepdims=keepdims)


@set_module_as('brainunit.math')
def nanargmin(
    a: Union[jax.typing.ArrayLike, Quantity],
    axis: int = None,
    keepdims: bool = False
) -> jax.Array:
    """
    Return the indices of the minimum values in the specified axis ignoring
    NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the results
    cannot be trusted if a slice contains only NaNs and Infs.

    Parameters
    ----------
    a : array_like, Quantity
      Input data.
    axis : int, optional
      Axis along which to operate.  By default flattened input is used.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the array.

    Returns
    -------
    index_array : ndarray
      An array of indices or a single index value.
    """
    return _fun_remove_unit_unary(jnp.nanargmin,
                                  a,
                                  axis=axis,
                                  keepdims=keepdims)


@set_module_as('brainunit.math')
def argwhere(
    a: Union[jax.typing.ArrayLike, Quantity],
    *,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike] = None,
) -> jax.Array:
    """
    Find the indices of array elements that are non-zero, grouped by element.

    Parameters
    ----------
    a : array_like, Quantity
      Input data.
    size : int, optional
      The length of the returned axis. By default, the length of the input array along the axis is used.
    fill_value : scalar, optional
      The value to use for elements in the output array that are not selected. If None, the output array has the same
      type as `a` and is filled with zeros.

    Returns
    -------
    res : ndarray
      The indices of elements that are non-zero. The indices are grouped by element.
    """
    return _fun_remove_unit_unary(jnp.argwhere, a, size=size, fill_value=fill_value)


@set_module_as('brainunit.math')
def nonzero(
    a: Union[jax.typing.ArrayLike, Quantity],
    *,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike] = None,
) -> Sequence[jax.Array]:
    """
    Return the indices of the elements that are non-zero.

    Parameters
    ----------
    a : array_like, Quantity
      Input data.
    size : int, optional
      The length of the returned axis. By default, the length of the input array along the axis is used.
    fill_value : scalar, optional
      The value to use for elements in the output array that are not selected. If None, the output array has the same
      type as `a` and is filled with zeros.

    Returns
    -------
    res : tuple of ndarrays
      Indices of elements that are non-zero along the specified axis. Each array in the tuple has the same shape as the
      input array.
    """
    return _fun_remove_unit_unary(jnp.nonzero, a, size=size, fill_value=fill_value)


@set_module_as('brainunit.math')
def flatnonzero(
    a: Union[jax.typing.ArrayLike, Quantity],
    *,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike] = None,
) -> jax.Array:
    """
    Return indices that are non-zero in the flattened version of the input quantity or array.

    Parameters
    ----------
    a : array_like, Quantity
      Input data.
    size : int, optional
      The length of the returned axis. By default, the length of the input array along the axis is used.
    fill_value : scalar, optional
      The value to use for elements in the output array that are not selected. If None, the output array has the same
      type as `a` and is filled with zeros.

    Returns
    -------
    res : ndarray
      Output array, containing the indices of the elements of `a.ravel()` that are non-zero.
    """
    a_unit = get_unit(a)
    if fill_value is not None:
        fill_value = Quantity(fill_value).in_unit(a_unit).mantissa
    return _fun_remove_unit_unary(jnp.flatnonzero, a, size=size, fill_value=fill_value)


@set_module_as('brainunit.math')
def count_nonzero(
    a: Union[jax.typing.ArrayLike, Quantity],
    axis: Optional[int] = None,
    keepdims: Optional[bool] = None
) -> jax.Array:
    """
    Count the number of non-zero values in the quantity or array `a`.

    Parameters
    ----------
    a : array_like, Quantity
      The array for which to count non-zeros.
    axis : int, optional
      The axis along which to count the non-zeros. If `None`, count non-zeros over the entire array.
    keepdims : bool, optional
      If this is set to `True`, the axes which are counted are left in the result as dimensions with size one. With this
      option, the result will broadcast correctly against the original array.

    Returns
    -------
    res : ndarray
      Number of non-zero values in the quantity or array along a given axis.
    """
    return _fun_remove_unit_unary(jnp.count_nonzero, a, axis=axis, keepdims=keepdims)


@set_module_as('brainunit.math')
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

    Find the indices into a sorted array `a` such that, if the corresponding elements in `v` were inserted before the
    indices, the order of `a` would be preserved.

    Parameters
    ----------
    a : array_like, Quantity
      Input array. It must be sorted in ascending order.
    v : array_like, Quantity
      Values to insert into `a`.
    side : {'left', 'right'}, optional
      If 'left', the index of the first suitable location found is given. If 'right', return the last such index. If
      there is no suitable index, return either 0 or N (where N is the length of `a`).
    sorter : 1-D array_like, optional
      Optional array of integer indices that sort array `a` into ascending order. They are typically the result of
      `argsort`.
    method : str
      One of 'scan' (default), 'scan_unrolled', 'sort' or 'compare_all'. Controls the method used by the
      implementation: 'scan' tends to be more performant on CPU (particularly when ``a`` is
      very large), 'scan_unrolled' is more performant on GPU at the expense of additional compile time,
      'sort' is often more performant on accelerator backends like GPU and TPU
      (particularly when ``v`` is very large), and 'compare_all' can be most performant
      when ``a`` is very small. The default is 'scan'.

    Returns
    -------
    out : ndarray
      Array of insertion points with the same shape as `v`.
    """
    a_unit = get_unit(a)
    v = Quantity(v).in_unit(a_unit).mantissa
    a = Quantity(a).mantissa
    r = jnp.searchsorted(a, v, side=side, sorter=sorter, method=method)
    return r

@set_module_as('brainunit.math')
def diag_indices_from(
    arr: Union[jax.typing.ArrayLike, Quantity],
) -> tuple[jax.Array, ...]:
    """Return indices for accessing the main diagonal of a given array.

    JAX implementation of :func:`numpy.diag_indices_from`.

    Args:
        arr: Input array. Must be at least 2-dimensional and have equal length along
            all dimensions.

    Returns:
        A tuple of arrays containing the indices to access the main diagonal of
        the input array.

    Examples:
    >>> arr = jnp.array([[1, 2, 3],
    ...                  [4, 5, 6],
    ...                  [7, 8, 9]])
    >>> jnp.diag_indices_from(arr)
    (Array([0, 1, 2], dtype=int32), Array([0, 1, 2], dtype=int32))
    >>> arr = jnp.array([[[1, 2], [3, 4]],
    ...                  [[5, 6], [7, 8]]])
    >>> jnp.diag_indices_from(arr)
    (Array([0, 1], dtype=int32),
    Array([0, 1], dtype=int32),
    Array([0, 1], dtype=int32))
    """
    return _fun_remove_unit_unary(jnp.diag_indices_from, arr)