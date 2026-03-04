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

from collections.abc import Sequence
from typing import Union, Optional, Tuple, Any, Callable

import jax
import jax.numpy as jnp

from saiunit._base_unit import UNITLESS
from saiunit._base_getters import maybe_decimal
from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as, maybe_custom_array, maybe_custom_array_tree
from ._fun_array_creation import asarray

__all__ = [

    # math funcs change unit (unary)
    'reciprocal', 'prod', 'product', 'nancumprod', 'nanprod', 'cumprod',
    'cumproduct', 'var', 'nanvar', 'cbrt', 'square', 'sqrt',

    # math funcs change unit (binary)
    'multiply', 'divide', 'power', 'cross',
    'true_divide', 'floor_divide', 'float_power',
    'divmod', 'convolve',

    # linear algebra
    'dot', 'multi_dot', 'vdot', 'vecdot', 'inner', 'outer', 'kron', 'matmul', 'tensordot',
    'matrix_power',
]


# math funcs change unit (unary)
# ------------------------------

def _fun_change_unit_unary(val_fun, unit_fun, x, *args, **kwargs):
    x = maybe_custom_array(x)
    args, kwargs = maybe_custom_array_tree((args, kwargs))
    if isinstance(x, Quantity):
        # x = x.factorless()
        r = Quantity(val_fun(x.mantissa, *args, **kwargs), unit=unit_fun(x.unit))
        return maybe_decimal(r)
    return val_fun(x, *args, **kwargs)


def unit_change(
    unit_change_fun: Callable
):
    def actual_decorator(func):
        func._unit_change_fun = unit_change_fun
        return set_module_as('saiunit.math')(func)

    return actual_decorator


@unit_change(lambda u: u ** -1)
def reciprocal(
    x: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.Array]:
    """
    Return the reciprocal of the argument, element-wise.

    Calculates ``1/x``. When the input carries a unit, the result carries the
    inverse of that unit.

    Parameters
    ----------
    x : array_like or Quantity
        Input array.

    Returns
    -------
    y : ndarray or Quantity
        Return array with the same shape as `x`.
        This is a scalar if `x` is a scalar.
        This is a Quantity if the unit of `x` is not dimensionless; the
        resulting unit is ``1 / x.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> result = u.math.reciprocal(u.math.array([2.0, 4.0]) * u.second)
        >>> result.mantissa  # array([0.5 , 0.25])
    """
    return _fun_change_unit_unary(jnp.reciprocal, lambda u: u ** -1, x)


@unit_change(lambda u: u ** 2)
def var(
    a: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[Any] = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: Optional[jax.typing.ArrayLike] = None
) -> Union[Quantity, jax.Array]:
    """
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a
    distribution. The variance is computed for the flattened array by default,
    otherwise over the specified axis. The resulting unit is the square of the
    input unit.

    Parameters
    ----------
    a : array_like or Quantity
        Array containing numbers whose variance is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed. The default is to
        compute the variance of the flattened array.
    dtype : data-type, optional
        Type to use in computing the variance. For arrays of integer type
        the default is ``float64``; for arrays of float types it is the same as
        the array type.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of elements. By
        default ``ddof`` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the input array.
    where : array_like of bool, optional
        Elements to include in the variance.

    Returns
    -------
    variance : ndarray or Quantity
        If the input has a unit, the result is a Quantity whose unit is the
        square of the input unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> q = u.math.array([1.0, 2.0, 3.0]) * u.meter
        >>> u.math.var(q)  # unit becomes meter ** 2
    """
    return _fun_change_unit_unary(jnp.var,
                                  lambda u: u ** 2,
                                  a,
                                  axis=axis,
                                  dtype=dtype,
                                  ddof=ddof,
                                  keepdims=keepdims,
                                  where=where)


@unit_change(lambda u: u ** 2)
def nanvar(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[Any] = None,
    ddof: int = 0,
    keepdims: bool = False,
    where: Optional[jax.typing.ArrayLike] = None
) -> Union[Quantity, jax.Array]:
    """
    Compute the variance along the specified axis, while ignoring NaNs.

    Returns the variance of the array elements, a measure of the spread of a
    distribution. NaN values are treated as missing. The resulting unit is the
    square of the input unit.

    Parameters
    ----------
    x : array_like or Quantity
        Array containing numbers whose variance is desired. If `x` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the variance is computed. The default is to
        compute the variance of the flattened array.
    dtype : data-type, optional
        Type to use in computing the variance. For arrays of integer type the
        default is ``float64``; for arrays of float types it is the same as
        the array type.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of non-NaN elements.
        By default ``ddof`` is zero.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as
        dimensions with size one.
    where : array_like of bool, optional
        Elements to include in the variance.

    Returns
    -------
    variance : ndarray or Quantity
        The variance of the non-NaN elements. If the input has a unit, the
        result is a Quantity whose unit is the square of the input unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> q = u.math.array([1.0, jnp.nan, 3.0]) * u.meter
        >>> u.math.nanvar(q)  # unit becomes meter ** 2
    """
    return _fun_change_unit_unary(jnp.nanvar,
                                  lambda u: u ** 2,
                                  x,
                                  axis=axis,
                                  dtype=dtype,
                                  ddof=ddof,
                                  keepdims=keepdims,
                                  where=where)


@unit_change(lambda u: u ** 0.5)
def sqrt(
    x: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.Array]:
    """
    Compute the positive square root of each element.

    When the input carries a unit, the resulting unit is the square root of
    that unit (e.g. ``meter ** 2`` becomes ``meter``).

    Parameters
    ----------
    x : array_like or Quantity
        The values whose square-roots are required.

    Returns
    -------
    y : ndarray or Quantity
        An array of the same shape as `x`, containing the positive
        square-root of each element. If `x` carries a unit, the result is
        a Quantity whose unit is ``x.unit ** 0.5``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> q = u.math.array([4.0, 9.0, 16.0]) * (u.meter ** 2)
        >>> u.math.sqrt(q)  # Quantity with unit meter
    """
    return _fun_change_unit_unary(jnp.sqrt, lambda u: u ** 0.5, x)


@unit_change(lambda u: u ** (1 / 3))
def cbrt(
    x: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.Array]:
    """
    Compute the cube root of each element.

    When the input carries a unit, the resulting unit is the cube root of
    that unit (e.g. ``meter ** 3`` becomes ``meter``).

    Parameters
    ----------
    x : array_like or Quantity
        The values whose cube-roots are required.

    Returns
    -------
    y : ndarray or Quantity
        An array of the same shape as `x`, containing the cube root of each
        element. If `x` carries a unit, the result is a Quantity whose unit
        is ``x.unit ** (1/3)``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> q = u.math.array([8.0, 27.0]) * (u.meter ** 3)
        >>> u.math.cbrt(q)  # Quantity with unit meter
    """
    return _fun_change_unit_unary(jnp.cbrt, lambda u: u ** (1 / 3), x)


@unit_change(lambda u: u ** 2)
def square(
    x: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.Array]:
    """
    Compute the square of each element.

    When the input carries a unit, the resulting unit is the square of that
    unit (e.g. ``meter`` becomes ``meter ** 2``).

    Parameters
    ----------
    x : array_like or Quantity
        Input data.

    Returns
    -------
    out : ndarray or Quantity
        Element-wise ``x * x``, of the same shape and dtype as `x`. This is
        a scalar if `x` is a scalar. If `x` carries a unit, the result is a
        Quantity whose unit is ``x.unit ** 2``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> q = u.math.array([2.0, 3.0, 4.0]) * u.meter
        >>> u.math.square(q)  # Quantity with unit meter ** 2
    """
    return _fun_change_unit_unary(jnp.square, lambda u: u ** 2, x)


@set_module_as('saiunit.math')
def prod(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    dtype: Optional[jax.typing.DTypeLike] = None,
    keepdims: Optional[bool] = False,
    initial: Union[Quantity, jax.typing.ArrayLike] = None,
    where: Union[Quantity, jax.typing.ArrayLike] = None,
    promote_integers: bool = True
) -> Union[Quantity, jax.Array]:
    """
    Return the product of array elements over a given axis.

    When the input is a Quantity, the resulting unit is the input unit raised
    to the power equal to the number of elements along the reduced axis.

    Parameters
    ----------
    x : array_like or Quantity
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed. The default,
        ``axis=None``, will calculate the product of all the elements in the
        input array.
    dtype : dtype, optional
        The type of the returned array, as well as of the accumulator in
        which the elements are multiplied.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as
        dimensions with size one.
    initial : scalar, optional
        The starting value for this product.
    where : array_like of bool, optional
        Elements to include in the product.
    promote_integers : bool, optional
        Whether to promote integer dtypes to the default platform integer.

    Returns
    -------
    product_along_axis : ndarray or Quantity
        An array shaped as `x` but with the specified axis removed. If `x`
        carries a unit, the result is a Quantity whose unit is
        ``x.unit ** n`` where ``n`` is the number of elements reduced.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> q = u.math.array([2.0, 3.0]) * u.meter
        >>> u.math.prod(q)  # product is 6.0, unit is meter ** 2
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        return x.prod(axis=axis,
                      dtype=dtype,
                      keepdims=keepdims,
                      initial=initial,
                      where=where,
                      promote_integers=promote_integers)
    else:
        return jnp.prod(x,
                        axis=axis,
                        dtype=dtype,
                        keepdims=keepdims,
                        initial=initial,
                        where=where,
                        promote_integers=promote_integers)


@set_module_as('saiunit.math')
def nanprod(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    dtype: Optional[jax.typing.DTypeLike] = None,
    keepdims: bool = False,
    initial: Union[Quantity, jax.typing.ArrayLike] = None,
    where: Union[Quantity, jax.typing.ArrayLike] = None
):
    """
    Return the product of array elements over a given axis treating NaNs as one.

    Behaves like :func:`prod` but treats NaN values as one, so they do not
    affect the product.

    Parameters
    ----------
    x : array_like or Quantity
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed. The default,
        ``axis=None``, will calculate the product of all elements.
    dtype : dtype, optional
        The type of the returned array, as well as of the accumulator in
        which the elements are multiplied.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as
        dimensions with size one.
    initial : scalar, optional
        The starting value for this product.
    where : array_like of bool, optional
        Elements to include in the product.

    Returns
    -------
    product_along_axis : ndarray or Quantity
        An array shaped as `x` but with the specified axis removed. If `x`
        carries a unit, the result is a Quantity whose unit is
        ``x.unit ** n`` where ``n`` is the number of elements reduced.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> q = u.math.array([2.0, jnp.nan, 3.0]) * u.meter
        >>> u.math.nanprod(q)  # NaN treated as 1, result is 6.0
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        return x.nanprod(axis=axis,
                         dtype=dtype,
                         keepdims=keepdims,
                         initial=initial,
                         where=where)
    else:
        return jnp.nanprod(x,
                           axis=axis,
                           dtype=dtype,
                           keepdims=keepdims,
                           initial=initial,
                           where=where)


product = prod


@set_module_as('saiunit.math')
def cumprod(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """
    Return the cumulative product of elements along a given axis.

    Each position *i* in the output contains the product of elements from
    index 0 to *i*. When the input carries a unit, the unit is raised to
    the corresponding cumulative power.

    Parameters
    ----------
    x : array_like or Quantity
        Input array.
    axis : int, optional
        Axis along which the cumulative product is computed. By default
        the input is flattened.
    dtype : dtype, optional
        Type of the returned array, as well as of the accumulator in which
        the elements are multiplied.

    Returns
    -------
    cumprod : ndarray or Quantity
        A new array holding the cumulative product. If `x` carries a unit,
        the result is a Quantity.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> q = u.math.array([1.0, 2.0, 3.0]) * u.meter
        >>> u.math.cumprod(q)
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        return x.cumprod(axis=axis, dtype=dtype)
    else:
        return jnp.cumprod(x, axis=axis, dtype=dtype)


@set_module_as('saiunit.math')
def nancumprod(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """
    Return the cumulative product of elements along a given axis treating NaNs as one.

    Behaves like :func:`cumprod` but treats NaN values as one, so they do not
    affect the cumulative product.

    Parameters
    ----------
    x : array_like or Quantity
        Input array.
    axis : int, optional
        Axis along which the cumulative product is computed. By default
        the input is flattened.
    dtype : dtype, optional
        Type of the returned array, as well as of the accumulator in which
        the elements are multiplied.

    Returns
    -------
    cumprod : ndarray or Quantity
        A new array holding the cumulative product with NaNs treated as one.
        If `x` carries a unit, the result is a Quantity.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> q = u.math.array([2.0, jnp.nan, 3.0]) * u.meter
        >>> u.math.nancumprod(q)
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        return x.nancumprod(axis=axis, dtype=dtype)
    else:
        return jnp.nancumprod(x, axis=axis, dtype=dtype)


cumproduct = cumprod


# math funcs change unit (binary)
# -------------------------------


def _fun_change_unit_binary(val_fun, unit_fun, x, y, *args, **kwargs):
    x = maybe_custom_array(x)
    y = maybe_custom_array(y)
    args, kwargs = maybe_custom_array_tree((args, kwargs))
    if isinstance(x, Quantity) and isinstance(y, Quantity):
        # x = x.factorless()
        # y = y.factorless()
        return maybe_decimal(
            Quantity(val_fun(x.mantissa, y.mantissa, *args, **kwargs), unit=unit_fun(x.unit, y.unit))
        )
    elif isinstance(x, Quantity):
        # x = x.factorless()
        return maybe_decimal(
            Quantity(val_fun(x.mantissa, y, *args, **kwargs), unit=unit_fun(x.unit, UNITLESS))
        )
    elif isinstance(y, Quantity):
        # y = y.factorless()
        return maybe_decimal(
            Quantity(val_fun(x, y.mantissa, *args, **kwargs), unit=unit_fun(UNITLESS, y.unit))
        )
    else:
        return val_fun(x, y, *args, **kwargs)


@unit_change(lambda ux, uy: ux * uy)
def multiply(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.typing.ArrayLike]:
    """
    Multiply arguments element-wise.

    The resulting unit is the product of the units of the two inputs.

    Parameters
    ----------
    x : array_like or Quantity
        First input array.
    y : array_like or Quantity
        Second input array. If ``x.shape != y.shape``, they must be
        broadcastable to a common shape.

    Returns
    -------
    out : ndarray or Quantity
        The product of `x` and `y`, element-wise. This is a scalar if both
        `x` and `y` are scalars. The resulting unit is ``x.unit * y.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([1.0, 2.0, 3.0]) * u.meter
        >>> b = u.math.array([4.0, 5.0, 6.0]) * u.second
        >>> u.math.multiply(a, b)  # unit is meter * second
    """
    return _fun_change_unit_binary(jnp.multiply,
                                   lambda ux, uy: ux * uy,
                                   x, y)


@unit_change(lambda ux, uy: ux / uy)
def divide(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.typing.ArrayLike]:
    """
    Divide arguments element-wise.

    The resulting unit is the quotient of the units of the two inputs.

    Parameters
    ----------
    x : array_like or Quantity
        Dividend array.
    y : array_like or Quantity
        Divisor array. If ``x.shape != y.shape``, they must be broadcastable
        to a common shape.

    Returns
    -------
    out : ndarray or Quantity
        The quotient of `x` and `y`, element-wise. This is a scalar if both
        `x` and `y` are scalars. The resulting unit is ``x.unit / y.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> distance = u.math.array([10.0, 20.0]) * u.meter
        >>> time = u.math.array([2.0, 4.0]) * u.second
        >>> u.math.divide(distance, time)  # unit is meter / second
    """
    return _fun_change_unit_binary(jnp.divide,
                                   lambda ux, uy: ux / uy,
                                   x, y)


@unit_change(lambda ux, uy: ux * uy)
def cross(
    a: Union[Quantity, jax.typing.ArrayLike],
    b: Union[Quantity, jax.typing.ArrayLike],
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: Optional[int] = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """
    Return the cross product of two (arrays of) vectors.

    The cross product of `a` and `b` in :math:`R^3` is a vector perpendicular
    to both `a` and `b`. The resulting unit is ``a.unit * b.unit``.

    Parameters
    ----------
    a : array_like or Quantity
        Components of the first vector(s).
    b : array_like or Quantity
        Components of the second vector(s).
    axisa : int, optional
        Axis of `a` that defines the vector(s). By default, the last axis.
    axisb : int, optional
        Axis of `b` that defines the vector(s). By default, the last axis.
    axisc : int, optional
        Axis of `c` containing the cross product vector(s). Ignored if both
        input vectors have dimension 2. By default, the last axis.
    axis : int, optional
        If defined, the axis of `a`, `b` and `c` that defines the vector(s)
        and cross product(s). Overrides `axisa`, `axisb` and `axisc`.

    Returns
    -------
    c : ndarray or Quantity
        Vector cross product(s). The resulting unit is
        ``a.unit * b.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([1.0, 0.0, 0.0]) * u.meter
        >>> b = u.math.array([0.0, 1.0, 0.0]) * u.second
        >>> u.math.cross(a, b)  # unit is meter * second
    """
    return _fun_change_unit_binary(jnp.cross,
                                   lambda ux, uy: ux * uy,
                                   a, b,
                                   axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)


@unit_change(lambda ux, uy: ux / uy)
def true_divide(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.typing.ArrayLike]:
    """
    Return a true division of the inputs, element-wise.

    Equivalent to ``divide``. The resulting unit is ``x.unit / y.unit``.

    Parameters
    ----------
    x : array_like or Quantity
        Dividend array.
    y : array_like or Quantity
        Divisor array. If ``x.shape != y.shape``, they must be broadcastable
        to a common shape.

    Returns
    -------
    out : ndarray or Quantity
        The quotient ``x / y``, element-wise. This is a scalar if both `x`
        and `y` are scalars. The resulting unit is ``x.unit / y.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([10.0, 20.0]) * u.meter
        >>> b = u.math.array([2.0, 5.0]) * u.second
        >>> u.math.true_divide(a, b)  # unit is meter / second
    """
    return _fun_change_unit_binary(jnp.true_divide,
                                   lambda ux, uy: ux / uy,
                                   x, y)


@set_module_as('saiunit.math')
def divmod(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Tuple[Union[Quantity, jax.typing.ArrayLike], Union[Quantity, jax.typing.ArrayLike]]:
    """
    Return element-wise quotient and remainder simultaneously.

    Equivalent to ``(x // y, x % y)``, but faster because it avoids
    redundant work. The quotient carries unit ``x.unit / y.unit`` and the
    remainder carries ``x.unit``.

    Parameters
    ----------
    x : array_like or Quantity
        Dividend array.
    y : array_like or Quantity
        Divisor array. If ``x.shape != y.shape``, they must be broadcastable
        to a common shape.

    Returns
    -------
    out1 : ndarray or Quantity
        Element-wise quotient resulting from floor division. Unit is
        ``x.unit / y.unit``.
    out2 : ndarray or Quantity
        Element-wise remainder from floor division. Unit is ``x.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([7.0, 9.0]) * u.meter
        >>> b = u.math.array([2.0, 4.0]) * u.second
        >>> quotient, remainder = u.math.divmod(a, b)
    """
    x = maybe_custom_array(x)
    y = maybe_custom_array(y)
    if isinstance(x, Quantity) and isinstance(y, Quantity):
        r = jnp.divmod(x.mantissa, y.mantissa)
        return Quantity(r[0], unit=x.unit / y.unit), Quantity(r[1], unit=x.unit)
    elif isinstance(x, Quantity):
        r = jnp.divmod(x.mantissa, y)
        return Quantity(r[0], unit=x.unit / UNITLESS), Quantity(r[1], unit=x.unit)
    elif isinstance(y, Quantity):
        r = jnp.divmod(x, y.mantissa)
        return Quantity(r[0], unit=UNITLESS / y.unit), Quantity(r[1], unit=UNITLESS)
    else:
        return jnp.divmod(x, y)


@unit_change(lambda ux, uy: ux * uy)
def convolve(
    a: Union[Quantity, jax.typing.ArrayLike],
    v: Union[Quantity, jax.typing.ArrayLike],
    mode: str = 'full',
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """
    Return the discrete, linear convolution of two one-dimensional sequences.

    The resulting unit is ``a.unit * v.unit``.

    Parameters
    ----------
    a : (N,) array_like or Quantity
        First one-dimensional input array.
    v : (M,) array_like or Quantity
        Second one-dimensional input array.
    mode : {'full', 'valid', 'same'}, optional
        'full' (default): output shape ``(N+M-1,)``.
        'same': output length ``max(M, N)``.
        'valid': output length ``max(M, N) - min(M, N) + 1``.
    precision : optional
        Precision for the computation.
    preferred_element_type : dtype, optional
        Accumulation and result dtype.

    Returns
    -------
    out : ndarray or Quantity
        Discrete, linear convolution of `a` and `v`. The resulting unit is
        ``a.unit * v.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([1.0, 2.0, 3.0]) * u.meter
        >>> v = u.math.array([0.5, 1.0]) * u.second
        >>> u.math.convolve(a, v)  # unit is meter * second
    """
    return _fun_change_unit_binary(
        jnp.convolve,
        lambda ux, uy: ux * uy,
        a, v,
        mode=mode,
        precision=precision,
        preferred_element_type=preferred_element_type
    )


@set_module_as('saiunit.math')
def power(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    """
    First array elements raised to powers from second array, element-wise.

    Raise each base in `x` to the positionally-corresponding power in `y`.
    The exponent `y` must be dimensionless. The resulting unit is
    ``x.unit ** y``.

    Parameters
    ----------
    x : array_like or Quantity
        The bases.
    y : array_like or Quantity
        The exponents. Must be dimensionless if a Quantity. If
        ``x.shape != y.shape``, they must be broadcastable to a common shape.

    Returns
    -------
    out : ndarray or Quantity
        The bases in `x` raised to the exponents in `y`. This is a scalar
        if both `x` and `y` are scalars. The resulting unit is
        ``x.unit ** y``.

    Raises
    ------
    TypeError
        If `y` is a Quantity that is not dimensionless.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> q = u.math.array([2.0, 3.0]) * u.meter
        >>> u.math.power(q, 3)  # unit is meter ** 3
    """
    x = maybe_custom_array(x)
    y = maybe_custom_array(y)
    if isinstance(x, Quantity):
        if isinstance(y, Quantity):
            if not y.is_unitless:
                raise TypeError(
                    f'power requires the exponent "y" to be dimensionless, '
                    f'but got y with unit={y.unit}. '
                    f'Strip the unit from y before raising a Quantity to a power.'
                )
            y = y.mantissa
        return maybe_decimal(Quantity(jnp.power(x.mantissa, y), unit=x.unit ** y))
    elif isinstance(y, Quantity):
        if not y.is_unitless:
            raise TypeError(
                f'power requires the exponent "y" to be dimensionless, '
                f'but got y with unit={y.unit}. '
                f'Strip the unit from y before raising a value to a power.'
            )
        y = y.mantissa
        return maybe_decimal(Quantity(jnp.power(x, y), unit=x ** y))
    else:
        return jnp.power(x, y)


@unit_change(lambda ux, uy: ux / uy)
def floor_divide(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.Array]:
    """
    Return the largest integer smaller or equal to the division of the inputs.

    Equivalent to the Python ``//`` operator. The resulting unit is
    ``x.unit / y.unit``.

    Parameters
    ----------
    x : array_like or Quantity
        Numerator.
    y : array_like or Quantity
        Denominator. If ``x.shape != y.shape``, they must be broadcastable
        to a common shape.

    Returns
    -------
    out : ndarray or Quantity
        ``floor(x / y)``, element-wise. This is a scalar if both `x` and
        `y` are scalars. The resulting unit is ``x.unit / y.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([7.0, 8.0]) * u.meter
        >>> b = u.math.array([2.0, 3.0]) * u.second
        >>> u.math.floor_divide(a, b)  # unit is meter / second
    """
    return _fun_change_unit_binary(jnp.floor_divide, lambda ux, uy: ux / uy, x, y)


@set_module_as('saiunit.math')
def float_power(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: jax.typing.ArrayLike
) -> Union[Quantity, jax.Array]:
    """
    First array elements raised to powers from second array, element-wise.

    Like :func:`power`, but integers, float16, and float32 are promoted to
    floats with a minimum precision of float64 so that the result is always
    inexact. The exponent must be dimensionless.

    Parameters
    ----------
    x : array_like or Quantity
        The bases.
    y : array_like
        The exponents. Must be dimensionless if a Quantity. If
        ``x.shape != y.shape``, they must be broadcastable to a common shape.

    Returns
    -------
    out : ndarray or Quantity
        The bases in `x` raised to the exponents in `y`. This is a scalar
        if both `x` and `y` are scalars. The resulting unit is
        ``x.unit ** y``.

    Raises
    ------
    TypeError
        If `y` is a Quantity that is not dimensionless.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> q = u.math.array([2.0, 3.0]) * u.meter
        >>> u.math.float_power(q, 2)  # unit is meter ** 2
    """
    x = maybe_custom_array(x)
    y = maybe_custom_array(y)
    if isinstance(x, Quantity):
        if isinstance(y, Quantity):
            if not y.is_unitless:
                raise TypeError(
                    f'float_power requires the exponent "y" to be dimensionless, '
                    f'but got y with unit={y.unit}. '
                    f'Strip the unit from y before raising a Quantity to a power.'
                )
            y = y.mantissa
        return maybe_decimal(Quantity(jnp.float_power(x.mantissa, y), unit=x.unit ** y))
    elif isinstance(y, Quantity):
        if not y.is_unitless:
            raise TypeError(
                f'float_power requires the exponent "y" to be dimensionless, '
                f'but got y with unit={y.unit}. '
                f'Strip the unit from y before raising a value to a power.'
            )
        y = y.mantissa
        return maybe_decimal(Quantity(jnp.float_power(x, y), unit=x ** y))
    else:
        return jnp.float_power(x, y)


# linear algebra
# --------------


@unit_change(lambda x, y: x * y)
def dot(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None
) -> Union[jax.Array, Quantity]:
    """
    Compute the dot product of two arrays or quantities.

    The resulting unit is ``a.unit * b.unit``.

    Parameters
    ----------
    a : array_like or Quantity
        First argument.
    b : array_like or Quantity
        Second argument.
    precision : optional
        Either ``None`` (default) or a :class:`~jax.lax.Precision` enum
        value, or a tuple of two such values for `a` and `b`.
    preferred_element_type : dtype, optional
        Accumulation and result dtype.

    Returns
    -------
    output : ndarray or Quantity
        The dot product of the inputs. The resulting unit is
        ``a.unit * b.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([1.0, 2.0, 3.0]) * u.meter
        >>> b = u.math.array([4.0, 5.0, 6.0]) * u.second
        >>> u.math.dot(a, b)  # scalar Quantity with unit meter * second
    """
    return _fun_change_unit_binary(jnp.dot,
                                   lambda x, y: x * y,
                                   a, b,
                                   precision=precision,
                                   preferred_element_type=preferred_element_type)


@unit_change(lambda x, y: x * y)
def multi_dot(
    arrays: Sequence[jax.typing.ArrayLike | Quantity],
    *,
    precision: jax.lax.PrecisionLike = None
) -> Union[jax.Array, Quantity]:
    """
    Efficiently compute matrix products between a sequence of arrays.

    JAX internally uses the opt_einsum library to compute the most efficient
    operation order. The resulting unit is the product of the units of all
    input arrays.

    Parameters
    ----------
    arrays : sequence of array_like or Quantity
        Sequence of arrays or quantities. All must be two-dimensional, except
        the first and last which may be one-dimensional.
    precision : optional
        Either ``None`` (default), or a :class:`~jax.lax.Precision` enum
        value.

    Returns
    -------
    output : ndarray or Quantity
        An array representing the equivalent of ``reduce(jnp.matmul, arrays)``,
        evaluated in the optimal order. The resulting unit is the product of
        all input units.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax
        >>> k1, k2 = jax.random.split(jax.random.key(0))
        >>> a = jax.random.normal(k1, shape=(3, 4)) * u.meter
        >>> b = jax.random.normal(k2, shape=(4, 2)) * u.second
        >>> u.math.multi_dot([a, b])  # unit is meter * second
    """
    new_arrays = []
    unit = UNITLESS
    for arr in arrays:
        arr = maybe_custom_array(arr)
        arr = asarray(arr)
        if isinstance(arr, Quantity):
            unit = unit * arr.unit
            arr = arr.mantissa
        new_arrays.append(arr)
    r = jnp.linalg.multi_dot(new_arrays, precision=precision)
    if unit.is_unitless:
        return r
    return Quantity(r, unit=unit)


@unit_change(lambda ux, uy: ux * uy)
def vdot(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None
) -> Union[jax.Array, Quantity]:
    """
    Perform a conjugate multiplication of two 1D vectors.

    Flattens both inputs and computes the dot product of the conjugate of
    `a` with `b`. The resulting unit is ``a.unit * b.unit``.

    Parameters
    ----------
    a : array_like or Quantity
        First argument.
    b : array_like or Quantity
        Second argument.
    precision : optional
        Either ``None`` (default) or a :class:`~jax.lax.Precision` enum
        value, or a tuple of two such values for `a` and `b`.
    preferred_element_type : dtype, optional
        Accumulation and result dtype.

    Returns
    -------
    output : ndarray or Quantity
        The conjugate dot product of the inputs. The resulting unit is
        ``a.unit * b.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([1.0, 2.0, 3.0]) * u.meter
        >>> b = u.math.array([4.0, 5.0, 6.0]) * u.second
        >>> u.math.vdot(a, b)  # scalar Quantity with unit meter * second
    """
    return _fun_change_unit_binary(jnp.vdot,
                                   lambda x, y: x * y,
                                   a, b,
                                   precision=precision,
                                   preferred_element_type=preferred_element_type)


@unit_change(lambda x, y: x * y)
def vecdot(
    a: jax.typing.ArrayLike | Quantity,
    b: jax.typing.ArrayLike | Quantity,
    /, *,
    axis: int = -1,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None
):
    """
    Perform a conjugate multiplication of two batched vectors.

    Computes the conjugate dot product of `a` and `b` along the given axis.
    The resulting unit is ``a.unit * b.unit``.

    Parameters
    ----------
    a : array_like or Quantity
        Left-hand side array.
    b : array_like or Quantity
        Right-hand side array. Size of ``b[axis]`` must match ``a[axis]``,
        and remaining dimensions must be broadcast-compatible.
    axis : int, optional
        Axis along which to compute the dot product (default: -1).
    precision : optional
        Either ``None`` (default) or a :class:`~jax.lax.Precision` enum
        value, or a tuple of two such values.
    preferred_element_type : dtype, optional
        Accumulation and result dtype.

    Returns
    -------
    output : ndarray or Quantity
        The conjugate dot product of `a` and `b` along `axis`. The resulting
        unit is ``a.unit * b.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([1.0, 2.0, 3.0]) * u.meter
        >>> b = u.math.array([4.0, 5.0, 6.0]) * u.second
        >>> u.math.vecdot(a, b)  # scalar Quantity with unit meter * second
    """
    return _fun_change_unit_binary(jnp.vecdot,
                                   lambda x, y: x * y,
                                   a, b,
                                   axis=axis,
                                   precision=precision,
                                   preferred_element_type=preferred_element_type)


@unit_change(lambda x, y: x * y)
def inner(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
    *,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None
) -> Union[jax.Array, Quantity]:
    """
    Compute the inner product of two arrays or quantities.

    For 1-D arrays, this is the ordinary dot product. For higher dimensions,
    it is the sum product over the last axes. The resulting unit is
    ``a.unit * b.unit``.

    Parameters
    ----------
    a : array_like or Quantity
        First argument.
    b : array_like or Quantity
        Second argument.
    precision : optional
        Either ``None`` (default) or a :class:`~jax.lax.Precision` enum
        value, or a tuple of two such values for `a` and `b`.
    preferred_element_type : dtype, optional
        Accumulation and result dtype.

    Returns
    -------
    output : ndarray or Quantity
        The inner product of the inputs. The resulting unit is
        ``a.unit * b.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([1.0, 2.0, 3.0]) * u.meter
        >>> b = u.math.array([4.0, 5.0, 6.0]) * u.second
        >>> u.math.inner(a, b)  # scalar Quantity with unit meter * second
    """
    return _fun_change_unit_binary(jnp.inner,
                                   lambda x, y: x * y,
                                   a, b,
                                   precision=precision,
                                   preferred_element_type=preferred_element_type)


@unit_change(lambda x, y: x * y)
def outer(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
    out: Optional[Any] = None
) -> Union[jax.Array, Quantity]:
    """
    Compute the outer product of two vectors or quantities.

    Given two 1-D arrays `a` of length M and `b` of length N, the outer
    product is an (M, N) array where ``out[i, j] = a[i] * b[j]``. The
    resulting unit is ``a.unit * b.unit``.

    Parameters
    ----------
    a : array_like or Quantity
        First argument (flattened if not 1-D).
    b : array_like or Quantity
        Second argument (flattened if not 1-D).
    out : ndarray, optional
        A location into which the result is stored. If not provided, a
        freshly-allocated array is returned.

    Returns
    -------
    output : ndarray or Quantity
        The outer product of the inputs, shape ``(M, N)``. The resulting
        unit is ``a.unit * b.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([1.0, 2.0]) * u.meter
        >>> b = u.math.array([3.0, 4.0, 5.0]) * u.second
        >>> u.math.outer(a, b)  # shape (2, 3), unit meter * second
    """
    return _fun_change_unit_binary(jnp.outer,
                                   lambda x, y: x * y,
                                   a, b,
                                   out=out)


@unit_change(lambda x, y: x * y)
def kron(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity]
) -> Union[jax.Array, Quantity]:
    """
    Compute the Kronecker product of two arrays or quantities.

    The resulting unit is ``a.unit * b.unit``.

    Parameters
    ----------
    a : array_like or Quantity
        First input.
    b : array_like or Quantity
        Second input.

    Returns
    -------
    output : ndarray or Quantity
        Kronecker product of `a` and `b`. The resulting unit is
        ``a.unit * b.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([1.0, 2.0]) * u.meter
        >>> b = u.math.array([3.0, 4.0]) * u.second
        >>> u.math.kron(a, b)  # unit is meter * second
    """
    return _fun_change_unit_binary(jnp.kron,
                                   lambda x, y: x * y,
                                   a, b)


@unit_change(lambda x, y: x * y)
def matmul(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None
) -> Union[jax.Array, Quantity]:
    """
    Compute the matrix product of two arrays or quantities.

    The resulting unit is ``a.unit * b.unit``.

    Parameters
    ----------
    a : array_like or Quantity
        First argument.
    b : array_like or Quantity
        Second argument.
    precision : optional
        Either ``None`` (default) or a :class:`~jax.lax.Precision` enum
        value, or a tuple of two such values for `a` and `b`.
    preferred_element_type : dtype, optional
        Accumulation and result dtype.

    Returns
    -------
    output : ndarray or Quantity
        The matrix product of the inputs. The resulting unit is
        ``a.unit * b.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([[1.0, 2.0], [3.0, 4.0]]) * u.meter
        >>> b = u.math.array([[5.0, 6.0], [7.0, 8.0]]) * u.second
        >>> u.math.matmul(a, b)  # shape (2, 2), unit meter * second
    """
    return _fun_change_unit_binary(jnp.matmul,
                                   lambda x, y: x * y,
                                   a, b,
                                   precision=precision,
                                   preferred_element_type=preferred_element_type)


@unit_change(lambda x, y: x * y)
def tensordot(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
    axes: int | Sequence[int] | Sequence[Sequence[int]] = 2,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None
) -> Union[jax.Array, Quantity]:
    """
    Compute tensor dot product along specified axes.

    The resulting unit is ``a.unit * b.unit``.

    Parameters
    ----------
    a : array_like or Quantity
        First tensor.
    b : array_like or Quantity
        Second tensor.
    axes : int or (2,) array_like
        If an int *N*, sum over the last *N* axes of `a` and the first *N*
        axes of `b`. Or a list of two sequences of axis indices.
    precision : optional
        Either ``None`` (default) or a :class:`~jax.lax.Precision` enum
        value, or a tuple of two such values.
    preferred_element_type : dtype, optional
        Accumulation and result dtype.

    Returns
    -------
    output : ndarray or Quantity
        The tensor dot product. The resulting unit is ``a.unit * b.unit``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> a = u.math.array([[1.0, 2.0], [3.0, 4.0]]) * u.meter
        >>> b = u.math.array([[5.0, 6.0], [7.0, 8.0]]) * u.second
        >>> u.math.tensordot(a, b, axes=1)  # unit is meter * second
    """
    return _fun_change_unit_binary(jnp.tensordot,
                                   lambda x, y: x * y,
                                   a, b,
                                   axes=axes,
                                   precision=precision,
                                   preferred_element_type=preferred_element_type)


@set_module_as('saiunit.math')
def matrix_power(
    a: Union[jax.typing.ArrayLike, Quantity],
    n: int
) -> Union[jax.typing.ArrayLike, Quantity]:
    """
    Raise a square matrix to the (integer) power `n`.

    The resulting unit is ``a.unit ** n``.

    Parameters
    ----------
    a : array_like or Quantity
        Square matrix to be "powered".
    n : int
        The exponent can be any integer or zero.

    Returns
    -------
    out : ndarray or Quantity
        The result of raising `a` to the power `n`. The resulting unit is
        ``a.unit ** n``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> m = u.math.array([[1.0, 2.0], [3.0, 4.0]]) * u.meter
        >>> u.math.matrix_power(m, 2)  # unit is meter ** 2
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return maybe_decimal(Quantity(jnp.linalg.matrix_power(a.mantissa, n), unit=a.unit ** n))
    else:
        return jnp.linalg.matrix_power(a, n)


@set_module_as('saiunit.math')
def det(
    a: Union[jax.typing.ArrayLike, Quantity],
) -> Union[jax.typing.ArrayLike, Quantity]:
    """Compute the determinant of a matrix.

    SaiUnit implementation of :func:`numpy.linalg.det`.

    For a Quantity with unit *u* and an ``(N, N)`` matrix the resulting
    unit is ``u ** N``.

    Parameters
    ----------
    a : array_like or Quantity
        Square input of shape ``(..., M, M)``.

    Returns
    -------
    out : ndarray or Quantity
        Determinant(s) of shape ``a.shape[:-2]``.  Carries unit
        ``a.unit ** M`` for an ``(M, M)`` matrix.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> a = jnp.array([[1., 2.],
        ...                [3., 4.]]) * u.meter
        >>> u.linalg.det(a)
        -2. * meter2
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        a_shape = a.shape
        if len(a_shape) >= 2 and a_shape[-1] == a_shape[-2]:
            new_unit = a.unit ** a_shape[-1]
        else:
            msg = "Argument to _det() must have shape [..., n, n], got {}"
            raise ValueError(msg.format(a_shape))
        return Quantity(jnp.linalg.det(a.mantissa), unit=new_unit)
    else:
        return jnp.linalg.det(a)
