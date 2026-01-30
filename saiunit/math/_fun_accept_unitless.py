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

from typing import Union, Optional, Tuple, Any, Callable

import jax
import jax.numpy as jnp

from ._exprel import exprel as _exprel_impl, set_exprel_order
from .._base import Quantity, Unit
from .._misc import set_module_as, maybe_custom_array_tree, maybe_custom_array

__all__ = [
    # math funcs only accept unitless (unary)
    'exprel', 'set_exprel_order', 'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2',
    'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
    'arctanh', 'cos', 'cosh', 'sin', 'sinc', 'sinh', 'tan',
    'tanh', 'deg2rad', 'rad2deg', 'degrees', 'radians', 'angle', 'frexp',

    # math funcs only accept unitless (binary)
    'hypot', 'arctan2', 'logaddexp', 'logaddexp2',
    'corrcoef', 'correlate', 'cov', 'ldexp',

    # Elementwise bit operations (unary)
    'bitwise_not', 'invert',

    # Elementwise bit operations (binary)
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'left_shift', 'right_shift',
]


# math funcs only accept unitless (unary)
# ---------------------------------------

def _fun_accept_unitless_unary(
    func: Callable,
    x: jax.typing.ArrayLike | Quantity,
    *args,
    unit_to_scale: Optional[Unit] = None,
    **kwargs
):
    x = maybe_custom_array(x)
    args = maybe_custom_array_tree(args)
    kwargs = maybe_custom_array_tree(kwargs)

    if isinstance(x, Quantity):
        # x = x.factorless()
        if unit_to_scale is None:
            assert x.dim.is_dimensionless, (
                f'{func} only support dimensionless input. But we got {x}. \n'
                f'If you want to scale the input, please provide the "unit_to_scale" parameter. Or '
                f'convert the input to a dimensionless Quantity manually.'
            )
            x = x.to_decimal()
            return func(x, *args, **kwargs)
        else:
            assert isinstance(unit_to_scale, Unit), f'unit_to_scale should be a Unit instance. Got {unit_to_scale}'
            return func(x.to_decimal(unit_to_scale), *args, **kwargs)
    else:
        assert unit_to_scale is None, (
            f'{func} only support dimensionless input. \n'
            'When the input is not a Quantity, the "unit_to_scale" parameter should not be provided.'
        )
        return func(x, *args, **kwargs)


@set_module_as('saiunit.math')
def exprel(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    """
    Relative error exponential, ``(exp(x) - 1)/x``.

    When ``x`` is near zero, ``exp(x)`` is near 1, so the numerical calculation of ``exp(x) - 1`` can
    suffer from catastrophic loss of precision. ``exprel(x)`` is implemented to avoid the loss of
    precision that occurs when ``x`` is near zero.

    The threshold for switching between Taylor series and direct computation is adaptive
    based on the input dtype for optimal numerical stability.

    Args:
      x: ndarray. Input array. ``x`` must contain real numbers.

    Returns:
      ``(exp(x) - 1)/x``, computed element-wise.

    Notes:
      Use ``saiunit.math.set_exprel_order(n)`` to control the Taylor series order (default: 5).
      Higher values provide better accuracy near x=0 but require more computation.
    """
    x = maybe_custom_array(x)
    return _fun_accept_unitless_unary(_exprel_impl, x)


@set_module_as('saiunit.math')
def exp(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Calculate the exponential of all elements in the input quantity or array.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.exp, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def exp2(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Calculate ``2**p`` for all p in the input quantity or array.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.exp2, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def expm1(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Calculate the exponential of the input elements minus 1.

    Calculate ``exp(x) - 1`` for all elements in the array.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.expm1, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def log(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Natural logarithm, element-wise.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.log, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def log10(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Base-10 logarithm of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.log10, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def log1p(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Natural logarithm of 1 + the input elements.

    Calculates ``log(1 + x)``.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.log1p, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def log2(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Base-2 logarithm of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.log2, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def arccos(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Compute the arccosine of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.arccos, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def arccosh(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Compute the hyperbolic arccosine of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.arccosh, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def arcsin(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Compute the arcsine of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.arcsin, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def arcsinh(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Compute the hyperbolic arcsine of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.arcsinh, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def arctan(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Compute the arctangent of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.arctan, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def arctanh(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Compute the hyperbolic arctangent of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.arctanh, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def cos(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Compute the cosine of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.cos, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def cosh(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Compute the hyperbolic cosine of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.cosh, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def sin(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Compute the sine of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.sin, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def sinc(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Compute the sinc function of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.sinc, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def sinh(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Compute the hyperbolic sine of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.sinh, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def tan(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Compute the tangent of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.tan, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def tanh(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Compute the hyperbolic tangent of the input elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.tanh, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def deg2rad(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.deg2rad, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def rad2deg(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.rad2deg, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def degrees(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.degrees, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def radians(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.radians, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def angle(
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Return the angle of the complex argument.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.angle, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def frexp(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> Tuple[jax.Array, jax.Array]:
    """
    Decompose the elements of x into mantissa and twos exponent.

    Returns (`mantissa`, `exponent`), where ``x = mantissa * 2**exponent``.
    The mantissa lies in the open interval(-1, 1), while the twos
    exponent is a signed integer.

    Parameters
    ----------
    x : array_like, Quantity
      Array of numbers to be decomposed.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    mantissa : ndarray
      Floating values between -1 and 1.
      This is a scalar if `x` is a scalar.
    exponent : ndarray
      Integer exponents of 2.
      This is a scalar if `x` is a scalar.
    """
    return _fun_accept_unitless_unary(jnp.frexp, x, unit_to_scale=unit_to_scale)


# math funcs only accept unitless (binary)
# ----------------------------------------


def _fun_accept_unitless_binary(
    func: Callable,
    x: jax.typing.ArrayLike | Quantity,
    y: jax.typing.ArrayLike | Quantity,
    *args,
    unit_to_scale: Optional[Unit] = None,
    **kwargs
):
    x = maybe_custom_array(x)
    y = maybe_custom_array(y)
    args = maybe_custom_array_tree(args)
    kwargs = maybe_custom_array_tree(kwargs)

    if isinstance(x, Quantity):
        # x = x.factorless()
        if unit_to_scale is None:
            assert x.dim.is_dimensionless, (
                f'{func} only support dimensionless input. But we got {x}. \n'
                f'If you want to scale the input, please provide the "unit_to_scale" parameter. Or '
                f'convert the input to a dimensionless Quantity manually.'
            )
            x = x.to_decimal()
        else:
            assert isinstance(unit_to_scale, Unit), f'unit_to_scale should be a Unit instance. Got {unit_to_scale}'
            x = x.to_decimal(unit_to_scale)
    if isinstance(y, Quantity):
        # y = y.factorless()
        if unit_to_scale is None:
            assert y.dim.is_dimensionless, (f'Input should be dimensionless for the function "{func}" '
                                            f'when scaling "unit_to_scale" is not provided.')
            y = y.to_decimal()
        else:
            assert isinstance(unit_to_scale, Unit), f'unit_to_scale should be a Unit instance. Got {unit_to_scale}'
            y = y.to_decimal(unit_to_scale)
    return func(x, y, *args, **kwargs)


@set_module_as('saiunit.math')
def hypot(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Given the “legs” of a right triangle, return its hypotenuse.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    y : array_like, Quantity
      Input array or Quantity.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_binary(jnp.hypot, x, y, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def arctan2(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    y : array_like, Quantity
      Input array or Quantity.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_binary(jnp.arctan2, x, y, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def logaddexp(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Logarithm of the sum of exponentiations of the inputs.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    y : array_like, Quantity
      Input array or Quantity.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_binary(jnp.logaddexp, x, y, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def logaddexp2(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Logarithm of the sum of exponentiations of the inputs in base-2.

    Parameters
    ----------
    x : array_like, Quantity
      Input array or Quantity.
    y : array_like, Quantity
      Input array or Quantity.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_binary(jnp.logaddexp2, x, y, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def corrcoef(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity] = None,
    rowvar: bool = True,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""
    Return Pearson product-moment correlation coefficients.

    Please refer to the documentation for `cov` for more detail.  The
    relationship between the correlation coefficient matrix, `R`, and the
    covariance matrix, `C`, is

    .. math:: R_{ij} = \frac{ C_{ij} } { \sqrt{ C_{ii} C_{jj} } }

    The values of `R` are between -1 and 1, inclusive.

    Parameters
    ----------
    x : array_like, Quantity
      A 1-D or 2-D array containing multiple variables and observations.
      Each row of `x` represents a variable, and each column a single
      observation of all those variables. Also see `rowvar` below.
    y : array_like, Quantity, optional
      An additional set of variables and observations. `y` has the same
      shape as `x`.
    rowvar : bool, optional
      If `rowvar` is True (default), then each row represents a
      variable, with observations in the columns. Otherwise, the relationship
      is transposed: each column represents a variable, while the rows
      contain observations.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    R : ndarray
      The correlation coefficient matrix of the variables.
    """
    return _fun_accept_unitless_binary(jnp.corrcoef, x, y, rowvar=rowvar, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.math')
def correlate(
    a: Union[jax.typing.ArrayLike, Quantity],
    v: Union[jax.typing.ArrayLike, Quantity],
    mode: str = 'valid',
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""
    Cross-correlation of two 1-dimensional sequences.

    This function computes the correlation as generally defined in signal
    processing texts:

    .. math:: c_k = \sum_n a_{n+k} \cdot \overline{v}_n

    with a and v sequences being zero-padded where necessary and
    :math:`\overline x` denoting complex conjugation.

    Parameters
    ----------
    a, v : array_like, Quantity
      Input sequences.
    mode : {'valid', 'same', 'full'}, optional
      Refer to the `convolve` docstring.  Note that the default
      is 'valid', unlike `convolve`, which uses 'full'.
    precision : Optional. Either ``None``, which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value
      (``Precision.DEFAULT``, ``Precision.HIGH`` or ``Precision.HIGHEST``), a
      string (e.g. 'highest' or 'fastest', see the
      ``jax.default_matmul_precision`` context manager), or a tuple of two
      :class:`~jax.lax.Precision` enums or strings indicating precision of
      ``lhs`` and ``rhs``.
    preferred_element_type : Optional. Either ``None``, which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : ndarray
      Discrete cross-correlation of `a` and `v`.
    """
    return _fun_accept_unitless_binary(
        jnp.correlate, a, v,
        mode=mode, precision=precision,
        preferred_element_type=preferred_element_type,
        unit_to_scale=unit_to_scale
    )


@set_module_as('saiunit.math')
def cov(
    m: Union[jax.typing.ArrayLike, Quantity],
    y: Optional[Union[jax.typing.ArrayLike, Quantity]] = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    fweights: Optional[jax.typing.ArrayLike] = None,
    aweights: Optional[jax.typing.ArrayLike] = None,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """
    Estimate a covariance matrix, given data and weights.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.

    See the notes for an outline of the algorithm.

    Parameters
    ----------
    m : array_like, Quantity
      A 1-D or 2-D array containing multiple variables and observations.
      Each row of `m` represents a variable, and each column a single
      observation of all those variables. Also see `rowvar` below.
    y : array_like, Quantity or optional
      An additional set of variables and observations. `y` has the same form
      as that of `m`.
    rowvar : bool, optional
      If `rowvar` is True (default), then each row represents a
      variable, with observations in the columns. Otherwise, the relationship
      is transposed: each column represents a variable, while the rows
      contain observations.
    bias : bool, optional
      Default normalization (False) is by ``(N - 1)``, where ``N`` is the
      number of observations given (unbiased estimate). If `bias` is True,
      then normalization is by ``N``. These values can be overridden by using
      the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
      If not ``None`` the default value implied by `bias` is overridden.
      Note that ``ddof=1`` will return the unbiased estimate, even if both
      `fweights` and `aweights` are specified, and ``ddof=0`` will return
      the simple average. See the notes for the details. The default value
      is ``None``.
    fweights : array_like, int, optional
      1-D array of integer frequency weights; the number of times each
      observation vector should be repeated.
    aweights : array_like, optional
      1-D array of observation vector weights. These relative weights are
      typically large for observations considered "important" and smaller for
      observations considered less "important". If ``ddof=0`` the array of
      weights can be used to assign probabilities to observation vectors.
    unit_to_scale : Unit, optional
      The unit to scale the ``x``.

    Returns
    -------
    out : ndarray
      The covariance matrix of the variables.
    """
    return _fun_accept_unitless_binary(
        jnp.cov, m, y,
        rowvar=rowvar, bias=bias, ddof=ddof, fweights=fweights,
        aweights=aweights, unit_to_scale=unit_to_scale
    )


@set_module_as('saiunit.math')
def ldexp(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: jax.typing.ArrayLike
) -> Union[Quantity, jax.typing.ArrayLike]:
    """
    Returns x * 2**y, element-wise.

    The mantissas `x` and twos exponents `y` are used to construct
    floating point numbers ``x * 2**y``.

    Parameters
    ----------
    x : array_like, Quantity
      Array of multipliers.
    y : array_like, int
      Array of twos exponents.
      If ``x.shape != y.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).

    Returns
    -------
    out : ndarray, quantity or scalar
      The result of ``x * 2**y``.
      This is a scalar if both `x` and `y` are scalars.

      This is a Quantity if the product of the square of the unit of `x` and the unit of `y` is not dimensionless.
    """
    x, y = maybe_custom_array_tree((x, y))
    if isinstance(x, Quantity):
        assert x.dim.is_dimensionless, f'Expected dimensionless array, got {x}'
        x = x.mantissa
    return jnp.ldexp(x, y)


# Elementwise bit operations (unary)
# ----------------------------------


@set_module_as('saiunit.math')
def bitwise_not(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    """
    Compute the bit-wise NOT of an array, element-wise.

    Parameters
    ----------
    x: array_like, quantity
      Input array.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.bitwise_not, x)


@set_module_as('saiunit.math')
def invert(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    """
    Compute bit-wise inversion, or bit-wise NOT, element-wise.

    Parameters
    ----------
    x: array_like, quantity
      Input array.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_accept_unitless_unary(jnp.invert, x)


# Elementwise bit operations (binary)
# -----------------------------------


def _fun_unitless_binary(func, x, y, *args, **kwargs):
    x = maybe_custom_array(x)
    y = maybe_custom_array(y)
    args = maybe_custom_array_tree(args)
    kwargs = maybe_custom_array_tree(kwargs)

    if isinstance(x, Quantity):
        # x = x.factorless()
        assert x.dim.is_dimensionless, f'Expected dimensionless array, got {x}'
        x = x.to_decimal()
    if isinstance(y, Quantity):
        # y = y.factorless()
        assert y.dim.is_dimensionless, f'Expected dimensionless array, got {y}'
        y = y.to_decimal()
    return func(x, y, *args, **kwargs)


@set_module_as('saiunit.math')
def bitwise_and(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> jax.Array:
    """
    Compute the bit-wise AND of two arrays element-wise.

    Parameters
    ----------
    x: array_like, quantity
      Input array.
    y: array_like, quantity
      Input array.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_unitless_binary(jnp.bitwise_and, x, y)


@set_module_as('saiunit.math')
def bitwise_or(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> jax.Array:
    """
    Compute the bit-wise OR of two arrays element-wise.

    Parameters
    ----------
    x: array_like, quantity
      Input array.
    y: array_like, quantity
      Input array.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_unitless_binary(jnp.bitwise_or, x, y)


@set_module_as('saiunit.math')
def bitwise_xor(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> jax.Array:
    """
    Compute the bit-wise XOR of two arrays element-wise.

    Parameters
    ----------
    x: array_like, quantity
      Input array.
    y: array_like, quantity
      Input array.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_unitless_binary(jnp.bitwise_xor, x, y)


@set_module_as('saiunit.math')
def left_shift(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> jax.Array:
    """
    Shift the bits of an integer to the left.

    Parameters
    ----------
    x: array_like, quantity
      Input array.
    y: array_like, quantity
      Input array.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_unitless_binary(jnp.left_shift, x, y)


@set_module_as('saiunit.math')
def right_shift(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> jax.Array:
    """
    Shift the bits of an integer to the right.

    Parameters
    ----------
    x: array_like, quantity
      Input array.
    y: array_like, quantity
      Input array.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    return _fun_unitless_binary(jnp.right_shift, x, y)
