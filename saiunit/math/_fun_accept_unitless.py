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

from typing import Union, Optional, Tuple, Any, Callable

from saiunit._jax_compat import jax, jnp, ArrayLike

from saiunit._backend import get_backend
from saiunit._base_unit import Unit
from saiunit._base_quantity import Quantity
from ._fun_keep_unit import _resolve_op
from saiunit._misc import set_module_as, maybe_custom_array_tree, maybe_custom_array
from ._exprel import exprel as _exprel_impl, set_exprel_order

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


def _func_name(func) -> str:
    if isinstance(func, str):
        return func
    return getattr(func, '__name__', repr(func))


def _quantity_summary(x: Quantity) -> str:
    return f"Quantity(unit={x.unit}, dim={x.dim})"


def _dimensionless_required_message(func: Callable, x: Quantity, arg_name: str = 'x') -> str:
    name = _func_name(func)
    summary = _quantity_summary(x)
    return (
        f'{name} requires a dimensionless "{arg_name}" when "unit_to_scale" is not provided. '
        f'Got {summary}. '
        f'Pass "unit_to_scale=<Unit>" to scale before applying {name}, or convert explicitly to '
        f'a dimensionless value first.'
    )


def _invalid_unit_to_scale_type_message(func: Callable, unit_to_scale: Any) -> str:
    name = _func_name(func)
    return (
        f'{name} expects "unit_to_scale" to be a Unit instance, but got '
        f'{type(unit_to_scale).__name__}: {unit_to_scale!r}.'
    )


def _unit_to_scale_without_quantity_message(func: Callable, x: Any) -> str:
    name = _func_name(func)
    return (
        f'{name} received "unit_to_scale" but input "x" is not a Quantity '
        f'(got type {type(x).__name__}). '
        f'Remove "unit_to_scale" or pass a Quantity input.'
    )


def _fun_accept_unitless_unary(
    func: Callable | str,
    x: ArrayLike | Quantity,
    *args,
    unit_to_scale: Optional[Unit] = None,
    **kwargs
):
    x = maybe_custom_array(x)
    args = maybe_custom_array_tree(args)
    kwargs = maybe_custom_array_tree(kwargs)

    if isinstance(x, Quantity):
        if unit_to_scale is None:
            if not x.dim.is_dimensionless:
                raise TypeError(_dimensionless_required_message(func, x, arg_name='x'))  # type: ignore[arg-type]
            x = x.to_decimal()
        else:
            if not isinstance(unit_to_scale, Unit):
                raise TypeError(_invalid_unit_to_scale_type_message(func, unit_to_scale))
            x = x.to_decimal(unit_to_scale)
        xp = get_backend(x)
        func = _resolve_op(func, xp)
        return func(x, *args, **kwargs)  # type: ignore[operator]
    else:
        if unit_to_scale is not None:
            raise TypeError(_unit_to_scale_without_quantity_message(func, x))  # type: ignore[arg-type]
        xp = get_backend(x)
        func = _resolve_op(func, xp)
        return func(x, *args, **kwargs)  # type: ignore[operator]


@set_module_as('saiunit.math')
def exprel(
    x: Union[Quantity, ArrayLike],
    **kwargs,
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
    return _fun_accept_unitless_unary(_exprel_impl, x, **kwargs)


@set_module_as('saiunit.math')
def exp(
    x: Union[Quantity, ArrayLike],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Calculate the exponential of all elements in the input.

    If ``x`` is a Quantity with physical units, ``unit_to_scale`` must be
    provided to convert ``x`` to a dimensionless value first.

    Parameters
    ----------
    x : array_like or Quantity
        Input array or Quantity.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number before
        applying the exponential.

    Returns
    -------
    out : jax.Array
        Element-wise exponential.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.exp(jnp.array([0.0, 1.0]))
        Array([1.       , 2.7182817], dtype=float32)
    """
    return _fun_accept_unitless_unary('exp', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def exp2(
    x: Union[Quantity, ArrayLike],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Calculate ``2**x`` element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input array or Quantity.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise ``2**x``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.exp2(jnp.array([0.0, 1.0, 2.0]))
        Array([1., 2., 4.], dtype=float32)
    """
    return _fun_accept_unitless_unary('exp2', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def expm1(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Calculate ``exp(x) - 1`` element-wise with improved precision near zero.

    Parameters
    ----------
    x : array_like or Quantity
        Input array or Quantity.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise ``exp(x) - 1``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.expm1(jnp.array([0.0, 1e-10]))
        Array([0.e+00, 1.e-10], dtype=float32)
    """
    return _fun_accept_unitless_unary('expm1', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def log(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Natural logarithm, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input array or Quantity. Must be positive.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise natural logarithm.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.log(jnp.array([1.0, jnp.e, jnp.e**2]))
        Array([0., 1., 2.], dtype=float32)
    """
    return _fun_accept_unitless_unary('log', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def log10(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Base-10 logarithm, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input array or Quantity. Must be positive.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise base-10 logarithm.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.log10(jnp.array([1.0, 10.0, 100.0]))
        Array([0., 1., 2.], dtype=float32)
    """
    return _fun_accept_unitless_unary('log10', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def log1p(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Natural logarithm of ``1 + x``, element-wise.

    More accurate than ``log(1 + x)`` for small ``x``.

    Parameters
    ----------
    x : array_like or Quantity
        Input array or Quantity.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise ``log(1 + x)``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.log1p(jnp.array([0.0, 1e-10]))
        Array([0.e+00, 1.e-10], dtype=float32)
    """
    return _fun_accept_unitless_unary('log1p', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def log2(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Base-2 logarithm, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input array or Quantity. Must be positive.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise base-2 logarithm.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.log2(jnp.array([1.0, 2.0, 4.0]))
        Array([0., 1., 2.], dtype=float32)
    """
    return _fun_accept_unitless_unary('log2', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def arccos(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Inverse cosine, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input values in the range ``[-1, 1]``.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Angle in radians, in ``[0, pi]``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.arccos(jnp.array([1.0, 0.0, -1.0]))
        Array([0.       , 1.5707964, 3.1415927], dtype=float32)
    """
    return _fun_accept_unitless_unary('arccos', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def arccosh(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Inverse hyperbolic cosine, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input values, must be >= 1.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise inverse hyperbolic cosine.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.arccosh(jnp.array([1.0, 2.0, 3.0]))
        Array([0.       , 1.3169578, 1.7627472], dtype=float32)
    """
    return _fun_accept_unitless_unary('arccosh', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def arcsin(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Inverse sine, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input values in the range ``[-1, 1]``.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Angle in radians, in ``[-pi/2, pi/2]``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.arcsin(jnp.array([0.0, 0.5, 1.0]))
        Array([0.       , 0.5235988, 1.5707964], dtype=float32)
    """
    return _fun_accept_unitless_unary('arcsin', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def arcsinh(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Inverse hyperbolic sine, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input values.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise inverse hyperbolic sine.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.arcsinh(jnp.array([0.0, 1.0]))
        Array([0.       , 0.8813736], dtype=float32)
    """
    return _fun_accept_unitless_unary('arcsinh', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def arctan(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Inverse tangent, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input values.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Angle in radians, in ``[-pi/2, pi/2]``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.arctan(jnp.array([0.0, 1.0]))
        Array([0.       , 0.7853982], dtype=float32)
    """
    return _fun_accept_unitless_unary('arctan', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def arctanh(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Inverse hyperbolic tangent, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input values in the range ``(-1, 1)``.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise inverse hyperbolic tangent.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.arctanh(jnp.array([0.0, 0.5]))
        Array([0.       , 0.5493061], dtype=float32)
    """
    return _fun_accept_unitless_unary('arctanh', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def cos(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Cosine, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Angle in radians.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise cosine.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.cos(jnp.array([0.0, jnp.pi / 2, jnp.pi]))
        Array([ 1.0000000e+00, -4.3711388e-08, -1.0000000e+00], dtype=float32)
    """
    return _fun_accept_unitless_unary('cos', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def cosh(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Hyperbolic cosine, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input values.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise hyperbolic cosine.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.cosh(jnp.array([0.0, 1.0]))
        Array([1.       , 1.5430806], dtype=float32)
    """
    return _fun_accept_unitless_unary('cosh', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def sin(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Sine, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Angle in radians.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise sine.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.sin(jnp.array([0.0, jnp.pi / 2, jnp.pi]))
        Array([ 0.0000000e+00,  1.0000000e+00, -8.7422777e-08], dtype=float32)
    """
    return _fun_accept_unitless_unary('sin', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def sinc(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Normalized sinc function, ``sin(pi*x) / (pi*x)``, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input values.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise sinc.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.sinc(jnp.array([0.0, 1.0]))
        Array([ 1.0000000e+00, -3.8981719e-09], dtype=float32)
    """
    return _fun_accept_unitless_unary('sinc', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def sinh(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Hyperbolic sine, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input values.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise hyperbolic sine.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.sinh(jnp.array([0.0, 1.0]))
        Array([0.       , 1.1752012], dtype=float32)
    """
    return _fun_accept_unitless_unary('sinh', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def tan(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Tangent, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Angle in radians.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise tangent.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.tan(jnp.array([0.0, jnp.pi / 4]))
        Array([0.       , 1.0000001], dtype=float32)
    """
    return _fun_accept_unitless_unary('tan', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def tanh(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Hyperbolic tangent, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Input values.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Element-wise hyperbolic tangent, in ``(-1, 1)``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.tanh(jnp.array([0.0, 1.0]))
        Array([0.       , 0.7615942], dtype=float32)
    """
    return _fun_accept_unitless_unary('tanh', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def deg2rad(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : array_like or Quantity
        Angle in degrees.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Angle in radians.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.deg2rad(jnp.array([0.0, 90.0, 180.0]))
        Array([0.       , 1.5707964, 3.1415927], dtype=float32)
    """
    return _fun_accept_unitless_unary('deg2rad', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def rad2deg(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : array_like or Quantity
        Angle in radians.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Angle in degrees.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.rad2deg(jnp.array([0.0, jnp.pi / 2, jnp.pi]))
        Array([  0.,  90., 180.], dtype=float32)
    """
    return _fun_accept_unitless_unary('rad2deg', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def degrees(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Convert angles from radians to degrees (alias for :func:`rad2deg`).

    Parameters
    ----------
    x : array_like or Quantity
        Angle in radians.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Angle in degrees.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.degrees(jnp.array([0.0, jnp.pi]))
        Array([  0., 180.], dtype=float32)
    """
    return _fun_accept_unitless_unary('degrees', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def radians(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Convert angles from degrees to radians (alias for :func:`deg2rad`).

    Parameters
    ----------
    x : array_like or Quantity
        Angle in degrees.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Angle in radians.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.radians(jnp.array([0.0, 180.0]))
        Array([0.       , 3.1415927], dtype=float32)
    """
    return _fun_accept_unitless_unary('radians', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def angle(
    x: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Return the angle of the complex argument, element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        Complex-valued input.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    out : jax.Array
        Angle in radians, in ``(-pi, pi]``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.angle(jnp.array([1.0 + 1.0j, 1.0 + 0.0j]))
        Array([0.7853982, 0.       ], dtype=float32)
    """
    return _fun_accept_unitless_unary('angle', x, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def frexp(
    x: Union[Quantity, ArrayLike],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> Tuple[jax.Array, jax.Array]:
    """
    Decompose elements into mantissa and base-2 exponent.

    Returns ``(mantissa, exponent)`` such that
    ``x = mantissa * 2**exponent``.

    Parameters
    ----------
    x : array_like or Quantity
        Array to decompose.
    unit_to_scale : Unit, optional
        Unit used to convert ``x`` to a dimensionless number first.

    Returns
    -------
    mantissa : jax.Array
        Floating values in ``(-1, 1)``.
    exponent : jax.Array
        Integer exponents of 2.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> m, e = u.math.frexp(jnp.array([1.0, 2.0, 4.0]))
        >>> m
        Array([0.5, 0.5, 0.5], dtype=float32)
        >>> e
        Array([1, 2, 3], dtype=int32)
    """
    return _fun_accept_unitless_unary('frexp', x, unit_to_scale=unit_to_scale, **kwargs)


# math funcs only accept unitless (binary)
# ----------------------------------------


def _fun_accept_unitless_binary(
    func: Callable | str,
    x: ArrayLike | Quantity,
    y: ArrayLike | Quantity,
    *args,
    unit_to_scale: Optional[Unit] = None,
    **kwargs
):
    x = maybe_custom_array(x)
    y = maybe_custom_array(y)
    args = maybe_custom_array_tree(args)
    kwargs = maybe_custom_array_tree(kwargs)

    if isinstance(x, Quantity):
        if unit_to_scale is None:
            if not x.dim.is_dimensionless:
                raise TypeError(_dimensionless_required_message(func, x, arg_name='x'))  # type: ignore[arg-type]
            x = x.to_decimal()
        else:
            if not isinstance(unit_to_scale, Unit):
                raise TypeError(_invalid_unit_to_scale_type_message(func, unit_to_scale))
            x = x.to_decimal(unit_to_scale)
    if isinstance(y, Quantity):
        if unit_to_scale is None:
            if not y.dim.is_dimensionless:
                raise TypeError(_dimensionless_required_message(func, y, arg_name='y'))  # type: ignore[arg-type]
            y = y.to_decimal()
        else:
            if not isinstance(unit_to_scale, Unit):
                raise TypeError(_invalid_unit_to_scale_type_message(func, unit_to_scale))
            y = y.to_decimal(unit_to_scale)
    xp = get_backend(x, y)
    func = _resolve_op(func, xp)
    return func(x, y, *args, **kwargs)  # type: ignore[operator]


@set_module_as('saiunit.math')
def hypot(
    x: Union[ArrayLike, Quantity],
    y: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Given the legs of a right triangle, return its hypotenuse.

    Computes ``sqrt(x**2 + y**2)`` element-wise.

    Parameters
    ----------
    x : array_like or Quantity
        First leg.
    y : array_like or Quantity
        Second leg. Must be broadcastable with ``x``.
    unit_to_scale : Unit, optional
        Unit used to convert both inputs to dimensionless numbers.

    Returns
    -------
    out : jax.Array
        Hypotenuse values.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.hypot(jnp.array([3.0]), jnp.array([4.0]))
        Array([5.], dtype=float32)
    """
    return _fun_accept_unitless_binary('hypot', x, y, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def arctan2(
    x: Union[ArrayLike, Quantity],
    y: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Element-wise arc tangent of ``x / y`` choosing the quadrant correctly.

    Parameters
    ----------
    x : array_like or Quantity
        y-coordinates (numerator).
    y : array_like or Quantity
        x-coordinates (denominator). Must be broadcastable with ``x``.
    unit_to_scale : Unit, optional
        Unit used to convert both inputs to dimensionless numbers.

    Returns
    -------
    out : jax.Array
        Angle in radians, in ``(-pi, pi]``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.arctan2(jnp.array([1.0, -1.0]),
        ...                 jnp.array([1.0, 1.0]))
        Array([ 0.7853982, -0.7853982], dtype=float32)
    """
    return _fun_accept_unitless_binary('arctan2', x, y, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def logaddexp(
    x: Union[ArrayLike, Quantity],
    y: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Logarithm of the sum of exponentiations of the inputs.

    Computes ``log(exp(x) + exp(y))`` in a numerically stable way.

    Parameters
    ----------
    x : array_like or Quantity
        First input.
    y : array_like or Quantity
        Second input. Must be broadcastable with ``x``.
    unit_to_scale : Unit, optional
        Unit used to convert both inputs to dimensionless numbers.

    Returns
    -------
    out : jax.Array
        Element-wise ``log(exp(x) + exp(y))``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.logaddexp(jnp.array([1.0]), jnp.array([2.0]))
        Array([2.3132617], dtype=float32)
    """
    return _fun_accept_unitless_binary('logaddexp', x, y, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def logaddexp2(
    x: Union[ArrayLike, Quantity],
    y: Union[ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
) -> jax.Array:
    """
    Logarithm of the sum of exponentiations of the inputs in base 2.

    Computes ``log2(2**x + 2**y)`` in a numerically stable way.

    Parameters
    ----------
    x : array_like or Quantity
        First input.
    y : array_like or Quantity
        Second input. Must be broadcastable with ``x``.
    unit_to_scale : Unit, optional
        Unit used to convert both inputs to dimensionless numbers.

    Returns
    -------
    out : jax.Array
        Element-wise ``log2(2**x + 2**y)``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.logaddexp2(jnp.array([1.0]), jnp.array([2.0]))
        Array([2.321928], dtype=float32)
    """
    return _fun_accept_unitless_binary('logaddexp2', x, y, unit_to_scale=unit_to_scale, **kwargs)


@set_module_as('saiunit.math')
def corrcoef(
    x: Union[ArrayLike, Quantity],
    y: Optional[Union[ArrayLike, Quantity]] = None,
    rowvar: bool = True,
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
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
    return _fun_accept_unitless_binary('corrcoef', x, y, rowvar=rowvar, unit_to_scale=unit_to_scale, **kwargs)  # type: ignore[arg-type]


@set_module_as('saiunit.math')
def correlate(
    a: Union[ArrayLike, Quantity],
    v: Union[ArrayLike, Quantity],
    mode: str = 'valid',
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None,
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
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
        'correlate', a, v,
        mode=mode, precision=precision,
        preferred_element_type=preferred_element_type,
        unit_to_scale=unit_to_scale,
        **kwargs,
    )


@set_module_as('saiunit.math')
def cov(
    m: Union[ArrayLike, Quantity],
    y: Optional[Union[ArrayLike, Quantity]] = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    fweights: Optional[ArrayLike] = None,
    aweights: Optional[ArrayLike] = None,
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
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
        'cov', m, y,  # type: ignore[arg-type]
        rowvar=rowvar, bias=bias, ddof=ddof, fweights=fweights,
        aweights=aweights, unit_to_scale=unit_to_scale,
        **kwargs,
    )


@set_module_as('saiunit.math')
def ldexp(
    x: Union[Quantity, ArrayLike],
    y: ArrayLike,
    **kwargs,
) -> Union[Quantity, ArrayLike]:
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
        if not x.dim.is_dimensionless:
            raise TypeError(_dimensionless_required_message('ldexp', x, arg_name='x'))
        x = x.mantissa
    xp = get_backend(x, y)
    return _resolve_op('ldexp', xp)(x, y, **kwargs)


# Elementwise bit operations (unary)
# ----------------------------------


@set_module_as('saiunit.math')
def bitwise_not(
    x: Union[Quantity, ArrayLike],
    **kwargs,
) -> jax.Array:
    """
    Compute bit-wise NOT, element-wise.

    The input must be dimensionless.

    Parameters
    ----------
    x : array_like or Quantity
        Input array of integers or booleans.

    Returns
    -------
    out : jax.Array
        Element-wise bit-wise NOT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.bitwise_not(jnp.array([True, False]))
        Array([False,  True], dtype=bool)
    """
    return _fun_accept_unitless_unary('bitwise_not', x, **kwargs)


@set_module_as('saiunit.math')
def invert(
    x: Union[Quantity, ArrayLike],
    **kwargs,
) -> jax.Array:
    """
    Compute bit-wise inversion (NOT), element-wise.

    Alias for :func:`bitwise_not`. The input must be dimensionless.

    Parameters
    ----------
    x : array_like or Quantity
        Input array of integers or booleans.

    Returns
    -------
    out : jax.Array
        Element-wise bit-wise inversion.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.invert(jnp.array([True, False]))
        Array([False,  True], dtype=bool)
    """
    return _fun_accept_unitless_unary('invert', x, **kwargs)


# Elementwise bit operations (binary)
# -----------------------------------


def _fun_unitless_binary(func, x, y, *args, **kwargs):
    x = maybe_custom_array(x)
    y = maybe_custom_array(y)
    args = maybe_custom_array_tree(args)
    kwargs = maybe_custom_array_tree(kwargs)

    if isinstance(x, Quantity):
        if not x.dim.is_dimensionless:
            raise TypeError(_dimensionless_required_message(func, x, arg_name='x'))
        x = x.to_decimal()
    if isinstance(y, Quantity):
        if not y.dim.is_dimensionless:
            raise TypeError(_dimensionless_required_message(func, y, arg_name='y'))
        y = y.to_decimal()
    xp = get_backend(x, y)
    func = _resolve_op(func, xp)
    return func(x, y, *args, **kwargs)


@set_module_as('saiunit.math')
def bitwise_and(
    x: Union[Quantity, ArrayLike],
    y: Union[Quantity, ArrayLike],
    **kwargs,
) -> jax.Array:
    """
    Compute bit-wise AND of two arrays, element-wise.

    Both inputs must be dimensionless.

    Parameters
    ----------
    x : array_like or Quantity
        First input.
    y : array_like or Quantity
        Second input.

    Returns
    -------
    out : jax.Array
        Element-wise bit-wise AND.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.bitwise_and(jnp.array([True, False]),
        ...                     jnp.array([True, True]))
        Array([ True, False], dtype=bool)
    """
    return _fun_unitless_binary('bitwise_and', x, y, **kwargs)


@set_module_as('saiunit.math')
def bitwise_or(
    x: Union[Quantity, ArrayLike],
    y: Union[Quantity, ArrayLike],
    **kwargs,
) -> jax.Array:
    """
    Compute bit-wise OR of two arrays, element-wise.

    Both inputs must be dimensionless.

    Parameters
    ----------
    x : array_like or Quantity
        First input.
    y : array_like or Quantity
        Second input.

    Returns
    -------
    out : jax.Array
        Element-wise bit-wise OR.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.bitwise_or(jnp.array([True, False]),
        ...                    jnp.array([False, False]))
        Array([ True, False], dtype=bool)
    """
    return _fun_unitless_binary('bitwise_or', x, y, **kwargs)


@set_module_as('saiunit.math')
def bitwise_xor(
    x: Union[Quantity, ArrayLike],
    y: Union[Quantity, ArrayLike],
    **kwargs,
) -> jax.Array:
    """
    Compute bit-wise XOR of two arrays, element-wise.

    Both inputs must be dimensionless.

    Parameters
    ----------
    x : array_like or Quantity
        First input.
    y : array_like or Quantity
        Second input.

    Returns
    -------
    out : jax.Array
        Element-wise bit-wise XOR.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.bitwise_xor(jnp.array([True, False]),
        ...                     jnp.array([True, True]))
        Array([False,  True], dtype=bool)
    """
    return _fun_unitless_binary('bitwise_xor', x, y, **kwargs)


@set_module_as('saiunit.math')
def left_shift(
    x: Union[Quantity, ArrayLike],
    y: Union[Quantity, ArrayLike],
    **kwargs,
) -> jax.Array:
    """
    Shift the bits of an integer to the left, element-wise.

    Both inputs must be dimensionless.

    Parameters
    ----------
    x : array_like or Quantity
        Input values.
    y : array_like or Quantity
        Number of bits to shift.

    Returns
    -------
    out : jax.Array
        Element-wise left shift.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.left_shift(jnp.array([1, 2]), jnp.array([1, 2]))
        Array([2, 8], dtype=int32)
    """
    return _fun_unitless_binary('left_shift', x, y, **kwargs)


@set_module_as('saiunit.math')
def right_shift(
    x: Union[Quantity, ArrayLike],
    y: Union[Quantity, ArrayLike],
    **kwargs,
) -> jax.Array:
    """
    Shift the bits of an integer to the right, element-wise.

    Both inputs must be dimensionless.

    Parameters
    ----------
    x : array_like or Quantity
        Input values.
    y : array_like or Quantity
        Number of bits to shift.

    Returns
    -------
    out : jax.Array
        Element-wise right shift.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.right_shift(jnp.array([8, 16]), jnp.array([1, 2]))
        Array([4, 4], dtype=int32)
    """
    return _fun_unitless_binary('right_shift', x, y, **kwargs)
