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

from typing import Union, Optional, Callable, Sequence

import jax
from jax import lax

from .._base import Quantity, Unit
from .._misc import set_module_as
from ..math._fun_accept_unitless import _fun_accept_unitless_unary, _fun_accept_unitless_binary, _fun_unitless_binary

__all__ = [
    # math funcs only accept unitless (unary)
    'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh',
    'collapse', 'cumlogsumexp',
    'bessel_i0e', 'bessel_i1e', 'digamma', 'lgamma', 'erf', 'erfc',
    'erf_inv', 'logistic',

    # math funcs only accept unitless (binary)
    'atan2', 'polygamma', 'igamma', 'igammac', 'igamma_grad_a', 'random_gamma_grad',
    'zeta',

    # math funcs only accept unitless (n-ary)
    'betainc',

    # Elementwise bit operations (unary)

    # Elementwise bit operations (binary)
    'shift_left', 'shift_right_arithmetic', 'shift_right_logical',

    # fft
    'fft',

    # misc
    'collapse',
]


# math funcs only accept unitless (unary)
# ---------------------------------------

@set_module_as('saiunit.lax')
def acos(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise arc cosine: :math:`\mathrm{acos}(x)`."""
    return _fun_accept_unitless_unary(lax.acos, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def acosh(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise inverse hyperbolic cosine: :math:`\mathrm{acosh}(x)`."""
    return _fun_accept_unitless_unary(lax.acosh, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def asin(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise arc sine: :math:`\mathrm{asin}(x)`."""
    return _fun_accept_unitless_unary(lax.asin, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def asinh(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise inverse hyperbolic sine: :math:`\mathrm{asinh}(x)`."""
    return _fun_accept_unitless_unary(lax.asinh, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def atan(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise arc tangent: :math:`\mathrm{atan}(x)`."""
    return _fun_accept_unitless_unary(lax.atan, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def atanh(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise inverse hyperbolic tangent: :math:`\mathrm{atanh}(x)`."""
    return _fun_accept_unitless_unary(lax.atanh, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def collapse(
    x: Union[Quantity, jax.typing.ArrayLike],
    start_dimension: int,
    stop_dimension: Optional[int] = None,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """Collapses dimensions of an array into a single dimension.

    For example, if ``operand`` is an array with shape ``[2, 3, 4]``,
    ``collapse(operand, 0, 2).shape == [6, 4]``. The elements of the collapsed
    dimension are laid out major-to-minor, i.e., with the lowest-numbered
    dimension as the slowest varying dimension.

    Args:
        x: an input array.
        start_dimension: the start of the dimensions to collapse (inclusive).
        stop_dimension: the end of the dimensions to collapse (exclusive). Pass None
          to collapse all the dimensions after start.
        unit_to_scale: the unit to scale the input to. If None, the input should be
            dimensionless.

    Returns:
        An array where dimensions ``[start_dimension, stop_dimension)`` have been
        collapsed (raveled) into a single dimension.
    """
    return _fun_accept_unitless_unary(lax.collapse, x, start_dimension=start_dimension, stop_dimension=stop_dimension,
                                      unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def cumlogsumexp(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = 0,
    reverse: Optional[bool] = False,
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    """Computes a cumulative logsumexp along `axis`."""
    return _fun_accept_unitless_unary(lax.cumlogsumexp, x, axis, reverse,
                                      unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def bessel_i0e(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Exponentially scaled modified Bessel function of order 0:
    :math:`\mathrm{i0e}(x) = e^{-|x|} \mathrm{i0}(x)`
    """
    return _fun_accept_unitless_unary(lax.bessel_i0e, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def bessel_i1e(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Exponentially scaled modified Bessel function of order 1:
    :math:`\mathrm{i1e}(x) = e^{-|x|} \mathrm{i1}(x)`
    """
    return _fun_accept_unitless_unary(lax.bessel_i1e, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def digamma(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise digamma: :math:`\psi(x)`."""
    return _fun_accept_unitless_unary(lax.digamma, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def lgamma(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise log gamma: :math:`\mathrm{log}(\Gamma(x))`."""
    return _fun_accept_unitless_unary(lax.lgamma, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def erf(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise error function: :math:`\mathrm{erf}(x)`."""
    return _fun_accept_unitless_unary(lax.erf, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def erfc(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise complementary error function:
    :math:`\mathrm{erfc}(x) = 1 - \mathrm{erf}(x)`."""
    return _fun_accept_unitless_unary(lax.erfc, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def erf_inv(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise inverse error function: :math:`\mathrm{erf}^{-1}(x)`."""
    return _fun_accept_unitless_unary(lax.erf_inv, x, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def logistic(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise logistic (sigmoid) function: :math:`\frac{1}{1 + e^{-x}}`."""
    return _fun_accept_unitless_unary(lax.logistic, x, unit_to_scale=unit_to_scale)


# math funcs only accept unitless (binary)
# ----------------------------------------
@set_module_as('saiunit.lax')
def atan2(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise arc tangent of two variables:
        :math:`\mathrm{atan}({x \over y})`."""
    return _fun_accept_unitless_binary(lax.atan2, x, y, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def polygamma(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise polygamma: :math:`\psi^{(m)}(x)`."""
    return _fun_accept_unitless_binary(lax.polygamma, x, y, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def igamma(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise regularized incomplete gamma function."""
    return _fun_accept_unitless_binary(lax.igamma, x, y, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def igammac(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise complementary regularized incomplete gamma function."""
    return _fun_accept_unitless_binary(lax.igammac, x, y, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def igamma_grad_a(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise derivative of the regularized incomplete gamma function."""
    return _fun_accept_unitless_binary(lax.igamma_grad_a, x, y, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def random_gamma_grad(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise derivative of samples from `Gamma(a, 1)`."""
    return _fun_accept_unitless_binary(lax.random_gamma_grad, x, y, unit_to_scale=unit_to_scale)


@set_module_as('saiunit.lax')
def zeta(
    x: Union[jax.typing.ArrayLike, Quantity],
    q: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise Hurwitz zeta function: :math:`\zeta(x, q)`"""
    return _fun_accept_unitless_binary(lax.zeta, x, q, unit_to_scale=unit_to_scale)


# math funcs only accept unitless (n-ary)
# ---------------------------------------

def _fun_accept_unitless_nary(
    func: Callable,
    *args,
    quantity_num: int,
    unit_to_scale: Optional[Unit] = None,
    **kwargs,
):
    if not isinstance(quantity_num, int):
        raise TypeError(f'quantity_num should be an integer. Got {quantity_num}')
    new_args = []
    for arg in args:
        if isinstance(arg, Quantity):
            if unit_to_scale is None:
                assert arg.dim.is_dimensionless, (
                    f'{func} only support dimensionless input. But we got {arg}. \n'
                    f'If you want to scale the input, please provide the "unit_to_scale" parameter. Or '
                    f'convert the input to a dimensionless Quantity manually.'
                )
                new_args.append(arg.to_decimal())
            else:
                assert isinstance(unit_to_scale, Unit), f'unit_to_scale should be a Unit instance. Got {unit_to_scale}'
                new_args.append(arg.to_decimal(unit_to_scale))
        else:
            new_args.append(arg)
    return func(*new_args, **kwargs)


@set_module_as('saiunit.lax')
def betainc(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
    x: Union[jax.typing.ArrayLike, Quantity],
    unit_to_scale: Optional[Unit] = None,
) -> jax.Array:
    r"""Elementwise regularized incomplete beta integral."""
    return _fun_accept_unitless_nary(lax.betainc, a, b, x,
                                     quantity_num=3, unit_to_scale=unit_to_scale)


# Elementwise bit operations (binary)
@set_module_as('saiunit.lax')
def shift_left(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> jax.Array:
    r"""Elementwise left shift: :math:`x \ll y`."""
    return _fun_unitless_binary(lax.shift_left, x, y)


@set_module_as('saiunit.lax')
def shift_right_arithmetic(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> jax.Array:
    r"""Elementwise arithmetic right shift: :math:`x \gg y`."""
    return _fun_unitless_binary(lax.shift_right_arithmetic, x, y)


@set_module_as('saiunit.lax')
def shift_right_logical(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> jax.Array:
    r"""Elementwise logical right shift: :math:`x \gg y`."""
    return _fun_unitless_binary(lax.shift_right_logical, x, y)


# fft
@set_module_as('saiunit.lax')
def fft(
    x: Union[Quantity, jax.typing.ArrayLike],
    fft_type: jax.lax.FftType | str,
    fft_lengths: Sequence[int],
    unit_to_scale: Optional[Unit] = None,
):
    return _fun_accept_unitless_unary(lax.fft, x, fft_type, fft_lengths,
                                      unit_to_scale=unit_to_scale)
