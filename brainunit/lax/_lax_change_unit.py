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

from typing import Callable, Union, Sequence

import jax
from jax import lax

from .._base import Quantity, maybe_decimal, UNITLESS
from .._misc import set_module_as
from ..math._fun_change_unit import _fun_change_unit_unary, _fun_change_unit_binary

__all__ = [
    # math funcs change unit (unary)
    'rsqrt',

    # math funcs change unit (binary)
    'div', 'pow', 'integer_pow', 'mul', 'rem', 'batch_matmul',

    # math funcs conv
    'conv', 'conv_transpose',

    # math funcs misc
    'dot_general',
]


def unit_change(
    unit_change_fun: Callable
):
    def actual_decorator(func):
        func._unit_change_fun = unit_change_fun
        return set_module_as('brainunit.lax')(func)

    return actual_decorator


# math funcs change unit (unary)
@unit_change(lambda u: u ** -0.5)
def rsqrt(
    x: Union[jax.typing.ArrayLike, Quantity],
) -> Union[Quantity, jax.Array]:
    r"""Elementwise reciprocal square root:  :math:`1 \over \sqrt{x}`."""
    return _fun_change_unit_unary(lax.rsqrt,
                                  lambda u: u ** -0.5,
                                  x)


# math funcs change unit (binary)
@unit_change(lambda x, y: x * y)
def conv(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
    window_strides: Sequence[int],
    padding: str,
    precision: lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None
) -> Union[Quantity, jax.Array]:
    """Convenience wrapper around `conv_general_dilated`.

    Args:
    lhs: a rank `n+2` dimensional input array.
    rhs: a rank `n+2` dimensional array of kernel weights.
    window_strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`.
    precision: Optional. Either ``None``, which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      :class:`~jax.lax.Precision` enums indicating precision of ``lhs``` and ``rhs``.
    preferred_element_type: Optional. Either ``None``, which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.

    Returns:
    An array containing the convolution result.
    """
    return _fun_change_unit_binary(lax.conv,
                                   lambda x, y: x * y,
                                   x, y,
                                   window_strides, padding, precision, preferred_element_type)


@unit_change(lambda x, y: x * y)
def conv_transpose(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
    strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
    transpose_kernel: bool = False,
    precision: lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None
) -> Union[Quantity, jax.Array]:
    """Convenience wrapper for calculating the N-d convolution "transpose".

    This function directly calculates a fractionally strided conv rather than
    indirectly calculating the gradient (transpose) of a forward convolution.

    Args:
      lhs: a rank `n+2` dimensional input array.
      rhs: a rank `n+2` dimensional array of kernel weights.
      strides: sequence of `n` integers, sets fractional stride.
      padding: 'SAME', 'VALID' will set as transpose of corresponding forward
        conv, or a sequence of `n` integer 2-tuples describing before-and-after
        padding for each `n` spatial dimension.
      rhs_dilation: `None`, or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `rhs`. RHS dilation
        is also known as atrous convolution.
      dimension_numbers: tuple of dimension descriptors as in
        lax.conv_general_dilated. Defaults to tensorflow convention.
      transpose_kernel: if True flips spatial axes and swaps the input/output
        channel axes of the kernel. This makes the output of this function identical
        to the gradient-derived functions like keras.layers.Conv2DTranspose
        applied to the same kernel. For typical use in neural nets this is completely
        pointless and just makes input/output channel specification confusing.
      precision: Optional. Either ``None``, which means the default precision for
        the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
        ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
        :class:`~jax.lax.Precision` enums indicating precision of ``lhs``` and ``rhs``.
      preferred_element_type: Optional. Either ``None``, which means the default
        accumulation type for the input types, or a datatype, indicating to
        accumulate results to and return a result with that datatype.

    Returns:
      Transposed N-d convolution, with output padding following the conventions of
      keras.layers.Conv2DTranspose.
    """
    return _fun_change_unit_binary(lax.conv_transpose,
                                   lambda x, y: x * y,
                                   x, y,
                                   strides, padding, rhs_dilation, dimension_numbers, transpose_kernel, precision,
                                   preferred_element_type)


@unit_change(lambda x, y: x / y)
def div(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
) -> Union[Quantity, jax.Array]:
    r"""Elementwise division: :math:`x \over y`.

    Integer division overflow
    (division by zero or signed division of INT_SMIN with -1)
    produces an implementation defined value.
    """
    return _fun_change_unit_binary(lax.div,
                                   lambda x, y: x / y,
                                   x, y)


@unit_change(lambda x, y: x * y)
def dot_general(
    x: Union[jax.typing.ArrayLike, Quantity],
    y: Union[jax.typing.ArrayLike, Quantity],
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    out_type=None
) -> Union[Quantity, jax.Array]:
    """General dot product/contraction operator.

    Wraps XLA's `DotGeneral
    <https://www.tensorflow.org/xla/operation_semantics#dotgeneral>`_
    operator.

    The semantics of ``dot_general`` are complicated, but most users should not have to
    use it directly. Instead, you can use higher-level functions like :func:`jax.numpy.dot`,
    :func:`jax.numpy.matmul`, :func:`jax.numpy.tensordot`, :func:`jax.numpy.einsum`,
    and others which will construct appropriate calls to ``dot_general`` under the hood.
    If you really want to understand ``dot_general`` itself, we recommend reading XLA's
    `DotGeneral  <https://www.tensorflow.org/xla/operation_semantics#dotgeneral>`_
    operator documentation.

    Args:
        lhs: an array
        rhs: an array
        dimension_numbers: a tuple of tuples of sequences of ints of the form
          ``((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims,
          rhs_batch_dims))``
        precision: Optional. This parameter controls the numerics of the
          computation, and it can be one of the following:

          - ``None``, which means the default precision for the current backend,
          - a :class:`~jax.lax.Precision` enum value or a tuple of two
            :class:`~jax.lax.Precision` enums indicating precision of ``lhs``` and
            ``rhs``, or
          - a :class:`~jax.lax.DotAlgorithm` or a
            :class:`~jax.lax.DotAlgorithmPreset` indicating the algorithm that
            must be used to accumulate the dot product.

        preferred_element_type: Optional. This parameter controls the data type
          output by the dot product. By default, the output element type of this
          operation will match the ``lhs`` and ``rhs`` input element types under
          the usual type promotion rules. Setting ``preferred_element_type`` to a
          specific ``dtype`` will mean that the operation returns that element type.
          When ``precision`` is not a :class:`~jax.lax.DotAlgorithm` or
          :class:`~jax.lax.DotAlgorithmPreset`, ``preferred_element_type`` provides
          a hint to the compiler to accumulate the dot product using this data type.

    Returns:
        An array whose first dimensions are the (shared) batch dimensions, followed
        by the ``lhs`` non-contracting/non-batch dimensions, and finally the ``rhs``
        non-contracting/non-batch dimensions.
    """
    try:
        return _fun_change_unit_binary(lax.dot_general,
                                       lambda x, y: x * y,
                                       x, y,
                                       dimension_numbers, precision, preferred_element_type, out_type)
    except:
        return _fun_change_unit_binary(lax.dot_general,
                                       lambda x, y: x * y,
                                       x, y,
                                       dimension_numbers, precision, preferred_element_type)


@set_module_as('brainunit.lax')
def pow(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    r"""Elementwise power: :math:`x^y`."""
    if isinstance(x, Quantity):
        if isinstance(y, Quantity):
            assert y.is_unitless, f'{jax.lax.pow.__name__} only supports scalar exponent'
            y = y.mantissa
        return maybe_decimal(Quantity(jax.lax.pow(x.mantissa, y), unit=x.unit ** y))
    elif isinstance(y, Quantity):
        assert y.is_unitless, f'{jax.lax.power.__name__} only supports scalar exponent'
        y = y.mantissa
        return maybe_decimal(Quantity(jax.lax.pow(x, y), unit=x ** y))
    else:
        return jax.lax.pow(x, y)


@set_module_as('brainunit.lax')
def integer_pow(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    r"""Elementwise power: :math:`x^y`, where :math:`y` is a fixed integer."""
    if isinstance(x, Quantity):
        if isinstance(y, Quantity):
            assert y.is_unitless, f'{jax.lax.integer_pow.__name__} only supports scalar exponent'
            y = y.mantissa
        return maybe_decimal(Quantity(jax.lax.integer_pow(x.mantissa, y), unit=x.unit ** y))
    elif isinstance(y, Quantity):
        assert y.is_unitless, f'{jax.lax.integer_power.__name__} only supports scalar exponent'
        y = y.mantissa
        return maybe_decimal(Quantity(jax.lax.integer_pow(x, y), unit=x ** y))
    else:
        return jax.lax.integer_pow(x, y)


@unit_change(lambda x, y: x * y)
def mul(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.typing.ArrayLike]:
    r"""Elementwise multiplication: :math:`x \times y`."""
    return _fun_change_unit_binary(lax.mul,
                                   lambda x, y: x * y,
                                   x, y)


@set_module_as('brainunit.lax')
def rem(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.typing.ArrayLike]:
    r"""Elementwise remainder: :math:`x \bmod y`.

    The sign of the result is taken from the dividend,
    and the absolute value of the result is always
    less than the divisor's absolute value.

    Integer division overflow
    (remainder by zero or remainder of INT_SMIN with -1)
    produces an implementation defined value.
    """
    if isinstance(x, Quantity) and isinstance(y, Quantity):
        return maybe_decimal(Quantity(lax.rem(x.mantissa, y.mantissa), unit=x.unit))
    elif isinstance(x, Quantity):
        return maybe_decimal(Quantity(lax.rem(x.mantissa, y), unit=x.unit))
    elif isinstance(y, Quantity):
        return maybe_decimal(Quantity(lax.rem(x, y.mantissa), unit=UNITLESS))
    else:
        return lax.rem(x, y)


@unit_change(lambda x, y: x * y)
def batch_matmul(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
    precision: jax.lax.PrecisionLike = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Batch matrix multiplication."""
    return _fun_change_unit_binary(lax.batch_matmul,
                                   lambda x, y: x * y,
                                   x, y, precision)
