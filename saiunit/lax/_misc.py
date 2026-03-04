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

from typing import Any, Callable, Sequence, Union

import jax
from jax import lax

from saiunit._base_getters import maybe_decimal
from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as, maybe_custom_array

__all__ = [
    'reduce', 'reduce_precision',

    # getting attribute funcs
    'broadcast_shapes',
]


# @set_module_as('saiunit.lax')
# def after_all(*operands):
#     """Merges one or more XLA token values. Experimental.
#
#     Wraps the XLA AfterAll operator."""
#     # new_operands = []
#     # for operand in operands:
#     #     if isinstance(operand, Quantity):
#     #         new_operands.append(operand.mantissa)
#     #     else:
#     #         new_operands.append(operand)
#     return lax.after_all(*operands)


@set_module_as('saiunit.lax')
def reduce(
    operands: Any,
    init_values: Any,
    computation: Callable[[Any, Any], Any],
    dimensions: Sequence[int]
) -> Any:
    """Reduce an array along dimensions using a computation.

    Wraps XLA's `Reduce
    <https://www.tensorflow.org/xla/operation_semantics#reduce>`_
    operator.

    ``init_values`` and ``computation`` together must form a `monoid
    <https://en.wikipedia.org/wiki/Monoid>`_
    for correctness: ``init_values`` must be an identity of
    ``computation``, and ``computation`` must be associative.

    Parameters
    ----------
    operands : array_like or Quantity
        The array(s) to reduce.  If a :class:`~saiunit.Quantity`, its
        underlying mantissa is extracted before the XLA operation.  If a
        :class:`~saiunit.CustomArray`, its ``.data`` attribute is unwrapped.
    init_values : array_like or Quantity
        The initial value(s) for the reduction.  Must be an identity element
        of ``computation``.  Accepts the same types as ``operands``.
    computation : callable
        A binary function used to combine elements (e.g. ``jax.lax.add``).
    dimensions : sequence of int
        The dimensions along which to reduce.

    Returns
    -------
    result : jax.Array
        The reduced result.  Note that unit information is not preserved
        through the raw XLA reduce; see Notes.

    Raises
    ------
    TypeError
        If ``operands`` and ``init_values`` have incompatible types after
        unwrapping.

    See Also
    --------
    jax.lax.reduce : The underlying JAX primitive.
    jax.numpy.sum : A higher-level sum that preserves units in saiunit.

    Notes
    -----
    Because this function delegates directly to :func:`jax.lax.reduce`, the
    unit metadata carried by a :class:`~saiunit.Quantity` is stripped before
    the reduction.  If you need the result to retain its unit, consider using
    the higher-level wrappers in :mod:`saiunit.math` (e.g. ``saiunit.math.sum``).

    Examples
    --------
    Reducing a plain array with ``lax.add``:

    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> import jax.numpy as jnp
        >>> from jax import lax
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> sulax.reduce(x, jnp.float32(0), lax.add, [0])
        Array(6., dtype=float32)

    Reducing a ``Quantity`` (unit is stripped, raw mantissa is reduced):

    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> import jax.numpy as jnp
        >>> from jax import lax
        >>> q = jnp.array([1.0, 2.0, 3.0]) * su.meter
        >>> sulax.reduce(q, jnp.float32(0) * su.meter, lax.add, [0])
        Array(6., dtype=float32)
    """
    operands = maybe_custom_array(operands)
    init_values = maybe_custom_array(init_values)
    return lax.reduce(operands, init_values, computation, dimensions)


def reduce_precision(
    operand: Union[jax.typing.ArrayLike, Quantity, float],
    exponent_bits: int,
    mantissa_bits: int
) -> jax.typing.ArrayLike:
    """Reduce the precision of array elements.

    Wraps XLA's `ReducePrecision
    <https://www.tensorflow.org/xla/operation_semantics#reduceprecision>`_
    operator.

    When the input is a :class:`~saiunit.Quantity`, the precision reduction
    is applied to the mantissa and the result is returned as a plain
    :class:`jax.Array` (the unit is stripped).

    Parameters
    ----------
    operand : array_like, Quantity, or float
        The input values whose precision will be reduced.  If a
        :class:`~saiunit.Quantity`, the precision reduction is applied to its
        mantissa and the result is a plain array.  If a
        :class:`~saiunit.CustomArray`, its ``.data`` attribute is unwrapped
        first.
    exponent_bits : int
        Number of exponent bits in the reduced-precision format.
    mantissa_bits : int
        Number of mantissa bits in the reduced-precision format.

    Returns
    -------
    result : jax.Array
        Array with reduced-precision values.  Unit information from a
        :class:`~saiunit.Quantity` input is not preserved.

    See Also
    --------
    jax.lax.reduce_precision : The underlying JAX primitive.

    Notes
    -----
    This function simulates the effect of converting values to a
    lower-precision floating-point format and back.  It is useful for
    exploring the numerical effects of quantization without actually changing
    the storage dtype.

    The ``exponent_bits`` and ``mantissa_bits`` together define a virtual
    floating-point format.  For example, ``exponent_bits=5`` and
    ``mantissa_bits=10`` correspond to IEEE float16.

    Examples
    --------
    Reducing precision of a plain array:

    .. code-block:: python

        >>> import saiunit.lax as sulax
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1.123456, 2.123456], dtype=jnp.float32)
        >>> sulax.reduce_precision(x, exponent_bits=5, mantissa_bits=10)
        Array([1.123047, 2.123047], dtype=float32)

    Reducing precision of a ``Quantity`` (mantissa is extracted, unit is stripped):

    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> import jax.numpy as jnp
        >>> q = jnp.array([1.123456, 2.123456], dtype=jnp.float32) * su.meter
        >>> sulax.reduce_precision(q, exponent_bits=5, mantissa_bits=10)
        Array([1.123047, 2.123047], dtype=float32)
    """
    operand = maybe_custom_array(operand)
    if isinstance(operand, Quantity):
        return maybe_decimal(lax.reduce_precision(operand.mantissa, exponent_bits, mantissa_bits))
    return lax.reduce_precision(operand, exponent_bits, mantissa_bits)


@set_module_as('saiunit.lax')
def broadcast_shapes(
    *shapes
):
    """Return the shape that results from NumPy broadcasting of ``shapes``.

    Computes the shape that would result from broadcasting arrays with the
    given shapes, following standard NumPy broadcasting rules.  This is a
    thin wrapper around :func:`jax.lax.broadcast_shapes` and does not involve
    any unit handling.

    Parameters
    ----------
    *shapes : tuple of int
        Two or more shapes to broadcast together.  Each shape is a tuple of
        non-negative integers.

    Returns
    -------
    result : tuple of int
        The broadcasted shape.

    Raises
    ------
    ValueError
        If the shapes are not broadcast-compatible (e.g. ``(2,)`` and
        ``(3,)``).

    See Also
    --------
    jax.lax.broadcast_shapes : The underlying JAX function.
    numpy.broadcast_shapes : The NumPy equivalent.

    Notes
    -----
    Broadcasting rules:

    1. If the shapes differ in length, the shorter shape is padded with ones
       on the left.
    2. Dimensions are compatible when they are equal, or one of them is 1.
    3. The resulting dimension is the maximum of the two.

    Examples
    --------
    Basic broadcasting of two shapes:

    .. code-block:: python

        >>> import saiunit.lax as sulax
        >>> sulax.broadcast_shapes((2, 3), (3,))
        (2, 3)

    Broadcasting with dimension expansion:

    .. code-block:: python

        >>> import saiunit.lax as sulax
        >>> sulax.broadcast_shapes((1, 5), (3, 1))
        (3, 5)

    Broadcasting three shapes together:

    .. code-block:: python

        >>> import saiunit.lax as sulax
        >>> sulax.broadcast_shapes((1,), (3, 1), (1, 1, 5))
        (1, 3, 5)
    """
    return lax.broadcast_shapes(*shapes)
