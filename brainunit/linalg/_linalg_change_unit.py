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

from collections.abc import Sequence
from typing import Union, Optional, Tuple, Any, Callable

import jax
import jax.numpy as jnp

from ..math._fun_array_creation import asarray
from ..math._fun_change_unit import unit_change, _fun_change_unit_binary
from .._base import UNITLESS, Quantity, maybe_decimal
from .._misc import set_module_as

__all__ = [
    'dot', 'multi_dot', 'vdot', 'vecdot', 'inner', 'outer', 'kron', 'matmul', 'tensordot',
    'matrix_power', 'det',
]

@unit_change(lambda x, y: x * y)
def dot(
    a: Union[jax.typing.ArrayLike, Quantity],
    b: Union[jax.typing.ArrayLike, Quantity],
    *,
    precision: Any = None,
    preferred_element_type: Optional[jax.typing.DTypeLike] = None
) -> Union[jax.Array, Quantity]:
    """
    Dot product of two arrays or quantities.

    Parameters
    ----------
    a : array_like, Quantity
      First argument.
    b : array_like, Quantity
      Second argument.
    precision : either ``None`` (default),
      which means the default precision for the backend, a :class:`~jax.lax.Precision`
      enum value (``Precision.DEFAULT``, ``Precision.HIGH`` or ``Precision.HIGHEST``)
      or a tuple of two such values indicating precision of ``a`` and ``b``.
    preferred_element_type : either ``None`` (default)
      which means the default accumulation type for the input types, or a datatype,
      indicating to accumulate results to and return a result with that datatype.

    Returns
    -------
    output : ndarray, Quantity
      array containing the dot product of the inputs, with batch dimensions of
      ``a`` and ``b`` stacked rather than broadcast.

      This is a Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
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
    operation order.

    Args:
      arrays: sequence of arrays / quantities. All must be two-dimensional, except the first
        and last which may be one-dimensional.
      precision: either ``None`` (default), which means the default precision for
        the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
        ``Precision.HIGH`` or ``Precision.HIGHEST``).

    Returns:
      an array representing the equivalent of ``reduce(jnp.matmul, arrays)``, but
      evaluated in the optimal order.

    This function exists because the cost of computing sequences of matmul operations
    can differ vastly depending on the order in which the operations are evaluated.
    For a single matmul, the number of floating point operations (flops) required to
    compute a matrix product can be approximated this way:

    >>> def approx_flops(x, y):
    ...   # for 2D x and y, with x.shape[1] == y.shape[0]
    ...   return 2 * x.shape[0] * x.shape[1] * y.shape[1]

    Suppose we have three matrices that we'd like to multiply in sequence:

    >>> import brainunit as bu
    >>> key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
    >>> x = jax.random.normal(key1, shape=(200, 5)) * bu.mA
    >>> y = jax.random.normal(key2, shape=(5, 100)) * bu.mV
    >>> z = jax.random.normal(key3, shape=(100, 10)) * bu.ohm

    Because of associativity of matrix products, there are two orders in which we might
    evaluate the product ``x @ y @ z``, and both produce equivalent outputs up to floating
    point precision:

    >>> result1 = (x @ y) @ z
    >>> result2 = x @ (y @ z)
    >>> bu.math.allclose(result1, result2, atol=1E-4)
    Array(True, dtype=bool)

    But the computational cost of these differ greatly:

    >>> print("(x @ y) @ z flops:", approx_flops(x, y) + approx_flops(x @ y, z))
    (x @ y) @ z flops: 600000
    >>> print("x @ (y @ z) flops:", approx_flops(y, z) + approx_flops(x, y @ z))
    x @ (y @ z) flops: 30000

    The second approach is about 20x more efficient in terms of estimated flops!

    ``multi_dot`` is a function that will automatically choose the fastest
    computational path for such problems:

    >>> result3 = bu.math.multi_dot([x, y, z])
    >>> bu.math.allclose(result1, result3, atol=1E-4)
    Array(True, dtype=bool)

    We can use JAX's :ref:`ahead-of-time-lowering` tools to estimate the total flops
    of each approach, and confirm that ``multi_dot`` is choosing the more efficient
    option:

    >>> jax.jit(lambda x, y, z: (x @ y) @ z).lower(x, y, z).cost_analysis()['flops']
    600000.0
    >>> jax.jit(lambda x, y, z: x @ (y @ z)).lower(x, y, z).cost_analysis()['flops']
    30000.0
    >>> jax.jit(bu.math.multi_dot).lower([x, y, z]).cost_analysis()['flops']
    30000.0
    """
    new_arrays = []
    unit = UNITLESS
    for arr in arrays:
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

    Parameters
    ----------
    a : array_like, Quantity
      First argument.
    b : array_like, Quantity
      Second argument.
    precision : either ``None`` (default),
      which means the default precision for the backend, a :class:`~jax.lax.Precision`
      enum value (``Precision.DEFAULT``, ``Precision.HIGH`` or ``Precision.HIGHEST``)
      or a tuple of two such values indicating precision of ``a`` and ``b``.
    preferred_element_type : either ``None`` (default)
      which means the default accumulation type for the input types, or a datatype,
      indicating to accumulate results to and return a result with that datatype.

    Returns
    -------
    output : ndarray, Quantity
      array containing the dot product of the inputs, with batch dimensions of
      ``a`` and ``b`` stacked rather than broadcast.

      This is a Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
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
    """Perform a conjugate multiplication of two batched vectors.

    Args:
      a: left-hand side array / Quantity.
      b: right-hand side array / Quantity. Size of ``b[axis]`` must match size of ``a[axis]``,
        and remaining dimensions must be broadcast-compatible.
      axis: axis along which to compute the dot product (default: -1)
      precision: either ``None`` (default), which means the default precision for
        the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
        ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
        such values indicating precision of ``a`` and ``b``.
      preferred_element_type: either ``None`` (default), which means the default
        accumulation type for the input types, or a datatype, indicating to
        accumulate results to and return a result with that datatype.

    Returns:
      array containing the conjugate dot product of ``a`` and ``b`` along ``axis``.
      The non-contracted dimensions are broadcast together.

    Examples:
      Vector conjugate-dot product of two 1D arrays:

      >>> import brainunit as bu
      >>> a = bu.math.array([1j, 2j, 3j]) * bu.ohm
      >>> b = bu.math.array([4., 5., 6.]) * bu.mA
      >>> bu.math.vecdot(a, b)
      Array(0.-32.j, dtype=complex64) * mvolt

      Batched vector dot product of two 2D arrays:

      >>> a = bu.math.array([[1, 2, 3],
      ...                   [4, 5, 6]]) * bu.ohm
      >>> b = bu.math.array([[2, 3, 4]]) * bu.mA
      >>> bu.math.vecdot(a, b, axis=-1)
      Array([20, 47], dtype=float32) * mV
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
    Inner product of two arrays or quantities.

    Parameters
    ----------
    a : array_like, Quantity
      First argument.
    b : array_like, Quantity
      Second argument.
    precision : either ``None`` (default),
      which means the default precision for the backend, a :class:`~jax.lax.Precision`
      enum value (``Precision.DEFAULT``, ``Precision.HIGH`` or ``Precision.HIGHEST``)
      or a tuple of two such values indicating precision of ``a`` and ``b``.
    preferred_element_type : either ``None`` (default)
      which means the default accumulation type for the input types, or a datatype,
      indicating to accumulate results to and return a result with that datatype.

    Returns
    -------
    output : ndarray, Quantity
      array containing the inner product of the inputs, with batch dimensions of
      ``a`` and ``b`` stacked rather than broadcast.

      This is a Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
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

    Parameters
    ----------
    a : array_like, Quantity
      First argument.
    b : array_like, Quantity
      Second argument.
    out : ndarray, optional
      A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
      If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    output : ndarray, Quantity
      array containing the outer product of the inputs, with batch dimensions of
      ``a`` and ``b`` stacked rather than broadcast.

      This is a Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
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

    Parameters
    ----------
    a : array_like, Quantity
      First input.
    b : array_like, Quantity
      Second input.

    Returns
    -------
    output : ndarray, Quantity
      Kronecker product of `a` and `b`.

      This is a Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
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
    Matrix product of two arrays or quantities.

    Parameters
    ----------
    a : array_like, Quantity
      First argument.
    b : array_like, Quantity
      Second argument.
    precision : either ``None`` (default),
      which means the default precision for the backend, a :class:`~jax.lax.Precision`
      enum value (``Precision.DEFAULT``, ``Precision.HIGH`` or ``Precision.HIGHEST``)
      or a tuple of two such values indicating precision of ``a`` and ``b``.
    preferred_element_type : either ``None`` (default)
      which means the default accumulation type for the input types, or a datatype,
      indicating to accumulate results to and return a result with that datatype.

    Returns
    -------
    output : ndarray, Quantity
      array containing the matrix product of the inputs, with batch dimensions of
      ``a`` and ``b`` stacked rather than broadcast.

      This is a Quantity if the final unit is the product of the unit of `a` and the unit of `b`, else an array.
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

    Given two tensors, `a` and `b`, and an array_like object containing
    two array_like objects, ``(a_axes, b_axes)``, sum the products of
    `a`'s and `b`'s elements (components) over the axes specified by
    ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
    of `a` and the first ``N`` dimensions of `b` are summed over.

    Parameters
    ----------
    a, b : array_like, Quantity
      Tensors to "dot".

    axes : int or (2,) array_like
      * integer_like
        If an int N, sum over the last N axes of `a` and the first N axes
        of `b` in order. The sizes of the corresponding axes must match.
      * (2,) array_like
        Or, a list of axes to be summed over, first sequence applying to `a`,
        second to `b`. Both elements array_like must be of the same length.
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

    Returns
    -------
    output : ndarray, Quantity
      The tensor dot product of the input.

      This is a quantity if the product of the units of `a` and `b` is not dimensionless.
    """
    return _fun_change_unit_binary(jnp.tensordot,
                                   lambda x, y: x * y,
                                   a, b,
                                   axes=axes,
                                   precision=precision,
                                   preferred_element_type=preferred_element_type)


@set_module_as('brainunit.math')
def matrix_power(
    a: Union[jax.typing.ArrayLike, Quantity],
    n: int
) -> Union[jax.typing.ArrayLike, Quantity]:
    """
    Raise a square matrix to the (integer) power `n`.

    Parameters
    ----------
    a : array_like, Quantity
      Matrix to be "powered".
    n : int
      The exponent can be any integer.

    Returns
    -------
    out : ndarray, Quantity
      The result of raising `a` to the power `n`.

      This is a Quantity if the final unit is the product of the unit of `a` and itself, else an array.
    """
    if isinstance(a, Quantity):
        return maybe_decimal(Quantity(jnp.linalg.matrix_power(a.mantissa, n), unit=a.unit ** n))
    else:
        return jnp.linalg.matrix_power(a, n)

@set_module_as('brainunit.math')
def det(
    a: Union[jax.typing.ArrayLike, Quantity],
) -> Union[jax.typing.ArrayLike, Quantity]:
    """
    Compute the determinant of an array.

    JAX implementation of :func:`numpy.linalg.det`.

    Args:
        a: array of shape ``(..., M, M)`` for which to compute the determinant.

    Returns:
        An array of determinants of shape ``a.shape[:-2]``.

    Examples:
    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.linalg.det(a)
    Array(-2., dtype=float32)
    """
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
