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
from typing import (Union, Optional, List, Any, Tuple)

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from saiunit._base_unit import UNITLESS, Unit
from saiunit._base_getters import fail_for_unit_mismatch, get_unit, unit_scale_align_to_first
from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as, maybe_custom_array_tree, maybe_custom_array

Shape = Union[int, Sequence[int]]

__all__ = [
    # array creation(given shape)
    'full', 'eye', 'identity', 'tri',
    'empty', 'ones', 'zeros',

    # array creation(given array)
    'full_like', 'diag', 'tril', 'triu',
    'empty_like', 'ones_like', 'zeros_like', 'fill_diagonal',

    # array creation(misc)
    'array', 'asarray', 'arange', 'linspace', 'logspace',
    'meshgrid', 'vander',

    # indexing funcs
    'tril_indices', 'tril_indices_from', 'triu_indices',
    'triu_indices_from',

    # others
    'from_numpy',
    'as_numpy',
    'tree_ones_like',
    'tree_zeros_like',
]


@set_module_as('saiunit.math')
def full(
    shape: Shape,
    fill_value: Union[Quantity, int, float],
    dtype: Optional[jax.typing.DTypeLike] = None,
) -> Union[Array, Quantity]:
    """
    Return a new quantity or array of given shape, filled with ``fill_value``.

    If ``fill_value`` is a :class:`~saiunit.Quantity`, the result is a
    ``Quantity`` carrying the same unit.  Otherwise a plain JAX array is
    returned.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar, array_like, or Quantity
        Fill value.  When a ``Quantity`` is given its unit is preserved.
    dtype : data-type, optional
        The desired data-type for the array.  The default, ``None``, means
        the dtype is inferred from ``fill_value``.

    Returns
    -------
    out : Quantity or jax.Array
        Array (or ``Quantity``) of ``fill_value`` with the given shape and
        dtype.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su.math.full((2, 3), 7.0)
        Array([[7., 7., 7.],
               [7., 7., 7.]], dtype=float32)
        >>> su.math.full((3,), 5.0 * su.meter)
        Quantity([5. 5. 5.], "m")
    """
    fill_value = maybe_custom_array(fill_value)
    if isinstance(fill_value, Quantity):
        return Quantity(jnp.full(shape, fill_value.mantissa, dtype=dtype), unit=fill_value.unit)
    return jnp.full(shape, fill_value, dtype=dtype)


@set_module_as('saiunit.math')
def eye(
    N: int,
    M: Optional[int] = None,
    k: int = 0,
    dtype: Optional[jax.typing.DTypeLike] = None,
    unit: Unit = UNITLESS,
) -> Union[Array, Quantity]:
    """
    Return a 2-D identity-like quantity or array with ones on the diagonal.

    Parameters
    ----------
    N : int
        Number of rows in the output.
    M : int, optional
        Number of columns in the output.  If ``None``, defaults to ``N``.
    k : int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value
        to a lower diagonal.
    dtype : data-type, optional
        Data-type of the returned array.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  When ``UNITLESS`` (the default)
        a plain array is returned.

    Returns
    -------
    out : Quantity or jax.Array
        An array of shape ``(N, M)`` where all elements are zero except for
        the ``k``-th diagonal, whose values are one (optionally carrying
        ``unit``).

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su.math.eye(2)
        Array([[1., 0.],
               [0., 1.]], dtype=float32)
        >>> su.math.eye(2, unit=su.meter)
        Quantity([[1. 0.]
                  [0. 1.]], "m")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'eye requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    if not unit.is_unitless:
        return jnp.eye(N, M, k, dtype=dtype) * unit
    else:
        return jnp.eye(N, M, k, dtype=dtype)


@set_module_as('saiunit.math')
def identity(
    n: int,
    dtype: Optional[jax.typing.DTypeLike] = None,
    unit: Unit = UNITLESS
) -> Union[Array, Quantity]:
    """
    Return the identity quantity or array.

    The identity array is a square array with ones on the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in the ``n x n`` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  When ``UNITLESS`` (the default)
        a plain array is returned.

    Returns
    -------
    out : Quantity or jax.Array
        ``n x n`` array with its main diagonal set to one and all other
        elements zero.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su.math.identity(3)
        Array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)
        >>> su.math.identity(2, unit=su.second)
        Quantity([[1. 0.]
                  [0. 1.]], "s")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'identity requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    if not unit.is_unitless:
        return jnp.identity(n, dtype=dtype) * unit
    else:
        return jnp.identity(n, dtype=dtype)


@set_module_as('saiunit.math')
def tri(
    N: int,
    M: Optional[int] = None,
    k: int = 0,
    dtype: Optional[jax.typing.DTypeLike] = None,
    unit: Unit = UNITLESS
) -> Union[Array, Quantity]:
    """
    Return an array with ones at and below the given diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array.  By default, ``M`` is taken equal
        to ``N``.
    k : int, optional
        The sub-diagonal at and below which the array is filled.
        ``k = 0`` is the main diagonal, ``k < 0`` is below it, and
        ``k > 0`` is above.  The default is 0.
    dtype : data-type, optional
        Data type of the returned array.  The default is ``float``.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or jax.Array
        Array of shape ``(N, M)`` with its lower triangle filled with ones
        and zero elsewhere; i.e. ``T[i, j] == 1`` for ``j <= i + k``,
        0 otherwise.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su.math.tri(3)
        Array([[1., 0., 0.],
               [1., 1., 0.],
               [1., 1., 1.]], dtype=float32)
        >>> su.math.tri(2, 3, unit=su.meter)
        Quantity([[1. 0. 0.]
                  [1. 1. 0.]], "m")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'tri requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    if not unit.is_unitless:
        return jnp.tri(N, M, k, dtype=dtype) * unit
    else:
        return jnp.tri(N, M, k, dtype=dtype)


@set_module_as('saiunit.math')
def empty(
    shape: Shape,
    dtype: Optional[jax.typing.DTypeLike] = None,
    unit: Unit = UNITLESS
) -> Union[Array, Quantity]:
    """
    Return a new quantity or array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the empty quantity or array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  When ``UNITLESS`` (the default)
        a plain array is returned.

    Returns
    -------
    out : Quantity or jax.Array
        Array of uninitialized (arbitrary) data of the given shape and dtype.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> result = su.math.empty((2, 3))
        >>> result.shape
        (2, 3)
        >>> result = su.math.empty((2,), unit=su.meter)
        >>> su.get_unit(result) == su.meter
        True
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'empty requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    if not unit.is_unitless:
        return jnp.empty(shape, dtype=dtype) * unit
    else:
        return jnp.empty(shape, dtype=dtype)


@set_module_as('saiunit.math')
def ones(
    shape: Shape,
    dtype: Optional[jax.typing.DTypeLike] = None,
    unit: Unit = UNITLESS
) -> Union[Array, Quantity]:
    """
    Return a new quantity or array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new quantity or array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array.  Default is ``float``.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  When ``UNITLESS`` (the default)
        a plain array is returned.

    Returns
    -------
    out : Quantity or jax.Array
        Array of ones with the given shape and dtype.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su.math.ones((3,))
        Array([1., 1., 1.], dtype=float32)
        >>> su.math.ones((2, 2), unit=su.meter)
        Quantity([[1. 1.]
                  [1. 1.]], "m")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'ones requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    if not unit.is_unitless:
        return jnp.ones(shape, dtype=dtype) * unit
    else:
        return jnp.ones(shape, dtype=dtype)


@set_module_as('saiunit.math')
def zeros(
    shape: Shape,
    dtype: Optional[jax.typing.DTypeLike] = None,
    unit: Unit = UNITLESS
) -> Union[Array, Quantity]:
    """
    Return a new quantity or array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new quantity or array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array.  Default is ``float``.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  When ``UNITLESS`` (the default)
        a plain array is returned.

    Returns
    -------
    out : Quantity or jax.Array
        Array of zeros with the given shape and dtype.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su.math.zeros((3,))
        Array([0., 0., 0.], dtype=float32)
        >>> su.math.zeros((2,), unit=su.second)
        Quantity([0. 0.], "s")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'zeros requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    if not unit.is_unitless:
        return jnp.zeros(shape, dtype=dtype) * unit
    else:
        return jnp.zeros(shape, dtype=dtype)


@set_module_as('saiunit.math')
def full_like(
    a: Union[Quantity, jax.typing.ArrayLike],
    fill_value: Union[Quantity, jax.typing.ArrayLike],
    dtype: Optional[jax.typing.DTypeLike] = None,
    shape: Shape = None
) -> Union[Quantity, jax.Array]:
    """
    Return a new quantity or array with the same shape and type as a given array, filled with ``fill_value``.

    Parameters
    ----------
    a : Quantity or array_like
        The shape and data-type of ``a`` define these same attributes of the
        returned array.
    fill_value : Quantity or array_like
        Value to fill the new quantity or array with.  When ``a`` is a
        ``Quantity``, ``fill_value`` must have a compatible unit.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or sequence of ints, optional
        Overrides the shape of the result.  If not given, ``a.shape`` is
        used.

    Returns
    -------
    out : Quantity or jax.Array
        New array with the same shape and type as ``a``, filled with
        ``fill_value``.

    Raises
    ------
    TypeError
        If ``fill_value`` carries a unit but ``a`` is a plain array (not
        unitless), or vice-versa.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> su.math.full_like(jnp.array([1.0, 2.0]), 9.0)
        Array([9., 9.], dtype=float32)
        >>> su.math.full_like(jnp.array([1.0, 2.0]) * su.meter, 9.0 * su.meter)
        Quantity([9. 9.], "m")
    """
    a = maybe_custom_array(a)
    fill_value = maybe_custom_array(fill_value)
    if isinstance(fill_value, Quantity):
        if isinstance(a, Quantity):
            fill_value = fill_value.in_unit(a.unit)
            return Quantity(
                jnp.full_like(a.mantissa, fill_value.mantissa, dtype=dtype, shape=shape),
                unit=a.unit
            )
        else:
            if not fill_value.is_unitless:
                raise TypeError(
                    f'full_like requires "fill_value" to be dimensionless when "a" is a plain array, '
                    f'but got fill_value with unit={fill_value.unit}. '
                    f'Either pass a plain number as fill_value or wrap "a" as a Quantity.'
                )
            return Quantity(
                jnp.full_like(a, fill_value.mantissa, dtype=dtype, shape=shape),
                unit=fill_value.unit
            )
    else:
        if isinstance(a, Quantity):
            if not a.is_unitless:
                raise TypeError(
                    f'full_like requires "a" to be dimensionless when "fill_value" is a plain value, '
                    f'but got a with unit={a.unit}. '
                    f'Either pass a Quantity as fill_value or use a plain array for "a".'
                )
            return jnp.full_like(a.mantissa, fill_value, dtype=dtype, shape=shape)
        else:
            return jnp.full_like(a, fill_value, dtype=dtype, shape=shape)


@set_module_as('saiunit.math')
def diag(
    v: Union[Quantity, jax.typing.ArrayLike],
    k: int = 0,
    unit: Unit = UNITLESS
) -> Union[Quantity, jax.Array]:
    """
    Extract a diagonal or construct a diagonal array.

    If ``v`` is a 1-D array, ``diag`` constructs a 2-D array with ``v`` on
    the ``k``-th diagonal.  If ``v`` is a 2-D array, ``diag`` extracts the
    ``k``-th diagonal and returns a 1-D array.

    Parameters
    ----------
    v : Quantity or array_like
        Input array.  1-D inputs produce a 2-D diagonal matrix; 2-D inputs
        have their ``k``-th diagonal extracted.
    k : int, optional
        Diagonal in question.  The default is 0.  Use ``k > 0`` for
        diagonals above the main diagonal and ``k < 0`` for diagonals below.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  Ignored when ``v`` already
        carries a unit.

    Returns
    -------
    out : Quantity or jax.Array
        The extracted diagonal or constructed diagonal array.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> su.math.diag(jnp.array([1.0, 2.0, 3.0]))
        Array([[1., 0., 0.],
               [0., 2., 0.],
               [0., 0., 3.]], dtype=float32)
        >>> su.math.diag(jnp.array([1.0, 2.0]), unit=su.meter)
        Quantity([[1. 0.]
                  [0. 2.]], "m")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'diag requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    v = maybe_custom_array(v)
    if isinstance(v, Quantity):
        if not unit.is_unitless:
            v = v.in_unit(unit)
        return Quantity(jnp.diag(v.mantissa, k=k), unit=v.unit)
    else:
        if not unit.is_unitless:
            return jnp.diag(v, k=k) * unit
        else:
            return jnp.diag(v, k=k)


@set_module_as('saiunit.math')
def tril(
    m: Union[Quantity, jax.typing.ArrayLike],
    k: int = 0,
    unit: Unit = UNITLESS
) -> Union[Quantity, jax.Array]:
    """
    Return the lower triangle of an array.

    Return a copy of a matrix with the elements above the ``k``-th diagonal
    zeroed.  For arrays with ``ndim > 2``, ``tril`` applies to the final two
    axes.

    Parameters
    ----------
    m : Quantity or array_like
        Input array.
    k : int, optional
        Diagonal above which to zero elements.  ``k = 0`` is the main
        diagonal, ``k < 0`` is below it, and ``k > 0`` is above.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or jax.Array
        Lower triangle of ``m``, of the same shape and data-type as ``m``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> su.math.tril(jnp.ones((3, 3)))
        Array([[1., 0., 0.],
               [1., 1., 0.],
               [1., 1., 1.]], dtype=float32)
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'tril requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    m = maybe_custom_array(m)
    if isinstance(m, Quantity):
        if not unit.is_unitless:
            m = m.in_unit(unit)
        return Quantity(jnp.tril(m.mantissa, k=k), unit=m.unit)
    else:
        if not unit.is_unitless:
            return jnp.tril(m, k=k) * unit
        else:
            return jnp.tril(m, k=k)


@set_module_as('saiunit.math')
def triu(
    m: Union[Quantity, jax.typing.ArrayLike],
    k: int = 0,
    unit: Unit = UNITLESS
) -> Union[Quantity, jax.Array]:
    """
    Return the upper triangle of an array.

    Return a copy of an array with the elements below the ``k``-th diagonal
    zeroed.  For arrays with ``ndim > 2``, ``triu`` applies to the final two
    axes.

    Parameters
    ----------
    m : Quantity or array_like
        Input array.
    k : int, optional
        Diagonal below which to zero elements.  ``k = 0`` is the main
        diagonal, ``k < 0`` is below it, and ``k > 0`` is above.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or jax.Array
        Upper triangle of ``m``, of the same shape and data-type as ``m``.

    See Also
    --------
    tril : lower triangle of an array

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> su.math.triu(jnp.ones((3, 3)))
        Array([[1., 1., 1.],
               [0., 1., 1.],
               [0., 0., 1.]], dtype=float32)
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'triu requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    m = maybe_custom_array(m)
    if isinstance(m, Quantity):
        if not unit.is_unitless:
            m = m.in_unit(unit)
        return Quantity(jnp.triu(m.mantissa, k=k), unit=m.unit)
    else:
        if not unit.is_unitless:
            return jnp.triu(m, k=k) * unit
        else:
            return jnp.triu(m, k=k)


@set_module_as('saiunit.math')
def empty_like(
    prototype: Union[Quantity, jax.typing.ArrayLike],
    dtype: Optional[jax.typing.DTypeLike] = None,
    shape: Shape = None,
    unit: Unit = UNITLESS
) -> Union[Quantity, jax.Array]:
    """
    Return a new uninitialized quantity or array with the same shape and type as a given array.

    Parameters
    ----------
    prototype : Quantity or array_like
        The shape and data-type of ``prototype`` define these same attributes
        of the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple of ints, optional
        Overrides the shape of the result.  If not given,
        ``prototype.shape`` is used.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or jax.Array
        Array of uninitialized (arbitrary) data with the same shape and type
        as ``prototype``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> result = su.math.empty_like(jnp.array([1.0, 2.0, 3.0]))
        >>> result.shape
        (3,)
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'empty_like requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    prototype = maybe_custom_array(prototype)
    if isinstance(prototype, Quantity):
        if not unit.is_unitless:
            prototype = prototype.in_unit(unit)
        return Quantity(jnp.empty_like(prototype.mantissa, dtype=dtype), unit=prototype.unit)
    else:
        if not unit.is_unitless:
            return jnp.empty_like(prototype, dtype=dtype, shape=shape) * unit
        else:
            return jnp.empty_like(prototype, dtype=dtype, shape=shape)


@set_module_as('saiunit.math')
def ones_like(
    a: Union[Quantity, jax.typing.ArrayLike],
    dtype: Optional[jax.typing.DTypeLike] = None,
    shape: Shape = None,
    unit: Unit = UNITLESS
) -> Union[Quantity, jax.Array]:
    """
    Return a quantity or array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : Quantity or array_like
        The shape and data-type of ``a`` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple of ints, optional
        Overrides the shape of the result.  If not given, ``a.shape`` is
        used.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or jax.Array
        Array of ones with the same shape and type as ``a``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> su.math.ones_like(jnp.array([1.0, 2.0, 3.0]))
        Array([1., 1., 1.], dtype=float32)
        >>> su.math.ones_like(jnp.array([1.0, 2.0]) * su.meter)
        Quantity([1. 1.], "m")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'ones_like requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        if not unit.is_unitless:
            a = a.in_unit(unit)
        return Quantity(jnp.ones_like(a.mantissa, dtype=dtype, shape=shape), unit=a.unit)
    else:
        if not unit.is_unitless:
            return jnp.ones_like(a, dtype=dtype, shape=shape) * unit
        else:
            return jnp.ones_like(a, dtype=dtype, shape=shape)


@set_module_as('saiunit.math')
def zeros_like(
    a: Union[Quantity, jax.typing.ArrayLike],
    dtype: Optional[jax.typing.DTypeLike] = None,
    shape: Shape = None,
    unit: Unit = UNITLESS
) -> Union[Quantity, jax.Array]:
    """
    Return a quantity or array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : Quantity or array_like
        The shape and data-type of ``a`` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple of ints, optional
        Overrides the shape of the result.  If not given, ``a.shape`` is
        used.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or jax.Array
        Array of zeros with the same shape and type as ``a``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> su.math.zeros_like(jnp.array([1.0, 2.0, 3.0]))
        Array([0., 0., 0.], dtype=float32)
        >>> su.math.zeros_like(jnp.array([1.0, 2.0]) * su.meter)
        Quantity([0. 0.], "m")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'zeros_like requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        if not unit.is_unitless:
            a = a.in_unit(unit)
        return Quantity(jnp.zeros_like(a.mantissa, dtype=dtype, shape=shape), unit=a.unit)
    else:
        if not unit.is_unitless:
            return jnp.zeros_like(a, dtype=dtype, shape=shape) * unit
        else:
            return jnp.zeros_like(a, dtype=dtype, shape=shape)


@set_module_as('saiunit.math')
def asarray(
    a: Any,
    dtype: Optional[jax.typing.DTypeLike] = None,
    order: Optional[str] = None,
    unit: Optional[Unit] = None,
) -> Quantity | jax.Array | None:
    """
    Convert the input to a quantity or array.

    If ``unit`` is provided, the input is checked for compatible units and
    converted accordingly.  If ``unit`` is not provided, the unit is inferred
    from the input data.

    The function ``array`` is an alias for ``asarray``.

    Parameters
    ----------
    a : Quantity, array_like, list[Quantity], or list[array_like]
        Input data, in any form that can be converted to an array.  When a
        list of ``Quantity`` objects is given, all elements must share the
        same dimension.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout.  Defaults to ``'K'``.
    unit : Unit, optional
        Target unit of the returned ``Quantity``.  When given, all elements
        are converted to this unit.

    Returns
    -------
    out : Quantity or jax.Array
        Array interpretation of ``a``.

    Raises
    ------
    UnitMismatchError
        If elements of ``a`` have incompatible units, or if ``unit`` is
        specified but does not match the dimension of ``a``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su.math.array([1, 2, 3])
        Array([1, 2, 3], dtype=int32)
        >>> su.math.array([1, 2, 3] * su.meter)
        Quantity([1 2 3], "m")
        >>> su.math.asarray([1 * su.meter, 2 * su.meter])
        Quantity([1 2], "m")
    """
    if a is None:
        return a

    # get leaves
    leaves, treedef = jax.tree.flatten(a, is_leaf=lambda x: isinstance(x, Quantity))
    leaves = unit_scale_align_to_first(*leaves)
    leaf_unit = leaves[0].unit

    # get unit
    if unit is not None and not leaf_unit.is_unitless:
        if not isinstance(unit, Unit):
            raise TypeError(f'asarray requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
        leaves = [leaf.in_unit(unit) for leaf in leaves]
    else:
        unit = leaf_unit

    # reconstruct mantissa
    a = treedef.unflatten([leaf.mantissa for leaf in leaves])
    a = jnp.asarray(a, dtype=dtype, order=order)

    # returns
    if unit.is_unitless:
        return a
    return Quantity(a, unit=unit)


array = asarray


@set_module_as('saiunit.math')
def arange(
    start: Union[Quantity, jax.typing.ArrayLike] = None,
    stop: Optional[Union[Quantity, jax.typing.ArrayLike]] = None,
    step: Optional[Union[Quantity, jax.typing.ArrayLike]] = None,
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[Quantity, jax.Array]:
    """
    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including ``start`` but excluding
    ``stop``).  All of ``start``, ``stop``, and ``step`` must share the
    same unit when any of them is a ``Quantity``.

    Parameters
    ----------
    start : Quantity or array_like, optional
        Start of the interval (inclusive).  The default start value is 0.
    stop : Quantity or array_like
        End of the interval (exclusive).
    step : Quantity or array_like, optional
        Spacing between values.  The default step size is 1.
    dtype : data-type, optional
        The type of the output array.  If not given, the dtype is inferred
        from the other input arguments.

    Returns
    -------
    out : Quantity or jax.Array
        Array of evenly spaced values.

    Raises
    ------
    UnitMismatchError
        If ``start``, ``stop``, and ``step`` do not share the same unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su.math.arange(5)
        Array([0, 1, 2, 3, 4], dtype=int32)
        >>> su.math.arange(0 * su.meter, 3 * su.meter, 1 * su.meter)
        Quantity([0 1 2], "m")
    """
    # apply maybe_custom_array to inputs
    start = maybe_custom_array(start) if start is not None else start
    stop = maybe_custom_array(stop) if stop is not None else stop
    step = maybe_custom_array(step) if step is not None else step

    # checking the dimension of the data
    non_none_data = [d for d in (start, stop, step) if d is not None]
    if len(non_none_data) == 0:
        raise ValueError('arange requires at least one of start, stop, or step to be provided.')
    d1 = non_none_data[0]
    for d2 in non_none_data[1:]:
        fail_for_unit_mismatch(
            d1,
            d2,
            error_message="Start value {d1} and stop value {d2} have to have the same units.",
            d1=d1,
            d2=d2
        )

    # convert to array
    unit = get_unit(d1)
    start = start.in_unit(unit).mantissa if isinstance(start, Quantity) else start
    stop = stop.in_unit(unit).mantissa if isinstance(stop, Quantity) else stop
    step = step.in_unit(unit).mantissa if isinstance(step, Quantity) else step
    # compute
    with jax.ensure_compile_time_eval():
        r = jnp.arange(start, stop, step, dtype=dtype)
    return r if unit.is_unitless else Quantity(r, unit=unit)


@set_module_as('saiunit.math')
def linspace(
    start: Union[Quantity, jax.typing.ArrayLike],
    stop: Union[Quantity, jax.typing.ArrayLike],
    num: int = 50,
    endpoint: Optional[bool] = True,
    retstep: Optional[bool] = False,
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[Quantity, jax.Array]:
    """
    Return evenly spaced numbers over a specified interval.

    Returns ``num`` evenly spaced samples, calculated over the interval
    ``[start, stop]``.  The endpoint of the interval can optionally be
    excluded.

    Parameters
    ----------
    start : Quantity or array_like
        The starting value of the sequence.
    stop : Quantity or array_like
        The end value of the sequence.  Must have the same unit as
        ``start`` when either is a ``Quantity``.
    num : int, optional
        Number of samples to generate.  Default is 50.
    endpoint : bool, optional
        If ``True``, ``stop`` is the last sample.  Otherwise, it is not
        included.  Default is ``True``.
    retstep : bool, optional
        If ``True``, return ``(samples, step)``, where ``step`` is the
        spacing between samples.
    dtype : data-type, optional
        The type of the output array.

    Returns
    -------
    samples : Quantity or jax.Array
        ``num`` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``.

    Raises
    ------
    UnitMismatchError
        If ``start`` and ``stop`` do not share the same unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su.math.linspace(0, 10, 5)
        Array([ 0. ,  2.5,  5. ,  7.5, 10. ], dtype=float32)
        >>> su.math.linspace(0 * su.meter, 10 * su.meter, 5)
        Quantity([ 0.   2.5  5.   7.5 10. ], "m")
    """
    start = maybe_custom_array(start)
    stop = maybe_custom_array(stop)
    fail_for_unit_mismatch(
        start,
        stop,
        error_message="Start value {start} and stop value {stop} have to have the same units.",
        start=start,
        stop=stop,
    )
    unit = get_unit(start)
    start = start.in_unit(unit).mantissa if isinstance(start, Quantity) else start
    stop = stop.in_unit(unit).mantissa if isinstance(stop, Quantity) else stop
    with jax.ensure_compile_time_eval():
        result = jnp.linspace(start, stop, num=num, endpoint=endpoint, retstep=retstep, dtype=dtype)
    return result if unit.is_unitless else Quantity(result, unit=unit)


@set_module_as('saiunit.math')
def logspace(
    start: Union[Quantity, jax.typing.ArrayLike],
    stop: Union[Quantity, jax.typing.ArrayLike],
    num: Optional[int] = 50,
    endpoint: Optional[bool] = True,
    base: Optional[float] = 10.0,
    dtype: Optional[jax.typing.DTypeLike] = None
):
    """
    Return numbers spaced evenly on a log scale.

    In linear space, the sequence starts at ``base ** start`` and ends with
    ``base ** stop`` in ``num`` steps.

    Parameters
    ----------
    start : Quantity or array_like
        ``base ** start`` is the starting value of the sequence.
    stop : Quantity or array_like
        ``base ** stop`` is the final value of the sequence (unless
        ``endpoint`` is ``False``).  Must share the same unit as ``start``
        when either is a ``Quantity``.
    num : int, optional
        Number of samples to generate.  Default is 50.
    endpoint : bool, optional
        If ``True``, ``stop`` is the last sample.  Otherwise, it is not
        included.  Default is ``True``.
    base : float, optional
        The base of the log space.  Default is 10.0.
    dtype : data-type, optional
        The type of the output array.

    Returns
    -------
    samples : Quantity or jax.Array
        ``num`` samples, equally spaced on a log scale.

    Raises
    ------
    UnitMismatchError
        If ``start`` and ``stop`` do not share the same unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su.math.logspace(0, 2, 4)
        Array([  1.       ,   4.6415887,  21.544348 , 100.       ], dtype=float32)
    """
    start = maybe_custom_array(start)
    stop = maybe_custom_array(stop)
    fail_for_unit_mismatch(
        start,
        stop,
        error_message="Start value {start} and stop value {stop} have to have the same units.",
        start=start,
        stop=stop,
    )
    unit = get_unit(start)
    start = start.in_unit(unit).mantissa if isinstance(start, Quantity) else start
    stop = stop.in_unit(unit).mantissa if isinstance(stop, Quantity) else stop
    with jax.ensure_compile_time_eval():
        result = jnp.logspace(start, stop, num=num, endpoint=endpoint, base=base, dtype=dtype)
    return result if unit.is_unitless else Quantity(result, unit=unit)


@set_module_as('saiunit.math')
def fill_diagonal(
    a: Union[Quantity, jax.typing.ArrayLike],
    val: Union[Quantity, jax.typing.ArrayLike],
    wrap: Optional[bool] = False,
    inplace: Optional[bool] = False
) -> Union[Quantity, jax.Array]:
    """
    Fill the main diagonal of the given array of any dimensionality.

    For an array ``a`` with ``a.ndim >= 2``, the diagonal is the list of
    locations with indices ``a[i, i, ..., i]`` all identical.

    Parameters
    ----------
    a : Quantity or array_like
        Array in which to fill the diagonal.
    val : Quantity or array_like
        Value to be written on the diagonal.  Its unit must be compatible
        with that of ``a``.
    wrap : bool, optional
        If ``True``, the diagonal is "wrapped" after ``a.shape[1]`` and
        continues in the first column (for tall matrices).  Default is
        ``False``.
    inplace : bool, optional
        If ``True``, the diagonal is filled in-place.  Default is ``False``.

    Returns
    -------
    out : Quantity or jax.Array
        The input array with the diagonal filled.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> su.math.fill_diagonal(jnp.zeros((3, 3)), 5.0)
        Array([[5., 0., 0.],
               [0., 5., 0.],
               [0., 0., 5.]], dtype=float32)
    """
    a = maybe_custom_array(a)
    val = maybe_custom_array(val)
    if isinstance(val, Quantity):
        if isinstance(a, Quantity):
            val = val.in_unit(a.unit)
            return Quantity(jnp.fill_diagonal(a.mantissa, val.mantissa, wrap, inplace=inplace), unit=a.unit)
        else:
            return Quantity(jnp.fill_diagonal(a, val.mantissa, wrap, inplace=inplace), unit=val.unit)
    else:
        if isinstance(a, Quantity):
            return Quantity(jnp.fill_diagonal(a.mantissa, val, wrap, inplace=inplace), unit=a.unit)
        else:
            return jnp.fill_diagonal(a, val, wrap, inplace=inplace)


@set_module_as('saiunit.math')
def meshgrid(
    *xi: Union[Quantity, jax.typing.ArrayLike],
    copy: Optional[bool] = True,
    sparse: Optional[bool] = False,
    indexing: Optional[str] = 'xy'
) -> List[Union[Quantity, jax.Array]]:
    """
    Return coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of N-D
    scalar/vector fields over N-D grids, given one-dimensional coordinate
    arrays ``x1, x2, ..., xn``.

    Parameters
    ----------
    xi : Quantity or array_like
        1-D arrays representing the coordinates of a grid.
    copy : bool, optional
        Must be ``True`` (the default).  JAX does not support
        ``copy=False``.
    sparse : bool, optional
        If ``True``, return a sparse grid instead of a dense grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian (``'xy'``, default) or matrix (``'ij'``) indexing of
        output.

    Returns
    -------
    X1, X2, ..., XN : list of Quantity or jax.Array
        Coordinate matrices.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> x, y = su.math.meshgrid(jnp.array([1, 2]), jnp.array([3, 4]))
        >>> x
        Array([[1, 2],
               [1, 2]], dtype=int32)
    """

    # Apply maybe_custom_array to inputs before processing
    xi = tuple(maybe_custom_array(x) for x in xi)
    args = [asarray(x) for x in xi]
    if not copy:
        raise ValueError("jax.numpy.meshgrid only supports copy=True")
    if indexing not in ["xy", "ij"]:
        raise ValueError(f"Valid values for indexing are 'xy' and 'ij', got {indexing}")
    if any(a.ndim != 1 for a in args):
        raise ValueError("Arguments to jax.numpy.meshgrid must be 1D, got shapes "
                         f"{[a.shape for a in args]}")
    if indexing == "xy" and len(args) >= 2:
        args[0], args[1] = args[1], args[0]
    shape = [1 if sparse else a.shape[0] for a in args]
    f_shape = lambda i, a: [*shape[:i], a.shape[0], *shape[i + 1:]] if sparse else shape
    # use jax.tree.map to compatible with Quantity
    output = [
        jax.tree.map(lambda x: jax.lax.broadcast_in_dim(x, f_shape(i, x), (i,)), a)
        for i, a, in enumerate(args)
    ]
    if indexing == "xy" and len(args) >= 2:
        output[0], output[1] = output[1], output[0]
    return output


@set_module_as('saiunit.math')
def vander(
    x: Union[Quantity, jax.typing.ArrayLike],
    N: Optional[bool] = None,
    increasing: Optional[bool] = False,
    unit: Unit = UNITLESS
) -> Union[Quantity, jax.Array]:
    """
    Generate a Vandermonde matrix.

    The columns of the output matrix are powers of the input vector.

    Parameters
    ----------
    x : Quantity or array_like
        1-D input array.  Must be dimensionless if a ``Quantity``.
    N : int, optional
        Number of columns in the output.  If ``N`` is not specified, a
        square array is returned (``N = len(x)``).
    increasing : bool, optional
        Order of the powers of the columns.  If ``True``, the powers
        increase from left to right; if ``False`` (the default), they are
        reversed.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or jax.Array
        Vandermonde matrix.

    Raises
    ------
    TypeError
        If ``x`` carries a non-trivial unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> su.math.vander(jnp.array([1, 2, 3]), 3)
        Array([[1, 1, 1],
               [4, 2, 1],
               [9, 3, 1]], dtype=int32)
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        if not x.is_unitless:
            raise TypeError(
                f'vander requires "x" to be dimensionless, '
                f'but got x with unit={x.unit}. '
                f'Pass "unit_to_scale" or strip the unit before calling vander.'
            )
        x = x.mantissa
    r = jnp.vander(x, N=N, increasing=increasing)
    if not isinstance(unit, Unit):
        raise TypeError(f'vander requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    if not unit.is_unitless:
        return Quantity(r, unit=unit)
    else:
        return r


# indexing funcs
# --------------

tril_indices = jnp.tril_indices


@set_module_as('saiunit.math')
def tril_indices_from(
    arr: Union[Quantity, jax.typing.ArrayLike],
    k: Optional[int] = 0
) -> Tuple[jax.Array, jax.Array]:
    """
    Return the indices for the lower-triangle of an ``(n, m)`` array.

    Parameters
    ----------
    arr : Quantity or array_like
        The array for which the returned indices will be valid.
    k : int, optional
        Diagonal offset.  ``k = 0`` is the main diagonal, ``k < 0`` is
        below, and ``k > 0`` is above.

    Returns
    -------
    out : tuple of jax.Array
        Row and column indices for the lower triangle.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> row, col = su.math.tril_indices_from(jnp.ones((3, 3)))
        >>> row
        Array([0, 1, 1, 2, 2, 2], dtype=int32)
    """
    arr = maybe_custom_array(arr)
    if isinstance(arr, Quantity):
        return jnp.tril_indices_from(arr.mantissa, k=k)
    else:
        return jnp.tril_indices_from(arr, k=k)


triu_indices = jnp.triu_indices


@set_module_as('saiunit.math')
def triu_indices_from(
    arr: Union[Quantity, jax.typing.ArrayLike],
    k: Optional[int] = 0
) -> Tuple[jax.Array, jax.Array]:
    """
    Return the indices for the upper-triangle of an ``(n, m)`` array.

    Parameters
    ----------
    arr : Quantity or array_like
        The array for which the returned indices will be valid.
    k : int, optional
        Diagonal offset.  ``k = 0`` is the main diagonal, ``k < 0`` is
        below, and ``k > 0`` is above.

    Returns
    -------
    out : tuple of jax.Array
        Row and column indices for the upper triangle.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> row, col = su.math.triu_indices_from(jnp.ones((3, 3)))
        >>> row
        Array([0, 0, 0, 1, 1, 2], dtype=int32)
    """
    arr = maybe_custom_array(arr)
    if isinstance(arr, Quantity):
        return jnp.triu_indices_from(arr.mantissa, k=k)
    else:
        return jnp.triu_indices_from(arr, k=k)


# --- others ---


@set_module_as('saiunit.math')
def from_numpy(
    x: np.ndarray,
    unit: Unit = UNITLESS
) -> jax.Array | Quantity:
    """
    Convert a NumPy array to a JAX array, optionally attaching a unit.

    Parameters
    ----------
    x : numpy.ndarray
        The NumPy array to convert.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  When ``UNITLESS`` (the default)
        a plain JAX array is returned.

    Returns
    -------
    out : Quantity or jax.Array
        JAX array (or ``Quantity``) created from ``x``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import numpy as np
        >>> su.math.from_numpy(np.array([1.0, 2.0]), unit=su.meter)
        Quantity([1. 2.], "m")
    """
    x = maybe_custom_array(x)
    if not isinstance(unit, Unit):
        raise TypeError(f'from_numpy requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    if not unit.is_unitless:
        return jnp.array(x) * unit
    return jnp.array(x)


@set_module_as('saiunit.math')
def as_numpy(x):
    """
    Convert a JAX array (or ``Quantity``) to a NumPy array.

    Parameters
    ----------
    x : Quantity or array_like
        The input to convert.  If ``x`` is a ``Quantity``, the underlying
        mantissa (in current unit scale) is returned as a NumPy array.

    Returns
    -------
    out : numpy.ndarray
        NumPy array representation of ``x``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su.math.as_numpy(su.math.ones((3,)))
        array([1., 1., 1.], dtype=float32)
    """
    x = maybe_custom_array(x)
    return np.array(x)


@set_module_as('saiunit.math')
def tree_zeros_like(tree):
    """
    Create a tree with the same structure as the input, but with zeros in each leaf.

    Parameters
    ----------
    tree : pytree
        A JAX-compatible pytree (nested dicts, lists, tuples, etc.) whose
        leaves are arrays or ``Quantity`` objects.

    Returns
    -------
    out : pytree
        A tree with the same structure, where every leaf is replaced by a
        zero-filled array (or ``Quantity``) of the same shape, dtype, and
        unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> su.math.tree_zeros_like(tree)
        {'a': Array([0., 0.], dtype=float32), 'b': Array([0.], dtype=float32)}
    """
    tree = maybe_custom_array_tree(tree)
    return jax.tree.map(zeros_like, tree)


@set_module_as('saiunit.math')
def tree_ones_like(tree):
    """
    Create a tree with the same structure as the input, but with ones in each leaf.

    Parameters
    ----------
    tree : pytree
        A JAX-compatible pytree (nested dicts, lists, tuples, etc.) whose
        leaves are arrays or ``Quantity`` objects.

    Returns
    -------
    out : pytree
        A tree with the same structure, where every leaf is replaced by a
        ones-filled array (or ``Quantity``) of the same shape, dtype, and
        unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import jax.numpy as jnp
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> su.math.tree_ones_like(tree)
        {'a': Array([1., 1.], dtype=float32), 'b': Array([1.], dtype=float32)}
    """
    tree = maybe_custom_array_tree(tree)
    return jax.tree.map(ones_like, tree)
