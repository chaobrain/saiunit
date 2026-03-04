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

from typing import Optional, Union, Sequence

import jax
from jax import lax
import jax.numpy as jnp

from saiunit._base_unit import Unit
from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as, maybe_custom_array

Shape = Union[int, Sequence[int]]

__all__ = [
    # array creation(given array)
    'zeros_like_array',

    # array creation(misc)
    'iota',
    'broadcasted_iota',
]


# array creation (given array)
@set_module_as('saiunit.lax')
def zeros_like_array(
    x: Union[Quantity, jax.typing.ArrayLike],
    unit: Optional[Unit] = None,
) -> Union[Quantity, jax.Array]:
    """Create a zero-filled array with the same shape and dtype as ``x``.

    Parameters
    ----------
    x : array_like or Quantity
        The template array whose shape and dtype are used.
    unit : Unit, optional
        If provided, the result will be a ``Quantity`` with this unit.
        If ``x`` is already a ``Quantity``, specifying ``unit`` converts
        ``x`` to that unit first.

    Returns
    -------
    result : jax.Array or Quantity
        A zero-filled array. If ``x`` is a ``Quantity`` (or ``unit`` is
        provided), the result is a ``Quantity`` with the corresponding unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import saiunit.lax as sulax
        >>> import jax.numpy as jnp
        >>> q = jnp.array([3.0, 5.0]) * u.meter
        >>> result = sulax.zeros_like_array(q)
        >>> result.mantissa
        Array([0., 0.], dtype=float32)
        >>> result.unit
        meter
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        if unit is not None:
            if not isinstance(unit, Unit):
                raise TypeError('unit must be an instance of Unit.')
            x = x.in_unit(unit)
        return Quantity(jnp.zeros_like(x.mantissa), unit=x.unit)
    else:
        if unit is not None:
            if not isinstance(unit, Unit):
                raise TypeError('unit must be an instance of Unit.')
            return jnp.zeros_like(x) * unit
        else:
            return jnp.zeros_like(x)


# array creation (misc)
@set_module_as('saiunit.lax')
def iota(
    dtype: jax.typing.DTypeLike,
    size: int,
    unit: Optional[Unit] = None,
) -> Union[Quantity, jax.Array]:
    """Create an iota array (integer sequence) with an optional unit.

    Wraps XLA's ``Iota`` operator.

    Parameters
    ----------
    dtype : DTypeLike
        The element type of the output array.
    size : int
        The number of elements.
    unit : Unit, optional
        If provided, the result is a ``Quantity`` with this unit.

    Returns
    -------
    result : jax.Array or Quantity
        An array ``[0, 1, 2, ..., size - 1]`` of the given dtype.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import saiunit.lax as sulax
        >>> result = sulax.iota(float, 5, unit=u.second)
        >>> result.mantissa
        Array([0., 1., 2., 3., 4.], dtype=float32)
        >>> result.unit
        second
    """
    if unit is not None:
        if not isinstance(unit, Unit):
            raise TypeError('unit must be an instance of Unit.')
        return lax.iota(dtype, size) * unit
    else:
        return lax.iota(dtype, size)


@set_module_as('saiunit.lax')
def broadcasted_iota(
    dtype: jax.typing.DTypeLike,
    shape: Shape,
    dimension: int,
    _sharding=None,
    unit: Optional[Unit] = None,
) -> Union[Quantity, jax.Array]:
    """Broadcast an iota array into the given shape along one dimension.

    Convenience wrapper around ``iota``.

    Parameters
    ----------
    dtype : DTypeLike
        The element type of the output array.
    shape : Shape
        The shape of the output array.
    dimension : int
        The dimension along which to broadcast the iota values.
    _sharding : optional
        Internal sharding parameter.
    unit : Unit, optional
        If provided, the result is a ``Quantity`` with this unit.

    Returns
    -------
    result : jax.Array or Quantity
        An array of the given shape with iota values along ``dimension``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import saiunit.lax as sulax
        >>> result = sulax.broadcasted_iota(float, (2, 3), 1, unit=u.meter)
        >>> result.mantissa
        Array([[0., 1., 2.],
               [0., 1., 2.]], dtype=float32)
    """
    if unit is not None:
        if not isinstance(unit, Unit):
            raise TypeError('unit must be an instance of Unit.')
        try:
            return lax.broadcasted_iota(dtype, shape, dimension, _sharding) * unit
        except TypeError:
            return lax.broadcasted_iota(dtype, shape, dimension) * unit
    else:
        try:
            return lax.broadcasted_iota(dtype, shape, dimension, _sharding)
        except TypeError:
            return lax.broadcasted_iota(dtype, shape, dimension)
