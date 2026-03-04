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

from typing import Union

import jax
from jax import lax

from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as
from saiunit.math._fun_remove_unit import _fun_remove_unit_unary, _fun_logic_binary

__all__ = [
    # math funcs remove unit (unary)
    'population_count', 'clz',

    # logic funcs (unary)

    # logic funcs (binary)
    'eq', 'ne', 'ge', 'gt', 'le', 'lt',

]


# math funcs remove unit (unary)
@set_module_as('saiunit.lax')
def population_count(
    x: Union[jax.typing.ArrayLike, Quantity],
) -> jax.Array:
    r"""Elementwise popcount: count the number of set bits in each element.

    Parameters
    ----------
    x : array_like or Quantity
        Input integer array. If a ``Quantity``, the unit is stripped before
        computing.

    Returns
    -------
    result : jax.Array
        The number of set bits in each element. Always unitless.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1, 3, 7])
        >>> sulax.population_count(x)
        Array([1, 2, 3], dtype=int32)
    """
    return _fun_remove_unit_unary(lax.population_count, x)


@set_module_as('saiunit.lax')
def clz(
    x: Union[jax.typing.ArrayLike, Quantity],
) -> jax.Array:
    r"""Elementwise count of leading zeros.

    Parameters
    ----------
    x : array_like or Quantity
        Input integer array. If a ``Quantity``, the unit is stripped before
        computing.

    Returns
    -------
    result : jax.Array
        The count of leading zeros in each element. Always unitless.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1, 2, 4], dtype=jnp.int32)
        >>> sulax.clz(x)
        Array([31, 30, 29], dtype=int32)
    """
    return _fun_remove_unit_unary(lax.clz, x)


# logic funcs (binary)
@set_module_as('saiunit.lax')
def eq(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    r"""Elementwise equals: :math:`x = y`.

    Parameters
    ----------
    x : array_like or Quantity
        First operand.
    y : array_like or Quantity
        Second operand. Must have the same unit as ``x``.

    Returns
    -------
    result : jax.Array
        Boolean array. Always unitless.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> import jax.numpy as jnp
        >>> a = jnp.array([1.0, 2.0, 3.0]) * su.meter
        >>> b = jnp.array([1.0, 5.0, 3.0]) * su.meter
        >>> sulax.eq(a, b)
        Array([ True, False,  True], dtype=bool)
    """
    return _fun_logic_binary(lax.eq, x, y)


@set_module_as('saiunit.lax')
def ne(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    r"""Elementwise not-equals: :math:`x \neq y`.

    Parameters
    ----------
    x : array_like or Quantity
        First operand.
    y : array_like or Quantity
        Second operand. Must have the same unit as ``x``.

    Returns
    -------
    result : jax.Array
        Boolean array. Always unitless.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> import jax.numpy as jnp
        >>> a = jnp.array([1.0, 2.0, 3.0]) * su.meter
        >>> b = jnp.array([1.0, 5.0, 3.0]) * su.meter
        >>> sulax.ne(a, b)
        Array([False,  True, False], dtype=bool)
    """
    return _fun_logic_binary(lax.ne, x, y)


@set_module_as('saiunit.lax')
def ge(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    r"""Elementwise greater-than-or-equals: :math:`x \geq y`.

    Parameters
    ----------
    x : array_like or Quantity
        First operand.
    y : array_like or Quantity
        Second operand. Must have the same unit as ``x``.

    Returns
    -------
    result : jax.Array
        Boolean array. Always unitless.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> import jax.numpy as jnp
        >>> a = jnp.array([1.0, 3.0, 2.0]) * su.second
        >>> b = jnp.array([2.0, 2.0, 2.0]) * su.second
        >>> sulax.ge(a, b)
        Array([False,  True,  True], dtype=bool)
    """
    return _fun_logic_binary(lax.ge, x, y)


@set_module_as('saiunit.lax')
def gt(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    r"""Elementwise greater-than: :math:`x > y`.

    Parameters
    ----------
    x : array_like or Quantity
        First operand.
    y : array_like or Quantity
        Second operand. Must have the same unit as ``x``.

    Returns
    -------
    result : jax.Array
        Boolean array. Always unitless.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> import jax.numpy as jnp
        >>> a = jnp.array([1.0, 3.0, 2.0]) * su.second
        >>> b = jnp.array([2.0, 2.0, 2.0]) * su.second
        >>> sulax.gt(a, b)
        Array([False,  True, False], dtype=bool)
    """
    return _fun_logic_binary(lax.gt, x, y)


@set_module_as('saiunit.lax')
def le(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    r"""Elementwise less-than-or-equals: :math:`x \leq y`.

    Parameters
    ----------
    x : array_like or Quantity
        First operand.
    y : array_like or Quantity
        Second operand. Must have the same unit as ``x``.

    Returns
    -------
    result : jax.Array
        Boolean array. Always unitless.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> import jax.numpy as jnp
        >>> a = jnp.array([1.0, 3.0, 2.0]) * su.second
        >>> b = jnp.array([2.0, 2.0, 2.0]) * su.second
        >>> sulax.le(a, b)
        Array([ True, False,  True], dtype=bool)
    """
    return _fun_logic_binary(lax.le, x, y)


@set_module_as('saiunit.lax')
def lt(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    r"""Elementwise less-than: :math:`x < y`.

    Parameters
    ----------
    x : array_like or Quantity
        First operand.
    y : array_like or Quantity
        Second operand. Must have the same unit as ``x``.

    Returns
    -------
    result : jax.Array
        Boolean array. Always unitless.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.lax as sulax
        >>> import jax.numpy as jnp
        >>> a = jnp.array([1.0, 3.0, 2.0]) * su.second
        >>> b = jnp.array([2.0, 2.0, 2.0]) * su.second
        >>> sulax.lt(a, b)
        Array([ True, False, False], dtype=bool)
    """
    return _fun_logic_binary(lax.lt, x, y)
