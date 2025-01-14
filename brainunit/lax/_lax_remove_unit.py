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

from typing import Union

import jax
from jax import lax

from .._base import Quantity
from .._misc import set_module_as
from ..math._fun_remove_unit import _fun_remove_unit_unary, _fun_logic_binary

__all__ = [
    # math funcs remove unit (unary)
    'population_count', 'clz',

    # logic funcs (unary)

    # logic funcs (binary)
    'eq', 'ne', 'ge', 'gt', 'le', 'lt',

]


# math funcs remove unit (unary)
@set_module_as('brainunit.lax')
def population_count(
    x: Union[jax.typing.ArrayLike, Quantity],
) -> jax.Array:
    r"""Elementwise popcount, count the number of set bits in each element."""
    return _fun_remove_unit_unary(lax.population_count, x)


@set_module_as('brainunit.lax')
def clz(
    x: Union[jax.typing.ArrayLike, Quantity],
) -> jax.Array:
    r"""Elementwise count-leading-zeros."""
    return _fun_remove_unit_unary(lax.clz, x)


# logic funcs (binary)
@set_module_as('brainunit.lax')
def eq(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    r"""Elementwise equals: :math:`x = y`."""
    return _fun_logic_binary(lax.eq, x, y)


@set_module_as('brainunit.lax')
def ne(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    r"""Elementwise not-equals: :math:`x \neq y`."""
    return _fun_logic_binary(lax.ne, x, y)


@set_module_as('brainunit.lax')
def ge(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    r"""Elementwise greater-than-or-equals: :math:`x \geq y`."""
    return _fun_logic_binary(lax.ge, x, y)


@set_module_as('brainunit.lax')
def gt(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    r"""Elementwise greater-than: :math:`x > y`."""
    return _fun_logic_binary(lax.gt, x, y)


@set_module_as('brainunit.lax')
def le(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    r"""Elementwise less-than-or-equals: :math:`x \leq y`."""
    return _fun_logic_binary(lax.le, x, y)


@set_module_as('brainunit.lax')
def lt(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[bool, jax.Array]:
    r"""Elementwise less-than: :math:`x < y`."""
    return _fun_logic_binary(lax.lt, x, y)
