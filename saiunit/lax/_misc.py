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

from typing import Any, Callable, Sequence, Union

import jax
from jax import lax

from .._base import Quantity, maybe_decimal
from .._misc import set_module_as, maybe_custom_array

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
    """Wraps XLA's `Reduce
    <https://www.tensorflow.org/xla/operation_semantics#reduce>`_
    operator.

    ``init_values`` and ``computation`` together must form a `monoid
    <https://en.wikipedia.org/wiki/Monoid>`_
    for correctness. That is ``init_values`` must be an identity of
    ``computation``, and ``computation`` must be associative. XLA may exploit both
    of these properties during code generation; if either is violated the result
    is undefined.
    """
    operands = maybe_custom_array(operands)
    init_values = maybe_custom_array(init_values)
    return lax.reduce(operands, init_values, computation, dimensions)


def reduce_precision(
    operand: Union[jax.typing.ArrayLike, Quantity, float],
    exponent_bits: int,
    mantissa_bits: int
) -> jax.typing.ArrayLike:
    """Wraps XLA's `ReducePrecision
    <https://www.tensorflow.org/xla/operation_semantics#reduceprecision>`_
    operator.
    """
    operand = maybe_custom_array(operand)
    if isinstance(operand, Quantity):
        return maybe_decimal(lax.reduce_precision(operand.mantissa, exponent_bits, mantissa_bits))
    return lax.reduce_precision(operand, exponent_bits, mantissa_bits)


@set_module_as('saiunit.lax')
def broadcast_shapes(
    *shapes
):
    """Returns the shape that results from NumPy broadcasting of `shapes`."""
    return lax.broadcast_shapes(*shapes)
