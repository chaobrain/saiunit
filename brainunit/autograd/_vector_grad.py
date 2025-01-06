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

from functools import wraps
from typing import Callable, Sequence

import jax
from jax import numpy as jnp
from jax._src.api import _vjp
from jax.api_util import argnums_partial
from jax.extend import linear_util

from brainunit._base import get_unit, maybe_decimal, Quantity, get_mantissa
from ._misc import _check_callable

__all__ = [
    'vector_grad',
]


def vector_grad(
    func: Callable,
    argnums: int | Sequence[int] = 0,
    return_value: bool = False,
    has_aux: bool = False,
    unit_aware: bool = True,
):
    """
    Unit-aware compute the gradient of a vector with respect to the input.

    Args:
        fun: A Python callable that computes a scalar loss given arguments.
        argnums: Optional, an integer or a tuple of integers. The argument number(s) to differentiate with respect to.
        return_value: Optional, bool. Whether to return the value of the function.
        has_aux: Optional, whether `fun` returns auxiliary data.
        unit_aware: Optional, whether to enable unit-aware computation.

    Returns:
        A function that computes the gradient of `fun` with respect to
        the argument(s) indicated by `argnums`.

    >>> import jax.numpy as jnp
    >>> import brainunit as u
    >>> def simple_function(x):
    ...    return x ** 2
    >>> vector_grad_fn = u.autograd.vector_grad(simple_function)
    >>> vector_grad_fn(jnp.array([3.0, 4.0]) * u.ms)
    [6.0, 8.0] * ms

    >>> import jax.numpy as jnp
    >>> import brainunit as u
    >>> def simple_function(x):
    ...    return x ** 2
    >>> vector_grad_fn = u.autograd.vector_grad(simple_function, return_value=True)
    >>> grad, value = vector_grad_fn(jnp.array([3.0, 4.0]) * u.ms)
    >>> grad
    [6.0, 8.0] * ms
    >>> value
    [9.0, 16.0] * ms ** 2
    """

    _check_callable(func)

    @wraps(func)
    def grad_fun(*args, **kwargs):
        f = linear_util.wrap_init(func, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args, require_static_args_hashable=False)
        if has_aux:
            y, vjp_fn, aux = _vjp(f_partial, *dyn_args, has_aux=True)
        else:
            y, vjp_fn = _vjp(f_partial, *dyn_args, has_aux=False)
        leaves, tree = jax.tree.flatten(y)
        if unit_aware:
            assert len(leaves) == 1, 'The function must return a single array when unit_aware is True.'
        tangents = jax.tree.unflatten(tree, [jnp.ones(l.shape, dtype=l.dtype) for l in leaves])
        grads = vjp_fn(tangents)
        if isinstance(argnums, int):
            grads = grads[0]
        if unit_aware:
            args_to_grad = jax.tree.map(lambda i: args[i], argnums)
            r_unit = get_unit(y)
            grads = jax.tree.map(
                lambda arg, grad: maybe_decimal(
                    Quantity(get_mantissa(grad), unit=r_unit / get_unit(arg))
                ),
                args_to_grad,
                grads,
                is_leaf=lambda x: isinstance(x, Quantity)
            )
        if has_aux:
            return (grads, y, aux) if return_value else (grads, aux)
        else:
            return (grads, y) if return_value else grads

    return grad_fun
