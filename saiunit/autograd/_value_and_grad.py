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
from typing import (Any, Sequence, Callable)

import jax

from ._misc import _ensure_index
from .._base import get_mantissa, get_unit, Quantity, maybe_decimal
from .._misc import maybe_custom_array_tree

__all__ = [
    'value_and_grad',
    'grad',
]


def value_and_grad(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable[..., tuple[Any, Any]]:
    """
    Physical unit-aware version of
    `jax.value_and_grad <https://jax.readthedocs.io/en/latest/_autosummary/jax.value_and_grad.html>`_.

    Example::

       >>> import jax.numpy as jnp
       >>> import saiunit as u
       >>> def simple_function(x):
       ...    return x ** 2
       >>> value_and_grad_fn = u.autograd.value_and_grad(simple_function)
       >>> value, grad = value_and_grad_fn(jnp.array(3.0) * u.ms)
       >>> value
       9.0 * ms ** 2
       >>> grad
       6.0 * ms

    Args:
        fun: A Python callable that computes a scalar loss given arguments.
        argnums: Optional, an integer or a tuple of integers. The argument number(s) to differentiate with respect to.
        has_aux: Optional, whether `fun` returns auxiliary data.
        holomorphic: Optional, whether to use a holomorphic or real-valued differentiation.
        allow_int: Optional, whether to allow differentiation through integer inputs.

    Returns:
        A function that computes the value and gradient of `fun` with respect to
        the argument(s) indicated by `argnums`.
    """

    argnums = jax.core.concrete_or_error(_ensure_index, argnums)

    def fun_return_unitless_loss(*args, **kwargs):
        if has_aux:
            loss, aux = fun(*args, **kwargs)
        else:
            loss = fun(*args, **kwargs)
            aux = None
        return get_mantissa(loss), (loss, aux)

    fun_transformed = jax.value_and_grad(
        fun_return_unitless_loss,
        argnums=argnums,
        has_aux=True,
        holomorphic=holomorphic,
        allow_int=allow_int,
    )

    @wraps(fun)
    def value_and_grad_fun(*args, **kwargs):
        args, kwargs = maybe_custom_array_tree((args, kwargs))

        # autograd as usual
        ((_, (loss, auxiliary_data)), gradient) = fun_transformed(*args, **kwargs)

        # gradient Quantity conversion
        args_to_grad = jax.tree.map(lambda i: args[i], argnums)
        loss_unit = get_unit(loss)
        gradient = jax.tree.map(
            lambda arg, grads: maybe_decimal(
                Quantity(get_mantissa(grads), unit=loss_unit / get_unit(arg))
            ),
            args_to_grad,
            gradient,
            is_leaf=lambda x: isinstance(x, Quantity)
        )

        # return
        if has_aux:
            return (loss, auxiliary_data), gradient
        else:
            return loss, gradient

    return value_and_grad_fun


def grad(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable:
    """
    Physical unit-aware version of `jax.grad <https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html>`_.

    Example::

       >>> import jax.numpy as jnp
       >>> import saiunit as u
       >>> def simple_function(x):
       ...    return x ** 2
       >>> grad_fn = u.autograd.grad(simple_function)
       >>> grad_fn(jnp.array(3.0) * u.ms)
       6.0 * ms

    Args:
        fun: A Python callable that computes a scalar loss given arguments.
        argnums: Optional, an integer or a tuple of integers. The argument number(s) to differentiate with respect to.
        has_aux: Optional, whether `fun` returns auxiliary data.
        holomorphic: Optional, whether to use a holomorphic or real-valued differentiation.
        allow_int: Optional, whether to allow differentiation through integer inputs.

    Returns:
        A function that computes the gradient of `fun` with respect to
        the argument(s) indicated by `argnums`.
    """
    value_and_grad_f = value_and_grad(
        fun,
        argnums,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int
    )

    @wraps(fun)
    def grad_f(*args, **kwargs):
        args, kwargs = maybe_custom_array_tree((args, kwargs))
        _, g = value_and_grad_f(*args, **kwargs)
        return g

    @wraps(fun)
    def grad_f_aux(*args, **kwargs):
        args, kwargs = maybe_custom_array_tree((args, kwargs))
        (_, aux), g = value_and_grad_f(*args, **kwargs)
        return g, aux

    return grad_f_aux if has_aux else grad_f
