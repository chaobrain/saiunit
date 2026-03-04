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

from functools import wraps
from typing import (Any, Sequence, Callable)

import jax

from saiunit._base_getters import get_mantissa, get_unit, maybe_decimal
from saiunit._base_quantity import Quantity
from saiunit._compatible_import import concrete_or_error
from saiunit._misc import maybe_custom_array_tree
from ._misc import _ensure_index

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

    Computes both the value and gradient of ``fun`` while correctly
    propagating physical units through the differentiation.

    Parameters
    ----------
    fun : callable
        A Python callable that computes a scalar loss given arguments.
        The output must be a scalar (possibly with physical units).
    argnums : int or tuple of int, optional
        Specifies which positional argument(s) to differentiate with
        respect to. Default is ``0``.
    has_aux : bool, optional
        If ``True``, ``fun`` is expected to return a pair ``(loss, aux)``
        where only ``loss`` is differentiated. Default is ``False``.
    holomorphic : bool, optional
        Whether to use holomorphic differentiation (for complex-valued
        functions). Default is ``False``.
    allow_int : bool, optional
        Whether to allow differentiation with respect to integer-valued
        inputs. Default is ``False``.

    Returns
    -------
    value_and_grad_fun : callable
        A function with the same signature as ``fun`` that returns a
        ``(value, gradient)`` pair. If ``has_aux=True``, it returns
        ``((value, aux), gradient)`` instead. Gradients carry the correct
        physical units derived from the output and input units.

    Examples
    --------
    Compute the value and gradient of a scalar function with units:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.autograd as suauto
        >>> def f(x):
        ...     return x ** 2
        >>> vg = suauto.value_and_grad(f)
        >>> value, grad = vg(jnp.array(3.0) * su.ms)
        >>> value
        9.0 * ms ** 2
        >>> grad
        6.0 * ms

    Differentiate with respect to multiple arguments:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.autograd as suauto
        >>> def g(x, y):
        ...     return x * y
        >>> vg = suauto.value_and_grad(g, argnums=(0, 1))
        >>> val, grads = vg(jnp.array(3.0) * su.ms, jnp.array(4.0) * su.mV)
        >>> grads[0]
        4.0 * mvolt
        >>> grads[1]
        3.0 * msecond
    """

    argnums = concrete_or_error(_ensure_index, argnums)

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
    Physical unit-aware version of
    `jax.grad <https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html>`_.

    Computes the gradient of ``fun`` while correctly propagating physical
    units through the differentiation.

    Parameters
    ----------
    fun : callable
        A Python callable that computes a scalar loss given arguments.
        The output must be a scalar (possibly with physical units).
    argnums : int or tuple of int, optional
        Specifies which positional argument(s) to differentiate with
        respect to. Default is ``0``.
    has_aux : bool, optional
        If ``True``, ``fun`` is expected to return a pair ``(loss, aux)``
        where only ``loss`` is differentiated. The returned function
        produces ``(gradient, aux)``. Default is ``False``.
    holomorphic : bool, optional
        Whether to use holomorphic differentiation (for complex-valued
        functions). Default is ``False``.
    allow_int : bool, optional
        Whether to allow differentiation with respect to integer-valued
        inputs. Default is ``False``.

    Returns
    -------
    grad_fun : callable
        A function with the same signature as ``fun`` that returns the
        gradient. If ``has_aux=True``, it returns ``(gradient, aux)``
        instead. Gradients carry the correct physical units derived
        from the output and input units.

    Examples
    --------
    Compute the gradient of a scalar function with units:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.autograd as suauto
        >>> def f(x):
        ...     return x ** 2
        >>> grad_fn = suauto.grad(f)
        >>> grad_fn(jnp.array(3.0) * su.ms)
        6.0 * ms

    Gradient with auxiliary data:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as su
        >>> import saiunit.autograd as suauto
        >>> def f_aux(x):
        ...     return x ** 2, x * 3
        >>> grad_fn = suauto.grad(f_aux, has_aux=True)
        >>> g, aux = grad_fn(jnp.array(3.0) * su.mV)
        >>> g
        6.0 * mvolt
        >>> aux
        9.0 * mvolt
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
        _, g = value_and_grad_f(*args, **kwargs)
        return g

    @wraps(fun)
    def grad_f_aux(*args, **kwargs):
        (_, aux), g = value_and_grad_f(*args, **kwargs)
        return g, aux

    return grad_f_aux if has_aux else grad_f
