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

from functools import wraps, partial
from typing import Sequence, Callable

import jax
import numpy as np
from jax import numpy as jnp

from saiunit._base_getters import get_magnitude, get_unit, maybe_decimal
from saiunit._base_quantity import Quantity
from saiunit._compatible_import import safe_map
from saiunit._misc import maybe_custom_array_tree
from ._misc import _ensure_index, _check_callable, _argnums_partial

__all__ = [
    'jacrev',
    'jacfwd',
    'jacobian',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_quantity(x):
    return isinstance(x, Quantity)


def _check_dtype(x, *, holomorphic: bool, allowed_dtype=np.floating, name: str = ""):
    """Validate leaf dtype for Jacobian computation."""
    try:
        dtype = x.dtype
    except AttributeError:
        dtype = np.result_type(x)
    if holomorphic:
        if not np.issubdtype(dtype, np.complexfloating):
            raise TypeError(
                f"{name} with holomorphic=True requires complex dtype, got {dtype}."
            )
    elif allowed_dtype is not None and not np.issubdtype(dtype, allowed_dtype):
        raise TypeError(
            f"{name} requires {allowed_dtype.__name__} inputs, got {dtype}."
        )


def _split(x, indices, axis):
    if isinstance(x, np.ndarray):
        return np.split(x, indices, axis)
    elif isinstance(x, Quantity):
        return x.split(indices, axis)
    else:
        return jnp.split(x, indices, axis)


def _unravel_array_into_pytree(pytree, axis, arr, is_leaf=None, divide_units=False):
    """Unravel an array into a PyTree with a given structure.

    Args:
        pytree: The pytree that provides the structure.
        axis: The parameter axis is either -1, 0, or 1.
        arr: The array to be unraveled.
        is_leaf: Optional leaf predicate for tree flattening.
        divide_units: If True, divide each part's unit by the corresponding leaf's unit.
    """
    leaves, treedef = jax.tree.flatten(pytree, is_leaf=is_leaf)
    axis = axis % arr.ndim
    shapes = [arr.shape[:axis] + np.shape(l) + arr.shape[axis + 1:] for l in leaves]
    parts = _split(arr, np.cumsum(safe_map(np.size, leaves[:-1])), axis)
    reshaped_parts = [x.reshape(shape) for x, shape in zip(parts, shapes)]
    if divide_units:
        reshaped_parts = [
            maybe_decimal(
                Quantity(get_magnitude(part), unit=get_unit(part) / get_unit(leaf))
            )
            for part, leaf in zip(reshaped_parts, leaves)
        ]
    return jax.tree.unflatten(treedef, reshaped_parts)


def _std_basis(pytree):
    leaves, _ = jax.tree.flatten(pytree)
    ndim = sum(safe_map(np.size, leaves))
    dtype = jax.dtypes.result_type(*leaves)
    flat_basis = jnp.eye(ndim, dtype=dtype)
    return _unravel_array_into_pytree(pytree, 1, flat_basis)


def _tree_transpose(outer, inner, pytree_to_transpose):
    outer_leaves, outer_treedef = jax.tree.flatten(outer, is_leaf=_is_quantity)
    inner_leaves, inner_treedef = jax.tree.flatten(inner, is_leaf=_is_quantity)
    outer_leaf_units = [get_unit(leaf) for leaf in outer_leaves]
    inner_leaf_units = [get_unit(leaf) for leaf in inner_leaves]

    flat, treedef = jax.tree.flatten(pytree_to_transpose, is_leaf=_is_quantity)
    inner_size = inner_treedef.num_leaves
    outer_size = outer_treedef.num_leaves
    if treedef.num_leaves != (inner_size * outer_size):
        expected_treedef = outer_treedef.compose(inner_treedef)
        raise TypeError(f"Mismatch\n{treedef}\n != \n{expected_treedef}")
    iter_flat = iter(flat)

    lol = [
        [
            maybe_decimal(
                Quantity(
                    get_magnitude(next(iter_flat)),
                    unit=inner_leaf_units[j] / outer_leaf_units[i]
                )
            )
            for j in range(inner_size)
        ]
        for i in range(outer_size)
    ]
    transposed_lol = zip(*lol)
    subtrees = map(partial(jax.tree.unflatten, outer_treedef), transposed_lol)
    return jax.tree.unflatten(inner_treedef, subtrees)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def jacrev(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False
) -> Callable:
    """
    Physical unit-aware reverse-mode Jacobian of ``fun``.

    This is the unit-aware counterpart of
    `jax.jacrev <https://jax.readthedocs.io/en/latest/_autosummary/jax.jacrev.html>`_.
    It computes the Jacobian matrix via reverse-mode automatic
    differentiation while correctly propagating physical units.

    Parameters
    ----------
    fun : callable
        Function whose Jacobian is to be computed. Its arguments at
        positions specified by ``argnums`` should be arrays, scalars,
        or standard Python containers thereof (possibly carrying
        physical units).
    argnums : int or tuple of int, optional
        Specifies which positional argument(s) to differentiate with
        respect to. Default is ``0``.
    has_aux : bool, optional
        If ``True``, ``fun`` is expected to return ``(output, aux)``
        where only ``output`` is differentiated. Default is ``False``.
    holomorphic : bool, optional
        Whether ``fun`` is promised to be holomorphic. Default is
        ``False``.
    allow_int : bool, optional
        Whether integer-valued inputs are allowed. Default is ``False``.

    Returns
    -------
    jacfun : callable
        A function with the same signature as ``fun`` that returns the
        Jacobian computed via reverse-mode AD. If ``has_aux=True``, it
        returns ``(jacobian, aux)``. Each Jacobian leaf carries the
        correct physical units (output unit / input unit).

    Notes
    -----
    ``jacrev`` generalises the standard Jacobian to nested Python
    containers (pytrees). The tree structure of
    ``jacrev(fun)(x)`` is formed by taking a tree product of the
    structure of ``fun(x)`` with the structure of ``x``.

    Examples
    --------
    Jacobian of a scalar-to-scalar function with units:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as u
        >>> import saiunit.autograd as suauto
        >>> def f(x):
        ...     return x ** 2
        >>> jac_fn = suauto.jacrev(f)
        >>> jac_fn(jnp.array(3.0) * u.ms)
        6.0 * ms

    Jacobian with multiple arguments:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as u
        >>> import saiunit.autograd as suauto
        >>> def g(x, y):
        ...     return x * y
        >>> jac_fn = suauto.jacrev(g, argnums=(0, 1))
        >>> x = jnp.array([3.0, 4.0]) * u.ohm
        >>> y = jnp.array([5.0, 6.0]) * u.mA
        >>> jac_x, jac_y = jac_fn(x, y)
    """
    _check_callable(fun)
    argnums = _ensure_index(argnums)
    input_dtype = None if allow_int else np.floating

    @wraps(fun)
    def jacfun(*args, **kwargs):
        args, kwargs = maybe_custom_array_tree((args, kwargs))
        argnums_, f_partial, dyn_args = _argnums_partial(fun, argnums, args, kwargs)
        jax.tree.map(partial(_check_dtype, holomorphic=holomorphic, allowed_dtype=input_dtype, name="jacrev"), dyn_args)
        if not has_aux:
            y, pullback = jax.vjp(f_partial, *dyn_args)
        else:
            y, pullback, aux = jax.vjp(f_partial, *dyn_args, has_aux=True)
        jax.tree.map(partial(_check_dtype, holomorphic=holomorphic, name="jacrev"), y)
        jac = jax.vmap(pullback)(_std_basis(y))
        jac = jac[0] if isinstance(argnums_, int) else jac
        jac_tree = jax.tree.map(
            lambda arr: _unravel_array_into_pytree(y, 0, arr, is_leaf=_is_quantity),
            jac,
            is_leaf=_is_quantity,
        )
        example_args = dyn_args[0] if isinstance(argnums_, int) else dyn_args
        jac_tree = _tree_transpose(outer=example_args, inner=y, pytree_to_transpose=jac_tree)
        if not has_aux:
            return jac_tree
        else:
            return jac_tree, aux

    return jacfun


def jacobian(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False
) -> Callable:
    """
    Alias of :func:`jacrev`.

    This is a convenience alias that delegates directly to
    :func:`jacrev`. See :func:`jacrev` for full documentation.

    Parameters
    ----------
    fun : callable
        Function whose Jacobian is to be computed.
    argnums : int or tuple of int, optional
        Specifies which positional argument(s) to differentiate with
        respect to. Default is ``0``.
    has_aux : bool, optional
        If ``True``, ``fun`` returns ``(output, aux)`` and only
        ``output`` is differentiated. Default is ``False``.
    holomorphic : bool, optional
        Whether ``fun`` is promised to be holomorphic. Default is
        ``False``.
    allow_int : bool, optional
        Whether integer-valued inputs are allowed. Default is ``False``.

    Returns
    -------
    jacfun : callable
        A function that computes the Jacobian of ``fun`` through
        reverse-mode automatic differentiation.

    See Also
    --------
    jacrev : The primary implementation.
    jacfwd : Forward-mode Jacobian computation.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as u
        >>> import saiunit.autograd as suauto
        >>> def f(x):
        ...     return x ** 2
        >>> jac_fn = suauto.jacobian(f)
        >>> jac_fn(jnp.array(3.0) * u.ms)
        6.0 * ms
    """
    return jacrev(
        fun,
        argnums=argnums,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int
    )


def jacfwd(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
) -> Callable:
    """
    Physical unit-aware forward-mode Jacobian of ``fun``.

    This is the unit-aware counterpart of
    `jax.jacfwd <https://jax.readthedocs.io/en/latest/_autosummary/jax.jacfwd.html>`_.
    It computes the Jacobian matrix via forward-mode automatic
    differentiation while correctly propagating physical units.

    Parameters
    ----------
    fun : callable
        Function whose Jacobian is to be computed. Its arguments at
        positions specified by ``argnums`` should be arrays, scalars,
        or standard Python containers thereof (possibly carrying
        physical units).
    argnums : int or tuple of int, optional
        Specifies which positional argument(s) to differentiate with
        respect to. Default is ``0``.
    has_aux : bool, optional
        If ``True``, ``fun`` is expected to return ``(output, aux)``
        where only ``output`` is differentiated. Default is ``False``.
    holomorphic : bool, optional
        Whether ``fun`` is promised to be holomorphic. Default is
        ``False``.

    Returns
    -------
    jacfun : callable
        A function with the same signature as ``fun`` that returns the
        Jacobian computed via forward-mode AD. If ``has_aux=True``, it
        returns ``(jacobian, aux)``. Each Jacobian leaf carries the
        correct physical units (output unit / input unit).

    Notes
    -----
    Forward-mode (``jacfwd``) is more efficient than reverse-mode
    (``jacrev``) when the number of inputs is smaller than the number
    of outputs.

    ``jacfwd`` generalises the standard Jacobian to nested Python
    containers (pytrees). The tree structure of
    ``jacfwd(fun)(x)`` is formed by taking a tree product of the
    structure of ``fun(x)`` with the structure of ``x``.

    See Also
    --------
    jacrev : Reverse-mode Jacobian computation.
    jacobian : Alias of ``jacrev``.

    Examples
    --------
    Jacobian of a scalar-to-scalar function with units:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as u
        >>> import saiunit.autograd as suauto
        >>> def f(x):
        ...     return x ** 2
        >>> jac_fn = suauto.jacfwd(f)
        >>> jac_fn(jnp.array(3.0) * u.ms)
        6.0 * ms

    Jacobian with multiple arguments:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit as u
        >>> import saiunit.autograd as suauto
        >>> def g(x, y):
        ...     return x * y
        >>> jac_fn = suauto.jacfwd(g, argnums=(0, 1))
        >>> x = jnp.array([3.0, 4.0]) * u.ohm
        >>> y = jnp.array([5.0, 6.0]) * u.mA
        >>> jac_x, jac_y = jac_fn(x, y)
    """
    _check_callable(fun)
    argnums = _ensure_index(argnums)

    @wraps(fun)
    def jacfun(*args, **kwargs):
        args, kwargs = maybe_custom_array_tree((args, kwargs))
        argnums_, f_partial, dyn_args = _argnums_partial(fun, argnums, args, kwargs)
        jax.tree.map(partial(_check_dtype, holomorphic=holomorphic, allowed_dtype=np.inexact, name="jacfwd"), dyn_args)
        if not has_aux:
            pushfwd: Callable = partial(jax.jvp, f_partial, dyn_args)
            y, jac = jax.vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
        else:
            pushfwd: Callable = partial(jax.jvp, f_partial, dyn_args, has_aux=True)
            y, jac, aux = jax.vmap(pushfwd, out_axes=(None, -1, None))(_std_basis(dyn_args))
        jax.tree.map(partial(_check_dtype, holomorphic=holomorphic, name="jacfwd"), y)
        example_args = dyn_args[0] if isinstance(argnums_, int) else dyn_args
        jac_tree = jax.tree.map(
            lambda arr: _unravel_array_into_pytree(example_args, -1, arr, is_leaf=_is_quantity, divide_units=True),
            jac,
            is_leaf=_is_quantity,
        )
        if not has_aux:
            return jac_tree
        else:
            return jac_tree, aux

    return jacfun
