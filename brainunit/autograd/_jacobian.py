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

from functools import wraps, partial
from typing import (Sequence, Callable, Any)

import jax
import numpy as np
from jax._src.api import (_jvp,
                          _vjp,
                          _check_input_dtype_jacrev,
                          _check_output_dtype_jacrev,
                          _check_input_dtype_jacfwd,
                          _check_output_dtype_jacfwd)
from jax.api_util import argnums_partial
from jax.extend import linear_util

from brainunit._base import Quantity, maybe_decimal, get_magnitude, get_unit
from ._misc import _ensure_index, _check_callable

__all__ = [
    'jacrev',
    'jacfwd',
    'jacobian',
]


def jacrev(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False
) -> Callable:
    """
    Physical unit-aware version of `jax.jacrev <https://jax.readthedocs.io/en/latest/_autosummary/jax.jacrev.html>`_.

    Args:
        fun: Function whose Jacobian is to be computed.
        argnums: Optional, integer or sequence of integers. Specifies which
            positional argument(s) to differentiate with respect to (default ``0``).
        has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
            first element is considered the output of the mathematical function to be
            differentiated and the second element is auxiliary data. Default False.
        holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be holomorphic. Default False.
        allow_int: Optional, bool. Indicates whether integer arguments are allowed. Default False.

    Returns:
        A function that computes the Jacobian of ``fun`` through reverse-mode automatic differentiation.

    >>> import jax.numpy as jnp
    >>> import brainunit as u
    >>> def simple_function1(x):
    ...    return x ** 2
>>> jac_fn = u.autograd.jacrev(simple_function1)
    >>> jac_fn(jnp.array(3.0) * u.ms)
    6.0 * ms

    >>> import jax.numpy as jnp
    >>> import brainunit as u
    >>> def simple_function2(x, y):
    ...    return x * y
    >>> jac_fn = u.autograd.jacrev(simple_function2, argnums=(0, 1))
    >>> x = jnp.array([3.0, 4.0]) * u.ohm
    >>> y = jnp.array([5.0, 6.0]) * u.mA
    >>> jac_fn(x, y)
    ([[5., 0.],
      [0., 6.]] * mA,
     [[3., 0.],
      [0., 4.]] * ohm)

    `jacrev` is a generalization of the usual definition of the JacRev(Jacobian Reverse Mode).
    that supports nested Python containers (i.e. pytrees) as inputs and outputs.
    The tree structure of ``brainunit.autograd.jacrev(fun)(x)`` is given by forming a tree
    product of the structure of ``fun(x)`` with a tree product of two copies of
    the structure of ``x``. A tree product of two tree structures is formed by
    replacing each leaf of the first tree with a copy of the second. For example:

    >>> import jax.numpy as jnp
    >>> import brainunit as u
    >>> def dict_function(inputs):
    ...    o1 = inputs['x'] * inputs['y']
    ...    o2 = inputs['x'] * inputs['z']
    ...    r = {'o1': o1, 'o2': o2}
    ...    return r, r
    >>> jac_fn = u.autograd.jacrev(dict_function, has_aux=True)
    >>> x = jnp.array([3.0, 4.0]) * u.ohm
    >>> y = jnp.array([5.0, 6.0]) * u.mA
    >>> z = jnp.array([7.0, 8.0]) * u.siemens
    >>> inp = {'x': x, 'y': y, 'z': z}
    >>> jac_fn(inp)
    ({'o1': {'x': ArrayImpl([[5., 0.],
                  [0., 6.]], dtype=float32) * mampere,
       'y': ArrayImpl([[3., 0.],
                  [0., 4.]], dtype=float32) * ohm,
       'z': ArrayImpl([[0., 0.],
                  [0., 0.]], dtype=float32) * mvolt / siemens},
      'o2': {'x': ArrayImpl([[7., 0.],
                  [0., 8.]], dtype=float32) * siemens,
       'y': ArrayImpl([[0., 0.],
                  [0., 0.]], dtype=float32) * 10.0^3 * amp ** -1,
       'z': ArrayImpl([[3., 0.],
                  [0., 4.]], dtype=float32) * ohm}},
     {'o1': ArrayImpl([15., 24.], dtype=float32) * mvolt,
      'o2': Array([21., 32.], dtype=float32)})

    Thus each leaf in the tree structure of ``brainunit.autograd.jacrev(fun)(x)`` corresponds to
    a leaf of ``fun(x)`` and a pair of leaves of ``x``. For each leaf in
    ``brainunit.autograd.jacrev(fun)(x)``, if the corresponding array leaf of ``fun(x)`` has
    shape ``(out_1, out_2, ...)`` and the corresponding array leaves of ``x`` have
    shape ``(in_1_1, in_1_2, ...)`` and ``(in_2_1, in_2_2, ...)`` respectively,
    then the JacRev leaf has shape ``(out_1, out_2, ..., in_1_1, in_1_2, ...,
    in_2_1, in_2_2, ...)``. In other words, the Python tree structure represents
    the block structure of the Hessian, with blocks determined by the input and
    output pytrees.

    In particular, an array is produced (with no pytrees involved) when the
    function input ``x`` and output ``fun(x)`` are each a single array, as in the
    ``simple_function`` example above. If ``fun(x)`` has shape ``(out1, out2, ...)`` and ``x``
    has shape ``(in1, in2, ...)`` then ``brainunit.autograd.jacrec(fun)(x)`` has shape
    ``(out1, out2, ..., in1, in2, ..., in1, in2, ...)``. To flatten pytrees into
    1D vectors, consider using :py:func:`jax.flatten_util.flatten_pytree`.
    """
    _check_callable(fun)

    @wraps(fun)
    def jacfun(*args, **kwargs):
        f = linear_util.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args, require_static_args_hashable=False)
        jax.tree.map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
        if not has_aux:
            y, pullback = _vjp(f_partial, *dyn_args)
        else:
            y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
        jax.tree.map(partial(_check_output_dtype_jacrev, holomorphic), y)
        jac = jax.vmap(pullback)(_std_basis(y))
        jac = jac[0] if isinstance(argnums, int) else jac
        jac_tree = jax.tree.map(partial(_jacrev_unravel, y, is_leaf=_is_quantity),
                                jac,
                                is_leaf=_is_quantity)
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = _tree_transpose(outer=example_args, inner=y, pytree_to_transpose=jac_tree)
        if not has_aux:
            return jac_tree
        else:
            return jac_tree, aux

    return jacfun


def _tree_transpose(
    outer: Any,
    inner: Any,
    pytree_to_transpose: Any
) -> Any:
    outer_leaves, outer_treedef = jax.tree.flatten(outer, is_leaf=_is_quantity)
    inner_leaves, inner_treedef = jax.tree.flatten(inner, is_leaf=_is_quantity)
    outer_leaf_units = [get_unit(leaf) for leaf in outer_leaves]
    inner_leaf_units = [get_unit(leaf) for leaf in inner_leaves]

    # tree transpose
    flat, treedef = jax.tree.flatten(pytree_to_transpose, is_leaf=_is_quantity)
    inner_size = inner_treedef.num_leaves
    outer_size = outer_treedef.num_leaves
    if treedef.num_leaves != (inner_size * outer_size):
        expected_treedef = outer_treedef.compose(inner_treedef)
        raise TypeError(f"Mismatch\n{treedef}\n != \n{expected_treedef}")
    iter_flat = iter(flat)

    # unit-aware tree transpose
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


def jacobian(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False
) -> Callable:
    """
    Alias of :func:`jacrev`.

    Args:
        fun: Function whose Jacobian is to be computed.
        argnums: Optional, integer or sequence of integers. Specifies which
            positional argument(s) to differentiate with respect to (default ``0``).
        has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
            first element is considered the output of the mathematical function to be
            differentiated and the second element is auxiliary data. Default False.
        holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be holomorphic. Default False.
        allow_int: Optional, bool. Indicates whether integer arguments are allowed. Default False.

    Returns:
        A function that computes the Jacobian of ``fun`` through reverse-mode automatic differentiation.
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
    holomorphic: bool = False
) -> Callable:
    """
    Physical unit-aware version of `jax.jacfwd <https://jax.readthedocs.io/en/latest/_autosummary/jax.jacfwd.html>`_.

    Args:
        fun: Function whose Jacobian is to be computed.
        argnums: Optional, integer or sequence of integers. Specifies which
            positional argument(s) to differentiate with respect to (default ``0``).
        has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
            first element is considered the output of the mathematical function to be
            differentiated and the second element is auxiliary data. Default False.
        holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be holomorphic. Default False.

    Returns:
        A function that computes the Jacobian of ``fun`` through forward-mode automatic differentiation.

    >>> import jax.numpy as jnp
    >>> import brainunit as u
    >>> def simple_function(x):
    ...    return x ** 2
    >>> jac_fn = u.autograd.jacfwd(simple_function)
    >>> jac_fn(jnp.array(3.0) * u.ms)
    6.0 * ms

    >>> import jax.numpy as jnp
    >>> import brainunit as u
    >>> def simple_function(x, y):
    ...    return x * y
    >>> jac_fn = u.autograd.jacfwd(simple_function, argnums=(0, 1))
    >>> x = jnp.array([3.0, 4.0]) * u.ohm
    >>> y = jnp.array([5.0, 6.0]) * u.mA
    >>> jac_fn(x, y)
    ([[5., 0.],
      [0., 6.]] * mA,
     [[3., 0.],
      [0., 4.]] * ohm)

    `jacfwd` is a generalization of the usual definition of the JacFwd(Jacobian Reverse Mode).
    that supports nested Python containers (i.e. pytrees) as inputs and outputs.
    The tree structure of ``brainunit.autograd.jacfwd(fun)(x)`` is given by forming a tree
    product of the structure of ``fun(x)`` with a tree product of two copies of
    the structure of ``x``. A tree product of two tree structures is formed by
    replacing each leaf of the first tree with a copy of the second. For example:

    >>> import jax.numpy as jnp
    >>> import brainunit as u
    >>> def dict_function(inputs):
    ...    o1 = inputs['x'] * inputs['y']
    ...    o2 = inputs['x'] * inputs['z']
    ...    r = {'o1': o1, 'o2': o2}
    ...    return r, r
    >>> jac_fn = u.autograd.jacfwd(dict_function, has_aux=True)
    >>> x = jnp.array([3.0, 4.0]) * u.ohm
    >>> y = jnp.array([5.0, 6.0]) * u.mA
    >>> z = jnp.array([7.0, 8.0]) * u.siemens
    >>> inp = {'x': x, 'y': y, 'z': z}
    >>> jac_fn(inp)
    ({'o1': {'x': ArrayImpl([[5., 0.],
                  [0., 6.]], dtype=float32) * mampere,
       'y': ArrayImpl([[3., 0.],
                  [0., 4.]], dtype=float32) * ohm,
       'z': ArrayImpl([[0., 0.],
                  [0., 0.]], dtype=float32) * mvolt / siemens},
      'o2': {'x': ArrayImpl([[7., 0.],
                  [0., 8.]], dtype=float32) * siemens,
       'y': ArrayImpl([[0., 0.],
                  [0., 0.]], dtype=float32) * 10.0^3 * amp ** -1,
       'z': ArrayImpl([[3., 0.],
                  [0., 4.]], dtype=float32) * ohm}},
     {'o1': ArrayImpl([15., 24.], dtype=float32) * mvolt,
      'o2': Array([21., 32.], dtype=float32)})

    Thus each leaf in the tree structure of ``brainunit.autograd.jacfwd(fun)(x)`` corresponds to
    a leaf of ``fun(x)`` and a pair of leaves of ``x``. For each leaf in
    ``brainunit.autograd.jacfwd(fun)(x)``, if the corresponding array leaf of ``fun(x)`` has
    shape ``(out_1, out_2, ...)`` and the corresponding array leaves of ``x`` have
    shape ``(in_1_1, in_1_2, ...)`` and ``(in_2_1, in_2_2, ...)`` respectively,
    then the JacFwd leaf has shape ``(out_1, out_2, ..., in_1_1, in_1_2, ...,
    in_2_1, in_2_2, ...)``. In other words, the Python tree structure represents
    the block structure of the Hessian, with blocks determined by the input and
    output pytrees.

    In particular, an array is produced (with no pytrees involved) when the
    function input ``x`` and output ``fun(x)`` are each a single array, as in the
    ``simple_function`` example above. If ``fun(x)`` has shape ``(out1, out2, ...)`` and ``x``
    has shape ``(in1, in2, ...)`` then ``brainunit.autograd.jacrec(fun)(x)`` has shape
    ``(out1, out2, ..., in1, in2, ..., in1, in2, ...)``. To flatten pytrees into
    1D vectors, consider using :py:func:`jax.flatten_util.flatten_pytree`.
    """
    _check_callable(fun)
    argnums = _ensure_index(argnums)

    @wraps(fun)
    def jacfun(*args, **kwargs):
        f = linear_util.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args, require_static_args_hashable=False)
        jax.tree.map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
        if not has_aux:
            pushfwd: Callable = partial(_jvp, f_partial, dyn_args)
            y, jac = jax.vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
        else:
            pushfwd: Callable = partial(_jvp, f_partial, dyn_args, has_aux=True)
            y, jac, aux = jax.vmap(pushfwd, out_axes=(None, -1, None))(_std_basis(dyn_args))
        jax.tree.map(partial(_check_output_dtype_jacfwd, holomorphic), y)
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = jax.tree.map(partial(_jacfwd_unravel, example_args, is_leaf=_is_quantity),
                                jac,
                                is_leaf=_is_quantity)
        if not has_aux:
            return jac_tree
        else:
            return jac_tree, aux

    return jacfun


def _std_basis(pytree):
    leaves, _ = jax.tree.flatten(pytree)
    ndim = sum(jax.util.safe_map(np.size, leaves))
    dtype = jax.dtypes.result_type(*leaves)
    flat_basis = jax.numpy.eye(ndim, dtype=dtype)
    return _unravel_array_into_pytree(pytree, 1, flat_basis)


def _jacfwd_unravel(input_pytree, arr, is_leaf=None):
    """
    Unravel an array into a PyTree with a given structure.

    Args:
        input_pytree: The pytree that provides the structure.
        arr: The array to be unraveled.
    """
    axis = -1
    leaves, treedef = jax.tree.flatten(input_pytree, is_leaf=is_leaf)
    axis = axis % arr.ndim
    shapes = [arr.shape[:axis] + np.shape(l) + arr.shape[axis + 1:] for l in leaves]
    parts = _split(arr, np.cumsum(jax.util.safe_map(np.size, leaves[:-1])), axis)
    reshaped_parts = [x.reshape(shape) for x, shape in zip(parts, shapes)]
    unit_reshaped_parts = [
        maybe_decimal(
            Quantity(
                get_magnitude(part),
                unit=get_unit(part) / get_unit(leaf)
            )
        )
        for part, leaf in zip(reshaped_parts, leaves)
    ]
    return jax.tree.unflatten(treedef, unit_reshaped_parts)


def _jacrev_unravel(output_pytree, arr, is_leaf=None):
    return _unravel_array_into_pytree(output_pytree, 0, arr, is_leaf=is_leaf)


def _unravel_array_into_pytree(pytree, axis, arr, is_leaf=None):
    """
    Unravel an array into a PyTree with a given structure.

    Args:
        pytree: The pytree that provides the structure.
        axis: The parameter axis is either -1, 0, or 1.  It controls the
          resulting shapes.
        arr: The array to be unraveled.
    """
    leaves, treedef = jax.tree.flatten(pytree, is_leaf=is_leaf)
    axis = axis % arr.ndim
    shapes = [arr.shape[:axis] + np.shape(l) + arr.shape[axis + 1:] for l in leaves]
    parts = _split(arr, np.cumsum(jax.util.safe_map(np.size, leaves[:-1])), axis)
    reshaped_parts = [x.reshape(shape) for x, shape in zip(parts, shapes)]
    return jax.tree.unflatten(treedef, reshaped_parts)


def _split(x, indices, axis):
    if isinstance(x, np.ndarray):
        return np.split(x, indices, axis)
    elif isinstance(x, Quantity):
        return x.split(indices, axis)
    else:
        return x._split(indices, axis)


def _is_quantity(x):
    return isinstance(x, Quantity)
