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

import inspect
import operator
from functools import partial
from typing import Any, Callable, Sequence

from .._compatible_import import concrete_or_error


def _ensure_index(x: Any) -> int | tuple[int, ...]:
    """
    Ensure x is either an index or a tuple of indices.
    """
    x = concrete_or_error(None, x, "expected a static index or sequence of indices.")
    try:
        return operator.index(x)
    except TypeError:
        return tuple(map(operator.index, x))


def _argnums_partial(
    fun: Callable,
    argnums: int | Sequence[int],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[int | tuple[int, ...], Callable, tuple[Any, ...]]:
    """
    Build a function that accepts only differentiable positional args.

    This avoids depending on JAX private internals while preserving argnums
    semantics (including negative indices).
    """
    argnums = _ensure_index(argnums)
    nargs = len(args)

    def _normalize_index(i: int) -> int:
        i = i + nargs if i < 0 else i
        if i < 0 or i >= nargs:
            raise ValueError(f'argnums index {i} is out of bounds for {nargs} positional args.')
        return i

    if isinstance(argnums, int):
        normalized_argnums: int | tuple[int, ...] = _normalize_index(argnums)
        argnums_tuple = (normalized_argnums,)
    else:
        argnums_tuple = tuple(_normalize_index(i) for i in argnums)
        if len(argnums_tuple) == 0:
            raise ValueError('argnums must be non-empty.')
        if len(set(argnums_tuple)) != len(argnums_tuple):
            raise ValueError(f'argnums must not contain duplicate entries, got {argnums}.')
        normalized_argnums = argnums_tuple

    dynamic_args = tuple(args[i] for i in argnums_tuple)
    static_args = list(args)

    def partial_fun(*dyn_args):
        if len(dyn_args) != len(argnums_tuple):
            raise TypeError(f'Expected {len(argnums_tuple)} differentiated args, got {len(dyn_args)}.')
        merged_args = list(static_args)
        for idx, value in zip(argnums_tuple, dyn_args):
            merged_args[idx] = value
        return fun(*merged_args, **kwargs)

    return normalized_argnums, partial_fun, dynamic_args


def _isgeneratorfunction(fun):
    # re-implemented here because of https://bugs.python.org/issue33261
    while inspect.ismethod(fun):
        fun = fun.__func__
    while isinstance(fun, partial):
        fun = fun.func
    return inspect.isfunction(fun) and bool(fun.__code__.co_flags & inspect.CO_GENERATOR)


def _check_callable(fun):
    # In Python 3.10+, the only thing stopping us from supporting staticmethods
    # is that we can't take weak references to them, which the C++ JIT requires.
    if isinstance(fun, staticmethod):
        raise TypeError(f"staticmethod arguments are not supported, got {fun}")
    if not callable(fun):
        raise TypeError(f"Expected a callable value, got {fun}")
    if _isgeneratorfunction(fun):
        raise TypeError(f"Expected a function, got a generator function: {fun}")
