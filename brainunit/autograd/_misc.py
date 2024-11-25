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

import inspect
import operator
from functools import partial
from typing import Any

import jax


def _ensure_index(x: Any) -> int | tuple[int, ...]:
    """
    Ensure x is either an index or a tuple of indices.
    """
    x = jax.core.concrete_or_error(None, x, "expected a static index or sequence of indices.")
    try:
        return operator.index(x)
    except TypeError:
        return tuple(map(operator.index, x))


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
