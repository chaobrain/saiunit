# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

from typing import TypeVar, Iterable, Callable

import jax

__all__ = [
    'safe_map',
    'unzip2',
    'debug_info',
    'wrap_init',
]

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")

from jax.extend import linear_util

if jax.__version_info__ < (0, 5, 0):
    from jax._src.api_util import debug_info

else:
    from jax.api_util import debug_info


def wrap_init(fun: Callable, args: tuple, kwargs: dict, name: str):
    if jax.__version_info__ < (0, 5, 0):
        f = linear_util.wrap_init(fun, kwargs)
    else:
        f = linear_util.wrap_init(fun, kwargs, debug_info=debug_info(name, fun, args, kwargs))
    return f



if jax.__version_info__ < (0, 6, 0):
    from jax.util import safe_map, unzip2

else:

    def safe_map(f, *args):
        args = list(map(list, args))
        n = len(args[0])
        for arg in args[1:]:
            assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
        return list(map(f, *args))


    def unzip2(xys: Iterable[tuple[T1, T2]] ) -> tuple[tuple[T1, ...], tuple[T2, ...]]:
        """Unzip sequence of length-2 tuples into two tuples."""
        # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
        # is too permissive about inputs, and does not guarantee a length-2 output.
        xs: list[T1] = []
        ys: list[T2] = []
        for x, y in xys:
            xs.append(x)
            ys.append(y)
        return tuple(xs), tuple(ys)

