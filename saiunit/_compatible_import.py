# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Cross-version JAX compatibility shims.

These helpers paper over churn in the ``jax.core`` / ``jax.extend`` / ``jax.util``
namespaces across JAX versions. When JAX is not installed, calling any of these
symbols raises :class:`BackendError`; importing this module is still safe.
"""

from typing import TypeVar, Iterable, Callable

from saiunit._jax_compat import HAS_JAX, jax, require_jax

__all__ = [
    'safe_map',
    'unzip2',
    'wrap_init',
    'Primitive',
    'concrete_or_error',
]

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


if HAS_JAX:
    from jax.extend import linear_util

    if jax.__version_info__ < (0, 4, 38):
        from jax.core import Primitive  # type: ignore[attr-defined]
    else:
        from jax.extend.core import Primitive

    # ``concrete_or_error`` moved from ``jax.core`` to ``jax.extend.core`` in
    # JAX 0.10; the ``jax.core`` alias is deprecated there and emits a
    # DeprecationWarning on import. Prefer the new ``jax.extend.core`` location,
    # fall back to ``jax.core`` for pre-0.10 JAX, and finally to a minimal shim.
    try:
        from jax.extend.core import concrete_or_error
    except ImportError:
        try:
            from jax.core import concrete_or_error
        except ImportError:
            def concrete_or_error(typ, val, context=""):
                """Minimal shim used when concrete_or_error is unavailable.

                Raises TypeError for traced/abstract values so callers don't silently
                receive a tracer where they expected a static Python value.
                """
                from jax.core import Tracer
                if isinstance(val, Tracer):
                    raise TypeError(
                        f"Expected a concrete value but got a JAX tracer ({val!r}). {context}"
                    )
                if typ is None:
                    return val
                return typ(val)


    def wrap_init(fun: Callable, args: tuple, kwargs: dict, name: str):
        if jax.__version_info__ < (0, 6, 0):
            f = linear_util.wrap_init(fun, kwargs)
        else:
            from jax.api_util import debug_info
            f = linear_util.wrap_init(fun, kwargs, debug_info=debug_info(name, fun, args, kwargs))
        return f


    if jax.__version_info__ < (0, 6, 0):
        from jax.util import safe_map, unzip2  # type: ignore[import-not-found]

    else:

        def safe_map(f, *args):
            args = list(map(list, args))
            n = len(args[0])
            for arg in args[1:]:
                if len(arg) != n:
                    raise ValueError(f'safe_map: length mismatch: {list(map(len, args))}')
            return list(map(f, *args))


        def unzip2(xys: Iterable[tuple[T1, T2]]) -> tuple[tuple[T1, ...], tuple[T2, ...]]:
            """
            Unzip sequence of length-2 tuples into two tuples.
            """
            # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
            # is too permissive about inputs, and does not guarantee a length-2 output.
            xs: list[T1] = []
            ys: list[T2] = []
            for x, y in xys:
                xs.append(x)
                ys.append(y)
            return tuple(xs), tuple(ys)

else:
    # No-JAX fallbacks: this module is only imported by JAX-using subpackages
    # (autograd, lax, sparse), so reaching these stubs at runtime means the
    # caller went past the import-time gate in saiunit/__init__.py — give the
    # same actionable install hint.

    class Primitive:  # type: ignore[no-redef]
        """Placeholder for ``jax.extend.core.Primitive`` (raises on construction)."""

        def __init__(self, *args, **kwargs):
            require_jax("jax primitives")

    def concrete_or_error(typ, val, context=""):  # type: ignore[no-redef]
        require_jax("concrete_or_error")

    def wrap_init(fun: Callable, args: tuple, kwargs: dict, name: str):  # type: ignore[no-redef]
        require_jax("wrap_init")

    def safe_map(f, *args):  # type: ignore[no-redef]
        require_jax("safe_map")

    def unzip2(xys):  # type: ignore[no-redef]
        require_jax("unzip2")
