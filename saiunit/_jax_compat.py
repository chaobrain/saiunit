# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Optional-JAX compatibility shim.

This module centralizes every JAX symbol that saiunit's core uses. When JAX is
installed, the imports pass through to the real ``jax`` package. When JAX is
not installed, fallbacks are provided so that:

* ``import saiunit`` succeeds with NumPy as the default backend.
* ``isinstance(x, Array)`` and similar checks return ``False`` instead of
  crashing, because the sentinel classes can never have instances.
* Decorators like :func:`register_pytree_node_class` become no-ops.
* Basic pytree utilities (``tree.map``, ``tree.structure``) are implemented
  over Python containers (``list``, ``tuple``, ``dict``, ``None``) so that
  saiunit's decorators and indexing helpers keep working.
* Operations that genuinely require JAX (e.g. ``device_put``) raise
  :class:`BackendError` with an install hint when called.

External callers should not import this module directly. Inside saiunit, prefer
``from saiunit._jax_compat import jax, jnp, Array, ...``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable

import numpy as np

from saiunit._exceptions import BackendError

__all__ = [
    "HAS_JAX",
    "jax",
    "jnp",
    "Array",
    "ArrayLike",
    "ScalarOrArrayLike",
    "DTypeLike",
    "Tracer",
    "ShapedArray",
    "ShapeDtypeStruct",
    "DynamicJaxprTracer",
    "TracerArrayConversionError",
    "register_pytree_node_class",
    "ensure_compile_time_eval",
    "result_type",
    "canonicalize_dtype",
    "tree_map",
    "tree_structure",
    "tree",
    "device_put",
    "devices",
    "require_jax",
]


def require_jax(feature: str = "this feature") -> None:
    """Raise :class:`BackendError` when ``feature`` needs JAX but it's missing."""
    if not HAS_JAX:
        raise BackendError(
            f"{feature} requires JAX. Install with: pip install saiunit[jax]"
        )


try:
    import jax as _jax
    import jax.numpy as _jnp

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only in no-jax CI job
    _jax = None  # type: ignore[assignment]
    _jnp = None  # type: ignore[assignment]
    HAS_JAX = False


if HAS_JAX:
    jax = _jax
    jnp = _jnp
    Array = _jax.Array
    # Narrow array-like alias: types that genuinely support ``.shape`` /
    # ``.ndim`` / ``.dtype``. Excludes bare Python scalars (``bool``, ``int``,
    # ``float``, ``complex``) so that mypy/pyright don't emit ``union-attr``
    # false positives when downstream code reads those attributes. Callers
    # that want to accept Python scalars too should use :data:`ScalarOrArrayLike`.
    ArrayLike = _jax.Array | np.ndarray | np.number | np.bool_
    # Wide alias: everything :data:`jax.typing.ArrayLike` accepts, including
    # bare Python scalars. Use sparingly — prefer :data:`ArrayLike`.
    ScalarOrArrayLike = ArrayLike | bool | int | float | complex
    DTypeLike = _jax.typing.DTypeLike
    Tracer = _jax.core.Tracer
    ShapedArray = _jax.core.ShapedArray
    ShapeDtypeStruct = _jax.ShapeDtypeStruct
    from jax.interpreters.partial_eval import DynamicJaxprTracer  # noqa: F401
    TracerArrayConversionError = _jax.errors.TracerArrayConversionError
    from jax.tree_util import register_pytree_node_class  # noqa: F401
    ensure_compile_time_eval = _jax.ensure_compile_time_eval
    result_type = _jax.dtypes.result_type
    canonicalize_dtype = _jax.dtypes.canonicalize_dtype
    tree = _jax.tree
    tree_map = _jax.tree.map
    tree_structure = _jax.tree.structure
    device_put = _jax.device_put
    devices = _jax.devices
else:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]

    class _JaxSentinel:
        """Class that no real object can ever be an instance of.

        Used as a placeholder for ``jax.Array``, ``jax.core.Tracer``, etc.
        ``isinstance(x, _JaxSentinel)`` is always ``False``, which is the
        correct answer when JAX is not installed.
        """

        __slots__ = ()

        def __init__(self, *args, **kwargs):  # pragma: no cover
            raise BackendError(
                "Cannot instantiate JAX sentinel type without JAX. "
                "Install with: pip install saiunit[jax]"
            )

    Array = type("Array", (_JaxSentinel,), {})  # type: ignore[misc, assignment]
    Tracer = type("Tracer", (_JaxSentinel,), {})  # type: ignore[misc, assignment]
    ShapedArray = type("ShapedArray", (_JaxSentinel,), {})  # type: ignore[misc, assignment]
    ShapeDtypeStruct = type("ShapeDtypeStruct", (_JaxSentinel,), {})  # type: ignore[misc, assignment]
    DynamicJaxprTracer = type("DynamicJaxprTracer", (_JaxSentinel,), {})  # type: ignore[misc, assignment]
    ArrayLike = np.ndarray | np.number | np.bool_  # type: ignore[misc, assignment]
    ScalarOrArrayLike = ArrayLike | bool | int | float | complex  # type: ignore[misc]
    DTypeLike = Any  # type: ignore[assignment, misc]

    class TracerArrayConversionError(Exception):  # type: ignore[no-redef]
        """Placeholder for ``jax.errors.TracerArrayConversionError``.

        Never raised in the no-JAX path, but referenced by ``except`` clauses
        in saiunit core.
        """

    def register_pytree_node_class(cls):  # type: ignore[no-redef]
        """No-op decorator used when JAX is missing.

        Real registration with ``jax.tree_util`` happens only when JAX is
        installed; without JAX, pytree-flattening is unreachable, so this
        decorator simply returns the class unchanged.
        """
        return cls

    @contextlib.contextmanager
    def ensure_compile_time_eval():  # type: ignore[no-redef]
        """No-op replacement for ``jax.ensure_compile_time_eval()``.

        Outside a JAX trace there is nothing to "ensure"; the block body runs
        as ordinary Python code.
        """
        yield

    def result_type(*args):  # type: ignore[no-redef, misc]
        """Fallback for ``jax.dtypes.result_type`` via :func:`numpy.result_type`."""
        return np.result_type(*args)

    def canonicalize_dtype(dtype):  # type: ignore[no-redef, misc]
        """Fallback for ``jax.dtypes.canonicalize_dtype`` via :class:`numpy.dtype`."""
        return np.dtype(dtype)

    # -- Minimal pytree implementation over Python containers -----------------
    #
    # Real ``jax.tree.map`` understands registered pytree nodes (Quantity,
    # equinox modules, dataclasses, …). Without JAX, registration is skipped,
    # so every saiunit type is a leaf already. This fallback only needs to
    # traverse the standard Python containers — list, tuple, dict, None — plus
    # honour the caller's ``is_leaf`` predicate.

    class _TreeDef:
        """Opaque structure record returned by :func:`tree_structure`."""

        __slots__ = ("kind", "keys", "children")

        def __init__(self, kind, keys, children):
            self.kind = kind
            self.keys = keys
            self.children = children

        def __eq__(self, other):
            if not isinstance(other, _TreeDef):
                return NotImplemented
            return (
                self.kind == other.kind
                and self.keys == other.keys
                and self.children == other.children
            )

        def __hash__(self):
            return hash((self.kind, self.keys, tuple(self.children)))

        def __repr__(self):
            return f"PyTreeDef({self.kind}, keys={self.keys}, children={self.children})"

    def _flatten(x, is_leaf):
        if is_leaf is not None and is_leaf(x):
            return [x], _TreeDef("leaf", None, ())
        if x is None:
            return [], _TreeDef("none", None, ())
        if isinstance(x, list):
            leaves: list = []
            child_defs: list = []
            for item in x:
                sub_leaves, sub_def = _flatten(item, is_leaf)
                leaves.extend(sub_leaves)
                child_defs.append(sub_def)
            return leaves, _TreeDef("list", None, tuple(child_defs))
        if isinstance(x, tuple):
            leaves = []
            child_defs = []
            for item in x:
                sub_leaves, sub_def = _flatten(item, is_leaf)
                leaves.extend(sub_leaves)
                child_defs.append(sub_def)
            return leaves, _TreeDef("tuple", None, tuple(child_defs))
        if isinstance(x, dict):
            keys = tuple(sorted(x.keys(), key=lambda k: (str(type(k)), repr(k))))
            leaves = []
            child_defs = []
            for k in keys:
                sub_leaves, sub_def = _flatten(x[k], is_leaf)
                leaves.extend(sub_leaves)
                child_defs.append(sub_def)
            return leaves, _TreeDef("dict", keys, tuple(child_defs))
        return [x], _TreeDef("leaf", None, ())

    def _unflatten(treedef: _TreeDef, leaves_iter):
        if treedef.kind == "leaf":
            return next(leaves_iter)
        if treedef.kind == "none":
            return None
        if treedef.kind == "list":
            return [_unflatten(c, leaves_iter) for c in treedef.children]
        if treedef.kind == "tuple":
            return tuple(_unflatten(c, leaves_iter) for c in treedef.children)
        if treedef.kind == "dict":
            return {k: _unflatten(c, leaves_iter) for k, c in zip(treedef.keys, treedef.children)}
        raise RuntimeError(f"Unknown TreeDef kind: {treedef.kind!r}")

    def tree_map(f: Callable, tree_obj, *rest, is_leaf: Callable | None = None):  # type: ignore[no-redef, misc]
        """Fallback ``jax.tree.map`` over list/tuple/dict/None containers."""
        leaves, treedef = _flatten(tree_obj, is_leaf)
        rest_leaves = [_flatten(t, is_leaf)[0] for t in rest]
        for rl in rest_leaves:
            if len(rl) != len(leaves):
                raise ValueError(
                    "tree_map: trees have unequal leaf counts in the no-JAX fallback"
                )
        mapped = [f(*([leaves[i]] + [rl[i] for rl in rest_leaves])) for i in range(len(leaves))]
        return _unflatten(treedef, iter(mapped))

    def tree_structure(tree_obj, is_leaf: Callable | None = None):  # type: ignore[no-redef, misc]
        """Fallback ``jax.tree.structure`` returning an opaque :class:`_TreeDef`."""
        _, treedef = _flatten(tree_obj, is_leaf)
        return treedef

    class _TreeNamespace:
        """Namespace object that mimics ``jax.tree``'s ``map`` / ``structure``."""

        map = staticmethod(tree_map)
        structure = staticmethod(tree_structure)

    tree = _TreeNamespace()  # type: ignore[assignment]

    def device_put(x, device=None):  # type: ignore[no-redef, misc]
        """Reject ``jax.device_put`` calls with a clear install hint."""
        require_jax("jax.device_put")

    def devices(*args, **kwargs):  # type: ignore[no-redef]
        """Reject ``jax.devices`` calls with a clear install hint."""
        require_jax("jax.devices")
