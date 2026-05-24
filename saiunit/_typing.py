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

"""Internal type aliases used across saiunit.

Centralized here so the no-JAX path stays import-safe and so saiunit core
modules don't depend on the runtime shim (``_jax_compat``) just to grab a
type alias. External users should import these from :mod:`saiunit.typing`,
which re-exports every name in :data:`__all__`.

Aliases
-------
Array
    Either ``jax.Array`` (when JAX is installed) or a sentinel class whose
    ``isinstance`` check is always False (when JAX is absent). Safe to use in
    annotations and ``isinstance`` checks regardless of backend.
ArrayLike
    Narrow array-like alias: types that genuinely support ``.shape`` /
    ``.ndim`` / ``.dtype``. Excludes bare Python scalars so that static
    checkers don't emit ``union-attr`` false positives when callers read
    those attributes.
ScalarOrArrayLike
    Wide alias: everything :data:`jax.typing.ArrayLike` accepts, including
    bare Python scalars. Use sparingly — prefer :data:`ArrayLike`.
DTypeLike
    Anything acceptable as a NumPy/JAX dtype specifier.
Shape
    Sequence of integer dimensions, e.g. ``(3, 4)`` or ``[3, 4]``.
Axis
    Single axis index.
Axes
    A single axis index or a sequence of axis indices.
PyTree
    Opaque pytree alias (an arbitrarily-nested structure of registered
    container nodes whose leaves are arrays/scalars/etc.). See
    https://docs.jax.dev/en/latest/pytrees.html.
"""

from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np

try:
    import jax as _jax
    _HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only in no-jax CI job
    _jax = None  # type: ignore[assignment]
    _HAS_JAX = False


if _HAS_JAX:
    Array = _jax.Array
    ArrayLike = _jax.Array | np.ndarray | np.number | np.bool_
    ScalarOrArrayLike = ArrayLike | bool | int | float | complex
    DTypeLike = _jax.typing.DTypeLike
else:
    class _JaxSentinel:
        """Class that no real object can ever be an instance of.

        Used as a placeholder for ``jax.Array`` when JAX is not installed.
        ``isinstance(x, _JaxSentinel)`` is always ``False``, which is the
        correct answer when JAX is not installed.
        """

        __slots__ = ()

        def __init__(self, *args, **kwargs):  # pragma: no cover
            raise RuntimeError(
                "Cannot instantiate JAX sentinel type without JAX. "
                "Install with: pip install saiunit[jax]"
            )

    Array = type("Array", (_JaxSentinel,), {})  # type: ignore[misc, assignment]
    ArrayLike = np.ndarray | np.number | np.bool_  # type: ignore[misc, assignment]
    ScalarOrArrayLike = ArrayLike | bool | int | float | complex  # type: ignore[misc]
    DTypeLike = Any  # type: ignore[assignment, misc]


Shape = Sequence[int]
Axis = int
Axes = Union[int, Sequence[int]]
PyTree = Any


__all__ = [
    "Array",
    "ArrayLike",
    "ScalarOrArrayLike",
    "DTypeLike",
    "Shape",
    "Axis",
    "Axes",
    "PyTree",
]
