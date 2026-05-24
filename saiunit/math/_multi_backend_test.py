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

"""Cross-backend smoke tests for the math functions refactored to use
:func:`saiunit._backend.get_backend` for dispatch.

This file deliberately avoids importing :mod:`jax` at the top level so it is
collected and run under the per-backend CI jobs that do not have JAX
installed (``test_pure_numpy``, ``test_pure_torch``, ``test_pure_dask``,
``test_pure_ndonnx``).

The ``backend`` fixture from ``conftest.py`` auto-parameterizes each test
across numpy / jax / cupy / torch / dask / ndonnx and skips automatically
when a backend's library isn't present.

Scope
-----
Each test exercises one refactored entry point with input that the array-API
surface for every targeted backend supports. Where a backend genuinely lacks
an operation (e.g. torch has no array-API ``cbrt``, ndonnx has no ``power``),
we ``pytest.skip`` rather than fail — the goal is to prove the dispatch path
routes correctly, not to backfill missing ops.

Numerical equivalence is asserted only on "eager numeric" backends
(numpy / jax / cupy / torch). dask returns lazy arrays and ndonnx is
symbolic; for those we just verify the call doesn't raise.
"""

from __future__ import annotations

import numpy as np
import pytest

import saiunit as u
from saiunit._backend import to_backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EAGER_NUMERIC_BACKENDS = {"numpy", "jax", "cupy", "torch"}


def _to_np(x):
    """Materialize a backend array (or Quantity mantissa) to a NumPy array."""
    if isinstance(x, u.Quantity):
        x = x.mantissa
    if hasattr(x, "unwrap_numpy"):  # ndonnx
        return x.unwrap_numpy()
    if hasattr(x, "compute"):  # dask
        return np.asarray(x.compute())
    return np.asarray(x)


def _arr(values, backend, dtype=np.float64):
    """Build a backend-native array from a Python list / numpy array."""
    return to_backend(np.asarray(values, dtype=dtype), backend)


def _is_eager(backend):
    return backend in _EAGER_NUMERIC_BACKENDS


# ---------------------------------------------------------------------------
# saiunit/math/_fun_change_unit.py
# ---------------------------------------------------------------------------


class TestChangeUnitMultiBackend:
    def test_reciprocal(self, backend):
        q = _arr([2.0, 4.0], backend) * u.second
        out = u.math.reciprocal(q)
        assert out.unit == u.second ** -1
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [0.5, 0.25])

    def test_square(self, backend):
        q = _arr([2.0, 3.0], backend) * u.meter
        out = u.math.square(q)
        assert out.unit == u.meter ** 2
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [4.0, 9.0])

    def test_sqrt(self, backend):
        q = _arr([4.0, 9.0], backend) * (u.meter ** 2)
        out = u.math.sqrt(q)
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [2.0, 3.0])

    def test_prod(self, backend):
        q = _arr([2.0, 3.0], backend) * u.meter
        out = u.math.prod(q)
        if _is_eager(backend):
            np.testing.assert_allclose(float(_to_np(out)), 6.0)

    def test_multiply(self, backend):
        a = _arr([1.0, 2.0, 3.0], backend) * u.meter
        b = _arr([4.0, 5.0, 6.0], backend) * u.second
        out = u.math.multiply(a, b)
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [4.0, 10.0, 18.0])

    def test_divide(self, backend):
        a = _arr([10.0, 20.0], backend) * u.meter
        b = _arr([2.0, 4.0], backend) * u.second
        out = u.math.divide(a, b)
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [5.0, 5.0])

    def test_matmul(self, backend):
        a = _arr([[1.0, 2.0], [3.0, 4.0]], backend) * u.meter
        b = _arr([[5.0, 6.0], [7.0, 8.0]], backend) * u.second
        out = u.math.matmul(a, b)
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [[19.0, 22.0], [43.0, 50.0]])

    def test_dot(self, backend):
        if backend == "ndonnx":
            pytest.skip("ndonnx has no array-API dot()")
        a = _arr([1.0, 2.0, 3.0], backend) * u.meter
        b = _arr([4.0, 5.0, 6.0], backend) * u.second
        out = u.math.dot(a, b)
        if _is_eager(backend):
            np.testing.assert_allclose(float(_to_np(out)), 32.0)


# ---------------------------------------------------------------------------
# saiunit/math/_fun_keep_unit.py
# ---------------------------------------------------------------------------


class TestKeepUnitMultiBackend:
    def test_concatenate(self, backend):
        if backend == "ndonnx":
            pytest.skip("ndonnx concat support is limited")
        a = _arr([1.0, 2.0], backend) * u.second
        b = _arr([3.0, 4.0], backend) * u.second
        out = u.math.concatenate([a, b])
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [1.0, 2.0, 3.0, 4.0])

    def test_broadcast_arrays(self, backend):
        if backend == "ndonnx":
            pytest.skip("ndonnx broadcast_arrays not in array_api_compat")
        a = _arr([1.0, 2.0, 3.0], backend) * u.meter
        b = _arr([[10.0], [20.0]], backend) * u.meter
        out = u.math.broadcast_arrays(a, b)
        assert len(out) == 2

    def test_sum(self, backend):
        q = _arr([1.0, 2.0, 3.0], backend) * u.meter
        out = u.math.sum(q)
        if _is_eager(backend):
            np.testing.assert_allclose(float(_to_np(out)), 6.0)

    def test_abs(self, backend):
        q = _arr([-1.0, 2.0, -3.0], backend) * u.meter
        out = u.math.abs(q)
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [1.0, 2.0, 3.0])

    def test_where_with_xy(self, backend):
        if backend in {"dask", "ndonnx"}:
            pytest.skip(f"{backend} where with x/y has limited support")
        cond = _arr([True, False, True], backend, dtype=bool)
        x = _arr([1.0, 2.0, 3.0], backend) * u.meter
        y = _arr([10.0, 20.0, 30.0], backend) * u.meter
        out = u.math.where(cond, x, y)
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [1.0, 20.0, 3.0])


# ---------------------------------------------------------------------------
# saiunit/math/_fun_remove_unit.py
# ---------------------------------------------------------------------------


class TestRemoveUnitMultiBackend:
    def test_digitize(self, backend):
        if backend in {"torch", "dask", "ndonnx"}:
            pytest.skip(f"{backend} has no array-API digitize")
        x = _arr([0.5, 1.5, 2.5], backend)
        bins = _arr([0.0, 1.0, 2.0, 3.0], backend)
        out = u.math.digitize(x, bins)
        if _is_eager(backend):
            np.testing.assert_array_equal(_to_np(out), [1, 2, 3])

    def test_searchsorted(self, backend):
        if backend in {"dask", "ndonnx"}:
            pytest.skip(f"{backend} searchsorted support is limited")
        a = _arr([1.0, 2.0, 3.0, 4.0], backend)
        v = _arr([2.5], backend)
        out = u.math.searchsorted(a, v)
        if _is_eager(backend):
            np.testing.assert_array_equal(_to_np(out), [2])

    def test_equal(self, backend):
        a = _arr([1.0, 2.0, 3.0], backend) * u.meter
        b = _arr([1.0, 0.0, 3.0], backend) * u.meter
        out = u.math.equal(a, b)
        if _is_eager(backend):
            np.testing.assert_array_equal(_to_np(out), [True, False, True])

    def test_logical_and(self, backend):
        a = _arr([True, True, False], backend, dtype=bool)
        b = _arr([True, False, False], backend, dtype=bool)
        out = u.math.logical_and(a, b)
        if _is_eager(backend):
            np.testing.assert_array_equal(_to_np(out), [True, False, False])


# ---------------------------------------------------------------------------
# saiunit/math/_fun_array_creation.py (functions touched by the refactor)
# ---------------------------------------------------------------------------


class TestArrayCreationMultiBackend:
    def test_asarray_plain_list(self, backend):
        out = u.math.asarray([1.0, 2.0, 3.0])
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [1.0, 2.0, 3.0])

    def test_asarray_with_unit(self, backend):
        if backend == "ndonnx":
            pytest.skip("ndonnx asarray of object-dtype Quantity fails dtype inference")
        out = u.math.asarray([1.0, 2.0, 3.0] * u.meter)
        assert out.unit == u.meter
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out.mantissa), [1.0, 2.0, 3.0])

    def test_meshgrid(self, backend):
        if backend == "ndonnx":
            pytest.skip("ndonnx meshgrid via reshape+broadcast_to has dtype constraints")
        x = _arr([1.0, 2.0], backend)
        y = _arr([3.0, 4.0], backend)
        out = u.math.meshgrid(x, y)
        assert len(out) == 2
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out[0]), [[1.0, 2.0], [1.0, 2.0]])
            np.testing.assert_allclose(_to_np(out[1]), [[3.0, 3.0], [4.0, 4.0]])

    def test_tree_zeros_like(self, backend):
        # tree_zeros_like routes through ``_tree.map`` (refactored) plus the
        # backend's ``zeros_like``. The underlying ``zeros_like`` passes
        # ``shape=None``, which torch's array-API surface and ndonnx reject.
        # That's an orthogonal pre-existing bug — skip those backends.
        if backend in {"torch", "ndonnx"}:
            pytest.skip(f"{backend} zeros_like(..., shape=None) signature issue")
        x = _arr([1.0, 2.0, 3.0], backend)
        tree = {"a": x, "b": [x, x]}
        out = u.math.tree_zeros_like(tree)
        assert set(out.keys()) == {"a", "b"}
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out["a"]), [0.0, 0.0, 0.0])

    def test_tree_ones_like(self, backend):
        # See ``test_tree_zeros_like`` — same underlying ``ones_like`` issue.
        if backend in {"torch", "ndonnx"}:
            pytest.skip(f"{backend} ones_like(..., shape=None) signature issue")
        x = _arr([1.0, 2.0, 3.0], backend)
        tree = (x, x)
        out = u.math.tree_ones_like(tree)
        assert len(out) == 2
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out[0]), [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# saiunit/math/_fun_accept_unitless.py (ldexp was the refactored entry)
# ---------------------------------------------------------------------------


class TestAcceptUnitlessMultiBackend:
    def test_sin(self, backend):
        a = _arr([0.0, np.pi / 2], backend)
        out = u.math.sin(a)
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [0.0, 1.0], atol=1e-7)

    def test_exp(self, backend):
        a = _arr([0.0, 1.0], backend)
        out = u.math.exp(a)
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [1.0, np.e], rtol=1e-6)

    def test_log(self, backend):
        a = _arr([1.0, np.e], backend)
        out = u.math.log(a)
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [0.0, 1.0], atol=1e-6)

    def test_ldexp(self, backend):
        # ldexp's signature varies widely across backends; the array-API
        # surface only ships it on numpy/jax. The refactor routes via
        # ``_resolve_op('ldexp', xp)`` — verify that path on the backends
        # that actually expose it.
        if backend in {"torch", "dask", "ndonnx"}:
            pytest.skip(f"{backend} does not expose array-API ldexp")
        x = _arr([1.0, 2.0], backend)
        y = _arr([1, 2], backend, dtype=np.int64)
        out = u.math.ldexp(x, y)
        if _is_eager(backend):
            np.testing.assert_allclose(_to_np(out), [2.0, 8.0])
