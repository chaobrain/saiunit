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

"""Cross-backend smoke tests for ``saiunit.linalg``.

Each test below is parameterized via the ``backend`` fixture defined in the
top-level ``conftest.py``, so it runs once per available array backend
(numpy / jax / cupy / torch / dask / ndonnx).

Quantities are built with the ``_quantity`` helper, which wraps a NumPy
array and then calls ``Quantity.to_<backend>()`` — this avoids the
``array_api_compat.torch.asarray`` failure on nested Python lists, which
otherwise blocks Quantity construction for 2-D inputs under the torch
default backend.

The module is jax-free at import time so it is collected by every
``test_pure_*`` CI job. Known per-backend gaps are listed in
``_UNSUPPORTED`` and reported as ``pytest.skip``.
"""

import numpy as np
import numpy.testing as npt
import pytest

import saiunit as u
from saiunit import meter, second


# Map (backend, op) → reason for known per-backend gaps. These are
# limitations of the underlying array library / array_api_compat shim, not
# of saiunit itself, so a clean SKIP is the correct CI signal.
_UNSUPPORTED = {
    # torch (array_api_compat shim) is missing kwargs / attrs saiunit needs
    ("torch", "det"): "array_api_compat.torch has no attribute 'shape' (used by saiunit det path)",
    ("torch", "trace"): "array_api_compat.torch.trace rejects axis1/axis2 kwargs",
    ("torch", "diagonal"): "array_api_compat.torch.diagonal kwarg mismatch",
    ("torch", "cross"): "array_api_compat.torch.cross rejects axis kwarg",
    # dask: ops not exposed by array_api_compat.dask.array or its linalg submodule
    ("dask", "det"): "array_api_compat.dask.array exposes no linalg.det",
    ("dask", "cross"): "array_api_compat.dask.array exposes no cross",
    ("dask", "cond"): "array_api_compat.dask.array exposes no linalg.cond",
    ("dask", "slogdet"): "array_api_compat.dask.array.linalg exposes no slogdet",
    # ndonnx: matmul is the only linalg op currently dispatchable through saiunit
    ("ndonnx", "norm"): "ndonnx exposes no linalg.norm",
    ("ndonnx", "vector_norm"): "ndonnx exposes no linalg.vector_norm",
    ("ndonnx", "matrix_norm"): "ndonnx exposes no linalg.matrix_norm",
    ("ndonnx", "inv"): "ndonnx exposes no linalg.inv",
    ("ndonnx", "det"): "ndonnx exposes no shape attribute used by saiunit det path",
    ("ndonnx", "solve"): "ndonnx exposes no linalg.solve",
    ("ndonnx", "cross"): "ndonnx exposes no cross",
    ("ndonnx", "trace"): "ndonnx exposes no trace",
    ("ndonnx", "diagonal"): "ndonnx exposes no diagonal",
    ("ndonnx", "matrix_transpose"): "ndonnx exposes no linalg.matrix_transpose",
    ("ndonnx", "cond"): "ndonnx exposes no linalg.cond",
    ("ndonnx", "slogdet"): "ndonnx exposes no linalg submodule",
}


def _skip_if_unsupported(backend, op):
    reason = _UNSUPPORTED.get((backend, op))
    if reason is not None:
        pytest.skip(reason)


def _quantity(values, unit, backend):
    """Build a Quantity with mantissa native to ``backend``.

    Construct via ``np.asarray`` then convert with ``to_<backend>()`` —
    needed because ``array_api_compat.torch.asarray`` raises on nested
    Python lists used directly under the torch default backend.
    """
    q = u.Quantity(np.asarray(values, dtype=float), unit=unit)
    return getattr(q, f"to_{backend}")()


def test_norm(backend):
    _skip_if_unsupported(backend, "norm")
    q = _quantity([3.0, 4.0], meter, backend)
    r = u.linalg.norm(q)
    assert isinstance(r, u.Quantity)
    assert r.unit == meter
    assert r.backend == backend
    assert float(r.to_numpy().mantissa) == pytest.approx(5.0)


def test_vector_norm(backend):
    _skip_if_unsupported(backend, "vector_norm")
    q = _quantity([3.0, 4.0], meter, backend)
    r = u.linalg.vector_norm(q)
    assert isinstance(r, u.Quantity)
    assert r.unit == meter
    assert r.backend == backend
    assert float(r.to_numpy().mantissa) == pytest.approx(5.0)


def test_matrix_norm(backend):
    _skip_if_unsupported(backend, "matrix_norm")
    q = _quantity([[3.0, 4.0], [0.0, 0.0]], meter, backend)
    r = u.linalg.matrix_norm(q)
    assert isinstance(r, u.Quantity)
    assert r.unit == meter
    assert r.backend == backend


def test_inv(backend):
    _skip_if_unsupported(backend, "inv")
    q = _quantity(np.eye(3) * 2.0, meter, backend)
    r = u.linalg.inv(q)
    assert isinstance(r, u.Quantity)
    assert r.unit == meter ** -1
    assert r.backend == backend
    npt.assert_allclose(r.to_numpy().mantissa, np.eye(3) * 0.5, atol=1e-6)


def test_det(backend):
    _skip_if_unsupported(backend, "det")
    q = _quantity(np.eye(3) * 2.0, meter, backend)
    r = u.linalg.det(q)
    assert isinstance(r, u.Quantity)
    assert r.unit == meter ** 3
    assert r.backend == backend
    assert float(r.to_numpy().mantissa) == pytest.approx(8.0)


def test_matmul(backend):
    _skip_if_unsupported(backend, "matmul")
    a = _quantity(np.eye(3), meter, backend)
    b = _quantity(np.eye(3), second, backend)
    r = u.linalg.matmul(a, b)
    assert isinstance(r, u.Quantity)
    assert r.unit == meter * second
    assert r.backend == backend


def test_solve(backend):
    _skip_if_unsupported(backend, "solve")
    A = _quantity(np.eye(3) * 2.0, meter, backend)
    b = _quantity([1.0, 2.0, 3.0], second, backend)
    r = u.linalg.solve(A, b)
    assert isinstance(r, u.Quantity)
    assert r.unit == second / meter
    assert r.backend == backend
    npt.assert_allclose(r.to_numpy().mantissa, [0.5, 1.0, 1.5], atol=1e-6)


def test_cross(backend):
    _skip_if_unsupported(backend, "cross")
    a = _quantity([1.0, 0.0, 0.0], meter, backend)
    b = _quantity([0.0, 1.0, 0.0], second, backend)
    r = u.linalg.cross(a, b)
    assert isinstance(r, u.Quantity)
    assert r.unit == meter * second
    assert r.backend == backend
    npt.assert_allclose(r.to_numpy().mantissa, [0.0, 0.0, 1.0], atol=1e-6)


def test_trace(backend):
    _skip_if_unsupported(backend, "trace")
    q = _quantity(np.eye(3), meter, backend)
    r = u.linalg.trace(q)
    assert isinstance(r, u.Quantity)
    assert r.unit == meter
    assert r.backend == backend
    assert float(r.to_numpy().mantissa) == pytest.approx(3.0)


def test_diagonal(backend):
    _skip_if_unsupported(backend, "diagonal")
    q = _quantity([[1.0, 2.0], [3.0, 4.0]], meter, backend)
    r = u.linalg.diagonal(q)
    assert isinstance(r, u.Quantity)
    assert r.unit == meter
    assert r.backend == backend
    npt.assert_allclose(r.to_numpy().mantissa, [1.0, 4.0])


def test_matrix_transpose(backend):
    _skip_if_unsupported(backend, "matrix_transpose")
    q = _quantity([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], meter, backend)
    r = u.linalg.matrix_transpose(q)
    assert isinstance(r, u.Quantity)
    assert r.unit == meter
    assert r.backend == backend
    assert r.mantissa.shape == (3, 2)


def test_cond_returns_dimensionless(backend):
    _skip_if_unsupported(backend, "cond")
    q = _quantity(np.eye(3) * 2.0, meter, backend)
    r = u.linalg.cond(q)
    # cond strips units — the result is a raw backend scalar/array.
    arr = np.asarray(r.compute()) if hasattr(r, "compute") else np.asarray(r)
    assert float(arr) == pytest.approx(1.0, abs=1e-6)


def test_slogdet_returns_dimensionless(backend):
    _skip_if_unsupported(backend, "slogdet")
    q = _quantity(np.eye(3) * 2.0, meter, backend)
    sign, logabsdet = u.linalg.slogdet(q)
    sign_arr = np.asarray(sign.compute()) if hasattr(sign, "compute") else np.asarray(sign)
    logdet_arr = np.asarray(logabsdet.compute()) if hasattr(logabsdet, "compute") else np.asarray(logabsdet)
    assert float(sign_arr) == pytest.approx(1.0)
    assert float(logdet_arr) == pytest.approx(np.log(8.0), abs=1e-6)
