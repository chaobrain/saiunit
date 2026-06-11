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

"""Multi-backend tests for ``Quantity.at[...].<op>(...)``.

Each test consumes the ``backend`` fixture from the project-level
``conftest.py`` and runs once per installed backend. Skips automatically when
a backend's library isn't installed.

ndonnx is covered separately: indexed updates raise ``BackendError`` there
because the symbolic graph has no notion of in-place update.
"""

from __future__ import annotations

import numpy as np
import pytest

import saiunit as u
from saiunit import BackendError, Quantity, meter, mV


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(q: Quantity) -> np.ndarray:
    """Return mantissa as a numpy array regardless of backend."""
    m = q.mantissa
    if hasattr(m, "compute"):  # dask
        m = m.compute()
    if hasattr(m, "detach"):  # torch
        m = m.detach().cpu().numpy()
    if hasattr(m, "unwrap_numpy"):  # ndonnx
        m = m.unwrap_numpy()
    return np.asarray(m)


def _make_quantity(values, unit, backend_name: str) -> Quantity:
    """Build a Quantity whose mantissa lives on ``backend_name``."""
    arr_np = np.asarray(values, dtype=np.float64)
    if backend_name == "numpy":
        return Quantity(arr_np, unit=unit)
    if backend_name == "jax":
        import jax.numpy as jnp
        return Quantity(jnp.asarray(arr_np), unit=unit)
    if backend_name == "cupy":
        import cupy as cp  # type: ignore[import-not-found]
        return Quantity(cp.asarray(arr_np), unit=unit)
    if backend_name == "torch":
        import torch
        return Quantity(torch.as_tensor(arr_np), unit=unit)
    if backend_name == "dask":
        import dask.array as da
        return Quantity(da.from_array(arr_np, chunks=2), unit=unit)
    if backend_name == "ndonnx":
        import ndonnx
        return Quantity(ndonnx.asarray(arr_np), unit=unit)
    raise AssertionError(f"unknown backend {backend_name!r}")


# ---------------------------------------------------------------------------
# ndonnx: every op raises BackendError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "op",
    ["set", "add", "multiply", "divide", "power", "min", "max"],
)
def test_ndonnx_at_raises(op):
    pytest.importorskip("ndonnx")
    q = _make_quantity([1.0, 2.0, 3.0], mV, "ndonnx")
    with pytest.raises(BackendError, match="ndonnx"):
        ref = q.at[0]
        if op == "power":
            ref.power(2)
        elif op == "multiply":
            ref.multiply(2.0)
        elif op == "divide":
            ref.divide(2.0)
        elif op == "min":
            ref.min(0.5 * mV)
        elif op == "max":
            ref.max(10.0 * mV)
        else:
            getattr(ref, op)(5.0 * mV)


def test_ndonnx_at_get_raises():
    pytest.importorskip("ndonnx")
    q = _make_quantity([1.0, 2.0, 3.0], mV, "ndonnx")
    with pytest.raises(BackendError, match="ndonnx"):
        q.at[0].get()


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------

def test_set_scalar_idx(backend):
    if backend == "ndonnx":
        pytest.skip("ndonnx covered by test_ndonnx_at_raises")
    q = _make_quantity([1.0, 2.0, 3.0, 4.0], mV, backend)
    r = q.at[1].set(99.0 * mV)
    assert r.unit == mV
    np.testing.assert_allclose(_to_numpy(r), [1.0, 99.0, 3.0, 4.0])


def test_set_int_array_idx(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([0.0, 0.0, 0.0, 0.0], mV, backend)
    idx = np.asarray([0, 2])
    r = q.at[idx].set(7.0 * mV)
    np.testing.assert_allclose(_to_numpy(r), [7.0, 0.0, 7.0, 0.0])


def test_set_does_not_mutate_original(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([1.0, 2.0, 3.0], mV, backend)
    _ = q.at[0].set(99.0 * mV)
    np.testing.assert_allclose(_to_numpy(q), [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# add — repeated-index accumulation (must match JAX semantics on jax/numpy/cupy/torch)
# ---------------------------------------------------------------------------

def test_add_scalar(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([0.0, 0.0, 0.0], mV, backend)
    r = q.at[1].add(5.0 * mV)
    np.testing.assert_allclose(_to_numpy(r), [0.0, 5.0, 0.0])


def test_add_repeated_indices_accumulates(backend):
    if backend == "ndonnx":
        pytest.skip()
    if backend == "dask":
        pytest.skip("dask uses bool-mask scatter; repeated-index 1D-int idx accumulation differs by design")
    q = _make_quantity([0.0, 0.0, 0.0, 0.0], mV, backend)
    idx = np.asarray([0, 0, 1])
    val = np.asarray([1.0, 2.0, 5.0])
    r = q.at[idx].add(val * mV)
    np.testing.assert_allclose(_to_numpy(r), [3.0, 5.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# multiply / divide
# ---------------------------------------------------------------------------

def test_multiply_scalar(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([2.0, 4.0, 8.0], mV, backend)
    r = q.at[1].multiply(0.5)
    np.testing.assert_allclose(_to_numpy(r), [2.0, 2.0, 8.0])
    assert r.unit == mV


def test_divide_scalar(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([2.0, 4.0, 8.0], mV, backend)
    r = q.at[2].divide(2.0)
    np.testing.assert_allclose(_to_numpy(r), [2.0, 4.0, 4.0])


# ---------------------------------------------------------------------------
# min / max
# ---------------------------------------------------------------------------

def test_min(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([3.0, 3.0, 3.0], mV, backend)
    r = q.at[0].min(1.0 * mV)
    np.testing.assert_allclose(_to_numpy(r), [1.0, 3.0, 3.0])


def test_max(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([1.0, 1.0, 1.0], mV, backend)
    r = q.at[2].max(5.0 * mV)
    np.testing.assert_allclose(_to_numpy(r), [1.0, 1.0, 5.0])


# ---------------------------------------------------------------------------
# power
# ---------------------------------------------------------------------------

def test_power_integer(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([2.0, 3.0, 4.0], meter, backend)
    r = q.at[1].power(2)
    np.testing.assert_allclose(_to_numpy(r), [2.0, 9.0, 4.0])
    # Quantity.at.power changes the unit on the entire array (this matches JAX's
    # behavior — repeated multiplication of unit). We assert that the unit
    # dimension is squared:
    assert r.unit == meter ** 2


def test_power_repeated_indices_accumulates(backend):
    # JAX applies the power once per index occurrence, so a repeated index
    # raises that element multiple times: ((x**2)**2) at idx 0. numpy/cupy must
    # match. (torch/dask are documented last-write-wins / mask-based.)
    if backend in ("ndonnx", "torch", "dask"):
        pytest.skip("repeated-index power accumulation is a documented limitation here")
    q = _make_quantity([2.0, 3.0, 4.0, 5.0], meter, backend)
    idx = np.asarray([0, 0])
    r = q.at[idx].power(2)
    # (2**2)**2 == 16 at idx 0; others unchanged.
    np.testing.assert_allclose(_to_numpy(r), [16.0, 3.0, 4.0, 5.0])
    assert r.unit == meter ** 2


def test_power_non_integer_raises(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([2.0, 3.0], meter, backend)
    with pytest.raises(TypeError):
        q.at[0].power(2.5)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def test_apply(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([1.0, 4.0, 9.0], mV, backend)
    r = q.at[1].apply(lambda x: x * 2)
    np.testing.assert_allclose(_to_numpy(r), [1.0, 8.0, 9.0])
    assert r.unit == mV


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_scalar(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([10.0, 20.0, 30.0], mV, backend)
    r = q.at[1].get()
    assert r.unit == mV
    np.testing.assert_allclose(_to_numpy(r), 20.0)


def test_get_array_idx(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([10.0, 20.0, 30.0, 40.0], mV, backend)
    idx = np.asarray([0, 2])
    r = q.at[idx].get()
    np.testing.assert_allclose(_to_numpy(r), [10.0, 30.0])


# ---------------------------------------------------------------------------
# OOB mode emulation (numpy/cupy/torch/jax — dask handles OOB via mask)
# ---------------------------------------------------------------------------

def test_set_oob_drop_default(backend):
    """Default ``mode=None`` drops out-of-bounds updates (JAX semantics)."""
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([1.0, 2.0, 3.0], mV, backend)
    # idx 10 is out of bounds → dropped → no change
    r = q.at[10].add(5.0 * mV)
    np.testing.assert_allclose(_to_numpy(r), [1.0, 2.0, 3.0])


def test_set_oob_clip_mode(backend):
    """mode='clip' clamps the index to the valid range."""
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([1.0, 2.0, 3.0], mV, backend)
    # idx 20 clipped to 2 (last position) → adds 5 there
    r = q.at[20].add(5.0 * mV, mode="clip")
    np.testing.assert_allclose(_to_numpy(r), [1.0, 2.0, 8.0])


def test_get_oob_clip(backend):
    """get with OOB and default mode returns clipped value (JAX semantics)."""
    if backend == "ndonnx":
        pytest.skip()
    if backend == "dask":
        pytest.skip("dask uses native indexing for get and raises on OOB ints")
    q = _make_quantity([10.0, 20.0, 30.0], mV, backend)
    r = q.at[20].get()
    np.testing.assert_allclose(_to_numpy(r), 30.0)


def test_get_oob_fill_default(backend):
    """get(mode='fill') with no fill_value uses dtype-specific default (NaN for float)."""
    if backend == "ndonnx":
        pytest.skip()
    if backend == "dask":
        pytest.skip("dask not exercised for fill mode in v0.3.0")
    q = _make_quantity([10.0, 20.0, 30.0], mV, backend)
    r = q.at[20].get(mode="fill")
    val = _to_numpy(r)
    assert np.isnan(val)


def test_get_oob_fill_with_quantity_fill_value(backend):
    """fill_value must be unit-compatible; its mantissa is used at OOB positions."""
    if backend == "ndonnx":
        pytest.skip()
    if backend == "dask":
        pytest.skip("dask not exercised for fill mode in v0.3.0")
    q = _make_quantity([10.0, 20.0, 30.0], mV, backend)
    r = q.at[20].get(mode="fill", fill_value=-1.0 * mV)
    np.testing.assert_allclose(_to_numpy(r), -1.0)


def test_get_oob_fill_value_dim_mismatch_raises(backend):
    if backend == "ndonnx":
        pytest.skip()
    if backend == "dask":
        pytest.skip()
    q = _make_quantity([10.0, 20.0, 30.0], mV, backend)
    with pytest.raises(u.UnitMismatchError):
        q.at[20].get(mode="fill", fill_value=-1.0)


# ---------------------------------------------------------------------------
# Unit checks (backend-agnostic — must hold across every backend)
# ---------------------------------------------------------------------------

def test_set_requires_compatible_unit(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([1.0, 2.0, 3.0], mV, backend)
    with pytest.raises(u.UnitMismatchError):
        q.at[0].set(5.0 * meter)


def test_add_requires_compatible_unit(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([1.0, 2.0, 3.0], mV, backend)
    with pytest.raises(u.UnitMismatchError):
        q.at[0].add(5.0)  # plain number, but quantity is not unitless


def test_multiply_unit_bearing_factor_raises(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([2.0, 4.0, 8.0], mV, backend)
    # A unit-bearing factor would silently re-label the untouched elements
    # (e.g. 2 mV -> 2 mV*m), so it must be rejected.
    with pytest.raises(TypeError, match="dimensionless"):
        q.at[1].multiply(3.0 * meter)
    with pytest.raises(TypeError, match="dimensionless"):
        q.at[1].divide(3.0 * meter)


# ---------------------------------------------------------------------------
# Backend-preserving (mantissa stays in the original backend)
# ---------------------------------------------------------------------------

def test_set_preserves_backend(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([1.0, 2.0, 3.0], mV, backend)
    r = q.at[0].set(99.0 * mV)
    assert r.backend == backend


# ---------------------------------------------------------------------------
# Dask-specific: laziness preserved
# ---------------------------------------------------------------------------

def test_dask_at_stays_lazy():
    da = pytest.importorskip("dask.array")
    q = _make_quantity([1.0, 2.0, 3.0, 4.0], mV, "dask")
    r = q.at[1].set(99.0 * mV)
    # The mantissa should still be a dask array (i.e., we haven't materialized)
    assert isinstance(r.mantissa, da.Array)
    np.testing.assert_allclose(r.mantissa.compute(), [1.0, 99.0, 3.0, 4.0])


def test_dask_bool_mask_set():
    pytest.importorskip("dask.array")
    q = _make_quantity([1.0, 2.0, 3.0, 4.0], mV, "dask")
    mask = np.array([True, False, True, False])
    r = q.at[mask].set(0.0 * mV)
    np.testing.assert_allclose(_to_numpy(r), [0.0, 2.0, 0.0, 4.0])


def test_dask_multidim_fancy_idx_not_implemented():
    da = pytest.importorskip("dask.array")
    arr = da.from_array(np.zeros((3, 3)), chunks=(2, 2))
    q = Quantity(arr, unit=mV)
    # 2D fancy integer indexing isn't supported in v0.3.0 for dask
    with pytest.raises(NotImplementedError):
        q.at[np.array([[0, 1]])].set(5.0 * mV)


# ---------------------------------------------------------------------------
# Scatter methods (Quantity.__setitem__, scatter_add/mul/div/max/min)
# also route through the new dispatch — smoke-test them per backend.
# ---------------------------------------------------------------------------

def test_scatter_add_method(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([1.0, 2.0, 3.0], mV, backend)
    r = q.scatter_add(0, 10.0 * mV)
    np.testing.assert_allclose(_to_numpy(r), [11.0, 2.0, 3.0])


def test_scatter_mul_method(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([1.0, 2.0, 3.0], mV, backend)
    r = q.scatter_mul(2, 10.0)
    np.testing.assert_allclose(_to_numpy(r), [1.0, 2.0, 30.0])


def test_scatter_max_method(backend):
    if backend == "ndonnx":
        pytest.skip()
    q = _make_quantity([1.0, 2.0, 3.0], mV, backend)
    r = q.scatter_max(0, 10.0 * mV)
    np.testing.assert_allclose(_to_numpy(r), [10.0, 2.0, 3.0])
