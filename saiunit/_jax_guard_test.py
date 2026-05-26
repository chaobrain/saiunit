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

import jax.numpy as jnp
import numpy as np
import pytest

import saiunit as u
from saiunit._jax_guard import require_jax_backend, jax_only


def test_require_jax_passes_for_jax():
    q = u.Quantity(jnp.array([1.0]), unit=u.meter)
    require_jax_backend("test_fn", q)  # no raise


def test_require_jax_raises_for_numpy():
    q = u.Quantity(np.array([1.0]), unit=u.meter)
    with pytest.raises(u.BackendError, match="requires the jax backend"):
        require_jax_backend("test_fn", q)


def test_require_jax_ignores_plain_numpy_array():
    # Plain ndarrays (not wrapped in Quantity) are allowed.
    arr = np.array([1.0])
    require_jax_backend("test_fn", arr)  # no raise


def test_require_jax_ignores_python_scalar():
    require_jax_backend("test_fn", 1.0, 2)  # no raise


def test_require_jax_raises_for_torch_quantity():
    torch = pytest.importorskip("torch")
    q = u.Quantity(torch.tensor([1.0]), unit=u.meter)
    with pytest.raises(u.BackendError, match="torch-backed Quantity"):
        require_jax_backend("test_fn", q)


def test_require_jax_raises_for_cupy_quantity():
    cupy = pytest.importorskip("cupy")
    q = u.Quantity(cupy.array([1.0]), unit=u.meter)
    with pytest.raises(u.BackendError, match="cupy-backed Quantity"):
        require_jax_backend("test_fn", q)


def test_require_jax_message_for_numpy_quantity_names_backend():
    q = u.Quantity(np.array([1.0]), unit=u.meter)
    with pytest.raises(u.BackendError, match="numpy-backed Quantity"):
        require_jax_backend("test_fn", q)


def test_require_jax_rejects_bare_torch_tensor():
    torch = pytest.importorskip("torch")
    t = torch.tensor([1.0])
    with pytest.raises(u.BackendError, match="torch"):
        require_jax_backend("test_fn", t)


def test_require_jax_rejects_bare_cupy_array():
    cupy = pytest.importorskip("cupy")
    arr = cupy.array([1.0])
    with pytest.raises(u.BackendError, match="cupy"):
        require_jax_backend("test_fn", arr)


def test_require_jax_raises_for_dask_quantity():
    da = pytest.importorskip("dask.array")
    q = u.Quantity(da.from_array(np.array([1.0]), chunks=1), unit=u.meter)
    with pytest.raises(u.BackendError, match="dask-backed Quantity"):
        require_jax_backend("test_fn", q)


def test_require_jax_rejects_bare_dask_array():
    da = pytest.importorskip("dask.array")
    arr = da.from_array(np.array([1.0]), chunks=1)
    with pytest.raises(u.BackendError, match="dask"):
        require_jax_backend("test_fn", arr)


def test_require_jax_raises_for_ndonnx_quantity():
    ndonnx = pytest.importorskip("ndonnx")
    q = u.Quantity(ndonnx.asarray(np.array([1.0])), unit=u.meter)
    with pytest.raises(u.BackendError, match="ndonnx-backed Quantity"):
        require_jax_backend("test_fn", q)


def test_require_jax_rejects_bare_ndonnx_array():
    ndonnx = pytest.importorskip("ndonnx")
    arr = ndonnx.asarray(np.array([1.0]))
    with pytest.raises(u.BackendError, match="ndonnx"):
        require_jax_backend("test_fn", arr)


def test_jax_only_passes_for_jax_quantity():
    @jax_only
    def f(x):
        return x

    q = u.Quantity(jnp.array([1.0]), unit=u.meter)
    assert f(q) is q


def test_jax_only_raises_for_numpy_quantity():
    @jax_only
    def f(x):
        return x

    q = u.Quantity(np.array([1.0]), unit=u.meter)
    with pytest.raises(u.BackendError, match="requires the jax backend"):
        f(q)


def test_jax_only_preserves_metadata():
    @jax_only
    def my_func(x):
        """my docstring"""
        return x

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "my docstring"
