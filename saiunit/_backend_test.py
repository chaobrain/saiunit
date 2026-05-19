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
from saiunit._backend import (
    get_backend,
    get_default_backend,
    is_jax_array,
    is_numpy_array,
    set_default_backend,
    to_backend,
    using_backend,
)


def test_is_numpy_array_true_for_ndarray():
    assert is_numpy_array(np.array([1.0])) is True
    assert is_numpy_array(jnp.array([1.0])) is False
    assert is_numpy_array(1.0) is False


def test_is_jax_array_true_for_jax_array():
    assert is_jax_array(jnp.array([1.0])) is True
    assert is_jax_array(np.array([1.0])) is False


def test_get_backend_numpy_only():
    xp = get_backend(np.array([1.0]))
    # numpy namespace must be array_api_compat.numpy (the wrapper)
    import array_api_compat.numpy as expected
    assert xp is expected
    # smoke: works as a namespace
    assert float(xp.sin(np.array([0.0]))[0]) == 0.0


def test_get_backend_jax_only():
    # In array_api_compat 1.14+, JAX is treated as already array-API-compatible:
    # array_namespace(jax_array) returns jax.numpy directly. We mirror that.
    import jax.numpy as expected
    xp = get_backend(jnp.array([1.0]))
    assert xp is expected


def test_get_backend_mixed_no_default_jax_wins():
    set_default_backend(None)
    import jax.numpy as expected
    xp = get_backend(np.array([1.0]), jnp.array([1.0]))
    assert xp is expected


def test_get_backend_mixed_with_numpy_default():
    set_default_backend("numpy")
    try:
        xp = get_backend(np.array([1.0]), jnp.array([1.0]))
        import array_api_compat.numpy as expected
        assert xp is expected
    finally:
        set_default_backend(None)


def test_set_default_backend_rejects_invalid():
    with pytest.raises(ValueError, match="must be 'numpy', 'jax', 'cupy', 'torch', 'dask', or None"):
        set_default_backend("notabackend")


def test_using_backend_context_manager():
    set_default_backend(None)
    assert get_default_backend() is None
    with using_backend("numpy"):
        assert get_default_backend() == "numpy"
    assert get_default_backend() is None


def test_using_backend_nested():
    set_default_backend(None)
    with using_backend("numpy"):
        with using_backend("jax"):
            assert get_default_backend() == "jax"
        assert get_default_backend() == "numpy"
    assert get_default_backend() is None


def test_to_backend_numpy_to_jax():
    arr = np.array([1.0, 2.0])
    out = to_backend(arr, "jax")
    assert is_jax_array(out)
    assert np.allclose(np.asarray(out), arr)


def test_to_backend_jax_to_numpy():
    arr = jnp.array([1.0, 2.0])
    out = to_backend(arr, "numpy")
    assert is_numpy_array(out)
    assert np.allclose(out, np.asarray(arr))


def test_to_backend_noop():
    arr = np.array([1.0])
    out = to_backend(arr, "numpy")
    assert out is arr  # no copy when already on target backend


def test_get_backend_scalar_falls_back_to_default():
    # No arrays at all — must use the default backend.
    set_default_backend("numpy")
    try:
        xp = get_backend(1.0, 2.0)
        import array_api_compat.numpy as expected
        assert xp is expected
    finally:
        set_default_backend(None)


def test_get_backend_scalar_no_default_uses_jax():
    set_default_backend(None)
    import jax.numpy as expected
    xp = get_backend(1.0)
    assert xp is expected


def test_try_import_returns_module_when_present():
    from saiunit._backend import _try_import
    np_mod = _try_import("numpy")
    assert np_mod is not None
    assert hasattr(np_mod, "asarray")


def test_try_import_returns_none_when_missing():
    from saiunit._backend import _try_import
    assert _try_import("definitely_not_a_real_package_xyz") is None


def test_try_import_is_cached():
    from saiunit._backend import _try_import
    a = _try_import("numpy")
    b = _try_import("numpy")
    assert a is b


def test_is_cupy_array_false_when_cupy_missing_or_not_cupy():
    from saiunit._backend import is_cupy_array
    # Non-cupy inputs always return False (works whether or not cupy is installed).
    assert is_cupy_array(np.array([1.0])) is False
    assert is_cupy_array(jnp.array([1.0])) is False
    assert is_cupy_array(1.0) is False


def test_is_cupy_array_true_when_available():
    cupy = pytest.importorskip("cupy")
    from saiunit._backend import is_cupy_array
    arr = cupy.array([1.0, 2.0])
    assert is_cupy_array(arr) is True


def test_is_torch_array_false_for_non_torch():
    from saiunit._backend import is_torch_array
    assert is_torch_array(np.array([1.0])) is False
    assert is_torch_array(jnp.array([1.0])) is False
    assert is_torch_array(1.0) is False


def test_is_torch_array_true_when_available():
    torch = pytest.importorskip("torch")
    from saiunit._backend import is_torch_array
    t = torch.tensor([1.0, 2.0])
    assert is_torch_array(t) is True


def test_backend_name_includes_cupy_and_torch():
    from saiunit._backend import BackendName
    import typing
    args = typing.get_args(BackendName)
    assert "cupy" in args
    assert "torch" in args
    assert "numpy" in args
    assert "jax" in args


def test_get_backend_cupy_only():
    cupy = pytest.importorskip("cupy")
    from saiunit._backend import get_backend
    import array_api_compat.cupy as expected
    xp = get_backend(cupy.array([1.0]))
    assert xp is expected


def test_get_backend_torch_only():
    torch = pytest.importorskip("torch")
    from saiunit._backend import get_backend
    import array_api_compat.torch as expected
    xp = get_backend(torch.tensor([1.0]))
    assert xp is expected


def test_get_backend_mixed_torch_jax_default_jax_wins():
    torch = pytest.importorskip("torch")
    from saiunit._backend import get_backend, set_default_backend
    set_default_backend(None)
    import jax.numpy as expected
    xp = get_backend(torch.tensor([1.0]), jnp.array([1.0]))
    assert xp is expected


def test_get_backend_mixed_with_torch_default():
    torch = pytest.importorskip("torch")
    from saiunit._backend import get_backend, set_default_backend
    set_default_backend("torch")
    try:
        import array_api_compat.torch as expected
        xp = get_backend(torch.tensor([1.0]), jnp.array([1.0]))
        assert xp is expected
    finally:
        set_default_backend(None)


def test_to_backend_numpy_to_cupy():
    cupy = pytest.importorskip("cupy")
    from saiunit._backend import to_backend, is_cupy_array
    arr = np.array([1.0, 2.0])
    out = to_backend(arr, "cupy")
    assert is_cupy_array(out)
    assert cupy.allclose(out, cupy.asarray(arr))


def test_to_backend_cupy_noop():
    cupy = pytest.importorskip("cupy")
    from saiunit._backend import to_backend
    arr = cupy.array([1.0])
    out = to_backend(arr, "cupy")
    assert out is arr


def test_to_backend_cupy_with_device_kwarg():
    cupy = pytest.importorskip("cupy")
    from saiunit._backend import to_backend, is_cupy_array
    arr = np.array([1.0])
    out = to_backend(arr, "cupy", device=0)
    assert is_cupy_array(out)


def test_to_backend_numpy_rejects_unknown_kwargs():
    from saiunit._backend import to_backend
    with pytest.raises(TypeError, match="does not accept"):
        to_backend(np.array([1.0]), "numpy", device="cuda")


def test_to_backend_numpy_to_torch():
    torch = pytest.importorskip("torch")
    from saiunit._backend import to_backend, is_torch_array
    arr = np.array([1.0, 2.0])
    out = to_backend(arr, "torch")
    assert is_torch_array(out)
    assert torch.allclose(out, torch.tensor([1.0, 2.0], dtype=out.dtype))


def test_to_backend_torch_noop():
    torch = pytest.importorskip("torch")
    from saiunit._backend import to_backend
    t = torch.tensor([1.0])
    out = to_backend(t, "torch")
    assert out is t


def test_to_backend_torch_with_dtype_torch_native():
    torch = pytest.importorskip("torch")
    from saiunit._backend import to_backend
    out = to_backend(np.array([1.0, 2.0]), "torch", dtype=torch.float64)
    assert out.dtype == torch.float64


def test_to_backend_torch_with_dtype_numpy_mapped():
    torch = pytest.importorskip("torch")
    from saiunit._backend import to_backend
    out = to_backend(np.array([1.0, 2.0]), "torch", dtype=np.float64)
    assert out.dtype == torch.float64


def test_to_backend_torch_rejects_unknown_kwarg():
    pytest.importorskip("torch")
    from saiunit._backend import to_backend
    with pytest.raises(TypeError, match="does not accept"):
        to_backend(np.array([1.0]), "torch", chunks="auto")


def test_set_default_backend_accepts_cupy():
    from saiunit._backend import set_default_backend, get_default_backend
    set_default_backend("cupy")
    try:
        assert get_default_backend() == "cupy"
    finally:
        set_default_backend(None)


def test_set_default_backend_accepts_torch():
    from saiunit._backend import set_default_backend, get_default_backend
    set_default_backend("torch")
    try:
        assert get_default_backend() == "torch"
    finally:
        set_default_backend(None)


def test_using_backend_accepts_cupy():
    from saiunit._backend import using_backend, get_default_backend
    with using_backend("cupy"):
        assert get_default_backend() == "cupy"


def test_using_backend_accepts_torch():
    from saiunit._backend import using_backend, get_default_backend
    with using_backend("torch"):
        assert get_default_backend() == "torch"


def test_top_level_exports_new_detectors():
    import saiunit as u
    assert hasattr(u, "is_cupy_array")
    assert hasattr(u, "is_torch_array")
    assert "is_cupy_array" in u.__all__
    assert "is_torch_array" in u.__all__


def test_is_dask_array_false_for_non_dask():
    from saiunit._backend import is_dask_array
    assert is_dask_array(np.array([1.0])) is False
    assert is_dask_array(jnp.array([1.0])) is False
    assert is_dask_array(1.0) is False


def test_is_dask_array_true_when_available():
    da = pytest.importorskip("dask.array")
    from saiunit._backend import is_dask_array
    arr = da.from_array(np.array([1.0, 2.0]), chunks=1)
    assert is_dask_array(arr) is True


def test_backend_name_includes_dask():
    from saiunit._backend import BackendName
    import typing
    assert "dask" in typing.get_args(BackendName)


def test_get_backend_dask_only():
    da = pytest.importorskip("dask.array")
    from saiunit._backend import get_backend
    import array_api_compat.dask.array as expected
    xp = get_backend(da.from_array(np.array([1.0]), chunks=1))
    assert xp is expected


def test_get_backend_dask_default_for_mixed():
    da = pytest.importorskip("dask.array")
    from saiunit._backend import get_backend, set_default_backend
    set_default_backend("dask")
    try:
        import array_api_compat.dask.array as expected
        xp = get_backend(da.from_array(np.array([1.0]), chunks=1), jnp.array([1.0]))
        assert xp is expected
    finally:
        set_default_backend(None)


def test_to_backend_numpy_to_dask_default_chunks():
    da = pytest.importorskip("dask.array")
    from saiunit._backend import to_backend, is_dask_array
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    out = to_backend(arr, "dask")
    assert is_dask_array(out)
    assert tuple(out.compute()) == (1.0, 2.0, 3.0, 4.0)


def test_to_backend_numpy_to_dask_custom_chunks():
    da = pytest.importorskip("dask.array")
    from saiunit._backend import to_backend
    arr = np.arange(8, dtype=np.float64)
    out = to_backend(arr, "dask", chunks=2)
    assert out.numblocks == (4,)


def test_to_backend_dask_noop():
    da = pytest.importorskip("dask.array")
    from saiunit._backend import to_backend
    arr = da.from_array(np.array([1.0]), chunks=1)
    out = to_backend(arr, "dask")
    assert out is arr


def test_to_backend_dask_rejects_unknown_kwarg():
    pytest.importorskip("dask.array")
    from saiunit._backend import to_backend
    with pytest.raises(TypeError, match="does not accept"):
        to_backend(np.array([1.0]), "dask", device="cuda")


def test_set_default_backend_accepts_dask():
    from saiunit._backend import set_default_backend, get_default_backend
    set_default_backend("dask")
    try:
        assert get_default_backend() == "dask"
    finally:
        set_default_backend(None)


def test_using_backend_accepts_dask():
    from saiunit._backend import using_backend, get_default_backend
    with using_backend("dask"):
        assert get_default_backend() == "dask"


def test_top_level_exports_is_dask_array():
    import saiunit as u
    assert hasattr(u, "is_dask_array")
    assert "is_dask_array" in u.__all__


def test_is_ndonnx_array_false_for_non_ndonnx():
    from saiunit._backend import is_ndonnx_array
    assert is_ndonnx_array(np.array([1.0])) is False
    assert is_ndonnx_array(jnp.array([1.0])) is False
    assert is_ndonnx_array(1.0) is False


def test_is_ndonnx_array_true_when_available():
    ndonnx = pytest.importorskip("ndonnx")
    from saiunit._backend import is_ndonnx_array
    arr = ndonnx.asarray(np.array([1.0, 2.0]))
    assert is_ndonnx_array(arr) is True


def test_backend_name_includes_ndonnx():
    from saiunit._backend import BackendName
    import typing
    assert "ndonnx" in typing.get_args(BackendName)


def test_get_backend_ndonnx_only():
    ndonnx = pytest.importorskip("ndonnx")
    from saiunit._backend import get_backend
    xp = get_backend(ndonnx.asarray(np.array([1.0])))
    assert xp is ndonnx


def test_to_backend_numpy_to_ndonnx():
    ndonnx = pytest.importorskip("ndonnx")
    from saiunit._backend import to_backend, is_ndonnx_array
    arr = np.array([1.0, 2.0])
    out = to_backend(arr, "ndonnx")
    assert is_ndonnx_array(out)


def test_to_backend_ndonnx_noop():
    ndonnx = pytest.importorskip("ndonnx")
    from saiunit._backend import to_backend
    arr = ndonnx.asarray(np.array([1.0]))
    out = to_backend(arr, "ndonnx")
    assert out is arr


def test_to_backend_ndonnx_rejects_kwargs():
    pytest.importorskip("ndonnx")
    from saiunit._backend import to_backend
    with pytest.raises(TypeError, match="does not accept"):
        to_backend(np.array([1.0]), "ndonnx", device="cuda")
