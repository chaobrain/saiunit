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
    with pytest.raises(ValueError, match="must be 'numpy', 'jax', or None"):
        set_default_backend("torch")


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
