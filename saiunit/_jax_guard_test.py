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
from saiunit._jax_guard import require_jax_backend


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
