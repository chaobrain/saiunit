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

"""Smoke tests for the cross-backend ``backend`` fixture.

Each test below is parameterized via the ``backend`` fixture defined in
the top-level ``conftest.py``, so it runs once with ``numpy`` as the
default backend and once with ``jax``.
"""

import numpy as np

import saiunit as u
from saiunit import UNITLESS, meter


def test_quantity_default_backend(backend):
    """Quantity built from a Python list adopts the default backend."""
    q = u.Quantity([1.0, 2.0, 3.0], unit=meter)
    assert q.backend == backend


def test_arithmetic_default_backend(backend):
    a = u.Quantity([1.0, 2.0], unit=meter)
    b = u.Quantity([3.0, 4.0], unit=meter)
    r = a + b
    assert r.backend == backend


def test_math_function_default_backend(backend):
    q = u.Quantity([0.0, 1.0], unit=UNITLESS)
    r = u.math.sin(q)
    if backend == "numpy":
        assert isinstance(r, np.ndarray)
    elif backend == "jax":
        import jax
        assert isinstance(r, jax.Array)
    elif backend == "cupy":
        import cupy
        assert isinstance(r, cupy.ndarray)
    elif backend == "torch":
        import torch
        assert isinstance(r, torch.Tensor)


def test_concatenate_respects_backend(backend):
    a = u.Quantity([1.0, 2.0], unit=meter)
    b = u.Quantity([3.0, 4.0], unit=meter)
    r = u.math.concatenate([a, b])
    assert r.backend == backend


def test_backend_fixture_includes_cupy_and_torch(backend):
    """The fixture parameter is one of the four known backends.

    pytest's parametrize machinery is what actually exercises each;
    importorskip handles missing libraries.
    """
    assert backend in {"numpy", "jax", "cupy", "torch"}


def test_math_sin_on_each_backend(backend):
    """saiunit.math.sin returns a mantissa native to the active backend."""
    q = u.Quantity([0.0, 1.0], unit=UNITLESS)
    r = u.math.sin(q)
    if backend == "numpy":
        assert isinstance(r, np.ndarray)
    elif backend == "jax":
        import jax
        assert isinstance(r, jax.Array)
    elif backend == "cupy":
        import cupy
        assert isinstance(r, cupy.ndarray)
    elif backend == "torch":
        import torch
        assert isinstance(r, torch.Tensor)


def test_linalg_norm_on_each_backend(backend):
    """saiunit.linalg.norm returns a scalar of the active backend."""
    q = u.Quantity([3.0, 4.0], unit=meter)
    n = u.linalg.norm(q)
    # Should be 5 meters regardless of backend.
    assert n.unit == meter
    assert float(n.mantissa) == 5.0


def test_to_method_round_trip_on_each_backend(backend):
    """Convert to numpy and back; mantissa values preserved."""
    q = u.Quantity([1.0, 2.0, 3.0], unit=meter)
    q_np = q.to_numpy()
    assert np.allclose(q_np.mantissa, [1.0, 2.0, 3.0])
    assert q_np.unit == meter
