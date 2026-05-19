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
    else:
        import jax
        assert isinstance(r, jax.Array)


def test_concatenate_respects_backend(backend):
    a = u.Quantity([1.0, 2.0], unit=meter)
    b = u.Quantity([3.0, 4.0], unit=meter)
    r = u.math.concatenate([a, b])
    assert r.backend == backend
