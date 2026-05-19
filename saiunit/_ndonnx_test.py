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

"""Symbolic-composition smoke tests for the ndonnx backend.

ndonnx arrays are symbolic — operations build an ONNX graph rather than
eagerly computing. These tests verify that saiunit dispatch routes
correctly through ndonnx, not that every saiunit.math function works
(ndonnx may not implement every op).
"""

import numpy as np
import pytest

ndonnx = pytest.importorskip("ndonnx")

import saiunit as u
from saiunit._backend import is_ndonnx_array


def test_ndonnx_quantity_preserves_symbolic_type():
    q = u.Quantity(ndonnx.asarray(np.array([1.0, 2.0, 3.0])), unit=u.meter)
    assert is_ndonnx_array(q.mantissa)
    assert q.backend == "ndonnx"


def test_ndonnx_arithmetic_stays_symbolic():
    q = u.Quantity(ndonnx.asarray(np.array([1.0, 2.0])), unit=u.meter)
    r = q + q
    assert is_ndonnx_array(r.mantissa)
    assert r.backend == "ndonnx"
    assert r.unit == u.meter


def test_ndonnx_math_sin_dispatches():
    q = u.Quantity(ndonnx.asarray(np.array([0.0, 1.0])), unit=u.UNITLESS)
    r = u.math.sin(q)
    assert is_ndonnx_array(r)


def test_ndonnx_unit_check_still_fires():
    """Dimensional analysis is independent of backend — the check fires
    on the units, not the mantissa type."""
    a = u.Quantity(ndonnx.asarray(np.array([1.0])), unit=u.meter)
    b = u.Quantity(ndonnx.asarray(np.array([1.0])), unit=u.second)
    with pytest.raises(u.UnitMismatchError):
        a + b
