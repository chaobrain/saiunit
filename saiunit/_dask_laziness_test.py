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

"""Lazy-safety tests for the dask backend.

These tests construct a dask Quantity from a source array that *counts*
materializations. Any saiunit operation that triggers compute() will bump
the counter; the assertions catch unintended materialization.
"""

import numpy as np
import pytest


class _ComputeCounter:
    """Wraps a numpy array; counts every time it's read."""

    def __init__(self, arr):
        self._arr = arr
        self.reads = 0

    def __getitem__(self, idx):
        self.reads += 1
        return self._arr[idx]

    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def ndim(self):
        return self._arr.ndim


@pytest.fixture
def dask_quantity():
    da = pytest.importorskip("dask.array")
    import saiunit as u
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    counter = _ComputeCounter(arr)
    darr = da.from_array(counter, chunks=2)
    q = u.Quantity(darr, unit=u.meter)
    # Dask's own metadata probe in from_array() and saiunit's Quantity
    # construction may read once or twice for setup. We measure laziness
    # of operations *after* construction, so reset the counter here.
    counter.reads = 0
    return q, counter


def test_dask_quantity_shape_does_not_compute(dask_quantity):
    q, counter = dask_quantity
    _ = q.shape
    assert counter.reads == 0, f"q.shape triggered {counter.reads} reads"


def test_dask_quantity_repr_does_not_compute(dask_quantity):
    q, counter = dask_quantity
    s = repr(q)
    assert counter.reads == 0, f"repr(q) triggered {counter.reads} reads"
    assert "dask" in s.lower() or "Quantity" in s


def test_dask_quantity_addition_does_not_compute(dask_quantity):
    q, counter = dask_quantity
    _ = q + q
    assert counter.reads == 0, f"q + q triggered {counter.reads} reads"


def test_dask_quantity_compute_does_materialize(dask_quantity):
    q, counter = dask_quantity
    _ = q.mantissa.compute()
    assert counter.reads > 0, "explicit .compute() did not actually materialize"
