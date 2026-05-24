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

"""Smoke tests that must pass with or without JAX installed.

These tests verify that the core Quantity/Unit machinery, NumPy backend, and
the lazy submodule gate all behave correctly when JAX is absent. They use
only the NumPy backend so they never trigger a JAX import.
"""

import numpy as np
import pytest

import saiunit as u
from saiunit._exceptions import BackendError
from saiunit._jax_compat import HAS_JAX


class TestCoreImports:
    def test_top_level_import_works(self):
        assert hasattr(u, "Quantity")
        assert hasattr(u, "Unit")
        assert hasattr(u, "meter")
        assert hasattr(u, "second")

    def test_eager_submodules_available(self):
        assert u.math is not None
        assert u.linalg is not None
        assert u.fft is not None
        assert u.constants is not None

    def test_has_jax_flag_consistent(self):
        try:
            import jax  # noqa: F401
            assert HAS_JAX
        except ImportError:
            assert not HAS_JAX


class TestQuantityWithNumpy:
    def test_scalar_construction(self):
        q = u.Quantity(3.0, u.meter)
        assert float(q.mantissa) == 3.0
        assert q.unit == u.meter

    def test_array_construction(self):
        q = np.array([1.0, 2.0, 3.0]) * u.second
        assert q.unit == u.second
        np.testing.assert_array_equal(np.asarray(q.mantissa), [1.0, 2.0, 3.0])

    def test_arithmetic_preserves_dimension(self):
        distance = 100.0 * u.meter
        time = 10.0 * u.second
        speed = distance / time
        assert speed.dim == (u.meter / u.second).dim

    def test_dimension_mismatch_raises(self):
        with pytest.raises((u.DimensionMismatchError, u.UnitMismatchError)):
            (1.0 * u.meter) + (1.0 * u.second)


class TestBackendSelection:
    def test_numpy_default_when_no_jax(self):
        if HAS_JAX:
            return
        # The per-backend CI jobs (e.g. test_pure_ndonnx) deliberately
        # override the default via SAIUNIT_DEFAULT_BACKEND. This test asks the
        # narrower question "does saiunit pick numpy when nothing else is
        # specified", so honour the env-var if the operator set one.
        import os
        if os.environ.get("SAIUNIT_DEFAULT_BACKEND") not in (None, "", "numpy"):
            pytest.skip(
                "SAIUNIT_DEFAULT_BACKEND is set; test_numpy_default_when_no_jax "
                "tests the auto-default, not the env-var override path"
            )
        assert u.get_default_backend() == "numpy"

    def test_using_numpy_backend_context(self):
        with u.using_backend("numpy"):
            q = u.Quantity(np.asarray(2.0), u.meter)
            assert u.is_numpy_array(q.mantissa)

    def test_jax_backend_raises_when_missing(self):
        if HAS_JAX:
            pytest.skip("JAX is installed; cannot test missing-JAX path")
        with pytest.raises(BackendError, match="(?i)jax"):
            u.set_default_backend("jax")


class TestJaxOnlySubmodulesGated:
    @pytest.mark.parametrize("name", ["autograd", "lax", "sparse"])
    def test_access_when_jax_missing_raises_backend_error(self, name):
        if HAS_JAX:
            pytest.skip("JAX is installed; gate does not trigger")
        with pytest.raises(BackendError, match="(?i)jax"):
            getattr(u, name)

    def test_exprel_raises_when_jax_missing(self):
        if HAS_JAX:
            pytest.skip("JAX is installed; gate does not trigger")
        with pytest.raises(BackendError, match="(?i)jax"):
            u.math.exprel(np.asarray([0.0, 0.5]))


class TestUnitArithmetic:
    def test_unit_multiplication(self):
        unit = u.meter / u.second
        assert unit.dim == (u.meter / u.second).dim

    def test_celsius_conversion(self):
        k = u.celsius2kelvin(0.0)
        assert abs(float(k.to_decimal(u.kelvin)) - 273.15) < 1e-9
