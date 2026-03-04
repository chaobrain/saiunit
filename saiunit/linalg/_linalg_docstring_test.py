# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Docstring example tests for key linalg functions.

Each test mirrors the ``.. code-block:: python`` example shown in the
corresponding docstring so that we catch any drift between docs and
behaviour.
"""

import jax.numpy as jnp
import numpy.testing as npt

import saiunit as su


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------

class TestMatmulDocstring:
    def test_matmul_basic(self):
        """Reproduce the matmul docstring example."""
        a = jnp.array([[1., 2.], [3., 4.]]) * su.meter
        b = jnp.array([[5., 6.], [7., 8.]]) * su.second
        result = su.linalg.matmul(a, b)
        expected = jnp.array([[19., 22.], [43., 50.]])
        npt.assert_allclose(result.mantissa, expected, rtol=1e-5)
        assert result.unit == su.meter * su.second

    def test_matmul_unitless(self):
        """matmul on plain arrays returns a plain array."""
        a = jnp.array([[1., 2.], [3., 4.]])
        b = jnp.array([[5., 6.], [7., 8.]])
        result = su.linalg.matmul(a, b)
        expected = jnp.matmul(a, b)
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_matmul_mixed_units(self):
        """matmul with one Quantity and one plain array."""
        a = jnp.array([[1., 0.], [0., 1.]]) * su.meter
        b = jnp.array([[3.], [4.]])
        result = su.linalg.matmul(a, b)
        expected = jnp.array([[3.], [4.]])
        npt.assert_allclose(result.mantissa, expected, rtol=1e-5)
        assert result.unit == su.meter


# ---------------------------------------------------------------------------
# dot
# ---------------------------------------------------------------------------

class TestDotDocstring:
    def test_dot_1d(self):
        """Reproduce the dot docstring example."""
        a = jnp.array([1., 2., 3.]) * su.meter
        b = jnp.array([4., 5., 6.]) * su.second
        result = su.linalg.dot(a, b)
        npt.assert_allclose(float(result.mantissa), 32., rtol=1e-5)
        assert result.unit == su.meter * su.second

    def test_dot_2d(self):
        """dot on 2-D arrays behaves like matmul."""
        a = jnp.array([[1., 2.], [3., 4.]]) * su.meter
        b = jnp.array([[5., 6.], [7., 8.]]) * su.second
        result = su.linalg.dot(a, b)
        expected = jnp.dot(jnp.array([[1., 2.], [3., 4.]]),
                           jnp.array([[5., 6.], [7., 8.]]))
        npt.assert_allclose(result.mantissa, expected, rtol=1e-5)
        assert result.unit == su.meter * su.second


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------

class TestNormDocstring:
    def test_norm_vector_l2(self):
        """Reproduce the norm docstring example (vector 2-norm)."""
        x = jnp.array([3., 4., 12.]) * su.meter
        result = su.linalg.norm(x)
        npt.assert_allclose(float(result.mantissa), 13., rtol=1e-5)
        assert result.unit == su.meter

    def test_norm_vector_l1(self):
        """L1 vector norm preserves unit."""
        x = jnp.array([3., 4., 12.]) * su.meter
        result = su.linalg.norm(x, ord=1)
        npt.assert_allclose(float(result.mantissa), 19., rtol=1e-5)
        assert result.unit == su.meter

    def test_norm_matrix_frobenius(self):
        """Frobenius matrix norm preserves unit."""
        m = jnp.array([[1., 2., 3.],
                        [4., 5., 7.]]) * su.meter
        result = su.linalg.norm(m)
        expected = jnp.linalg.norm(jnp.array([[1., 2., 3.], [4., 5., 7.]]))
        npt.assert_allclose(float(result.mantissa), float(expected), rtol=1e-5)
        assert result.unit == su.meter


# ---------------------------------------------------------------------------
# det
# ---------------------------------------------------------------------------

class TestDetDocstring:
    def test_det_2x2(self):
        """Reproduce the det docstring example."""
        a = jnp.array([[1., 2.],
                        [3., 4.]]) * su.meter
        result = su.linalg.det(a)
        npt.assert_allclose(float(result.mantissa), -2., rtol=1e-5)
        assert result.unit == su.meter ** 2

    def test_det_3x3(self):
        """det of 3x3 carries unit**3."""
        a = jnp.array([[1., 2., 3.],
                        [4., 5., 6.],
                        [7., 8., 10.]]) * su.second
        result = su.linalg.det(a)
        expected = jnp.linalg.det(jnp.array([[1., 2., 3.],
                                              [4., 5., 6.],
                                              [7., 8., 10.]]))
        npt.assert_allclose(float(result.mantissa), float(expected), rtol=1e-4)
        assert result.unit == su.second ** 3


# ---------------------------------------------------------------------------
# inv
# ---------------------------------------------------------------------------

class TestInvDocstring:
    def test_inv_unit(self):
        """Reproduce the inv docstring example -- unit check."""
        a = jnp.array([[1., 2., 3.],
                        [2., 4., 2.],
                        [3., 2., 1.]]) * su.second
        a_inv = su.linalg.inv(a)
        assert a_inv.unit == su.second ** -1

    def test_inv_identity_reconstruction(self):
        """A @ inv(A) should give the identity matrix."""
        a = jnp.array([[2., 1.],
                        [1., 3.]]) * su.meter
        a_inv = su.linalg.inv(a)
        product = su.linalg.matmul(a, a_inv)
        # meter * meter^-1 is dimensionless, so result is a plain array
        npt.assert_allclose(product, jnp.eye(2), atol=1e-5)

    def test_inv_unitless(self):
        """inv on a plain array returns a plain array."""
        a = jnp.array([[2., 1.], [1., 3.]])
        result = su.linalg.inv(a)
        expected = jnp.linalg.inv(a)
        npt.assert_allclose(result, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------

class TestSolveDocstring:
    def test_solve_basic(self):
        """Reproduce the solve docstring example."""
        A = jnp.array([[1., 2., 3.],
                        [2., 4., 2.],
                        [3., 2., 1.]]) * su.meter
        b = jnp.array([14., 16., 10.]) * su.second
        x = su.linalg.solve(A, b)
        npt.assert_allclose(x.mantissa, jnp.array([1., 2., 3.]), rtol=1e-4)
        assert x.unit == su.second / su.meter

    def test_solve_reconstructs_rhs(self):
        """A @ solve(A, b) should give back b."""
        A = jnp.array([[1., 2.],
                        [3., 4.]]) * su.meter
        b = jnp.array([5., 6.]) * su.second
        x = su.linalg.solve(A, b)
        reconstructed = su.linalg.matmul(A, x)
        npt.assert_allclose(reconstructed.mantissa, b.mantissa, rtol=1e-4)
        assert reconstructed.unit == b.unit

    def test_solve_unitless(self):
        """solve on plain arrays returns a plain array."""
        A = jnp.array([[1., 2.], [3., 4.]])
        b = jnp.array([5., 6.])
        result = su.linalg.solve(A, b)
        expected = jnp.linalg.solve(A, b)
        npt.assert_allclose(result, expected, rtol=1e-5)
