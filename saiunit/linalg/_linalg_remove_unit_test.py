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

"""Tests for saiunit.linalg._linalg_remove_unit (cond, matrix_rank, slogdet)."""

import jax.numpy as jnp
import numpy.testing as npt
from absl.testing import parameterized

import saiunit as u
from saiunit import meter, second
from saiunit._base_getters import assert_quantity


class Array(u.CustomArray):
    def __init__(self, value):
        self.data = value


# ---------------------------------------------------------------------------
# cond
# ---------------------------------------------------------------------------

class TestCond(parameterized.TestCase):
    @parameterized.product(
        unit=[meter, second],
    )
    def test_cond_with_quantity(self, unit):
        x = jnp.array([[1., 2.], [2., 1.]]) * unit
        result = u.linalg.cond(x)
        expected = jnp.linalg.cond(jnp.array([[1., 2.], [2., 1.]]))
        npt.assert_allclose(float(result), float(expected), rtol=1e-5)

    def test_cond_without_unit(self):
        x = jnp.array([[1., 2.], [2., 1.]])
        result = u.linalg.cond(x)
        expected = jnp.linalg.cond(x)
        npt.assert_allclose(float(result), float(expected), rtol=1e-5)

    def test_cond_ill_conditioned(self):
        x = jnp.array([[1., 2.], [0., 0.]]) * meter
        result = u.linalg.cond(x)
        assert jnp.isinf(result)

    def test_cond_with_p(self):
        x = jnp.array([[1., 2.], [3., 4.]]) * meter
        result = u.linalg.cond(x, p=1)
        expected = jnp.linalg.cond(jnp.array([[1., 2.], [3., 4.]]), p=1)
        npt.assert_allclose(float(result), float(expected), rtol=1e-5)

    def test_cond_custom_array(self):
        x = jnp.array([[1., 2.], [2., 1.]]) * meter
        arr = Array(x)
        result = u.linalg.cond(arr.data)
        expected = jnp.linalg.cond(jnp.array([[1., 2.], [2., 1.]]))
        npt.assert_allclose(float(result), float(expected), rtol=1e-5)


# ---------------------------------------------------------------------------
# matrix_rank
# ---------------------------------------------------------------------------

class TestMatrixRank(parameterized.TestCase):
    @parameterized.product(
        unit=[meter, second],
    )
    def test_matrix_rank_full_rank(self, unit):
        a = jnp.array([[1., 2.], [3., 4.]]) * unit
        result = u.linalg.matrix_rank(a)
        assert int(result) == 2

    def test_matrix_rank_deficient(self):
        a = jnp.array([[1., 0.], [0., 0.]]) * meter
        result = u.linalg.matrix_rank(a)
        assert int(result) == 1

    def test_matrix_rank_without_unit(self):
        a = jnp.array([[1., 2.], [3., 4.]])
        result = u.linalg.matrix_rank(a)
        expected = jnp.linalg.matrix_rank(a)
        assert int(result) == int(expected)

    def test_matrix_rank_zero_matrix(self):
        a = jnp.zeros((3, 3)) * meter
        result = u.linalg.matrix_rank(a)
        assert int(result) == 0

    def test_matrix_rank_custom_array(self):
        a = jnp.array([[1., 2.], [3., 4.]]) * meter
        arr = Array(a)
        result = u.linalg.matrix_rank(arr.data)
        assert int(result) == 2


# ---------------------------------------------------------------------------
# slogdet
# ---------------------------------------------------------------------------

class TestSlogdet(parameterized.TestCase):
    @parameterized.product(
        unit=[meter, second],
    )
    def test_slogdet_with_quantity(self, unit):
        a = jnp.array([[1., 2.], [3., 4.]]) * unit
        sign, logabsdet = u.linalg.slogdet(a)
        exp_sign, exp_logabsdet = jnp.linalg.slogdet(
            jnp.array([[1., 2.], [3., 4.]])
        )
        npt.assert_allclose(float(sign), float(exp_sign), rtol=1e-5)
        npt.assert_allclose(float(logabsdet), float(exp_logabsdet), rtol=1e-5)

    def test_slogdet_without_unit(self):
        a = jnp.array([[1., 2.], [3., 4.]])
        sign, logabsdet = u.linalg.slogdet(a)
        exp_sign, exp_logabsdet = jnp.linalg.slogdet(a)
        npt.assert_allclose(float(sign), float(exp_sign), rtol=1e-5)
        npt.assert_allclose(float(logabsdet), float(exp_logabsdet), rtol=1e-5)

    def test_slogdet_positive_det(self):
        a = jnp.array([[2., 0.], [0., 3.]]) * meter
        sign, logabsdet = u.linalg.slogdet(a)
        npt.assert_allclose(float(sign), 1., rtol=1e-5)
        npt.assert_allclose(float(jnp.exp(logabsdet)), 6., rtol=1e-5)

    def test_slogdet_custom_array(self):
        a = jnp.array([[1., 2.], [3., 4.]]) * meter
        arr = Array(a)
        sign, logabsdet = u.linalg.slogdet(arr.data)
        exp_sign, exp_logabsdet = jnp.linalg.slogdet(
            jnp.array([[1., 2.], [3., 4.]])
        )
        npt.assert_allclose(float(sign), float(exp_sign), rtol=1e-5)
        npt.assert_allclose(float(logabsdet), float(exp_logabsdet), rtol=1e-5)


# --- Docstring example tests ---


def test_docstring_example_cond():
    """Reproduce the cond docstring example: well-conditioned matrix."""
    x = jnp.array([[1., 2.],
                    [2., 1.]]) * u.meter
    result = u.linalg.cond(x)
    npt.assert_allclose(float(result), 3., rtol=1e-5)


def test_docstring_example_cond_ill_conditioned():
    """Reproduce the cond docstring example: ill-conditioned matrix."""
    x = jnp.array([[1., 2.],
                    [0., 0.]]) * u.meter
    result = u.linalg.cond(x)
    assert jnp.isinf(result)


def test_docstring_example_matrix_rank():
    """Reproduce the matrix_rank docstring example: full-rank matrix."""
    a = jnp.array([[1., 2.],
                    [3., 4.]]) * u.meter
    result = u.linalg.matrix_rank(a)
    assert int(result) == 2


def test_docstring_example_matrix_rank_deficient():
    """Reproduce the matrix_rank docstring example: rank-deficient."""
    b = jnp.array([[1., 0.],
                    [0., 0.]]) * u.meter
    result = u.linalg.matrix_rank(b)
    assert int(result) == 1


def test_docstring_example_slogdet():
    """Reproduce the slogdet docstring example."""
    a = jnp.array([[1., 2.],
                    [3., 4.]]) * u.meter
    sign, logabsdet = u.linalg.slogdet(a)
    npt.assert_allclose(float(sign), -1., rtol=1e-5)
    npt.assert_allclose(float(jnp.exp(logabsdet)), 2., rtol=1e-5)
