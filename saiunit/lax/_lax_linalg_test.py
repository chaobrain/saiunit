# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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
from absl.testing import parameterized
from jax import lax

import saiunit as u
import saiunit.lax as ulax
from saiunit._base import assert_quantity

lax_linear_algebra_change_unit_unary = [
    'cholesky',
]

lax_linear_algebra_keep_unit_unary_return_2 = [
    'eigh', 'hessenberg', 'qr',
]

lax_linear_algebra_keep_unit_unary_return_3 = [
    'eig', 'lu',
]

lax_linear_algebra_qdwh = [
    'qdwh',
]

lax_linear_algebra_schur = [
    'schur',
]

lax_linear_algebra_svd = [
    'svd',
]

lax_linear_algebra_tridiagonal = [
    'tridiagonal',
]

lax_linear_algebra_binary = [
    'householder_product', 'triangular_solve',
]

lax_linear_algebra_nary = [
    'tridiagonal_solve',
]


class TestLaxLinalg(parameterized.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestLaxLinalg, self).__init__(*args, **kwargs)

        print()

    def test_eig(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        w, vl, vr = ulax.eig(x)
        w_e, vl_e, vr_e = lax.linalg.eig(x)

        assert_quantity(w, w_e)
        assert_quantity(vl, vl_e)
        assert_quantity(vr, vr_e)

        x = x * u.second
        w, vl, vr = ulax.eig(x)
        assert_quantity(w, w_e, u.second)
        assert_quantity(vl, vl_e)
        assert_quantity(vr, vr_e)

    def test_cholesky(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        result = ulax.cholesky(x)
        expected = lax.linalg.cholesky(x)

        assert_quantity(result, expected)

        x = x * u.second
        result_q = ulax.cholesky(x)
        assert_quantity(result_q, expected, u.second ** 0.5)

    def test_eigh(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        v, w = ulax.eigh(x)
        v_e, w_e = lax.linalg.eigh(x)

        assert_quantity(w, w_e)
        assert_quantity(v, v_e)

        x = x * u.second
        v, w = ulax.eigh(x)
        assert_quantity(w, w_e, u.second)
        assert_quantity(v, v_e)

    def test_hessenberg(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        h, q = ulax.hessenberg(x)
        h_e, q_e = lax.linalg.hessenberg(x)

        assert_quantity(h, h_e)
        assert_quantity(q, q_e)

        x = x * u.second
        h, q = ulax.hessenberg(x)
        assert_quantity(h, h_e, u.second)
        assert_quantity(q, q_e)

    def test_qr(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        q, r = ulax.qr(x)
        q_e, r_e = lax.linalg.qr(x)

        assert_quantity(q, q_e)
        assert_quantity(r, r_e)

        x = x * u.second
        q, r = ulax.qr(x)
        assert_quantity(q, q_e)
        assert_quantity(r, r_e, u.second)

    def test_lu(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        p, l, _u = ulax.lu(x)
        p_e, l_e, _u_e = lax.linalg.lu(x)

        assert_quantity(p, p_e)
        assert_quantity(l, l_e)
        assert_quantity(_u, _u_e)

        x = x * u.second
        p, l, _u = ulax.lu(x)
        assert_quantity(p, p_e, u.second)
        assert_quantity(l, l_e)
        assert_quantity(_u, _u_e)

    def test_qdwh(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        _u, h, num_iters, is_converged = ulax.qdwh(x)
        _u_e, h_e, num_iters_e, is_converged_e = lax.linalg.qdwh(x)
        assert_quantity(_u, _u_e)
        assert_quantity(h, h_e)
        assert_quantity(num_iters, num_iters_e)
        assert_quantity(is_converged, is_converged_e)

        x = x * u.second
        _u, h, num_iters, is_converged = ulax.qdwh(x)
        assert_quantity(_u, _u_e)
        assert_quantity(h, h_e, u.second)
        assert_quantity(num_iters, num_iters_e)
        assert_quantity(is_converged, is_converged_e)

    def test_schur(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        t, q = ulax.schur(x)
        t_e, q_e = lax.linalg.schur(x)
        assert_quantity(t, t_e)
        assert_quantity(q, q_e)

        x = x * u.second
        t, q = ulax.schur(x)
        assert_quantity(t, t_e)
        assert_quantity(q, q_e, u.second)

    def test_svd(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        _u, s, vh = ulax.svd(x)
        _u_e, s_e, vh_e = lax.linalg.svd(x)
        assert_quantity(_u, _u_e)
        assert_quantity(s, s_e)
        assert_quantity(vh, vh_e)

        x = x * u.second
        _u, s, vh = ulax.svd(x)
        assert_quantity(_u, _u_e)
        assert_quantity(s, s_e, u.second)
        assert_quantity(vh, vh_e)

    def test_tridiagonal(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        arr, d, e, taus = ulax.tridiagonal(x)
        arr_e, d_e, e_e, taus_e = ulax.tridiagonal(x)
        assert_quantity(arr, arr_e)
        assert_quantity(d, d_e)
        assert_quantity(e, e_e)
        assert_quantity(taus, taus_e)

        x = x * u.second
        arr, d, e, taus = ulax.tridiagonal(x)
        assert_quantity(arr, arr_e, u.second)
        assert_quantity(d, d_e, u.second)
        assert_quantity(e, e_e, u.second)
        assert_quantity(taus, taus_e)

    def test_householder_product(self):
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        taus = jnp.array([1.0])

        result = ulax.householder_product(a, taus)
        expected = lax.linalg.householder_product(a, taus)
        assert_quantity(result, expected)

        a = a * u.second
        taus = taus * u.second
        result_q = ulax.householder_product(a, taus)
        assert_quantity(result_q, expected)

    def test_triangular_solve(self):
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        result = ulax.triangular_solve(a, b)
        expected = lax.linalg.triangular_solve(a, b)
        assert_quantity(result, expected)

        a = a * u.second
        result_q = ulax.triangular_solve(a, b)
        assert_quantity(result_q, expected)
        b = b * u.second
        result_q = ulax.triangular_solve(a, b)
        assert_quantity(result_q, expected, u.second)

    def test_tridiagonal_solve(self):
        dl = jnp.array([0.0, 1.0, 1.0])
        d = jnp.array([2.0, 2.0, 2.0])
        du = jnp.array([1.0, 1.0, 0.0])
        b = jnp.array([[1.0], [2.0], [3.0]])

        result = ulax.tridiagonal_solve(dl, d, du, b)
        expected = lax.linalg.tridiagonal_solve(dl, d, du, b)
        assert_quantity(result, expected)

        b = b * u.second
        result_q = ulax.tridiagonal_solve(dl, d, du, b)
        assert_quantity(result_q, expected, u.second)
