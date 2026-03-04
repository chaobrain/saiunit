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


from __future__ import annotations

import unittest

import brainstate as bst
import jax

import saiunit as u


class TestCOO(unittest.TestCase):
    def test_matvec(self):
        for ux, uy in [
            (u.ms, u.mV),
            (u.UNITLESS, u.UNITLESS),
            (u.mV, u.UNITLESS),
            (u.UNITLESS, u.mV),
        ]:
            data = bst.random.rand(10, 20)
            data = data * (data < 0.3) * ux

            coo = u.sparse.COO.fromdense(data)

            x = bst.random.random((10,)) * uy
            self.assertTrue(
                u.math.allclose(
                    x @ data,
                    x @ coo
                )
            )

            x = bst.random.random((20,)) * uy
            self.assertTrue(
                u.math.allclose(
                    data @ x,
                    coo @ x
                )
            )

    def test_matmul(self):
        for ux, uy in [
            (u.ms, u.mV),
            (u.UNITLESS, u.UNITLESS),
            (u.mV, u.UNITLESS),
            (u.UNITLESS, u.mV),
        ]:
            data = bst.random.rand(10, 20)
            data = data * (data < 0.3) * ux
            coo = u.sparse.COO.fromdense(data)

            data2 = bst.random.rand(20, 30) * uy

            self.assertTrue(
                u.math.allclose(
                    data @ data2,
                    coo @ data2
                )
            )

            data2 = bst.random.rand(30, 10) * uy
            self.assertTrue(
                u.math.allclose(
                    data2 @ data,
                    data2 @ coo
                )
            )

    def test_pos(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data = bst.random.rand(10, 20)
            data = data * (data < 0.3) * ux

            coo = u.sparse.COO.fromdense(data)

            self.assertTrue(
                u.math.allclose(
                    coo.__pos__().data,
                    coo.data
                )
            )

    def test_neg(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data = bst.random.rand(10, 20)
            data = data * (data < 0.3) * ux

            coo = u.sparse.COO.fromdense(data)

            self.assertTrue(
                u.math.allclose(
                    (-coo).data,
                    -coo.data
                )
            )

    def test_abs(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data = bst.random.rand(10, 20)
            data = data * (data < 0.3) * ux

            coo = u.sparse.COO.fromdense(data)

            self.assertTrue(
                u.math.allclose(
                    abs(coo).data,
                    abs(coo.data)
                )
            )

    def test_with_data_preserves_sorted_flags(self):
        coo = u.sparse.COO._eye(4, 4, 0)
        self.assertTrue(coo._rows_sorted)
        self.assertTrue(coo._cols_sorted)

        coo_new = coo.with_data(coo.data)
        self.assertTrue(coo_new._rows_sorted)
        self.assertTrue(coo_new._cols_sorted)

    def test_add(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.rand(10, 20)
            data1 = data1 * (data1 < 0.3) * ux

            data2 = 2. * ux

            coo1 = u.sparse.COO.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (coo1 + data2).data,
                    coo1.data + data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 + coo1).data,
                    data2 + coo1.data
                )
            )

    def test_sub(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.rand(10, 20)
            data1 = data1 * (data1 < 0.3) * ux

            data2 = 2. * ux

            coo1 = u.sparse.COO.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (coo1 - data2).data,
                    coo1.data - data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 - coo1).data,
                    data2 - coo1.data
                )
            )

    def test_mul(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.rand(10, 20)
            data1 = data1 * (data1 < 0.3) * ux

            data2 = 2. * ux

            coo1 = u.sparse.COO.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (coo1 * data2).data,
                    coo1.data * data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 * coo1).data,
                    data2 * coo1.data
                )
            )

    def test_div(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.rand(10, 20)
            data1 = data1 * (data1 < 0.3) * ux

            data2 = 2. * u.ohm

            coo1 = u.sparse.COO.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (coo1 / data2).data,
                    coo1.data / data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 / coo1).data,
                    data2 / coo1.data
                )
            )

    def test_mod(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.rand(10, 20)
            data1 = data1 * (data1 < 0.3) * ux

            data2 = 2. * ux

            coo1 = u.sparse.COO.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (coo1 % data2).data,
                    coo1.data % data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 % coo1).data,
                    data2 % coo1.data
                )
            )

    def test_grad(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.randn(10, 20) * ux
            sp = u.sparse.COO.fromdense(data1)

            def f(data, x):
                return u.get_mantissa((sp.with_data(data) @ x).sum())

            xs = bst.random.randn(20)

            grads = jax.grad(f)(sp.data, xs)

    def test_grad2(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.randn(10, 20) * ux
            sp = u.sparse.CSR.fromdense(data1)

            def f(sp, x):
                return u.get_mantissa((sp @ x).sum())

            xs = bst.random.randn(20)

            grads = jax.grad(f)(sp, xs)

            sp = sp + grads * 1e-3
            sp = sp + 1e-3 * grads

    def test_jit(self):
        @jax.jit
        def f(sp, x):
            return sp @ x

        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.randn(10, 20) * ux
            sp = u.sparse.CSR.fromdense(data1)

            xs = bst.random.randn(20)
            ys = f(sp, xs)


class TestCOODocstringExamples(unittest.TestCase):
    """Tests verifying the docstring examples for COO and related functions."""

    def test_coo_fromdense_basic(self):
        """Verify COO.fromdense round-trips through todense."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[1., 0., 2.], [0., 0., 3.]])
        coo = susparse.COO.fromdense(dense)
        self.assertEqual(coo.shape, (2, 3))
        self.assertTrue(jnp.allclose(coo.todense(), dense))

    def test_coo_fromdense_function(self):
        """Verify coo_fromdense function round-trips through todense."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[1., 0., 0.], [0., 2., 3.]])
        coo = susparse.coo_fromdense(dense)
        self.assertEqual(coo.shape, (2, 3))
        self.assertTrue(jnp.allclose(coo.todense(), dense))

    def test_coo_with_data(self):
        """Verify COO.with_data replaces values but preserves structure."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[1., 0.], [0., 2.]])
        coo = susparse.COO.fromdense(dense)
        new_coo = coo.with_data(coo.data * 5)
        expected = jnp.array([[5., 0.], [0., 10.]])
        self.assertTrue(jnp.allclose(new_coo.todense(), expected))

    def test_coo_todense_function(self):
        """Verify the coo_todense standalone function."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[5., 0.], [0., 6.]])
        coo = susparse.coo_fromdense(dense)
        result = susparse.coo_todense(coo)
        self.assertTrue(jnp.allclose(result, dense))

    def test_coo_isinstance_sparse_matrix(self):
        """Verify COO is a SparseMatrix instance."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[1., 0.], [0., 2.]])
        coo = susparse.COO.fromdense(dense)
        self.assertIsInstance(coo, susparse.SparseMatrix)
