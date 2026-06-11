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

import brainstate as bst  # type: ignore[import-untyped]
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


def test_coo_jit_repeated_calls():
    import jax.numpy as jnp

    @jax.jit
    def f(sp, x):
        return sp @ x

    x = jnp.ones(3)
    dense1 = jnp.array([[1., 0., 2.], [0., 3., 0.]]) * u.mV
    dense2 = jnp.array([[4., 0., 5.], [0., 6., 0.]]) * u.mV
    dense3 = jnp.array([[0., 7., 0.], [8., 0., 9.]]) * u.mV

    coo1 = u.sparse.COO.fromdense(dense1)
    # Same unit and same sparsity pattern, different values.
    coo2 = u.sparse.COO.fromdense(dense2)
    # Different sparsity pattern.
    coo3 = u.sparse.COO.fromdense(dense3)

    assert u.math.allclose(f(coo1, x), dense1 @ x)
    assert u.math.allclose(f(coo2, x), dense2 @ x)  # crashed before fix
    assert u.math.allclose(f(coo1, x), dense1 @ x)  # cache hit
    assert u.math.allclose(f(coo3, x), dense3 @ x)  # recompile


def test_coo_binary_op_with_1x1_operand():
    import jax.numpy as jnp

    dense = jnp.array([[1., 0., 2.], [0., 3., 0.]])
    coo = u.sparse.COO.fromdense(dense)

    out = coo * jnp.array([[3.0]])
    assert out.data.ndim == 1
    assert u.math.allclose(out.todense(), dense * 3.0)

    out = jnp.array([[3.0]]) * coo
    assert out.data.ndim == 1
    assert u.math.allclose(out.todense(), dense * 3.0)

    out = coo * (jnp.array([[3.0]]) * u.mV)
    assert out.data.ndim == 1
    assert u.math.allclose(out.todense(), dense * (3.0 * u.mV))


def test_coo_add_shape_mismatch_raises():
    import jax.numpy as jnp
    import pytest

    c1 = u.sparse.COO.fromdense(jnp.array([[1., 2., 3.]]))
    c2 = u.sparse.COO.fromdense(jnp.array([[1., 2., 3., 0.]]))
    with pytest.raises(ValueError, match="shape mismatch"):
        c1 + c2
    with pytest.raises(ValueError, match="shape mismatch"):
        c1 - c2


def test_numpy_left_operands_defer_to_coo():
    import jax.numpy as jnp
    import numpy as np

    dense = jnp.array([[1., 0., 2.], [0., 3., 0.]])
    coo = u.sparse.COO.fromdense(dense)

    expected = jnp.ones(2) @ coo
    out = np.ones(2) @ coo
    assert u.math.allclose(out, expected)

    out = np.float64(3.0) * coo
    assert isinstance(out, u.sparse.COO)
    assert u.math.allclose(out.todense(), dense * 3.0)


def test_coo_block_until_ready_with_quantity():
    import jax.numpy as jnp

    dense = jnp.array([[1., 0.], [0., 2.]]) * u.mV
    coo = u.sparse.COO.fromdense(dense)
    assert coo.block_until_ready() is coo


def test_coo_fromdense_raises_on_numpy_quantity():
    import numpy as np
    import pytest
    from saiunit import meter

    q = u.Quantity(np.array([[1.0, 0.0], [0.0, 2.0]]), unit=meter)
    with pytest.raises(u.BackendError, match="requires the jax backend"):
        u.sparse.COO.fromdense(q)
    with pytest.raises(u.BackendError, match="requires the jax backend"):
        u.sparse.coo_fromdense(q)
