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


class TestCSR(unittest.TestCase):
    def test_matvec(self):
        for ux, uy in [
            (u.ms, u.mV),
            (u.UNITLESS, u.UNITLESS),
            (u.mV, u.UNITLESS),
            (u.UNITLESS, u.mV),
        ]:
            data = bst.random.rand(10, 20)
            data = data * (data < 0.3) * ux

            csr = u.sparse.CSR.fromdense(data)

            x = bst.random.random((10,)) * uy
            self.assertTrue(
                u.math.allclose(
                    x @ data,
                    x @ csr
                )
            )

            x = bst.random.random((20,)) * uy
            self.assertTrue(
                u.math.allclose(
                    data @ x,
                    csr @ x
                )
            )

    def test_matvec_non_unit(self):
        data = bst.random.rand(10, 20)
        data = data * (data < 0.3)

        csr = u.sparse.CSR.fromdense(data)

        x = bst.random.random((10,))

        self.assertTrue(
            u.math.allclose(
                x @ data,
                x @ csr
            )
        )

        x = bst.random.random((20,))
        self.assertTrue(
            u.math.allclose(
                data @ x,
                csr @ x
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
            csr = u.sparse.CSR.fromdense(data)

            data2 = bst.random.rand(20, 30) * uy

            self.assertTrue(
                u.math.allclose(
                    data @ data2,
                    csr @ data2
                )
            )

            data2 = bst.random.rand(30, 10) * uy
            self.assertTrue(
                u.math.allclose(
                    data2 @ data,
                    data2 @ csr
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

            csr = u.sparse.CSR.fromdense(data)

            self.assertTrue(
                u.math.allclose(
                    csr.__pos__().data,
                    csr.data
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

            csr = u.sparse.CSR.fromdense(data)

            self.assertTrue(
                u.math.allclose(
                    (-csr).data,
                    -csr.data
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

            csr = u.sparse.CSR.fromdense(data)

            self.assertTrue(
                u.math.allclose(
                    abs(csr).data,
                    abs(csr.data)
                )
            )

    def test_add(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.rand(10, 20)
            data1 = data1 * (data1 < 0.3) * ux

            data2 = 2. * ux

            csr1 = u.sparse.CSR.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (csr1 + data2).data,
                    csr1.data + data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 + csr1).data,
                    data2 + csr1.data
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

            csr1 = u.sparse.CSR.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (csr1 - data2).data,
                    csr1.data - data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 - csr1).data,
                    data2 - csr1.data
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

            csr1 = u.sparse.CSR.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (csr1 * data2).data,
                    csr1.data * data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 * csr1).data,
                    data2 * csr1.data
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

            csr1 = u.sparse.CSR.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (csr1 / data2).data,
                    csr1.data / data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 / csr1).data,
                    data2 / csr1.data
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

            csr1 = u.sparse.CSR.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (csr1 % data2).data,
                    csr1.data % data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 % csr1).data,
                    data2 % csr1.data
                )
            )

    def test_grad(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.randn(10, 20) * ux
            csr = u.sparse.CSR.fromdense(data1)

            def f(csr_data, x):
                return u.get_mantissa((csr.with_data(csr_data) @ x).sum())

            xs = bst.random.randn(20)

            grads = jax.grad(f)(csr.data, xs)

    def test_grad2(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.randn(10, 20) * ux
            csr = u.sparse.CSR.fromdense(data1)

            def f(csr, x):
                return u.get_mantissa((csr @ x).sum())

            xs = bst.random.randn(20)

            grads = jax.grad(f)(csr, xs)

            csr = csr + grads * 1e-3
            csr = csr + 1e-3 * grads

    def test_jit(self):
        @jax.jit
        def f(csr, x):
            return csr @ x

        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.randn(10, 20) * ux
            csr = u.sparse.CSR.fromdense(data1)

            xs = bst.random.randn(20)
            ys = f(csr, xs)


class TestCSC(unittest.TestCase):
    def test_fromdense_with_nse(self):
        data = bst.random.rand(10, 20)
        data = data * (data < 0.3) * u.ms

        nse = int((u.get_mantissa(data) != 0).sum())
        csc = u.sparse.csc_fromdense(data, nse=nse)
        self.assertTrue(u.math.allclose(csc.todense(), data))

    def test_matvec(self):
        for ux, uy in [
            (u.ms, u.mV),
            (u.UNITLESS, u.UNITLESS),
            (u.mV, u.UNITLESS),
            (u.UNITLESS, u.mV),
        ]:
            data = bst.random.rand(10, 20)
            data = data * (data < 0.3) * ux

            csc = u.sparse.CSC.fromdense(data)

            x = bst.random.random((20,)) * uy
            self.assertTrue(
                u.math.allclose(
                    data @ x,
                    csc @ x
                )
            )

            x = bst.random.random((10,)) * uy
            self.assertTrue(
                u.math.allclose(
                    x @ data,
                    x @ csc
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
            csr = u.sparse.CSC.fromdense(data)

            data2 = bst.random.rand(20, 30) * uy

            self.assertTrue(
                u.math.allclose(
                    data @ data2,
                    csr @ data2
                )
            )

            data2 = bst.random.rand(30, 10) * uy
            self.assertTrue(
                u.math.allclose(
                    data2 @ data,
                    data2 @ csr
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

            csc = u.sparse.CSC.fromdense(data)

            self.assertTrue(
                u.math.allclose(
                    csc.__pos__().data,
                    csc.data
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

            csc = u.sparse.CSC.fromdense(data)

            self.assertTrue(
                u.math.allclose(
                    (-csc).data,
                    -csc.data
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

            csc = u.sparse.CSC.fromdense(data)

            self.assertTrue(
                u.math.allclose(
                    abs(csc).data,
                    abs(csc.data)
                )
            )

    def test_add(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.rand(10, 20)
            data1 = data1 * (data1 < 0.3) * ux

            data2 = 2. * ux

            csc1 = u.sparse.CSC.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (csc1 + data2).data,
                    csc1.data + data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 + csc1).data,
                    data2 + csc1.data
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

            csc1 = u.sparse.CSC.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (csc1 - data2).data,
                    csc1.data - data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 - csc1).data,
                    data2 - csc1.data
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

            csc1 = u.sparse.CSC.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (csc1 * data2).data,
                    csc1.data * data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 * csc1).data,
                    data2 * csc1.data
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

            csc1 = u.sparse.CSC.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (csc1 / data2).data,
                    csc1.data / data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 / csc1).data,
                    data2 / csc1.data
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

            csc1 = u.sparse.CSC.fromdense(data1)

            self.assertTrue(
                u.math.allclose(
                    (csc1 % data2).data,
                    csc1.data % data2
                )
            )

            self.assertTrue(
                u.math.allclose(
                    (data2 % csc1).data,
                    data2 % csc1.data
                )
            )

    def test_grad(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.randn(10, 20) * ux
            csc = u.sparse.CSC.fromdense(data1)

            def f(data, x):
                return u.get_mantissa((csc.with_data(data) @ x).sum())

            xs = bst.random.randn(20)

            grads = jax.grad(f)(csc.data, xs)

    def test_grad2(self):
        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.randn(10, 20) * ux
            csc = u.sparse.CSC.fromdense(data1)

            def f(csc, x):
                return u.get_mantissa((csc @ x).sum())

            xs = bst.random.randn(20)

            grads = jax.grad(f)(csc, xs)

            csc = csc + grads * 1e-3
            csc = csc + 1e-3 * grads

    def test_jit(self):

        @jax.jit
        def f(csc, x):
            return csc @ x

        for ux in [
            u.ms,
            u.UNITLESS,
            u.mV,
        ]:
            data1 = bst.random.randn(10, 20) * ux
            csc = u.sparse.CSC.fromdense(data1)

            xs = bst.random.randn(20)
            ys = f(csc, xs)


class TestCSRDocstringExamples(unittest.TestCase):
    """Tests verifying the docstring examples for CSR and related functions."""

    def test_csr_fromdense_basic(self):
        """Verify CSR.fromdense round-trips through todense."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[1., 0., 2.], [0., 0., 3.]])
        csr = susparse.CSR.fromdense(dense)
        self.assertEqual(csr.shape, (2, 3))
        self.assertTrue(jnp.allclose(csr.todense(), dense))

    def test_csr_fromdense_function(self):
        """Verify csr_fromdense function round-trips through todense."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[1., 0., 0.], [0., 2., 3.]])
        csr = susparse.csr_fromdense(dense)
        self.assertEqual(csr.shape, (2, 3))
        self.assertTrue(jnp.allclose(csr.todense(), dense))

    def test_csr_with_data(self):
        """Verify CSR.with_data replaces values but preserves structure."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[1., 0.], [0., 2.]])
        csr = susparse.CSR.fromdense(dense)
        new_csr = csr.with_data(csr.data * 5)
        expected = jnp.array([[5., 0.], [0., 10.]])
        self.assertTrue(jnp.allclose(new_csr.todense(), expected))

    def test_csr_todense_function(self):
        """Verify the csr_todense standalone function."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[5., 0.], [0., 6.]])
        csr = susparse.csr_fromdense(dense)
        result = susparse.csr_todense(csr)
        self.assertTrue(jnp.allclose(result, dense))

    def test_csr_isinstance_sparse_matrix(self):
        """Verify CSR is a SparseMatrix instance."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[1., 0.], [0., 2.]])
        csr = susparse.CSR.fromdense(dense)
        self.assertIsInstance(csr, susparse.SparseMatrix)


class TestCSCDocstringExamples(unittest.TestCase):
    """Tests verifying the docstring examples for CSC and related functions."""

    def test_csc_fromdense_basic(self):
        """Verify CSC.fromdense round-trips through todense."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[1., 0., 2.], [0., 0., 3.]])
        csc = susparse.CSC.fromdense(dense)
        self.assertEqual(csc.shape, (2, 3))
        self.assertTrue(jnp.allclose(csc.todense(), dense))

    def test_csc_fromdense_function(self):
        """Verify csc_fromdense function round-trips through todense."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[1., 0., 0.], [0., 2., 3.]])
        csc = susparse.csc_fromdense(dense)
        self.assertEqual(csc.shape, (2, 3))
        self.assertTrue(jnp.allclose(csc.todense(), dense))

    def test_csc_with_data(self):
        """Verify CSC.with_data replaces values but preserves structure."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[1., 0.], [0., 2.]])
        csc = susparse.CSC.fromdense(dense)
        new_csc = csc.with_data(csc.data * 5)
        expected = jnp.array([[5., 0.], [0., 10.]])
        self.assertTrue(jnp.allclose(new_csc.todense(), expected))

    def test_csc_todense_function(self):
        """Verify the csc_todense standalone function."""
        import jax.numpy as jnp
        import saiunit as u
        import saiunit.sparse as susparse

        dense = jnp.array([[5., 0.], [0., 6.]])
        csc = susparse.csc_fromdense(dense)
        result = susparse.csc_todense(csc)
        self.assertTrue(jnp.allclose(result, dense))


def test_csr_fromdense_raises_on_numpy_quantity():
    import numpy as np
    import pytest
    import saiunit as u
    from saiunit import meter
    q = u.Quantity(np.array([[1.0, 0.0], [0.0, 2.0]]), unit=meter)
    with pytest.raises(u.BackendError, match="requires the jax backend"):
        u.sparse.CSR.fromdense(q)


def test_csr_fromdense_function_raises_on_numpy_quantity():
    import numpy as np
    import pytest
    import saiunit as u
    from saiunit import meter
    q = u.Quantity(np.array([[1.0, 0.0], [0.0, 2.0]]), unit=meter)
    with pytest.raises(u.BackendError, match="requires the jax backend"):
        u.sparse.csr_fromdense(q)


def test_csc_fromdense_raises_on_numpy_quantity():
    import numpy as np
    import pytest
    import saiunit as u
    from saiunit import meter
    q = u.Quantity(np.array([[1.0, 0.0], [0.0, 2.0]]), unit=meter)
    with pytest.raises(u.BackendError, match="requires the jax backend"):
        u.sparse.CSC.fromdense(q)
    with pytest.raises(u.BackendError, match="requires the jax backend"):
        u.sparse.csc_fromdense(q)


def test_csr_jit_repeated_calls():
    import jax.numpy as jnp

    @jax.jit
    def f(sp, x):
        return sp @ x

    x = jnp.ones(3)
    dense1 = jnp.array([[1., 0., 2.], [0., 3., 0.]]) * u.mV
    dense2 = jnp.array([[4., 0., 5.], [0., 6., 0.]]) * u.mV
    dense3 = jnp.array([[0., 7., 0.], [8., 0., 9.]]) * u.mV

    csr1 = u.sparse.CSR.fromdense(dense1)
    # Same unit and same sparsity pattern, different values.
    csr2 = u.sparse.CSR.fromdense(dense2)
    # Different sparsity pattern.
    csr3 = u.sparse.CSR.fromdense(dense3)

    assert u.math.allclose(f(csr1, x), dense1 @ x)
    assert u.math.allclose(f(csr2, x), dense2 @ x)  # crashed before fix
    assert u.math.allclose(f(csr1, x), dense1 @ x)  # cache hit
    assert u.math.allclose(f(csr3, x), dense3 @ x)  # recompile


def test_csc_jit_repeated_calls():
    import jax.numpy as jnp

    @jax.jit
    def f(sp, x):
        return sp @ x

    x = jnp.ones(3)
    dense1 = jnp.array([[1., 0., 2.], [0., 3., 0.]]) * u.mV
    dense2 = jnp.array([[4., 0., 5.], [0., 6., 0.]]) * u.mV
    dense3 = jnp.array([[0., 7., 0.], [8., 0., 9.]]) * u.mV

    csc1 = u.sparse.CSC.fromdense(dense1)
    # Same unit and same sparsity pattern, different values.
    csc2 = u.sparse.CSC.fromdense(dense2)
    # Different sparsity pattern.
    csc3 = u.sparse.CSC.fromdense(dense3)

    assert u.math.allclose(f(csc1, x), dense1 @ x)
    assert u.math.allclose(f(csc2, x), dense2 @ x)  # crashed before fix
    assert u.math.allclose(f(csc1, x), dense1 @ x)  # cache hit
    assert u.math.allclose(f(csc3, x), dense3 @ x)  # recompile


def test_csr_binary_op_with_1x1_operand():
    import jax.numpy as jnp

    dense = jnp.array([[1., 0., 2.], [0., 3., 0.]])
    csr = u.sparse.CSR.fromdense(dense)

    out = csr * jnp.array([[3.0]])
    assert out.data.ndim == 1
    assert u.math.allclose(out.todense(), dense * 3.0)

    out = jnp.array([[3.0]]) * csr
    assert out.data.ndim == 1
    assert u.math.allclose(out.todense(), dense * 3.0)

    out = csr * (jnp.array([[3.0]]) * u.mV)
    assert out.data.ndim == 1
    assert u.math.allclose(out.todense(), dense * (3.0 * u.mV))


def test_csc_binary_op_with_1x1_operand():
    import jax.numpy as jnp

    dense = jnp.array([[1., 0., 2.], [0., 3., 0.]])
    csc = u.sparse.CSC.fromdense(dense)

    out = csc * jnp.array([[3.0]])
    assert out.data.ndim == 1
    assert u.math.allclose(out.todense(), dense * 3.0)

    out = jnp.array([[3.0]]) * csc
    assert out.data.ndim == 1
    assert u.math.allclose(out.todense(), dense * 3.0)

    out = csc * (jnp.array([[3.0]]) * u.mV)
    assert out.data.ndim == 1
    assert u.math.allclose(out.todense(), dense * (3.0 * u.mV))


def test_csr_add_shape_mismatch_raises():
    import jax.numpy as jnp
    import pytest

    c1 = u.sparse.CSR.fromdense(jnp.array([[1., 2., 3.]]))
    c2 = u.sparse.CSR.fromdense(jnp.array([[1., 2., 3., 0.]]))
    with pytest.raises(ValueError, match="shape mismatch"):
        c1 + c2
    with pytest.raises(ValueError, match="shape mismatch"):
        c1 - c2


def test_csc_add_shape_mismatch_raises():
    import jax.numpy as jnp
    import pytest

    c1 = u.sparse.CSC.fromdense(jnp.array([[1.], [2.], [3.]]))
    c2 = u.sparse.CSC.fromdense(jnp.array([[1.], [2.], [3.], [0.]]))
    with pytest.raises(ValueError, match="shape mismatch"):
        c1 + c2
    with pytest.raises(ValueError, match="shape mismatch"):
        c1 - c2


def test_numpy_left_operands_defer_to_csr():
    import jax.numpy as jnp
    import numpy as np

    dense = jnp.array([[1., 0., 2.], [0., 3., 0.]])
    csr = u.sparse.CSR.fromdense(dense)

    expected = jnp.ones(2) @ csr
    out = np.ones(2) @ csr
    assert u.math.allclose(out, expected)

    out = np.float64(3.0) * csr
    assert isinstance(out, u.sparse.CSR)
    assert u.math.allclose(out.todense(), dense * 3.0)


def test_csr_block_until_ready_with_quantity():
    import jax.numpy as jnp

    dense = jnp.array([[1., 0.], [0., 2.]]) * u.mV
    csr = u.sparse.CSR.fromdense(dense)
    assert csr.block_until_ready() is csr
