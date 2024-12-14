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

from __future__ import annotations

import unittest

import brainstate as bst
import jax

import brainunit as u


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
