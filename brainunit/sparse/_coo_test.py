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
