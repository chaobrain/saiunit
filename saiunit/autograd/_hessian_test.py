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

import jax
import numpy as np

import saiunit as u


class TestHessianFunction(unittest.TestCase):
    def test_hessian_scalar_function(self):
        def scalar_function(x):
            return x ** 2 + 3 * x + 2

        hess = u.autograd.hessian(scalar_function)
        x = np.array(1.0)
        expected_hessian = np.array([[2.0]])
        np.testing.assert_array_almost_equal(hess(x), expected_hessian)

    def test_hessian_scalar_function_with_unit(self):
        unit = u.ms

        def scalar_function(x):
            return x ** 2 + 3 * x * unit + 2 * unit * unit

        hess = u.autograd.hessian(scalar_function)
        x = np.array(1.0) * unit
        res = hess(x)
        expected_hessian = np.array([[2.0]])
        np.testing.assert_array_almost_equal(res, expected_hessian)

    def test_hessian_scalar_function_with_unit2(self):
        unit = u.ms

        def scalar_function(x):
            return x ** 3 + 3 * x * unit * unit + 2 * unit * unit * unit

        hess = u.autograd.hessian(scalar_function)
        x = np.array(1.0) * unit
        res = hess(x)
        expected_hessian = np.array([[6.0]]) * unit
        assert u.math.allclose(res, expected_hessian)

    def test_hessian_vector_function(self):
        def vector_function(x):
            return np.sum(x ** 2)

        hess = u.autograd.hessian(vector_function)
        x = np.array([1.0, 2.0])
        expected_hessian = np.array([[2.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(hess(x), expected_hessian)

    def test_hessian_with_aux(self):
        unit = u.ms

        def function_with_aux(x):
            return x ** 2, x + u.math.ones_like(x)

        hess = u.autograd.hessian(function_with_aux, has_aux=True)
        x = np.array(1.0)
        expected_hessian = np.array([[2.0]])
        result, aux = hess(x)
        np.testing.assert_array_almost_equal(result, expected_hessian)
        np.testing.assert_array_almost_equal(aux, np.array(2.0))

        x = np.array(1.0) * unit
        result, aux = hess(x)
        expected_hessian = np.array([[2.0]])
        assert u.math.allclose(result, expected_hessian)
        assert u.math.allclose(aux, np.array(2.0) * unit)

    def test_hessian_multiple_arguments(self):
        def multi_arg_function(x, y):
            return x ** 2 + y ** 2

        hess = u.autograd.hessian(multi_arg_function, argnums=(0, 1))
        x = np.array(1.0)
        y = np.array(2.0)
        expected_hessian = np.array([[2.0, 0.0], [0.0, 2.0]])
        res = hess(x, y)
        np.testing.assert_array_almost_equal(res, expected_hessian)

        x = np.array(1.0) * u.ms
        y = np.array(2.0) * u.ms
        res = hess(x, y)
        assert u.math.allclose(u.math.asarray(res), expected_hessian)

    def test_hessian_dict(self):
        def dict_function(x):
            return {'z': x['a'] ** 3 + x['b'] ** 3}

        unit = u.ms
        x = {'a': np.array(1.0) * unit, 'b': np.array(2.0) * unit}
        res = u.autograd.hessian(dict_function)(x)

        expected_hessian = {'z': {'a': {'a': 6.0 * unit, 'b': 0.0 * unit},
                                  'b': {'a': 0.0 * unit, 'b': 12.0 * unit}}}

        def check(a, b):
            assert u.math.allclose(a, b)

        jax.tree.map(check, res, expected_hessian, is_leaf=u.math.is_quantity)


if __name__ == '__main__':
    unittest.main()
