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

import brainstate as bst
import jax.numpy as jnp
import numpy as np
from scipy.special import exprel

from saiunit import math


def test_exprel():
    np.printoptions(precision=30)

    print()
    with bst.environ.context(precision=64):
        # Test with float64 input
        x = jnp.array([0.0, 1e-17, 1e-16, 1e-15, 1e-12, 1e-9, 1.0, 10.0, 100.0, 717.0, 718.0], dtype=jnp.float64)
        print(math.exprel(x), '\n', exprel(np.asarray(x)))
        assert jnp.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-6)

    with bst.environ.context(precision=32):
        # Test with float32 input
        x = jnp.array([0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1.0, 10.0, 100.0], dtype=jnp.float32)
        print(math.exprel(x), '\n', exprel(np.asarray(x)))
        assert jnp.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-6)

    # Test with float16 input
    x = jnp.array([0.0, 1e-5, 1e-4, 1e-3, 1.0, 10.0], dtype=jnp.float16)
    print(math.exprel(x), '\n', exprel(np.asarray(x)))
    assert np.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-03, atol=1e-05)

    # # Test with float8 input
    # x = jnp.array([0.0, 1e-5, 1e-4, 1e-3, 1.0, ], dtype=jnp.float8_e5m2fnuz)
    # print(math.exprel(x), '\n', exprel(np.asarray(x)))
    # assert np.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-03, atol=1e-05)

    # Test with int input
    x = jnp.array([0., 1., 10.])
    print(math.exprel(x), '\n', exprel(np.asarray(x)))
    assert jnp.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-6)

    with bst.environ.context(precision=64):
        # Test with negative input
        x = jnp.array([-1.0, -10.0, -100.0], dtype=jnp.float64)
        print(math.exprel(x), '\n', exprel(np.asarray(x)))
        assert jnp.allclose(math.exprel(x), exprel(np.asarray(x)), rtol=1e-6)
