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

import jax.numpy as jnp

import saiunit as u
from saiunit.sparse._block_csr import sample_sparse_matrix as sample_block_csr
from saiunit.sparse._block_ell import BlockELL


class TestBlockSparse(unittest.TestCase):
    def test_block_csr_can_instantiate_and_todense(self):
        mat = sample_block_csr(8, 8, 2, 2, sparse_prob=0.3)
        dense = mat.todense()
        self.assertEqual(dense.shape, (8, 8))

    def test_block_ell_fromdense_handles_ragged_rows(self):
        dense = jnp.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ], dtype=jnp.float32)

        mat = BlockELL.fromdense(dense, block_size=(2, 2))
        self.assertEqual(mat.indices.shape, (2, 1, 2))
        self.assertTrue(jnp.array_equal(mat.blocks_per_row, jnp.array([1, 0], dtype=jnp.int32)))
        self.assertTrue(u.math.allclose(mat.todense(), dense))

    def test_block_ell_round_trip_with_units(self):
        dense = jnp.array([
            [0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ], dtype=jnp.float32) * u.mV

        mat = BlockELL.fromdense(dense, block_size=(2, 2))
        self.assertTrue(u.math.allclose(mat.todense(), dense))
