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

import jax.numpy as jnp

import saiunit as u
from saiunit._sparse_base import SparseMatrix, _same_sparsity_pattern


def test_sparsematrix_is_exported():
    assert SparseMatrix.__name__ == "SparseMatrix"
    assert "SparseMatrix" in __import__("saiunit._sparse_base", fromlist=["__all__"]).__all__


def test_same_sparsity_pattern_identity():
    a = jnp.array([0, 1, 2])
    assert _same_sparsity_pattern(a, a) is True


def test_same_sparsity_pattern_equal_values():
    a = jnp.array([0, 1, 2])
    b = jnp.array([0, 1, 2])
    assert _same_sparsity_pattern(a, b) is True


def test_same_sparsity_pattern_different_shape():
    a = jnp.array([0, 1, 2])
    b = jnp.array([0, 1])
    assert _same_sparsity_pattern(a, b) is False


def test_concrete_subclasses_are_sparsematrix():
    # COO / CSR subclass the shared base.
    assert issubclass(u.sparse.COO, SparseMatrix)
    assert issubclass(u.sparse.CSR, SparseMatrix)
