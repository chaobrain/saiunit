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

# Top-level jax import flags this module for collect_ignore in the no-JAX CI
# jobs (see conftest._scan_jax_only_test_files); these smoke tests assume
# HAS_JAX is True and must not be collected when JAX is absent.
import jax  # noqa: F401
import numpy as np
import pytest

import saiunit._jax_compat as jc

EXPORTS = [
    "HAS_JAX", "jax", "jnp", "Tracer", "ShapedArray", "ShapeDtypeStruct",
    "DynamicJaxprTracer", "TracerArrayConversionError",
    "register_pytree_node_class", "ensure_compile_time_eval", "result_type",
    "canonicalize_dtype", "tree_map", "tree_structure", "tree_flatten", "tree",
    "device_put", "devices", "require_jax",
]


def test_all_exports_present():
    assert set(jc.__all__) == set(EXPORTS)
    for name in EXPORTS:
        assert hasattr(jc, name), f"_jax_compat.{name} missing"


def test_has_jax_true_in_test_env():
    # The default test environment installs JAX.
    assert jc.HAS_JAX is True
    assert jc.jax is not None
    assert jc.jnp is not None


def test_require_jax_noop_when_jax_present():
    # Should not raise when JAX is installed.
    jc.require_jax("anything")


def test_result_type_and_canonicalize_dtype():
    assert jc.result_type(np.int32, np.float32) == np.float32
    assert jc.canonicalize_dtype(np.float64) is not None


def test_tree_helpers_roundtrip():
    tree_obj = {"a": [1, 2], "b": 3}
    leaves, treedef = jc.tree_flatten(tree_obj)
    assert sorted(leaves) == [1, 2, 3]
    rebuilt = treedef.unflatten(leaves)
    assert rebuilt == tree_obj
    doubled = jc.tree_map(lambda x: x * 2, tree_obj)
    assert doubled == {"a": [2, 4], "b": 6}
    assert jc.tree_structure(tree_obj) == treedef


def test_register_pytree_node_class_returns_class():
    @jc.register_pytree_node_class
    class Dummy:
        def __init__(self, value):
            self.value = value

        def tree_flatten(self):
            return (self.value,), None

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)

    # The decorator returns the class, now registered as a pytree node.
    assert Dummy is not None
    leaves, treedef = jc.tree_flatten(Dummy(5))
    assert leaves == [5]
    assert treedef.unflatten(leaves).value == 5
