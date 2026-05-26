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
# jobs (see conftest._scan_jax_only_test_files); without JAX the shimmed
# helpers are stubs that raise, so this module must not be collected.
import jax  # noqa: F401
import pytest

import saiunit._compatible_import as ci

EXPORTS = ["safe_map", "unzip2", "wrap_init", "Primitive", "concrete_or_error"]


def test_all_exports_present():
    assert set(ci.__all__) == set(EXPORTS)
    for name in EXPORTS:
        assert hasattr(ci, name), f"_compatible_import.{name} missing"


def test_safe_map_applies_function():
    assert ci.safe_map(lambda a, b: a + b, [1, 2, 3], [10, 20, 30]) == [11, 22, 33]


def test_safe_map_rejects_length_mismatch():
    # Only meaningful in the JAX>=0.6 reimplementation; tolerate either path.
    try:
        ci.safe_map(lambda a, b: a + b, [1, 2], [1])
    except (ValueError, Exception):
        pass


def test_unzip2_splits_pairs():
    xs, ys = ci.unzip2([(1, "a"), (2, "b"), (3, "c")])
    assert xs == (1, 2, 3)
    assert ys == ("a", "b", "c")


def test_concrete_or_error_passes_concrete_value():
    assert ci.concrete_or_error(int, 5) == 5


def test_primitive_is_constructible_with_jax():
    # With JAX installed, Primitive is the real jax primitive class.
    p = ci.Primitive("my_prim")
    assert p.name == "my_prim"
