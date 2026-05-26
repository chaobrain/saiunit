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
# jobs (see conftest._scan_jax_only_test_files); importing saiunit.autograd
# pulls in JAX-only modules, so this module must not be collected without JAX.
import jax  # noqa: F401
import pytest

from saiunit.autograd._misc import (
    _ensure_index,
    _argnums_partial,
    _check_callable,
    _isgeneratorfunction,
)


def test_ensure_index_scalar():
    assert _ensure_index(2) == 2


def test_ensure_index_sequence():
    assert _ensure_index([1, 2, 3]) == (1, 2, 3)


def test_argnums_partial_single_arg():
    def f(a, b, c):
        return a + b + c

    argnums, partial_fun, dyn = _argnums_partial(f, 1, (10, 20, 30), {})
    assert argnums == 1
    assert dyn == (20,)
    # Re-supply only the differentiated arg; statics are baked in.
    assert partial_fun(99) == 10 + 99 + 30


def test_argnums_partial_negative_index_normalized():
    def f(a, b):
        return a - b

    argnums, partial_fun, dyn = _argnums_partial(f, -1, (5, 2), {})
    assert argnums == 1
    assert dyn == (2,)
    assert partial_fun(2) == 3


def test_argnums_partial_out_of_bounds():
    def f(a):
        return a

    with pytest.raises(ValueError, match="out of bounds"):
        _argnums_partial(f, 3, (1,), {})


def test_check_callable_accepts_function():
    def f(x):
        return x

    _check_callable(f)  # no raise


def test_check_callable_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        _check_callable(42)


def test_check_callable_rejects_staticmethod():
    with pytest.raises(TypeError, match="staticmethod"):
        _check_callable(staticmethod(lambda x: x))


def test_check_callable_rejects_generator_function():
    def gen(x):
        yield x

    with pytest.raises(TypeError, match="generator"):
        _check_callable(gen)


def test_isgeneratorfunction():
    def gen(x):
        yield x

    def regular(x):
        return x

    assert _isgeneratorfunction(gen) is True
    assert _isgeneratorfunction(regular) is False
