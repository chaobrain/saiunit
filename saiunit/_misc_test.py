# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import numpy as np

import saiunit as u


class MyArray(u.CustomArray):
    def __init__(self, value):
        self.data = value


# --- Docstring example tests ---


def test_docstring_example_maybe_custom_array():
    # Plain values pass through unchanged
    assert u._misc.maybe_custom_array(5) == 5
    assert u._misc.maybe_custom_array(3.14) == 3.14
    assert u._misc.maybe_custom_array("hello") == "hello"

    # CustomArray instances are unwrapped to their .data
    arr = MyArray(np.array([1, 2, 3]))
    result = u._misc.maybe_custom_array(arr)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    # Scalar CustomArray
    scalar_arr = MyArray(42)
    assert u._misc.maybe_custom_array(scalar_arr) == 42


def test_docstring_example_maybe_custom_array_tree():
    # Nested structure with CustomArray instances
    tree = [MyArray(np.array([1, 2])), 3, {'k': MyArray(np.array([4]))}]
    result = u._misc.maybe_custom_array_tree(tree)

    np.testing.assert_array_equal(result[0], np.array([1, 2]))
    assert result[1] == 3
    np.testing.assert_array_equal(result[2]['k'], np.array([4]))

    # Tuple structure
    tree_tuple = (MyArray(np.array([10])), 20)
    result_tuple = u._misc.maybe_custom_array_tree(tree_tuple)
    np.testing.assert_array_equal(result_tuple[0], np.array([10]))
    assert result_tuple[1] == 20

    # Plain tree without any CustomArray should pass through unchanged
    plain_tree = [1, 2, {'a': 3}]
    result_plain = u._misc.maybe_custom_array_tree(plain_tree)
    assert result_plain[0] == 1
    assert result_plain[1] == 2
    assert result_plain[2]['a'] == 3
