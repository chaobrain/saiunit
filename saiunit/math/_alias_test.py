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

import saiunit as u
import saiunit.math as bm
from saiunit.math import _alias as alias_mod

GETTER_NAMES = [
    'is_dimensionless', 'is_unitless', 'get_dim', 'get_unit', 'get_mantissa',
    'get_magnitude', 'display_in_unit', 'maybe_decimal',
    'fail_for_dimension_mismatch', 'fail_for_unit_mismatch', 'assert_quantity',
    'get_or_create_dimension',
]
DECORATOR_NAMES = ['check_dims', 'check_units']


def test_alias_all_matches_expected():
    assert set(alias_mod.__all__) == set(GETTER_NAMES + DECORATOR_NAMES)


def test_aliases_exposed_on_math_namespace():
    for name in GETTER_NAMES + DECORATOR_NAMES:
        assert hasattr(bm, name), f"saiunit.math.{name} missing"


def test_getter_aliases_are_same_object_as_top_level():
    # The re-exported getters are the exact objects on the top-level package.
    for name in GETTER_NAMES:
        if hasattr(u, name):
            assert getattr(bm, name) is getattr(u, name), f"math.{name} != u.{name}"
