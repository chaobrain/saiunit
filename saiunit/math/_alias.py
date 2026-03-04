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


from saiunit._base_decorators import check_dims, check_units
from saiunit._base_dimension import get_or_create_dimension
from saiunit._base_getters import (
    assert_quantity,
    display_in_unit,
    fail_for_dimension_mismatch,
    fail_for_unit_mismatch,
    get_dim,
    get_magnitude,
    get_mantissa,
    get_unit,
    is_dimensionless,
    is_unitless,
    maybe_decimal,
)

__all__ = [
    'is_dimensionless',
    'is_unitless',
    'get_dim',
    'get_unit',
    'get_mantissa',
    'get_magnitude',
    'display_in_unit',
    'maybe_decimal',

    # functions for checking
    'check_dims',
    'check_units',
    'fail_for_dimension_mismatch',
    'fail_for_unit_mismatch',
    'assert_quantity',

    # advanced functions
    'get_or_create_dimension',
]
