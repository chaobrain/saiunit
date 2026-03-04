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

"""Re-export module for backward compatibility.

All implementation has been split into:
  - _base_dimension.py: Dimension class, DIMENSIONLESS, error classes
  - _base_unit.py: Unit class, UNITLESS, display helpers
  - _base_getters.py: getter/helper functions (get_dim, get_unit, etc.)
  - _base_quantity.py: Quantity class, wrapping functions
  - _base_decorators.py: check_dims, check_units, assign_units decorators
"""

from ._base_decorators import *
from ._base_dimension import *
from ._base_getters import *
from ._base_quantity import *

__all__ = [
    # three base objects
    'Dimension',
    'Unit',
    'Quantity',

    # errors
    'DimensionMismatchError',
    'UnitMismatchError',
    'DIMENSIONLESS',
    'UNITLESS',

    # helpers
    'is_dimensionless',
    'is_unitless',
    'get_dim',
    'get_unit',
    'get_mantissa',
    'get_magnitude',
    'display_in_unit',
    'split_mantissa_unit',
    'maybe_decimal',

    # functions for checking
    'check_dims',
    'check_units',
    'assign_units',
    'fail_for_dimension_mismatch',
    'fail_for_unit_mismatch',
    'assert_quantity',

    # advanced functions
    'get_or_create_dimension',
]
