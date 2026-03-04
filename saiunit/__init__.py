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

"""
saiunit -- Physical units for JAX arrays.

``saiunit`` provides a :class:`Quantity` type that pairs a JAX array with a
physical :class:`Unit`, ensuring dimensional correctness at every arithmetic
operation.  It also supplies the standard SI base and derived units (e.g.
``meter``, ``second``, ``volt``), physical constants, and unit-aware wrappers
for NumPy/JAX math functions.

Subpackages
-----------
math
    Unit-aware wrappers for NumPy-style math functions.
lax
    Unit-aware wrappers for ``jax.lax`` primitives.
linalg
    Unit-aware linear-algebra routines.
fft
    Unit-aware FFT functions.
autograd
    Unit-aware automatic differentiation (grad, jacobian, hessian).
constants
    Physical constants as :class:`Quantity` objects.
sparse
    Unit-aware sparse matrix types (CSR, CSC, COO).

Examples
--------

.. code-block:: python

    >>> import saiunit as u
    >>> distance = 100.0 * u.meter
    >>> time = 9.58 * u.second
    >>> speed = distance / time
    >>> speed.dim == (u.meter / u.second).dim
    True
"""

from . import _matplotlib_compat
from . import autograd
from . import constants
from . import fft
from . import lax
from . import linalg
from . import math
from . import sparse
from ._base_decorators import assign_units, check_dims, check_units
from ._base_dimension import (
    DIMENSIONLESS,
    Dimension,
    DimensionMismatchError,
    UnitMismatchError,
    get_dim_for_display,
    get_or_create_dimension,
)
from ._base_getters import (
    array_with_unit,
    assert_quantity,
    display_in_unit,
    fail_for_dimension_mismatch,
    fail_for_unit_mismatch,
    get_dim,
    get_magnitude,
    get_mantissa,
    get_unit,
    has_same_unit,
    have_same_dim,
    is_dimensionless,
    is_scalar_type,
    is_unitless,
    maybe_decimal,
    split_mantissa_unit,
    unit_scale_align_to_first,
)
from ._base_quantity import Quantity, compatible_with_equinox
from ._base_unit import UNITLESS, Unit, add_standard_unit, parse_unit
from ._celsius import celsius2kelvin, kelvin2celsius
from ._misc import maybe_custom_array, maybe_custom_array_tree
from ._unit_common import *
from ._unit_common import __all__ as _common_all
from ._unit_constants import *
from ._unit_constants import __all__ as _constants_all
from ._unit_shortcuts import *
from ._unit_shortcuts import __all__ as _std_units_all
from ._version import __version__, __version_info__
from .custom_array import CustomArray

# old version compatibility
avogadro_constant = constants.avogadro
boltzmann_constant = constants.boltzmann
electric_constant = constants.electric
electron_mass = constants.electron_mass
elementary_charge = constants.elementary_charge
faraday_constant = constants.faraday
gas_constant = constants.gas
magnetic_constant = constants.magnetic
molar_mass_constant = constants.molar_mass

__all__ = [
              # version control
              '__version__',
              '__version_info__',

              # submodules
              'math',
              'linalg',
              'autograd',
              'fft',
              'constants',
              'sparse',

              # misc
              'maybe_custom_array',
              'maybe_custom_array_tree',
              'CustomArray',

              # _base_dimension
              'Dimension',
              'DIMENSIONLESS',
              'DimensionMismatchError',
              'UnitMismatchError',
              'get_or_create_dimension',
              'get_dim_for_display',

              # _base_unit
              'Unit',
              'UNITLESS',
              'add_standard_unit',
              'parse_unit',

              # _base_getters
              'is_dimensionless',
              'is_unitless',
              'is_scalar_type',
              'get_dim',
              'get_unit',
              'get_mantissa',
              'get_magnitude',
              'display_in_unit',
              'split_mantissa_unit',
              'maybe_decimal',
              'fail_for_dimension_mismatch',
              'fail_for_unit_mismatch',
              'assert_quantity',
              'have_same_dim',
              'has_same_unit',
              'unit_scale_align_to_first',
              'array_with_unit',

              # _base_quantity
              'Quantity',
              'compatible_with_equinox',

              # _base_decorators
              'check_dims',
              'check_units',
              'assign_units',

              # _celsius
              'celsius2kelvin',
              'kelvin2celsius',

              # old version compatibility
              'avogadro_constant',
              'boltzmann_constant',
              'electric_constant',
              'electron_mass',
              'elementary_charge',
              'faraday_constant',
              'gas_constant',
              'magnetic_constant',
              'molar_mass_constant',
          ] + _common_all + _std_units_all + _constants_all
del _common_all, _std_units_all, _matplotlib_compat, _constants_all
