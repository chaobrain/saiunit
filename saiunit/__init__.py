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
saiunit -- Physical units for JAX and NumPy arrays.

``saiunit`` provides a :class:`Quantity` type that pairs a JAX array or
NumPy array with a physical :class:`Unit`, ensuring dimensional correctness
at every arithmetic operation. The backend is detected from the mantissa
type; users can force a default with :func:`set_default_backend` or the
:func:`using_backend` context manager. ``saiunit`` also supplies the standard
SI base and derived units (e.g. ``meter``, ``second``, ``volt``), physical
constants, and unit-aware wrappers for NumPy/JAX math functions.

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

from ._matplotlib_compat import enable_matplotlib_support
from . import constants
from . import fft
from . import linalg
from . import math
from . import typing
from ._jax_compat import HAS_JAX as _HAS_JAX
from ._base_decorators import assign_units, check_dims, check_units
from ._base_dimension import (
    DIMENSIONLESS,
    Dimension,
    DimensionMismatchError,
    UnitMismatchError,
    get_dim_for_display,
    get_or_create_dimension,
)
from ._backend import (
    get_default_backend,
    is_cupy_array,
    is_dask_array,
    is_jax_array,
    is_ndonnx_array,
    is_numpy_array,
    is_torch_array,
    set_default_backend,
    using_backend,
)
from ._exceptions import BackendError
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
    is_unit_equal_math,
    is_unitless,
    maybe_decimal,
    split_mantissa_unit,
    unit_scale_align_to_first,
)
from ._base_quantity import Quantity
from ._base_unit import UNITLESS, Unit, add_standard_unit, parse_unit
from ._celsius import celsius2kelvin, kelvin2celsius, fahrenheit2kelvin, kelvin2fahrenheit
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

              # submodules ('autograd', 'lax' and 'sparse' are appended below
              # together with the lazy-loading machinery)
              'math',
              'linalg',
              'fft',
              'constants',
              'typing',

              # misc
              'maybe_custom_array',
              'maybe_custom_array_tree',
              'CustomArray',

              # _base_dimension
              'Dimension',
              'DIMENSIONLESS',
              'DimensionMismatchError',
              'UnitMismatchError',
              'BackendError',

              # _backend
              'get_default_backend',
              'set_default_backend',
              'using_backend',
              'is_jax_array',
              'is_numpy_array',
              'is_cupy_array',
              'is_torch_array',
              'is_dask_array',
              'is_ndonnx_array',
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
              'is_unit_equal_math',
              'unit_scale_align_to_first',
              'array_with_unit',

              # _base_quantity
              'Quantity',

              # _base_decorators
              'check_dims',
              'check_units',
              'assign_units',

              # _celsius
              'celsius2kelvin',
              'kelvin2celsius',
              'fahrenheit2kelvin',
              'kelvin2fahrenheit',

              # _matplotlib_compat
              'enable_matplotlib_support',

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
del _common_all, _std_units_all, _constants_all

# ---------------------------------------------------------------------------
# Lazy submodule loading for JAX-only features.
#
# ``autograd``, ``lax`` and ``sparse`` use JAX primitives that have no NumPy
# equivalent. Importing them eagerly would force a hard JAX dependency on
# every ``import saiunit`` call. Instead, expose them through a module-level
# ``__getattr__``: the first attribute access triggers the real import and,
# if JAX is missing, raises :class:`~saiunit._exceptions.BackendError` with
# an actionable install hint.
# ---------------------------------------------------------------------------

_JAX_ONLY_SUBMODULES = ("autograd", "lax", "sparse")
__all__ = __all__ + ["autograd", "lax", "sparse"]


def __getattr__(name):
    if name in _JAX_ONLY_SUBMODULES:
        if not _HAS_JAX:
            from ._exceptions import BackendError
            raise BackendError(
                f"saiunit.{name} requires JAX. Install with: pip install saiunit[jax]"
            )
        import importlib
        mod = importlib.import_module(f"saiunit.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'saiunit' has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(globals()))
