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
Unit-aware mathematical functions for JAX arrays.

This subpackage provides unit-aware wrappers for NumPy-style functions,
organized by how they handle units:

- **Array creation**: ``array``, ``zeros``, ``ones``, ``arange``, ``linspace``, etc.
- **Keep unit**: functions that preserve the input unit (``sum``, ``mean``,
  ``concatenate``, ``reshape``, ``abs``, ``round``, etc.).
- **Change unit**: functions whose output unit differs from the input
  (``multiply``, ``divide``, ``square``, ``sqrt``, ``dot``, ``matmul``, etc.).
- **Remove unit**: functions that return dimensionless results
  (``equal``, ``greater``, ``argmax``, ``argsort``, ``sign``, etc.).
- **Accept unitless**: functions that require unitless inputs
  (``exp``, ``log``, ``sin``, ``cos``, ``arctan``, etc.).
- **Activations**: neural-network activation functions
  (``relu``, ``sigmoid``, ``gelu``, ``silu``, etc.).
- **Einops**: Einstein-notation operations (``einsum``, ``einrearrange``, etc.).
"""

from . import linalg, fft
from ._activation import *
from ._activation import __all__ as _activation_all
from ._alias import *
from ._alias import __all__ as _alias_all
from ._einops import *
from ._einops import __all__ as _einops_all
from ._fun_accept_unitless import *
from ._fun_accept_unitless import __all__ as _compat_funcs_accept_unitless_all
from ._fun_array_creation import *
from ._fun_array_creation import __all__ as _compat_array_creation_all
from ._fun_change_unit import *
from ._fun_change_unit import __all__ as _compat_funcs_change_unit_all
from ._fun_keep_unit import *
from ._fun_keep_unit import __all__ as _compat_funcs_keep_unit_all
from ._fun_remove_unit import *
from ._fun_remove_unit import __all__ as _compat_funcs_remove_unit_all
from ._misc import *
from ._misc import __all__ as _compat_misc_all

__all__ = (
    _compat_array_creation_all +
    _alias_all +
    _compat_funcs_change_unit_all +
    _compat_funcs_keep_unit_all +
    _compat_funcs_accept_unitless_all +
    _compat_funcs_remove_unit_all +
    _compat_misc_all +
    _einops_all +
    _activation_all +
    [
        'linalg', 'fft'
    ]
)

del (
    _compat_array_creation_all,
    _alias_all,
    _compat_funcs_change_unit_all,
    _compat_funcs_keep_unit_all,
    _compat_funcs_accept_unitless_all,
    _compat_funcs_remove_unit_all,
    _compat_misc_all,
    _einops_all,
    _activation_all
)
