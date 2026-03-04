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
Unit-aware automatic differentiation.

Provides ``grad``, ``value_and_grad``, ``jacobian`` (``jacrev``/``jacfwd``),
``hessian``, and ``vector_grad`` that correctly track physical units
through JAX's autodiff transformations.
"""

__all__ = [
    'value_and_grad',
    'grad',
    'vector_grad',
    'jacobian',
    'jacrev',
    'jacfwd',
    'hessian',
]

from ._hessian import *
from ._jacobian import *
from ._value_and_grad import *
from ._vector_grad import *
