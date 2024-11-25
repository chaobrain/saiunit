# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from __future__ import annotations

from typing import (Sequence, Callable)

from ._jacobian import jacrev, jacfwd
from ._misc import _check_callable

__all__ = [
    'hessian'
]


def hessian(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False
) -> Callable:
    """
    Physical unit-aware version of `jax.hessian <https://jax.readthedocs.io/en/latest/_autosummary/jax.hessian.html>`_,
    computing Hessian of ``fun`` as a dense array.

    Args:
      fun: Function whose Hessian is to be computed.  Its arguments at positions
        specified by ``argnums`` should be arrays, scalars, or standard Python
        containers thereof. It should return arrays, scalars, or standard Python
        containers thereof.
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default ``0``).
      has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data. Default False.
      holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
        holomorphic. Default False.

    Returns:
      A function with the same arguments as ``fun``, that evaluates the Hessian of
      ``fun``.
    """
    _check_callable(fun)

    return jacfwd(
        jacrev(fun, argnums, has_aux=has_aux, holomorphic=holomorphic),
        argnums, has_aux=has_aux, holomorphic=holomorphic,
    )
