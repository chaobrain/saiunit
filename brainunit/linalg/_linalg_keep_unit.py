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

import functools
from typing import (Union, Sequence, Tuple, Optional)

import jax
import jax.numpy as jnp
import numpy as np

from ..math._fun_array_creation import asarray
from ..math._fun_keep_unit import _fun_keep_unit_unary
from .._base import (
    Quantity,
    fail_for_dimension_mismatch,
    get_unit,
    UNITLESS,
    unit_scale_align_to_first,
    maybe_decimal
)
from .._misc import set_module_as

__all__ = [
    'norm',
]

@set_module_as('brainunit.math')
def norm(
    x: Union[jax.typing.ArrayLike, Quantity],
    ord: int | str | None = None,
    axis: None | tuple[int, ...] | int = None,
    keepdims: bool = False,
) -> Union[jax.Array, Quantity]:
    """Compute the norm of a matrix or vector.

    Args:
        x: N-dimensional array for which the norm will be computed.
        ord: specify the kind of norm to take. Default is Frobenius norm for matrices,
            and the 2-norm for vectors. For other options, see Notes below.
        axis: integer or sequence of integers specifying the axes over which the norm
            will be computed. Defaults to all axes of ``x``.
        keepdims: if True, the output array will have the same number of dimensions as
            the input, with the size of reduced axes replaced by ``1`` (default: False).

    Returns:
        array containing the specified norm of x.

    Notes:
    The flavor of norm computed depends on the value of ``ord`` and the number of
    axes being reduced.

    For **vector norms** (i.e. a single axis reduction):

    - ``ord=None`` (default) computes the 2-norm
    - ``ord=inf`` computes ``max(abs(x))``
    - ``ord=-inf`` computes min(abs(x))``
    - ``ord=0`` computes ``sum(x!=0)``
    - for other numerical values, computes ``sum(abs(x) ** ord)**(1/ord)``

    For **matrix norms** (i.e. two axes reductions):

    - ``ord='fro'`` or ``ord=None`` (default) computes the Frobenius norm
    - ``ord='nuc'`` computes the nuclear norm, or the sum of the singular values
    - ``ord=1`` computes ``max(abs(x).sum(0))``
    - ``ord=-1`` computes ``min(abs(x).sum(0))``
    - ``ord=2`` computes the 2-norm, i.e. the largest singular value
    - ``ord=-2`` computes the smallest singular value

    Examples:
    Vector norms:

    >>> x = jnp.array([3., 4., 12.])
    >>> jnp.linalg.norm(x)
    Array(13., dtype=float32)
    >>> jnp.linalg.norm(x, ord=1)
    Array(19., dtype=float32)
    >>> jnp.linalg.norm(x, ord=0)
    Array(3., dtype=float32)

    Matrix norms:

    >>> x = jnp.array([[1., 2., 3.],
    ...                [4., 5., 7.]])
    >>> jnp.linalg.norm(x)  # Frobenius norm
    Array(10.198039, dtype=float32)
    >>> jnp.linalg.norm(x, ord='nuc')  # nuclear norm
    Array(10.762535, dtype=float32)
    >>> jnp.linalg.norm(x, ord=1)  # 1-norm
    Array(10., dtype=float32)

    Batched vector norm:

    >>> jnp.linalg.norm(x, axis=1)
    Array([3.7416575, 9.486833 ], dtype=float32)
    """
    return _fun_keep_unit_unary(jnp.linalg.norm, x, ord=ord, axis=axis, keepdims=keepdims)

