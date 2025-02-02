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

    >>> import jax.numpy as jnp
    >>> import saiunit as u
    >>> def scalar_function1(x):
    ...    return x ** 2 + 3 * x * u.ms + 2 * u.msecond2
    >>> hess_fn = u.autograd.hessian(scalar_function1)
    >>> hess_fn(jnp.array(1.0) * u.ms)
    [2]

    >>> import jax.numpy as jnp
    >>> import saiunit as u
    >>> def scalar_function2(x):
    ...     return x ** 3 + 3 * x * u.msecond2 + 2 * u.msecond3
    >>> hess_fn = u.autograd.hessian(scalar_function2)
    >>> hess_fn(jnp.array(1.0) * u.ms)
    [6] * ms

    `hessian` is a generalization of the usual definition of the Hessian
    that supports nested Python containers (i.e. pytrees) as inputs and outputs.
    The tree structure of ``saiunit.autograd.hessian(fun)(x)`` is given by forming a tree
    product of the structure of ``fun(x)`` with a tree product of two copies of
    the structure of ``x``. A tree product of two tree structures is formed by
    replacing each leaf of the first tree with a copy of the second. For example:

    >>> import jax.numpy as jnp
    >>> import saiunit as u
    >>> def dict_function(x):
    ...     return {'z': x['a'] ** 3 + x['b'] ** 3}
    >>> x = {'a': jnp.array(1.0) * u.ms, 'b': jnp.array(2.0) * u.ms}
    >>> u.autograd.hessian(dict_function)(x)
    {'z': {'a': {'a': 6. * msecond, 'b': 0. * msecond},
     'b': {'a': 0. * msecond, 'b': 12. * msecond}}}

    Thus each leaf in the tree structure of ``saiunit.autograd.hessian(fun)(x)`` corresponds to
    a leaf of ``fun(x)`` and a pair of leaves of ``x``. For each leaf in
    ``saiunit.autograd.hessian(fun)(x)``, if the corresponding array leaf of ``fun(x)`` has
    shape ``(out_1, out_2, ...)`` and the corresponding array leaves of ``x`` have
    shape ``(in_1_1, in_1_2, ...)`` and ``(in_2_1, in_2_2, ...)`` respectively,
    then the Hessian leaf has shape ``(out_1, out_2, ..., in_1_1, in_1_2, ...,
    in_2_1, in_2_2, ...)``. In other words, the Python tree structure represents
    the block structure of the Hessian, with blocks determined by the input and
    output pytrees.

    In particular, an array is produced (with no pytrees involved) when the
    function input ``x`` and output ``fun(x)`` are each a single array, as in the
    ``dict_function`` example above. If ``fun(x)`` has shape ``(out1, out2, ...)`` and ``x``
    has shape ``(in1, in2, ...)`` then ``saiunit.autograd.hessian(fun)(x)`` has shape
    ``(out1, out2, ..., in1, in2, ..., in1, in2, ...)``. To flatten pytrees into
    1D vectors, consider using :py:func:`jax.flatten_util.flatten_pytree`.
    """
    _check_callable(fun)

    return jacfwd(
        jacrev(fun, argnums, has_aux=has_aux, holomorphic=holomorphic),
        argnums, has_aux=has_aux, holomorphic=holomorphic,
    )
