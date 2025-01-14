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

import builtins
from typing import Union, Sequence, Callable

import jax
import numpy as np
from jax import lax
from jax._src.typing import Shape

from .._base import Quantity, maybe_decimal, has_same_unit
from .._misc import set_module_as
from ..math._fun_keep_unit import _fun_keep_unit_unary, _fun_keep_unit_binary

__all__ = [
    # sequence inputs

    # array manipulation
    'slice', 'dynamic_slice', 'dynamic_update_slice', 'gather',
    'index_take', 'slice_in_dim', 'index_in_dim', 'dynamic_slice_ind_dim', 'dynamic_index_in_dim',
    'dynamic_update_slice_in_dim', 'dynamic_update_index_in_dim',
    'sort', 'sort_key_val',

    # math funcs keep unit (unary)
    'neg',
    'cummax', 'cummin', 'cumsum',
    'scatter', 'scatter_add', 'scatter_sub', 'scatter_mul', 'scatter_min', 'scatter_max', 'scatter_apply',

    # math funcs keep unit (binary)
    'sub', 'complex', 'pad',

    # math funcs keep unit (n-ary)
    'clamp',

    # type conversion
    'convert_element_type', 'bitcast_convert_type',

    # math funcs keep unit (return Quantity and index)
    'approx_max_k', 'approx_min_k', 'top_k',

    # math funcs only accept unitless (unary) can return Quantity

    # broadcasting arrays
    'broadcast', 'broadcast_in_dim', 'broadcast_to_rank',
]


# array manipulation
@set_module_as('brainunit.math')
def slice(
    operand: Union[Quantity, jax.typing.ArrayLike],
    start_indices: Sequence[int],
    limit_indices: Sequence[int],
    strides: Sequence[int] | None = None
) -> Union[Quantity, jax.Array]:
    """Wraps XLA's `Slice
    <https://www.tensorflow.org/xla/operation_semantics#slice>`_
    operator.

    Args:
        operand: an array to slice
        start_indices: a sequence of ``operand.ndim`` start indices.
        limit_indices: a sequence of ``operand.ndim`` limit indices.
        strides: an optional sequence of ``operand.ndim`` strides.

    Returns:
        The sliced array

    Examples:
        Here are some examples of simple two-dimensional slices:

        >>> x = jnp.arange(12).reshape(3, 4)
        >>> x
        Array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]], dtype=int32)

        >>> lax.slice(x, (1, 0), (3, 2))
        Array([[4, 5],
               [8, 9]], dtype=int32)

        >>> lax.slice(x, (0, 0), (3, 4), (1, 2))
        Array([[ 0,  2],
               [ 4,  6],
               [ 8, 10]], dtype=int32)

        These two examples are equivalent to the following Python slicing syntax:

        >>> x[1:3, 0:2]
        Array([[4, 5],
               [8, 9]], dtype=int32)

        >>> x[0:3, 0:4:2]
        Array([[ 0,  2],
               [ 4,  6],
               [ 8, 10]], dtype=int32)
    """
    return _fun_keep_unit_unary(lax.slice, operand, start_indices, limit_indices, strides)


@set_module_as('brainunit.math')
def dynamic_slice(
    operand: Union[Quantity, jax.typing.ArrayLike],
    start_indices: jax.typing.ArrayLike | Sequence[jax.typing.ArrayLike],
    slice_sizes: Shape,
) -> Union[Quantity, jax.Array]:
    """Wraps XLA's `DynamicSlice
    <https://www.tensorflow.org/xla/operation_semantics#dynamicslice>`_
    operator.

    Args:
        operand: an array to slice.
        start_indices: a list of scalar indices, one per dimension. These values
          may be dynamic.
        slice_sizes: the size of the slice. Must be a sequence of non-negative
          integers with length equal to `ndim(operand)`. Inside a JIT compiled
          function, only static values are supported (all JAX arrays inside JIT
          must have statically known size).

    Returns:
        An array containing the slice.

    Examples:
        Here is a simple two-dimensional dynamic slice:

        >>> x = jnp.arange(12).reshape(3, 4)
        >>> x
        Array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]], dtype=int32)

        >>> dynamic_slice(x, (1, 1), (2, 3))
        Array([[ 5,  6,  7],
               [ 9, 10, 11]], dtype=int32)

        Note the potentially surprising behavior for the case where the requested slice
        overruns the bounds of the array; in this case the start index is adjusted to
        return a slice of the requested size:

        >>> dynamic_slice(x, (1, 1), (2, 4))
        Array([[ 4,  5,  6,  7],
               [ 8,  9, 10, 11]], dtype=int32)
    """
    return _fun_keep_unit_unary(lax.dynamic_slice, operand, start_indices, slice_sizes)


@set_module_as('brainunit.math')
def dynamic_update_slice(
    operand: Union[Quantity, jax.typing.ArrayLike],
    update: Union[Quantity, jax.typing.ArrayLike],
    start_indices: jax.typing.ArrayLike | Sequence[jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    """Wraps XLA's `DynamicUpdateSlice
    <https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice>`_
    operator.

    Args:
        operand: an array to slice.
        update: an array containing the new values to write onto `operand`.
        start_indices: a list of scalar indices, one per dimension.

    Returns:
        An array containing the slice.

    Examples:
        Here is an example of updating a one-dimensional slice update:

        >>> x = jnp.zeros(6)
        >>> y = jnp.ones(3)
        >>> dynamic_update_slice(x, y, (2,))
        Array([0., 0., 1., 1., 1., 0.], dtype=float32)

        If the update slice is too large to fit in the array, the start
        index will be adjusted to make it fit

        >>> dynamic_update_slice(x, y, (3,))
        Array([0., 0., 0., 1., 1., 1.], dtype=float32)
        >>> dynamic_update_slice(x, y, (5,))
        Array([0., 0., 0., 1., 1., 1.], dtype=float32)

        Here is an example of a two-dimensional slice update:

        >>> x = jnp.zeros((4, 4))
        >>> y = jnp.ones((2, 2))
        >>> dynamic_update_slice(x, y, (1, 2))
        Array([[0., 0., 0., 0.],
               [0., 0., 1., 1.],
               [0., 0., 1., 1.],
               [0., 0., 0., 0.]], dtype=float32)
    """
    return _fun_keep_unit_binary(lax.dynamic_update_slice, operand, update, start_indices)


@set_module_as('brainunit.math')
def gather(
    operand: Union[Quantity, jax.typing.ArrayLike],
    start_indices: jax.typing.ArrayLike,
    dimension_numbers: jax.lax.GatherDimensionNumbers,
    slice_sizes: Shape,
    *,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    mode: str | jax.lax.GatherScatterMode | None = None,
    fill_value: Union[Quantity, jax.typing.ArrayLike] = None
) -> Union[Quantity, jax.Array]:
    """Gather operator.

    Wraps `XLA's Gather operator
    <https://www.tensorflow.org/xla/operation_semantics#gather>`_.

    :func:`gather` is a low-level operator with complicated semantics, and most JAX
    users will never need to call it directly. Instead, you should prefer using
    `Numpy-style indexing`_, and/or :func:`jax.numpy.ndarray.at`, perhaps in combination
    with :func:`jax.vmap`.

    Args:
        operand: an array from which slices should be taken
        start_indices: the indices at which slices should be taken
        dimension_numbers: a `lax.GatherDimensionNumbers` object that describes
            how dimensions of `operand`, `start_indices` and the output relate.
        slice_sizes: the size of each slice. Must be a sequence of non-negative
            integers with length equal to `ndim(operand)`.
        indices_are_sorted: whether `indices` is known to be sorted. If
            true, may improve performance on some backends.
        unique_indices: whether the elements gathered from ``operand`` are
              guaranteed not to overlap with each other. If ``True``, this may improve
              performance on some backends. JAX does not check this promise: if
              the elements overlap the behavior is undefined.
        mode: how to handle indices that are out of bounds: when set to ``'clip'``,
              indices are clamped so that the slice is within bounds, and when
              set to ``'fill'`` or ``'drop'`` gather returns a slice full of
              ``fill_value`` for the affected slice. The behavior for out-of-bounds
              indices when set to ``'promise_in_bounds'`` is implementation-defined.
        fill_value: the fill value to return for out-of-bounds slices when `mode`
              is ``'fill'``. Ignored otherwise. Defaults to ``NaN`` for inexact types,
              the largest negative value for signed types, the largest positive value
              for unsigned types, and ``True`` for booleans.

    Returns:
        An array containing the gather output.

    Examples:
        As mentioned above, you should basically never use :func:`gather` directly,
        and instead use NumPy-style indexing expressions to gather values from
        arrays.

        For example, here is how you can extract values at particular indices using
        straightforward indexing semantics, which will lower to XLA's Gather operator:

        >>> import jax.numpy as jnp
        >>> x = jnp.array([10, 11, 12])
        >>> indices = jnp.array([0, 1, 1, 2, 2, 2])

        >>> x[indices]
        Array([10, 11, 11, 12, 12, 12], dtype=int32)

        For control over settings like ``indices_are_sorted``, ``unique_indices``, ``mode``,
        and ``fill_value``, you can use the :attr:`jax.numpy.ndarray.at` syntax:

        >>> x.at[indices].get(indices_are_sorted=True, mode="promise_in_bounds")
        Array([10, 11, 11, 12, 12, 12], dtype=int32)

        By comparison, here is the equivalent function call using :func:`gather` directly,
        which is not something typical users should ever need to do:

        >>> from jax import lax
        >>> lax.gather(x, indices[:, None], slice_sizes=(1,),
        ...            dimension_numbers=lax.GatherDimensionNumbers(
        ...                offset_dims=(),
        ...                collapsed_slice_dims=(0,),
        ...                start_index_map=(0,)),
        ...            indices_are_sorted=True,
        ...            mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS)
        Array([10, 11, 11, 12, 12, 12], dtype=int32)
    """
    if isinstance(operand, Quantity) and isinstance(fill_value, Quantity):
        return maybe_decimal(
            Quantity(lax.gather(operand.mantissa, start_indices, dimension_numbers, slice_sizes,
                                unique_indices=unique_indices, indices_are_sorted=indices_are_sorted,
                                mode=mode, fill_value=fill_value.in_unit(operand.unit)),
                     unit=operand.unit)
        )
    elif isinstance(operand, Quantity):
        if fill_value is not None:
            raise ValueError('fill_value must be a Quantity if operand is a Quantity')
        return maybe_decimal(
            Quantity(lax.gather(operand.mantissa, start_indices, dimension_numbers, slice_sizes,
                                unique_indices=unique_indices, indices_are_sorted=indices_are_sorted,
                                mode=mode), unit=operand.unit)
        )
    elif isinstance(fill_value, Quantity):
        raise ValueError('fill_value must be None if operand is not a Quantity')
    return lax.gather(operand, start_indices, dimension_numbers, slice_sizes,
                      unique_indices=unique_indices, indices_are_sorted=indices_are_sorted,
                      mode=mode, fill_value=fill_value)


@set_module_as('brainunit.math')
def index_take(
    src: Union[Quantity, jax.typing.ArrayLike],
    idxs: jax.typing.ArrayLike,
    axes: Sequence[int]
) -> Union[Quantity, jax.Array]:
    return _fun_keep_unit_unary(lax.index_take, src, idxs, axes)


@set_module_as('brainunit.math')
def slice_in_dim(
    operand: Union[Quantity, jax.typing.ArrayLike],
    start_index: int | None,
    limit_index: int | None,
    stride: int = 1,
    axis: int = 0
) -> Union[Quantity, jax.Array]:
    """Convenience wrapper around :func:`lax.slice` applying to only one dimension.

    This is effectively equivalent to ``operand[..., start_index:limit_index:stride]``
    with the indexing applied on the specified axis.

    Args:
        operand: an array to slice.
        start_index: an optional start index (defaults to zero)
        limit_index: an optional end index (defaults to operand.shape[axis])
        stride: an optional stride (defaults to 1)
        axis: the axis along which to apply the slice (defaults to 0)

    Returns:
        An array containing the slice.

    Examples:
        Here is a one-dimensional example:

        >>> x = jnp.arange(4)
        >>> lax.slice_in_dim(x, 1, 3)
        Array([1, 2], dtype=int32)

        Here are some two-dimensional examples:

        >>> x = jnp.arange(12).reshape(4, 3)
        >>> x
        Array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11]], dtype=int32)

        >>> lax.slice_in_dim(x, 1, 3)
        Array([[3, 4, 5],
               [6, 7, 8]], dtype=int32)

        >>> lax.slice_in_dim(x, 1, 3, axis=1)
        Array([[ 1,  2],
               [ 4,  5],
               [ 7,  8],
               [10, 11]], dtype=int32)
    """
    return _fun_keep_unit_unary(lax.slice_in_dim, operand, start_index, limit_index, stride, axis)


@set_module_as('brainunit.math')
def index_in_dim(
    operand: Union[Quantity, jax.typing.ArrayLike],
    index: int,
    axis: int = 0,
    keepdims: bool = True
) -> Union[Quantity, jax.Array]:
    """Convenience wrapper around :func:`lax.slice` to perform int indexing.

    This is effectively equivalent to ``operand[..., start_index:limit_index:stride]``
    with the indexing applied on the specified axis.

    Args:
        operand: an array to index.
        index: integer index
        axis: the axis along which to apply the index (defaults to 0)
        keepdims: boolean specifying whether the output array should preserve the
          rank of the input (default=True)

    Returns:
        The subarray at the specified index.

    Examples:
        Here is a one-dimensional example:

        >>> x = jnp.arange(4)
        >>> lax.index_in_dim(x, 2)
        Array([2], dtype=int32)

        >>> lax.index_in_dim(x, 2, keepdims=False)
        Array(2, dtype=int32)

        Here are some two-dimensional examples:

        >>> x = jnp.arange(12).reshape(3, 4)
        >>> x
        Array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]], dtype=int32)

        >>> lax.index_in_dim(x, 1)
        Array([[4, 5, 6, 7]], dtype=int32)

        >>> lax.index_in_dim(x, 1, axis=1, keepdims=False)
        Array([1, 5, 9], dtype=int32)
    """
    return _fun_keep_unit_unary(lax.index_in_dim, operand, index, axis, keepdims)


@set_module_as('brainunit.math')
def dynamic_slice_ind_dim(
    operand: Union[Quantity, jax.typing.ArrayLike],
    start_index: jax.typing.ArrayLike,
    slice_size: int,
    axis: int = 0
) -> Union[Quantity, jax.Array]:
    """Convenience wrapper around :func:`lax.dynamic_slice` applied to one dimension.

    This is roughly equivalent to the following Python indexing syntax applied
    along the specified axis: ``operand[..., start_index:start_index + slice_size]``.

    Args:
        operand: an array to slice.
        start_index: the (possibly dynamic) start index
        slice_size: the static slice size
        axis: the axis along which to apply the slice (defaults to 0)

    Returns:
        An array containing the slice.

    Examples:
        Here is a one-dimensional example:

        >>> x = jnp.arange(5)
        >>> dynamic_slice_in_dim(x, 1, 3)
        Array([1, 2, 3], dtype=int32)

        Like `jax.lax.dynamic_slice`, out-of-bound slices will be clipped to the
        valid range:

        >>> dynamic_slice_in_dim(x, 4, 3)
        Array([2, 3, 4], dtype=int32)

        Here is a two-dimensional example:

        >>> x = jnp.arange(12).reshape(3, 4)
        >>> x
        Array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]], dtype=int32)

        >>> dynamic_slice_in_dim(x, 1, 2, axis=1)
        Array([[ 1,  2],
               [ 5,  6],
               [ 9, 10]], dtype=int32)
    """
    return _fun_keep_unit_unary(lax.dynamic_slice_in_dim, operand, start_index, slice_size, axis)


@set_module_as('brainunit.math')
def dynamic_index_in_dim(
    operand: Union[Quantity, jax.typing.ArrayLike],
    index: int | jax.typing.ArrayLike,
    axis: int = 0, keepdims: bool = True
) -> Union[Quantity, jax.Array]:
    """Convenience wrapper around dynamic_slice to perform int indexing.

    This is roughly equivalent to the following Python indexing syntax applied
    along the specified axis: ``operand[..., index]``.

    Args:
        operand: an array to slice.
        index: the (possibly dynamic) start index
        axis: the axis along which to apply the slice (defaults to 0)
        keepdims: boolean specifying whether the output should have the same rank as
          the input (default = True)

    Returns:
        An array containing the slice.

    Examples:
        Here is a one-dimensional example:

        >>> x = jnp.arange(5)
        >>> dynamic_index_in_dim(x, 1)
        Array([1], dtype=int32)

        >>> dynamic_index_in_dim(x, 1, keepdims=False)
        Array(1, dtype=int32)

        Here is a two-dimensional example:

        >>> x = jnp.arange(12).reshape(3, 4)
        >>> x
        Array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]], dtype=int32)

        >>> dynamic_index_in_dim(x, 1, axis=1, keepdims=False)
        Array([1, 5, 9], dtype=int32)
    """
    return _fun_keep_unit_unary(lax.dynamic_index_in_dim, operand, index, axis, keepdims)


@set_module_as('brainunit.math')
def dynamic_update_slice_in_dim(
    operand: Union[Quantity, jax.typing.ArrayLike],
    update: Union[Quantity, jax.typing.ArrayLike],
    start_index: jax.typing.ArrayLike, axis: int
) -> Union[Quantity, jax.Array]:
    """Convenience wrapper around :func:`dynamic_update_slice` to update
    a slice in a single ``axis``.

    Args:
        operand: an array to slice.
        update: an array containing the new values to write onto `operand`.
        start_index: a single scalar index
        axis: the axis of the update.

    Returns:
        The updated array

    Examples:

        >>> x = jnp.zeros(6)
        >>> y = jnp.ones(3)
        >>> dynamic_update_slice_in_dim(x, y, 2, axis=0)
        Array([0., 0., 1., 1., 1., 0.], dtype=float32)

        If the update slice is too large to fit in the array, the start
        index will be adjusted to make it fit:

        >>> dynamic_update_slice_in_dim(x, y, 3, axis=0)
        Array([0., 0., 0., 1., 1., 1.], dtype=float32)
        >>> dynamic_update_slice_in_dim(x, y, 5, axis=0)
        Array([0., 0., 0., 1., 1., 1.], dtype=float32)

        Here is an example of a two-dimensional slice update:

        >>> x = jnp.zeros((4, 4))
        >>> y = jnp.ones((2, 4))
        >>> dynamic_update_slice_in_dim(x, y, 1, axis=0)
        Array([[0., 0., 0., 0.],
               [1., 1., 1., 1.],
               [1., 1., 1., 1.],
               [0., 0., 0., 0.]], dtype=float32)

        Note that the shape of the additional axes in ``update`` need not
        match the associated dimensions of the ``operand``:

        >>> y = jnp.ones((2, 3))
        >>> dynamic_update_slice_in_dim(x, y, 1, axis=0)
        Array([[0., 0., 0., 0.],
               [1., 1., 1., 0.],
               [1., 1., 1., 0.],
               [0., 0., 0., 0.]], dtype=float32)
    """
    return _fun_keep_unit_binary(lax.dynamic_update_slice_in_dim, operand, update, start_index, axis)


@set_module_as('brainunit.math')
def dynamic_update_index_in_dim(
    operand: Union[Quantity, jax.typing.ArrayLike],
    update: Union[Quantity, jax.typing.ArrayLike],
    index: jax.typing.ArrayLike,
    axis: int
) -> Union[Quantity, jax.Array]:
    """Convenience wrapper around :func:`dynamic_update_slice` to update a slice
    of size 1 in a single ``axis``.

    Args:
        operand: an array to slice.
        update: an array containing the new values to write onto `operand`.
        index: a single scalar index
        axis: the axis of the update.

    Returns:
        The updated array

    Examples:

        >>> x = jnp.zeros(6)
        >>> y = 1.0
        >>> dynamic_update_index_in_dim(x, y, 2, axis=0)
        Array([0., 0., 1., 0., 0., 0.], dtype=float32)

        >>> y = jnp.array([1.0])
        >>> dynamic_update_index_in_dim(x, y, 2, axis=0)
        Array([0., 0., 1., 0., 0., 0.], dtype=float32)

        If the specified index is out of bounds, the index will be clipped to the
        valid range:

        >>> dynamic_update_index_in_dim(x, y, 10, axis=0)
        Array([0., 0., 0., 0., 0., 1.], dtype=float32)

        Here is an example of a two-dimensional dynamic index update:

        >>> x = jnp.zeros((4, 4))
        >>> y = jnp.ones(4)
        >>> dynamic_update_index_in_dim(x, y, 1, axis=0)
        Array([[0., 0., 0., 0.],
              [1., 1., 1., 1.],
              [0., 0., 0., 0.],
              [0., 0., 0., 0.]], dtype=float32)

        Note that the shape of the additional axes in ``update`` need not
        match the associated dimensions of the ``operand``:

        >>> y = jnp.ones((1, 3))
        >>> dynamic_update_index_in_dim(x, y, 1, 0)
        Array([[0., 0., 0., 0.],
               [1., 1., 1., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]], dtype=float32)
    """
    return _fun_keep_unit_binary(lax.dynamic_update_index_in_dim, operand, update, index, axis)


@set_module_as('brainunit.math')
def sort(
    operand: Union[Quantity, jax.typing.ArrayLike] | Sequence[Union[Quantity, jax.typing.ArrayLike]],
    dimension: int = -1,
    is_stable: bool = True, num_keys: int = 1
) -> Union[Quantity, jax.Array] | Sequence[Union[Quantity, jax.Array]]:
    """Wraps XLA's `Sort
    <https://www.tensorflow.org/xla/operation_semantics#sort>`_ operator.

    For floating point inputs, -0.0 and 0.0 are treated as equivalent, and NaN values
    are sorted to the end of the array. For complex inputs, the sort order is
    lexicographic over the real and imaginary parts, with the real part primary.

    Args:
        operand : Array or sequence of arrays
        dimension : integer dimension along which to sort. Default: -1.
        is_stable : boolean specifying whether to use a stable sort. Default: True.
        num_keys : number of operands to treat as sort keys. Default: 1.
              For num_keys > 1, the sort order will be determined lexicographically using
              the first `num_keys` arrays, with the first key being primary.
              The remaining operands will be returned with the same permutation.

    Returns:
        operand : sorted version of the input or inputs.
    """
    # check if operand is a sequence
    if isinstance(operand, Sequence):
        output = []
        for op in operand:
            if isinstance(op, Quantity):
                output.append(Quantity(lax.sort(op.mantissa, dimension, is_stable, num_keys), unit=op.unit))
            else:
                output.append(lax.sort(op, dimension, is_stable, num_keys))
        return output
    else:
        if isinstance(operand, Quantity):
            return Quantity(lax.sort(operand.mantissa, dimension, is_stable, num_keys), unit=operand.unit)
        return lax.sort(operand, dimension, is_stable, num_keys)


@set_module_as('brainunit.math')
def sort_key_val(
    keys: Union[Quantity, jax.typing.ArrayLike],
    values: Union[Quantity, jax.typing.ArrayLike],
    dimension: int = -1,
    is_stable: bool = True
) -> tuple[Union[Quantity, jax.Array], Union[Quantity, jax.Array]]:
    """Sorts ``keys`` along ``dimension`` and applies the same permutation to ``values``."""
    if isinstance(keys, Quantity) and isinstance(values, Quantity):
        k, v = lax.sort_key_val(keys.mantissa, values.mantissa, dimension, is_stable)
        return maybe_decimal(Quantity(k, unit=keys.unit)), maybe_decimal(Quantity(v, unit=values.unit))
    elif isinstance(keys, Quantity):
        k, v = lax.sort_key_val(keys.mantissa, values, dimension, is_stable)
        return maybe_decimal(Quantity(k, unit=keys.unit)), v
    elif isinstance(values, Quantity):
        k, v = lax.sort_key_val(keys, values.mantissa, dimension, is_stable)
        return k, maybe_decimal(Quantity(v, unit=values.unit))
    return lax.sort_key_val(keys, values, dimension, is_stable)


# math funcs keep unit (unary)
@set_module_as('brainunit.math')
def neg(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    r"""Elementwise negation: :math:`-x`."""
    return _fun_keep_unit_unary(lax.neg, x)


@set_module_as('brainunit.math')
def cummax(
    operand: Union[Quantity, jax.typing.ArrayLike],
    axis: int = 0,
    reverse: bool = False
) -> Union[Quantity, jax.Array]:
    """Computes a cumulative maximum along `axis`."""
    return _fun_keep_unit_unary(lax.cummax, operand, axis, reverse)


@set_module_as('brainunit.math')
def cummin(
    operand: Union[Quantity, jax.typing.ArrayLike],
    axis: int = 0,
    reverse: bool = False
) -> Union[Quantity, jax.Array]:
    """Computes a cumulative minimum along `axis`."""
    return _fun_keep_unit_unary(lax.cummin, operand, axis, reverse)


@set_module_as('brainunit.math')
def cumsum(
    operand: Union[Quantity, jax.typing.ArrayLike],
    axis: int = 0,
    reverse: bool = False
) -> Union[Quantity, jax.Array]:
    """Computes a cumulative sum along `axis`."""
    return _fun_keep_unit_unary(lax.cumsum, operand, axis, reverse)


def _fun_lax_scatter(
    fun: Callable,
    operand,
    scatter_indices,
    updates,
    dimension_numbers,
    indices_are_sorted,
    unique_indices,
    mode
) -> Union[Quantity, jax.Array]:
    if isinstance(operand, Quantity) and isinstance(updates, Quantity):
        assert has_same_unit(operand,
                             updates), f'operand(unit:{operand.unit}) and updates(unit:{updates.unit}) do not have same unit'
        return maybe_decimal(Quantity(fun(operand.mantissa, scatter_indices, updates.mantissa, dimension_numbers,
                                          indices_are_sorted=indices_are_sorted,
                                          unique_indices=unique_indices,
                                          mode=mode), unit=operand.unit))
    elif isinstance(operand, Quantity) or isinstance(updates, Quantity):
        raise AssertionError(
            f'operand and updates should both be `Quantity` or Array, now we got {type(operand)} and {type(updates)}')
    else:
        return fun(operand, scatter_indices, updates, dimension_numbers,
                   indices_are_sorted=indices_are_sorted,
                   unique_indices=unique_indices,
                   mode=mode)


@set_module_as('brainunit.math')
def scatter(
    operand: Union[Quantity, jax.typing.ArrayLike],
    scatter_indices: jax.typing.ArrayLike,
    updates: jax.typing.ArrayLike,
    dimension_numbers: jax.lax.ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    """Scatter-update operator.

    Wraps `XLA's Scatter operator
    <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where updates
    replace values from `operand`.

    If multiple updates are performed to the same index of operand, they may be
    applied in any order.

    :func:`scatter` is a low-level operator with complicated semantics, and most
    JAX users will never need to call it directly. Instead, you should prefer using
    :func:`jax.numpy.ndarray.at` for more familiary NumPy-style indexing syntax.

    Args:
        operand: an array to which the scatter should be applied
        scatter_indices: an array that gives the indices in `operand` to which each
            update in `updates` should be applied.
        updates: the updates that should be scattered onto `operand`.
        dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
              how dimensions of `operand`, `start_indices`, `updates` and the output
              relate.
        indices_are_sorted: whether `scatter_indices` is known to be sorted. If
            true, may improve performance on some backends.
        unique_indices: whether the elements to be updated in ``operand`` are
              guaranteed to not overlap with each other. If true, may improve performance on
              some backends. JAX does not check this promise: if the updated elements
              overlap when ``unique_indices`` is ``True`` the behavior is undefined.
        mode: how to handle indices that are out of bounds: when set to 'clip',
              indices are clamped so that the slice is within bounds, and when
              set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
              for out-of-bounds indices when set to 'promise_in_bounds' is
              implementation-defined.

    Returns:
        An array containing the sum of `operand` and the scattered updates.

    Examples:
        As mentioned above, you should basically never use :func:`scatter` directly,
        and instead perform scatter-style operations using NumPy-style indexing
        expressions via :attr:`jax.numpy.ndarray.at`.

        Here is and example of updating entries in an array using :attr:`jax.numpy.ndarray.at`,
        which lowers to an XLA Scatter operation:

        >>> x = jnp.zeros(5)
        >>> indices = jnp.array([1, 2, 4])
        >>> values = jnp.array([2.0, 3.0, 4.0])

        >>> x.at[indices].set(values)
        Array([0., 2., 3., 0., 4.], dtype=float32)

        This syntax also supports several of the optional arguments to :func:`scatter`,
        for example:

        >>> x.at[indices].set(values, indices_are_sorted=True, mode='promise_in_bounds')
        Array([0., 2., 3., 0., 4.], dtype=float32)

        By comparison, here is the equivalent function call using :func:`scatter` directly,
        which is not something typical users should ever need to do:

        >>> lax.scatter(x, indices[:, None], values,
        ...             dimension_numbers=lax.ScatterDimensionNumbers(
        ...                 update_window_dims=(),
        ...                 inserted_window_dims=(0,),
        ...                 scatter_dims_to_operand_dims=(0,)),
        ...             indices_are_sorted=True,
        ...             mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS)
        Array([0., 2., 3., 0., 4.], dtype=float32)
    """
    return _fun_lax_scatter(lax.scatter, operand, scatter_indices, updates, dimension_numbers, indices_are_sorted,
                            unique_indices, mode)


@set_module_as('brainunit.math')
def scatter_add(
    operand: Union[Quantity, jax.typing.ArrayLike],
    scatter_indices: jax.typing.ArrayLike,
    updates: jax.typing.ArrayLike,
    dimension_numbers: jax.lax.ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    """Scatter-add operator.

    Wraps `XLA's Scatter operator
    <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
    addition is used to combine updates and values from `operand`.

    The semantics of scatter are complicated, and its API might change in the
    future. For most use cases, you should prefer the
    :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
    the familiar NumPy indexing syntax.

    Args:
        operand: an array to which the scatter should be applied
        scatter_indices: an array that gives the indices in `operand` to which each
            update in `updates` should be applied.
        updates: the updates that should be scattered onto `operand`.
        dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
              how dimensions of `operand`, `scatter_indices`, `updates` and the output
              relate.
        indices_are_sorted: whether `scatter_indices` is known to be sorted. If
            true, may improve performance on some backends.
        unique_indices: whether the elements to be updated in ``operand`` are
              guaranteed to not overlap with each other. If true, may improve performance on
              some backends. JAX does not check this promise: if the updated elements
              overlap when ``unique_indices`` is ``True`` the behavior is undefined.
        mode: how to handle indices that are out of bounds: when set to 'clip',
              indices are clamped so that the slice is within bounds, and when
              set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
              for out-of-bounds indices when set to 'promise_in_bounds' is
              implementation-defined.

    Returns:
        An array containing the sum of `operand` and the scattered updates.
    """
    return _fun_lax_scatter(lax.scatter_add, operand, scatter_indices, updates, dimension_numbers, indices_are_sorted,
                            unique_indices, mode)


@set_module_as('brainunit.math')
def scatter_sub(
    operand: Union[Quantity, jax.typing.ArrayLike],
    scatter_indices: jax.typing.ArrayLike,
    updates: jax.typing.ArrayLike,
    dimension_numbers: jax.lax.ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    """Scatter-sub operator.

    Wraps `XLA's Scatter operator
    <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
    subtraction is used to combine updates and values from `operand`.

    The semantics of scatter are complicated, and its API might change in the
    future. For most use cases, you should prefer the
    :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
    the familiar NumPy indexing syntax.

    Args:
        operand: an array to which the scatter should be applied
        scatter_indices: an array that gives the indices in `operand` to which each
            update in `updates` should be applied.
        updates: the updates that should be scattered onto `operand`.
        dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
              how dimensions of `operand`, `scatter_indices`, `updates` and the output
              relate.
        indices_are_sorted: whether `scatter_indices` is known to be sorted. If
            true, may improve performance on some backends.
        unique_indices: whether the elements to be updated in ``operand`` are
              guaranteed to not overlap with each other. If true, may improve performance on
              some backends. JAX does not check this promise: if the updated elements
              overlap when ``unique_indices`` is ``True`` the behavior is undefined.
        mode: how to handle indices that are out of bounds: when set to 'clip',
              indices are clamped so that the slice is within bounds, and when
              set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
              for out-of-bounds indices when set to 'promise_in_bounds' is
              implementation-defined.

    Returns:
        An array containing the sum of `operand` and the scattered updates.
    """
    return _fun_lax_scatter(lax.scatter_sub, operand, scatter_indices, updates, dimension_numbers, indices_are_sorted,
                            unique_indices, mode)


@set_module_as('brainunit.math')
def scatter_mul(
    operand: Union[Quantity, jax.typing.ArrayLike],
    scatter_indices: jax.typing.ArrayLike,
    updates: jax.typing.ArrayLike,
    dimension_numbers: jax.lax.ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    """Scatter-multiply operator.

    Wraps `XLA's Scatter operator
    <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
    multiplication is used to combine updates and values from `operand`.

    The semantics of scatter are complicated, and its API might change in the
    future. For most use cases, you should prefer the
    :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
    the familiar NumPy indexing syntax.

    Args:
        operand: an array to which the scatter should be applied
        scatter_indices: an array that gives the indices in `operand` to which each
            update in `updates` should be applied.
        updates: the updates that should be scattered onto `operand`.
        dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
              how dimensions of `operand`, `scatter_indices`, `updates` and the output
              relate.
        indices_are_sorted: whether `scatter_indices` is known to be sorted. If
            true, may improve performance on some backends.
        unique_indices: whether the elements to be updated in ``operand`` are
              guaranteed to not overlap with each other. If true, may improve performance on
              some backends. JAX does not check this promise: if the updated elements
              overlap when ``unique_indices`` is ``True`` the behavior is undefined.
        mode: how to handle indices that are out of bounds: when set to 'clip',
              indices are clamped so that the slice is within bounds, and when
              set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
              for out-of-bounds indices when set to 'promise_in_bounds' is
              implementation-defined.

    Returns:
        An array containing the sum of `operand` and the scattered updates.
    """
    return _fun_lax_scatter(lax.scatter_mul, operand, scatter_indices, updates, dimension_numbers, indices_are_sorted,
                            unique_indices, mode)


@set_module_as('brainunit.math')
def scatter_min(
    operand: Union[Quantity, jax.typing.ArrayLike],
    scatter_indices: jax.typing.ArrayLike,
    updates: jax.typing.ArrayLike,
    dimension_numbers: jax.lax.ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    """Scatter-min operator.

    Wraps `XLA's Scatter operator
    <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
    the `min` function  is used to combine updates and values from `operand`.

    The semantics of scatter are complicated, and its API might change in the
    future. For most use cases, you should prefer the
    :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
    the familiar NumPy indexing syntax.

    Args:
        operand: an array to which the scatter should be applied
        scatter_indices: an array that gives the indices in `operand` to which each
            update in `updates` should be applied.
        updates: the updates that should be scattered onto `operand`.
        dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
              how dimensions of `operand`, `scatter_indices`, `updates` and the output
              relate.
        indices_are_sorted: whether `scatter_indices` is known to be sorted. If
            true, may improve performance on some backends.
        unique_indices: whether the elements to be updated in ``operand`` are
              guaranteed to not overlap with each other. If true, may improve performance on
              some backends. JAX does not check this promise: if the updated elements
              overlap when ``unique_indices`` is ``True`` the behavior is undefined.
        mode: how to handle indices that are out of bounds: when set to 'clip',
              indices are clamped so that the slice is within bounds, and when
              set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
              for out-of-bounds indices when set to 'promise_in_bounds' is
              implementation-defined.

    Returns:
        An array containing the sum of `operand` and the scattered updates.
    """
    return _fun_lax_scatter(lax.scatter_min, operand, scatter_indices, updates, dimension_numbers, indices_are_sorted,
                            unique_indices, mode)


@set_module_as('brainunit.math')
def scatter_max(
    operand: Union[Quantity, jax.typing.ArrayLike],
    scatter_indices: jax.typing.ArrayLike,
    updates: jax.typing.ArrayLike,
    dimension_numbers: jax.lax.ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    """Scatter-max operator.

    Wraps `XLA's Scatter operator
    <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where
    the `max` function  is used to combine updates and values from `operand`.

    The semantics of scatter are complicated, and its API might change in the
    future. For most use cases, you should prefer the
    :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
    the familiar NumPy indexing syntax.

    Args:
        operand: an array to which the scatter should be applied
        scatter_indices: an array that gives the indices in `operand` to which each
            update in `updates` should be applied.
        updates: the updates that should be scattered onto `operand`.
        dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
              how dimensions of `operand`, `scatter_indices`, `updates` and the output
              relate.
        indices_are_sorted: whether `scatter_indices` is known to be sorted. If
            true, may improve performance on some backends.
        unique_indices: whether the elements to be updated in ``operand`` are
              guaranteed to not overlap with each other. If true, may improve performance on
              some backends. JAX does not check this promise: if the updated elements
              overlap when ``unique_indices`` is ``True`` the behavior is undefined.
        mode: how to handle indices that are out of bounds: when set to 'clip',
              indices are clamped so that the slice is within bounds, and when
              set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
              for out-of-bounds indices when set to 'promise_in_bounds' is
              implementation-defined.

    Returns:
        An array containing the sum of `operand` and the scattered updates.
    """
    return _fun_lax_scatter(lax.scatter_max, operand, scatter_indices, updates, dimension_numbers, indices_are_sorted,
                            unique_indices, mode)


@set_module_as('brainunit.math')
def scatter_apply(
    operand: Union[Quantity, jax.typing.ArrayLike],
    scatter_indices: jax.typing.ArrayLike,
    func: Callable,
    dimension_numbers: jax.lax.ScatterDimensionNumbers,
    *,
    update_shape: Shape = (),
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | jax.lax.GatherScatterMode | None = None
) -> Union[Quantity, jax.Array]:
    """Scatter-apply operator.

    Wraps `XLA's Scatter operator
    <https://www.tensorflow.org/xla/operation_semantics#scatter>`_, where values
    from ``operand`` are replaced with ``func(operand)``, with duplicate indices
    resulting in multiple applications of ``func``.

    The semantics of scatter are complicated, and its API might change in the
    future. For most use cases, you should prefer the
    :attr:`jax.numpy.ndarray.at` property on JAX arrays which uses
    the familiar NumPy indexing syntax.

    Note that in the current implementation, ``scatter_apply`` is not compatible
    with automatic differentiation.

    Args:
        operand: an array to which the scatter should be applied
        scatter_indices: an array that gives the indices in `operand` to which each
            update in `updates` should be applied.
        func: unary function that will be applied at each index.
        dimension_numbers: a `lax.ScatterDimensionNumbers` object that describes
              how dimensions of `operand`, `start_indices`, `updates` and the output
              relate.
        update_shape: the shape of the updates at the given indices.
        indices_are_sorted: whether `scatter_indices` is known to be sorted. If
            true, may improve performance on some backends.
        unique_indices: whether the elements to be updated in ``operand`` are
              guaranteed to not overlap with each other. If true, may improve performance on
              some backends. JAX does not check this promise: if the updated elements
              overlap when ``unique_indices`` is ``True`` the behavior is undefined.
        mode: how to handle indices that are out of bounds: when set to 'clip',
              indices are clamped so that the slice is within bounds, and when
              set to 'fill' or 'drop' out-of-bounds updates are dropped. The behavior
              for out-of-bounds indices when set to 'promise_in_bounds' is
              implementation-defined.

    Returns:
        An array containing the result of applying `func` to `operand` at the given indices.
    """
    if isinstance(operand, Quantity):
        return maybe_decimal(Quantity(lax.scatter_apply(operand.mantissa, scatter_indices, func, dimension_numbers,
                                                        update_shape=update_shape,
                                                        indices_are_sorted=indices_are_sorted,
                                                        unique_indices=unique_indices,
                                                        mode=mode), unit=operand.unit))
    else:
        return lax.scatter_apply(operand, scatter_indices, func, dimension_numbers,
                                 update_shape=update_shape,
                                 indices_are_sorted=indices_are_sorted,
                                 unique_indices=unique_indices,
                                 mode=mode)


# math funcs keep unit (binary)
@set_module_as('brainunit.math')
def complex(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    r"""Elementwise make complex number: :math:`x + jy`.

    Builds a complex number from real and imaginary parts.
    """
    return _fun_keep_unit_binary(lax.complex, x, y)


@set_module_as('brainunit.math')
def pad(
    operand: Union[Quantity, jax.typing.ArrayLike],
    padding_value: Union[Quantity, jax.typing.ArrayLike],
    padding_config: Sequence[tuple[int, int, int]]
) -> Union[Quantity, jax.Array]:
    """Applies low, high, and/or interior padding to an array.

    Wraps XLA's `Pad
    <https://www.tensorflow.org/xla/operation_semantics#pad>`_
    operator.

    Args:
        operand: an array to be padded.
        padding_value: the value to be inserted as padding. Must have the same dtype
            as ``operand``.
        padding_config: a sequence of ``(low, high, interior)`` tuples of integers,
              giving the amount of low, high, and interior (dilation) padding to insert
              in each dimension.

    Returns:
        The ``operand`` array with padding value ``padding_value`` inserted in each
        dimension according to the ``padding_config``.
    """
    return _fun_keep_unit_binary(lax.pad, operand, padding_value, padding_config)


@set_module_as('brainunit.math')
def sub(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    r"""Elementwise subtraction: :math:`x - y`."""
    return _fun_keep_unit_binary(lax.sub, x, y)


# type conversion
@set_module_as('brainunit.math')
def convert_element_type(
    operand: Union[Quantity, jax.typing.ArrayLike],
    new_dtype: jax.typing.DTypeLike
) -> Union[Quantity, jax.Array]:
    """Elementwise cast.

    Wraps XLA's `ConvertElementType
    <https://www.tensorflow.org/xla/operation_semantics#convertelementtype>`_
    operator, which performs an elementwise conversion from one type to another.
    Similar to a C++ `static_cast`.

    Args:
        operand: an array or scalar value to be cast.
        new_dtype: a NumPy dtype representing the target type.

    Returns:
        An array with the same shape as `operand`, cast elementwise to `new_dtype`.
    """
    return _fun_keep_unit_unary(lax.convert_element_type, operand, new_dtype)


@set_module_as('brainunit.math')
def bitcast_convert_type(
    operand: Union[Quantity, jax.typing.ArrayLike],
    new_dtype: jax.typing.DTypeLike
) -> Union[Quantity, jax.Array]:
    """Elementwise bitcast.

    Wraps XLA's `BitcastConvertType
    <https://www.tensorflow.org/xla/operation_semantics#bitcastconverttype>`_
    operator, which performs a bit cast from one type to another.

    The output shape depends on the size of the input and output dtypes with
    the following logic::

    if new_dtype.itemsize == operand.dtype.itemsize:
        output_shape = operand.shape
    if new_dtype.itemsize < operand.dtype.itemsize:
        output_shape = (*operand.shape, operand.dtype.itemsize // new_dtype.itemsize)
    if new_dtype.itemsize > operand.dtype.itemsize:
          assert operand.shape[-1] * operand.dtype.itemsize == new_dtype.itemsize
          output_shape = operand.shape[:-1]

    Args:
        operand: an array or scalar value to be cast
        new_dtype: the new type. Should be a NumPy type.

    Returns:
        An array of shape `output_shape` (see above) and type `new_dtype`,
        constructed from the same bits as operand.
    """
    return _fun_keep_unit_unary(lax.bitcast_convert_type, operand, new_dtype)


# math funcs keep unit (n-ary)
@set_module_as('brainunit.math')
def clamp(
    min: Union[Quantity, jax.typing.ArrayLike],
    x: Union[Quantity, jax.typing.ArrayLike],
    max: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    r"""Elementwise clamp.

    Returns :math:`\mathrm{clamp}(x) = \begin{cases}
    \mathit{min} & \text{if } x < \mathit{min},\\
    \mathit{max} & \text{if } x > \mathit{max},\\
    x & \text{otherwise}
    \end{cases}`.
    """
    if all(isinstance(i, Quantity) for i in (min, x, max)):
        unit = min.unit
        return maybe_decimal(Quantity(lax.clamp(min.mantissa, x.to_decimal(unit), max.to_decimal(unit)), unit=unit))
    elif all(isinstance(i, (jax.Array, np.ndarray, np.bool_, np.number, bool, int, float, builtins.complex)) for i in
             (min, x, max)):
        return lax.clamp(min, x, max)
    else:
        raise AssertionError('All inputs must be Quantity or jax.typing.ArrayLike')


# math funcs keep unit (return Quantity and index)
@set_module_as('brainunit.math')
def approx_max_k(
    operand: Union[Quantity, jax.typing.ArrayLike],
    k: int,
    reduction_dimension: int = -1,
    recall_target: float = 0.95,
    reduction_input_size_override: int = -1,
    aggregate_to_topk: bool = True
) -> tuple[Union[Quantity, jax.Array], jax.typing.ArrayLike]:
    """Returns max ``k`` values and their indices of the ``operand`` in an approximate manner.

    See https://arxiv.org/abs/2206.14286 for the algorithm details.

    Args:
        operand : Array to search for max-k. Must be a floating number type.
        k : Specifies the number of max-k.
        reduction_dimension : Integer dimension along which to search. Default: -1.
        recall_target : Recall target for the approximation.
        reduction_input_size_override : When set to a positive value, it overrides
              the size determined by ``operand[reduction_dim]`` for evaluating the
              recall. This option is useful when the given ``operand`` is only a subset
              of the overall computation in SPMD or distributed pipelines, where the
              true input size cannot be deferred by the operand shape.
        aggregate_to_topk : When true, aggregates approximate results to the top-k
              in sorted order. When false, returns the approximate results unsorted. In
              this case, the number of the approximate results is implementation defined
              and is greater or equal to the specified ``k``.

    Returns:
        Tuple of two arrays. The arrays are the max ``k`` values and the
        corresponding indices along the ``reduction_dimension`` of the input
        ``operand``. The arrays' dimensions are the same as the input ``operand``
        except for the ``reduction_dimension``: when ``aggregate_to_topk`` is true,
        the reduction dimension is ``k``; otherwise, it is greater equals to ``k``
        where the size is implementation-defined.

    We encourage users to wrap ``approx_max_k`` with jit. See the following
    example for maximal inner production search (MIPS):

    >>> import functools
    >>> import jax
    >>> import numpy as np
    >>> @functools.partial(jax.jit, static_argnames=["k", "recall_target"])
    ... def mips(qy, db, k=10, recall_target=0.95):
    ...   dists = jax.lax.dot(qy, db.transpose())
    ...   # returns (f32[qy_size, k], i32[qy_size, k])
    ...   return jax.lax.approx_max_k(dists, k=k, recall_target=recall_target)
    >>>
    >>> qy = jax.numpy.array(np.random.rand(50, 64))
    >>> db = jax.numpy.array(np.random.rand(1024, 64))
    >>> dot_products, neighbors = mips(qy, db, k=10)
    """
    if isinstance(operand, Quantity):
        r = lax.approx_max_k(operand.mantissa, k, reduction_dimension, recall_target, reduction_input_size_override,
                             aggregate_to_topk)
        return maybe_decimal(Quantity(r[0], unit=operand.unit)), r[1]
    return lax.approx_max_k(operand, k, reduction_dimension, recall_target, reduction_input_size_override,
                            aggregate_to_topk)


@set_module_as('brainunit.math')
def approx_min_k(
    operand: Union[Quantity, jax.typing.ArrayLike],
    k: int,
    reduction_dimension: int = -1,
    recall_target: float = 0.95,
    reduction_input_size_override: int = -1,
    aggregate_to_topk: bool = True
) -> tuple[Union[Quantity, jax.Array], jax.typing.ArrayLike]:
    """Returns min ``k`` values and their indices of the ``operand`` in an approximate manner.

    See https://arxiv.org/abs/2206.14286 for the algorithm details.

    Args:
        operand : Array to search for min-k. Must be a floating number type.
        k : Specifies the number of min-k.
        reduction_dimension: Integer dimension along which to search. Default: -1.
        recall_target: Recall target for the approximation.
        reduction_input_size_override : When set to a positive value, it overrides
              the size determined by ``operand[reduction_dim]`` for evaluating the
              recall. This option is useful when the given operand is only a subset of
              the overall computation in SPMD or distributed pipelines, where the true
              input size cannot be deferred by the ``operand`` shape.
        aggregate_to_topk : When true, aggregates approximate results to the top-k
              in sorted order. When false, returns the approximate results unsorted. In
              this case, the number of the approximate results is implementation defined
              and is greater or equal to the specified ``k``.

    Returns:
        Tuple of two arrays. The arrays are the least ``k`` values and the
        corresponding indices along the ``reduction_dimension`` of the input
        ``operand``.  The arrays' dimensions are the same as the input ``operand``
        except for the ``reduction_dimension``: when ``aggregate_to_topk`` is true,
        the reduction dimension is ``k``; otherwise, it is greater equals to ``k``
        where the size is implementation-defined.

    We encourage users to wrap ``approx_min_k`` with jit. See the following example
    for nearest neighbor search over the squared l2 distance:

    >>> import functools
    >>> import jax
    >>> import numpy as np
    >>> @functools.partial(jax.jit, static_argnames=["k", "recall_target"])
    ... def l2_ann(qy, db, half_db_norms, k=10, recall_target=0.95):
    ...   dists = half_db_norms - jax.lax.dot(qy, db.transpose())
    ...   return jax.lax.approx_min_k(dists, k=k, recall_target=recall_target)
    >>>
    >>> qy = jax.numpy.array(np.random.rand(50, 64))
    >>> db = jax.numpy.array(np.random.rand(1024, 64))
    >>> half_db_norm_sq = jax.numpy.linalg.norm(db, axis=1)**2 / 2
    >>> dists, neighbors = l2_ann(qy, db, half_db_norm_sq, k=10)

    In the example above, we compute ``db^2/2 - dot(qy, db^T)`` instead of
    ``qy^2 - 2 dot(qy, db^T) + db^2`` for performance reason. The former uses less
    arithmetic and produces the same set of neighbors.
    """
    if isinstance(operand, Quantity):
        r = lax.approx_min_k(operand.mantissa, k, reduction_dimension, recall_target, reduction_input_size_override,
                             aggregate_to_topk)
        return maybe_decimal(Quantity(r[0], unit=operand.unit)), r[1]
    return lax.approx_min_k(operand, k, reduction_dimension, recall_target, reduction_input_size_override,
                            aggregate_to_topk)


@set_module_as('brainunit.math')
def top_k(
    operand: Union[Quantity, jax.typing.ArrayLike],
    k: int
) -> tuple[Union[Quantity, jax.Array], jax.typing.ArrayLike]:
    """Returns top ``k`` values and their indices along the last axis of ``operand``.

    Args:
        operand: N-dimensional array of non-complex type.
        k: integer specifying the number of top entries.

    Returns:
        A tuple ``(values, indices)`` where

        - ``values`` is an array containing the top k values along the last axis.
        - ``indices`` is an array containing the indices corresponding to values.

    Examples:
        Find the largest three values, and their indices, within an array:

        >>> x = jnp.array([9., 3., 6., 4., 10.])
        >>> values, indices = jax.lax.top_k(x, 3)
        >>> values
        Array([10.,  9.,  6.], dtype=float32)
        >>> indices
        Array([4, 0, 2], dtype=int32)
    """
    if isinstance(operand, Quantity):
        r = lax.top_k(operand.mantissa, k)
        return maybe_decimal(Quantity(r[0], unit=operand.unit)), r[1]
    return lax.top_k(operand, k)


# broadcasting arrays
def broadcast(
    operand: Union[Quantity, jax.typing.ArrayLike],
    sizes: Sequence[int]
) -> Union[Quantity, jax.Array]:
    """Broadcasts an array, adding new leading dimensions

    Args:
        operand: an array
        sizes: a sequence of integers, giving the sizes of new leading dimensions
            to add to the front of the array.

    Returns:
        An array containing the result.
    """
    return _fun_keep_unit_unary(lax.broadcast, operand, sizes)


def broadcast_in_dim(
    operand: Union[Quantity, jax.typing.ArrayLike],
    shape: Shape,
    broadcast_dimensions: Sequence[int]
) -> Union[Quantity, jax.Array]:
    """Wraps XLA's `BroadcastInDim
    <https://www.tensorflow.org/xla/operation_semantics#broadcastindim>`_
    operator.

    Args:
        operand: an array
        shape: the shape of the target array
        broadcast_dimensions: to which dimension in the target shape each dimension
              of the operand shape corresponds to.  That is, dimension i of the operand
              becomes dimension broadcast_dimensions[i] of the result.

    Returns:
        An array containing the result.
    """
    return _fun_keep_unit_unary(lax.broadcast_in_dim, operand, shape, broadcast_dimensions)


def broadcast_to_rank(
    x: Union[Quantity, jax.typing.ArrayLike],
    rank: int
) -> Union[Quantity, jax.Array]:
    """Adds leading dimensions of ``1`` to give ``x`` rank ``rank``."""
    return _fun_keep_unit_unary(lax.broadcast_to_rank, x, rank)
