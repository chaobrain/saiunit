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
from jax._src.numpy.util import promote_dtypes as _promote_dtypes
import numpy as np

from ._fun_array_creation import asarray
from .._base import (
    Quantity,
    fail_for_dimension_mismatch,
    get_unit,
    split_mantissa_unit,
    UNITLESS,
    unit_scale_align_to_first,
    maybe_decimal
)
from .._misc import set_module_as

__all__ = [
    # sequence inputs
    'row_stack', 'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack', 'block', 'append',

    # sequence outputs
    'split', 'array_split', 'dsplit', 'hsplit', 'vsplit',

    # broadcasting arrays
    'atleast_1d', 'atleast_2d', 'atleast_3d', 'broadcast_arrays', 'broadcast_to',

    # array manipulation
    'reshape', 'moveaxis', 'transpose', 'swapaxes', 'tile', 'repeat',
    'flip', 'fliplr', 'flipud', 'roll', 'expand_dims', 'squeeze',
    'sort', 'max', 'min', 'amax', 'amin', 'diagflat', 'diagonal', 'choose', 'ravel',
    'flatten', 'unflatten', 'remove_diag',

    # math funcs keep unit (unary)
    'astype', 'real', 'imag', 'conj', 'conjugate', 'negative', 'positive',
    'abs', 'sum', 'nancumsum', 'nansum',
    'cumsum', 'ediff1d', 'absolute', 'fabs', 'median',
    'nanmin', 'nanmax', 'ptp', 'average', 'mean', 'std',
    'nanmedian', 'nanmean', 'nanstd', 'diff', 'rot90', 'intersect1d', 'nan_to_num',
    'percentile', 'nanpercentile', 'quantile', 'nanquantile',

    # math funcs only accept unitless (unary) can return Quantity
    'round', 'around', 'rint', 'floor', 'ceil', 'trunc', 'fix', 'modf',

    # math funcs keep unit (binary)
    'fmod', 'mod', 'copysign', 'remainder',
    'maximum', 'minimum', 'fmax', 'fmin', 'lcm', 'gcd', 'trace',
    'add', 'subtract', 'nextafter', 'promote_dtypes',

    # math funcs keep unit
    'interp', 'clip', 'histogram',

    # selection
    'compress', 'extract', 'take', 'select', 'where', 'unique',
]


# -------------------------------------------------------------------


def _fun_keep_unit_sequence(
    func,
    *args,
    **kwargs
):
    leaves, treedef = jax.tree.flatten(args, is_leaf=lambda x: isinstance(x, Quantity))
    # leaves = jax.tree.map(
    #     lambda x: x.factorless() if isinstance(x, Quantity) else x,
    #     leaves,
    #     is_leaf=lambda x: isinstance(x, Quantity)
    # )
    leaves = unit_scale_align_to_first(*leaves)
    unit = leaves[0].unit
    leaves = [x.mantissa for x in leaves]
    args = treedef.unflatten(leaves)
    r = func(*args, **kwargs)
    if unit.is_unitless:
        return r
    return Quantity(r, unit=unit)


@set_module_as('saiunit.math')
def concatenate(
    arrays: Union[Sequence[jax.typing.ArrayLike], Sequence[Quantity]],
    axis: Optional[int] = None,
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[jax.Array, Quantity]:
    """
    Join a sequence of quantities or arrays along an existing axis.

    Parameters
    ----------
    arrays : sequence of array_like, Quantity
      The arrays must have the same shape, except in the dimension corresponding
      to `axis` (the first, by default).
    axis : int, optional
      The axis along which the arrays will be joined.  Default is 0.
    dtype : dtype, optional
      If provided, the concatenation will be done using this dtype. Otherwise, the
      array with the highest precision will be used.

    Returns
    -------
    res : ndarray, Quantity
      The concatenated array. The type of the array is the same as that of the
      first array passed in.
    """
    return _fun_keep_unit_sequence(jnp.concatenate, arrays, axis=axis, dtype=dtype)


@set_module_as('saiunit.math')
def stack(
    arrays: Union[Sequence[jax.typing.ArrayLike], Sequence[Quantity]],
    axis: int = 0,
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[jax.Array, Quantity]:
    """
    Join a sequence of quantities or arrays along a new axis.

    Parameters
    ----------
    arrays : sequence of array_like, Quantity
      The arrays must have the same shape.
    axis : int, optional
      The axis in the result array along which the input arrays are stacked.
    dtype : dtype, optional
      If provided, the concatenation will be done using this dtype. Otherwise, the
      array with the highest precision will be used.

    Returns
    -------
    res : ndarray, Quantity
      The stacked array has one more dimension than the input arrays.
    """
    return _fun_keep_unit_sequence(jnp.stack, arrays, axis=axis, dtype=dtype)


@set_module_as('saiunit.math')
def vstack(
    tup: Union[Sequence[jax.typing.ArrayLike], Sequence[Quantity]],
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[jax.Array, Quantity]:
    """
    Stack quantities or arrays in sequence vertically (row wise).

    Parameters
    ----------
    tup : sequence of array_like, Quantity
      The arrays must have the same shape along all but the first axis.
    dtype : dtype, optional
      If provided, the concatenation will be done using this dtype. Otherwise, the
      array with the highest precision will be used.

    Returns
    -------
    res : ndarray, Quantity
      The array formed by stacking the given arrays.
    """
    return _fun_keep_unit_sequence(jnp.vstack, tup, dtype=dtype)


row_stack = vstack


@set_module_as('saiunit.math')
def hstack(
    arrays: Union[Sequence[jax.typing.ArrayLike], Sequence[Quantity]],
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[jax.Array, Quantity]:
    """
    Stack quantities arrays in sequence horizontally (column wise).

    Parameters
    ----------
    arrays : sequence of array_like, Quantity
      The arrays must have the same shape along all but the second axis.
    dtype : dtype, optional
      If provided, the concatenation will be done using this dtype. Otherwise, the
      array with the highest precision will be used.

    Returns
    -------
    res : ndarray, Quantity
      The array formed by stacking the given arrays.
    """
    return _fun_keep_unit_sequence(jnp.hstack, arrays, dtype=dtype)


@set_module_as('saiunit.math')
def dstack(
    arrays: Union[Sequence[jax.typing.ArrayLike], Sequence[Quantity]],
    dtype: Optional[jax.typing.DTypeLike] = None
) -> Union[jax.Array, Quantity]:
    """
    Stack quantities or arrays in sequence depth wise (along third axis).

    Parameters
    ----------
    arrays : sequence of array_like, Quantity
      The arrays must have the same shape along all but the third axis.
    dtype : dtype, optional
      If provided, the concatenation will be done using this dtype. Otherwise, the
      array with the highest precision will be used.

    Returns
    -------
    res : ndarray, Quantity
      The array formed by stacking the given arrays.
    """
    return _fun_keep_unit_sequence(jnp.dstack, arrays, dtype=dtype)


@set_module_as('saiunit.math')
def column_stack(
    tup: Union[Sequence[jax.typing.ArrayLike], Sequence[Quantity]]
) -> Union[jax.Array, Quantity]:
    """
    Stack 1-D arrays as columns into a 2-D array.

    Take a sequence of 1-D arrays and stack them as columns to make a single
    2-D array. 2-D arrays are stacked as-is, just like with hstack.

    Parameters
    ----------
    tup : sequence of 1-D array_like, Quantity
      1-D arrays to stack as columns.

    Returns
    -------
    res : ndarray, Quantity
      The array formed by stacking the given arrays.
    """
    return _fun_keep_unit_sequence(jnp.column_stack, tup)


@set_module_as('saiunit.math')
def block(
    arrays: Sequence[Union[jax.Array, Quantity]]
) -> Union[jax.Array, Quantity]:
    """
    Assemble a quantity or an array from nested lists of blocks.

    Parameters
    ----------
    arrays : sequence of array_like, Quantity
      Each element in `arrays` can itself be a nested sequence of arrays, in which case the blocks in the corresponding
      cells are recursively stacked as the elements of the resulting array.

    Returns
    -------
    res : ndarray, Quantity
      The array constructed from the given blocks.
    """
    return _fun_keep_unit_sequence(jnp.block, arrays)


@set_module_as('saiunit.math')
def append(
    arr: Union[jax.Array, Quantity],
    values: Union[jax.Array, Quantity],
    axis: Optional[int] = None
) -> Union[jax.Array, Quantity]:
    """
    Append values to the end of a quantity or an array.

    Parameters
    ----------
    arr : array_like, Quantity
      Values are appended to a copy of this array.
    values : array_like, Quantity
      These values are appended to a copy of `arr`.
      It must be of the correct shape (the same shape as `arr`, excluding `axis`).
    axis : int, optional
      The axis along which `values` are appended. If `axis` is None, `values` is flattened before use.

    Returns
    -------
    res : ndarray, Quantity
      A copy of `arr` with `values` appended to `axis`. Note that `append` does not occur in-place:
      a new array is allocated and filled.
    """
    return _fun_keep_unit_sequence(jnp.append, arr, values, axis=axis)


def _fun_keep_unit_return_sequence(
    func,
    x: jax.typing.ArrayLike | Quantity,
    *args,
    **kwargs
):
    if isinstance(x, Quantity):
        r = func(x.mantissa, *args, **kwargs)
        return [maybe_decimal(Quantity(rr, unit=x.unit)) for rr in r]
    return func(x, *args, **kwargs)


@set_module_as('saiunit.math')
def split(
    a: Union[jax.Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]],
    axis: int = 0
) -> Union[Sequence[jax.Array | Quantity]]:
    """
    Split quantity or array into a list of multiple sub-arrays.

    Parameters
    ----------
    a : array_like, Quantity
      Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
      If `indices_or_sections` is an integer, N, the array will be divided into
      N equal arrays along `axis`. If such a split is not possible, an error is
      raised. If `indices_or_sections` is a 1-D array of sorted integers, the
      entries indicate where along `axis` the array is split. For example,
      `[2, 3]` would, for `axis=0`, result in
      - `a[:2]`
      - `a[2:3]`
      - `a[3:]`
    axis : int, optional
      The axis along which to split, default is 0.

    Returns
    -------
    res : list of ndarrays, Quantity
      A list of sub-arrays.
    """
    return _fun_keep_unit_return_sequence(jnp.split, a, indices_or_sections=indices_or_sections, axis=axis)


@set_module_as('saiunit.math')
def array_split(
    ary: Union[Quantity, jax.typing.ArrayLike],
    indices_or_sections: Union[int, jax.typing.ArrayLike],
    axis: Optional[int] = 0
) -> Union[Sequence[Quantity | jax.Array]]:
    """
    Split an array into multiple sub-arrays.

    Parameters
    ----------
    ary : Quantity or array
      Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
      If `indices_or_sections` is an integer, `ary` is divided into `indices_or_sections` sub-arrays along `axis`.
      If such a split is not possible, an error is raised.
      If `indices_or_sections` is a 1-D array of sorted integers, the entries indicate where along `axis` the array is split.
    axis : int, optional
      The axis along which to split, default is 0.

    Returns
    -------
    sub-arrays : list of Quantity or list of array
      A list of sub-arrays.
    """
    return _fun_keep_unit_return_sequence(jnp.split, ary, indices_or_sections=indices_or_sections, axis=axis)


@set_module_as('saiunit.math')
def dsplit(
    a: Union[jax.Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]]
) -> Union[Sequence[jax.Array | Quantity]]:
    """
    Split a quantity or an array into multiple sub-arrays along the 3rd axis (depth).

    Parameters
    ----------
    a : array_like, Quantity
      Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
      If `indices_or_sections` is an integer, N, the array will be divided into
      N equal arrays along the third axis (depth). If such a split is not possible,
      an error is raised. If `indices_or_sections` is a 1-D array of sorted integers,
      the entries indicate where along the third axis the array is split.

    Returns
    -------
    res : list of ndarrays, Quantity
      A list of sub-arrays.
    """
    return _fun_keep_unit_return_sequence(jnp.dsplit, a, indices_or_sections)


@set_module_as('saiunit.math')
def hsplit(
    a: Union[jax.Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]]
) -> Union[Sequence[jax.Array | Quantity]]:
    """
    Split a quantity or an array into multiple sub-arrays horizontally (column-wise).

    Parameters
    ----------
    a : array_like, Quantity
      Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
      If `indices_or_sections` is an integer, N, the array will be divided into
      N equal arrays along the second axis. If such a split is not possible, an
      error is raised. If `indices_or_sections` is a 1-D array of sorted integers,
      the entries indicate where along the second axis the array is split.

    Returns
    -------
    res : list of ndarrays, Quantity
      A list of sub-arrays.
    """
    return _fun_keep_unit_return_sequence(jnp.hsplit, a, indices_or_sections)


@set_module_as('saiunit.math')
def vsplit(
    a: Union[jax.Array, Quantity],
    indices_or_sections: Union[int, Sequence[int]]
) -> Union[Sequence[jax.Array | Quantity]]:
    """
    Split a quantity or an array into multiple sub-arrays vertically (row-wise).

    Parameters
    ----------
    a : array_like, Quantity
      Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
      If `indices_or_sections` is an integer, N, the array will be divided into
      N equal arrays along the first axis. If such a split is not possible, an
      error is raised. If `indices_or_sections` is a 1-D array of sorted integers,
      the entries indicate where along the first axis the array is split.

    Returns
    -------
    res : list of ndarrays, Quantity
      A list of sub-arrays.
    """
    return _fun_keep_unit_return_sequence(jnp.vsplit, a, indices_or_sections)


# broadcasting arrays
# -------------------


def _broadcast_fun(func, *args, **kwargs):
    args = [asarray(x) for x in args]
    args, treedef = jax.tree.flatten(args)
    r = func(*args, **kwargs)
    r = treedef.unflatten([r] if isinstance(r, jax.Array) else r)
    if len(r) == 1:
        return r[0]
    return r


# more
# ----
@set_module_as('saiunit.math')
def broadcast_arrays(
    *args: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity | jax.Array | Sequence[jax.Array | Quantity]]:
    """
    Broadcast any number of arrays against each other.

    Parameters
    ----------
    `*args` : array_likes
        The arrays to broadcast.

    Returns
    -------
    broadcasted : list of arrays
        These arrays are views on the original arrays.  They are typically
        not contiguous.  Furthermore, more than one element of a
        broadcasted array may refer to a single memory location. If you need
        to write to the arrays, make copies first. While you can set the
        ``writable`` flag True, writing to a single output value may end up
        changing more than one location in the output array.
    """
    return _broadcast_fun(jnp.broadcast_arrays, *args)


@set_module_as('saiunit.math')
def promote_dtypes(
    *args: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity | jax.Array | Sequence[jax.Array | Quantity]]:
    """
    Promote the data types of the inputs to a common type.

    Parameters
    ----------
    `*args` : array_likes
        The arrays to promote.

    Returns
    -------
    promoted : list of arrays
        These arrays have the same shape as the input arrays, with the
        data type of the most precise input.
    """
    return _broadcast_fun(_promote_dtypes, *args)


@set_module_as('saiunit.math')
def broadcast_to(
    array: Union[Quantity, jax.typing.ArrayLike],
    shape: Tuple[int, ...]
) -> Quantity | jax.Array:
    """
    Broadcast an array to a new shape.

    Parameters
    ----------
    array : array_like
        The array to broadcast.
    shape : tuple or int
        The shape of the desired array. A single integer ``i`` is interpreted
        as ``(i,)``.

    Returns
    -------
    broadcast : array
        A readonly view on the original array with the given shape. It is
        typically not contiguous. Furthermore, more than one element of a
        broadcasted array may refer to a single memory location.

    Raises
    ------
    ValueError
        If the array is not compatible with the new shape according to NumPy's
        broadcasting rules.

    """
    return _fun_keep_unit_unary(jnp.broadcast_to, array, shape=shape)


@set_module_as('saiunit.math')
def atleast_1d(
    *arys: Union[jax.Array, Quantity]
) -> Union[Quantity | jax.Array | Sequence[jax.Array | Quantity]]:
    """
    View inputs as quantities or arrays with at least one dimension.

    Parameters
    ----------
    *arys : array_like, Quantity
      One or more input arrays or quantities.

    Returns
    -------
    res : ndarray, Quantity
      An array or a quantity, or a tuple of arrays or quantities, each with `a.ndim >= 1`.
    """
    return _broadcast_fun(jnp.atleast_1d, *arys)


@set_module_as('saiunit.math')
def atleast_2d(
    *arys: Union[jax.Array, Quantity]
) -> Union[Quantity | jax.Array | Sequence[jax.Array | Quantity]]:
    """
    View inputs as quantities or arrays with at least two dimensions.

    Parameters
    ----------
    *arys : array_like, Quantity
      One or more input arrays or quantities.

    Returns
    -------
    res : ndarray, Quantity
      An array or a quantity, or a tuple of arrays or quantities, each with `a.ndim >= 2`.
    """
    return _broadcast_fun(jnp.atleast_2d, *arys)


@set_module_as('saiunit.math')
def atleast_3d(
    *arys: Union[jax.Array, Quantity]
) -> Union[Quantity | jax.Array | Sequence[jax.Array | Quantity]]:
    """
    View inputs as quantities or arrays with at least three dimensions.

    Parameters
    ----------
    *arys : array_like, Quantity
      One or more input arrays or quantities.

    Returns
    -------
    res : ndarray, Quantity
      An array or a quantity, or a tuple of arrays or quantities, each with `a.ndim >= 3`.
    """
    return _broadcast_fun(jnp.atleast_3d, *arys)


# array manipulation
# ------------------


@set_module_as('saiunit.math')
def reshape(
    a: Union[jax.Array, Quantity],
    shape: Union[int, Tuple[int, ...]],
    order: str = 'C'
) -> Union[jax.Array, Quantity]:
    """
    Gives a new shape to a quantity or an array without changing its data.

    Parameters
    ----------
    a : array_like, Quantity
      Array to be reshaped.
    shape : int or tuple of ints
      The new shape should be compatible with the original shape. If
      an integer, then the result will be a 1-D array of that length.
      One shape dimension can be -1. In this case, the value is
      inferred from the length of the array and remaining dimensions.
    order : {'C', 'F', 'A'}, optional
      Read the elements of `a` using this index order, and place the
      elements into the reshaped array using this index order.  'C'
      means to read / write the elements using C-like index order,
      with the last axis index changing fastest, back to the first
      axis index changing slowest. 'F' means to read / write the
      elements using Fortran-like index order, with the first index
      changing fastest, and the last index changing slowest. Note that
      the 'C' and 'F' options take no account of the memory layout of
      the underlying array, and only refer to the order of indexing.
      'A' means to read / write the elements in Fortran-like index
      order if `a` is Fortran *contiguous* in memory, C-like order
      otherwise.

    Returns
    -------
    reshaped_array : ndarray, Quantity
      This will be a new view object if possible; otherwise, it will
      be a copy.  Note there is no guarantee of the *memory layout* (C- or
      Fortran- contiguous) of the returned array.
    """
    return _fun_keep_unit_unary(jnp.reshape, a, shape=shape, order=order)


@set_module_as('saiunit.math')
def moveaxis(
    a: Union[jax.Array, Quantity],
    source: Union[int, Tuple[int, ...]],
    destination: Union[int, Tuple[int, ...]]
) -> Union[jax.Array, Quantity]:
    """
    Moves axes of a quantity or an array to new positions.
    Other axes remain in their original order.

    Parameters
    ----------
    a : array_like, Quantity
      The array whose axes should be reordered.
    source : int or sequence of int
      Original positions of the axes to move. These must be unique.
    destination : int or sequence of int
      Destination positions for each of the original axes. These must also be
      unique.

    Returns
    -------
    result : ndarray, Quantity
      Array with moved axes. This array is a view of the input array.
    """
    return _fun_keep_unit_unary(jnp.moveaxis, a, source=source, destination=destination)


@set_module_as('saiunit.math')
def transpose(
    a: Union[jax.Array, Quantity],
    axes: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[jax.Array, Quantity]:
    """
    Permute the dimensions of a quantity or an array.

    Parameters
    ----------
    a : array_like, Quantity
      Input array.
    axes : list of ints, optional
      By default, reverse the dimensions, otherwise permute the axes
      according to the values given.

    Returns
    -------
    p : ndarray, Quantity
      `a` with its axes permuted.  A view is returned whenever
      possible.
    """
    return _fun_keep_unit_unary(jnp.transpose, a, axes=axes)


@set_module_as('saiunit.math')
def swapaxes(
    a: Union[jax.Array, Quantity],
    axis1: int,
    axis2: int
) -> Union[jax.Array, Quantity]:
    """
    Interchange two axes of a quantity or an array.

    Parameters
    ----------
    a : array_like, Quantity
      Input array.
    axis1 : int
      First axis.
    axis2 : int
      Second axis.

    Returns
    -------
    a_swapped : ndarray, Quantity
      a new array where the axes are swapped.
    """
    return _fun_keep_unit_unary(jnp.swapaxes, a, axis1=axis1, axis2=axis2)


@set_module_as('saiunit.math')
def tile(
    A: Union[jax.Array, Quantity],
    reps: Union[int, Tuple[int, ...]]
) -> Union[jax.Array, Quantity]:
    """
    Construct a quantity or an array by repeating A the number of times given by reps.

    Parameters
    ----------
    A : array_like, Quantity
      The input array.
    reps : array_like
      The number of repetitions of A along each axis.

    Returns
    -------
    res : ndarray, Quantity
      The tiled output array.
    """
    return _fun_keep_unit_unary(jnp.tile, A, reps=reps)


@set_module_as('saiunit.math')
def repeat(
    a: Union[jax.Array, Quantity],
    repeats: Union[int, Tuple[int, ...]],
    axis: Optional[int] = None,
    total_repeat_length: Optional[int] = None
) -> Union[jax.Array, Quantity]:
    """
    Repeat elements of a quantity or an array.

    Parameters
    ----------
    a : array_like, Quantity
      Input array.
    repeats : int or tuple of ints
      The number of repetitions for each element. `repeats` is broadcasted to fit the shape of the given axis.
    axis : int, optional
      The axis along which to repeat values. By default, use the flattened input array, and return a flat output array.
    total_repeat_length : int, optional
      The total length of the repeated array. If `total_repeat_length` is not None, the output array
      will have the length of `total_repeat_length`.

    Returns
    -------
    res : ndarray, Quantity
      Output array which has the same shape as `a`, except along the given axis.
    """
    return _fun_keep_unit_unary(jnp.repeat, a, repeats=repeats, axis=axis, total_repeat_length=total_repeat_length)


@set_module_as('saiunit.math')
def flip(
    m: Union[jax.Array, Quantity],
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[jax.Array, Quantity]:
    """
    Reverse the order of elements in a quantity or an array along the given axis.

    Parameters
    ----------
    m : array_like, Quantity
      Input array.
    axis : int or tuple of ints, optional
      Axis or axes along which to flip over. The default, axis=None, will flip over all of the axes of the input array.

    Returns
    -------
    res : ndarray, Quantity
      A view of `m` with the entries of axis reversed. Since a view is returned, this operation is done in constant time.
    """
    return _fun_keep_unit_unary(jnp.flip, m, axis=axis)


@set_module_as('saiunit.math')
def fliplr(
    m: Union[jax.Array, Quantity]
) -> Union[jax.Array, Quantity]:
    """
    Flip quantity or array in the left/right direction.

    Parameters
    ----------
    m : array_like, Quantity
      Input array.

    Returns
    -------
    res : ndarray, Quantity
      A view of `m` with the columns reversed. Since a view is returned, this operation is done in constant time.
    """
    return _fun_keep_unit_unary(jnp.fliplr, m)


@set_module_as('saiunit.math')
def flipud(
    m: Union[jax.Array, Quantity]
) -> Union[jax.Array, Quantity]:
    """
    Flip quantity or array in the up/down direction.

    Parameters
    ----------
    m : array_like, Quantity
      Input array.

    Returns
    -------
    res : ndarray, Quantity
      A view of `m` with the rows reversed.
    """
    return _fun_keep_unit_unary(jnp.flipud, m)


@set_module_as('saiunit.math')
def roll(
    a: Union[jax.Array, Quantity],
    shift: Union[int, Tuple[int, ...]],
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[jax.Array, Quantity]:
    """
    Roll quantity or array elements along a given axis.

    Parameters
    ----------
    a : array_like, Quantity
      Input array.
    shift : int or tuple of ints
      The number of places by which elements are shifted. If a tuple, then `axis` must be a tuple of the same size,
      and each of the given axes is shifted by the corresponding number. If an int while `axis` is a tuple of ints,
      then the same value is used for all given axes.
    axis : int or tuple of ints, optional
      Axis or axes along which elements are shifted. By default, the array is flattened before shifting, after which
      the original shape is restored.

    Returns
    -------
    res : ndarray, Quantity
      Output array, with the same shape as `a`.
    """
    return _fun_keep_unit_unary(jnp.roll, a, shift=shift, axis=axis)


@set_module_as('saiunit.math')
def expand_dims(
    a: Union[jax.Array, Quantity],
    axis: int
) -> Union[jax.Array, Quantity]:
    """
    Expand the shape of a quantity or an array.

    Parameters
    ----------
    a : array_like, Quantity
      Input array.
    axis : int
      Position in the expanded axes where the new axis is placed.

    Returns
    -------
    res : ndarray, Quantity
      View of `a` with the number of dimensions increased by one.
    """
    return _fun_keep_unit_unary(jnp.expand_dims, a, axis=axis)


@set_module_as('saiunit.math')
def squeeze(
    a: Union[jax.Array, Quantity],
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[jax.Array, Quantity]:
    """
    Remove single-dimensional entries from the shape of a quantity or an array.

    Parameters
    ----------
    a : array_like, Quantity
      Input data.
    axis : None or int or tuple of ints, optional
      Selects a subset of the single-dimensional entries in the shape. If an axis is selected with shape entry greater
      than one, an error is raised.

    Returns
    -------
    res : ndarray, Quantity
      An array with the same data as `a`, but with a lower dimension.
    """
    return _fun_keep_unit_unary(jnp.squeeze, a, axis=axis)


@set_module_as('saiunit.math')
def sort(
    a: Union[jax.Array, Quantity],
    axis: Optional[int] = -1,
    *,
    kind: None = None,
    order: None = None,
    stable: bool = True,
    descending: bool = False,
) -> Union[jax.Array, Quantity]:
    """
    Return a sorted copy of a quantity or an array.

    Parameters
    ----------
    a : array_like, Quantity
      Array or quantity to be sorted.
    axis : int or None, optional
      Axis along which to sort. If None, the array is flattened before sorting. The default is -1, which sorts along
      the last axis.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
      Sorting algorithm. The default is 'quicksort'.
    order : str or list of str, optional
      When `a` is a quantity, it can be a string or a sequence of strings, which is interpreted as an order the quantity
      should be sorted. The default is None.
    stable : bool, optional
      Whether to use a stable sorting algorithm. The default is True.
    descending : bool, optional
      Whether to sort in descending order. The default is False.

    Returns
    -------
    res : ndarray, Quantity
      Sorted copy of the input array.
    """
    return _fun_keep_unit_unary(jnp.sort, a, axis=axis, kind=kind, order=order, stable=stable, descending=descending)


@set_module_as('saiunit.math')
def max(
    a: Union[jax.Array, Quantity],
    axis: Optional[int] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float, Quantity]] = None,
    where: Optional[jax.Array] = None,
) -> Union[jax.Array, Quantity]:
    """
    Return the maximum of a quantity or an array or maximum along an axis.

    Parameters
    ----------
    a : array_like, Quantity
      Array or quantity containing numbers whose maximum is desired.
    axis : int or None, optional
      Axis or axes along which to operate. By default, flattened input is used.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
      option, the result will broadcast correctly against the input array.
    initial : scalar, optional
      The minimum value of an output element. Must be present to allow computation on empty slice.
      See `numpy.ufunc.reduce`.
    where : array_like, optional
      Values of True indicate to calculate the ufunc at that position, values of False indicate to leave the value in the
      output alone.

    Returns
    -------
    res : ndarray, Quantity
      Maximum of `a`. If `axis` is None, the result is a scalar value. If `axis` is given, the result is an array of
      dimension `a.ndim - 1`.
    """
    return _fun_keep_unit_unary(jnp.max, a, axis=axis, keepdims=keepdims, initial=initial, where=where)


@set_module_as('saiunit.math')
def min(
    a: Union[jax.Array, Quantity],
    axis: Optional[int] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float, Quantity]] = None,
    where: Optional[jax.Array] = None,
) -> Union[jax.Array, Quantity]:
    """
    Return the minimum of a quantity or an array or minimum along an axis.

    Parameters
    ----------
    a : array_like, Quantity
      Array or quantity containing numbers whose minimum is desired.
    axis : int or None, optional
      Axis or axes along which to operate. By default, flattened input is used.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
      option, the result will broadcast correctly against the input array.
    initial : scalar, optional
      The maximum value of an output element. Must be present to allow computation on empty slice.
      See `numpy.ufunc.reduce`.
    where : array_like, optional
      Values of True indicate to calculate the ufunc at that position, values of False indicate to leave the value in the
      output alone.

    Returns
    -------
    res : ndarray, Quantity
      Minimum of `a`. If `axis` is None, the result is a scalar value. If `axis` is given, the result is an array of
      dimension `a.ndim - 1`.
    """
    return _fun_keep_unit_unary(jnp.min, a, axis=axis, keepdims=keepdims, initial=initial, where=where)


amax = max
amin = min


@set_module_as('saiunit.math')
def diagonal(
    a: Union[jax.Array, Quantity],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1
) -> Union[jax.Array, Quantity]:
    """
    Return specified diagonals.

    Parameters
    ----------
    a : array_like, Quantity
      Array from which the diagonals are taken.
    offset : int, optional
      Offset of the diagonal from the main diagonal. Can be positive or negative. Defaults to main diagonal (0).
    axis1 : int, optional
      Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken. Defaults to first
      axis (0).
    axis2 : int, optional
      Axis to be used as the second axis of the 2-D sub-arrays from which the diagonals should be taken. Defaults to
      second axis (1).

    Returns
    -------
    res : ndarray
      The extracted diagonals. The shape of the output is determined by considering the shape of the input array with
      the specified axis removed.
    """
    return _fun_keep_unit_unary(jnp.diagonal, a, offset=offset, axis1=axis1, axis2=axis2)


@set_module_as('saiunit.math')
def ravel(
    a: Union[jax.Array, Quantity],
    order: str = 'C'
) -> Union[jax.Array, Quantity]:
    """
    Return a contiguous flattened quantity or array.

    Parameters
    ----------
    a : array_like, Quantity
      Input array. The elements in `a` are read in the order specified by `order`, and packed as a 1-D array.
    order : {'C', 'F', 'A', 'K'}, optional
      The elements of `a` are read using this index order. 'C' means to index the elements in row-major, C-style order,
      with the last axis index changing fastest, back to the first axis index changing slowest. 'F' means to index the
      elements in column-major, Fortran-style order, with the first index changing fastest, and the last index changing
      slowest. 'A' means to read the elements in Fortran-like index order if `a` is Fortran contiguous in memory, C-like
      order otherwise. 'K' means to read the elements in the order they occur in memory, except for reversing the data
      when strides are negative. By default, 'C' index order is used.

    Returns
    -------
    res : ndarray, Quantity
      The flattened quantity or array. The shape of the output is the same as `a`, but the array is 1-D.
    """
    return _fun_keep_unit_unary(jnp.ravel, a, order=order)


@set_module_as('saiunit.math')
def flatten(
    x: jax.typing.ArrayLike | Quantity,
    start_axis: Optional[int] = None,
    end_axis: Optional[int] = None
) -> jax.Array | Quantity:
    """Flattens input by reshaping it into a one-dimensional tensor.
    If ``start_dim`` or ``end_dim`` are passed, only dimensions starting
    with ``start_dim`` and ending with ``end_dim`` are flattened.
    The order of elements in input is unchanged.

    .. note::
       Flattening a zero-dimensional tensor will return a one-dimensional view.

    Parameters
    ----------
    x: Array, Quantity
      The input array.
    start_axis: int
      the first dim to flatten
    end_axis: int
      the last dim to flatten

    Returns
    -------
    out: Array, Quantity
    """
    shape = x.shape
    ndim = x.ndim
    if ndim == 0:
        ndim = 1
    if start_axis is None:
        start_axis = 0
    elif start_axis < 0:
        start_axis = ndim + start_axis
    if end_axis is None:
        end_axis = ndim - 1
    elif end_axis < 0:
        end_axis = ndim + end_axis
    end_axis += 1
    if start_axis < 0 or start_axis > ndim:
        raise ValueError(f'start_axis {start_axis} is out of size.')
    if end_axis < 0 or end_axis > ndim:
        raise ValueError(f'end_axis {end_axis} is out of size.')
    new_shape = shape[:start_axis] + (np.prod(shape[start_axis: end_axis], dtype=int),) + shape[end_axis:]
    return _fun_keep_unit_unary(jnp.reshape, x, shape=new_shape)


@set_module_as('saiunit.math')
def unflatten(
    x: jax.typing.ArrayLike | Quantity,
    axis: int,
    sizes: Sequence[int]
) -> jax.Array | Quantity:
    """
    Expands a dimension of the input tensor over multiple dimensions.

    Args:
      x: input tensor.
      axis: Dimension to be unflattened, specified as an index into ``x.shape``.
      sizes: New shape of the unflattened dimension. One of its elements can be -1
          in which case the corresponding output dimension is inferred.
          Otherwise, the product of ``sizes`` must equal ``input.shape[dim]``.

    Returns:
      A tensor with the same data as ``input``, but with ``dim`` split into multiple dimensions.
      The returned tensor has one more dimension than the input tensor.
      The returned tensor shares the same underlying data with this tensor.
    """
    assert x.ndim > axis, ('The dimension to be unflattened should '
                           'be less than the tensor dimension. '
                           f'Got {axis} and {x.ndim}.')
    shape = x.shape
    new_shape = shape[:axis] + tuple(sizes) + shape[axis + 1:]
    return _fun_keep_unit_unary(jnp.reshape, x, shape=new_shape)


@set_module_as('saiunit.math')
def remove_diag(x: jax.typing.ArrayLike | Quantity) -> jax.Array | Quantity:
    """Remove the diagonal of the matrix.

    Parameters
    ----------
    x: Array, Quantity
      The matrix with the shape of `(M, N)`.

    Returns
    -------
    arr: Array, Quantity
      The matrix without diagonal which has the shape of `(M, N-1)`.
    """
    unit = UNITLESS
    if isinstance(x, Quantity):
        x = x.mantissa
        unit = x.unit

    if x.ndim != 2:
        raise ValueError(f'Only support 2D matrix, while we got a {x.ndim}D array.')
    eyes = jnp.fill_diagonal(jnp.ones(x.shape, dtype=bool), False)
    x = jnp.reshape(x[eyes], (x.shape[0], x.shape[1] - 1))
    if unit.is_unitless:
        return x
    return Quantity(x, unit=unit)


# ----------  selection


@set_module_as('saiunit.math')
def choose(
    a: Union[jax.Array, Quantity],
    choices: Sequence[Union[jax.Array, Quantity]],
    mode: str = 'raise',
) -> Union[jax.Array, Quantity]:
    """
    Construct a quantity or an array from an index array and a set of arrays to choose from.

    Parameters
    ----------
    a : array_like, Quantity
      This array must be an integer array of the same shape as `choices`. The elements of `a` are used to select elements
      from `choices`.
    choices : sequence of array_like
      Choice arrays. `a` and all `choices` must be broadcastable to the same shape.
    mode : {'raise', 'wrap', 'clip'}, optional
      Specifies how indices outside [0, n-1] will be treated:
      - 'raise' : raise an error (default)
      - 'wrap' : wrap around
      - 'clip' : clip to the range [0, n-1]

    Returns
    -------
    res : ndarray, Quantity
      The constructed array. The shape is identical to the shape of `a`, and the data type is the data type of `choices`.
    """
    return _fun_keep_unit_unary(jnp.choose, a, choices=choices, mode=mode)


@set_module_as('saiunit.math')
def diagflat(
    v: Union[jax.Array, Quantity],
    k: int = 0
) -> Union[jax.Array, Quantity]:
    """
    Create a two-dimensional a quantity or array with the flattened input as a diagonal.

    Parameters
    ----------
    v : array_like, Quantity
      Input data, which is flattened and set as the `k`-th diagonal of the output.
    k : int, optional
      Diagonal in question. The default is 0.

    Returns
    -------
    res : ndarray, Quantity
      The 2-D output array.
    """
    return _fun_keep_unit_unary(jnp.diagflat, v, k=k)


# math funcs keep unit (unary)
# ----------------------------


def _fun_keep_unit_unary(func, x, *args, **kwargs):
    if isinstance(x, Quantity):
        return Quantity(func(x.mantissa, *args, **kwargs), unit=x.unit)
    else:
        return func(x, *args, **kwargs)


def astype(
    x: Union[jax.typing.ArrayLike, Quantity],
    dtype: jax.typing.DTypeLike
) -> Union[jax.Array, Quantity]:
    """
    Copy of the array, cast to a specified type.

    Parameters
    ----------
    x : array_like, Quantity
        Input array.
    dtype : dtype
        Typecode or data-type to which the array is cast.

    Returns
    -------
    out : ndarray, Quantity
        A copy of the array, cast to a specified type.
    """
    return _fun_keep_unit_unary(jnp.astype, x, dtype)


@set_module_as('saiunit.math')
def real(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
    """
    Return the real part of the complex argument.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.real, x)


@set_module_as('saiunit.math')
def imag(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
    """
    Return the imaginary part of the complex argument.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.imag, x)


@set_module_as('saiunit.math')
def conj(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
    """
    Return the complex conjugate of the argument.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.conj, x)


@set_module_as('saiunit.math')
def conjugate(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
    """
    Return the complex conjugate of the argument.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.conjugate, x)


@set_module_as('saiunit.math')
def negative(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
    """
    Return the negative of the argument.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.negative, x)


@set_module_as('saiunit.math')
def positive(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
    """
    Return the positive of the argument.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.positive, x)


@set_module_as('saiunit.math')
def abs(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
    """
    Return the absolute value of the argument.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.abs, x)


@set_module_as('saiunit.math')
def sum(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
    keepdims: bool = False,
    initial: Union[jax.typing.ArrayLike, Quantity, None] = None,
    where: Union[jax.typing.ArrayLike, None] = None,
    promote_integers: bool = True
) -> Union[Quantity, jax.Array]:
    """
    Return the sum of the array elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : None or int or tuple of ints, optional
      Axis or axes along which a sum is performed.  The default,
      axis=None, will sum all of the elements of the input array.  If
      axis is negative it counts from the last to the first axis.

      If axis is a tuple of ints, a sum is performed on all of the axes
      specified in the tuple instead of a single axis or all the axes as
      before.
    dtype : dtype, optional
      The type of the returned array and of the accumulator in which the
      elements are summed.  The dtype of `a` is used by default unless `a`
      has an integer dtype of less precision than the default platform
      integer.  In that case, if `a` is signed then the platform integer
      is used while if `a` is unsigned then an unsigned integer of the
      same precision as the platform integer is used.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the input array.

      If the default value is passed, then `keepdims` will not be
      passed through to the `sum` method of sub-classes of
      `ndarray`, however any non-default value will be.  If the
      sub-class' method does not implement `keepdims` any
      exceptions will be raised.
    initial : scalar, optional
      Starting value for the sum. See `~numpy.ufunc.reduce` for details.
    where : array_like of bool, optional
      Elements to include in the sum. See `~numpy.ufunc.reduce` for details.
    promote_integers : bool, optional
      If True, and if the accumulator is an integer type, then the

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    if initial is not None:
        initial = Quantity(initial).in_unit(get_unit(x)).mantissa
    return _fun_keep_unit_unary(jnp.sum,
                                x,
                                axis=axis,
                                dtype=dtype,
                                keepdims=keepdims,
                                initial=initial,
                                where=where,
                                promote_integers=promote_integers)


@set_module_as('saiunit.math')
def nancumsum(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
) -> Union[Quantity, jax.Array]:
    """
    Return the cumulative sum of the array elements, ignoring NaNs.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : int, optional
      Axis along which the cumulative sum is computed. The default
      (None) is to compute the cumsum over the flattened array.
    dtype : dtype, optional
      Type of the returned array and of the accumulator in which the
      elements are summed.  If `dtype` is not specified, it defaults
      to the dtype of `a`, unless `a` has an integer dtype with a
      precision less than that of the default platform integer.  In
      that case, the default platform integer is used.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.nancumsum, x, axis=axis, dtype=dtype)


@set_module_as('saiunit.math')
def nansum(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
    keepdims: bool = False,
    initial: Union[jax.typing.ArrayLike, Quantity, None] = None,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> Union[Quantity, jax.Array]:
    """
    Return the sum of the array elements, ignoring NaNs.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : {int, tuple of int, None}, optional
      Axis or axes along which the sum is computed. The default is to compute the
      sum of the flattened array.
    dtype : data-type, optional
      The type of the returned array and of the accumulator in which the
      elements are summed.  By default, the dtype of `a` is used.  An
      exception is when `a` has an integer type with less precision than
      the platform (u)intp. In that case, the default will be either
      (u)int32 or (u)int64 depending on whether the platform is 32 or 64
      bits. For inexact inputs, dtype must be inexact.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `mean` or `sum` methods
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    initial : scalar, Quantity, optional
        Starting value for the sum. See `~numpy.ufunc.reduce` for details.
    where : array_like of bool, optional
        Elements to include in the sum. See `~numpy.ufunc.reduce` for details.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    if initial is not None:
        initial = Quantity(initial).in_unit(get_unit(x)).mantissa
    return _fun_keep_unit_unary(jnp.nansum,
                                x,
                                axis=axis,
                                dtype=dtype,
                                keepdims=keepdims,
                                initial=initial,
                                where=where)


@set_module_as('saiunit.math')
def cumsum(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
) -> Union[Quantity, jax.Array]:
    """
    Return the cumulative sum of the array elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : int, optional
      Axis along which the cumulative sum is computed. The default
      (None) is to compute the cumsum over the flattened array.
    dtype : dtype, optional
      Type of the returned array and of the accumulator in which the
      elements are summed.  If `dtype` is not specified, it defaults
      to the dtype of `a`, unless `a` has an integer dtype with a
      precision less than that of the default platform integer.  In
      that case, the default platform integer is used.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.cumsum, x, axis=axis, dtype=dtype)


@set_module_as('saiunit.math')
def ediff1d(
    x: Quantity | jax.typing.ArrayLike,
    to_end: jax.typing.ArrayLike | Quantity = None,
    to_begin: jax.typing.ArrayLike | Quantity = None
) -> Union[Quantity, jax.Array]:
    """
    Return the differences between consecutive elements of the array.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    to_end : array_like, optional
      Number(s) to append at the end of the returned differences.
    to_begin : array_like, optional
      Number(s) to prepend at the beginning of the returned differences.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    x_unit = get_unit(x)
    if to_end is not None:
        to_end = Quantity(to_end).in_unit(x_unit).mantissa
    if to_begin is not None:
        to_begin = Quantity(to_begin).in_unit(x_unit).mantissa
    return _fun_keep_unit_unary(jnp.ediff1d, x, to_end=to_end, to_begin=to_begin)


@set_module_as('saiunit.math')
def absolute(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
    """
    Return the absolute value of the argument.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.absolute, x)


@set_module_as('saiunit.math')
def fabs(x: Union[Quantity, jax.typing.ArrayLike]) -> Union[Quantity, jax.Array]:
    """
    Return the absolute value of the argument.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.fabs, x)


@set_module_as('saiunit.math')
def median(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    overwrite_input: bool = False,
    keepdims: bool = False
) -> Union[Quantity, jax.Array]:
    """
    Return the median of the array elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : {int, sequence of int, None}, optional
      Axis or axes along which the medians are computed. The default
      is to compute the median along a flattened version of the array.
      A sequence of axes is supported since version 1.9.0.
    overwrite_input : bool, optional
     If True, then allow use of memory of input array `a` for
     calculations. The input array will be modified by the call to
     `median`. This will save memory when you do not need to preserve
     the contents of the input array. Treat the input as undefined,
     but it will probably be fully or partially sorted. Default is
     False. If `overwrite_input` is ``True`` and `a` is not already an
     `ndarray`, an error will be raised.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the original `arr`.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.median, x, axis=axis, overwrite_input=overwrite_input, keepdims=keepdims)


@set_module_as('saiunit.math')
def nanmin(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    keepdims: bool = False,
    initial: Union[jax.typing.ArrayLike, Quantity, None] = None,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> Union[Quantity, jax.Array]:
    """
    Return the minimum of the array elements, ignoring NaNs.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : {int, tuple of int, None}, optional
      Axis or axes along which the minimum is computed. The default is to compute
      the minimum of the flattened array.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the original `a`.

      If the value is anything but the default, then
      `keepdims` will be passed through to the `min` method
      of sub-classes of `ndarray`.  If the sub-classes methods
      does not implement `keepdims` any exceptions will be raised.
    initial : scalar, optional
      The maximum value of an output element. Must be present to allow
      computation on empty slice. See `~numpy.ufunc.reduce` for details.
    where : array_like of bool, optional
      Elements to compare for the minimum. See `~numpy.ufunc.reduce`
      for details.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    if initial is not None:
        initial = Quantity(initial).in_unit(get_unit(x)).mantissa
    return _fun_keep_unit_unary(jnp.nanmin, x, axis=axis, keepdims=keepdims, initial=initial, where=where)


@set_module_as('saiunit.math')
def nanmax(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    keepdims: bool = False,
    initial: Union[jax.typing.ArrayLike, None] = None,
    where: Union[jax.typing.ArrayLike, None] = None,
) -> Union[Quantity, jax.Array]:
    """
    Return the maximum of the array elements, ignoring NaNs.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : {int, tuple of int, None}, optional
      Axis or axes along which the minimum is computed. The default is to compute
      the minimum of the flattened array.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the original `a`.

      If the value is anything but the default, then
      `keepdims` will be passed through to the `min` method
      of sub-classes of `ndarray`.  If the sub-classes methods
      does not implement `keepdims` any exceptions will be raised.
    initial : scalar, optional
      The maximum value of an output element. Must be present to allow
      computation on empty slice. See `~numpy.ufunc.reduce` for details.
    where : array_like of bool, optional
      Elements to compare for the minimum. See `~numpy.ufunc.reduce`
      for details.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    if initial is not None:
        initial = Quantity(initial).in_unit(get_unit(x)).mantissa
    return _fun_keep_unit_unary(jnp.nanmax, x, axis=axis, keepdims=keepdims, initial=initial, where=where)


@set_module_as('saiunit.math')
def ptp(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    keepdims: bool = False,
) -> Union[Quantity, jax.Array]:
    """
    Return the range of the array elements (maximum - minimum).

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : None or int or tuple of ints, optional
      Axis along which to find the peaks.  By default, flatten the
      array.  `axis` may be negative, in
      which case it counts from the last to the first axis.

      If this is a tuple of ints, a reduction is performed on multiple
      axes, instead of a single axis or all the axes as before.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the input array.

      If the default value is passed, then `keepdims` will not be
      passed through to the `ptp` method of sub-classes of
      `ndarray`, however any non-default value will be.  If the
      sub-class' method does not implement `keepdims` any
      exceptions will be raised.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.ptp, x, axis=axis, keepdims=keepdims)


@set_module_as('saiunit.math')
def average(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    weights: Union[jax.typing.ArrayLike, None] = None,
    returned: bool = False,
    keepdims: bool = False
) -> Union[Quantity, jax.Array]:
    """
    Return the weighted average of the array elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : None or int or tuple of ints, optional
      Axis or axes along which to average `a`.  The default,
      axis=None, will average over all of the elements of the input array.
      If axis is negative it counts from the last to the first axis.

      If axis is a tuple of ints, averaging is performed on all of the axes
      specified in the tuple instead of a single axis or all the axes as
      before.
    weights : array_like, optional
      An array of weights associated with the values in `a`. Each value in
      `a` contributes to the average according to its associated weight.
      The weights array can either be 1-D (in which case its length must be
      the size of `a` along the given axis) or of the same shape as `a`.
      If `weights=None`, then all data in `a` are assumed to have a
      weight equal to one.  The 1-D calculation is::

          avg = sum(a * weights) / sum(weights)

      The only constraint on `weights` is that `sum(weights)` must not be 0.
    returned : bool, optional
      Default is `False`. If `True`, the tuple (`average`, `sum_of_weights`)
      is returned, otherwise only the average is returned.
      If `weights=None`, `sum_of_weights` is equivalent to the number of
      elements over which the average is taken.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the original `a`.
      *Note:* `keepdims` will not work with instances of `numpy.matrix`
      or other classes whose methods do not support `keepdims`.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.average, x, axis=axis, weights=weights, returned=returned, keepdims=keepdims)


@set_module_as('saiunit.math')
def mean(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
    keepdims: bool = False, *,
    where: Union[jax.typing.ArrayLike, None] = None
) -> Union[Quantity, jax.Array]:
    """
    Return the mean of the array elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : None or int or tuple of ints, optional
      Axis or axes along which the means are computed. The default is to
      compute the mean of the flattened array.

      If this is a tuple of ints, a mean is performed over multiple axes,
      instead of a single axis or all the axes as before.
    dtype : data-type, optional
      Type to use in computing the mean.  For integer inputs, the default
      is `float64`; for floating point inputs, it is the same as the
      input dtype.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the input array.

      If the default value is passed, then `keepdims` will not be
      passed through to the `mean` method of sub-classes of
      `ndarray`, however any non-default value will be.  If the
      sub-class' method does not implement `keepdims` any
      exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in the mean.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.mean, x, axis=axis, dtype=dtype, keepdims=keepdims, where=where)


@set_module_as('saiunit.math')
def std(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
    ddof: int = 0,
    keepdims: bool = False, *,
    where: Union[jax.typing.ArrayLike, None] = None
) -> Union[Quantity, jax.Array]:
    """
    Return the standard deviation of the array elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : None or int or tuple of ints, optional
      Axis or axes along which the standard deviation is computed. The
      default is to compute the standard deviation of the flattened array.

      If this is a tuple of ints, a standard deviation is performed over
      multiple axes, instead of a single axis or all the axes as before.
    dtype : dtype, optional
      Type to use in computing the standard deviation. For arrays of
      integer type the default is float64, for arrays of float types it is
      the same as the array type.
    ddof : int, optional
      Means Delta Degrees of Freedom.  The divisor used in calculations
      is ``N - ddof``, where ``N`` represents the number of elements.
      By default `ddof` is zero.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the input array.

      If the default value is passed, then `keepdims` will not be
      passed through to the `std` method of sub-classes of
      `ndarray`, however any non-default value will be.  If the
      sub-class' method does not implement `keepdims` any
      exceptions will be raised.
    where : array_like of bool, optional
      Elements to include in the standard deviation.
      See `~numpy.ufunc.reduce` for details.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.std, x, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)


@set_module_as('saiunit.math')
def nanmedian(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, tuple[int, ...], None] = None,
    overwrite_input: bool = False,
    keepdims: bool = False
) -> Union[Quantity, jax.Array]:
    """
    Return the median of the array elements, ignoring NaNs.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : {int, sequence of int, None}, optional
      Axis or axes along which the medians are computed. The default
      is to compute the median along a flattened version of the array.
      A sequence of axes is supported since version 1.9.0.
    overwrite_input : bool, optional
     If True, then allow use of memory of input array `a` for
     calculations. The input array will be modified by the call to
     `median`. This will save memory when you do not need to preserve
     the contents of the input array. Treat the input as undefined,
     but it will probably be fully or partially sorted. Default is
     False. If `overwrite_input` is ``True`` and `a` is not already an
     `ndarray`, an error will be raised.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the original `a`.

      If this is anything but the default value it will be passed
      through (in the special case of an empty array) to the
      `mean` function of the underlying array.  If the array is
      a sub-class and `mean` does not have the kwarg `keepdims` this
      will raise a RuntimeError.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.nanmedian, x, axis=axis, overwrite_input=overwrite_input, keepdims=keepdims)


@set_module_as('saiunit.math')
def nanmean(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
    keepdims: bool = False, *,
    where: Union[jax.typing.ArrayLike, None] = None
) -> Union[Quantity, jax.Array]:
    """
    Return the mean of the array elements, ignoring NaNs.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : None or int or tuple of ints, optional
      Axis or axes along which the means are computed. The default is to
      compute the mean of the flattened array.

      If this is a tuple of ints, a mean is performed over multiple axes,
      instead of a single axis or all the axes as before.
    dtype : data-type, optional
      Type to use in computing the mean.  For integer inputs, the default
      is `float64`; for floating point inputs, it is the same as the
      input dtype.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the input array.

      If the default value is passed, then `keepdims` will not be
      passed through to the `mean` method of sub-classes of
      `ndarray`, however any non-default value will be.  If the
      sub-class' method does not implement `keepdims` any
      exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in the mean.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.nanmean, x, axis=axis, dtype=dtype, keepdims=keepdims, where=where)


@set_module_as('saiunit.math')
def nanstd(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: Union[int, Sequence[int], None] = None,
    dtype: Union[jax.typing.DTypeLike, None] = None,
    ddof: int = 0,
    keepdims: bool = False, *,
    where: Union[jax.typing.ArrayLike, None] = None
) -> Union[Quantity, jax.Array]:
    """
    Return the standard deviation of the array elements, ignoring NaNs.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    axis : None or int or tuple of ints, optional
      Axis or axes along which the standard deviation is computed. The
      default is to compute the standard deviation of the flattened array.

      If this is a tuple of ints, a standard deviation is performed over
      multiple axes, instead of a single axis or all the axes as before.
    dtype : dtype, optional
      Type to use in computing the standard deviation. For arrays of
      integer type the default is float64, for arrays of float types it is
      the same as the array type.
    ddof : int, optional
      Means Delta Degrees of Freedom.  The divisor used in calculations
      is ``N - ddof``, where ``N`` represents the number of elements.
      By default `ddof` is zero.
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left
      in the result as dimensions with size one. With this option,
      the result will broadcast correctly against the input array.

      If the default value is passed, then `keepdims` will not be
      passed through to the `std` method of sub-classes of
      `ndarray`, however any non-default value will be.  If the
      sub-class' method does not implement `keepdims` any
      exceptions will be raised.
    where : array_like of bool, optional
      Elements to include in the standard deviation.
      See `~numpy.ufunc.reduce` for details.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.nanstd, x, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims,
                                where=where)


@set_module_as('saiunit.math')
def diff(
    x: Union[Quantity, jax.typing.ArrayLike],
    n: int = 1,
    axis: int = -1,
    prepend: Union[jax.typing.ArrayLike, Quantity, None] = None,
    append: Union[jax.typing.ArrayLike, Quantity, None] = None
) -> Union[Quantity, jax.Array]:
    """
    Return the differences between consecutive elements of the array.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    n : int, optional
      The number of times values are differenced. If zero, the input
      is returned as-is.
    axis : int, optional
      The axis along which the difference is taken, default is the
      last axis.
    prepend, append : array_like, optional
      Values to prepend or append to `a` along axis prior to
      performing the difference.  Scalar values are expanded to
      arrays with length 1 in the direction of axis and the shape
      of the input array in along all other axes.  Otherwise the
      dimension and shape must match `a` except along axis.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x` is a Quantity, else an array.
    """
    x_unit = get_unit(x)
    if prepend is not None:
        prepend = Quantity(prepend).in_unit(x_unit).mantissa
    if append is not None:
        append = Quantity(append).in_unit(x_unit).mantissa
    return _fun_keep_unit_unary(jnp.diff, x, n=n, axis=axis, prepend=prepend, append=append)


@set_module_as('saiunit.math')
def rot90(
    m: Union[jax.typing.ArrayLike, Quantity],
    k: int = 1,
    axes: Tuple[int, int] = (0, 1)
) -> Union[
    jax.Array, Quantity]:
    """
    Rotate an array by 90 degrees in the plane specified by axes.

    Rotation direction is from the first towards the second axis.

    Parameters
    ----------
    m : array_like, Quantity
      Array of two or more dimensions.
    k : integer
      Number of times the array is rotated by 90 degrees.
    axes : (2,) array_like
      The array is rotated in the plane defined by the axes.
      Axes must be different.

    Returns
    -------
    y : ndarray, Quantity
      A rotated view of `m`.

      This is a quantity if `m` is a quantity.
    """
    return _fun_keep_unit_unary(jnp.rot90, m, k=k, axes=axes)


@set_module_as('saiunit.math')
def intersect1d(
    ar1: Union[jax.typing.ArrayLike, Quantity],
    ar2: Union[jax.typing.ArrayLike, Quantity],
    assume_unique: bool = False,
    return_indices: bool = False
) -> Union[jax.Array, Quantity, tuple[Union[jax.Array, Quantity], jax.Array, jax.Array]]:
    """
    Find the intersection of two arrays.

    Return the sorted, unique values that are in both of the input arrays.

    Parameters
    ----------
    ar1, ar2 : array_like, Quantity
      Input arrays. Will be flattened if not already 1D.
    assume_unique : bool
      If True, the input arrays are both assumed to be unique, which
      can speed up the calculation.  If True but ``ar1`` or ``ar2`` are not
      unique, incorrect results and out-of-bounds indices could result.
      Default is False.
    return_indices : bool
      If True, the indices which correspond to the intersection of the two
      arrays are returned. The first instance of a value is used if there are
      multiple. Default is False.

    Returns
    -------
    intersect1d : ndarray, Quantity
      Sorted 1D array of common and unique elements.
    comm1 : ndarray
      The indices of the first occurrences of the common values in `ar1`.
      Only provided if `return_indices` is True.
    comm2 : ndarray
      The indices of the first occurrences of the common values in `ar2`.
      Only provided if `return_indices` is True.
    """
    fail_for_dimension_mismatch(ar1, ar2, 'intersect1d')
    unit = UNITLESS
    if isinstance(ar1, Quantity):
        unit = ar1.unit
    ar1 = ar1.in_unit(unit).mantissa if isinstance(ar1, Quantity) else ar1
    ar2 = ar2.in_unit(unit).mantissa if isinstance(ar2, Quantity) else ar2
    result = jnp.intersect1d(ar1, ar2, assume_unique=assume_unique, return_indices=return_indices)
    if return_indices:
        if unit.is_unitless:
            return result
        else:
            return Quantity(result[0], unit=unit), result[1], result[2]
    else:
        if unit.is_unitless:
            return result
        else:
            return Quantity(result, unit=unit)


@set_module_as('saiunit.math')
def nan_to_num(
    x: Union[jax.typing.ArrayLike, Quantity],
    nan: float | Quantity = None,
    posinf: float | Quantity = None,
    neginf: float | Quantity = None
) -> Union[jax.Array, Quantity]:
    """
    Replace NaN with zero and infinity with large finite numbers (default
    behaviour) or with the numbers defined by the user using the `nan`,
    `posinf` and/or `neginf` keywords.

    If `x` is inexact, NaN is replaced by zero or by the user defined value in
    `nan` keyword, infinity is replaced by the largest finite floating point
    values representable by ``x.dtype`` or by the user defined value in
    `posinf` keyword and -infinity is replaced by the most negative finite
    floating point values representable by ``x.dtype`` or by the user defined
    value in `neginf` keyword.

    For complex dtypes, the above is applied to each of the real and
    imaginary components of `x` separately.

    If `x` is not inexact, then no replacements are made.

    Parameters
    ----------
    x : scalar, array_like or Quantity
      Input data.
    nan : int, float, optional
      Value to be used to fill NaN values. If no value is passed
      then NaN values will be replaced with 0.0.
    posinf : int, float, optional
      Value to be used to fill positive infinity values. If no value is
      passed then positive infinity values will be replaced with a very
      large number.
    neginf : int, float, optional
      Value to be used to fill negative infinity values. If no value is
      passed then negative infinity values will be replaced with a very
      small (or negative) number.

    Returns
    -------
    out : ndarray, Quantity
      `x`, with the non-finite values replaced. If `copy` is False, this may
      be `x` itself.
    """
    x_unit = get_unit(x)
    if isinstance(x, Quantity):
        if nan is not None:
            nan = Quantity(nan).in_unit(x_unit).mantissa
        else:
            nan = 0.0
        if posinf is not None:
            posinf = Quantity(posinf).in_unit(x_unit).mantissa
        if neginf is not None:
            neginf = Quantity(neginf).in_unit(x_unit).mantissa
        r = jnp.nan_to_num(x.mantissa, nan=nan, posinf=posinf, neginf=neginf)
        return r if x_unit.is_unitless else Quantity(r, unit=x_unit)
    else:
        nan = 0.0 if nan is None else nan
        return jnp.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


@set_module_as('saiunit.math')
def trace(
    a: Union[jax.Array, Quantity],
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: Optional[jax.typing.DTypeLike] = None,
) -> Union[jax.Array, Quantity]:
    """
    Return the sum along diagonals of the array.

    If `a` is 2-D, the sum along its diagonal with the given offset
    is returned, i.e., the sum of elements ``a[i,i+offset]`` for all i.

    If `a` has more than two dimensions, then the axes specified by axis1 and
    axis2 are used to determine the 2-D sub-arrays whose traces are returned.
    The shape of the resulting array is the same as that of `a` with `axis1`
    and `axis2` removed.

    Parameters
    ----------
    a : array_like, Quantity
      Input array, from which the diagonals are taken.
    offset : int, optional
      Offset of the diagonal from the main diagonal. Can be both positive
      and negative. Defaults to 0.
    axis1, axis2 : int, optional
      Axes to be used as the first and second axis of the 2-D sub-arrays
      from which the diagonals should be taken. Defaults are the first two
      axes of `a`.
    dtype : dtype, optional
      Determines the data-type of the returned array and of the accumulator
      where the elements are summed. If dtype has the value None and `a` is
      of integer type of precision less than the default integer
      precision, then the default integer precision is used. Otherwise,
      the precision is the same as that of `a`.

    Returns
    -------
    sum_along_diagonals : ndarray
      If `a` is 2-D, the sum along the diagonal is returned.  If `a` has
      larger dimensions, then an array of sums along diagonals is returned.

      This is a Quantity if `a` is a Quantity, else an array.
    """
    return _fun_keep_unit_unary(jnp.trace, a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)


@set_module_as('saiunit.math')
def percentile(
    a: Union[jax.Array, Quantity],
    q: jax.typing.ArrayLike,
    axis: Optional[Union[int, Tuple[int]]] = None,
    method: str = 'linear',
    keepdims: Optional[bool] = False,
) -> jax.Array:
    """
    Compute the q-th percentile of the data along the specified axis.

    Returns the q-th percentile(s) of the array elements.

    Parameters
    ----------
    a : array_like, Quantity
      Input array or Quantity.
    q : array_like
      Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
    method : str, optional
      This parameter specifies the method to use for estimating the
      percentile.  There are many different methods, some unique to NumPy.
      See the notes for explanation.  The options sorted by their R type
      as summarized in the H&F paper [1]_ are:

      1. 'inverted_cdf'
      2. 'averaged_inverted_cdf'
      3. 'closest_observation'
      4. 'interpolated_inverted_cdf'
      5. 'hazen'
      6. 'weibull'
      7. 'linear'  (default)
      8. 'median_unbiased'
      9. 'normal_unbiased'

      The first three methods are discontinuous.  NumPy further defines the
      following discontinuous variations of the default 'linear' (7.) option:

      * 'lower'
      * 'higher',
      * 'midpoint'
      * 'nearest'
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    if isinstance(q, Quantity):
        assert q.is_unitless, 'Percentile should be unitless.'
        q = q.mantissa
    return _fun_keep_unit_unary(
        jnp.percentile, a, q=q, axis=axis, method=method, keepdims=keepdims,
    )


@set_module_as('saiunit.math')
def nanpercentile(
    a: Union[jax.Array, Quantity],
    q: jax.typing.ArrayLike,
    axis: Optional[Union[int, Tuple[int]]] = None,
    method: str = 'linear',
    keepdims: Optional[bool] = False,
) -> jax.Array:
    """
    Compute the q-th percentile of the data along the specified axis, while ignoring nan values.

    Returns the q-th percentile(s) of the array elements, while ignoring nan values.

    Parameters
    ----------
    a : array_like, Quantity
      Input array or Quantity.
    q : array_like, Quantity
      Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
    method : str, optional
      This parameter specifies the method to use for estimating the
      percentile.  There are many different methods, some unique to NumPy.
      See the notes for explanation.  The options sorted by their R type
      as summarized in the H&F paper [1]_ are:

      1. 'inverted_cdf'
      2. 'averaged_inverted_cdf'
      3. 'closest_observation'
      4. 'interpolated_inverted_cdf'
      5. 'hazen'
      6. 'weibull'
      7. 'linear'  (default)
      8. 'median_unbiased'
      9. 'normal_unbiased'

      The first three methods are discontinuous.  NumPy further defines the
      following discontinuous variations of the default 'linear' (7.) option:

      * 'lower'
      * 'higher',
      * 'midpoint'
      * 'nearest'
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    if isinstance(q, Quantity):
        assert q.is_unitless, 'Percentile should be unitless.'
        q = q.mantissa
    return _fun_keep_unit_unary(
        jnp.nanpercentile, a, q=q, axis=axis, method=method, keepdims=keepdims,
    )


@set_module_as('saiunit.math')
def quantile(
    a: Union[jax.Array, Quantity],
    q: jax.typing.ArrayLike,
    axis: Optional[Union[int, Tuple[int]]] = None,
    method: str = 'linear',
    keepdims: Optional[bool] = False,
) -> jax.Array:
    """
    Compute the q-th percentile of the data along the specified axis.

    Returns the q-th percentile(s) of the array elements.

    Parameters
    ----------
    a : array_like, Quantity
      Input array or Quantity.
    q : array_like, Quantity
      Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
    method : str, optional
      This parameter specifies the method to use for estimating the
      percentile.  There are many different methods, some unique to NumPy.
      See the notes for explanation.  The options sorted by their R type
      as summarized in the H&F paper [1]_ are:

      1. 'inverted_cdf'
      2. 'averaged_inverted_cdf'
      3. 'closest_observation'
      4. 'interpolated_inverted_cdf'
      5. 'hazen'
      6. 'weibull'
      7. 'linear'  (default)
      8. 'median_unbiased'
      9. 'normal_unbiased'

      The first three methods are discontinuous.  NumPy further defines the
      following discontinuous variations of the default 'linear' (7.) option:

      * 'lower'
      * 'higher',
      * 'midpoint'
      * 'nearest'
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    if isinstance(q, Quantity):
        assert q.is_unitless, 'Percentile should be unitless.'
        q = q.mantissa
    return _fun_keep_unit_unary(
        jnp.quantile, a, q=q, axis=axis, method=method, keepdims=keepdims
    )


@set_module_as('saiunit.math')
def nanquantile(
    a: Union[jax.Array, Quantity],
    q: jax.typing.ArrayLike,
    axis: Optional[Union[int, Tuple[int]]] = None,
    method: str = 'linear',
    keepdims: Optional[bool] = False,
) -> jax.Array:
    """
    Compute the q-th percentile of the data along the specified axis, while ignoring nan values.

    Returns the q-th percentile(s) of the array elements, while ignoring nan values.

    Parameters
    ----------
    a : array_like, Quantity
      Input array or Quantity.
    q : array_like, Quantity
      Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
    method : str, optional
      This parameter specifies the method to use for estimating the
      percentile.  There are many different methods, some unique to NumPy.
      See the notes for explanation.  The options sorted by their R type
      as summarized in the H&F paper [1]_ are:

      1. 'inverted_cdf'
      2. 'averaged_inverted_cdf'
      3. 'closest_observation'
      4. 'interpolated_inverted_cdf'
      5. 'hazen'
      6. 'weibull'
      7. 'linear'  (default)
      8. 'median_unbiased'
      9. 'normal_unbiased'

      The first three methods are discontinuous.  NumPy further defines the
      following discontinuous variations of the default 'linear' (7.) option:

      * 'lower'
      * 'higher',
      * 'midpoint'
      * 'nearest'
    keepdims : bool, optional
      If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns
    -------
    out : jax.Array
      Output array.
    """
    if isinstance(q, Quantity):
        assert q.is_unitless, 'Percentile should be unitless.'
        q = q.mantissa
    return _fun_keep_unit_unary(
        jnp.nanquantile, a, q=q, axis=axis, method=method, keepdims=keepdims,
    )


# math funcs keep unit (binary)
# -----------------------------

def _fun_keep_unit_binary(func, x1, x2, *args, **kwargs):
    if isinstance(x1, Quantity) and isinstance(x2, Quantity):
        return Quantity(func(x1.mantissa, x2.in_unit(x1.unit).mantissa, *args, **kwargs), unit=x1.unit)
    elif isinstance(x1, Quantity):
        assert x1.is_unitless, f'Expected unitless array when x2 is not Quantity, while got {x1}'
        return func(x1.mantissa, x2, *args, **kwargs)
    elif isinstance(x2, Quantity):
        assert x2.is_unitless, f'Expected unitless array when x1 is not Quantity, while got {x2}'
        return func(x1, x2.mantissa, *args, **kwargs)
    else:
        return func(x1, x2, *args, **kwargs)


@set_module_as('saiunit.math')
def fmod(x1: Union[Quantity, jax.typing.ArrayLike],
         x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
    """
    Return the element-wise remainder of division.

    Parameters
    ----------
    x1: array_like, Quantity
      Input array.
    x2: array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
    """
    return _fun_keep_unit_binary(jnp.fmod, x1, x2)


@set_module_as('saiunit.math')
def mod(x1: Union[Quantity, jax.typing.ArrayLike], x2: Union[Quantity, jax.Array]) -> Union[Quantity, jax.Array]:
    """
    Return the element-wise modulus of division.

    Parameters
    ----------
    x1: array_like, Quantity
      Input array.
    x2: array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
    """
    return _fun_keep_unit_binary(jnp.mod, x1, x2)


@set_module_as('saiunit.math')
def copysign(
    x1: Union[Quantity, jax.typing.ArrayLike],
    x2: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    """
    Return a copy of the first array elements with the sign of the second array.

    Parameters
    ----------
    x1: array_like, Quantity
      Input array.
    x2: array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
    """
    x2 = x2.mantissa if isinstance(x2, Quantity) else x2
    return _fun_keep_unit_unary(jnp.copysign, x1, x2)


@set_module_as('saiunit.math')
def maximum(
    x1: Union[Quantity, jax.typing.ArrayLike],
    x2: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    """
    Element-wise maximum of array elements.

    Parameters
    ----------
    x1: array_like, Quantity
      Input array.
    x2: array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
    """
    return _fun_keep_unit_binary(jnp.maximum, x1, x2)


@set_module_as('saiunit.math')
def minimum(
    x1: Union[Quantity, jax.typing.ArrayLike],
    x2: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    """
    Element-wise minimum of array elements.

    Parameters
    ----------
    x1: array_like, Quantity
      Input array.
    x2: array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
    """
    return _fun_keep_unit_binary(jnp.minimum, x1, x2)


@set_module_as('saiunit.math')
def fmax(
    x1: Union[Quantity, jax.typing.ArrayLike],
    x2: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    """
    Element-wise maximum of array elements ignoring NaNs.

    Parameters
    ----------
    x1: array_like, Quantity
      Input array.
    x2: array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
    """
    return _fun_keep_unit_binary(jnp.fmax, x1, x2)


@set_module_as('saiunit.math')
def fmin(
    x1: Union[Quantity, jax.typing.ArrayLike],
    x2: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    """
    Element-wise minimum of array elements ignoring NaNs.

    Parameters
    ----------
    x1: array_like, Quantity
      Input array.
    x2: array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
    """
    return _fun_keep_unit_binary(jnp.fmin, x1, x2)


@set_module_as('saiunit.math')
def lcm(
    x1: Union[Quantity, jax.typing.ArrayLike],
    x2: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    """
    Return the least common multiple of `x1` and `x2`.

    Parameters
    ----------
    x1: array_like, Quantity
      Input array.
    x2: array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
    """
    return _fun_keep_unit_binary(jnp.lcm, x1, x2)


@set_module_as('saiunit.math')
def gcd(x1: Union[Quantity, jax.typing.ArrayLike],
        x2: Union[Quantity, jax.typing.ArrayLike], ) -> Union[Quantity, jax.Array]:
    """
    Return the greatest common divisor of `x1` and `x2`.

    Parameters
    ----------
    x1: array_like, Quantity
      Input array.
    x2: array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array, Quantity
      Quantity if `x1` and `x2` are Quantities that have the same unit, else an array.
    """
    return _fun_keep_unit_binary(jnp.gcd, x1, x2)


@set_module_as('saiunit.math')
def add(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    """
    Add arguments element-wise.

    Parameters
    ----------
    x, y : array_like, Quantity
      The arrays to be added.
      If ``x.shape != y.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).

    Returns
    -------
    add : ndarray or scalar
      The sum of `x` and `y`, element-wise.
      This is a scalar if both `x` and `y` are scalars.
    """
    return _fun_keep_unit_binary(jnp.add, x, y)


@set_module_as('saiunit.math')
def subtract(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    """
    subtract(x1, x2, /, out=None, *, where=True, casting='same_kind',
    order='K', dtype=None, subok=True[, signature, extobj])

    Subtract arguments, element-wise.

    Parameters
    ----------
    x, y : array_like
      The arrays to be subtracted from each other.
      If ``x.shape != y.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).

    Returns
    -------
    subtract : ndarray
      The difference of `x` and `y`, element-wise.
      This is a scalar if both `x` and `y` are scalars.
    """
    return _fun_keep_unit_binary(jnp.subtract, x, y)


@set_module_as('saiunit.math')
def remainder(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.Array]:
    """
    Returns the element-wise remainder of division.

    Computes the remainder complementary to the `floor_divide` function.  It is
    equivalent to the Python modulus operator``x1 % x2`` and has the same sign
    as the divisor `x2`. The MATLAB function equivalent to ``np.remainder``
    is ``mod``.

    Parameters
    ----------
    x : array_like, Quantity
      Dividend array.
    y : array_like, Quantity
      Divisor array.
      If ``x1.shape != x2.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).

    Returns
    -------
    out : ndarray, Quantity
      The element-wise remainder of the quotient ``floor_divide(x1, x2)``.
      This is a scalar if both `x1` and `x2` are scalars.

      This is a Quantity if division of `x1` by `x2` is not dimensionless.
    """
    return _fun_keep_unit_binary(jnp.remainder, x, y)


@set_module_as('saiunit.math')
def nextafter(
    x: Union[Quantity, jax.typing.ArrayLike],
    y: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    """
    nextafter(x, y, /, out=None, *, where=True, casting='same_kind',
    order='K', dtype=None, subok=True[, signature, extobj])

    Return the next floating-point value after x1 towards x2, element-wise.

    Parameters
    ----------
    x : array_like, Quantity
      Values to find the next representable value of.
    y : array_like, Quantity
      The direction where to look for the next representable value of `x`.
      If ``x.shape != y.shape``, they must be broadcastable to a common
      shape (which becomes the shape of the output).

    Returns
    -------
    out : ndarray or scalar
      The next representable values of `x` in the direction of `y`.
      This is a scalar if both `x` and `y` are scalars.
    """
    return _fun_keep_unit_binary(jnp.nextafter, x, y)


# math funcs keep unit (n-ary)
# ----------------------------
@set_module_as('saiunit.math')
def interp(
    x: Union[Quantity, jax.typing.ArrayLike],
    xp: Union[Quantity, jax.typing.ArrayLike],
    fp: Union[Quantity, jax.typing.ArrayLike],
    left: Union[Quantity, jax.typing.ArrayLike] = None,
    right: Union[Quantity, jax.typing.ArrayLike] = None,
    period: Union[Quantity, jax.typing.ArrayLike] = None
) -> Union[Quantity, jax.Array]:
    """
    One-dimensional linear interpolation.

    Args:
      x: array_like, Quantity
      xp: array_like, Quantity
      fp: array_like, Quantity
      left: array_like, Quantity, optional
      right: array_like, Quantity, optional
      period: array_like, Quantity, optional

    Returns:
      Union[jax.Array, Quantity]: Quantity if `x`, `xp`, and `fp` are Quantities that have the same unit, else an array.
    """
    x_unit = get_unit(x)
    fp, y_unit = split_mantissa_unit(fp)
    x, xp, fp, left, right, period = (
        x.mantissa if isinstance(x, Quantity) else x,
        Quantity(xp).in_unit(x_unit).mantissa,
        fp,
        Quantity(left).in_unit(x_unit).mantissa if left is not None else left,
        Quantity(right).in_unit(x_unit).mantissa if right is not None else right,
        Quantity(period).in_unit(x_unit).mantissa if period is not None else period
    )
    r = jnp.interp(x, xp=xp, fp=fp, left=left, right=right, period=period)
    return maybe_decimal(Quantity(r, unit=y_unit))


@set_module_as('saiunit.math')
def clip(
    a: Union[Quantity, jax.typing.ArrayLike],
    a_min: Union[Quantity, jax.typing.ArrayLike],
    a_max: Union[Quantity, jax.typing.ArrayLike]
) -> Union[Quantity, jax.Array]:
    """
    Clip (limit) the values in an array.

    Args:
      a: array_like, Quantity
      a_min: array_like, Quantity
      a_max: array_like, Quantity

    Returns:
      Union[jax.Array, Quantity]: Quantity if `a`, `a_min`, and `a_max` are Quantities that have the same unit, else an array.
    """
    a_unit = get_unit(a)
    if a_min is not None:
        a_min = Quantity(a_min).in_unit(a_unit).mantissa
    if a_max is not None:
        a_max = Quantity(a_max).in_unit(a_unit).mantissa
    return _fun_keep_unit_unary(jnp.clip, a, min=a_min, max=a_max)


@set_module_as('saiunit.math')
def histogram(
    x: Union[jax.Array, Quantity],
    bins: jax.typing.ArrayLike = 10,
    range: Optional[Sequence[jax.typing.ArrayLike | Quantity]] = None,
    weights: Optional[jax.typing.ArrayLike] = None,
    density: Optional[bool] = None
) -> Tuple[jax.Array, jax.Array | Quantity]:
    """
    Compute the histogram of a set of data.

    Parameters
    ----------
    x : array_like, Quantity
      Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars or str, optional
      If `bins` is an int, it defines the number of equal-width
      bins in the given range (10, by default). If `bins` is a
      sequence, it defines a monotonically increasing array of bin edges,
      including the rightmost edge, allowing for non-uniform bin widths.

      If `bins` is a string, it defines the method used to calculate the
      optimal bin width, as defined by `histogram_bin_edges`.
    range : (float, float), (Quantity, Quantity) optional
      The lower and upper range of the bins.  If not provided, range
      is simply ``(a.min(), a.max())``.  Values outside the range are
      ignored. The first element of the range must be less than or
      equal to the second. `range` affects the automatic bin
      computation as well. While bin width is computed to be optimal
      based on the actual data within `range`, the bin count will fill
      the entire range including portions containing no data.
    weights : array_like, optional
      An array of weights, of the same shape as `a`.  Each value in
      `a` only contributes its associated weight towards the bin count
      (instead of 1). If `density` is True, the weights are
      normalized, so that the integral of the density over the range
      remains 1.
    density : bool, optional
      If ``False``, the result will contain the number of samples in
      each bin. If ``True``, the result is the value of the
      probability *density* function at the bin, normalized such that
      the *integral* over the range is 1. Note that the sum of the
      histogram values will not be equal to 1 unless bins of unity
      width are chosen; it is not a probability *mass* function.

    Returns
    -------
    hist : array
      The values of the histogram. See `density` and `weights` for a
      description of the possible semantics.
    bin_edges : array of dtype float
      Return the bin edges ``(length(hist)+1)``.
    """
    unit = UNITLESS
    if isinstance(x, Quantity):
        unit = x.unit
        x = x.mantissa
    if range is not None:
        range = (
            Quantity(range[0]).in_unit(unit).mantissa,
            Quantity(range[1]).in_unit(unit).mantissa
        )
    hist, bin_edges = jnp.histogram(x, bins, range=range, weights=weights, density=density)
    if unit.is_unitless:
        return hist, bin_edges
    return hist, Quantity(bin_edges, unit=unit)


@set_module_as('saiunit.math')
def compress(
    condition: jax.Array,
    a: Union[jax.Array, Quantity],
    axis: Optional[int] = None,
    *,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike] = None,
) -> Union[jax.Array, Quantity]:
    """
    Return selected slices of a quantity or an array along given axis.

    Parameters
    ----------
    condition : array_like, Quantity
      An array of boolean values that selects which slices to return. If the shape of condition is not the same as `a`,
      it must be broadcastable to `a`.
    a : array_like, Quantity
      Array from which to extract a part.
    axis : int or None, optional
      The axis along which to take slices. If axis is None, `condition` must be a 1-D array with the same length as `a`.
      If axis is an integer, `condition` must be broadcastable to the same shape as `a` along all axes except `axis`.
    size : int, optional
      The length of the returned axis. By default, the length of the input array along the axis is used.
    fill_value : scalar, optional
      The value to use for elements in the output array that are not selected. If None, the output array has the same
      type as `a` and is filled with zeros.

    Returns
    -------
    res : ndarray, Quantity
      A new array that has the same number of dimensions as `a`, and the same shape as `a` with axis `axis` removed.
    """
    if isinstance(condition, Quantity):
        assert condition.is_unitless, 'condition must be unitless.'
        condition = condition.mantissa
    a_unit = get_unit(a)
    if fill_value is not None:
        fill_value = Quantity(fill_value).in_unit(a_unit).mantissa
    else:
        fill_value = 0
    return _fun_keep_unit_unary(functools.partial(jnp.compress, condition),
                                a, axis=axis, size=size, fill_value=fill_value)


@set_module_as('saiunit.math')
def extract(
    condition: jax.Array,
    arr: Union[jax.Array, Quantity],
    *,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike | Quantity] = None,
) -> jax.Array | Quantity:
    """
    Return the elements of an array that satisfy some condition.

    Parameters
    ----------
    condition : array_like, Quantity
      An array of boolean values that selects which elements to extract.
    arr : array_like, Quantity
      The array from which to extract elements.
    size: int
      optional static size for output. Must be specified in order for ``extract``
      to be compatible with JAX transformations like :func:`~jax.jit` or :func:`~jax.vmap`.
    fill_value: array_like
      if ``size`` is specified, fill padded entries with this value (default: 0).

    Returns
    -------
    res : ndarray
      The extracted elements. The shape of `res` is the same as that of `condition`.
    """
    if isinstance(condition, Quantity):
        assert condition.is_unitless, 'condition must be unitless.'
        condition = condition.mantissa
    a_unit = get_unit(arr)
    if fill_value is not None:
        fill_value = Quantity(fill_value).in_unit(a_unit).mantissa
    else:
        fill_value = 0
    return _fun_keep_unit_unary(functools.partial(jnp.extract, condition),
                                arr, size=size, fill_value=fill_value)


@set_module_as('saiunit.math')
def take(
    a: Union[Quantity, jax.typing.ArrayLike],
    indices: Union[Quantity, jax.typing.ArrayLike],
    axis: Optional[int] = None,
    mode: Optional[str] = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value: Optional[Union[Quantity, jax.typing.ArrayLike]] = None,
) -> Union[Quantity, jax.Array]:
    """
    Take elements from an array along an axis.

    When axis is not None, this function does the same thing as "fancy"
    indexing (indexing arrays using arrays); however, it can be easier to use
    if you need elements along a given axis. A call such as
    ``np.take(arr, indices, axis=3)`` is equivalent to
    ``arr[:,:,:,indices,...]``.

    Explained without fancy indexing, this is equivalent to the following use
    of `ndindex`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of
    indices::

      Ni, Nk = a.shape[:axis], a.shape[axis+1:]
      Nj = indices.shape
      for ii in ndindex(Ni):
          for jj in ndindex(Nj):
              for kk in ndindex(Nk):
                  out[ii + jj + kk] = a[ii + (indices[jj],) + kk]

    Parameters
    ----------
    a : array_like (Ni..., M, Nk...)
      The source array.
    indices : array_like (Nj...)
      The indices of the values to extract.

      Also allow scalars for indices.
    axis : int, optional
      The axis over which to select values. By default, the flattened
      input array is used.
    mode : string, default="fill"
      Out-of-bounds indexing mode. The default mode="fill" returns invalid values
      (e.g. NaN) for out-of bounds indices (see also ``fill_value`` below).
      For more discussion of mode options, see :attr:`jax.numpy.ndarray.at`.
    fill_value : optional
      The fill value to return for out-of-bounds slices when mode is 'fill'. Ignored
      otherwise. Defaults to NaN for inexact types, the largest negative value for
      signed types, the largest positive value for unsigned types, and True for booleans.
    unique_indices : bool, default=False
      If True, the implementation will assume that the indices are unique,
      which can result in more efficient execution on some backends.
    indices_are_sorted : bool, default=False
      If True, the implementation will assume that the indices are sorted in
      ascending order, which can lead to more efficient execution on some backends.

    Returns
    -------
    out : ndarray (Ni..., Nj..., Nk...)
      The returned array has the same type as `a`.
    """
    if isinstance(a, Quantity):
        return a.take(indices, axis=axis, mode=mode, unique_indices=unique_indices,
                      indices_are_sorted=indices_are_sorted, fill_value=fill_value)
    else:
        return jnp.take(a, indices, axis=axis, mode=mode, unique_indices=unique_indices,
                        indices_are_sorted=indices_are_sorted, fill_value=fill_value)


@set_module_as('saiunit.math')
def select(
    condlist: list[Union[jax.typing.ArrayLike]],
    choicelist: Union[Quantity, jax.typing.ArrayLike],
    default: int = 0
) -> Union[Quantity, jax.Array]:
    """
    Return an array drawn from elements in choicelist, depending on conditions.

    Parameters
    ----------
    condlist : list of bool ndarrays
      The list of conditions which determine from which array in `choicelist`
      the output elements are taken. When multiple conditions are satisfied,
      the first one encountered in `condlist` is used.
    choicelist : list of ndarrays or Quantity
      The list of arrays from which the output elements are taken. It has
      to be of the same length as `condlist`.
    default : scalar, optional
      The element inserted in `output` when all conditions evaluate to False.

    Returns
    -------
    output : ndarray, Quantity
      The output at position m is the m-th element of the array in
      `choicelist` where the m-th element of the corresponding array in
      `condlist` is True.
    """
    for cond in condlist:
        assert not isinstance(cond, Quantity), "condlist should not contain Quantity."
    return _fun_keep_unit_sequence(functools.partial(jnp.select, condlist), choicelist, default=default)


@set_module_as('saiunit.math')
def where(condition, x=None, y=None, /, *, size=None, fill_value=None):
    """
    Return elements chosen from `x` or `y` depending on `condition`.

    .. note::
      When only `condition` is provided, this function is a shorthand for
      ``np.asarray(condition).nonzero()``. Using `nonzero` directly should be
      preferred, as it behaves correctly for subclasses. The rest of this
      documentation covers only the case where all three arguments are
      provided.

    Parameters
    ----------
    condition : array_like, bool,
      Where True, yield `x`, otherwise yield `y`.
    x, y : array_like, Quantity
      Values from which to choose. `x`, `y` and `condition` need to be
      broadcastable to some shape.
    size : int, optional
      The length of the output array. If `size` is not None, the output array will have the length of `size`.
    fill_value : scalar, Quantity, optional
      The value to use for missing values. If `fill_value` is not None, the output array will have the length of `size`.

    Returns
    -------
    out : ndarray
      An array with elements from `x` where `condition` is True, and elements
      from `y` elsewhere.

    See Also
    --------
    choose
    nonzero : The function that is called when x and y are omitted
    """
    assert not isinstance(condition, Quantity), "condition should not be a Quantity."
    if x is None and y is None:
        assert not isinstance(fill_value, Quantity), "fill_value should not be a Quantity."
        return jnp.where(condition, size=size, fill_value=fill_value)

    assert size is None and fill_value is None, "size and fill_value are only supported when x and y are not None."
    if isinstance(x, Quantity) and isinstance(y, Quantity):
        y = y.in_unit(x.unit)
        return Quantity(jnp.where(condition, x.mantissa, y.mantissa), unit=x.unit)
    elif isinstance(x, Quantity):
        assert x.is_unitless, 'x must be unitless when y is not a Quantity.'
        x = x.mantissa
    elif isinstance(y, Quantity):
        assert y.is_unitless, 'y must be unitless when x is not a Quantity.'
        y = y.mantissa
    return jnp.where(condition, x, y)


@set_module_as('saiunit.math')
def unique(
    a: Union[jax.Array, Quantity],
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: Optional[int] = None,
    *,
    equal_nan: bool = False,
    size: Optional[int] = None,
    fill_value: Optional[jax.typing.ArrayLike, Quantity] = None
) -> Sequence[jax.Array | Quantity] | jax.Array | Quantity:
    """
    Find the unique elements of a quantity or an array.

    Parameters
    ----------
    a : array_like, Quantity
      Input array.
    return_index : bool, optional
      If True, also return the indices of `a` (along the specified axis, if provided) that result in the unique array.
    return_inverse : bool, optional
      If True, also return the indices of the unique array (for the specified axis, if provided)
      that can be used to reconstruct `a`.
    return_counts : bool, optional
      If True, also return the number of times each unique item appears in `a`.
    axis : int, optional
      The axis along which to operate. If None, the array is flattened before use. Default is None.
    equal_nan : bool, optional
      Whether to compare NaN's as equal. If True, NaN's in `a` will be considered equal to each other in the unique array.
    size : int, optional
      The length of the output array. If `size` is not None, the output array will have the length of `size`.
    fill_value : scalar, optional
      The value to use for missing values. If `fill_value` is not None, the output array will have the length of `size`.

    Returns
    -------
    res : ndarray, Quantity
      The sorted unique values.
    """
    a_unit = get_unit(a)
    if fill_value is not None:
        fill_value = Quantity(fill_value).in_unit(a_unit).mantissa
    if isinstance(a, Quantity):
        result = jnp.unique(a.mantissa,
                            return_index=return_index,
                            return_inverse=return_inverse,
                            return_counts=return_counts,
                            axis=axis,
                            equal_nan=equal_nan,
                            size=size,
                            fill_value=fill_value)
        if isinstance(result, tuple):
            output = []
            output.append(Quantity(result[0], unit=a_unit))
            for r in result[1:]:
                output.append(r)
            return tuple(output)
        else:
            return Quantity(result, unit=a_unit)
    else:
        return jnp.unique(a,
                          return_index=return_index,
                          return_inverse=return_inverse,
                          return_counts=return_counts,
                          axis=axis,
                          equal_nan=equal_nan,
                          size=size,
                          fill_value=fill_value)



@set_module_as('saiunit.math')
def round(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array | Quantity:
    """
    Round an array to the nearest integer.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array
    """
    return _fun_keep_unit_unary(jnp.round, x)


@set_module_as('saiunit.math')
def around(
    x: Union[Quantity, jax.typing.ArrayLike],
    decimals: int = 0,
) -> jax.Array | Quantity:
    """
    Round an array to the nearest integer.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.
    decimals : int, optional
      Number of decimal places to round to (default is 0).

    Returns
    -------
    out : jax.Array
    """
    return _fun_keep_unit_unary(jnp.around, x, decimals=decimals)


@set_module_as('saiunit.math')
def rint(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    """
    Round an array to the nearest integer.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array
    """
    return _fun_keep_unit_unary(jnp.rint, x)


@set_module_as('saiunit.math')
def floor(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array | Quantity:
    """
    Return the floor of the argument.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array
    """
    return _fun_keep_unit_unary(jnp.floor, x)


@set_module_as('saiunit.math')
def ceil(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array | Quantity:
    """
    Return the ceiling of the argument.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array
    """
    return _fun_keep_unit_unary(jnp.ceil, x)


@set_module_as('saiunit.math')
def trunc(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array | Quantity:
    """
    Return the truncated value of the argument.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array
    """
    return _fun_keep_unit_unary(jnp.trunc, x)


@set_module_as('saiunit.math')
def fix(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array | Quantity:
    """
    Return the nearest integer towards zero.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    out : jax.Array
    """
    return _fun_keep_unit_unary(jnp.fix, x)


@set_module_as('saiunit.math')
def modf(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> Tuple[jax.Array | Quantity, jax.Array | Quantity]:
    """
    Return the fractional and integer parts of the array elements.

    Parameters
    ----------
    x : array_like, Quantity
      Input array.

    Returns
    -------
    The fractional and integral parts of the input, both with the same dimension.
    """
    if isinstance(x, Quantity):
        return jax.tree.map(lambda y: Quantity(y, unit=x.unit), jnp.modf(x.mantissa))
    return jnp.modf(x)
