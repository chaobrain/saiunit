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

from collections.abc import Sequence
from typing import (Union, TypeVar, Any)

import jax
import jax.numpy as jnp
import numpy as np

from .._base import (Unit,
                     Quantity,
                     get_unit,
                     is_unitless)
from .._misc import set_module_as, maybe_custom_array_tree, maybe_custom_array

T = TypeVar("T")

__all__ = [
    'bool_',
    'uint2',
    'uint4',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'int2',
    'int4',
    'int8',
    'int16',
    'int32',
    'int64',
    'float4_e2m1fn',
    'float8_e3m4',
    'float8_e4m3',
    'float8_e8m0fnu',
    'float8_e4m3fn',
    'float8_e4m3fnuz',
    'float8_e5m2',
    'float8_e5m2fnuz',
    'float8_e4m3b11fnuz',
    'bfloat16',
    'float16',
    'float32',
    'float64',
    'complex64',
    'complex128',
    'int_',
    'uint',
    'float_',
    'complex_',
    'single',
    'double',
    'csingle',
    'cdouble',

    # constants
    'e', 'pi', 'inf', 'nan', 'euler_gamma', 'inexact',

    # data types
    'dtype', 'finfo', 'iinfo', 'newaxis',

    # getting attribute funcs
    'is_quantity', 'issubdtype', 'result_type',
    'ndim', 'isreal', 'isscalar', 'isfinite', 'isinf',
    'isnan', 'shape', 'size', 'get_dtype',
    'is_float', 'is_int', 'broadcast_shapes',

    # more
    'gradient',

    # window funcs
    'bartlett', 'blackman', 'hamming', 'hanning', 'kaiser',
]

bool_ = jnp.bool_
uint2 = jnp.uint2
uint4 = jnp.uint4
uint8 = jnp.uint8
uint16 = jnp.uint16
uint32 = jnp.uint32
uint64 = jnp.uint64
int2 = jnp.int2
int4 = jnp.int4
int8 = jnp.int8
int16 = jnp.int16
int32 = jnp.int32
int64 = jnp.int64
float4_e2m1fn = jnp.float4_e2m1fn
float8_e3m4 = jnp.float8_e3m4
float8_e4m3 = jnp.float8_e4m3
float8_e8m0fnu = jnp.float8_e8m0fnu
float8_e4m3fn = jnp.float8_e4m3fn
float8_e4m3fnuz = jnp.float8_e4m3fnuz
float8_e5m2 = jnp.float8_e5m2
float8_e5m2fnuz = jnp.float8_e5m2fnuz
float8_e4m3b11fnuz = jnp.float8_e4m3b11fnuz
bfloat16 = jnp.bfloat16
float16 = jnp.float16
float32 = single = jnp.float32
float64 = double = jnp.float64
complex64 = csingle = jnp.complex64
complex128 = cdouble = jnp.complex128
int_ = jnp.int_
uint = jnp.uint
float_ = jnp.float_
complex_ = jnp.complex_


def _removechars(s, chars):
    return s.translate(str.maketrans(dict.fromkeys(chars)))


# constants
# ---------
e = np.e
pi = np.pi
inf = np.inf
nan = np.nan
inexact = jnp.inexact
euler_gamma = np.euler_gamma

# data types
# ----------
dtype = jnp.dtype
newaxis = jnp.newaxis


def is_quantity(x: Any) -> bool:
    """
    Check if x is a Quantity.

    Parameters
    ----------
    x : Any
        The input object.

    Returns
    -------
    bool
        A boolean value indicating if x is a Quantity.
    """
    x = maybe_custom_array(x)
    return isinstance(x, Quantity)


@set_module_as('saiunit.math')
def issubdtype(a: T, b: T) -> bool:
    """
    Returns True if first argument is a typecode lower/equal in type hierarchy.

    Args:
      a: dtype
      b: dtype

    Returns:
      bool
    """
    return jnp.issubdtype(a, b)


@set_module_as('saiunit.math')
def result_type(*args):
    """
    Determine the result data type.

    Args:
      *args: array_like

    Returns:
      dtype: dtype
    """
    args = maybe_custom_array_tree(args)
    return jnp.result_type(*jax.tree.leaves(args))


@set_module_as('saiunit.math')
def ndim(a: Union[Quantity, jax.typing.ArrayLike]) -> int:
    """
    Return the number of dimensions of an array.

    Args:
      a: array_like, Quantity

    Returns:
      Union[jax.Array, Quantity]: int
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return a.ndim
    else:
        return jnp.ndim(a)


@set_module_as('saiunit.math')
def isreal(a: Union[Quantity, jax.typing.ArrayLike]) -> jax.Array:
    """
    Return True if the input array is real.

    Args:
      a: array_like, Quantity

    Returns:
      Union[jax.Array, Quantity]: boolean array
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return a.isreal
    else:
        return jnp.isreal(a)


@set_module_as('saiunit.math')
def isscalar(a: Union[Quantity, jax.typing.ArrayLike]) -> bool:
    """
    Return True if the input is a scalar.

    Args:
      a: array_like, Quantity

    Returns:
      Union[jax.Array, Quantity]: boolean array
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return a.isscalar
    else:
        return jnp.isscalar(a)


@set_module_as('saiunit.math')
def isfinite(a: Union[Quantity, jax.typing.ArrayLike]) -> jax.Array:
    """
    Return each element of the array is finite or not.

    Args:
      a: array_like, Quantity

    Returns:
      Union[jax.Array, Quantity]: boolean array
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return a.isfinite
    else:
        return jnp.isfinite(a)


@set_module_as('saiunit.math')
def isinf(a: Union[Quantity, jax.typing.ArrayLike]) -> jax.Array:
    """
    Return each element of the array is infinite or not.

    Args:
      a: array_like, Quantity

    Returns:
      Union[jax.Array, Quantity]: boolean array
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return a.isinf
    else:
        return jnp.isinf(a)


@set_module_as('saiunit.math')
def isnan(a: Union[Quantity, jax.typing.ArrayLike]) -> jax.Array:
    """
    Return each element of the array is NaN or not.

    Args:
      a: array_like, Quantity

    Returns:
      Union[jax.Array, Quantity]: boolean array
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return a.isnan
    else:
        return jnp.isnan(a)


@set_module_as('saiunit.math')
def shape(a: Union[Quantity, jax.typing.ArrayLike]) -> tuple[int, ...]:
    """
    Return the shape of an array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    shape : tuple of ints
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    See Also
    --------
    len : ``len(a)`` is equivalent to ``np.shape(a)[0]`` for N-D arrays with
          ``N>=1``.
    ndarray.shape : Equivalent array method.

    Examples
    --------
    >>> saiunit.math.shape(saiunit.math.eye(3))
    (3, 3)
    >>> saiunit.math.shape([[1, 3]])
    (1, 2)
    >>> saiunit.math.shape([0])
    (1,)
    >>> saiunit.math.shape(0)
    ()

    """
    a = maybe_custom_array(a)
    if isinstance(a, (Quantity, jax.Array, np.ndarray)):
        return a.shape
    else:
        return np.shape(a)


@set_module_as('saiunit.math')
def size(a: Union[Quantity, jax.typing.ArrayLike], axis: int = None) -> int:
    """
    Return the number of elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which the elements are counted.  By default, give
        the total number of elements.

    Returns
    -------
    element_count : int
        Number of elements along the specified axis.

    See Also
    --------
    shape : dimensions of array
    Array.shape : dimensions of array
    Array.size : number of elements in array

    Examples
    --------
    >>> a = Quantity([[1,2,3], [4,5,6]])
    >>> saiunit.math.size(a)
    6
    >>> saiunit.math.size(a, 1)
    3
    >>> saiunit.math.size(a, 0)
    2
    """
    a = maybe_custom_array(a)
    if isinstance(a, (Quantity, jax.Array, np.ndarray)):
        if axis is None:
            return a.size
        else:
            return a.shape[axis]
    else:
        return np.size(a, axis=axis)


@set_module_as('saiunit.math')
def finfo(a: Union[Quantity, jax.typing.ArrayLike]) -> jnp.finfo:
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return jnp.finfo(a.mantissa)
    else:
        return jnp.finfo(a)


@set_module_as('saiunit.math')
def iinfo(a: Union[Quantity, jax.typing.ArrayLike]) -> jnp.iinfo:
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return jnp.iinfo(a.mantissa)
    else:
        return jnp.iinfo(a)


@set_module_as('saiunit.math')
def broadcast_shapes(*shapes):
    """
    Broadcast a sequence of array shapes.

    Parameters
    ----------
    *shapes : tuple of ints
        The shapes of the arrays to broadcast.

    Returns
    -------
    broadcast_shape : tuple of ints
        The shape of the broadcasted arrays.
    """
    return jnp.broadcast_shapes(*shapes)


environ = None  # type: ignore[assignment]


@set_module_as('brainstate.math')
def get_dtype(a):
    """
    Get the dtype of a.
    """
    a = maybe_custom_array(a)
    if hasattr(a, 'dtype'):
        return a.dtype
    else:
        global environ
        if isinstance(a, bool):
            return bool
        elif isinstance(a, int):
            if environ is None:
                from brainstate import environ
            return environ.ditype()
        elif isinstance(a, float):
            if environ is None:
                from brainstate import environ
            return environ.dftype()
        elif isinstance(a, complex):
            if environ is None:
                from brainstate import environ
            return environ.dctype()
        else:
            raise ValueError(f'Can not get dtype of {a}.')


@set_module_as('brainstate.math')
def is_float(array):
    """
    Check if the array is a floating point array.

    Args:
      array: The input array.

    Returns:
      A boolean value indicating if the array is a floating point array.
    """
    array = maybe_custom_array(array)
    return jnp.issubdtype(get_dtype(array), jnp.floating)


@set_module_as('brainstate.math')
def is_int(array):
    """
    Check if the array is an integer array.

    Args:
      array: The input array.

    Returns:
      A boolean value indicating if the array is an integer array.
    """
    array = maybe_custom_array(array)
    return jnp.issubdtype(get_dtype(array), jnp.integer)


@set_module_as('saiunit.math')
def gradient(
    f: Union[jax.typing.ArrayLike, Quantity],
    *varargs: Union[jax.typing.ArrayLike, Quantity],
    axis: Union[int, Sequence[int], None] = None,
    edge_order: Union[int, None] = None,
) -> Union[jax.Array, list[jax.Array], Quantity, list[Quantity]]:
    """
    Computes the gradient of a scalar field.

    Return the gradient of an N-dimensional array.

    The gradient is computed using second order accurate central differences
    in the interior points and either first or second order accurate one-sides
    (forward or backwards) differences at the boundaries.
    The returned gradient hence has the same shape as the input array.

    Parameters
    ----------
    f : array_like, Quantity
      An N-dimensional array containing samples of a scalar function.
    varargs : list of scalar or array, optional
      Spacing between f values. Default unitary spacing for all dimensions.
      Spacing can be specified using:

      1. single scalar to specify a sample distance for all dimensions.
      2. N scalars to specify a constant sample distance for each dimension.
         i.e. `dx`, `dy`, `dz`, ...
      3. N arrays to specify the coordinates of the values along each
         dimension of F. The length of the array must match the size of
         the corresponding dimension
      4. Any combination of N scalars/arrays with the meaning of 2. and 3.

      If `axis` is given, the number of varargs must equal the number of axes.
      Default: 1.
    edge_order : {1, 2}, optional
      Gradient is calculated using N-th order accurate differences
      at the boundaries. Default: 1.
    axis : None or int or tuple of ints, optional
      Gradient is calculated only along the given axis or axes
      The default (axis = None) is to calculate the gradient for all the axes
      of the input array. axis may be negative, in which case it counts from
      the last to the first axis.

    Returns
    -------
    gradient : ndarray or list of ndarray or Quantity
      A list of ndarrays (or a single ndarray if there is only one dimension)
      corresponding to the derivatives of f with respect to each dimension.
      Each derivative has the same shape as f.
    """
    f, varargs = maybe_custom_array_tree((f, varargs))
    if edge_order is not None:
        raise NotImplementedError("The 'edge_order' argument to jnp.gradient is not supported.")

    if len(varargs) == 0:
        if isinstance(f, Quantity) and not is_unitless(f):
            return Quantity(jnp.gradient(f.mantissa, axis=axis), unit=f.unit)
        else:
            return jnp.gradient(f)
    elif len(varargs) == 1:
        unit = get_unit(f) / get_unit(varargs[0])
        if isinstance(unit, Unit) and unit.is_unitless:
            return jnp.gradient(f, varargs[0], axis=axis)
        else:
            return [Quantity(r, unit=unit) for r in jnp.gradient(f.mantissa, Quantity(varargs[0]).mantissa, axis=axis)]
    else:
        unit_list = [get_unit(f) / get_unit(v) for v in varargs]
        f = f.mantissa if isinstance(f, Quantity) else f
        varargs = [v.mantissa if isinstance(v, Quantity) else v for v in varargs]
        result_list = jnp.gradient(f, *varargs, axis=axis)
        return [(Quantity(r, unit=unit) if unit is not None else r) for r, unit in zip(result_list, unit_list)]


# window funcs
# ------------

bartlett = jnp.bartlett
blackman = jnp.blackman
hamming = jnp.hamming
hanning = jnp.hanning
kaiser = jnp.kaiser
