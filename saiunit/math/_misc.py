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

from __future__ import annotations

from collections.abc import Sequence
from typing import (Union, TypeVar, Any)

from saiunit._jax_compat import jax, jnp, ArrayLike
import numpy as np

from saiunit._base_unit import Unit
from saiunit._base_getters import get_unit, is_unitless
from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as, maybe_custom_array_tree, maybe_custom_array

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

def _dtype_or_none(attr_name: str):
    """Return ``jnp.<attr>`` if JAX provides it, else fall back to ``np.<attr>``.

    Some dtypes (``uint2``, ``uint4``, ``int2``, ``int4``, ``bfloat16``) exist
    only in JAX. Without JAX they evaluate to ``None`` and using them as a
    dtype argument will raise downstream. Standard widths (``float32`` etc.)
    transparently fall back to NumPy.
    """
    src = jnp if jnp is not None else None
    if src is not None and hasattr(src, attr_name):
        return getattr(src, attr_name)
    return getattr(np, attr_name, None)


bool_ = _dtype_or_none('bool_')
uint2 = _dtype_or_none('uint2')
uint4 = _dtype_or_none('uint4')
uint8 = _dtype_or_none('uint8')
uint16 = _dtype_or_none('uint16')
uint32 = _dtype_or_none('uint32')
uint64 = _dtype_or_none('uint64')
int2 = _dtype_or_none('int2')
int4 = _dtype_or_none('int4')
int8 = _dtype_or_none('int8')
int16 = _dtype_or_none('int16')
int32 = _dtype_or_none('int32')
int64 = _dtype_or_none('int64')
bfloat16 = _dtype_or_none('bfloat16')
float16 = _dtype_or_none('float16')
float32 = single = _dtype_or_none('float32')
float64 = double = _dtype_or_none('float64')
complex64 = csingle = _dtype_or_none('complex64')
complex128 = cdouble = _dtype_or_none('complex128')
int_ = _dtype_or_none('int_')
uint = _dtype_or_none('uint')
float_ = _dtype_or_none('float_')
complex_ = _dtype_or_none('complex_')


def _removechars(s, chars):
    return s.translate(str.maketrans(dict.fromkeys(chars)))


# constants
# ---------
e = np.e  # type: ignore[misc]
pi = np.pi
inf = np.inf
nan = np.nan
inexact = _dtype_or_none('inexact')
euler_gamma = np.euler_gamma

# data types
# ----------
dtype = (jnp.dtype if jnp is not None else np.dtype)
newaxis = (jnp.newaxis if jnp is not None else np.newaxis)


def is_quantity(x: Any) -> bool:
    """Check whether *x* is a ``Quantity`` instance.

    Parameters
    ----------
    x : Any
        The object to test.

    Returns
    -------
    out : bool
        ``True`` if *x* is a ``Quantity``, ``False`` otherwise.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.math.is_quantity(u.Quantity(1.0, unit=u.meter))
        True
        >>> u.math.is_quantity(1.0)
        False
    """
    x = maybe_custom_array(x)
    return isinstance(x, Quantity)


@set_module_as('saiunit.math')
def issubdtype(a: T, b: T) -> bool:
    """Check if a dtype is a sub-dtype of another in the type hierarchy.

    Parameters
    ----------
    a : dtype
        First dtype to check.
    b : dtype
        Second dtype (abstract type class or concrete dtype).

    Returns
    -------
    out : bool
        ``True`` if *a* is lower or equal in the type hierarchy to *b*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.issubdtype(jnp.float32, jnp.floating)
        True
        >>> sumath.issubdtype(jnp.int32, jnp.floating)
        False
    """
    return jnp.issubdtype(a, b)  # type: ignore[arg-type]


@set_module_as('saiunit.math')
def result_type(*args):
    """Determine the result dtype from a set of input arrays or dtypes.

    Parameters
    ----------
    *args : array_like or dtype
        Input arrays or dtypes.

    Returns
    -------
    out : dtype
        The result dtype that would arise from operating on the inputs.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.result_type(jnp.float32, jnp.int32)
        dtype('float32')
    """
    args = maybe_custom_array_tree(args)
    return jnp.result_type(*jax.tree.leaves(args))


@set_module_as('saiunit.math')
def ndim(a: Union[Quantity, ArrayLike]) -> int:
    """Return the number of dimensions of an array or ``Quantity``.

    Parameters
    ----------
    a : array_like or Quantity
        Input array.

    Returns
    -------
    out : int
        Number of dimensions.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.ndim(jnp.zeros((2, 3)))
        2
        >>> import saiunit as u
        >>> sumath.ndim(u.Quantity(jnp.zeros((2, 3, 4)), unit=u.meter))
        3
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return a.ndim
    else:
        return jnp.ndim(a)


@set_module_as('saiunit.math')
def isreal(a: Union[Quantity, ArrayLike]) -> jax.Array:
    """Test element-wise whether each element is real (has zero imaginary part).

    Parameters
    ----------
    a : array_like or Quantity
        Input array.

    Returns
    -------
    out : jax.Array
        Boolean array of the same shape as *a*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.isreal(jnp.array([1.0, 2.0 + 0j, 3.0 + 1j]))
        Array([ True,  True, False], dtype=bool)
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return a.isreal
    else:
        return jnp.isreal(a)


@set_module_as('saiunit.math')
def isscalar(a: Union[Quantity, ArrayLike]) -> bool:
    """Return ``True`` if the input is a scalar (zero-dimensional).

    Parameters
    ----------
    a : array_like or Quantity
        Input value.

    Returns
    -------
    out : bool
        ``True`` if *a* is a scalar.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit.math as sumath
        >>> sumath.isscalar(3.14)
        True
        >>> import jax.numpy as jnp
        >>> sumath.isscalar(jnp.array([1, 2]))
        False
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return a.isscalar
    else:
        return jnp.isscalar(a)


@set_module_as('saiunit.math')
def isfinite(a: Union[Quantity, ArrayLike]) -> jax.Array:
    """Test element-wise for finiteness (not inf and not NaN).

    Parameters
    ----------
    a : array_like or Quantity
        Input array.

    Returns
    -------
    out : jax.Array
        Boolean array of the same shape as *a*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.isfinite(jnp.array([1.0, jnp.inf, jnp.nan]))
        Array([ True, False, False], dtype=bool)
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return a.isfinite
    else:
        return jnp.isfinite(a)


@set_module_as('saiunit.math')
def isinf(a: Union[Quantity, ArrayLike]) -> jax.Array:
    """Test element-wise for positive or negative infinity.

    Parameters
    ----------
    a : array_like or Quantity
        Input array.

    Returns
    -------
    out : jax.Array
        Boolean array of the same shape as *a*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.isinf(jnp.array([1.0, jnp.inf, -jnp.inf]))
        Array([False,  True,  True], dtype=bool)
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return a.isinf
    else:
        return jnp.isinf(a)


@set_module_as('saiunit.math')
def isnan(a: Union[Quantity, ArrayLike]) -> jax.Array:
    """Test element-wise for NaN.

    Parameters
    ----------
    a : array_like or Quantity
        Input array.

    Returns
    -------
    out : jax.Array
        Boolean array of the same shape as *a*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.isnan(jnp.array([1.0, jnp.nan, 3.0]))
        Array([False,  True, False], dtype=bool)
    """
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return a.isnan
    else:
        return jnp.isnan(a)


@set_module_as('saiunit.math')
def shape(a: Union[Quantity, ArrayLike]) -> tuple[int, ...]:
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
def size(a: Union[Quantity, ArrayLike], axis: int | None = None) -> int:
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
def finfo(a: Union[Quantity, ArrayLike]) -> jnp.finfo:
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return jnp.finfo(a.mantissa)
    else:
        return jnp.finfo(a)


@set_module_as('saiunit.math')
def iinfo(a: Union[Quantity, ArrayLike]) -> jnp.iinfo:
    a = maybe_custom_array(a)
    if isinstance(a, Quantity):
        return jnp.iinfo(a.mantissa)
    else:
        return jnp.iinfo(a)


@set_module_as('saiunit.math')
def broadcast_shapes(*shapes):
    """Broadcast a sequence of array shapes.

    Parameters
    ----------
    *shapes : tuple of int
        The shapes of the arrays to broadcast.

    Returns
    -------
    broadcast_shape : tuple of int
        The broadcasted shape.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit.math as sumath
        >>> sumath.broadcast_shapes((2, 1), (1, 3))
        (2, 3)
    """
    return jnp.broadcast_shapes(*shapes)


environ = None  # type: ignore[assignment]


@set_module_as('brainstate.math')
def get_dtype(a):
    """Get the dtype of an array, ``Quantity``, or Python scalar.

    Parameters
    ----------
    a : array_like, Quantity, or scalar
        The input whose dtype is to be determined.

    Returns
    -------
    out : dtype
        The data type of *a*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.get_dtype(jnp.array([1.0, 2.0]))
        dtype('float32')
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
                from brainstate import environ  # type: ignore[import-untyped]
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
    """Check if the array has a floating-point dtype.

    Parameters
    ----------
    array : array_like or Quantity
        The input array.

    Returns
    -------
    out : bool
        ``True`` if the array dtype is a floating-point type.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.is_float(jnp.array([1.0]))
        True
        >>> sumath.is_float(jnp.array([1]))
        False
    """
    array = maybe_custom_array(array)
    return jnp.issubdtype(get_dtype(array), jnp.floating)


@set_module_as('brainstate.math')
def is_int(array):
    """Check if the array has an integer dtype.

    Parameters
    ----------
    array : array_like or Quantity
        The input array.

    Returns
    -------
    out : bool
        ``True`` if the array dtype is an integer type.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.is_int(jnp.array([1]))
        True
        >>> sumath.is_int(jnp.array([1.0]))
        False
    """
    array = maybe_custom_array(array)
    return jnp.issubdtype(get_dtype(array), jnp.integer)


@set_module_as('saiunit.math')
def gradient(
    f: Union[ArrayLike, Quantity],
    *varargs: Union[ArrayLike, Quantity],
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> f = jnp.array([1., 2., 4., 7., 11.])
        >>> sumath.gradient(f)
        Array([1. , 1.5, 2.5, 3.5, 4. ], dtype=float32)
    """
    f, varargs = maybe_custom_array_tree((f, varargs))
    if edge_order is not None:
        raise NotImplementedError("The 'edge_order' argument to jnp.gradient is not supported.")

    if len(varargs) == 0:
        if isinstance(f, Quantity) and not is_unitless(f):
            return Quantity(jnp.gradient(f.mantissa, axis=axis), unit=f.unit)
        else:
            return jnp.gradient(f)  # type: ignore[arg-type]
    elif len(varargs) == 1:
        unit = get_unit(f) / get_unit(varargs[0])
        if isinstance(unit, Unit) and unit.is_unitless:
            return jnp.gradient(f, varargs[0], axis=axis)  # type: ignore[arg-type]
        else:
            return [Quantity(r, unit=unit) for r in jnp.gradient(f.mantissa, Quantity(varargs[0]).mantissa, axis=axis)]  # type: ignore[union-attr]
    else:
        unit_list = [get_unit(f) / get_unit(v) for v in varargs]
        f = f.mantissa if isinstance(f, Quantity) else f
        varargs = [v.mantissa if isinstance(v, Quantity) else v for v in varargs]  # type: ignore[assignment]
        result_list = jnp.gradient(f, *varargs, axis=axis)  # type: ignore[arg-type]
        return [(Quantity(r, unit=unit) if unit is not None else r) for r, unit in zip(result_list, unit_list)]


# window funcs
# ------------

def _np_or_jnp_attr(attr_name: str):
    """Window-function lookup: prefer ``jnp.<attr>``; fall back to ``np.<attr>``."""
    if jnp is not None and hasattr(jnp, attr_name):
        return getattr(jnp, attr_name)
    return getattr(np, attr_name)


bartlett = _np_or_jnp_attr('bartlett')
blackman = _np_or_jnp_attr('blackman')
hamming = _np_or_jnp_attr('hamming')
hanning = _np_or_jnp_attr('hanning')
kaiser = _np_or_jnp_attr('kaiser')
