# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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


from typing import Any, Optional, Union, Sequence

import operator
import jax.numpy as jnp
import jax.typing
import numpy as np
from saiunit import math


ArrayLike = jax.typing.ArrayLike

__all__ = [
    'CustomArray',
]


class CustomArray:
    """
    A custom array wrapper providing comprehensive array operations and cross-framework compatibility.

    CustomArray is a versatile array wrapper that provides a unified interface for array
    operations while maintaining compatibility with NumPy, JAX, and PyTorch ecosystems.
    It serves as a drop-in replacement for standard array types with enhanced functionality
    and cross-framework interoperability.

    Attributes
    ----------
    data : Any
        The underlying array data. Can be a NumPy array, JAX array, or any array-like
        object that supports the required operations.

    Properties
    ----------
    dtype : numpy.dtype or equivalent
        Data type of the array elements.
    shape : tuple of ints
        Tuple representing the dimensions of the array.
    ndim : int
        Number of array dimensions.
    size : int
        Total number of elements in the array.
    real : array_like
        Real part of the array elements.
    imag : array_like
        Imaginary part of the array elements. Note: Currently contains a typo,
        accessing 'data.image' instead of 'data.imag'.
    T : array_like
        Transposed view of the array.

    Methods
    -------
    Arithmetic Operations:
        __add__(other), __radd__(other), __iadd__(other)
            Addition operations (+, +=).
        __sub__(other), __rsub__(other), __isub__(other)
            Subtraction operations (-, -=).
        __mul__(other), __rmul__(other), __imul__(other)
            Multiplication operations (*, *=).
        __truediv__(other), __rtruediv__(other), __itruediv__(other)
            True division operations (/, /=).
        __floordiv__(other), __rfloordiv__(other), __ifloordiv__(other)
            Floor division operations (//, //=).
        __pow__(other), __rpow__(other), __ipow__(other)
            Power operations (**, **=).
        __matmul__(other), __rmatmul__(other), __imatmul__(other)
            Matrix multiplication operations (@, @=).

    Comparison Operations:
        __eq__(other), __ne__(other), __lt__(other), __le__(other),
        __gt__(other), __ge__(other)
            Element-wise comparison operations.

    Unary Operations:
        __neg__(), __pos__(), __abs__(), __invert__()
            Unary arithmetic and bitwise operations.

    Statistical Methods:
        mean(axis=None, dtype=None, keepdims=False)
            Compute the arithmetic mean along the specified axis.
        sum(axis=None, dtype=None, keepdims=False, initial=0, where=True)
            Return the sum of array elements over a given axis.
        min(axis=None, keepdims=False), max(axis=None, keepdims=False)
            Return minimum/maximum datas along an axis.
        std(axis=None, dtype=None, ddof=0, keepdims=False)
            Compute the standard deviation along the specified axis.
        var(axis=None, dtype=None, ddof=0, keepdims=False)
            Compute the variance along the specified axis.

    Array Manipulation:
        reshape(*shape, order='C')
            Return an array with a new shape.
        transpose(*axes)
            Return a view of the array with axes transposed.
        flatten()
            Return a copy of the array collapsed into one dimension.
        squeeze(axis=None)
            Remove axes of length one.
        expand_dims(axis)
            Expand the shape of an array.

    Indexing and Selection:
        take(indices, axis=None, mode=None)
            Return an array formed from elements at given indices.
        compress(condition, axis=None)
            Return selected slices along given axis.
        nonzero()
            Return indices of elements that are non-zero.

    Sorting and Searching:
        sort(axis=-1, stable=True, order=None)
            Sort an array in-place.
        argsort(axis=-1, kind=None, order=None)
            Return indices that would sort an array.
        argmax(axis=None), argmin(axis=None)
            Return indices of maximum/minimum datas.

    Numerical Operations:
        round(decimals=0)
            Round array elements to given number of decimals.
        clip(min=None, max=None)
            Clip datas to specified range.
        cumsum(axis=None, dtype=None), cumprod(axis=None, dtype=None)
            Return cumulative sum/product along axis.

    Type Conversion:
        astype(dtype)
            Copy array cast to specified type.
        to_numpy(dtype=None)
            Convert to numpy.ndarray.
        to_jax(dtype=None)
            Convert to jax.numpy.ndarray.

    PyTorch Compatibility:
        unsqueeze(dim), clamp(min_data=None, max_data=None), clone()
            PyTorch-style operations.

    Examples
    --------
    Basic usage with NumPy arrays:

    >>> import numpy as np
    >>> from saiunit import CustomArray
    >>> import brainstate
    >>>
    >>> class Array(brainstate.State, u.CustomArray):
    >>>>   pass
    >>> arr = Array()
    >>> arr.data = np.array([1, 2, 3, 4, 5])
    >>> print(arr.shape)
    (5,)
    >>> print(arr.mean())
    3.0

    Arithmetic operations:

    >>> result = arr * 2 + 10
    >>> print(result)
    [12 14 16 18 20]

    JAX compatibility:

    >>> import jax.numpy as jnp
    >>> jax_arr = Array()
    >>> jax_arr.data = jnp.array([1.0, 2.0, 3.0])
    >>> squared = jax_arr ** 2
    >>> print(squared)
    [1. 4. 9.]

    Array manipulation:

    >>> matrix = Array()
    >>> matrix.data = np.array([[1, 2], [3, 4]])
    >>> transposed = matrix.T
    >>> reshaped = matrix.reshape(4)
    >>> print(reshaped)
    [1 2 3 4]

    Statistical operations:

    >>> data = Array()
    >>> data.data = np.array([1, 2, 3, 4, 5])
    >>> print(f"Mean: {data.mean()}, Std: {data.std()}")
    Mean: 3.0, Std: 1.4142135623730951

    Notes
    -----
    - This class uses duck typing and delegates operations to the underlying array
    - In-place operations modify the internal `data` attribute directly
    - Some methods return the underlying array type rather than CustomArray instances
    - The `imag` property currently has a typo and accesses `data.image`
    - Thread safety depends on the underlying array implementation
    - JAX transformations (jit, grad, vmap) work seamlessly with CustomArray instances

    See Also
    --------
    numpy.ndarray : NumPy's N-dimensional array
    jax.numpy.ndarray : JAX's array implementation
    torch.Tensor : PyTorch's tensor class

    References
    ----------
    .. [1] NumPy documentation: https://numpy.org/doc/
    .. [2] JAX documentation: https://jax.readthedocs.io/
    .. [3] PyTorch documentation: https://pytorch.org/docs/
    """
    data: Any

    def __hash__(self):
        return hash(self.data)

    @property
    def dtype(self):
        """Variable dtype."""
        return math.get_dtype(self.data)

    @property
    def shape(self):
        """Variable shape."""
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def imag(self):
        return self.data.image

    @property
    def real(self):
        return self.data.real

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        return self.data.T

    def __format__(self, format_spec: str) -> str:
        return format(self.data)

    def __iter__(self):
        """Solve the issue of DeviceArray.__iter__.

        Details please see JAX issues:

        - https://github.com/google/jax/issues/7713
        - https://github.com/google/jax/pull/3821
        """
        for i in range(self.data.shape[0]):
            yield self.data[i]

    def __getitem__(self, index):
        if isinstance(index, slice) and (index == slice(None)):
            return self.data
        return self.data[index]

    def __setitem__(self, index, data: ArrayLike):
        if isinstance(data, np.ndarray):
            data = math.asarray(data)

        # update
        self_data = math.asarray(self.data)
        self.data = self_data.at[index].set(data)

    # ---------- #
    # operations #
    # ---------- #

    def __len__(self) -> int:
        return len(self.data)

    def __neg__(self):
        return self.data.__neg__()

    def __pos__(self):
        return self.data.__pos__()

    def __abs__(self):
        return self.data.__abs__()

    def __invert__(self):
        return self.data.__invert__()

    def __eq__(self, oc):
        return self.data == oc

    def __ne__(self, oc):
        return self.data != oc

    def __lt__(self, oc):
        return self.data < oc

    def __le__(self, oc):
        return self.data <= oc

    def __gt__(self, oc):
        return self.data > oc

    def __ge__(self, oc):
        return self.data >= oc

    def __add__(self, oc):
        return self.data + oc

    def __radd__(self, oc):
        return self.data + oc

    def __iadd__(self, oc):
        # a += b
        self.data = self.data + oc
        return self

    def __sub__(self, oc):
        return self.data - oc

    def __rsub__(self, oc):
        return oc - self.data

    def __isub__(self, oc):
        # a -= b
        self.data = self.data - oc
        return self

    def __mul__(self, oc):
        return self.data * oc

    def __rmul__(self, oc):
        return oc * self.data

    def __imul__(self, oc):
        # a *= b
        self.data = self.data * oc
        return self

    def __rdiv__(self, oc):
        return oc / self.data

    def __truediv__(self, oc):
        return self.data / oc

    def __rtruediv__(self, oc):
        return oc / self.data

    def __itruediv__(self, oc):
        # a /= b
        self.data = self.data / oc
        return self

    def __floordiv__(self, oc):
        return self.data // oc

    def __rfloordiv__(self, oc):
        return oc // self.data

    def __ifloordiv__(self, oc):
        # a //= b
        self.data = self.data // oc
        return self

    def __divmod__(self, oc):
        return self.data.__divmod__(oc)

    def __rdivmod__(self, oc):
        return self.data.__rdivmod__(oc)

    def __mod__(self, oc):
        return self.data % oc

    def __rmod__(self, oc):
        return oc % self.data

    def __imod__(self, oc):
        # a %= b
        self.data = self.data % oc
        return self

    def __pow__(self, oc):
        return self.data ** oc

    def __rpow__(self, oc):
        return oc ** self.data

    def __ipow__(self, oc):
        # a **= b
        self.data = self.data ** oc
        return self

    def __matmul__(self, oc):
        return self.data @ oc

    def __rmatmul__(self, oc):
        return oc @ self.data

    def __imatmul__(self, oc):
        # a @= b
        self.data = self.data @ oc
        return self

    def __and__(self, oc):
        return self.data & oc

    def __rand__(self, oc):
        return oc & self.data

    def __iand__(self, oc):
        # a &= b
        self.data = self.data & oc
        return self

    def __or__(self, oc):
        return self.data | oc

    def __ror__(self, oc):
        return oc | self.data

    def __ior__(self, oc):
        # a |= b
        self.data = self.data | oc
        return self

    def __xor__(self, oc):
        return self.data ^ oc

    def __rxor__(self, oc):
        return oc ^ self.data

    def __ixor__(self, oc):
        # a ^= b
        self.data = self.data ^ oc
        return self

    def __lshift__(self, oc):
        return self.data << oc

    def __rlshift__(self, oc):
        return oc << self.data

    def __ilshift__(self, oc):
        # a <<= b
        self.data = self.data << oc
        return self

    def __rshift__(self, oc):
        return self.data >> oc

    def __rrshift__(self, oc):
        return oc >> self.data

    def __irshift__(self, oc):
        # a >>= b
        self.data = self.data >> oc
        return self

    def __round__(self, ndigits=None):
        return self.data.__round__(ndigits)

    # ----------------------- #
    #      NumPy methods      #
    # ----------------------- #

    def all(self, axis=None, keepdims=False):
        """Returns True if all elements evaluate to True."""
        r = self.data.all(axis=axis, keepdims=keepdims)
        return r

    def any(self, axis=None, keepdims=False):
        """Returns True if any of the elements of a evaluate to True."""
        r = self.data.any(axis=axis, keepdims=keepdims)
        return r

    def argmax(self, axis=None):
        """Return indices of the maximum datas along the given axis."""
        return self.data.argmax(axis=axis)

    def argmin(self, axis=None):
        """Return indices of the minimum datas along the given axis."""
        return self.data.argmin(axis=axis)

    def argpartition(self, kth, axis: int = -1, kind: str = 'introselect', order=None):
        """Returns the indices that would partition this array."""
        return self.data.argpartition(kth=kth, axis=axis, kind=kind, order=order)

    def argsort(self, axis=-1, kind=None, order=None):
        """Returns the indices that would sort this array."""
        return self.data.argsort(axis=axis, kind=kind, order=order)

    def astype(self, dtype):
        """Copy of the array, cast to a specified type.

        Parameters::

        dtype: str, dtype
          Typecode or data-type to which the array is cast.
        """
        if dtype is None:
            return self.data
        else:
            return self.data.astype(dtype)

    def byteswap(self, inplace=False):
        """Swap the bytes of the array elements

        Toggle between low-endian and big-endian data representation by
        returning a byteswapped array, optionally swapped in-place.
        Arrays of byte-strings are not swapped. The real and imaginary
        parts of a complex number are swapped individually."""
        return self.data.byteswap(inplace=inplace)

    def choose(self, choices, mode='raise'):
        """Use an index array to construct a new array from a set of choices."""
        return self.data.choose(choices=choices, mode=mode)

    def clip(self, min=None, max=None):
        """Return an array whose datas are limited to [min, max]. One of max or min must be given."""
        r = self.data.clip(min=min, max=max)
        return r

    def compress(self, condition, axis=None):
        """Return selected slices of this array along given axis."""
        return self.data.compress(condition=condition, axis=axis)

    def conj(self):
        """Complex-conjugate all elements."""
        return self.data.conj()

    def conjugate(self):
        """Return the complex conjugate, element-wise."""
        return self.data.conjugate()

    def copy(self):
        """Return a copy of the array."""
        return self.data.copy()

    def cumprod(self, axis=None, dtype=None):
        """Return the cumulative product of the elements along the given axis."""
        return self.data.cumprod(axis=axis, dtype=dtype)

    def cumsum(self, axis=None, dtype=None):
        """Return the cumulative sum of the elements along the given axis."""
        return self.data.cumsum(axis=axis, dtype=dtype)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """Return specified diagonals."""
        return self.data.diagonal(offset=offset, axis1=axis1, axis2=axis2)

    def dot(self, b):
        """Dot product of two arrays."""
        return self.data.dot(b)

    def fill(self, data: ArrayLike):
        """Fill the array with a scalar data."""
        self.data = math.ones_like(self.data) * data

    def flatten(self):
        return self.data.flatten()

    def item(self, *args):
        """Copy an element of an array to a standard Python scalar and return it."""
        return self.data.item(*args)

    def max(self, axis=None, keepdims=False, *args, **kwargs):
        """Return the maximum along a given axis."""
        res = self.data.max(axis=axis, keepdims=keepdims, *args, **kwargs)
        return res

    def mean(self, axis=None, dtype=None, keepdims=False, *args, **kwargs):
        """Returns the average of the array elements along given axis."""
        res = self.data.mean(axis=axis, dtype=dtype, keepdims=keepdims, *args, **kwargs)
        return res

    def min(self, axis=None, keepdims=False, *args, **kwargs):
        """Return the minimum along a given axis."""
        res = self.data.min(axis=axis, keepdims=keepdims, *args, **kwargs)
        return res

    def nonzero(self):
        """Return the indices of the elements that are non-zero."""
        return tuple(a for a in self.data.nonzero())

    def prod(self, axis=None, dtype=None, keepdims=False, initial=1, where=True):
        """Return the product of the array elements over the given axis."""
        res = self.data.prod(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
        return res

    def ptp(self, axis=None, keepdims=False):
        """Peak to peak (maximum - minimum) data along a given axis."""
        r = self.data.ptp(axis=axis, keepdims=keepdims)
        return r

    def put(self, indices, datas):
        """Replaces specified elements of an array with given datas.

        Parameters::

        indices: array_like
          Target indices, interpreted as integers.
        datas: array_like
          Values to place in the array at target indices.
        """
        self.__setitem__(indices, datas)

    def ravel(self, order=None):
        """Return a flattened array."""
        return self.data.ravel(order=order)

    def repeat(self, repeats, axis=None):
        """Repeat elements of an array."""
        return self.data.repeat(repeats=repeats, axis=axis)

    def reshape(self, *shape, order='C'):
        """Returns an array containing the same data with a new shape."""
        return self.data.reshape(*shape, order=order)

    def resize(self, new_shape):
        """Change shape and size of array in-place."""
        self.data = self.data.reshape(new_shape)

    def round(self, decimals=0):
        """Return ``a`` with each element rounded to the given number of decimals."""
        return self.data.round(decimals=decimals)

    def searchsorted(self, v, side='left', sorter=None):
        return self.data.searchsorted(v=v, side=side, sorter=sorter)

    def sort(self, axis=-1, stable=True, order=None):
        """Sort an array in-place.

        Parameters::

        axis : int, optional
            Axis along which to sort. Default is -1, which means sort along the
            last axis.
        stable : bool, optional
            Whether to use a stable sorting algorithm. The default is True.
        order : str or list of str, optional
            When `a` is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.  A single field can
            be specified as a string, and not all fields need be specified,
            but unspecified fields will still be used, in the order in which
            they come up in the dtype, to break ties.
        """
        self.data = self.data.sort(axis=axis, stable=stable, order=order)

    def squeeze(self, axis=None):
        """Remove axes of length one from ``a``."""
        return self.data.squeeze(axis=axis)

    def std(self, axis=None, dtype=None, ddof=0, keepdims=False):
        r = self.data.std(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        return r

    def sum(self, axis=None, dtype=None, keepdims=False, initial=0, where=True):
        """Return the sum of the array elements over the given axis."""
        res = self.data.sum(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
        return res

    def swapaxes(self, axis1, axis2):
        """Return a view of the array with `axis1` and `axis2` interchanged."""
        return self.data.swapaxes(axis1, axis2)

    def split(self, indices_or_sections, axis=0):
        return [a for a in math.split(self.data, indices_or_sections, axis=axis)]

    def take(self, indices, axis=None, mode=None):
        """Return an array formed from the elements of a at the given indices."""
        return self.data.take(indices=indices, axis=axis, mode=mode)

    def tolist(self):
        return self.data.tolist()

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
        """Return the sum along diagonals of the array."""
        return self.data.trace(offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

    def transpose(self, *axes):
        return self.data.transpose(*axes)

    def tile(self, reps):
        return self.data.tile(reps)

    def var(self, axis=None, dtype=None, ddof=0, keepdims=False):
        """Returns the variance of the array elements, along given axis."""
        r = self.data.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
        return r

    def view(self, *args, dtype=None):
        if len(args) == 0:
            if dtype is None:
                raise ValueError('Provide dtype or shape.')
            else:
                return self.data.view(dtype)
        else:
            if isinstance(args[0], int):  # shape
                if dtype is not None:
                    raise ValueError('Provide one of dtype or shape. Not both.')
                return self.data.reshape(*args)
            else:  # dtype
                assert not isinstance(args[0], int)
                assert dtype is None
                return self.data.view(args[0])

    # ------------------
    # NumPy support
    # ------------------

    def numpy(self, dtype=None):
        """Convert to numpy.ndarray."""
        # warnings.warn('Deprecated since 2.1.12. Please use ".to_numpy()" instead.', DeprecationWarning)
        return np.asarray(self.data, dtype=dtype)

    def to_numpy(self, dtype=None):
        """Convert to numpy.ndarray."""
        return np.asarray(self.data, dtype=dtype)

    def to_jax(self, dtype=None):
        """Convert to jax.numpy.ndarray."""
        if dtype is None:
            return self.data
        else:
            return math.asarray(self.data, dtype=dtype)

    def __array__(self, dtype=None):
        """Support ``numpy.array()`` and ``numpy.asarray()`` functions."""
        return np.asarray(self.data, dtype=dtype)

    def __jax_array__(self):
        return self.data

    def __bool__(self) -> bool:
        return bool(self.data)

    def __float__(self):
        return self.data.__float__()

    def __int__(self):
        return self.data.__int__()

    def __complex__(self):
        return self.data.__complex__()

    def __hex__(self):
        assert self.ndim == 0, 'hex only works on scalar datas'
        return hex(self.data)  # type: ignore

    def __oct__(self):
        assert self.ndim == 0, 'oct only works on scalar datas'
        return oct(self.data)  # type: ignore

    def __index__(self):
        return operator.index(self.data)

    # ----------------------
    # PyTorch compatibility
    # ----------------------

    def unsqueeze(self, dim: int) -> ArrayLike:
        """
        Array.unsqueeze(dim) -> Array, or so called Tensor
        equals
        Array.expand_dims(dim)

        See :func:`brainpy.math.unsqueeze`
        """
        return math.expand_dims(self.data, dim)

    def expand_dims(self, axis: Union[int, Sequence[int]]) -> ArrayLike:
        return math.expand_dims(self.data, axis)

    def expand_as(self, array: ArrayLike) -> ArrayLike:
        return math.broadcast_to(self.data, array)

    def pow(self, index: int):
        return self.data ** index

    def addr(
        self,
        vec1: ArrayLike,
        vec2: ArrayLike,
        *,
        beta: float = 1.0,
        alpha: float = 1.0,
    ) -> Optional[ArrayLike]:
        r = alpha * math.outer(vec1, vec2) + beta * self.data
        return r

    def outer(self, other: ArrayLike) -> ArrayLike:
        return math.outer(self.data, other.data)

    def abs(self) -> Optional[ArrayLike]:
        r = math.abs(self.data)
        return r

    def absolute(self) -> Optional[ArrayLike]:
        """
        alias of Array.abs
        """
        return self.abs()

    def mul(self, data: ArrayLike):
        return self.data * data

    def multiply(self, data: ArrayLike):  # real signature unknown; restored from __doc__
        """
        multiply(data) -> Tensor

        See :func:`torch.multiply`.
        """
        return self.data * data

    def sin(self) -> Optional[ArrayLike]:
        r = math.sin(self.data)
        return r

    def sin_(self):
        self.data = math.sin(self.data)
        return self

    def cos_(self):
        self.data = math.cos(self.data)
        return self

    def cos(self) -> Optional[ArrayLike]:
        r = math.cos(self.data)
        return r

    def tan_(self):
        self.data = math.tan(self.data)
        return self

    def tan(self) -> Optional[ArrayLike]:
        r = math.tan(self.data)
        return r

    def sinh_(self):
        self.data = math.sinh(self.data)
        return self

    def sinh(self) -> Optional[ArrayLike]:
        r = math.sinh(self.data)
        return r

    def cosh(self) -> Optional[ArrayLike]:
        r = math.cosh(self.data)
        return r

    def tanh_(self):
        self.data = math.tanh(self.data)
        return self

    def tanh(self) -> Optional[ArrayLike]:
        r = math.tanh(self.data)
        return r

    def arcsin_(self):
        self.data = math.arcsin(self.data)
        return self

    def arcsin(self) -> Optional[ArrayLike]:
        r = math.arcsin(self.data)
        return r

    def arccos_(self):
        self.data = math.arccos(self.data)
        return self

    def arccos(self) -> Optional[ArrayLike]:
        r = math.arccos(self.data)
        return r

    def arctan_(self):
        self.data = math.arctan(self.data)
        return self

    def arctan(self) -> Optional[ArrayLike]:
        r = math.arctan(self.data)
        return r

    def clamp(
        self,
        min_data: Optional[ArrayLike] = None,
        max_data: Optional[ArrayLike] = None,
    ) -> Optional[ArrayLike]:
        """
        return the data between min_data and max_data,
        if min_data is None, then no lower bound,
        if max_data is None, then no upper bound.
        """
        r = math.clip(self.data, min_data, max_data)
        return r

    def clamp_(
        self,
        min_data: Optional[ArrayLike] = None,
        max_data: Optional[ArrayLike] = None
    ):
        """
        return the data between min_data and max_data,
        if min_data is None, then no lower bound,
        if max_data is None, then no upper bound.
        """
        self.clamp(min_data, max_data)
        return self

    def clone(self) -> ArrayLike:
        return self.data.copy()

    def expand(self, *sizes) -> ArrayLike:
        """
        Expand an array to a new shape.

        Parameters::

        sizes : tuple or int
            The shape of the desired array. A single integer ``i`` is interpreted
            as ``(i,)``.

        Returns::

        expanded : Array
            A readonly view on the original array with the given shape. It is
            typically not contiguous. Furthermore, more than one element of a
            expanded array may refer to a single memory location.
        """
        l_ori = len(self.shape)
        l_tar = len(sizes)
        base = l_tar - l_ori
        sizes_list = list(sizes)
        if base < 0:
            raise ValueError(f'the number of sizes provided ({len(sizes)}) must be greater or equal to the number of '
                             f'dimensions in the tensor ({len(self.shape)})')
        for i, v in enumerate(sizes[:base]):
            if v < 0:
                raise ValueError(
                    f'The expanded size of the tensor ({v}) isn\'t allowed in a leading, non-existing dimension {i + 1}')
        for i, v in enumerate(self.shape):
            sizes_list[base + i] = v if sizes_list[base + i] == -1 else sizes_list[base + i]
            if v != 1 and sizes_list[base + i] != v:
                raise ValueError(
                    f'The expanded size of the tensor ({sizes_list[base + i]}) must match the existing size ({v}) at non-singleton '
                    f'dimension {i}.  Target sizes: {sizes}.  Tensor sizes: {self.shape}')
        return math.broadcast_to(self.data, tuple(sizes_list))

    def zero_(self):
        self.data = math.zeros_like(self.data)
        return self

    def bool(self):
        return math.asarray(self.data, dtype=jnp.bool_)

    def int(self):
        return math.asarray(self.data, dtype=jnp.int32)

    def long(self):
        return math.asarray(self.data, dtype=jnp.int64)

    def half(self):
        return math.asarray(self.data, dtype=jnp.float16)

    def float(self):
        return math.asarray(self.data, dtype=jnp.float32)

    def double(self):
        return math.asarray(self.data, dtype=jnp.float64)

    def tree_flatten(self):
        return (self.data,), None

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        return cls(*flat_contents)

