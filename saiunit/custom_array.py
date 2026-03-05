# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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


import operator
from typing import Any, Optional, Union, Sequence

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
    A custom array wrapper providing comprehensive array operations and
    cross-framework compatibility.

    ``CustomArray`` is a mix-in class that delegates every operation to its
    ``data`` attribute.  Subclasses must provide a ``data`` property (or
    attribute) that returns the underlying array.  The class exposes
    NumPy-style methods, PyTorch-style convenience methods, and full
    interoperability with JAX transformations (``jit``, ``grad``, ``vmap``).

    Attributes
    ----------
    data : array-like
        The underlying array storage.  Subclasses are responsible for
        providing this attribute (e.g. via a property backed by some
        internal state).

    Methods
    -------
    NumPy-style methods
        ``all``, ``any``, ``argmax``, ``argmin``, ``argsort``,
        ``astype``, ``clip``, ``copy``, ``cumsum``, ``cumprod``,
        ``diagonal``, ``dot``, ``flatten``, ``max``, ``mean``, ``min``,
        ``nonzero``, ``prod``, ``ravel``, ``repeat``, ``reshape``,
        ``round``, ``squeeze``, ``std``, ``sum``, ``swapaxes``,
        ``take``, ``tolist``, ``trace``, ``transpose``, ``var``
    PyTorch-style methods
        ``unsqueeze``, ``expand``, ``expand_as``, ``clamp``, ``clone``,
        ``zero_``, ``bool``, ``int``, ``long``, ``half``, ``float``,
        ``double``
    Conversion methods
        ``to_numpy``, ``to_jax``, ``numpy``
    Trigonometric methods
        ``sin``, ``cos``, ``tan``, ``sinh``, ``cosh``, ``tanh``,
        ``arcsin``, ``arccos``, ``arctan`` (and in-place ``_`` variants)

    Examples
    --------
    ``CustomArray`` is designed to be used as a mix-in.  A minimal
    standalone subclass needs only a ``data`` attribute and JAX pytree
    registration:

    .. code-block:: python

        import jax
        import numpy as np
        from saiunit import CustomArray

        @jax.tree_util.register_pytree_node_class
        class SimpleArray(CustomArray):
            def __init__(self, value):
                self.data = value

            def tree_flatten(self):
                return (self.data,), None

            @classmethod
            def tree_unflatten(cls, aux_data, flat_contents):
                return cls(*flat_contents)

    Basic properties and arithmetic:

    .. code-block:: python

        arr = SimpleArray(np.array([1.0, 2.0, 3.0]))
        arr.shape   # (3,)
        arr.ndim    # 1
        arr + 10    # array([11., 12., 13.])
        arr ** 2    # array([1., 4., 9.])

    Statistical operations:

    .. code-block:: python

        arr = SimpleArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        arr.mean()  # 3.0
        arr.std()   # ~1.414
        arr.sum()   # 15.0

    Array manipulation:

    .. code-block:: python

        matrix = SimpleArray(np.array([[1, 2, 3], [4, 5, 6]]))
        matrix.T           # transposed (3, 2) array
        matrix.reshape(6)  # array([1, 2, 3, 4, 5, 6])
        matrix.flatten()   # array([1, 2, 3, 4, 5, 6])

    JAX compatibility (``jit``, ``grad``):

    .. code-block:: python

        import jax
        import jax.numpy as jnp

        arr = SimpleArray(jnp.array([1.0, 2.0, 3.0]))

        @jax.jit
        def square(x):
            return x * x

        square(arr)  # Array([1., 4., 9.], ...)

    Notes
    -----
    - This class uses duck typing and delegates operations to the
      underlying ``data`` attribute.
    - In-place operations (``+=``, ``-=``, etc.) modify ``data`` directly
      and return ``self``.
    - Most methods return the raw underlying array type, **not** a new
      ``CustomArray`` instance.
    - Thread safety depends on the underlying array implementation.
    - JAX transformations (``jit``, ``grad``, ``vmap``) work seamlessly
      when the subclass is registered as a JAX pytree.

    See Also
    --------
    numpy.ndarray : NumPy's N-dimensional array.
    jax.Array : JAX's array implementation.

    References
    ----------
    - NumPy documentation: https://numpy.org/doc/
    - JAX documentation: https://jax.readthedocs.io/
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
        return self.data.imag

    @property
    def real(self):
        return self.data.real

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        return self.data.T

    @property
    def mT(self):
        """Transpose the last two dimensions (for batched matrix operations)."""
        return jnp.swapaxes(self.data, -1, -2)

    @property
    def nbytes(self):
        """Total bytes consumed by the array elements."""
        return self.data.nbytes

    @property
    def itemsize(self):
        """Length of one array element in bytes."""
        return self.data.itemsize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.data})"

    def __str__(self) -> str:
        return str(self.data)

    def __format__(self, format_spec: str) -> str:
        return format(self.data, format_spec)

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

        Parameters
        ----------
        dtype : str or dtype
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

        Parameters
        ----------
        indices : array_like
            Target indices, interpreted as integers.
        datas : array_like
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

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. Default is -1, which means sort along the
            last axis.
        stable : bool, optional
            Whether to use a stable sorting algorithm. The default is True.
        order : str or list of str, optional
            When ``a`` is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.
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
        return math.tile(self.data, reps)

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
        return self.to_numpy(dtype=dtype)

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
        return math.broadcast_to(self.data, jnp.asarray(array).shape)

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
        self.data = math.clip(self.data, min_data, max_data)
        return self

    def clone(self) -> ArrayLike:
        return self.data.copy()

    def expand(self, *sizes) -> ArrayLike:
        """
        Expand an array to a new shape.

        Parameters
        ----------
        sizes : tuple or int
            The shape of the desired array. A single integer ``i`` is interpreted
            as ``(i,)``.

        Returns
        -------
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
            raise ValueError(
                f'the number of sizes provided ({len(sizes)}) '
                f'must be greater or equal to the number of '
                f'dimensions in the tensor ({len(self.shape)})'
            )
        for i, v in enumerate(sizes[:base]):
            if v < 0:
                raise ValueError(
                    f'The expanded size of the tensor ({v}) '
                    f'isn\'t allowed in a leading, non-existing dimension {i + 1}'
                )
        for i, v in enumerate(self.shape):
            sizes_list[base + i] = v if sizes_list[base + i] == -1 else sizes_list[base + i]
            if v != 1 and sizes_list[base + i] != v:
                raise ValueError(
                    f'The expanded size of the tensor ({sizes_list[base + i]}) '
                    f'must match the existing size ({v}) at non-singleton '
                    f'dimension {i}.  Target sizes: {sizes}.  Tensor sizes: {self.shape}'
                )
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
