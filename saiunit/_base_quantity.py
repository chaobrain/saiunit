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

import numbers
import operator
import re
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from ._base_dimension import Dimension, UnitMismatchError, _is_tracer
from ._base_getters import (
    get_dim,
    fail_for_dimension_mismatch,
    maybe_decimal,
    _to_quantity,
    unit_scale_align_to_first,
)
from ._base_unit import Unit, UNITLESS
from ._misc import maybe_custom_array_tree
from ._sparse_base import SparseMatrix

__all__ = [
    'Quantity',
    'compatible_with_equinox',
]

# ---------------------------------------------------------------------------
# Module-level type aliases and globals
# ---------------------------------------------------------------------------

StaticScalar = (
    np.bool_ | np.number |  # NumPy scalar types
    bool | int | float | complex  # Python scalar types
)
PyTree = Any
_all_slice = slice(None, None, None)
compat_with_equinox = False


def compatible_with_equinox(mode: bool = True):
    """
    This function is developed to set the compatibility with equinox.
    See `unit-aware diffrax <https://github.com/chaoming0625/diffrax>`_.

    Args:
        mode: bool, optional. The mode to set the compatibility with equinox.
    """
    global compat_with_equinox
    compat_with_equinox = mode


# ---------------------------------------------------------------------------
# Wrapping functions
# ---------------------------------------------------------------------------

def _wrap_function_keep_unit(func):
    """
    Returns a new function that wraps the given function `func` so that it
    keeps the dimensions of its input. Arrays are transformed to
    unitless jax numpy arrays before calling `func`, the output is a array
    with the original dimensions re-attached.

    These transformations apply only to the very first argument, all
    other arguments are ignored/untouched, allowing to work functions like
    ``sum`` to work as expected with additional ``axis`` etc. arguments.
    """

    def f(x: 'Quantity', *args, **kwds):  # pylint: disable=C0111
        return Quantity(func(x.mantissa, *args, **kwds), unit=x.unit)

    f._arg_units = [None]
    f._return_unit = lambda u: u
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    f._do_not_run_doctests = True
    return f


def _wrap_function_change_unit(func, unit_fun):
    """
    Returns a new function that wraps the given function `func` so that it
    changes the dimensions of its input. Arrays are transformed to
    unitless jax numpy arrays before calling `func`, the output is a array
    with the original dimensions passed through the function
    `unit_fun`. A typical use would be a ``sqrt`` function that uses
    ``lambda d: d ** 0.5`` as ``unit_fun``.

    These transformations apply only to the very first argument, all
    other arguments are ignored/untouched.
    """

    def f(x, *args, **kwds):  # pylint: disable=C0111
        assert isinstance(x, Quantity), "Only Quantity objects can be passed to this function"
        x = x.factorless()
        return maybe_decimal(Quantity(func(x.mantissa, *args, **kwds), unit=unit_fun(x.unit, x.unit)))

    f._arg_units = [None]
    f._return_unit = unit_fun
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    f._do_not_run_doctests = True
    return f


def _wrap_function_remove_unit(func):
    """
    Returns a new function that wraps the given function `func` so that it
    removes any dimensions from its input. Useful for functions that are
    returning integers (indices) or booleans, irrespective of the datatype
    contained in the array.

    These transformations apply only to the very first argument, all
    other arguments are ignored/untouched.
    """

    def f(x, *args, **kwds):  # pylint: disable=C0111
        assert isinstance(x, Quantity), "Only Quantity objects can be passed to this function"
        return func(x.mantissa, *args, **kwds)

    f._arg_units = [None]
    f._return_unit = 1
    f.__name__ = func.__name__
    f.__doc__ = func.__doc__
    f._do_not_run_doctests = True
    return f


# ---------------------------------------------------------------------------
# List processing helpers
# ---------------------------------------------------------------------------

def _zoom_values_with_units(
    values: Sequence[jax.typing.ArrayLike],
    units: Sequence[Unit]
):
    """
    Zoom values with units.

    Parameters
    ----------
    values : `Array`
        The values to zoom.
    units : `Array`
        The units to use for zooming.

    Returns
    -------
    zoomed_values : `Array`
        The zoomed values.
    """
    assert len(values) == len(units), "The number of values and units must be the same"
    values = list(values)
    first_unit = units[0]
    for i in range(1, len(values)):
        if not units[i].has_same_magnitude(first_unit):
            values[i] = values[i] * (units[i].magnitude / first_unit.magnitude)
    return values


def _check_units_and_collect_values(lst) -> tuple[jax.typing.ArrayLike, Unit]:
    units = []
    values = []

    for item in lst:
        if isinstance(item, (list, tuple)):
            val, unit = _check_units_and_collect_values(item)
            values.append(val)
            if unit != UNITLESS:
                units.append(unit)
        elif isinstance(item, Quantity):
            values.append(item.mantissa)
            units.append(item.unit)
        elif isinstance(item, Unit):
            values.append(1)
            units.append(item)
        else:
            values.append(item)
            units.append(None)

    if len(units):
        first_unit = units[0]
        if first_unit is None:
            if not all(unit is None for unit in units):
                raise TypeError(f"All elements must have the same units, but got {units}")
            first_unit = UNITLESS
            units = [UNITLESS] * len(units)
        else:
            if not all(first_unit.has_same_dim(unit) for unit in units):
                raise TypeError(f"All elements must have the same units, but got {units}")
        return jnp.asarray(_zoom_values_with_units(values, units)), first_unit
    else:
        return jnp.asarray(values), UNITLESS


def _process_list_with_units(value: list) -> tuple[jax.typing.ArrayLike, Unit]:
    values, unit = _check_units_and_collect_values(value)
    return values, unit


def _element_not_quantity(x):
    assert not isinstance(x, Quantity), f"Expected not a Quantity object, but got {x}"
    return x


# ---------------------------------------------------------------------------
# Pickle helper
# ---------------------------------------------------------------------------

def _quantity_with_unit(mantissa, unit):
    """Private reconstruction helper for Quantity pickling.

    Must live at module level *without* ``@set_module_as`` so that pickle can
    locate it as ``saiunit._base._quantity_with_unit``.
    """
    return Quantity(mantissa, unit=unit)


_quantity_with_unit.__module__ = 'saiunit._base'


# ---------------------------------------------------------------------------
# Quantity class
# ---------------------------------------------------------------------------

@register_pytree_node_class
class Quantity:
    """
    The `Quantity` class represents a physical quantity with a mantissa and a unit.
    It is used to represent all physical quantities in ``saiunit``.
    """

    __module__ = "saiunit"
    __slots__ = ('_mantissa', '_unit')
    __array_priority__ = 1000
    _mantissa: jax.Array | np.ndarray
    _unit: Unit

    def __init__(
        self,
        mantissa: PyTree | Unit,
        unit: Unit | jax.typing.ArrayLike | None = UNITLESS,
        dtype: jax.typing.DTypeLike | None = None,
    ):

        with jax.ensure_compile_time_eval():  # inside JIT, this can avoid to trace the constant mantissa value

            # Handle custom arrays in the mantissa tree structure
            mantissa = maybe_custom_array_tree(mantissa)

            if isinstance(mantissa, Unit):
                if unit is not UNITLESS:
                    raise ValueError(
                        "Cannot create a Quantity object with a unit and a mantissa that is a Unit object.")
                unit = mantissa
                mantissa = 1.

            if isinstance(mantissa, (list, tuple)):
                mantissa, new_unit = _process_list_with_units(mantissa)
                if unit is UNITLESS:
                    unit = new_unit
                elif new_unit != UNITLESS:
                    if not new_unit.has_same_dim(unit):
                        raise TypeError(f"All elements must have the same unit. But got {unit} != {new_unit}")
                    if not new_unit.has_same_magnitude(unit):
                        mantissa = mantissa * (new_unit.magnitude / unit.magnitude)
                mantissa = jnp.array(mantissa, dtype=dtype)

            # array mantissa
            elif isinstance(mantissa, Quantity):
                if unit is UNITLESS:
                    unit = mantissa.unit
                elif not unit.has_same_dim(mantissa.unit):
                    raise ValueError("Cannot create a Quantity object with a different unit.")
                mantissa = mantissa.in_unit(unit)
                mantissa = mantissa.mantissa

            elif isinstance(mantissa, (np.ndarray, jax.Array)):
                if dtype is not None:
                    mantissa = jnp.array(mantissa, dtype=dtype)
                # skip 'asarray' if dtype is not provided

            elif isinstance(mantissa, (jnp.number, numbers.Number)):
                pass  # keep as-is; jnp.array conversion deferred to use-site

            else:
                pass  # keep as-is for other pytree types

        # mantissa
        self._mantissa = mantissa

        # dimension
        self._unit = unit

    @property
    def at(self):
        """
        Helper property for index update functionality.

        The ``at`` property provides a functionally pure equivalent of in-place
        array modifications.

        In particular:

        ==============================  ================================
        Alternate syntax                Equivalent In-place expression
        ==============================  ================================
        ``x = x.at[idx].set(y)``        ``x[idx] = y``
        ``x = x.at[idx].add(y)``        ``x[idx] += y``
        ``x = x.at[idx].multiply(y)``   ``x[idx] *= y``
        ``x = x.at[idx].divide(y)``     ``x[idx] /= y``
        ``x = x.at[idx].power(y)``      ``x[idx] **= y``
        ``x = x.at[idx].min(y)``        ``x[idx] = minimum(x[idx], y)``
        ``x = x.at[idx].max(y)``        ``x[idx] = maximum(x[idx], y)``
        ``x = x.at[idx].apply(ufunc)``  ``ufunc.at(x, idx)``
        ``x = x.at[idx].get()``         ``x = x[idx]``
        ==============================  ================================

        None of the ``x.at`` expressions modify the original ``x``; instead they return
        a modified copy of ``x``. However, inside a :py:func:`~jax.jit` compiled function,
        expressions like :code:`x = x.at[idx].set(y)` are guaranteed to be applied in-place.

        Unlike NumPy in-place operations such as :code:`x[idx] += y`, if multiple
        indices refer to the same location, all updates will be applied (NumPy would
        only apply the last update, rather than applying all updates.) The order
        in which conflicting updates are applied is implementation-defined and may be
        nondeterministic (e.g., due to concurrency on some hardware platforms).

        By default, JAX assumes that all indices are in-bounds. Alternative out-of-bound
        index semantics can be specified via the ``mode`` parameter (see below).

        Arguments
        ---------
        mode : str
            Specify out-of-bound indexing mode. Options are:

            - ``"promise_in_bounds"``: (default) The user promises that indices are in bounds.
              No additional checking will be performed. In practice, this means that
              out-of-bounds indices in ``get()`` will be clipped, and out-of-bounds indices
              in ``set()``, ``add()``, etc. will be dropped.
            - ``"clip"``: clamp out of bounds indices into valid range.
            - ``"drop"``: ignore out-of-bound indices.
            - ``"fill"``: alias for ``"drop"``.  For `get()`, the optional ``fill_value``
              argument specifies the value that will be returned.
        indices_are_sorted : bool
            If True, the implementation will assume that the indices passed to ``at[]``
            are sorted in ascending order, which can lead to more efficient execution
            on some backends.
        unique_indices : bool
            If True, the implementation will assume that the indices passed to ``at[]``
            are unique, which can result in more efficient execution on some backends.
        fill_value : Any
            Only applies to the ``get()`` method: the fill value to return for out-of-bounds
            slices when `mode` is ``'fill'``. Ignored otherwise. Defaults to ``NaN`` for
            inexact types, the largest negative value for signed types, the largest positive
            value for unsigned types, and ``True`` for booleans.

        Examples
        --------
        >>> import saiunit as bu
        >>> x = jnp.arange(5.0) * bu.mV
        >>> x
        Array([0., 1., 2., 3., 4.], dtype=float32) * mvolt
        >>> x.at[2].add(10)
        saiunit.UnitMismatchError: Cannot convert to a unit with different dimensions. (units are Unit(1.0) and mV).
        >>> x.at[2].add(10 * bu.mV)
        ArrayImpl([ 0.,  1., 12.,  3.,  4.], dtype=float32) * mvolt
        >>> x.at[10].add(10 * bu.mV)  # out-of-bounds indices are ignored
        ArrayImpl([0., 1., 2., 3., 4.], dtype=float32) * mvolt
        >>> x.at[20].add(10 * bu.mV, mode='clip')
        ArrayImpl([ 0.,  1.,  2.,  3., 14.], dtype=float32) * mvolt
        >>> x.at[2].get()
        2. * mvolt
        >>> x.at[20].get()  # out-of-bounds indices clipped
        4. * mvolt
        >>> x.at[20].get(mode='fill')  # out-of-bounds indices filled with NaN
        nan * mvolt
        >>> x.at[20].get(mode='fill', fill_value=-1)  # custom fill value
        saiunit.UnitMismatchError: Cannot convert to a unit with different dimensions. (units are Unit(1.0) and mV).
        >>> x.at[20].get(mode='fill', fill_value=-1 * bu.mV)  # custom fill value
        -1. * mvolt
        """
        return _IndexUpdateHelper(self)

    @property
    def mantissa(self) -> jax.typing.ArrayLike:
        r"""
        The mantissa of the array.

        In the scientific notation, :math:`x = a * 10^b`, the mantissa :math:`a` is the part of
        a floating-point number that contains its significant digits. For example, in the number
        :math:`3.14 * 10^5`, the mantissa is :math:`3.14`.

        Returns:
          The mantissa of the array.
        """
        return self._mantissa

    @property
    def magnitude(self) -> jax.typing.ArrayLike:
        """
        The magnitude of the array.

        Same as :py:meth:`mantissa`.

        In the scientific notation, :math:`x = a * 10^b`, the magnitude :math:`b` is the exponent
        of the power of ten. For example, in the number :math:`3.14 * 10^5`, the magnitude is :math:`5`.

        Returns:
          The magnitude of the array.
        """
        return self.mantissa

    def update_mantissa(self, mantissa: PyTree) -> None:
        """
        Set the mantissa of the array.

        Examples::

        >>> a = jax.numpy.array([1, 2, 3]) * mV
        >>> a[:] = jax.numpy.array([4, 5, 6]) * mV

        Args:
          mantissa: The new mantissa of the array.
        """
        self_value = self.mantissa
        if isinstance(mantissa, Quantity):
            raise ValueError("Cannot set the mantissa of a Quantity to another Quantity.")
        if isinstance(mantissa, np.ndarray):
            mantissa = jnp.asarray(mantissa, dtype=self.dtype)
        elif isinstance(mantissa, jax.Array):
            pass
        else:
            mantissa = jnp.asarray(mantissa, dtype=self.dtype)
        # check
        if mantissa.shape != jnp.shape(self_value):
            raise ValueError(f"The shape of the original data is {jnp.shape(self_value)}, "
                             f"while we got {mantissa.shape}.")
        if mantissa.dtype != jax.dtypes.result_type(self_value):
            raise ValueError(f"The dtype of the original data is {jax.dtypes.result_type(self_value)}, "
                             f"while we got {mantissa.dtype}.")
        self._mantissa = mantissa

    @property
    def dim(self) -> Dimension:
        """
        Returns the physical dimensions of this Quantity object.

        The dimensions represent the physical properties (such as length, mass, time)
        that define the quantity, independent of the specific units used.

        Returns
        -------
        Dimension
            The physical dimensions of this Quantity object, accessed through its unit.

        Examples
        --------
        >>> from saiunit import *
        >>> q = Quantity(5, metre)
        >>> q.dim  # Returns dimensions of length
        metre

        See Also
        --------
        unit : The complete unit information including scale and factor
        """
        return self.unit.dim

    @dim.setter
    def dim(self, value):
        # Do not support setting the unit directly
        raise NotImplementedError(
            "Cannot set the dimension of a Quantity object directly,"
            "Please create a new Quantity object with the dimension you want."
        )

    @property
    def unit(self) -> 'Unit':
        """
        Returns the unit of this Quantity object.

        The unit contains both the dimensions (such as length, mass) and the specific
        scale information (e.g., meters vs kilometers).

        Returns
        -------
        Unit
            The complete unit information of this Quantity object.

        Examples
        --------
        >>> from saiunit import *
        >>> q = Quantity(5, kilometre)
        >>> q.unit  # Returns kilometre unit
        kilometre
        >>> q.unit.magnitude  # Access the magnitude through the unit
        1000.0

        See Also
        --------
        dim : The physical dimensions without scale information
        mantissa : The numerical value of the quantity
        """
        return self._unit

    @unit.setter
    def unit(self, value):
        # Do not support setting the unit directly
        raise NotImplementedError(
            "Cannot set the unit of a Quantity object directly,"
            "Please create a new Quantity object with the unit you want."
        )

    def to(self, new_unit: Unit) -> 'Quantity':
        """
        Convert the given :py:class:`Quantity` into the given unit.

        Examples::

        >>> a = jax.numpy.array([1, 2, 3]) * mV
        >>> a.to(volt)
        array([0.001, 0.002, 0.003]) * volt

        Args:
          new_unit: The new unit to convert the quantity to.

        Returns:
          The new quantity with the given unit.
        """
        return self.in_unit(new_unit)

    def to_decimal(self, unit: Unit = UNITLESS) -> jax.typing.ArrayLike:
        """
        Convert the given :py:class:`Quantity` into the decimal number.

        Examples::

        >>> a = jax.numpy.array([1, 2, 3]) * mV
        >>> a.to_decimal(volt)
        array([0.001, 0.002, 0.003])

        Args:
          unit: The new unit to convert the quantity to.

        Returns:
          The decimal number of the quantity based on the given unit.
        """
        if not isinstance(unit, Unit):
            raise TypeError(f"Expected a Unit, but got {unit}.")
        if not unit.has_same_dim(self.unit):
            raise UnitMismatchError(
                f"Cannot convert to the decimal number using a unit with different dimensions.",
                self.unit,
                unit,
            )
        if not unit.has_same_magnitude(self.unit):
            return self.mantissa * (self.unit.magnitude / unit.magnitude)
        else:
            return self.mantissa

    def in_unit(self, unit: Unit, err_msg: str = None) -> 'Quantity':
        """
        Convert the given :py:class:`Quantity` into the given unit.

        Examples::

        >>> a = jax.numpy.array([1, 2, 3]) * mV
        >>> a.in_unit(volt)
        array([0.001, 0.002, 0.003]) * volt

        Args:
            unit: The new unit to convert the quantity to.
            err_msg: The error message to show when the conversion is not possible.

        Returns:
            The new quantity with the given unit.
        """
        if not isinstance(unit, Unit):
            raise TypeError(f"Expected a Unit, but got {unit}.")
        if not unit.has_same_dim(self.unit):
            if err_msg is None:
                raise UnitMismatchError(f"Cannot convert to a unit with different dimensions.", self.unit, unit)
            else:
                raise UnitMismatchError(err_msg)
        self_mag = self.unit.magnitude
        target_mag = unit.magnitude
        if self_mag == target_mag:
            u = Quantity(self.mantissa, unit=unit)
        else:
            u = Quantity(self.mantissa * (self_mag / target_mag), unit=unit)
        return u

    @staticmethod
    def with_unit(mantissa: PyTree, unit: Unit):
        """
        Create a `Array` object with the given units.

        Parameters
        ----------
        mantissa : {array_like, number}
            The mantissa of the dimension
        unit : Unit
            The unit of the dimension

        Returns
        -------
        q : `Quantity`
            A `Array` object with the given dim

        Examples
        --------
        All of these define an equivalent `Array` object:

        >>> from saiunit import *
        >>> Quantity.with_unit(2, unit=metre)
        2. * metre
        """
        return Quantity(mantissa, unit=unit)

    @property
    def is_unitless(self) -> bool:
        """
        Whether the array does not have unit.

        Returns:
          bool: True if the array does not have unit.
        """
        return self.unit.is_unitless

    def has_same_unit(self, other):
        """
        Whether this Array has the same unit dimensions as another Array

        Parameters
        ----------
        other : Unit
            The other Array to compare with

        Returns
        -------
        bool
            Whether the two Arrays have the same unit dimensions
        """
        self_dim = get_dim(self.dim)
        other_dim = get_dim(other.dim)
        return (self_dim is other_dim) or (self_dim == other_dim)

    def _format_value(self, precision: int | None = None) -> str:
        """Format the mantissa value as a string."""
        m = self.mantissa
        if isinstance(m, jax.Array):
            value = m
        else:
            try:
                value = jnp.asarray(m)
            except TypeError:
                value = m

        if _is_tracer(value):
            return str(value)

        try:
            if value.shape == ():
                s = np.array_str(np.array([value]), precision=precision)
                return s.replace("[", "").replace("]", "").strip()
            # Use numpy's built-in summarization for large arrays
            if value.size > 100:
                kw = {}
                if precision is not None:
                    kw['precision'] = precision
                with np.printoptions(threshold=10, **kw):
                    return np.array_str(value)
            return np.array_str(value, precision=precision)
        except (TypeError, AttributeError):
            return str(value)

    def repr_in_unit(
        self,
        precision: int | None = None,
    ) -> str:
        """
        Represent the Quantity in its current unit.

        Returns the canonical ``"value unit"`` format, e.g.
        ``"3.0 mV"`` or ``"[1. 2. 3.] mV"``.

        Parameters
        ----------
        precision : `int`, optional
            The number of digits of precision (in the given unit).
            If no value is given, numpy's `get_printoptions` is used.

        Returns
        -------
        s : `str`
            The string representation of the Quantity.

        Examples
        --------
        >>> from saiunit import *
        >>> x = 25.123456 * mV
        >>> x.repr_in_unit()
        '25.123456 mV'
        >>> x.in_unit(volt).repr_in_unit(3)
        '0.025 V'
        """
        s = self._format_value(precision=precision)
        if self.unit.should_display_unit:
            s += f" {str(self.unit)}"
        return s.strip()

    def factorless(self) -> 'Quantity':
        """
        Return the Quantity object without the factor.

        Returns
        -------
        out : Quantity
            The Quantity object without the factor.
        """
        if self.unit.factor != 1.0:
            return Quantity(self.mantissa * self.unit.factor, unit=self.unit.factorless())
        else:
            return self

    @property
    def dtype(self):
        """Variable dtype."""
        a = self.mantissa
        if hasattr(a, 'dtype'):
            return a.dtype
        else:
            if isinstance(a, bool):
                return bool
            elif isinstance(a, int):
                return jax.dtypes.canonicalize_dtype(int)
            elif isinstance(a, float):
                return jax.dtypes.canonicalize_dtype(float)
            elif isinstance(a, complex):
                return jax.dtypes.canonicalize_dtype(complex)
            else:
                raise TypeError(f'Can not get dtype of {a}.')

    @property
    def shape(self) -> tuple[int, ...]:
        """Variable shape."""
        return jnp.shape(self.mantissa)

    @property
    def ndim(self) -> int:
        return jnp.ndim(self.mantissa)

    @property
    def imag(self) -> 'Quantity':
        return Quantity(jnp.imag(self.mantissa), unit=self.unit)

    @property
    def real(self) -> 'Quantity':
        return Quantity(jnp.real(self.mantissa), unit=self.unit)

    @property
    def size(self) -> int:
        return jnp.size(self.mantissa)

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by the mantissa array."""
        return jnp.asarray(self.mantissa).nbytes

    @property
    def itemsize(self) -> int:
        """Length (in bytes) of one array element."""
        return jnp.asarray(self.mantissa).itemsize

    @property
    def strides(self):
        """Tuple of byte-steps in each dimension (mirrors numpy.ndarray.strides)."""
        return np.asarray(self.mantissa).strides

    @property
    def flat(self):
        """1-D iterator over the mantissa elements, unit preserved."""
        for v in jnp.asarray(self.mantissa).flat:
            yield Quantity(v, unit=self.unit)

    @property
    def T(self) -> 'Quantity':
        return Quantity(jnp.asarray(self.mantissa).T, unit=self.unit)

    @property
    def mT(self) -> 'Quantity':
        return Quantity(jnp.asarray(self.mantissa).mT, unit=self.unit)

    @property
    def isreal(self) -> jax.Array:
        return jnp.isreal(self.mantissa)

    @property
    def isscalar(self) -> bool:
        return self.ndim == 0

    @property
    def isfinite(self) -> jax.Array:
        return jnp.isfinite(self.mantissa)

    @property
    def isinfinite(self) -> jax.Array:
        return jnp.isinf(self.mantissa)

    @property
    def isinf(self) -> jax.Array:
        return jnp.isinf(self.mantissa)

    @property
    def isnan(self) -> jax.Array:
        return jnp.isnan(self.mantissa)

    # ----------------------- #
    # Python inherent methods #
    # ----------------------- #

    def __hash__(self):
        """
        Hash the Quantity object.

        Returns:
          int: The hash value of the Quantity object.
        """
        try:
            return hash((np.asarray(self.mantissa).tobytes(), self.unit))
        except Exception:
            return hash((id(self.mantissa), self.unit))

    def __repr__(self) -> str:
        value_str = self._format_value()
        unit_str = str(self.unit)
        prefix = "Quantity("
        if self.unit.should_display_unit:
            suffix = f", \"{unit_str}\")"
        else:
            suffix = ")"
        # Indent continuation lines to align with prefix
        if "\n" in value_str:
            indent = " " * len(prefix)
            lines = value_str.split("\n")
            value_str = lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])
        return f"{prefix}{value_str}{suffix}"

    def __str__(self) -> str:
        return self.repr_in_unit()

    def __format__(self, format_spec) -> str:
        if not format_spec:
            return str(self)
        # Block '%' format on quantities with units — "50% mV" is meaningless
        if '%' in format_spec and not self.unit.is_unitless:
            raise ValueError(
                f"'%' format is not supported for Quantity with unit {str(self.unit)!r}. "
                f"Convert to a dimensionless value first."
            )
        unit_str = str(self.unit)
        show_unit = self.unit.should_display_unit
        if self.shape == ():
            formatted_value = format(self.mantissa, format_spec)
            if not show_unit:
                return formatted_value
            return f"{formatted_value} {unit_str}"
        else:
            # Parse precision from standard format specs like .2f, .3e, .4g,
            # 10.2f, +.2f, etc.  Use a regex to extract the precision field.
            m = re.match(r'^[^.]*\.(\d+)[feEgGn%]?$', format_spec)
            if m is not None:
                precision = int(m.group(1))
                value = np.asarray(self.mantissa)
                s = np.array_str(np.round(value, precision), precision=precision)
                if not show_unit:
                    return s
                return f"{s} {unit_str}"
            return str(self)

    def __iter__(self):
        """Solve the issue of DeviceArray.__iter__.

        Details please see JAX issues:

        - https://github.com/google/jax/issues/7713
        - https://github.com/google/jax/pull/3821
        """

        if self.ndim == 0:
            raise TypeError("iteration over a 0-d Quantity is not allowed")
        for i in range(self.shape[0]):
            yield Quantity(self.mantissa[i], unit=self.unit)

    def __getitem__(self, index) -> 'Quantity':

        if isinstance(index, slice) and (index == _all_slice):
            return Quantity(self.mantissa, unit=self.unit)
        elif isinstance(index, tuple):
            for x in index:
                if isinstance(x, Quantity):
                    raise TypeError("Array indices must be integers or slices, not Array")
        elif isinstance(index, Quantity):
            raise TypeError("Array indices must be integers or slices, not Array")
        return Quantity(self.mantissa[index], unit=self.unit)

    def __setitem__(self, index, value: 'Quantity | jax.typing.ArrayLike'):
        # check value
        if not isinstance(value, Quantity):
            if self.is_unitless:
                value = Quantity(value)
            else:
                raise TypeError(f"Only Quantity can be assigned to Quantity. But got {value}")
        value = value.in_unit(self.unit)

        # check index
        index = jax.tree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))

        # update
        self_value = jnp.asarray(self.mantissa).at[index].set(value.mantissa)
        self.update_mantissa(self_value)

    def scatter_add(
        self,
        index: jax.typing.ArrayLike,
        value: 'Quantity | jax.typing.ArrayLike'
    ) -> 'Quantity':
        """
        Scatter-add the given value to the given index.

        Parameters
        ----------
        index : int or array_like
            The index to scatter-add the value to.
        value : Quantity
            The value to scatter-add.

        Returns
        -------
        out : Quantity
            The scatter-added value.
        """

        # check value
        if not isinstance(value, Quantity):
            if self.is_unitless:
                value = Quantity(value)
            else:
                raise TypeError(f"Only Quantity can be assigned to Quantity. But got {value}")
        value = value.in_unit(self.unit)

        # check index
        index = jax.tree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))

        # scatter-add
        self_value = jnp.asarray(self.mantissa)
        self_value = self_value.at[index].add(value.mantissa)
        return Quantity(self_value, unit=self.unit)

    def scatter_sub(
        self,
        index: jax.typing.ArrayLike,
        value: 'Quantity | jax.typing.ArrayLike'
    ) -> 'Quantity':
        """
        Scatter-sub the given value to the given index.

        Parameters
        ----------
        index : int or array_like
            The index to scatter-add the value to.
        value : Quantity
            The value to scatter-add.

        Returns
        -------
        out : Quantity
            The scatter-subbed value.
        """
        return self.scatter_add(index, -value)

    def scatter_mul(
        self,
        index: jax.typing.ArrayLike,
        value: 'Quantity | jax.typing.ArrayLike'
    ) -> 'Quantity':
        """
        Scatter-mul the given value to the given index.

        Parameters
        ----------
        index : int or array_like
            The index to scatter-mul the value to.
        value : Quantity
            The value to scatter-mul.

        Returns
        -------
        out : Quantity
            The scatter-multiplied value.
        """

        # check value: scatter_mul requires a dimensionless scale factor
        if not isinstance(value, Quantity):
            value = Quantity(value)
        if not value.is_unitless:
            raise TypeError(
                f"scatter_mul requires a dimensionless scale factor, "
                f"but got {value}. Use Quantity.__mul__ for unit-changing multiplication."
            )

        # check index
        index = jax.tree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))

        # scatter-mul
        self_value = jnp.asarray(self.mantissa)
        self_value = self_value.at[index].mul(value.mantissa)
        return Quantity(self_value, unit=self.unit)

    def scatter_div(
        self,
        index: jax.typing.ArrayLike,
        value: 'Quantity | jax.typing.ArrayLike'
    ) -> 'Quantity':
        """
        Scatter-div the given value to the given index.

        Parameters
        ----------
        index : int or array_like
            The index to scatter-div the value to.
        value : Quantity
            The value to scatter-div.

        Returns
        -------
        out : Quantity
            The scatter-divided value.
        """

        # check value: scatter_div requires a dimensionless scale factor
        if not isinstance(value, Quantity):
            value = Quantity(value)
        if not value.is_unitless:
            raise TypeError(
                f"scatter_div requires a dimensionless scale factor, "
                f"but got {value}. Use Quantity.__truediv__ for unit-changing division."
            )

        # check index
        index = jax.tree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))

        # scatter-div
        self_value = jnp.asarray(self.mantissa)
        self_value = self_value.at[index].divide(value.mantissa)
        return Quantity(self_value, unit=self.unit)

    def scatter_max(
        self,
        index: jax.typing.ArrayLike,
        value: 'Quantity | jax.typing.ArrayLike'
    ) -> 'Quantity':
        """
        Scatter-max the given value to the given index.

        Parameters
        ----------
        index : int or array_like
            The index to scatter-max the value to.
        value : Quantity
            The value to scatter-max.

        Returns
        -------
        out : Quantity
            The scatter-maximum value.
        """

        # check value
        if not isinstance(value, Quantity):
            if self.is_unitless:
                value = Quantity(value)
            else:
                raise TypeError(f"Only Quantity can be assigned to Quantity. But got {value}")
        value = value.in_unit(self.unit)

        # check index
        index = jax.tree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))

        # scatter-max
        self_value = jnp.asarray(self.mantissa)
        self_value = self_value.at[index].max(value.mantissa)
        return Quantity(self_value, unit=self.unit)

    def scatter_min(
        self,
        index: jax.typing.ArrayLike,
        value: 'Quantity | jax.typing.ArrayLike'
    ) -> 'Quantity':
        """
        Scatter-min the given value to the given index.

        Parameters
        ----------
        index : int or array_like
            The index to scatter-min the value to.
        value : Quantity
            The value to scatter-min.

        Returns
        -------
        out : Quantity
            The scatter-minimum value.
        """

        # check value
        if not isinstance(value, Quantity):
            if self.is_unitless:
                value = Quantity(value)
            else:
                raise TypeError(f"Only Quantity can be assigned to Quantity. But got {value}")
        value = value.in_unit(self.unit)

        # check index
        index = jax.tree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))

        # scatter-min
        self_value = jnp.asarray(self.mantissa)
        self_value = self_value.at[index].min(value.mantissa)
        return Quantity(self_value, unit=self.unit)

    # ---------- #
    # operations #
    # ---------- #

    def __len__(self) -> int:
        return len(self.mantissa)

    def __neg__(self) -> 'Quantity':
        return Quantity(self.mantissa.__neg__(), unit=self.unit)

    def __pos__(self) -> 'Quantity':
        return Quantity(self.mantissa.__pos__(), unit=self.unit)

    def __abs__(self) -> 'Quantity':
        return Quantity(self.mantissa.__abs__(), unit=self.unit)

    def __invert__(self) -> 'Quantity':
        return Quantity(self.mantissa.__invert__(), unit=self.unit)

    def _comparison(self, other: Any, operator_str: str, operation: Callable):
        other = _to_quantity(other)
        try:
            other_value = other.in_unit(self.unit).mantissa
        except UnitMismatchError as e:
            raise UnitMismatchError(
                f"Cannot compare {self} {operator_str} {other}",
                self.unit, other.unit,
            ) from e
        return operation(self.mantissa, other_value)

    def __eq__(self, oc) -> jax.typing.ArrayLike:
        return self._comparison(oc, "==", operator.eq)

    def __ne__(self, oc) -> jax.typing.ArrayLike:
        return self._comparison(oc, "!=", operator.ne)

    def __lt__(self, oc) -> jax.typing.ArrayLike:
        return self._comparison(oc, "<", operator.lt)

    def __le__(self, oc) -> jax.typing.ArrayLike:
        return self._comparison(oc, "<=", operator.le)

    def __gt__(self, oc) -> jax.typing.ArrayLike:
        return self._comparison(oc, ">", operator.gt)

    def __ge__(self, oc) -> jax.typing.ArrayLike:
        return self._comparison(oc, ">=", operator.ge)

    def _binary_operation(
        self,
        other,
        value_operation: Callable,
        unit_operation: Callable = lambda a, b: a,
        fail_for_mismatch: bool = False,
        operator_str: str = None,
        inplace: bool = False,
    ):
        """
        General implementation for binary operations.

        Parameters
        ----------
        other : {`Array`, `ndarray`, scalar}
            The object with which the operation should be performed.
        value_operation : function of two variables
            The function with which the two objects are combined. For example,
            `operator.mul` for a multiplication.
        unit_operation : function of two variables, optional
            The function with which the dimension of the resulting object is
            calculated (as a function of the dimensions of the two involved
            objects). For example, `operator.mul` for a multiplication. If not
            specified, the dimensions of `self` are used for the resulting
            object.
        fail_for_mismatch : bool, optional
            Whether to fail for a dimension mismatch between `self` and `other`
            (defaults to ``False``)
        operator_str : str, optional
            The string to use for the operator in an error message.
        inplace: bool, optional
            Whether to do the operation in-place (defaults to ``False``).
        """

        # format "other"
        if not isinstance(other, Quantity):
            other = _to_quantity(other)

        # format the unit and mantissa of "other"
        if fail_for_mismatch:
            other = other.in_unit(
                self.unit,
                err_msg=f"Cannot calculate \n"
                        f"{self} {operator_str} {other}, "
                        f"because units do not match: {self.unit} != {other.unit}"
            )
        other_value = other.mantissa
        other_unit = other.unit

        # calculate the new unit and mantissa
        r = Quantity(
            value_operation(self.mantissa, other_value),
            unit=unit_operation(self.unit, other_unit)
        )

        # update the mantissa in-place or not
        if inplace:
            self.update_mantissa(r.mantissa)
            return self
        else:
            return r

    def __add__(self, oc):
        if isinstance(oc, SparseMatrix):
            return oc.__radd__(self)
        return self._binary_operation(oc, operator.add, fail_for_mismatch=True, operator_str="+")

    def __radd__(self, oc):
        return self.__add__(oc)

    def __iadd__(self, oc):
        # a += b
        return self._binary_operation(oc, operator.add, fail_for_mismatch=True, operator_str="+=", inplace=True)

    def __sub__(self, oc):
        if isinstance(oc, SparseMatrix):
            return oc.__rsub__(self)
        return self._binary_operation(oc, operator.sub, fail_for_mismatch=True, operator_str="-")

    def __rsub__(self, oc):
        return Quantity(oc).__sub__(self)

    def __isub__(self, oc):
        # a -= b
        return self._binary_operation(oc, operator.sub, fail_for_mismatch=True, operator_str="-=", inplace=True)

    def __mul__(self, oc):
        if isinstance(oc, SparseMatrix):
            return oc.__rmul__(self)
        r = self._binary_operation(oc, operator.mul, operator.mul)
        return maybe_decimal(r)

    def __rmul__(self, oc):
        return self.__mul__(oc)

    def __imul__(self, oc):
        # a *= b
        raise NotImplementedError("In-place multiplication is not supported, since it changes the unit.")

    def __div__(self, oc):
        # self / oc
        if isinstance(oc, SparseMatrix):
            return oc.__rdiv__(self)
        r = self._binary_operation(oc, operator.truediv, operator.truediv)
        return maybe_decimal(r)

    def __idiv__(self, oc):
        raise NotImplementedError("In-place division is not supported, since it changes the unit.")

    def __truediv__(self, oc):
        # self / oc
        if isinstance(oc, SparseMatrix):
            return oc.__rtruediv__(self)
        return self.__div__(oc)

    def __rdiv__(self, oc):
        # oc / self
        # division with swapped arguments
        rdiv = lambda a, b: operator.truediv(b, a)
        r = self._binary_operation(oc, rdiv, rdiv)
        return maybe_decimal(r)

    def __rtruediv__(self, oc):
        # oc / self
        return self.__rdiv__(oc)

    def __itruediv__(self, oc):
        # a /= b
        raise NotImplementedError("In-place true division is not supported, since it changes the unit.")

    def __floordiv__(self, oc):
        # self // oc
        if isinstance(oc, SparseMatrix):
            return oc.__rfloordiv__(self)
        r = self._binary_operation(oc, operator.floordiv, operator.truediv)
        return maybe_decimal(r)

    def __rfloordiv__(self, oc):
        # oc // self
        rdiv = lambda a, b: operator.truediv(b, a)
        rfloordiv = lambda a, b: operator.floordiv(b, a)
        r = self._binary_operation(oc, rfloordiv, rdiv)
        return maybe_decimal(r)

    def __ifloordiv__(self, oc):
        # a //= b
        raise NotImplementedError("In-place floor division is not supported, since it changes the unit.")

    def __mod__(self, oc):
        # self % oc
        if isinstance(oc, SparseMatrix):
            return oc.__rmod__(self)
        r = self._binary_operation(oc, operator.mod, lambda ua, ub: ua, fail_for_mismatch=True, operator_str=r"%")
        return maybe_decimal(r)

    def __rmod__(self, oc):
        # oc % self
        oc = _to_quantity(oc)
        r = oc._binary_operation(self, operator.mod, lambda ua, ub: ua, fail_for_mismatch=True, operator_str=r"%")
        return maybe_decimal(r)

    def __imod__(self, oc):
        raise NotImplementedError("In-place mod is not supported, since it changes the unit.")

    def __divmod__(self, oc):
        return self.__floordiv__(oc), self.__mod__(oc)

    def __rdivmod__(self, oc):
        return self.__rfloordiv__(oc), self.__rmod__(oc)

    def __matmul__(self, oc):
        if isinstance(oc, SparseMatrix):
            return oc.__rmatmul__(self)
        r = self._binary_operation(oc, operator.matmul, operator.mul, operator_str="@")
        return maybe_decimal(r)

    def __rmatmul__(self, oc):
        oc = _to_quantity(oc)
        r = oc._binary_operation(self, operator.matmul, operator.mul, operator_str="@")
        return maybe_decimal(r)

    def __imatmul__(self, oc):
        # a @= b
        raise NotImplementedError("In-place matrix multiplication is not supported, since it changes the unit.")

    # -------------------- #

    def __pow__(self, oc):
        self = self.factorless()
        if compat_with_equinox:
            try:
                from equinox.internal._omega import ω  # noqa
                if isinstance(oc, ω):
                    return ω(self)
            except (ImportError, ModuleNotFoundError):
                pass
        if isinstance(oc, Quantity):
            if not oc.is_unitless:
                raise ValueError(f"Cannot calculate {self} ** {oc}, the exponent has to be dimensionless")
            oc = oc.mantissa
        r = Quantity(jnp.array(self.mantissa) ** oc, unit=self.unit ** oc)
        return maybe_decimal(r)

    def __rpow__(self, oc):
        # oc ** self
        if not self.is_unitless:
            raise ValueError(f"Cannot calculate {oc} ** {self}, the exponent has to be dimensionless")
        return oc ** self.mantissa

    def __ipow__(self, oc):
        # a **= b
        raise NotImplementedError("In-place power is not supported, since it changes the unit.")

    def __and__(self, oc):
        # Remove the unit from the result
        raise NotImplementedError("Bitwise operations are not supported")

    def __rand__(self, oc):
        # Remove the unit from the result
        raise NotImplementedError("Bitwise operations are not supported")

    def __iand__(self, oc):
        # Remove the unit from the result
        raise NotImplementedError("Bitwise operations are not supported")

    def __or__(self, oc):
        # Remove the unit from the result
        raise NotImplementedError("Bitwise operations are not supported")

    def __ror__(self, oc):
        # Remove the unit from the result
        raise NotImplementedError("Bitwise operations are not supported")

    def __ior__(self, oc):
        # Remove the unit from the result
        # a |= b
        raise NotImplementedError("Bitwise operations are not supported")

    def __xor__(self, oc):
        # Remove the unit from the result
        raise NotImplementedError("Bitwise operations are not supported")

    def __rxor__(self, oc):
        # Remove the unit from the result
        raise NotImplementedError("Bitwise operations are not supported")

    def __ixor__(self, oc) -> 'Quantity':
        # Remove the unit from the result
        # a ^= b
        raise NotImplementedError("Bitwise operations are not supported")

    def __lshift__(self, oc) -> 'Quantity':
        # self << oc
        if isinstance(oc, Quantity):
            if not oc.is_unitless:
                raise ValueError("The shift amount must be dimensionless")
            oc = oc.mantissa
        r = Quantity(self.mantissa << oc, unit=self.unit)
        return maybe_decimal(r)

    def __rlshift__(self, oc) -> 'Quantity | jax.typing.ArrayLike':
        # oc << self
        if not self.is_unitless:
            raise ValueError("The shift amount must be dimensionless")
        return oc << self.mantissa

    def __ilshift__(self, oc) -> 'Quantity':
        # self <<= oc
        r = self.__lshift__(oc)
        self.update_mantissa(r.mantissa)
        return self

    def __rshift__(self, oc) -> 'Quantity':
        # self >> oc
        if isinstance(oc, Quantity):
            if not oc.is_unitless:
                raise ValueError("The shift amount must be dimensionless")
            oc = oc.mantissa
        r = Quantity(self.mantissa >> oc, unit=self.unit)
        return maybe_decimal(r)

    def __rrshift__(self, oc) -> 'Quantity | jax.typing.ArrayLike':
        # oc >> self
        if not self.is_unitless:
            raise ValueError("The shift amount must be dimensionless")
        return oc >> self.mantissa

    def __irshift__(self, oc) -> 'Quantity':
        # self >>= oc
        r = self.__rshift__(oc)
        self.update_mantissa(r.mantissa)
        return self

    def __round__(self, ndigits: int = None) -> 'Quantity':
        """
        Round the mantissa to the given number of decimals.

        :param ndigits: The number of decimals to round to.
        :return: The rounded Quantity.
        """
        return Quantity(self.mantissa.__round__(ndigits), unit=self.unit)

    def __reduce__(self):
        """
        Method used by Pickle object serialization.

        Returns ``(array_with_unit, (mantissa, unit))`` so that
        ``pickle.loads(pickle.dumps(q))`` reconstructs an identical Quantity
        without bypassing ``__init__`` validation.  Using ``array_with_unit``
        (rather than ``Quantity`` directly) mirrors the pattern used by
        ``Unit.__reduce__`` and avoids issues with ``__slots__``.

        Returns
        -------
        tuple
            ``(callable, args)`` such that ``callable(*args)`` reconstructs
            the object.
        """
        return _quantity_with_unit, (self.mantissa, self.unit)

    # ----------------------- #
    #      NumPy methods      #
    # ----------------------- #

    all = _wrap_function_remove_unit(jnp.all)
    any = _wrap_function_remove_unit(jnp.any)
    nonzero = _wrap_function_remove_unit(jnp.nonzero)
    argmax = _wrap_function_remove_unit(jnp.argmax)
    argmin = _wrap_function_remove_unit(jnp.argmin)
    argsort = _wrap_function_remove_unit(jnp.argsort)

    var = _wrap_function_change_unit(jnp.var, lambda val, unit: unit ** 2)

    std = _wrap_function_keep_unit(jnp.std)
    sum = _wrap_function_keep_unit(jnp.sum)
    trace = _wrap_function_keep_unit(jnp.trace)
    cumsum = _wrap_function_keep_unit(jnp.cumsum)
    diagonal = _wrap_function_keep_unit(jnp.diagonal)
    max = _wrap_function_keep_unit(jnp.max)
    mean = _wrap_function_keep_unit(jnp.mean)
    min = _wrap_function_keep_unit(jnp.min)
    ptp = _wrap_function_keep_unit(jnp.ptp)
    ravel = _wrap_function_keep_unit(jnp.ravel)

    def __deepcopy__(self, memodict: dict):
        return Quantity(
            deepcopy(self.mantissa),
            unit=self.unit.__deepcopy__(memodict)
        )

    def round(
        self,
        decimals: int = 0,
    ) -> 'Quantity':
        """
        Evenly round to the given number of decimals.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to (default: 0).  If
            decimals is negative, it specifies the number of positions to
            the left of the decimal point.

        Returns
        -------
        rounded_array : Quantity
            An array of the same type as `a`, containing the rounded values.
            Unless `out` was specified, a new array is created.  A reference to
            the result is returned.

            The real and imaginary parts of complex numbers are rounded
            separately.  The result of rounding a float is a float.
        """
        return Quantity(jnp.round(self.mantissa, decimals), unit=self.unit)

    def astype(
        self,
        dtype: jax.typing.DTypeLike
    ) -> 'Quantity':
        """Copy of the array, cast to a specified type.

        Parameters
        ----------
        dtype: str, dtype
          Typecode or data-type to which the array is cast.
        """
        if dtype is None:
            return Quantity(self.mantissa, unit=self.unit)
        else:
            return Quantity(jnp.astype(self.mantissa, dtype), unit=self.unit)

    def clip(
        self,
        min: 'Quantity | jax.typing.ArrayLike' = None,
        max: 'Quantity | jax.typing.ArrayLike' = None,
    ) -> 'Quantity':
        """
        Return an array whose values are limited to [min, max]. One of max or min must be given.
        """
        _, min = unit_scale_align_to_first(self, min)
        _, max = unit_scale_align_to_first(self, max)
        return Quantity(jnp.clip(self.mantissa, min.mantissa, max.mantissa), unit=self.unit)

    def conj(self) -> 'Quantity':
        """Complex-conjugate all elements."""
        return Quantity(jnp.conj(self.mantissa), unit=self.unit)

    def conjugate(self) -> 'Quantity':
        """Return the complex conjugate, element-wise."""
        return Quantity(jnp.conjugate(self.mantissa), unit=self.unit)

    def copy(self) -> 'Quantity':
        """Return a copy of the quantity."""
        return type(self)(jnp.copy(self.mantissa), unit=self.unit)

    def dot(self, b) -> 'Quantity':
        """Dot product of two arrays."""
        r = self._binary_operation(b, jnp.dot, operator.mul, operator_str="@")
        return maybe_decimal(r)

    def trace(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> 'Quantity':
        """Sum along diagonals of the array, preserving units."""
        return Quantity(jnp.trace(self.mantissa, offset=offset, axis1=axis1, axis2=axis2), unit=self.unit)

    def diagonal(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> 'Quantity':
        """Return specified diagonals, preserving units."""
        return Quantity(jnp.diagonal(self.mantissa, offset=offset, axis1=axis1, axis2=axis2), unit=self.unit)

    def outer(self, b: 'Quantity') -> 'Quantity':
        """Outer product of two 1-D arrays; result unit = self.unit * b.unit."""
        b = _to_quantity(b)
        r = self._binary_operation(b, jnp.outer, operator.mul, operator_str="outer")
        return maybe_decimal(r)

    def cross(self, b: 'Quantity', axisa: int = -1, axisb: int = -1, axisc: int = -1, axis: int = None) -> 'Quantity':
        """Cross product of two arrays; result unit = self.unit * b.unit."""
        b = _to_quantity(b)
        kwargs = dict(axisa=axisa, axisb=axisb, axisc=axisc)
        if axis is not None:
            kwargs['axis'] = axis
        result_mantissa = jnp.cross(self.mantissa, b.mantissa, **kwargs)
        result_unit = self.unit * b.unit
        r = Quantity(result_mantissa, unit=result_unit)
        return maybe_decimal(r)

    def searchsorted(self, v, side: str = 'left', sorter=None) -> jax.Array:
        """Find indices where elements should be inserted to maintain order."""
        if isinstance(v, Quantity):
            v = v.in_unit(self.unit).mantissa
        return jnp.searchsorted(self.mantissa, v, side=side, sorter=sorter)

    def fill(self, value: 'Quantity') -> 'Quantity':
        """Fill the array with a scalar mantissa."""
        fail_for_dimension_mismatch(self, value, "fill")
        self[:] = value
        return self

    def flatten(self) -> 'Quantity':
        return Quantity(jnp.reshape(self.mantissa, -1), unit=self.unit)

    def item(self, *args) -> 'Quantity':
        """Copy an element of an array to a standard Python scalar and return it."""
        return Quantity(self.mantissa.item(*args), unit=self.unit)

    def prod(self, *args, **kwds) -> 'Quantity':  # TODO: check error when axis is not None
        """Return the product of the array elements over the given axis."""
        self = self.factorless()

        prod_res = jnp.prod(self.mantissa, *args, **kwds)
        # Calculating the correct dimensions is not completly trivial (e.g.
        # like doing self.dim**self.size) because prod can be called on
        # multidimensional arrays along a certain axis.
        # Our solution: Use a "dummy matrix" containing a 1 (without units) at
        # each entry and sum it, using the same keyword arguments as provided.
        # The result gives the exponent for the dimensions.
        # This relies on sum and prod having the same arguments, which is true
        # now and probably remains like this in the future
        dim_exponent = jnp.ones_like(self.mantissa).sum(*args, **kwds)
        # The result is possibly multidimensional but all entries should be
        # identical
        if dim_exponent.size > 1:
            dim_exponent = dim_exponent[-1]
        r = Quantity(jnp.array(prod_res), unit=self.unit ** dim_exponent)
        return maybe_decimal(r)

    def nanprod(self, *args, **kwds) -> 'Quantity':  # TODO: check error when axis is not None
        """Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones."""
        self = self.factorless()

        prod_res = jnp.nanprod(self.mantissa, *args, **kwds)
        nan_mask = jnp.isnan(self.mantissa)
        dim_exponent = jnp.cumsum(jnp.where(nan_mask, 0, 1), *args)
        if dim_exponent.size > 1:
            dim_exponent = dim_exponent[-1]
        r = Quantity(jnp.array(prod_res), unit=self.unit ** dim_exponent)
        return maybe_decimal(r)

    def cumprod(self, *args, **kwds):  # TODO: check error when axis is not None
        self = self.factorless()

        prod_res = jnp.cumprod(self.mantissa, *args, **kwds)
        dim_exponent = jnp.ones_like(self.mantissa).cumsum(*args, **kwds)
        if dim_exponent.size > 1:
            dim_exponent = dim_exponent[-1]
        r = Quantity(jnp.array(prod_res), unit=self.unit ** dim_exponent)
        return maybe_decimal(r)

    def nancumprod(self, *args, **kwds):  # TODO: check error when axis is not None
        self = self.factorless()

        prod_res = jnp.nancumprod(self.mantissa, *args, **kwds)
        nan_mask = jnp.isnan(self.mantissa)
        dim_exponent = jnp.cumsum(jnp.where(nan_mask, 0, 1), *args)
        if dim_exponent.size > 1:
            dim_exponent = dim_exponent[-1]
        r = Quantity(jnp.array(prod_res), unit=self.unit ** dim_exponent)
        return maybe_decimal(r)

    def put(self, indices, values) -> 'Quantity':
        """Replaces specified elements of an array with given values.

        Parameters
        ----------
        indices: array_like
          Target indices, interpreted as integers.
        values: array_like
          Values to place in the array at target indices.
        """
        fail_for_dimension_mismatch(self, values, "put")
        self.__setitem__(indices, values)
        return self

    def repeat(self, repeats, axis=None) -> 'Quantity':
        """Repeat elements of an array."""
        r = jnp.repeat(self.mantissa, repeats=repeats, axis=axis)
        return Quantity(r, unit=self.unit)

    def reshape(self, shape, order='C') -> 'Quantity':
        """Returns an array containing the same data with a new shape."""
        return Quantity(jnp.reshape(self.mantissa, shape, order=order), unit=self.unit)

    def resize(self, new_shape) -> 'Quantity':
        """Change shape and size of array in-place."""
        self.update_mantissa(jnp.resize(self.mantissa, new_shape))
        return self

    def sort(self, axis=-1, stable=True, order=None) -> 'Quantity':
        """Sort an array in-place.

        Parameters
        ----------
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
        self.update_mantissa(jnp.sort(self.mantissa, axis=axis, stable=stable, order=order))
        return self

    def squeeze(self, axis=None) -> 'Quantity':
        """Remove axes of length one from ``a``."""
        return Quantity(jnp.squeeze(self.mantissa, axis=axis), unit=self.unit)

    def swapaxes(self, axis1, axis2) -> 'Quantity':
        """Return a view of the array with `axis1` and `axis2` interchanged."""
        return Quantity(jnp.swapaxes(self.mantissa, axis1, axis2), unit=self.unit)

    def split(self, indices_or_sections, axis=0) -> 'list[Quantity]':
        """Split an array into multiple sub-arrays as views into ``ary``.

        Parameters
        ----------
        indices_or_sections : int, 1-D array
          If `indices_or_sections` is an integer, N, the array will be divided
          into N equal arrays along `axis`.  If such a split is not possible,
          an error is raised.

          If `indices_or_sections` is a 1-D array of sorted integers, the entries
          indicate where along `axis` the array is split.  For example,
          ``[2, 3]`` would, for ``axis=0``, result in

            - ary[:2]
            - ary[2:3]
            - ary[3:]

          If an index exceeds the dimension of the array along `axis`,
          an empty sub-array is returned correspondingly.
        axis : int, optional
          The axis along which to split, default is 0.

        Returns
        -------
        sub-arrays : list of ndarrays
          A list of sub-arrays as views into `ary`.
        """
        return [Quantity(a, unit=self.unit) for a in jnp.split(self.mantissa, indices_or_sections, axis=axis)]

    def take(
        self,
        indices,
        axis=None,
        mode=None,
        unique_indices=False,
        indices_are_sorted=False,
        fill_value=None,
    ) -> 'Quantity':
        """Return an array formed from the elements of a at the given indices."""

        if isinstance(fill_value, Quantity):
            fail_for_dimension_mismatch(self, fill_value, "take")
            fill_value = unit_scale_align_to_first(self, fill_value)[1].mantissa
        elif fill_value is not None:
            if not self.is_unitless:
                raise TypeError(f"fill_value must be a Quantity when the unit {self.unit}. But got {fill_value}")
        return Quantity(
            jnp.take(
                self.mantissa,
                indices=indices,
                axis=axis,
                mode=mode,
                unique_indices=unique_indices,
                indices_are_sorted=indices_are_sorted,
                fill_value=fill_value
            ),
            unit=self.unit
        )

    def tolist(self):
        """Return the array as an ``a.ndim``-levels deep nested list of Python scalars.

        Return a copy of the array data as a (nested) Python list.
        Data items are converted to the nearest compatible builtin Python type, via
        the `~numpy.ndarray.item` function.

        If ``a.ndim`` is 0, then since the depth of the nested list is 0, it will
        not be a list at all, but a simple Python scalar.
        """
        if isinstance(self.mantissa, numbers.Number):
            list_mantissa = self.mantissa
        else:
            list_mantissa = self.mantissa.tolist()
        return _replace_with_array(list_mantissa, self.unit)

    def transpose(self, *axes) -> 'Quantity':
        """Returns a view of the array with axes transposed.

        For a 1-D array this has no effect, as a transposed vector is simply the
        same vector. To convert a 1-D array into a 2D column vector, an additional
        dimension must be added. `jnp.atleast2d(a).T` achieves this, as does
        `a[:, jnp.newaxis]`.
        For a 2-D array, this is a standard matrix transpose.
        For an n-D array, if axes are given, their order indicates how the
        axes are permuted (see Examples). If axes are not provided and
        ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
        ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

        Parameters
        ----------
        axes : None, tuple of ints, or `n` ints

         * None or no argument: reverses the order of the axes.

         * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
           `i`-th axis becomes `a.transpose()`'s `j`-th axis.

         * `n` ints: same as an n-tuple of the same ints (this form is
           intended simply as a "convenience" alternative to the tuple form)

        Returns
        -------
        out : ndarray
            View of `a`, with axes suitably permuted.
        """
        return Quantity(jnp.transpose(self.mantissa, *axes), unit=self.unit)

    def tile(self, reps) -> 'Quantity':
        """Construct an array by repeating A the number of times given by reps.

        If `reps` has length ``d``, the result will have dimension of
        ``max(d, A.ndim)``.

        If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
        axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
        or shape (1, 1, 3) for 3-D replication. If this is not the desired
        behavior, promote `A` to d-dimensions manually before calling this
        function.

        If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
        Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
        (1, 1, 2, 2).

        Note : Although tile may be used for broadcasting, it is strongly
        recommended to use numpy's broadcasting operations and functions.

        Parameters
        ----------
        reps : array_like
            The number of repetitions of `A` along each axis.

        Returns
        -------
        c : ndarray
            The tiled output array.
        """
        return Quantity(jnp.tile(self.mantissa, reps), unit=self.unit)

    def view(self, *args, dtype=None) -> 'Quantity':
        r"""New view of array with the same data.

        This function is compatible with pytorch syntax.

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`shape`.

        The returned tensor shares the same data and must have the same number
        of elements, but may have a different size. For a tensor to be viewed, the new
        view size must be compatible with its original size and stride, i.e., each new
        view dimension must either be a subspace of an original dimension, or only span
        across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
        contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

        .. math::

          \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

        Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
        without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
        :meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
        returns a view if the shapes are compatible, and copies (equivalent to calling
        :meth:`contiguous`) otherwise.

        Args:
            shape (int...): the desired size

        Example::

            >>> import jax.numpy as jnp, saiunit
            >>> x = saiunit.Quantity(jnp.ones((4, 4)))
            >>> x.shape
            (4, 4)
            >>> y = x.view(16)
            >>> y.shape
            (16,)
            >>> z = x.view(2, 8)
            >>> z.shape
            (2, 8)


        .. method:: view(dtype) -> Tensor
           :noindex:

        Returns a new tensor with the same data as the :attr:`self` tensor but of a
        different :attr:`dtype`.

        If the element size of :attr:`dtype` is different than that of ``self.dtype``,
        then the size of the last dimension of the output will be scaled
        proportionally.  For instance, if :attr:`dtype` element size is twice that of
        ``self.dtype``, then each pair of elements in the last dimension of
        :attr:`self` will be combined, and the size of the last dimension of the output
        will be half that of :attr:`self`. If :attr:`dtype` element size is half that
        of ``self.dtype``, then each element in the last dimension of :attr:`self` will
        be split in two, and the size of the last dimension of the output will be
        double that of :attr:`self`. For this to be possible, the following conditions
        must be true:

            * ``self.dim()`` must be greater than 0.
            * ``self.stride(-1)`` must be 1.

        Additionally, if the element size of :attr:`dtype` is greater than that of
        ``self.dtype``, the following conditions must be true as well:

            * ``self.size(-1)`` must be divisible by the ratio between the element
              sizes of the dtypes.
            * ``self.storage_offset()`` must be divisible by the ratio between the
              element sizes of the dtypes.
            * The strides of all dimensions, except the last dimension, must be
              divisible by the ratio between the element sizes of the dtypes.

        If any of the above conditions are not met, an error is thrown.


        Args:
            dtype (:class:`dtype`): the desired dtype

        Example::

            >>> x = brainstate.random.randn(4, 4)
            >>> x
            Array([[ 0.9482, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])
            >>> x.dtype
            brainstate.math.float32

            >>> y = x.view(numpy.int32)
            >>> y
            tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
                    [-1105482831,  1061112040,  1057999968, -1084397505],
                    [-1071760287, -1123489973, -1097310419, -1084649136],
                    [-1101533110,  1073668768, -1082790149, -1088634448]],
                dtype=numpy.int32)
            >>> y[0, 0] = 1000000000
            >>> x
            tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
                    [-0.1520,  0.7472,  0.5617, -0.8649],
                    [-2.4724, -0.0334, -0.2976, -0.8499],
                    [-0.2109,  1.9913, -0.9607, -0.6123]])

            >>> x.view(numpy.complex64)
            tensor([[ 0.0047-0.0310j,  1.4999-0.5316j],
                    [-0.1520+0.7472j,  0.5617-0.8649j],
                    [-2.4724-0.0334j, -0.2976-0.8499j],
                    [-0.2109+1.9913j, -0.9607-0.6123j]])
            >>> x.view(numpy.complex64).size
            [4, 2]

            >>> x.view(numpy.uint8)
            tensor([[  0, 202, 154,  59, 182, 243, 253, 188, 185, 252, 191,  63, 240,  22,
                         8, 191],
                    [227, 165,  27, 190, 128,  72,  63,  63, 146, 203,  15,  63,  22, 106,
                        93, 191],
                    [205,  59,  30, 192, 112, 206,   8, 189,   7,  95, 152, 190,  12, 147,
                        89, 191],
                    [ 43, 246,  87, 190, 235, 226, 254,  63, 111, 240, 117, 191, 177, 191,
                        28, 191]], dtype=uint8)
            >>> x.view(numpy.uint8).size
            [4, 16]

        """
        if len(args) == 0:
            if dtype is None:
                raise ValueError('Provide dtype or shape.')
            else:
                return Quantity(self.mantissa.view(dtype), unit=self.unit)
        else:
            if isinstance(args[0], int):  # shape
                if dtype is not None:
                    raise ValueError('Provide one of dtype or shape. Not both.')
                return Quantity(self.mantissa.reshape(*args), unit=self.unit)
            else:  # dtype
                assert not isinstance(args[0], int)
                assert dtype is None
                return Quantity(self.mantissa.view(args[0]), unit=self.unit)

    # ------------------
    # NumPy support
    # ------------------

    def __array__(self, dtype: jax.typing.DTypeLike | None = None) -> np.ndarray:
        """Support ``numpy.array()`` and ``numpy.asarray()`` functions."""
        if self.dim.is_dimensionless:
            return np.asarray(self.to_decimal(), dtype=dtype)
        else:
            raise TypeError(
                f"Only dimensionless quantities can be "
                f"converted to NumPy arrays. But got {self}"
            )

    def __float__(self):
        if self.dim.is_dimensionless and self.ndim == 0:
            return float(self.to_decimal())
        else:
            raise TypeError(
                "Only dimensionless scalar quantities can be "
                f"converted to Python scalars. But got {self}"
            )

    def __int__(self):
        if self.dim.is_dimensionless and self.ndim == 0:
            return int(self.to_decimal())
        else:
            raise TypeError(
                "only dimensionless scalar quantities can be "
                f"converted to Python scalars. But got {self}"
            )

    def __index__(self):
        if self.dim.is_dimensionless:
            return operator.index(self.to_decimal())
        else:
            raise TypeError(
                "only dimensionless quantities can be "
                f"converted to a Python index. But got {self}"
            )

    # ----------------------
    # PyTorch compatibility
    # ----------------------

    def unsqueeze(self, axis: int) -> 'Quantity':
        """
        Array.unsqueeze(dim) -> Array, or so called Tensor
        equals
        Array.expand_dims(dim)

        See :func:`brainstate.math.unsqueeze`
        """
        return Quantity(jnp.expand_dims(self.mantissa, axis), unit=self.unit)

    def expand_dims(self, axis: int | Sequence[int]) -> 'Quantity':
        """
        Expand the shape of an array.

        Parameters
        ----------
        axis : int or tuple of ints
            Position in the expanded axes where the new axis is placed.

        Returns
        -------
        expanded : Quantity
            A view with the new axis inserted.
        """
        return Quantity(jnp.expand_dims(self.mantissa, axis), unit=self.unit)

    def expand_as(self, array: 'Quantity | jax.typing.ArrayLike') -> 'Quantity':
        """
        Expand an array to a shape of another array.

        Parameters
        ----------
        array : Quantity

        Returns
        -------
        expanded : Quantity
            A readonly view on the original array with the given shape of array. It is
            typically not contiguous. Furthermore, more than one element of a
            expanded array may refer to a single memory location.
        """
        if isinstance(array, Quantity):
            fail_for_dimension_mismatch(self, array, "expand_as (Quantity)")
            array = array.mantissa
        return Quantity(jnp.broadcast_to(self.mantissa, array), unit=self.unit)

    def pow(self, oc) -> 'Quantity':
        return self.__pow__(oc)

    def clone(self) -> 'Quantity':
        return self.copy()

    def tree_flatten(self) -> tuple[tuple[jax.typing.ArrayLike], Unit]:
        """
        Tree flattens the data.

        Returns:
          The data and the dimension.
        """
        return (self.mantissa,), self.unit

    @classmethod
    def tree_unflatten(cls, unit, values) -> 'Quantity':
        """
        Tree unflattens the data.

        Args:
          unit: The unit.
          values: The data.

        Returns:
          The Quantity object.
        """
        return cls(*values, unit=unit)

    def cuda(self, device=None) -> 'Quantity':
        device = jax.devices('cuda')[0] if device is None else device
        self.update_mantissa(jax.device_put(self.mantissa, device))
        return self

    def cpu(self, device=None) -> 'Quantity':
        device = jax.devices('cpu')[0] if device is None else device
        self.update_mantissa(jax.device_put(self.mantissa, device))
        return self

    # dtype exchanging #
    # ---------------- #
    def half(self) -> 'Quantity':
        return Quantity(jnp.asarray(self.mantissa, dtype=jnp.float16), unit=self.unit)

    def float(self) -> 'Quantity':
        return Quantity(jnp.asarray(self.mantissa, dtype=jnp.float32), unit=self.unit)

    def double(self) -> 'Quantity':
        return Quantity(jnp.asarray(self.mantissa, dtype=jnp.float64), unit=self.unit)


# ---------------------------------------------------------------------------
# _IndexUpdateHelper
# ---------------------------------------------------------------------------

class _IndexUpdateHelper:
    """
    Helper property for index update functionality.
    """
    __slots__ = ("quantity",)

    def __init__(self, quantity: Quantity):
        if not isinstance(quantity, Quantity):
            raise TypeError(f"quantity must be a Quantity object, but got {quantity}")
        self.quantity = quantity

    def __getitem__(self, index: Any) -> '_IndexUpdateRef':
        return _IndexUpdateRef(index, self.quantity)

    def __repr__(self):
        return f"_IndexUpdateHelper({self.quantity})"


# ---------------------------------------------------------------------------
# _IndexUpdateRef
# ---------------------------------------------------------------------------

class _IndexUpdateRef:
    """
    Helper object to call indexed update functions for an (advanced) index.

    This object references a source array and a specific indexer into that array.
    Methods on this object return copies of the source array that have been
    modified at the positions specified by the indexer.
    """
    __slots__ = ("quantity", "index", "mantissa_at", "unit")

    def __init__(self, index, quantity: Quantity):
        self.index = jax.tree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))
        self.quantity = quantity
        self.mantissa_at = jnp.asarray(quantity.mantissa).at
        self.unit = quantity.unit

    def __repr__(self) -> str:
        return f"_IndexUpdateRef({self.quantity}, {self.index!r})"

    def get(
        self,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | None = None,
        fill_value: StaticScalar | None = None
    ) -> Quantity:
        """Equivalent to ``x[idx]``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexing <numpy.doc.indexing>` ``x[idx]``. This function differs from
        the usual array indexing syntax in that it allows additional keyword
        arguments ``indices_are_sorted`` and ``unique_indices`` to be passed.
        """
        if fill_value is not None:
            fill_value = Quantity(fill_value).in_unit(self.unit).mantissa
        return Quantity(
            self.mantissa_at[self.index].get(
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode,
                fill_value=fill_value
            ),
            unit=self.unit
        )

    def set(
        self,
        values: Any,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | None = None,
    ) -> Quantity:
        """Pure equivalent of ``x[idx] = y``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:`indexed assignment <numpy.doc.indexing>` ``x[idx] = y``.
        """
        values = Quantity(values).in_unit(self.unit).mantissa
        return Quantity(
            self.mantissa_at[self.index].set(
                values,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode,
            ),
            unit=self.unit
        )

    def add(
        self,
        values: Any,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | None = None
    ) -> Quantity:
        """Pure equivalent of ``x[idx] += y``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] += y``.

        """
        values = Quantity(values).in_unit(self.unit).mantissa
        return Quantity(
            self.mantissa_at[self.index].add(
                values,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode
            ),
            unit=self.unit
        )

    def multiply(
        self,
        values: Any,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | None = None
    ) -> Quantity:
        """Pure equivalent of ``x[idx] *= y``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] *= y``.

        """
        values = Quantity(values)
        return Quantity(
            self.mantissa_at[self.index].multiply(
                values.mantissa,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode
            ),
            unit=self.unit * values.unit
        )

    mul = multiply

    def divide(
        self,
        values: Any,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | None = None
    ) -> Quantity:
        """Pure equivalent of ``x[idx] /= y``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] /= y``.

        """
        values = Quantity(values)
        return Quantity(
            self.mantissa_at[self.index].divide(
                values.mantissa,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode
            ),
            unit=self.unit / values.unit
        )

    div = divide

    def power(
        self,
        values: Any,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | None = None
    ) -> Quantity:
        """Pure equivalent of ``x[idx] **= y``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] **= y``.

        """
        if not isinstance(values, int):
            raise TypeError(f"values must be an integer, but got {values}")
        return Quantity(
            self.mantissa_at[self.index].power(
                values,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode
            ),
            unit=self.unit ** values
        )

    def min(
        self,
        values: Any,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | None = None
    ) -> Quantity:
        """Pure equivalent of ``x[idx] = minimum(x[idx], y)``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>`
        ``x[idx] = minimum(x[idx], y)``.

        """
        values = Quantity(values).in_unit(self.unit).mantissa
        return Quantity(
            self.mantissa_at[self.index].min(
                values,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode
            ),
            unit=self.unit
        )

    def max(
        self,
        values: Any,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | None = None
    ) -> Quantity:
        """Pure equivalent of ``x[idx] = maximum(x[idx], y)``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexed assignment <numpy.doc.indexing>`
        ``x[idx] = maximum(x[idx], y)``.

        """
        values = Quantity(values).in_unit(self.unit).mantissa
        return Quantity(
            self.mantissa_at[self.index].max(
                values,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode
            ),
            unit=self.unit
        )

    def apply(
        self,
        mantissa_fun: Callable[[jax.typing.ArrayLike], jax.typing.ArrayLike],
        unit_fun: Callable[[Unit], Unit] | None = None,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | None = None
    ) -> Quantity:
        """Pure equivalent of ``func.at(x, idx)`` for a unary ufunc ``func``.

        Returns the value of ``x`` that would result from applying the unary
        function ``func`` to ``x`` at the given indices. This is similar to
        ``x.at[idx].set(func(x[idx]))``, but differs in the case of repeated indices:
        in ``x.at[idx].apply(func)``, repeated indices result in the function being
        applied multiple times.

        Note that in the current implementation, ``scatter_apply`` is not compatible
        with automatic differentiation.

        Parameters
        ----------
        mantissa_fun : callable
            Applied to the mantissa values at the given indices.
        unit_fun : callable, optional
            Transforms the unit of the result. If omitted the unit is preserved
            (unit-preserving operations such as ``jnp.abs``).
        """
        result_unit = unit_fun(self.unit) if unit_fun is not None else self.unit
        return Quantity(
            self.mantissa_at[self.index].apply(
                mantissa_fun,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode
            ),
            unit=result_unit
        )


# ---------------------------------------------------------------------------
# _replace_with_array
# ---------------------------------------------------------------------------

def _replace_with_array(seq, unit):
    """
    Replace all the elements in the list with an equivalent `Array`
    with the given `unit`.
    """
    # No recursion needed for single values
    if not isinstance(seq, list):
        return Quantity(seq, unit=unit)

    def top_replace(s):
        """
        Recursively descend into the list.
        """
        for i in s:
            if not isinstance(i, list):
                yield Quantity(i, unit=unit)
            else:
                yield list(top_replace(i))

    return list(top_replace(seq))
