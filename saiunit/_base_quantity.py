# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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
import numbers
import operator
import re
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from ._backend import get_backend, is_jax_array, is_numpy_array
from ._jax_compat import (
    HAS_JAX,
    jax,
    jnp,
    canonicalize_dtype as _canonicalize_dtype,
    ensure_compile_time_eval as _ensure_compile_time_eval,
    register_pytree_node_class,
    result_type as _result_type,
    tree as _jtree,
)
from ._typing import (
    Array,
    ArrayLike,
    Array as _JaxArray,
    DTypeLike,
)
from ._base_dimension import Dimension, UnitMismatchError, _is_tracer
from ._base_getters import (
    get_dim,
    fail_for_dimension_mismatch,
    maybe_decimal,
    _to_quantity,
    unit_scale_align_to_first,
)
from ._base_unit import Unit, UNITLESS
from ._exceptions import BackendError
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


def _xp_attr(name: str):
    """Return a callable that resolves ``<name>`` on the input's backend.

    Used at class-body time to wrap ``Quantity`` reduction methods (``sum``,
    ``mean``, …).  Backend resolution is deferred until call time so that a
    ``Quantity`` over a torch / dask / ndonnx mantissa dispatches into that
    backend instead of unconditionally hitting JAX/NumPy.
    """
    def _resolved(x, *args, **kwargs):
        from saiunit._backend import get_backend
        xp = get_backend(x)
        op = getattr(xp, name, None)
        if op is None:
            backend = getattr(xp, "__name__", repr(xp))
            raise AttributeError(
                f"saiunit: backend {backend!r} has no operation {name!r}"
            )
        return op(x, *args, **kwargs)
    _resolved.__name__ = name
    return _resolved


def compatible_with_equinox(mode: bool = True):
    """
    Enable or disable compatibility with the Equinox library.

    When enabled, ``Quantity`` objects interact correctly with Equinox
    transformations such as those used in
    `unit-aware diffrax <https://github.com/chaoming0625/diffrax>`_.

    Parameters
    ----------
    mode : bool, optional
        If ``True`` (default), enable Equinox compatibility.
        If ``False``, disable it.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.compatible_with_equinox(True)   # enable
        >>> u.compatible_with_equinox(False)  # disable

    See Also
    --------
    Quantity : The core physical-quantity class affected by this setting.
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

    @functools.wraps(func)
    def f(x: 'Quantity', *args, **kwds):  # pylint: disable=C0111
        return Quantity(func(x.mantissa, *args, **kwds), unit=x.unit)

    f._arg_units = [None]
    f._return_unit = lambda u: u
    f._do_not_run_doctests = True
    return f


def _dask_materialization_guard(mantissa, op_name: str) -> None:
    """Raise ``BackendError`` if ``mantissa`` is a dask array.

    Used by ``Quantity`` methods that would otherwise call ``.compute()``
    implicitly (``__float__``, ``__int__``, ``__array__``, ``tolist``, etc.).
    The caller passes a human-readable ``op_name`` for the error message.
    """
    from saiunit._backend import is_dask_array
    if is_dask_array(mantissa):
        from saiunit._exceptions import BackendError
        raise BackendError(
            f"{op_name} would materialize a dask-backed Quantity. "
            f"Call `q.mantissa.compute()` first."
        )


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

    @functools.wraps(func)
    def f(x, *args, **kwds):  # pylint: disable=C0111
        assert isinstance(x, Quantity), "Only Quantity objects can be passed to this function"
        x = x.factorless()
        return maybe_decimal(Quantity(func(x.mantissa, *args, **kwds), unit=unit_fun(x.unit, x.unit)))

    f._arg_units = [None]
    f._return_unit = unit_fun
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

    @functools.wraps(func)
    def f(x, *args, **kwds):  # pylint: disable=C0111
        assert isinstance(x, Quantity), "Only Quantity objects can be passed to this function"
        return func(x.mantissa, *args, **kwds)

    f._arg_units = [None]
    f._return_unit = 1
    f._do_not_run_doctests = True
    return f


# ---------------------------------------------------------------------------
# List processing helpers
# ---------------------------------------------------------------------------

def _zoom_values_with_units(
    values: Sequence[ArrayLike],
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


def _check_units_and_collect_values(lst) -> tuple[ArrayLike, Unit]:
    units: list = []
    values: list = []

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

    # Respect the active default backend rather than always falling to
    # ``jnp.asarray`` when JAX is installed — otherwise a user inside
    # ``using_backend("torch")`` would still get a JAX array out of
    # ``Quantity([1*mV, 2*mV])``.
    from saiunit._backend import _xp_for, get_default_backend
    _xp = _xp_for(get_default_backend() or ("jax" if HAS_JAX else "numpy"))

    if len(units):
        # Normalize None (plain scalars) to UNITLESS so they are
        # compatible with explicitly unitless Quantity values.
        units = [UNITLESS if u is None else u for u in units]
        first_unit = units[0]
        if not all(first_unit.has_same_dim(unit) for unit in units):
            raise TypeError(f"All elements must have the same units, but got {units}")
        return _xp.asarray(_zoom_values_with_units(values, units)), first_unit
    else:
        return _xp.asarray(values), UNITLESS


def _process_list_with_units(value: 'list | tuple') -> tuple[ArrayLike, Unit]:
    values, unit = _check_units_and_collect_values(value)
    return values, unit


def _element_not_quantity(x):
    assert not isinstance(x, Quantity), f"Expected not a Quantity object, but got {x}"
    return x


def _reduction_count_from_shape(shape, axis) -> int:
    """Number of elements reduced over ``axis`` for an array of static ``shape``.

    JIT-safe: depends only on the static shape, never on values.
    """
    if axis is None:
        n = 1
        for d in shape:
            n *= int(d)
        return n
    if isinstance(axis, (int, np.integer)):
        return int(shape[int(axis)])
    n = 1
    for a in axis:
        n *= int(shape[int(a)])
    return n


def _is_concrete_zero(mantissa) -> bool:
    """Return True iff ``mantissa`` is a concrete (non-traced) numeric zero.

    Used to honour the physics convention that ``0`` is compatible with any
    dimension. Returns False under JIT tracing or for any non-zero value.
    """
    if _is_tracer(mantissa):
        return False
    if isinstance(mantissa, (int, float, complex)):
        return mantissa == 0
    if isinstance(mantissa, (np.ndarray, _JaxArray, np.generic)):
        try:
            arr = np.asarray(mantissa)
        except Exception:
            return False
        if arr.size == 0:
            return False
        return bool(np.all(arr == 0))
    return False


# ---------------------------------------------------------------------------
# Pickle helper
# ---------------------------------------------------------------------------

def _quantity_with_unit(mantissa, unit):
    """Private reconstruction helper for Quantity pickling.
    """
    return Quantity(mantissa, unit=unit)


_quantity_with_unit.__module__ = 'saiunit._base_quantity'


def _reject_lazy_materialization(m, attr_name: str) -> None:
    """Raise :class:`BackendError` if accessing ``attr_name`` would force
    materialisation of a lazy / symbolic mantissa.

    Used by property fallbacks (``Quantity.strides``, ``.flat``, ``.T`` …)
    that would otherwise silently call ``.compute()`` on a dask graph or
    ``.unwrap_numpy()`` on an ndonnx symbolic array.
    """
    from saiunit._backend import is_dask_array, is_ndonnx_array
    if is_dask_array(m):
        raise BackendError(
            f"{attr_name} requires materializing a lazy dask array. "
            f"Call .compute() on the mantissa explicitly first."
        )
    if is_ndonnx_array(m):
        raise BackendError(
            f"{attr_name} is not supported for ndonnx symbolic arrays."
        )


# ---------------------------------------------------------------------------
# Scatter helper (backend-aware ``arr.at[idx].<op>(value)`` equivalent)
# ---------------------------------------------------------------------------

# Map legacy op names used by Quantity.__setitem__ / scatter_* to the
# canonical names exported by saiunit._scatter.
_LEGACY_SCATTER_OP_ALIASES = {"mul": "multiply"}


def _scatter(mantissa, index, value, op: str):
    """Return a new array with ``op`` applied at ``index``.

    Thin wrapper around :func:`saiunit._scatter.scatter` that translates the
    legacy op names used internally (``"mul"``) into the canonical ones
    (``"multiply"``). All supported backends (numpy, jax, cupy, torch, dask,
    ndonnx) are routed through the unified dispatch.
    """
    from saiunit._scatter import scatter as _scatter_dispatch
    canonical = _LEGACY_SCATTER_OP_ALIASES.get(op, op)
    return _scatter_dispatch(mantissa, index, value, canonical)


# ---------------------------------------------------------------------------
# numpy ufunc dispatch table (used by Quantity.__array_ufunc__)
# ---------------------------------------------------------------------------

_UFUNC_DISPATCH: dict = {}

# Map of numpy binary ufuncs → (forward dunder, reverse dunder) on Quantity.
# Used to delegate mixed Quantity+plain ufunc calls back to operator dispatch.
_BINARY_UFUNC_OPNAMES: dict = {
    np.add: ("__add__", "__radd__"),
    np.subtract: ("__sub__", "__rsub__"),
    np.multiply: ("__mul__", "__rmul__"),
    np.true_divide: ("__truediv__", "__rtruediv__"),
    np.floor_divide: ("__floordiv__", "__rfloordiv__"),
    np.mod: ("__mod__", "__rmod__"),
    np.power: ("__pow__", "__rpow__"),
    np.equal: ("__eq__", "__eq__"),
    np.not_equal: ("__ne__", "__ne__"),
    np.less: ("__lt__", "__gt__"),
    np.less_equal: ("__le__", "__ge__"),
    np.greater: ("__gt__", "__lt__"),
    np.greater_equal: ("__ge__", "__le__"),
}


def _build_ufunc_dispatch() -> dict:
    """Lazily build the numpy ufunc → saiunit.math function table.

    Only ufuncs in this table are unit-safe. Anything else returns
    ``NotImplemented`` from ``__array_ufunc__``.
    """
    from saiunit import math as _u_math
    table = {
        # arithmetic
        np.add: _u_math.add,
        np.subtract: _u_math.subtract,
        np.multiply: _u_math.multiply,
        np.true_divide: _u_math.true_divide,
        np.floor_divide: _u_math.floor_divide,
        np.mod: _u_math.mod,
        np.power: _u_math.power,
        np.negative: _u_math.negative,
        np.positive: _u_math.positive,
        np.absolute: _u_math.absolute,
        np.abs: _u_math.absolute,
        # comparison
        np.equal: _u_math.equal,
        np.not_equal: _u_math.not_equal,
        np.less: _u_math.less,
        np.less_equal: _u_math.less_equal,
        np.greater: _u_math.greater,
        np.greater_equal: _u_math.greater_equal,
        # trig
        np.sin: _u_math.sin,
        np.cos: _u_math.cos,
        np.tan: _u_math.tan,
        np.arcsin: _u_math.arcsin,
        np.arccos: _u_math.arccos,
        np.arctan: _u_math.arctan,
        np.arctan2: _u_math.arctan2,
        # exp/log
        np.exp: _u_math.exp,
        np.log: _u_math.log,
        np.log2: _u_math.log2,
        np.log10: _u_math.log10,
        # other common
        np.sqrt: _u_math.sqrt,
        np.square: _u_math.square,
        np.isfinite: _u_math.isfinite,
        np.isnan: _u_math.isnan,
        np.isinf: _u_math.isinf,
    }
    return table


# ---------------------------------------------------------------------------
# Quantity class
# ---------------------------------------------------------------------------

@register_pytree_node_class
class Quantity:
    """
    A numerical value paired with a physical unit.

    ``Quantity`` is the central data structure in ``saiunit``.  It stores a
    *mantissa* (the raw numerical data, typically a JAX array) together with a
    :class:`Unit` that describes the physical dimensions and scale.  Arithmetic
    on ``Quantity`` objects automatically tracks and checks units, raising
    :class:`UnitMismatchError` when incompatible quantities are combined.

    ``Quantity`` is registered as a JAX pytree, so it works transparently with
    ``jax.jit``, ``jax.grad``, ``jax.vmap``, and other JAX transformations.

    Parameters
    ----------
    mantissa : array_like, number, Unit, or Quantity
        The numerical value(s).  If a :class:`Unit` is passed, the mantissa is
        set to ``1.0`` and that unit is adopted.  If a :class:`Quantity` is
        passed, its mantissa and unit are used (converted to *unit* when
        given).
    unit : Unit, optional
        The physical unit.  Defaults to ``UNITLESS``.
    dtype : dtype, optional
        If provided, the mantissa is cast to this dtype on construction.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> # Scalar with unit
        >>> q = u.Quantity(3.0, unit=u.mV)
        >>> q
        Quantity(3., "mV")
        >>> # Array with unit via multiplication shorthand
        >>> arr = jnp.array([1.0, 2.0, 3.0]) * u.mV
        >>> arr.shape
        (3,)
        >>> # From a Unit object directly
        >>> u.Quantity(u.metre)
        Quantity(1., "m")

    See Also
    --------
    Unit : Represents a physical unit (dimension + scale).
    compatible_with_equinox : Toggle Equinox interoperability.
    """

    __module__ = "saiunit"
    __slots__ = ('_mantissa', '_unit')
    __array_priority__ = 1000
    _mantissa: 'ArrayLike'
    _unit: Unit

    def __class_getitem__(cls, item: Unit | str) -> type['Quantity']:
        """Enable ``Quantity[unit]`` and ``Quantity["physical_type"]`` annotations.

        Returns a type that supports ``isinstance`` checks and can be used as
        a type annotation.

        Parameters
        ----------
        item : Unit or str
            A :class:`Unit` instance (e.g. ``u.meter``) or a string naming a
            physical type (e.g. ``"length"``, ``"speed"``).

        Returns
        -------
        type
            A class supporting ``isinstance(quantity, Quantity[unit])``.

        Examples
        --------
        >>> import saiunit as u
        >>> x = 2.0 * u.kmeter
        >>> isinstance(x, u.Quantity[u.meter])    # dimension check
        True
        >>> isinstance(x, u.Quantity["length"])    # physical type check
        True
        >>> isinstance(x, u.Quantity["mass"])      # wrong dimension
        False

        Notes
        -----
        Some static analyzers may report warnings for
        ``isinstance(x, Quantity["..."])`` because they interpret this syntax
        as parameterized generics. For IDE-safe runtime checks, use
        :func:`saiunit.typing.quantity_type`.
        """
        from .typing import _make_annotated_quantity_type
        return _make_annotated_quantity_type(item)

    def __init__(
        self,
        mantissa: PyTree | Unit,
        unit: 'Unit | str | None' = UNITLESS,
        dtype: DTypeLike | None = None,
    ):

        # ``ensure_compile_time_eval`` is a no-op outside tracing; inside a
        # JIT trace it forces constant mantissas (lists, scalars, parse_unit
        # output) through eager evaluation so they are baked into the
        # traced graph rather than promoted to tracers.
        with _ensure_compile_time_eval():

            # String-based unit: Quantity(1.0, "mV")
            if isinstance(unit, str):
                from ._base_unit import parse_unit
                unit = parse_unit(unit)
            if unit is None:
                unit = UNITLESS
            assert isinstance(unit, Unit), f"Expected a Unit, got {type(unit).__name__}: {unit!r}"

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
                # Respect the default backend for list/tuple inputs.
                from saiunit._backend import _xp_for, get_default_backend
                default = get_default_backend() or "jax"
                xp = _xp_for(default)
                # ndonnx.asarray can't infer dtype from a JAX array; route via numpy.
                if default == "ndonnx" and isinstance(mantissa, _JaxArray):
                    mantissa = np.asarray(mantissa)
                if dtype is not None:
                    mantissa = xp.asarray(mantissa, dtype=dtype)
                else:
                    mantissa = xp.asarray(mantissa)

            # array mantissa
            elif isinstance(mantissa, Quantity):
                if unit is UNITLESS:
                    unit = mantissa.unit
                elif not unit.has_same_dim(mantissa.unit):
                    raise ValueError("Cannot create a Quantity object with a different unit.")
                mantissa = mantissa.in_unit(unit)
                mantissa = mantissa.mantissa

            elif isinstance(mantissa, (np.ndarray, _JaxArray)):
                # Preserve the input backend: NumPy stays NumPy, JAX stays JAX.
                if dtype is not None and mantissa.dtype != dtype:
                    if isinstance(mantissa, _JaxArray):
                        mantissa = jnp.asarray(mantissa, dtype=dtype)
                    else:
                        mantissa = mantissa.astype(dtype)
                # skip if dtype matches or is not provided

            elif isinstance(mantissa, ((jnp.number,) if HAS_JAX else ()) + (numbers.Number,)):
                pass  # keep as-is; jnp.array conversion deferred to use-site

            else:
                # cupy / torch / dask / ndonnx mantissas: preserve the input's
                # native backend, honouring ``dtype`` when requested.
                from saiunit._backend import (
                    get_backend as _gb,
                    is_cupy_array, is_torch_array, is_dask_array,
                    is_ndonnx_array,
                )
                if (
                    is_cupy_array(mantissa)
                    or is_torch_array(mantissa)
                    or is_dask_array(mantissa)
                    or is_ndonnx_array(mantissa)
                ):
                    if dtype is not None and getattr(mantissa, "dtype", None) != dtype:
                        xp = _gb(mantissa)
                        mantissa = xp.asarray(mantissa, dtype=dtype)
                # else: keep as-is for arbitrary pytree leaves

        # mantissa
        self._mantissa = mantissa  # type: ignore[assignment]

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

        ``Quantity.at`` is multi-backend: it works for ``numpy``, ``jax``, ``cupy``,
        ``torch``, and ``dask`` mantissas. The ``ndonnx`` backend cannot represent
        functional in-place updates in its symbolic graph and raises
        :class:`saiunit.BackendError` — call ``.to_numpy()`` first. On ``dask`` the
        update is expressed via ``da.where`` so the task graph stays lazy; only
        slice / scalar-int / 1-D integer / boolean-mask indices are supported on
        dask (multi-dim fancy-integer indexing raises ``NotImplementedError`` —
        use ``.to_numpy()`` for that case).

        Repeated-index semantics differ across backends. When multiple indices
        refer to the same location, JAX applies *all* updates (NumPy in-place
        ``x[idx] += y`` would apply only the last). The summary:

        ============  ========================  =========================
        Backend       ``add``                   ``multiply / divide /
                                                min / max / apply``
        ============  ========================  =========================
        ``jax``       accumulates               accumulates
        ``numpy``     accumulates (np.add.at)   accumulates (np.<op>.at)
        ``cupy``      accumulates               accumulates
        ``torch``     accumulates               last-write-wins
                      (index_put_(accumulate))
        ``dask``      last-write-wins via mask  last-write-wins via mask
        ``ndonnx``    raises BackendError       raises BackendError
        ============  ========================  =========================

        By default, JAX assumes that all indices are in-bounds. Alternative
        out-of-bound semantics can be specified via the ``mode`` parameter
        (see below). On non-JAX backends ``mode`` and ``fill_value`` are
        emulated for scalar-int and 1-D integer-array indices; for slice /
        boolean / ellipsis / tuple indices the kwarg is silently ignored
        because out-of-bounds cannot occur for same-shape sources.
        ``indices_are_sorted`` and ``unique_indices`` are hints only —
        silently ignored outside JAX.

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
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> x = jnp.arange(5.0) * u.mV
            >>> x.at[2].add(10 * u.mV)
            Quantity([ 0.  1. 12.  3.  4.], "mV")
            >>> x.at[2].get()
            Quantity(2., "mV")
        """
        return _IndexUpdateHelper(self)

    @property
    def mantissa(self) -> ArrayLike:
        r"""
        The raw numerical data of this quantity (without the unit).

        In scientific notation :math:`x = a \times 10^{b}`, the *mantissa* is
        the coefficient :math:`a`.  For a ``Quantity``, it is the underlying
        JAX/NumPy array (or Python scalar) that stores the numeric value.

        Returns
        -------
        array_like
            The mantissa array or scalar.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> q = u.Quantity(3.0, unit=u.mV)
            >>> q.mantissa
            3.0

        See Also
        --------
        magnitude : Alias for ``mantissa``.
        unit : The physical unit attached to this quantity.
        """
        return self._mantissa

    @property
    def magnitude(self) -> ArrayLike:
        """
        Alias for :attr:`mantissa`.

        Returns
        -------
        array_like
            The raw numerical data of this quantity.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> q = u.Quantity(5.0, unit=u.metre)
            >>> q.magnitude
            5.0

        See Also
        --------
        mantissa : Primary accessor for the numerical data.
        """
        return self.mantissa

    def update_mantissa(self, mantissa: PyTree) -> None:
        """
        Replace the mantissa in-place, keeping the same unit.

        The new mantissa must have the same shape and dtype as the current one.

        Parameters
        ----------
        mantissa : array_like
            The new numerical data.  Must not be a :class:`Quantity`.

        Raises
        ------
        ValueError
            If *mantissa* is a ``Quantity``, or if shape/dtype do not match.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> q.update_mantissa(jnp.array([4.0, 5.0, 6.0]))
            >>> q
            Quantity([4. 5. 6.], "mV")
        """
        self_value = self.mantissa
        if _is_tracer(self_value):
            raise RuntimeError(
                "update_mantissa() cannot mutate a Quantity whose mantissa is a "
                "JAX tracer (e.g., inside jit/vmap/grad). Construct a new Quantity "
                "instead, e.g. Quantity(new_mantissa, unit=q.unit)."
            )
        if isinstance(mantissa, Quantity):
            raise ValueError("Cannot set the mantissa of a Quantity to another Quantity.")
        if is_numpy_array(self_value):
            # Coerce input into a NumPy array to match self's backend.
            mantissa = np.asarray(mantissa, dtype=self.dtype)
        elif isinstance(mantissa, _JaxArray):
            pass
        else:
            # Coerce ``mantissa`` to whatever backend ``self`` already lives on
            # rather than silently lifting torch / cupy / dask / ndonnx mantissas
            # to JAX.
            from saiunit._backend import get_backend as _gb
            xp = _gb(self_value)
            mantissa = xp.asarray(mantissa, dtype=self.dtype)
        # check
        if mantissa.shape != self_value.shape:
            raise ValueError(f"The shape of the original data is {self_value.shape}, "
                             f"while we got {mantissa.shape}.")
        # Dtype check: use numpy-canonical comparison for numpy-backed Quantity,
        # otherwise apply jax's result-type (which may downcast under x32 mode).
        expected_dtype = self_value.dtype if is_numpy_array(self_value) else _result_type(self_value)
        if mantissa.dtype != expected_dtype:
            raise ValueError(f"The dtype of the original data is {expected_dtype}, "
                             f"while we got {mantissa.dtype}.")
        self._mantissa = mantissa

    @property
    def dim(self) -> Dimension:
        """
        The physical dimension of this quantity (e.g. length, mass, time).

        The dimension is independent of scale (metres vs kilometres both have
        the *length* dimension).

        Returns
        -------
        Dimension
            The physical dimension object.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> q = u.Quantity(5.0, unit=u.metre)
            >>> q.dim
            m

        See Also
        --------
        unit : The full unit (dimension + scale).
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
        The :class:`Unit` attached to this quantity.

        The unit carries both the physical dimension and the scale factor
        (e.g. ``mV`` has dimension ``voltage`` with scale ``1e-3``).

        Returns
        -------
        Unit
            The unit of this quantity.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> q = u.Quantity(5.0, unit=u.mV)
            >>> q.unit
            mV

        See Also
        --------
        dim : The physical dimension without scale information.
        mantissa : The numerical value.
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
        Convert this quantity to a different (compatible) unit.

        The mantissa is rescaled so that the physical value stays the same,
        and the returned ``Quantity`` carries *new_unit*.

        Parameters
        ----------
        new_unit : Unit
            Target unit.  Must have the same dimension as ``self.unit``.

        Returns
        -------
        Quantity
            A new ``Quantity`` expressed in *new_unit*.

        Raises
        ------
        UnitMismatchError
            If *new_unit* has a different dimension.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> q.to(u.volt)
            Quantity([0.001 0.002 0.003], "V")

        See Also
        --------
        in_unit : Identical behaviour (``to`` delegates to ``in_unit``).
        to_decimal : Convert to a plain number in the target unit.
        """
        return self.in_unit(new_unit)

    def to_decimal(self, unit: Unit = UNITLESS) -> ArrayLike:
        """
        Return the numerical value expressed in the given unit, without wrapping
        the result in a ``Quantity``.

        This is useful when you need a plain JAX array for downstream
        computation that does not support units.

        Parameters
        ----------
        unit : Unit, optional
            The reference unit.  Defaults to ``UNITLESS``.

        Returns
        -------
        array_like
            A plain number or JAX array representing the quantity in *unit*.

        Raises
        ------
        TypeError
            If *unit* is not a :class:`Unit`.
        UnitMismatchError
            If *unit* has a different dimension than ``self.unit``.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> q.to_decimal(u.volt)
            Array([0.001, 0.002, 0.003], dtype=float32)

        See Also
        --------
        to : Convert while keeping the ``Quantity`` wrapper.
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

    def in_unit(self, unit: Unit, err_msg: str | None = None) -> 'Quantity':
        """
        Convert this quantity to a compatible unit.

        Behaves identically to :meth:`to`; kept for API compatibility.

        Parameters
        ----------
        unit : Unit
            Target unit.  Must share the same dimension as ``self.unit``.
        err_msg : str, optional
            Custom error message used when the dimensions do not match.

        Returns
        -------
        Quantity
            A new ``Quantity`` expressed in *unit*.

        Raises
        ------
        UnitMismatchError
            If *unit* has a different dimension.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> q.in_unit(u.volt)
            Quantity([0.001 0.002 0.003], "V")
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
        Create a :class:`Quantity` from a raw value and a unit.

        This is a convenience factory that reads more naturally in some
        contexts than the standard constructor.

        Parameters
        ----------
        mantissa : array_like or number
            The numerical value(s).
        unit : Unit
            The physical unit.

        Returns
        -------
        Quantity
            A new ``Quantity`` with the given mantissa and unit.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.Quantity.with_unit(2.0, unit=u.metre)
            Quantity(2., "m")
        """
        return Quantity(mantissa, unit=unit)

    @property
    def is_unitless(self) -> bool:
        """
        ``True`` if this quantity is dimensionless (has no physical unit).

        Returns
        -------
        bool
            Whether the quantity is unitless.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> u.Quantity(5.0).is_unitless
            True
            >>> u.Quantity(5.0, unit=u.mV).is_unitless
            False
        """
        return self.unit.is_unitless

    def has_same_unit(self, other):
        """
        Check whether this quantity shares the same physical dimension as *other*.

        Two quantities that differ only in scale (e.g. ``mV`` vs ``V``) are
        considered to have the same unit dimension.

        Parameters
        ----------
        other : Quantity or Unit
            The object to compare with.

        Returns
        -------
        bool
            ``True`` if both have identical physical dimensions.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> a = u.Quantity(1.0, unit=u.mV)
            >>> b = u.Quantity(2.0, unit=u.volt)
            >>> a.has_same_unit(b)
            True
            >>> c = u.Quantity(1.0, unit=u.second)
            >>> a.has_same_unit(c)
            False
        """
        self_dim = get_dim(self.dim)
        other_dim = get_dim(other.dim)
        return (self_dim is other_dim) or (self_dim == other_dim)

    def _format_value(self, precision: int | None = None) -> str:
        """Format the mantissa value as a string."""
        m = self.mantissa
        # Lazy backends (dask) print their own task-graph summary; never
        # materialize to format. ``repr(m)`` is lazy-safe for dask arrays.
        from saiunit._backend import is_dask_array
        if is_dask_array(m):
            return repr(m)
        value: 'ArrayLike'
        if isinstance(m, (_JaxArray, np.ndarray)):
            value = m
        elif isinstance(m, (numbers.Number, list, tuple)):
            # Python scalars / sequences — promote so we get a printable array.
            # Prefer ``jnp.asarray`` when JAX is installed so the precision
            # matches JAX's x32 default (test fixtures key off this); fall
            # back to NumPy when JAX is unavailable.
            try:
                value = jnp.asarray(m) if HAS_JAX else np.asarray(m)
            except TypeError:
                value = m
        else:
            # cupy / torch / dask / ndonnx arrays: don't lift to JAX (that
            # would silently move the data between backends). Let the outer
            # try/except at the bottom of this function fall back to
            # ``str(value)`` if numpy printing can't handle it.
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
                with np.printoptions(threshold=10, **kw):  # type: ignore[arg-type]
                    return np.array_str(value)  # type: ignore[arg-type]
            return np.array_str(value, precision=precision)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            return str(value)

    def repr_in_unit(
        self,
        precision: int | None = None,
    ) -> str:
        """
        Return a human-readable string of this quantity in its current unit.

        The format is ``"<value> <unit>"``, e.g. ``"3. mV"`` or
        ``"[1. 2. 3.] mV"``.

        Parameters
        ----------
        precision : int, optional
            Number of significant digits.  When *None* the value from
            ``numpy.get_printoptions`` is used.

        Returns
        -------
        str
            The formatted string.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> x = u.Quantity(25.0, unit=u.mV)
            >>> x.repr_in_unit()
            '25. mV'
            >>> x.to(u.volt).repr_in_unit(3)
            '0.025 V'
        """
        s = self._format_value(precision=precision)
        if self.unit.should_display_unit:
            s += f" {str(self.unit)}"
        return s.strip()

    def factorless(self) -> 'Quantity':
        """
        Return an equivalent quantity whose unit has ``factor == 1.0``.

        If the unit already has no extra factor the original object is
        returned unchanged.

        Returns
        -------
        Quantity
            A quantity with the factor folded into the mantissa.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> q = u.Quantity(3.0, unit=u.mV)
            >>> q.factorless()
            Quantity(3., "mV")
        """
        if self.unit.factor != 1.0:
            return Quantity(self.mantissa * self.unit.factor, unit=self.unit.factorless())
        else:
            return self

    @property
    def dtype(self):
        """
        The data type of the mantissa.

        Returns
        -------
        dtype
            The JAX/NumPy dtype of the underlying array.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
            >>> q.dtype
            float32
        """
        a = self.mantissa
        if hasattr(a, 'dtype'):
            return a.dtype
        else:
            if isinstance(a, bool):
                return bool
            elif isinstance(a, int):
                return _canonicalize_dtype(int)
            elif isinstance(a, float):
                return _canonicalize_dtype(float)
            elif isinstance(a, complex):
                return _canonicalize_dtype(complex)
            else:
                raise TypeError(f'Can not get dtype of {a}.')

    @property
    def backend(self) -> str:
        """The backend of the underlying mantissa: one of
        ``'numpy'``, ``'jax'``, ``'cupy'``, ``'torch'``, ``'dask'``, ``'ndonnx'``."""
        from saiunit._backend import (
            is_numpy_array, is_cupy_array, is_torch_array,
            is_dask_array, is_ndonnx_array,
        )
        m = self._mantissa
        if is_numpy_array(m):
            return "numpy"
        if is_cupy_array(m):
            return "cupy"
        if is_torch_array(m):
            return "torch"
        if is_dask_array(m):
            return "dask"
        if is_ndonnx_array(m):
            return "ndonnx"
        return "jax"  # jax is the fallthrough (preserves existing behavior)

    def to_numpy(self) -> 'Quantity':
        """Return a new Quantity with mantissa converted to ``numpy.ndarray``.

        No-op (returns ``self``) if the mantissa is already a NumPy array.
        """
        from saiunit._backend import is_numpy_array, to_backend
        if is_numpy_array(self._mantissa):
            return self
        return Quantity(to_backend(self._mantissa, "numpy"), unit=self.unit)

    def to_jax(self) -> 'Quantity':
        """Return a new Quantity with mantissa converted to ``Array``.

        No-op (returns ``self``) if the mantissa is already a JAX array.
        """
        from saiunit._backend import is_jax_array, to_backend
        if is_jax_array(self._mantissa):
            return self
        return Quantity(to_backend(self._mantissa, "jax"), unit=self.unit)

    def to_cupy(self, *, device=None) -> 'Quantity':
        """Return a new Quantity with mantissa converted to a ``cupy.ndarray``.

        No-op (returns ``self``) if the mantissa is already a CuPy array and no
        ``device`` was specified.
        """
        from saiunit._backend import is_cupy_array, to_backend
        if is_cupy_array(self._mantissa) and device is None:
            return self
        kwargs = {} if device is None else {"device": device}
        return Quantity(to_backend(self._mantissa, "cupy", **kwargs), unit=self.unit)

    def to_torch(self, *, device=None, dtype=None) -> 'Quantity':
        """Return a new Quantity with mantissa converted to a ``torch.Tensor``.

        No-op (returns ``self``) if the mantissa is already a torch tensor and
        no ``device``/``dtype`` was specified. ``dtype`` accepts either a torch
        dtype (e.g. ``torch.float32``) or a numpy dtype (e.g. ``np.float32``).
        """
        from saiunit._backend import is_torch_array, to_backend
        if is_torch_array(self._mantissa) and device is None and dtype is None:
            return self
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return Quantity(to_backend(self._mantissa, "torch", **kwargs), unit=self.unit)

    def to_dask(self, *, chunks='auto') -> 'Quantity':
        """Return a new Quantity with mantissa converted to a ``dask.array.Array``.

        No-op (returns ``self``) if the mantissa is already a dask array and no
        ``chunks`` was specified.
        """
        from saiunit._backend import is_dask_array, to_backend
        if is_dask_array(self._mantissa) and chunks == 'auto':
            return self
        return Quantity(to_backend(self._mantissa, "dask", chunks=chunks), unit=self.unit)

    def to_ndonnx(self) -> 'Quantity':
        """Return a new Quantity with mantissa converted to an ``ndonnx.Array``.

        No-op (returns ``self``) if the mantissa is already an ndonnx array.
        ndonnx arrays are symbolic — operations build an ONNX graph rather than
        eagerly computing.
        """
        from saiunit._backend import is_ndonnx_array, to_backend
        if is_ndonnx_array(self._mantissa):
            return self
        return Quantity(to_backend(self._mantissa, "ndonnx"), unit=self.unit)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        The shape of the mantissa array.

        Returns
        -------
        tuple of int
            Shape tuple, identical to ``jnp.shape(self.mantissa)``.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.mV)
            >>> q.shape
            (2, 2)
        """
        m = self.mantissa
        return tuple(m.shape) if hasattr(m, "shape") else get_backend(self).shape(m)

    @property
    def ndim(self) -> int:
        return self._mantissa.ndim if hasattr(self._mantissa, "ndim") else get_backend(self).ndim(self.mantissa)

    @property
    def imag(self) -> 'Quantity':
        return Quantity(get_backend(self).imag(self.mantissa), unit=self.unit)

    @property
    def real(self) -> 'Quantity':
        return Quantity(get_backend(self).real(self.mantissa), unit=self.unit)

    @property
    def size(self) -> int:
        return self._mantissa.size if hasattr(self._mantissa, "size") else get_backend(self).size(self.mantissa)

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by the mantissa array."""
        m = self._mantissa
        if hasattr(m, "nbytes"):
            return m.nbytes
        _reject_lazy_materialization(m, "Quantity.nbytes")
        return np.asarray(m).nbytes

    @property
    def itemsize(self) -> int:
        """Length (in bytes) of one array element."""
        m = self._mantissa
        if hasattr(m, "itemsize"):
            return m.itemsize
        _reject_lazy_materialization(m, "Quantity.itemsize")
        return np.asarray(m).itemsize

    @property
    def strides(self):
        """Tuple of byte-steps in each dimension (mirrors numpy.ndarray.strides)."""
        m = self._mantissa
        if hasattr(m, "strides"):
            # numpy / cupy expose ``.strides`` natively; for torch this is a
            # method, so we still fall through to the materialise path.
            strides = m.strides
            if not callable(strides):
                return strides
        _reject_lazy_materialization(m, "Quantity.strides")
        return np.asarray(m).strides

    @property
    def flat(self):
        """1-D iterator over the mantissa elements, unit preserved."""
        m = self._mantissa
        if hasattr(m, "flat"):
            flat = m.flat
        else:
            _reject_lazy_materialization(m, "Quantity.flat")
            flat = np.asarray(m).flat
        for v in flat:
            yield Quantity(v, unit=self.unit)

    @property
    def T(self) -> 'Quantity':
        m = self._mantissa
        if hasattr(m, "T"):
            t = m.T
        else:
            _reject_lazy_materialization(m, "Quantity.T")
            t = np.asarray(m).T
        return Quantity(t, unit=self.unit)

    @property
    def mT(self) -> 'Quantity':
        """
        Matrix transpose of the last two dimensions, preserving units.

        The array must be at least 2-D.

        Returns
        -------
        Quantity
            The matrix-transposed quantity.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.mV)
            >>> q.mT.shape
            (2, 2)
        """
        m = self._mantissa
        if hasattr(m, "mT"):
            mt = m.mT
        else:
            _reject_lazy_materialization(m, "Quantity.mT")
            mt = np.asarray(m).mT
        return Quantity(mt, unit=self.unit)

    @property
    def isreal(self) -> Array:
        xp = get_backend(self)
        # array_api_compat.numpy lacks isreal; fall back to imag == 0.
        if hasattr(xp, "isreal"):
            return xp.isreal(self.mantissa)
        return xp.imag(self.mantissa) == 0

    @property
    def isscalar(self) -> bool:
        return self.ndim == 0

    @property
    def isfinite(self) -> Array:
        return get_backend(self).isfinite(self.mantissa)

    @property
    def isinfinite(self) -> Array:
        return get_backend(self).isinf(self.mantissa)

    @property
    def isinf(self) -> Array:
        return get_backend(self).isinf(self.mantissa)

    @property
    def isnan(self) -> Array:
        return get_backend(self).isnan(self.mantissa)

    # ----------------------- #
    # Python inherent methods #
    # ----------------------- #

    # Quantity is unhashable: ``__eq__`` returns an elementwise array, not a
    # bool, so any hash would silently violate the hash/eq invariant and
    # corrupt dict/set lookups. Use ``id(q)`` if a stable identity key is
    # really required.
    __hash__ = None  # type: ignore[assignment]

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
        return Quantity(self.mantissa[index], unit=self.unit)  # type: ignore[index]

    def __setitem__(self, index, value: 'Quantity | ArrayLike'):
        if _is_tracer(self.mantissa):
            raise RuntimeError(
                "Quantity[...] = value cannot mutate a Quantity whose mantissa "
                "is a JAX tracer (e.g., inside jit/vmap/grad). Use the functional "
                "form Quantity(q.mantissa.at[index].set(value.mantissa), unit=q.unit) "
                "instead, or perform the update outside the traced function."
            )
        # check value
        if not isinstance(value, Quantity):
            if self.is_unitless:
                value = Quantity(value)
            else:
                raise TypeError(f"Only Quantity can be assigned to Quantity. But got {value}")
        value = value.in_unit(self.unit)

        # check index
        index = _jtree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))

        # update
        self_value = _scatter(self.mantissa, index, value.mantissa, "set")
        self.update_mantissa(self_value)

    def scatter_add(
        self,
        index: ArrayLike,
        value: 'Quantity | ArrayLike'
    ) -> 'Quantity':
        """
        Return a copy with *value* added at *index*.

        Parameters
        ----------
        index : int or array_like
            Target index (indices).
        value : Quantity
            The value to add.  Must have the same unit dimension.

        Returns
        -------
        Quantity
            A new quantity with the update applied.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> q.scatter_add(0, u.Quantity(10.0, unit=u.mV))
            Quantity([11.  2.  3.], "mV")
        """

        # check value
        if not isinstance(value, Quantity):
            if self.is_unitless:
                value = Quantity(value)
            else:
                raise TypeError(f"Only Quantity can be assigned to Quantity. But got {value}")
        value = value.in_unit(self.unit)

        # check index
        index = _jtree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))

        # scatter-add
        self_value = _scatter(self.mantissa, index, value.mantissa, "add")
        return Quantity(self_value, unit=self.unit)

    def scatter_sub(
        self,
        index: ArrayLike,
        value: 'Quantity | ArrayLike'
    ) -> 'Quantity':
        """
        Return a copy with *value* subtracted at *index*.

        Parameters
        ----------
        index : int or array_like
            Target index (indices).
        value : Quantity
            The value to subtract.  Must have the same unit dimension.

        Returns
        -------
        Quantity
            A new quantity with the update applied.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> q.scatter_sub(0, u.Quantity(1.0, unit=u.mV))
            Quantity([0. 2. 3.], "mV")
        """
        return self.scatter_add(index, -value)  # type: ignore[operator]

    def scatter_mul(
        self,
        index: ArrayLike,
        value: 'Quantity | ArrayLike'
    ) -> 'Quantity':
        """
        Return a copy with the element at *index* multiplied by *value*.

        *value* must be dimensionless (a pure scale factor).

        Parameters
        ----------
        index : int or array_like
            Target index (indices).
        value : Quantity or number
            Dimensionless scale factor.

        Returns
        -------
        Quantity
            A new quantity with the update applied.

        Raises
        ------
        TypeError
            If *value* is not dimensionless.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> q.scatter_mul(0, u.Quantity(10.0))
            Quantity([10.  2.  3.], "mV")
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
        index = _jtree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))

        # scatter-mul
        self_value = _scatter(self.mantissa, index, value.mantissa, "mul")
        return Quantity(self_value, unit=self.unit)

    def scatter_div(
        self,
        index: ArrayLike,
        value: 'Quantity | ArrayLike'
    ) -> 'Quantity':
        """
        Return a copy with the element at *index* divided by *value*.

        *value* must be dimensionless (a pure scale factor).

        Parameters
        ----------
        index : int or array_like
            Target index (indices).
        value : Quantity or number
            Dimensionless scale factor.

        Returns
        -------
        Quantity
            A new quantity with the update applied.

        Raises
        ------
        TypeError
            If *value* is not dimensionless.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> q.scatter_div(0, u.Quantity(2.0))
            Quantity([0.5 2.  3. ], "mV")
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
        index = _jtree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))

        # scatter-div
        self_value = _scatter(self.mantissa, index, value.mantissa, "divide")
        return Quantity(self_value, unit=self.unit)

    def scatter_max(
        self,
        index: ArrayLike,
        value: 'Quantity | ArrayLike'
    ) -> 'Quantity':
        """
        Return a copy where the element at *index* is the maximum of
        the current value and *value*.

        Parameters
        ----------
        index : int or array_like
            Target index (indices).
        value : Quantity
            The comparison value.  Must have the same unit dimension.

        Returns
        -------
        Quantity
            A new quantity with the update applied.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> q.scatter_max(0, u.Quantity(10.0, unit=u.mV))
            Quantity([10.  2.  3.], "mV")
        """

        # check value
        if not isinstance(value, Quantity):
            if self.is_unitless:
                value = Quantity(value)
            else:
                raise TypeError(f"Only Quantity can be assigned to Quantity. But got {value}")
        value = value.in_unit(self.unit)

        # check index
        index = _jtree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))

        # scatter-max
        self_value = _scatter(self.mantissa, index, value.mantissa, "max")
        return Quantity(self_value, unit=self.unit)

    def scatter_min(
        self,
        index: ArrayLike,
        value: 'Quantity | ArrayLike'
    ) -> 'Quantity':
        """
        Return a copy where the element at *index* is the minimum of
        the current value and *value*.

        Parameters
        ----------
        index : int or array_like
            Target index (indices).
        value : Quantity
            The comparison value.  Must have the same unit dimension.

        Returns
        -------
        Quantity
            A new quantity with the update applied.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> q.scatter_min(0, u.Quantity(0.5, unit=u.mV))
            Quantity([0.5 2.  3. ], "mV")
        """

        # check value
        if not isinstance(value, Quantity):
            if self.is_unitless:
                value = Quantity(value)
            else:
                raise TypeError(f"Only Quantity can be assigned to Quantity. But got {value}")
        value = value.in_unit(self.unit)

        # check index
        index = _jtree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))

        # scatter-min
        self_value = _scatter(self.mantissa, index, value.mantissa, "min")
        return Quantity(self_value, unit=self.unit)

    # ---------- #
    # operations #
    # ---------- #

    def __len__(self) -> int:
        return len(self.mantissa)  # type: ignore[arg-type]

    def __neg__(self) -> 'Quantity':
        return Quantity(-self.mantissa, unit=self.unit)  # type: ignore[operator]

    def __pos__(self) -> 'Quantity':
        return Quantity(+self.mantissa, unit=self.unit)  # type: ignore[operator]

    def __abs__(self) -> 'Quantity':
        return Quantity(abs(self.mantissa), unit=self.unit)

    def __invert__(self) -> 'Quantity':
        return Quantity(~self.mantissa, unit=self.unit)  # type: ignore[operator]

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

    def __eq__(self, oc) -> ArrayLike:  # type: ignore[override]
        return self._comparison(oc, "==", operator.eq)

    def __ne__(self, oc) -> ArrayLike:  # type: ignore[override]
        return self._comparison(oc, "!=", operator.ne)

    def __lt__(self, oc) -> ArrayLike:
        return self._comparison(oc, "<", operator.lt)

    def __le__(self, oc) -> ArrayLike:
        return self._comparison(oc, "<=", operator.le)

    def __gt__(self, oc) -> ArrayLike:
        return self._comparison(oc, ">", operator.gt)

    def __ge__(self, oc) -> ArrayLike:
        return self._comparison(oc, ">=", operator.ge)

    def _binary_operation(
        self,
        other,
        value_operation: Callable,
        unit_operation: Callable = lambda a, b: a,
        fail_for_mismatch: bool = False,
        operator_str: str | None = None,
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
            # 0-as-any-dimension: a concrete zero (unitless) is treated as
            # dimensionally compatible with self, matching standard physics
            # convention so e.g. ``0 + 3*ms`` works.
            if (other.unit.is_unitless
                    and not self.unit.is_unitless
                    and _is_concrete_zero(other.mantissa)):
                other = Quantity(other.mantissa, unit=self.unit)
            elif (self.unit.is_unitless
                    and not other.unit.is_unitless
                    and _is_concrete_zero(self.mantissa)):
                # mirror case: ``Quantity(0) + 3*ms``
                self = Quantity(self.mantissa, unit=other.unit)
            else:
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
            if _is_tracer(self.mantissa):
                # Under JIT/vmap/grad, mutation isn't supported; degrade to the
                # functional form so `q += x` still works (Python rebinds the name).
                return r
            self.update_mantissa(r.mantissa)
            return self
        else:
            return r

    @staticmethod
    def _reject_bare_unit(oc, op: str):
        from ._base_unit import Unit
        if isinstance(oc, Unit):
            raise TypeError(
                f"Cannot {op} a Quantity with a bare Unit: addition and "
                "subtraction are defined on quantities, not units. Attach a "
                "mantissa first (e.g. 1*ms + 2*ms)."
            )

    def __add__(self, oc):
        self._reject_bare_unit(oc, "add")
        if isinstance(oc, SparseMatrix):
            return oc.__radd__(self)
        return self._binary_operation(oc, operator.add, fail_for_mismatch=True, operator_str="+")

    def __radd__(self, oc):
        self._reject_bare_unit(oc, "add")
        return self.__add__(oc)

    def __iadd__(self, oc):
        # a += b
        self._reject_bare_unit(oc, "add")
        return self._binary_operation(oc, operator.add, fail_for_mismatch=True, operator_str="+=", inplace=True)

    def __sub__(self, oc):
        self._reject_bare_unit(oc, "subtract")
        if isinstance(oc, SparseMatrix):
            return oc.__rsub__(self)
        return self._binary_operation(oc, operator.sub, fail_for_mismatch=True, operator_str="-")

    def __rsub__(self, oc):
        self._reject_bare_unit(oc, "subtract")
        return _to_quantity(oc).__sub__(self)

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
        # Preserve backend: use mantissa's own ** operator if available.
        m = self.mantissa
        if hasattr(m, "__pow__"):
            powered = m ** oc
        else:
            powered = get_backend(self).asarray(m) ** oc
        r = Quantity(powered, unit=self.unit ** oc)
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

    def __rlshift__(self, oc) -> 'Quantity | ArrayLike':
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

    def __rrshift__(self, oc) -> 'Quantity | ArrayLike':
        # oc >> self
        if not self.is_unitless:
            raise ValueError("The shift amount must be dimensionless")
        return oc >> self.mantissa

    def __irshift__(self, oc) -> 'Quantity':
        # self >>= oc
        r = self.__rshift__(oc)
        self.update_mantissa(r.mantissa)
        return self

    def __round__(self, ndigits: int | None = None) -> 'Quantity':
        """
        Round the mantissa to the given number of decimals.

        :param ndigits: The number of decimals to round to.
        :return: The rounded Quantity.
        """
        return Quantity(round(self.mantissa, ndigits), unit=self.unit)  # type: ignore[arg-type,call-overload]

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

    all = _wrap_function_remove_unit(_xp_attr('all'))
    any = _wrap_function_remove_unit(_xp_attr('any'))
    nonzero = _wrap_function_remove_unit(_xp_attr('nonzero'))
    argmax = _wrap_function_remove_unit(_xp_attr('argmax'))
    argmin = _wrap_function_remove_unit(_xp_attr('argmin'))
    argsort = _wrap_function_remove_unit(_xp_attr('argsort'))

    var = _wrap_function_change_unit(_xp_attr('var'), lambda val, unit: unit ** 2)

    std = _wrap_function_keep_unit(_xp_attr('std'))
    sum = _wrap_function_keep_unit(_xp_attr('sum'))
    trace = _wrap_function_keep_unit(_xp_attr('trace'))
    cumsum = _wrap_function_keep_unit(_xp_attr('cumsum'))
    diagonal = _wrap_function_keep_unit(_xp_attr('diagonal'))
    max = _wrap_function_keep_unit(_xp_attr('max'))
    mean = _wrap_function_keep_unit(_xp_attr('mean'))
    min = _wrap_function_keep_unit(_xp_attr('min'))
    ptp = _wrap_function_keep_unit(_xp_attr('ptp'))
    ravel = _wrap_function_keep_unit(_xp_attr('ravel'))

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
        Evenly round the mantissa to the given number of decimals.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places (default ``0``).  Negative values
            round to positions left of the decimal point.

        Returns
        -------
        Quantity
            A new quantity with the rounded mantissa.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> q = u.Quantity(1.567, unit=u.mV)
            >>> q.round(1)
            Quantity(1.6, "mV")
        """
        return Quantity(get_backend(self).round(self.mantissa, decimals), unit=self.unit)

    def astype(
        self,
        dtype: DTypeLike
    ) -> 'Quantity':
        """
        Return a copy of this quantity with the mantissa cast to *dtype*.

        Parameters
        ----------
        dtype : str or dtype
            Target data type (e.g. ``jnp.float64``).

        Returns
        -------
        Quantity
            A new quantity with the converted dtype.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
            >>> q.astype(jnp.float64).dtype
            float64
        """
        if dtype is None:
            return Quantity(self.mantissa, unit=self.unit)
        else:
            return Quantity(get_backend(self).astype(self.mantissa, dtype), unit=self.unit)

    def clip(
        self,
        min: 'Quantity | ArrayLike | None' = None,
        max: 'Quantity | ArrayLike | None' = None,
    ) -> 'Quantity':
        """
        Clip (limit) the values in the array to ``[min, max]``.

        At least one of *min* or *max* must be given.  Both must be
        compatible with the unit of ``self``.

        Parameters
        ----------
        min : Quantity or array_like, optional
            Minimum value.
        max : Quantity or array_like, optional
            Maximum value.

        Returns
        -------
        Quantity
            The clipped quantity.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> q.clip(min=u.Quantity(1.5, unit=u.mV), max=u.Quantity(2.5, unit=u.mV))
            Quantity([1.5 2.  2.5], "mV")
        """
        _, min = unit_scale_align_to_first(self, min)
        _, max = unit_scale_align_to_first(self, max)
        return Quantity(get_backend(self, min, max).clip(self.mantissa, min.mantissa, max.mantissa), unit=self.unit)  # type: ignore[union-attr]

    def conj(self) -> 'Quantity':
        """
        Return the complex conjugate, element-wise, preserving units.

        Returns
        -------
        Quantity
            The conjugated quantity.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> q = u.Quantity(1.0 + 2.0j, unit=u.mV)
            >>> q.conj()
            Quantity((1-2j), "mV")
        """
        return Quantity(get_backend(self).conj(self.mantissa), unit=self.unit)

    def conjugate(self) -> 'Quantity':
        """
        Return the complex conjugate, element-wise.

        Alias for :meth:`conj`.

        Returns
        -------
        Quantity
            The conjugated quantity.
        """
        return Quantity(get_backend(self).conj(self.mantissa), unit=self.unit)

    def copy(self) -> 'Quantity':
        """
        Return a deep copy of this quantity.

        Returns
        -------
        Quantity
            An independent copy with the same mantissa and unit.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> q = u.Quantity(3.0, unit=u.mV)
            >>> q2 = q.copy()
            >>> q2
            Quantity(3., "mV")
        """
        return type(self)(get_backend(self).copy(self.mantissa), unit=self.unit)

    def dot(self, b) -> 'Quantity':
        """
        Dot product of two arrays.

        The resulting unit is ``self.unit * b.unit``.

        Parameters
        ----------
        b : Quantity or array_like
            Second operand.

        Returns
        -------
        Quantity
            The dot product.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> a = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> b = u.Quantity(jnp.array([1.0, 1.0, 1.0]), unit=u.mV)
            >>> a.dot(b)
            Quantity(6., "mV^2")
        """
        xp = get_backend(self, b)
        r = self._binary_operation(b, xp.dot if hasattr(xp, "dot") else jnp.dot,
                                   operator.mul, operator_str="@")
        return maybe_decimal(r)

    def trace(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> 'Quantity':  # type: ignore[no-redef]
        """
        Sum along diagonals of the array, preserving units.

        Parameters
        ----------
        offset : int, optional
            Offset of the diagonal from the main diagonal (default ``0``).
        axis1 : int, optional
            First axis of the 2-D sub-arrays (default ``0``).
        axis2 : int, optional
            Second axis of the 2-D sub-arrays (default ``1``).

        Returns
        -------
        Quantity
            The trace value(s).

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.eye(3), unit=u.mV)
            >>> q.trace()
            Quantity(3., "mV")
        """
        return Quantity(get_backend(self).trace(self.mantissa, offset=offset, axis1=axis1, axis2=axis2), unit=self.unit)

    def diagonal(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> 'Quantity':  # type: ignore[no-redef]
        """
        Return specified diagonals, preserving units.

        Parameters
        ----------
        offset : int, optional
            Offset from the main diagonal (default ``0``).
        axis1 : int, optional
            First axis (default ``0``).
        axis2 : int, optional
            Second axis (default ``1``).

        Returns
        -------
        Quantity
            The diagonal elements.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.mV)
            >>> q.diagonal()
            Quantity([1. 4.], "mV")
        """
        return Quantity(get_backend(self).diagonal(self.mantissa, offset=offset, axis1=axis1, axis2=axis2), unit=self.unit)

    def outer(self, b: 'Quantity') -> 'Quantity':
        """
        Outer product of two 1-D arrays.

        The resulting unit is ``self.unit * b.unit``.

        Parameters
        ----------
        b : Quantity or array_like
            Second operand.

        Returns
        -------
        Quantity
            The outer product matrix.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> a = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
            >>> b = u.Quantity(jnp.array([3.0, 4.0]), unit=u.second)
            >>> a.outer(b).shape
            (2, 2)
        """
        b = _to_quantity(b)
        xp = get_backend(self, b)
        r = self._binary_operation(b, xp.outer if hasattr(xp, "outer") else jnp.outer,
                                   operator.mul, operator_str="outer")
        return maybe_decimal(r)

    def cross(self, b: 'Quantity', axisa: int = -1, axisb: int = -1, axisc: int = -1, axis: int | None = None) -> 'Quantity':
        """
        Cross product of two arrays.

        The resulting unit is ``self.unit * b.unit``.

        Parameters
        ----------
        b : Quantity
            Second operand.
        axisa : int, optional
            Axis of *self* that defines the vector(s) (default ``-1``).
        axisb : int, optional
            Axis of *b* that defines the vector(s) (default ``-1``).
        axisc : int, optional
            Axis of the result containing the cross product (default ``-1``).
        axis : int, optional
            Overrides *axisa*, *axisb*, and *axisc* simultaneously.

        Returns
        -------
        Quantity
            The cross product.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> a = u.Quantity(jnp.array([1.0, 0.0, 0.0]), unit=u.mV)
            >>> b = u.Quantity(jnp.array([0.0, 1.0, 0.0]), unit=u.second)
            >>> a.cross(b)
            Quantity([0. 0. 1.], "mV * s")
        """
        b = _to_quantity(b)
        kwargs = dict(axisa=axisa, axisb=axisb, axisc=axisc)
        if axis is not None:
            kwargs['axis'] = axis
        result_mantissa = get_backend(self, b).cross(self.mantissa, b.mantissa, **kwargs)
        result_unit = self.unit * b.unit
        r = Quantity(result_mantissa, unit=result_unit)  # type: ignore[arg-type]
        return maybe_decimal(r)

    def searchsorted(self, v, side: str = 'left', sorter=None) -> Array:
        """Find indices where elements should be inserted to maintain order."""
        if isinstance(v, Quantity):
            v = v.in_unit(self.unit).mantissa
        return get_backend(self, v).searchsorted(self.mantissa, v, side=side, sorter=sorter)

    def fill(self, value: 'Quantity') -> 'Quantity':
        """Fill the array with a scalar mantissa."""
        fail_for_dimension_mismatch(self, value, "fill")
        self[:] = value
        return self

    def flatten(self) -> 'Quantity':
        """
        Return a 1-D copy of this quantity.

        Returns
        -------
        Quantity
            Flattened quantity with the same unit.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.mV)
            >>> q.flatten()
            Quantity([1. 2. 3. 4.], "mV")
        """
        return Quantity(get_backend(self).reshape(self.mantissa, (-1,)), unit=self.unit)

    def item(self, *args) -> 'Quantity':
        """
        Extract a single element as a scalar ``Quantity``.

        Parameters
        ----------
        *args : int
            Index into the flat array.

        Returns
        -------
        Quantity
            A 0-D ``Quantity`` containing the selected element.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([10.0, 20.0]), unit=u.mV)
            >>> q.item(0)
            Quantity(10., "mV")
        """
        return Quantity(self.mantissa.item(*args), unit=self.unit)

    def prod(self, *args, **kwds) -> 'Quantity':
        """
        Return the product of array elements over the given axis.

        The unit of the result is ``self.unit ** n`` where *n* is the number
        of elements multiplied together.

        Returns
        -------
        Quantity
            The product.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([2.0, 3.0]), unit=u.mV)
            >>> q.prod()
            Quantity(6., "mV^2")
        """
        self = self.factorless()

        xp = get_backend(self)
        prod_res = xp.prod(self.mantissa, *args, **kwds)
        # The unit exponent is the number of elements multiplied along the
        # reduction axis. Derive it from the static shape so the computation
        # is JIT-safe (no `bool(traced_array)` or traced unit exponents).
        axis = args[0] if args else kwds.get('axis', None)
        dim_exponent = _reduction_count_from_shape(self.mantissa.shape, axis)
        r = Quantity(prod_res, unit=self.unit ** dim_exponent)
        return maybe_decimal(r)

    def nanprod(self, *args, **kwds) -> 'Quantity':
        """
        Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones.

        When reducing along a specific axis, the number of non-NaN elements
        must be the same for every position in the result so that a single
        unit exponent can be assigned.  If the non-NaN counts differ and the
        quantity is not dimensionless, a ``ValueError`` is raised.

        Returns
        -------
        Quantity
            The product (NaNs treated as ones).

        Raises
        ------
        ValueError
            If the non-NaN counts are not uniform along the reduction axis
            for a non-dimensionless quantity.
        """
        self = self.factorless()

        xp = get_backend(self)
        if not hasattr(xp, "nanprod"):
            backend = getattr(xp, "__name__", repr(xp))
            raise AttributeError(
                f"saiunit: backend {backend!r} has no operation 'nanprod'"
            )
        prod_res = xp.nanprod(self.mantissa, *args, **kwds)

        if self.is_unitless:
            return maybe_decimal(Quantity(prod_res, unit=self.unit))

        # Unit exponent is the count of elements along the reduction axis.
        # Use the static shape so the result is JIT-safe.
        axis = args[0] if args else kwds.get('axis', None)
        dim_exponent = _reduction_count_from_shape(self.mantissa.shape, axis)

        # Eager (non-traced) inputs: verify NaN counts are uniform along the
        # reduction axis. Otherwise different output elements would carry
        # different unit exponents, which is unrepresentable. Skipped under
        # tracing because traced booleans cannot drive a Python ``if``.
        if not _is_tracer(self.mantissa):
            nan_mask = xp.isnan(self.mantissa)
            non_nan_counts = xp.sum(xp.where(nan_mask, 0, 1), *args, **kwds)
            if non_nan_counts.ndim > 0:
                first = non_nan_counts.ravel()[0]
                if not bool(xp.all(non_nan_counts == first)):
                    raise ValueError(
                        "nanprod over an axis with non-uniform NaN counts is not "
                        "supported for quantities with units, because the resulting "
                        "elements would have different unit exponents."
                    )
                dim_exponent = int(first)
            else:
                dim_exponent = int(non_nan_counts)

        r = Quantity(prod_res, unit=self.unit ** dim_exponent)
        return maybe_decimal(r)

    def cumprod(self, *args, **kwds):
        """
        Return the cumulative product of elements along a given axis.

        Because each position in the result corresponds to a different number
        of multiplied elements, the unit exponent varies across the output.
        This is only representable when the quantity is dimensionless.

        Returns
        -------
        Quantity
            The cumulative product.

        Raises
        ------
        TypeError
            If the quantity is not dimensionless.
        """
        if not self.is_unitless:
            raise TypeError(
                "cumprod is not supported for quantities with units "
                f"(has unit {self.unit}), because each element of the result "
                "would have a different unit exponent. "
                "Use .prod() for a single reduction, or convert to "
                "dimensionless first."
            )
        return maybe_decimal(
            Quantity(get_backend(self).cumprod(self.mantissa, *args, **kwds), unit=self.unit)
        )

    def nancumprod(self, *args, **kwds):
        """
        Return the cumulative product of elements along a given axis,
        treating NaNs as ones.

        Because each position in the result corresponds to a different number
        of multiplied elements, the unit exponent varies across the output.
        This is only representable when the quantity is dimensionless.

        Returns
        -------
        Quantity
            The cumulative product (NaNs treated as ones).

        Raises
        ------
        TypeError
            If the quantity is not dimensionless.
        """
        if not self.is_unitless:
            raise TypeError(
                "nancumprod is not supported for quantities with units "
                f"(has unit {self.unit}), because each element of the result "
                "would have a different unit exponent. "
                "Use .nanprod() for a single reduction, or convert to "
                "dimensionless first."
            )
        return maybe_decimal(
            Quantity(get_backend(self).nancumprod(self.mantissa, *args, **kwds), unit=self.unit)
        )

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
        """
        Repeat elements of the array.

        Parameters
        ----------
        repeats : int or array of ints
            Number of repetitions for each element.
        axis : int, optional
            Axis along which to repeat.

        Returns
        -------
        Quantity
            The repeated quantity.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
            >>> q.repeat(2)
            Quantity([1. 1. 2. 2.], "mV")
        """
        r = get_backend(self).repeat(self.mantissa, repeats=repeats, axis=axis)
        return Quantity(r, unit=self.unit)

    def reshape(self, shape, order='C') -> 'Quantity':
        """
        Return a quantity with the same data but a new shape.

        Parameters
        ----------
        shape : int or tuple of ints
            New shape.
        order : {'C', 'F'}, optional
            Memory layout order (default ``'C'``).

        Returns
        -------
        Quantity
            Reshaped quantity.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> q.reshape((3, 1)).shape
            (3, 1)
        """
        xp = get_backend(self)
        try:
            return Quantity(xp.reshape(self.mantissa, shape, order=order), unit=self.unit)
        except TypeError:
            # array_api_compat.numpy.reshape may not accept order=; fall back.
            return Quantity(xp.reshape(self.mantissa, shape), unit=self.unit)

    def resize(self, new_shape) -> 'Quantity':
        """Change shape and size of array in-place."""
        # ``resize`` is not in the array-API spec; both numpy and jax expose it.
        xp = get_backend(self)
        resize_fn = getattr(xp, "resize", None) or np.resize
        self.update_mantissa(resize_fn(self.mantissa, new_shape))
        return self

    def sort(self, axis=-1, stable=True, order=None) -> 'Quantity':
        """
        Sort the array in-place along the given axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort (default ``-1``).
        stable : bool, optional
            Whether to use a stable sort (default ``True``).
        order : str or list of str, optional
            Field ordering for structured arrays.

        Returns
        -------
        Quantity
            ``self``, with the mantissa sorted in-place.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([3.0, 1.0, 2.0]), unit=u.mV)
            >>> q.sort()
            Quantity([1. 2. 3.], "mV")
        """
        xp = get_backend(self)
        # array-API sort has different kw names; try the standard first, fall back to numpy/jnp.
        try:
            sorted_arr = xp.sort(self.mantissa, axis=axis, stable=stable)
        except TypeError:
            sorted_arr = xp.sort(self.mantissa, axis=axis)
        self.update_mantissa(sorted_arr)
        return self

    def squeeze(self, axis=None) -> 'Quantity':
        """
        Remove length-one axes from the array.

        Parameters
        ----------
        axis : int or tuple of ints, optional
            Axes to remove.  If ``None``, all length-one axes are removed.

        Returns
        -------
        Quantity
            The squeezed quantity.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([[[1.0]]]), unit=u.mV)
            >>> q.squeeze().shape
            ()
        """
        return Quantity(get_backend(self).squeeze(self.mantissa, axis=axis), unit=self.unit)

    def swapaxes(self, axis1, axis2) -> 'Quantity':
        """
        Interchange two axes of the array.

        Parameters
        ----------
        axis1 : int
            First axis.
        axis2 : int
            Second axis.

        Returns
        -------
        Quantity
            The quantity with axes swapped.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.mV)
            >>> q.swapaxes(0, 1).shape
            (2, 2)
        """
        return Quantity(get_backend(self).swapaxes(self.mantissa, axis1, axis2), unit=self.unit)

    def split(self, indices_or_sections, axis=0) -> 'list[Quantity]':
        """
        Split the array into multiple sub-arrays.

        Parameters
        ----------
        indices_or_sections : int or 1-D array
            If an integer *N*, the array is divided into *N* equal parts.
            If a sorted 1-D array of indices, the entries indicate split
            points along *axis*.
        axis : int, optional
            Axis along which to split (default ``0``).

        Returns
        -------
        list of Quantity
            Sub-arrays, each carrying the same unit.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0, 3.0]), unit=u.mV)
            >>> parts = q.split(3)
            >>> len(parts)
            3
        """
        # ``split`` is part of array-API standard (with ``indices_or_sections``).
        xp = get_backend(self)
        split_fn = getattr(xp, "split", None) or np.split
        return [Quantity(a, unit=self.unit) for a in split_fn(self.mantissa, indices_or_sections, axis=axis)]

    def take(
        self,
        indices,
        axis=None,
        mode=None,
        unique_indices=False,
        indices_are_sorted=False,
        fill_value=None,
    ) -> 'Quantity':
        """
        Select elements from the array at the given indices.

        Parameters
        ----------
        indices : array_like
            Indices of the values to extract.
        axis : int, optional
            Axis along which to take (default flattened).
        mode : str, optional
            Out-of-bounds index handling.
        unique_indices : bool, optional
            Hint that indices are unique.
        indices_are_sorted : bool, optional
            Hint that indices are sorted.
        fill_value : Quantity or scalar, optional
            Value for out-of-bounds positions when *mode* is ``'fill'``.

        Returns
        -------
        Quantity
            The selected elements.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([10.0, 20.0, 30.0]), unit=u.mV)
            >>> q.take(jnp.array([0, 2]))
            Quantity([10. 30.], "mV")
        """

        if isinstance(fill_value, Quantity):
            fail_for_dimension_mismatch(self, fill_value, "take")
            fill_value = unit_scale_align_to_first(self, fill_value)[1].mantissa
        elif fill_value is not None:
            if not self.is_unitless:
                raise TypeError(f"fill_value must be a Quantity when the unit {self.unit}. But got {fill_value}")
        if is_numpy_array(self._mantissa):
            # NumPy ``take`` has a different signature; emulate the saiunit semantics.
            taken = np.take(self.mantissa, indices, axis=axis, mode=mode if mode else 'raise')
            return Quantity(taken, unit=self.unit)
        if is_jax_array(self._mantissa):
            return Quantity(
                jnp.take(
                    self.mantissa,
                    indices=indices,
                    axis=axis,
                    mode=mode,
                    unique_indices=unique_indices,
                    indices_are_sorted=indices_are_sorted,
                    fill_value=fill_value,
                ),
                unit=self.unit,
            )
        # Other backends: dispatch via the array-API ``take`` op.
        xp = get_backend(self)
        return Quantity(xp.take(self.mantissa, indices, axis=axis), unit=self.unit)

    def tolist(self):
        """
        Convert the array to a (nested) Python list of ``Quantity`` scalars.

        Each leaf element is a 0-D ``Quantity`` with the same unit.

        Returns
        -------
        list or Quantity
            A nested list of scalar ``Quantity`` objects, or a single
            ``Quantity`` for 0-D arrays.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
            >>> q.tolist()
            [Quantity(1., "mV"), Quantity(2., "mV")]
        """
        _dask_materialization_guard(self._mantissa, "Quantity.tolist()")
        if isinstance(self.mantissa, numbers.Number):
            list_mantissa = self.mantissa
        else:
            list_mantissa = self.mantissa.tolist()
        return _replace_with_array(list_mantissa, self.unit)

    def transpose(self, *axes) -> 'Quantity':
        """
        Return the array with axes transposed.

        For a 2-D array this is the standard matrix transpose.

        Parameters
        ----------
        *axes : None, tuple of ints, or n ints
            If omitted, axes are reversed.  Otherwise specifies the
            permutation.

        Returns
        -------
        Quantity
            Transposed quantity.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=u.mV)
            >>> q.transpose().shape
            (2, 2)
        """
        xp = get_backend(self)
        if not axes:
            return Quantity(xp.transpose(self.mantissa), unit=self.unit)
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        elif len(axes) == 1 and axes[0] is None:
            return Quantity(xp.transpose(self.mantissa), unit=self.unit)
        return Quantity(xp.transpose(self.mantissa, axes), unit=self.unit)

    def tile(self, reps) -> 'Quantity':
        """
        Construct an array by repeating this quantity.

        Parameters
        ----------
        reps : int or array_like
            Number of repetitions along each axis.

        Returns
        -------
        Quantity
            The tiled quantity.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
            >>> q.tile(2)
            Quantity([1. 2. 1. 2.], "mV")
        """
        return Quantity(get_backend(self).tile(self.mantissa, reps), unit=self.unit)

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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Intercept numpy ufunc calls so units are preserved or checked.

        For ufuncs in :data:`_UFUNC_DISPATCH` (``np.add``, ``np.sin``, …), route
        through the matching ``saiunit.math`` function. For anything else,
        return ``NotImplemented`` so NumPy raises a ``TypeError`` rather than
        silently stripping units.

        When a binary ufunc mixes a Quantity with a plain scalar/array, delegate
        to the Quantity's own arithmetic dunder so unit-mismatch errors keep
        their normal type (``UnitMismatchError``) rather than degrading to a
        generic ``TypeError`` from the underlying saiunit.math function.
        """
        if method != "__call__":
            # reduce / accumulate / outer / at / reduceat are not supported.
            return NotImplemented

        global _UFUNC_DISPATCH
        if not _UFUNC_DISPATCH:
            _UFUNC_DISPATCH = _build_ufunc_dispatch()

        saiunit_fn = _UFUNC_DISPATCH.get(ufunc)
        if saiunit_fn is None:
            return NotImplemented

        if kwargs.pop("out", None) is not None:
            # ``out=`` writes into a pre-allocated buffer; not supported.
            return NotImplemented

        # For binary ufuncs mixing a Quantity with a plain scalar/array,
        # route through Quantity's own dunder so the operator semantics
        # (including UnitMismatchError on a unit mismatch) are preserved.
        if len(inputs) == 2 and ufunc in _BINARY_UFUNC_OPNAMES and not kwargs:
            lhs, rhs = inputs
            lhs_is_q = isinstance(lhs, Quantity)
            rhs_is_q = isinstance(rhs, Quantity)
            if lhs_is_q != rhs_is_q:  # exactly one is a Quantity
                forward, reverse = _BINARY_UFUNC_OPNAMES[ufunc]
                if lhs_is_q:
                    result = getattr(lhs, forward)(rhs)
                else:
                    result = getattr(rhs, reverse)(lhs)
                if result is NotImplemented:
                    return saiunit_fn(*inputs, **kwargs)
                return result

        return saiunit_fn(*inputs, **kwargs)

    def __array__(self, dtype: DTypeLike | None = None) -> np.ndarray:
        """Support ``numpy.array()`` and ``numpy.asarray()`` functions.

        Only dimensionless quantities are coercible — converting a unit-bearing
        Quantity would silently drop the unit and produce a numerically
        misleading array, so we raise ``TypeError`` instead.
        """
        _dask_materialization_guard(self._mantissa, "np.asarray(Quantity)")
        if self.dim.is_dimensionless:
            return np.asarray(self.to_decimal(), dtype=dtype)
        else:
            raise TypeError(
                f"Only dimensionless quantities can be "
                f"converted to NumPy arrays. But got {self}"
            )

    def __float__(self):
        _dask_materialization_guard(self._mantissa, "float(Quantity)")
        if self.dim.is_dimensionless and self.ndim == 0:
            return float(self.to_decimal())
        else:
            raise TypeError(
                "Only dimensionless scalar quantities can be "
                f"converted to Python scalars. But got {self}"
            )

    def __int__(self):
        _dask_materialization_guard(self._mantissa, "int(Quantity)")
        if self.dim.is_dimensionless and self.ndim == 0:
            return int(self.to_decimal())
        else:
            raise TypeError(
                "only dimensionless scalar quantities can be "
                f"converted to Python scalars. But got {self}"
            )

    def __index__(self):
        _dask_materialization_guard(self._mantissa, "operator.index(Quantity)")
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
        Insert a length-one axis (PyTorch-style alias for :meth:`expand_dims`).

        Parameters
        ----------
        axis : int
            Position where the new axis is inserted.

        Returns
        -------
        Quantity
            The quantity with an extra dimension.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
            >>> q.unsqueeze(0).shape
            (1, 2)
        """
        return Quantity(get_backend(self).expand_dims(self.mantissa, axis), unit=self.unit)

    def expand_dims(self, axis: int | Sequence[int]) -> 'Quantity':
        """
        Insert new axes at the given positions.

        Parameters
        ----------
        axis : int or tuple of ints
            Position(s) where the new axis (axes) are placed.

        Returns
        -------
        Quantity
            The expanded quantity.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> import jax.numpy as jnp
            >>> q = u.Quantity(jnp.array([1.0, 2.0]), unit=u.mV)
            >>> q.expand_dims(0).shape
            (1, 2)
        """
        return Quantity(get_backend(self).expand_dims(self.mantissa, axis), unit=self.unit)

    def expand_as(self, array: 'Quantity | ArrayLike') -> 'Quantity':
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
        return Quantity(get_backend(self).broadcast_to(self.mantissa, array), unit=self.unit)

    def pow(self, oc) -> 'Quantity':
        """
        Raise this quantity to the power *oc*.

        The exponent must be dimensionless.  The resulting unit is
        ``self.unit ** oc``.

        Parameters
        ----------
        oc : int, float, or dimensionless Quantity
            The exponent.

        Returns
        -------
        Quantity
            ``self ** oc``.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> q = u.Quantity(2.0, unit=u.mV)
            >>> q.pow(2)
            Quantity(4., "mV^2")
        """
        return self.__pow__(oc)

    def clone(self) -> 'Quantity':
        """
        Return a copy of this quantity (PyTorch-style alias for :meth:`copy`).

        Returns
        -------
        Quantity
            An independent copy.

        Examples
        --------
        .. code-block:: python

            >>> import saiunit as u
            >>> q = u.Quantity(3.0, unit=u.mV)
            >>> q.clone()
            Quantity(3., "mV")
        """
        return self.copy()

    def tree_flatten(self) -> tuple[tuple[ArrayLike], Unit]:
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
        return cls(*values, unit=unit)  # type: ignore[misc]

    def cuda(self, device=None) -> 'Quantity':
        from saiunit._jax_compat import device_put as _device_put, devices as _devices
        device = _devices('cuda')[0] if device is None else device
        self.update_mantissa(_device_put(self.mantissa, device))
        return self

    def cpu(self, device=None) -> 'Quantity':
        from saiunit._jax_compat import device_put as _device_put, devices as _devices
        device = _devices('cpu')[0] if device is None else device
        self.update_mantissa(_device_put(self.mantissa, device))
        return self

    # dtype exchanging #
    # ---------------- #
    def half(self) -> 'Quantity':
        xp = get_backend(self)
        return Quantity(xp.asarray(self.mantissa, dtype=xp.float16), unit=self.unit)

    def float(self) -> 'Quantity':
        xp = get_backend(self)
        return Quantity(xp.asarray(self.mantissa, dtype=xp.float32), unit=self.unit)

    def double(self) -> 'Quantity':
        xp = get_backend(self)
        return Quantity(xp.asarray(self.mantissa, dtype=xp.float64), unit=self.unit)


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

    This object references a source array and a specific indexer into that
    array. Methods on this object return copies of the source array that have
    been modified at the positions specified by the indexer. Updates are
    dispatched through :func:`saiunit._scatter.scatter`, which supports every
    backend saiunit ships with: ``numpy``, ``jax``, ``cupy``, ``torch``,
    ``dask``, and ``ndonnx`` (ndonnx raises a :class:`BackendError` because
    its symbolic graph cannot represent in-place updates — call
    ``.to_numpy()`` first).
    """
    __slots__ = ("quantity", "index", "unit")

    def __init__(self, index, quantity: Quantity):
        self.index = _jtree.map(_element_not_quantity, index, is_leaf=lambda x: isinstance(x, Quantity))
        self.quantity = quantity
        self.unit = quantity.unit

    def __repr__(self) -> str:
        return f"_IndexUpdateRef({self.quantity}, {self.index!r})"

    def _scatter(self, op: str, value, *, indices_are_sorted, unique_indices,
                 mode, fill_value=None):
        """Internal: dispatch ``op`` on the underlying mantissa."""
        from saiunit._scatter import scatter as _scatter_dispatch
        return _scatter_dispatch(
            self.quantity.mantissa,
            self.index,
            value,
            op,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
            fill_value=fill_value,
        )

    def get(
        self,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | None = None,
        fill_value: StaticScalar | Quantity | None = None
    ) -> Quantity:
        """Equivalent to ``x[idx]``.

        Returns the value of ``x`` that would result from the NumPy-style
        :mod:indexing <numpy.doc.indexing>` ``x[idx]``. This function differs from
        the usual array indexing syntax in that it allows additional keyword
        arguments ``indices_are_sorted`` and ``unique_indices`` to be passed.
        """
        if fill_value is not None:
            fill_value = Quantity(fill_value).in_unit(self.unit).mantissa  # type: ignore[assignment]
        return Quantity(
            self._scatter(
                "get",
                None,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode,
                fill_value=fill_value,
            ),
            unit=self.unit,
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
            self._scatter(
                "set", values,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode,
            ),
            unit=self.unit,
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
            self._scatter(
                "add", values,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode,
            ),
            unit=self.unit,
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
            self._scatter(
                "multiply", values.mantissa,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode,
            ),
            unit=self.unit * values.unit,
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
            self._scatter(
                "divide", values.mantissa,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode,
            ),
            unit=self.unit / values.unit,
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
            self._scatter(
                "power", values,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode,
            ),
            unit=self.unit ** values,
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
            self._scatter(
                "min", values,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode,
            ),
            unit=self.unit,
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
            self._scatter(
                "max", values,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode,
            ),
            unit=self.unit,
        )

    def apply(
        self,
        mantissa_fun: Callable[[ArrayLike], ArrayLike],
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
            self._scatter(
                "apply", mantissa_fun,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode,
            ),
            unit=result_unit,
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
