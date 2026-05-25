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
from typing import (Union, Optional, List, Any, Tuple)

from saiunit._jax_compat import jax, jnp, tree as _tree
from saiunit._typing import Array, ArrayLike, DTypeLike
import numpy as np

from saiunit._backend import get_backend, get_default_backend, _xp_for, _translate_dtype
from saiunit._base_dimension import UnitMismatchError
from saiunit._base_unit import UNITLESS, Unit
from saiunit._base_getters import fail_for_unit_mismatch, get_unit, unit_scale_align_to_first
from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as, maybe_custom_array_tree, maybe_custom_array

import array_api_compat.numpy as _numpy_xp


def _safe_call_xp(fn, args, kwargs):
    """Local mirror of :func:`_fun_keep_unit._dispatch_call` for direct ``xp.fn(...)``
    call sites in this module. Imported lazily to avoid a circular import."""
    from ._fun_keep_unit import _dispatch_call
    return _dispatch_call(fn, args, kwargs)


def _default_xp():
    """Return the backend namespace selected by the current default backend.

    When no default is set, prefer JAX if installed (preserves legacy
    behaviour where ``jnp`` was the fallback), otherwise NumPy. If the
    configured default backend isn't importable — e.g. CI runs that test
    individual backends in isolation without JAX — fall back to NumPy.
    """
    name = get_default_backend()
    if name is None:
        return jnp if jnp is not None else _numpy_xp
    try:
        return _xp_for(name)
    except Exception:
        return _numpy_xp

Shape = Union[int, Sequence[int]]

__all__ = [
    # array creation(given shape)
    'full', 'eye', 'identity', 'tri',
    'empty', 'ones', 'zeros',

    # array creation(given array)
    'full_like', 'diag', 'tril', 'triu',
    'empty_like', 'ones_like', 'zeros_like', 'fill_diagonal',

    # array creation(misc)
    'array', 'asarray', 'arange', 'linspace', 'logspace',
    'meshgrid', 'vander',

    # indexing funcs
    'tril_indices', 'tril_indices_from', 'triu_indices',
    'triu_indices_from',

    # others
    'from_numpy',
    'as_numpy',
    'tree_ones_like',
    'tree_zeros_like',
]


@set_module_as('saiunit.math')
def full(
    shape: Shape,
    fill_value: Union[Quantity, int, float],
    dtype: Optional[DTypeLike] = None,
) -> Union[Array, Quantity]:
    """
    Return a new quantity or array of given shape, filled with ``fill_value``.

    If ``fill_value`` is a :class:`~saiunit.Quantity`, the result is a
    ``Quantity`` carrying the same unit.  Otherwise a plain JAX array is
    returned.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar, array_like, or Quantity
        Fill value.  When a ``Quantity`` is given its unit is preserved.
    dtype : data-type, optional
        The desired data-type for the array.  The default, ``None``, means
        the dtype is inferred from ``fill_value``.

    Returns
    -------
    out : Quantity or Array
        Array (or ``Quantity``) of ``fill_value`` with the given shape and
        dtype.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.math.full((2, 3), 7.0)
        Array([[7., 7., 7.],
               [7., 7., 7.]], dtype=float32)
        >>> u.math.full((3,), 5.0 * u.meter)
        Quantity([5. 5. 5.], "m")
    """
    fill_value = maybe_custom_array(fill_value)
    if isinstance(fill_value, Quantity):
        xp = get_backend(fill_value.mantissa)
        return Quantity(xp.full(shape, fill_value.mantissa, dtype=dtype), unit=fill_value.unit)
    xp = _default_xp()
    return xp.full(shape, fill_value, dtype=dtype)


@set_module_as('saiunit.math')
def eye(
    N: int,
    M: Optional[int] = None,
    k: int = 0,
    dtype: Optional[DTypeLike] = None,
    unit: Unit = UNITLESS,
) -> Union[Array, Quantity]:
    """
    Return a 2-D identity-like quantity or array with ones on the diagonal.

    Parameters
    ----------
    N : int
        Number of rows in the output.
    M : int, optional
        Number of columns in the output.  If ``None``, defaults to ``N``.
    k : int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value
        to a lower diagonal.
    dtype : data-type, optional
        Data-type of the returned array.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  When ``UNITLESS`` (the default)
        a plain array is returned.

    Returns
    -------
    out : Quantity or Array
        An array of shape ``(N, M)`` where all elements are zero except for
        the ``k``-th diagonal, whose values are one (optionally carrying
        ``unit``).

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.math.eye(2)
        Array([[1., 0.],
               [0., 1.]], dtype=float32)
        >>> u.math.eye(2, unit=u.meter)
        Quantity([[1. 0.]
                  [0. 1.]], "m")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'eye requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    # ``k`` is keyword-only under the array-API spec (e.g. ``array_api_compat.numpy.eye``).
    arr = _default_xp().eye(N, M, k=k, dtype=dtype)
    if not unit.is_unitless:
        return arr * unit
    return arr


@set_module_as('saiunit.math')
def identity(
    n: int,
    dtype: Optional[DTypeLike] = None,
    unit: Unit = UNITLESS
) -> Union[Array, Quantity]:
    """
    Return the identity quantity or array.

    The identity array is a square array with ones on the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in the ``n x n`` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  When ``UNITLESS`` (the default)
        a plain array is returned.

    Returns
    -------
    out : Quantity or Array
        ``n x n`` array with its main diagonal set to one and all other
        elements zero.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.math.identity(3)
        Array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)
        >>> u.math.identity(2, unit=u.second)
        Quantity([[1. 0.]
                  [0. 1.]], "s")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'identity requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    if not unit.is_unitless:
        return _default_xp().identity(n, dtype=dtype) * unit
    else:
        return _default_xp().identity(n, dtype=dtype)


@set_module_as('saiunit.math')
def tri(
    N: int,
    M: Optional[int] = None,
    k: int = 0,
    dtype: Optional[DTypeLike] = None,
    unit: Unit = UNITLESS
) -> Union[Array, Quantity]:
    """
    Return an array with ones at and below the given diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array.  By default, ``M`` is taken equal
        to ``N``.
    k : int, optional
        The sub-diagonal at and below which the array is filled.
        ``k = 0`` is the main diagonal, ``k < 0`` is below it, and
        ``k > 0`` is above.  The default is 0.
    dtype : data-type, optional
        Data type of the returned array.  The default is ``float``.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or Array
        Array of shape ``(N, M)`` with its lower triangle filled with ones
        and zero elsewhere; i.e. ``T[i, j] == 1`` for ``j <= i + k``,
        0 otherwise.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.math.tri(3)
        Array([[1., 0., 0.],
               [1., 1., 0.],
               [1., 1., 1.]], dtype=float32)
        >>> u.math.tri(2, 3, unit=u.meter)
        Quantity([[1. 0. 0.]
                  [1. 1. 0.]], "m")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'tri requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    xp = _default_xp()
    # ``dask.array.tri`` rejects ``dtype=None`` ("dtype must be known for auto-chunking"),
    # while numpy/jax default to ``float``. Materialize the default explicitly for portability.
    if dtype is None:
        dtype = xp.float64 if hasattr(xp, "float64") else float
    arr = xp.tri(N, M, k, dtype=dtype)
    if not unit.is_unitless:
        return arr * unit
    return arr


@set_module_as('saiunit.math')
def empty(
    shape: Shape,
    dtype: Optional[DTypeLike] = None,
    unit: Unit = UNITLESS
) -> Union[Array, Quantity]:
    """
    Return a new quantity or array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the empty quantity or array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  When ``UNITLESS`` (the default)
        a plain array is returned.

    Returns
    -------
    out : Quantity or Array
        Array of uninitialized (arbitrary) data of the given shape and dtype.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> result = u.math.empty((2, 3))
        >>> result.shape
        (2, 3)
        >>> result = u.math.empty((2,), unit=u.meter)
        >>> u.get_unit(result) == u.meter
        True
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'empty requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    if not unit.is_unitless:
        return _default_xp().empty(shape, dtype=dtype) * unit
    else:
        return _default_xp().empty(shape, dtype=dtype)


@set_module_as('saiunit.math')
def ones(
    shape: Shape,
    dtype: Optional[DTypeLike] = None,
    unit: Unit = UNITLESS
) -> Union[Array, Quantity]:
    """
    Return a new quantity or array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new quantity or array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array.  Default is ``float``.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  When ``UNITLESS`` (the default)
        a plain array is returned.

    Returns
    -------
    out : Quantity or Array
        Array of ones with the given shape and dtype.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.math.ones((3,))
        Array([1., 1., 1.], dtype=float32)
        >>> u.math.ones((2, 2), unit=u.meter)
        Quantity([[1. 1.]
                  [1. 1.]], "m")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'ones requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    if not unit.is_unitless:
        return _default_xp().ones(shape, dtype=dtype) * unit
    else:
        return _default_xp().ones(shape, dtype=dtype)


@set_module_as('saiunit.math')
def zeros(
    shape: Shape,
    dtype: Optional[DTypeLike] = None,
    unit: Unit = UNITLESS
) -> Union[Array, Quantity]:
    """
    Return a new quantity or array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new quantity or array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array.  Default is ``float``.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  When ``UNITLESS`` (the default)
        a plain array is returned.

    Returns
    -------
    out : Quantity or Array
        Array of zeros with the given shape and dtype.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.math.zeros((3,))
        Array([0., 0., 0.], dtype=float32)
        >>> u.math.zeros((2,), unit=u.second)
        Quantity([0. 0.], "s")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'zeros requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    if not unit.is_unitless:
        return _default_xp().zeros(shape, dtype=dtype) * unit
    else:
        return _default_xp().zeros(shape, dtype=dtype)


@set_module_as('saiunit.math')
def full_like(
    a: Union[Quantity, ArrayLike],
    fill_value: Union[Quantity, ArrayLike],
    dtype: Optional[DTypeLike] = None,
    shape: Shape | None = None
) -> Union[Quantity, Array]:
    """
    Return a new quantity or array with the same shape and type as a given array, filled with ``fill_value``.

    Parameters
    ----------
    a : Quantity or array_like
        The shape and data-type of ``a`` define these same attributes of the
        returned array.
    fill_value : Quantity or array_like
        Value to fill the new quantity or array with.  When ``a`` is a
        ``Quantity``, ``fill_value`` must have a compatible unit.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or sequence of ints, optional
        Overrides the shape of the result.  If not given, ``a.shape`` is
        used.

    Returns
    -------
    out : Quantity or Array
        New array with the same shape and type as ``a``, filled with
        ``fill_value``.

    Raises
    ------
    TypeError
        If ``fill_value`` carries a unit but ``a`` is a plain array (not
        unitless), or vice-versa.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.full_like(jnp.array([1.0, 2.0]), 9.0)
        Array([9., 9.], dtype=float32)
        >>> u.math.full_like(jnp.array([1.0, 2.0]) * u.meter, 9.0 * u.meter)
        Quantity([9. 9.], "m")
    """
    a = maybe_custom_array(a)
    fill_value = maybe_custom_array(fill_value)
    xp = get_backend(a.mantissa if isinstance(a, Quantity) else a)
    if isinstance(fill_value, Quantity):
        if isinstance(a, Quantity):
            fill_value = fill_value.in_unit(a.unit)
            return Quantity(
                _safe_call_xp(xp.full_like, (a.mantissa, fill_value.mantissa), dict(dtype=dtype, shape=shape)),
                unit=a.unit
            )
        else:
            if not fill_value.is_unitless:
                raise TypeError(
                    f'full_like requires "fill_value" to be dimensionless when "a" is a plain array, '
                    f'but got fill_value with unit={fill_value.unit}. '
                    f'Either pass a plain number as fill_value or wrap "a" as a Quantity.'
                )
            return Quantity(
                _safe_call_xp(xp.full_like, (a, fill_value.mantissa), dict(dtype=dtype, shape=shape)),
                unit=fill_value.unit
            )
    else:
        if isinstance(a, Quantity):
            if not a.is_unitless:
                raise TypeError(
                    f'full_like requires "a" to be dimensionless when "fill_value" is a plain value, '
                    f'but got a with unit={a.unit}. '
                    f'Either pass a Quantity as fill_value or use a plain array for "a".'
                )
            return _safe_call_xp(xp.full_like, (a.mantissa, fill_value), dict(dtype=dtype, shape=shape))
        else:
            return _safe_call_xp(xp.full_like, (a, fill_value), dict(dtype=dtype, shape=shape))


@set_module_as('saiunit.math')
def diag(
    v: Union[Quantity, ArrayLike],
    k: int = 0,
    unit: Unit = UNITLESS
) -> Union[Quantity, Array]:
    """
    Extract a diagonal or construct a diagonal array.

    If ``v`` is a 1-D array, ``diag`` constructs a 2-D array with ``v`` on
    the ``k``-th diagonal.  If ``v`` is a 2-D array, ``diag`` extracts the
    ``k``-th diagonal and returns a 1-D array.

    Parameters
    ----------
    v : Quantity or array_like
        Input array.  1-D inputs produce a 2-D diagonal matrix; 2-D inputs
        have their ``k``-th diagonal extracted.
    k : int, optional
        Diagonal in question.  The default is 0.  Use ``k > 0`` for
        diagonals above the main diagonal and ``k < 0`` for diagonals below.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  Ignored when ``v`` already
        carries a unit.

    Returns
    -------
    out : Quantity or Array
        The extracted diagonal or constructed diagonal array.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.diag(jnp.array([1.0, 2.0, 3.0]))
        Array([[1., 0., 0.],
               [0., 2., 0.],
               [0., 0., 3.]], dtype=float32)
        >>> u.math.diag(jnp.array([1.0, 2.0]), unit=u.meter)
        Quantity([[1. 0.]
                  [0. 2.]], "m")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'diag requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    v = maybe_custom_array(v)
    xp = get_backend(v.mantissa if isinstance(v, Quantity) else v)
    if isinstance(v, Quantity):
        if not unit.is_unitless:
            v = v.in_unit(unit)
        return Quantity(_safe_call_xp(xp.diag, (v.mantissa,), dict(k=k)), unit=v.unit)
    else:
        if not unit.is_unitless:
            return _safe_call_xp(xp.diag, (v,), dict(k=k)) * unit
        else:
            return _safe_call_xp(xp.diag, (v,), dict(k=k))


@set_module_as('saiunit.math')
def tril(
    m: Union[Quantity, ArrayLike],
    k: int = 0,
    unit: Unit = UNITLESS
) -> Union[Quantity, Array]:
    """
    Return the lower triangle of an array.

    Return a copy of a matrix with the elements above the ``k``-th diagonal
    zeroed.  For arrays with ``ndim > 2``, ``tril`` applies to the final two
    axes.

    Parameters
    ----------
    m : Quantity or array_like
        Input array.
    k : int, optional
        Diagonal above which to zero elements.  ``k = 0`` is the main
        diagonal, ``k < 0`` is below it, and ``k > 0`` is above.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or Array
        Lower triangle of ``m``, of the same shape and data-type as ``m``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.tril(jnp.ones((3, 3)))
        Array([[1., 0., 0.],
               [1., 1., 0.],
               [1., 1., 1.]], dtype=float32)
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'tril requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    m = maybe_custom_array(m)
    xp = get_backend(m.mantissa if isinstance(m, Quantity) else m)
    if isinstance(m, Quantity):
        if not unit.is_unitless:
            m = m.in_unit(unit)
        return Quantity(xp.tril(m.mantissa, k=k), unit=m.unit)
    else:
        if not unit.is_unitless:
            return xp.tril(m, k=k) * unit
        else:
            return xp.tril(m, k=k)


@set_module_as('saiunit.math')
def triu(
    m: Union[Quantity, ArrayLike],
    k: int = 0,
    unit: Unit = UNITLESS
) -> Union[Quantity, Array]:
    """
    Return the upper triangle of an array.

    Return a copy of an array with the elements below the ``k``-th diagonal
    zeroed.  For arrays with ``ndim > 2``, ``triu`` applies to the final two
    axes.

    Parameters
    ----------
    m : Quantity or array_like
        Input array.
    k : int, optional
        Diagonal below which to zero elements.  ``k = 0`` is the main
        diagonal, ``k < 0`` is below it, and ``k > 0`` is above.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or Array
        Upper triangle of ``m``, of the same shape and data-type as ``m``.

    See Also
    --------
    tril : lower triangle of an array

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.triu(jnp.ones((3, 3)))
        Array([[1., 1., 1.],
               [0., 1., 1.],
               [0., 0., 1.]], dtype=float32)
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'triu requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    m = maybe_custom_array(m)
    xp = get_backend(m.mantissa if isinstance(m, Quantity) else m)
    if isinstance(m, Quantity):
        if not unit.is_unitless:
            m = m.in_unit(unit)
        return Quantity(xp.triu(m.mantissa, k=k), unit=m.unit)
    else:
        if not unit.is_unitless:
            return xp.triu(m, k=k) * unit
        else:
            return xp.triu(m, k=k)


@set_module_as('saiunit.math')
def empty_like(
    prototype: Union[Quantity, ArrayLike],
    dtype: Optional[DTypeLike] = None,
    shape: Shape | None = None,
    unit: Unit = UNITLESS
) -> Union[Quantity, Array]:
    """
    Return a new uninitialized quantity or array with the same shape and type as a given array.

    Parameters
    ----------
    prototype : Quantity or array_like
        The shape and data-type of ``prototype`` define these same attributes
        of the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple of ints, optional
        Overrides the shape of the result.  If not given,
        ``prototype.shape`` is used.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or Array
        Array of uninitialized (arbitrary) data with the same shape and type
        as ``prototype``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> result = u.math.empty_like(jnp.array([1.0, 2.0, 3.0]))
        >>> result.shape
        (3,)
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'empty_like requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    prototype = maybe_custom_array(prototype)
    xp = get_backend(prototype.mantissa if isinstance(prototype, Quantity) else prototype)
    if isinstance(prototype, Quantity):
        if not unit.is_unitless:
            prototype = prototype.in_unit(unit)
        return Quantity(
            _safe_call_xp(xp.empty_like, (prototype.mantissa,), dict(dtype=dtype)),
            unit=prototype.unit,
        )
    else:
        if not unit.is_unitless:
            return _safe_call_xp(xp.empty_like, (prototype,), dict(dtype=dtype, shape=shape)) * unit
        else:
            return _safe_call_xp(xp.empty_like, (prototype,), dict(dtype=dtype, shape=shape))


@set_module_as('saiunit.math')
def ones_like(
    a: Union[Quantity, ArrayLike],
    dtype: Optional[DTypeLike] = None,
    shape: Shape | None = None,
    unit: Unit = UNITLESS
) -> Union[Quantity, Array]:
    """
    Return a quantity or array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : Quantity or array_like
        The shape and data-type of ``a`` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple of ints, optional
        Overrides the shape of the result.  If not given, ``a.shape`` is
        used.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or Array
        Array of ones with the same shape and type as ``a``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.ones_like(jnp.array([1.0, 2.0, 3.0]))
        Array([1., 1., 1.], dtype=float32)
        >>> u.math.ones_like(jnp.array([1.0, 2.0]) * u.meter)
        Quantity([1. 1.], "m")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'ones_like requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    a = maybe_custom_array(a)
    xp = get_backend(a.mantissa if isinstance(a, Quantity) else a)
    if isinstance(a, Quantity):
        if not unit.is_unitless:
            a = a.in_unit(unit)
        return Quantity(
            _safe_call_xp(xp.ones_like, (a.mantissa,), dict(dtype=dtype, shape=shape)),
            unit=a.unit,
        )
    else:
        if not unit.is_unitless:
            return _safe_call_xp(xp.ones_like, (a,), dict(dtype=dtype, shape=shape)) * unit
        else:
            return _safe_call_xp(xp.ones_like, (a,), dict(dtype=dtype, shape=shape))


@set_module_as('saiunit.math')
def zeros_like(
    a: Union[Quantity, ArrayLike],
    dtype: Optional[DTypeLike] = None,
    shape: Shape | None = None,
    unit: Unit = UNITLESS
) -> Union[Quantity, Array]:
    """
    Return a quantity or array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : Quantity or array_like
        The shape and data-type of ``a`` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple of ints, optional
        Overrides the shape of the result.  If not given, ``a.shape`` is
        used.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or Array
        Array of zeros with the same shape and type as ``a``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.zeros_like(jnp.array([1.0, 2.0, 3.0]))
        Array([0., 0., 0.], dtype=float32)
        >>> u.math.zeros_like(jnp.array([1.0, 2.0]) * u.meter)
        Quantity([0. 0.], "m")
    """
    if not isinstance(unit, Unit):
        raise TypeError(f'zeros_like requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    a = maybe_custom_array(a)
    xp = get_backend(a.mantissa if isinstance(a, Quantity) else a)
    if isinstance(a, Quantity):
        if not unit.is_unitless:
            a = a.in_unit(unit)
        return Quantity(
            _safe_call_xp(xp.zeros_like, (a.mantissa,), dict(dtype=dtype, shape=shape)),
            unit=a.unit,
        )
    else:
        if not unit.is_unitless:
            return _safe_call_xp(xp.zeros_like, (a,), dict(dtype=dtype, shape=shape)) * unit
        else:
            return _safe_call_xp(xp.zeros_like, (a,), dict(dtype=dtype, shape=shape))


@set_module_as('saiunit.math')
def asarray(
    a: Any,
    dtype: Optional[DTypeLike] = None,
    order: Optional[str] = None,
    unit: Optional[Unit] = None,
) -> Quantity | Array | None:
    """
    Convert the input to a quantity or array.

    If ``unit`` is provided, the input is checked for compatible units and
    converted accordingly.  If ``unit`` is not provided, the unit is inferred
    from the input data.

    The function ``array`` is an alias for ``asarray``.

    Parameters
    ----------
    a : Quantity, array_like, list[Quantity], or list[array_like]
        Input data, in any form that can be converted to an array.  When a
        list of ``Quantity`` objects is given, all elements must share the
        same dimension.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout.  Defaults to ``'K'``.
    unit : Unit, optional
        Target unit of the returned ``Quantity``.  When given, all elements
        are converted to this unit.

    Returns
    -------
    out : Quantity or Array
        Array interpretation of ``a``.

    Raises
    ------
    UnitMismatchError
        If elements of ``a`` have incompatible units, or if ``unit`` is
        specified but does not match the dimension of ``a``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.math.array([1, 2, 3])
        Array([1, 2, 3], dtype=int32)
        >>> u.math.array([1, 2, 3] * u.meter)
        Quantity([1 2 3], "m")
        >>> u.math.asarray([1 * u.meter, 2 * u.meter])
        Quantity([1 2], "m")
    """
    if a is None:
        return a
    if isinstance(a, dict):
        raise TypeError(
            f"asarray does not accept dict inputs (got {type(a).__name__}); "
            "pass an array, list, or Quantity."
        )

    # get leaves
    leaves, treedef = _tree.flatten(a, is_leaf=lambda x: isinstance(x, Quantity))
    leaves = unit_scale_align_to_first(*leaves)
    leaf_unit = leaves[0].unit

    # get unit
    if unit is not None and not leaf_unit.is_unitless:
        if not isinstance(unit, Unit):
            raise TypeError(f'asarray requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
        leaves = [leaf.in_unit(unit) for leaf in leaves]
    else:
        unit = leaf_unit

    # reconstruct mantissa
    a = treedef.unflatten([leaf.mantissa for leaf in leaves])  # type: ignore[attr-defined]
    xp = _default_xp()
    # ``order`` is a numpy/jax-only kwarg; torch / dask / ndonnx ``asarray``
    # reject it. Only forward when explicitly provided.
    extra = {}
    if order is not None:
        extra["order"] = order
    a = xp.asarray(a, dtype=dtype, **extra)

    # returns
    if unit.is_unitless:
        return a
    return Quantity(a, unit=unit)


array = asarray


@set_module_as('saiunit.math')
def arange(
    start: Optional[Union[Quantity, ArrayLike]] = None,
    stop: Optional[Union[Quantity, ArrayLike]] = None,
    step: Optional[Union[Quantity, ArrayLike]] = None,
    dtype: Optional[DTypeLike] = None
) -> Union[Quantity, Array]:
    """
    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including ``start`` but excluding
    ``stop``).  All of ``start``, ``stop``, and ``step`` must share the
    same unit when any of them is a ``Quantity``.

    Parameters
    ----------
    start : Quantity or array_like, optional
        Start of the interval (inclusive).  The default start value is 0.
    stop : Quantity or array_like
        End of the interval (exclusive).
    step : Quantity or array_like, optional
        Spacing between values.  The default step size is 1.
    dtype : data-type, optional
        The type of the output array.  If not given, the dtype is inferred
        from the other input arguments.

    Returns
    -------
    out : Quantity or Array
        Array of evenly spaced values.

    Raises
    ------
    UnitMismatchError
        If ``start``, ``stop``, and ``step`` do not share the same unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.math.arange(5)
        Array([0, 1, 2, 3, 4], dtype=int32)
        >>> u.math.arange(0 * u.meter, 3 * u.meter, 1 * u.meter)
        Quantity([0 1 2], "m")
    """
    # apply maybe_custom_array to inputs
    start = maybe_custom_array(start) if start is not None else start
    stop = maybe_custom_array(stop) if stop is not None else stop
    step = maybe_custom_array(step) if step is not None else step

    # checking the dimension of the data
    non_none_data = [d for d in (start, stop, step) if d is not None]
    if len(non_none_data) == 0:
        raise ValueError('arange requires at least one of start, stop, or step to be provided.')
    d1 = non_none_data[0]
    for d2 in non_none_data[1:]:
        fail_for_unit_mismatch(
            d1,
            d2,
            error_message="Start value {d1} and stop value {d2} have to have the same units.",
            d1=d1,
            d2=d2
        )

    # convert to array
    unit = get_unit(d1)
    start = start.in_unit(unit).mantissa if isinstance(start, Quantity) else start
    stop = stop.in_unit(unit).mantissa if isinstance(stop, Quantity) else stop
    step = step.in_unit(unit).mantissa if isinstance(step, Quantity) else step
    # Build positional args without leading/trailing ``None``s. torch / dask /
    # ndonnx reject a ``None`` ``step`` positional that numpy and jax silently
    # treat as the default 1, and ndonnx additionally requires a non-``None``
    # ``stop``. Normalize the single-arg form ``arange(stop)`` to ``(0, stop)``
    # the way numpy does, then drop ``step`` when not given.
    if stop is None:
        stop = start
        start = 0
    pos = (start, stop) if step is None else (start, stop, step)
    xp = _default_xp()
    kwargs = {} if dtype is None else {"dtype": _translate_dtype(dtype, xp)}
    with jax.ensure_compile_time_eval():
        r = xp.arange(*pos, **kwargs)
    return r if unit.is_unitless else Quantity(r, unit=unit)


@set_module_as('saiunit.math')
def linspace(
    start: Union[Quantity, ArrayLike],
    stop: Union[Quantity, ArrayLike],
    num: int = 50,
    endpoint: Optional[bool] = True,
    retstep: Optional[bool] = False,
    dtype: Optional[DTypeLike] = None
) -> Union[Quantity, Array]:
    """
    Return evenly spaced numbers over a specified interval.

    Returns ``num`` evenly spaced samples, calculated over the interval
    ``[start, stop]``.  The endpoint of the interval can optionally be
    excluded.

    Parameters
    ----------
    start : Quantity or array_like
        The starting value of the sequence.
    stop : Quantity or array_like
        The end value of the sequence.  Must have the same unit as
        ``start`` when either is a ``Quantity``.
    num : int, optional
        Number of samples to generate.  Default is 50.
    endpoint : bool, optional
        If ``True``, ``stop`` is the last sample.  Otherwise, it is not
        included.  Default is ``True``.
    retstep : bool, optional
        If ``True``, return ``(samples, step)``, where ``step`` is the
        spacing between samples.
    dtype : data-type, optional
        The type of the output array.

    Returns
    -------
    samples : Quantity or Array
        ``num`` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``.

    Raises
    ------
    UnitMismatchError
        If ``start`` and ``stop`` do not share the same unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.math.linspace(0, 10, 5)
        Array([ 0. ,  2.5,  5. ,  7.5, 10. ], dtype=float32)
        >>> u.math.linspace(0 * u.meter, 10 * u.meter, 5)
        Quantity([ 0.   2.5  5.   7.5 10. ], "m")
    """
    start = maybe_custom_array(start)
    stop = maybe_custom_array(stop)
    fail_for_unit_mismatch(
        start,
        stop,
        error_message="Start value {start} and stop value {stop} have to have the same units.",
        start=start,
        stop=stop,
    )
    unit = get_unit(start)
    start = start.in_unit(unit).mantissa if isinstance(start, Quantity) else start
    stop = stop.in_unit(unit).mantissa if isinstance(stop, Quantity) else stop
    with jax.ensure_compile_time_eval():
        xp = _default_xp()
        result = _safe_call_xp(
            xp.linspace, (start, stop),
            dict(num=num, endpoint=endpoint, retstep=retstep, dtype=dtype),
        )
    return result if unit.is_unitless else Quantity(result, unit=unit)


@set_module_as('saiunit.math')
def logspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: Optional[int] = 50,
    endpoint: Optional[bool] = True,
    base: Optional[float] = 10.0,
    dtype: Optional[DTypeLike] = None
):
    """
    Return numbers spaced evenly on a log scale.

    In linear space, the sequence starts at ``base ** start`` and ends with
    ``base ** stop`` in ``num`` steps. Because ``base ** x`` is dimensionless,
    ``start`` and ``stop`` must be dimensionless and the result is a plain
    array (never a :class:`Quantity`).

    Parameters
    ----------
    start : array_like
        ``base ** start`` is the starting value of the sequence. Must be
        dimensionless.
    stop : array_like
        ``base ** stop`` is the final value of the sequence (unless
        ``endpoint`` is ``False``). Must be dimensionless.
    num : int, optional
        Number of samples to generate.  Default is 50.
    endpoint : bool, optional
        If ``True``, ``stop`` is the last sample.  Otherwise, it is not
        included.  Default is ``True``.
    base : float, optional
        The base of the log space.  Default is 10.0.
    dtype : data-type, optional
        The type of the output array.

    Returns
    -------
    samples : Array
        ``num`` samples, equally spaced on a log scale.

    Raises
    ------
    UnitMismatchError
        If ``start`` or ``stop`` carries a non-trivial unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.math.logspace(0, 2, 4)
        Array([  1.       ,   4.6415887,  21.544348 , 100.       ], dtype=float32)
    """
    start = maybe_custom_array(start)
    stop = maybe_custom_array(stop)
    for argname, value in (("start", start), ("stop", stop)):
        u = get_unit(value)
        if not u.is_unitless:
            raise UnitMismatchError(
                f"logspace requires dimensionless `{argname}`, got unit {u!r}. "
                f"`base ** x` is intrinsically dimensionless; pass a plain "
                f"scalar/array instead.",
                u,
            )
    start = start.mantissa if isinstance(start, Quantity) else start
    stop = stop.mantissa if isinstance(stop, Quantity) else stop
    with jax.ensure_compile_time_eval():
        xp = _default_xp()
        return _safe_call_xp(
            xp.logspace, (start, stop),
            dict(num=num, endpoint=endpoint, base=base, dtype=dtype),
        )


@set_module_as('saiunit.math')
def fill_diagonal(
    a: Union[Quantity, ArrayLike],
    val: Union[Quantity, ArrayLike],
    wrap: Optional[bool] = False,
    inplace: Optional[bool] = False
) -> Union[Quantity, Array]:
    """
    Fill the main diagonal of the given array of any dimensionality.

    For an array ``a`` with ``a.ndim >= 2``, the diagonal is the list of
    locations with indices ``a[i, i, ..., i]`` all identical.

    Parameters
    ----------
    a : Quantity or array_like
        Array in which to fill the diagonal.
    val : Quantity or array_like
        Value to be written on the diagonal.  Its unit must be compatible
        with that of ``a``.
    wrap : bool, optional
        If ``True``, the diagonal is "wrapped" after ``a.shape[1]`` and
        continues in the first column (for tall matrices).  Default is
        ``False``.
    inplace : bool, optional
        If ``True``, the diagonal is filled in-place.  Default is ``False``.

    Returns
    -------
    out : Quantity or Array
        The input array with the diagonal filled.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.fill_diagonal(jnp.zeros((3, 3)), 5.0)
        Array([[5., 0., 0.],
               [0., 5., 0.],
               [0., 0., 5.]], dtype=float32)
    """
    a = maybe_custom_array(a)
    val = maybe_custom_array(val)
    xp = get_backend(a.mantissa if isinstance(a, Quantity) else a)
    if isinstance(val, Quantity):
        if isinstance(a, Quantity):
            val = val.in_unit(a.unit)
            return Quantity(
                _safe_call_xp(xp.fill_diagonal, (a.mantissa, val.mantissa, wrap), dict(inplace=inplace)),
                unit=a.unit,
            )
        else:
            return Quantity(
                _safe_call_xp(xp.fill_diagonal, (a, val.mantissa, wrap), dict(inplace=inplace)),
                unit=val.unit,
            )
    else:
        if isinstance(a, Quantity):
            return Quantity(
                _safe_call_xp(xp.fill_diagonal, (a.mantissa, val, wrap), dict(inplace=inplace)),
                unit=a.unit,
            )
        else:
            return _safe_call_xp(xp.fill_diagonal, (a, val, wrap), dict(inplace=inplace))


@set_module_as('saiunit.math')
def meshgrid(
    *xi: Union[Quantity, ArrayLike],
    copy: Optional[bool] = True,
    sparse: Optional[bool] = False,
    indexing: Optional[str] = 'xy'
) -> List[Union[Quantity, Array]]:
    """
    Return coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of N-D
    scalar/vector fields over N-D grids, given one-dimensional coordinate
    arrays ``x1, x2, ..., xn``.

    Parameters
    ----------
    xi : Quantity or array_like
        1-D arrays representing the coordinates of a grid.
    copy : bool, optional
        Must be ``True`` (the default).  JAX does not support
        ``copy=False``.
    sparse : bool, optional
        If ``True``, return a sparse grid instead of a dense grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian (``'xy'``, default) or matrix (``'ij'``) indexing of
        output.

    Returns
    -------
    X1, X2, ..., XN : list of Quantity or Array
        Coordinate matrices.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> x, y = u.math.meshgrid(jnp.array([1, 2]), jnp.array([3, 4]))
        >>> x
        Array([[1, 2],
               [1, 2]], dtype=int32)
    """

    # Apply maybe_custom_array to inputs before processing
    xi = tuple(maybe_custom_array(x) for x in xi)
    args = [asarray(x) for x in xi]
    if not copy:
        raise ValueError("jax.numpy.meshgrid only supports copy=True")
    if indexing not in ["xy", "ij"]:
        raise ValueError(f"Valid values for indexing are 'xy' and 'ij', got {indexing}")
    if any(a.ndim != 1 for a in args):
        raise ValueError("Arguments to jax.numpy.meshgrid must be 1D, got shapes "
                         f"{[a.shape for a in args]}")
    if indexing == "xy" and len(args) >= 2:
        args[0], args[1] = args[1], args[0]
    shape = [1 if sparse else a.shape[0] for a in args]
    f_shape = lambda i, a: [*shape[:i], a.shape[0], *shape[i + 1:]] if sparse else shape

    def _broadcast_in_dim(x, target_shape, i):
        xp = get_backend(x)
        reshape_shape = [1] * len(args)
        reshape_shape[i] = x.shape[0]
        return xp.broadcast_to(xp.reshape(x, reshape_shape), target_shape)

    # use ``_tree.map`` to be Quantity-aware (Quantity is a registered pytree
    # when JAX is installed; the fallback ``_tree`` only descends standard
    # containers so plain arrays are passed straight through).
    output = [
        _tree.map(lambda x: _broadcast_in_dim(x, f_shape(i, x), i), a)
        for i, a, in enumerate(args)
    ]
    if indexing == "xy" and len(args) >= 2:
        output[0], output[1] = output[1], output[0]
    return output


@set_module_as('saiunit.math')
def vander(
    x: Union[Quantity, ArrayLike],
    N: Optional[bool] = None,
    increasing: Optional[bool] = False,
    unit: Unit = UNITLESS
) -> Union[Quantity, Array]:
    """
    Generate a Vandermonde matrix.

    The columns of the output matrix are powers of the input vector.

    Parameters
    ----------
    x : Quantity or array_like
        1-D input array.  Must be dimensionless if a ``Quantity``.
    N : int, optional
        Number of columns in the output.  If ``N`` is not specified, a
        square array is returned (``N = len(x)``).
    increasing : bool, optional
        Order of the powers of the columns.  If ``True``, the powers
        increase from left to right; if ``False`` (the default), they are
        reversed.
    unit : Unit, optional
        Unit of the returned ``Quantity``.

    Returns
    -------
    out : Quantity or Array
        Vandermonde matrix.

    Raises
    ------
    TypeError
        If ``x`` carries a non-trivial unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> u.math.vander(jnp.array([1, 2, 3]), 3)
        Array([[1, 1, 1],
               [4, 2, 1],
               [9, 3, 1]], dtype=int32)
    """
    x = maybe_custom_array(x)
    if isinstance(x, Quantity):
        if not x.is_unitless:
            raise TypeError(
                f'vander requires "x" to be dimensionless, '
                f'but got x with unit={x.unit}. '
                f'Pass "unit_to_scale" or strip the unit before calling vander.'
            )
        x = x.mantissa
    r = get_backend(x).vander(x, N=N, increasing=increasing)
    if not isinstance(unit, Unit):
        raise TypeError(f'vander requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    if not unit.is_unitless:
        return Quantity(r, unit=unit)
    else:
        return r


# indexing funcs
# --------------

def tril_indices(n, k=0, m=None):
    xp = _default_xp()
    name = getattr(xp, '__name__', '')
    # torch's binding spells the signature ``tril_indices(row, col, offset)``;
    # array-API / numpy / jax / dask spell it ``(n, k=0, m=None)``.
    if 'torch' in name:
        return xp.tril_indices(n, n if m is None else m, offset=k)
    # ndonnx has no ``tril_indices``; compute the static index pair on numpy
    # and wrap with ``xp.asarray`` so the caller receives backend-native arrays.
    if 'ndonnx' in name:
        rows, cols = np.tril_indices(n, k=k, m=m)
        return (xp.asarray(rows), xp.asarray(cols))
    return _safe_call_xp(xp.tril_indices, (n,), {'k': k, 'm': m})


@set_module_as('saiunit.math')
def tril_indices_from(
    arr: Union[Quantity, ArrayLike],
    k: Optional[int] = 0
) -> Tuple[Array, Array]:
    """
    Return the indices for the lower-triangle of an ``(n, m)`` array.

    Parameters
    ----------
    arr : Quantity or array_like
        The array for which the returned indices will be valid.
    k : int, optional
        Diagonal offset.  ``k = 0`` is the main diagonal, ``k < 0`` is
        below, and ``k > 0`` is above.

    Returns
    -------
    out : tuple of Array
        Row and column indices for the lower triangle.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> row, col = u.math.tril_indices_from(jnp.ones((3, 3)))
        >>> row
        Array([0, 1, 1, 2, 2, 2], dtype=int32)
    """
    arr = maybe_custom_array(arr)
    inner = arr.mantissa if isinstance(arr, Quantity) else arr
    return get_backend(inner).tril_indices_from(inner, k=k)


def triu_indices(n, k=0, m=None):
    xp = _default_xp()
    name = getattr(xp, '__name__', '')
    if 'torch' in name:
        return xp.triu_indices(n, n if m is None else m, offset=k)
    if 'ndonnx' in name:
        rows, cols = np.triu_indices(n, k=k, m=m)
        return (xp.asarray(rows), xp.asarray(cols))
    return _safe_call_xp(xp.triu_indices, (n,), {'k': k, 'm': m})


@set_module_as('saiunit.math')
def triu_indices_from(
    arr: Union[Quantity, ArrayLike],
    k: Optional[int] = 0
) -> Tuple[Array, Array]:
    """
    Return the indices for the upper-triangle of an ``(n, m)`` array.

    Parameters
    ----------
    arr : Quantity or array_like
        The array for which the returned indices will be valid.
    k : int, optional
        Diagonal offset.  ``k = 0`` is the main diagonal, ``k < 0`` is
        below, and ``k > 0`` is above.

    Returns
    -------
    out : tuple of Array
        Row and column indices for the upper triangle.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> row, col = u.math.triu_indices_from(jnp.ones((3, 3)))
        >>> row
        Array([0, 0, 0, 1, 1, 2], dtype=int32)
    """
    arr = maybe_custom_array(arr)
    inner = arr.mantissa if isinstance(arr, Quantity) else arr
    return get_backend(inner).triu_indices_from(inner, k=k)


# --- others ---


@set_module_as('saiunit.math')
def from_numpy(
    x: np.ndarray,
    unit: Unit = UNITLESS
) -> Array | Quantity:
    """
    Convert a NumPy array to a JAX array, optionally attaching a unit.

    Parameters
    ----------
    x : numpy.ndarray
        The NumPy array to convert.
    unit : Unit, optional
        Unit of the returned ``Quantity``.  When ``UNITLESS`` (the default)
        a plain JAX array is returned.

    Returns
    -------
    out : Quantity or Array
        JAX array (or ``Quantity``) created from ``x``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import numpy as np
        >>> u.math.from_numpy(np.array([1.0, 2.0]), unit=u.meter)
        Quantity([1. 2.], "m")
    """
    x = maybe_custom_array(x)
    if not isinstance(unit, Unit):
        raise TypeError(f'from_numpy requires "unit" to be a Unit instance, got {type(unit).__name__}: {unit!r}.')
    xp = _default_xp()
    if not unit.is_unitless:
        return xp.asarray(x) * unit
    return xp.asarray(x)


@set_module_as('saiunit.math')
def as_numpy(x):
    """
    Convert a JAX array (or ``Quantity``) to a NumPy array.

    Parameters
    ----------
    x : Quantity or array_like
        The input to convert.  If ``x`` is a ``Quantity``, the underlying
        mantissa (in current unit scale) is returned as a NumPy array.

    Returns
    -------
    out : numpy.ndarray
        NumPy array representation of ``x``.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> u.math.as_numpy(u.math.ones((3,)))
        array([1., 1., 1.], dtype=float32)
    """
    x = maybe_custom_array(x)
    return np.array(x)


@set_module_as('saiunit.math')
def tree_zeros_like(tree):
    """
    Create a tree with the same structure as the input, but with zeros in each leaf.

    Parameters
    ----------
    tree : pytree
        A JAX-compatible pytree (nested dicts, lists, tuples, etc.) whose
        leaves are arrays or ``Quantity`` objects.

    Returns
    -------
    out : pytree
        A tree with the same structure, where every leaf is replaced by a
        zero-filled array (or ``Quantity``) of the same shape, dtype, and
        unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> u.math.tree_zeros_like(tree)
        {'a': Array([0., 0.], dtype=float32), 'b': Array([0.], dtype=float32)}
    """
    tree = maybe_custom_array_tree(tree)
    return _tree.map(zeros_like, tree)


@set_module_as('saiunit.math')
def tree_ones_like(tree):
    """
    Create a tree with the same structure as the input, but with ones in each leaf.

    Parameters
    ----------
    tree : pytree
        A JAX-compatible pytree (nested dicts, lists, tuples, etc.) whose
        leaves are arrays or ``Quantity`` objects.

    Returns
    -------
    out : pytree
        A tree with the same structure, where every leaf is replaced by a
        ones-filled array (or ``Quantity``) of the same shape, dtype, and
        unit.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import jax.numpy as jnp
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> u.math.tree_ones_like(tree)
        {'a': Array([1., 1.], dtype=float32), 'b': Array([1.], dtype=float32)}
    """
    tree = maybe_custom_array_tree(tree)
    return _tree.map(ones_like, tree)
