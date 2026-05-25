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

import functools
import importlib.util
import inspect
from typing import Any

import numpy as np

from ._base_getters import fail_for_dimension_mismatch
from ._base_quantity import Quantity
from ._base_unit import UNITLESS, Unit


def _has_matplotlib() -> bool:
    """Return whether Matplotlib is importable in the current environment."""
    return importlib.util.find_spec('matplotlib') is not None


def _load_matplotlib_units() -> Any | None:
    """Import and return ``matplotlib.units`` if available."""
    if not _has_matplotlib():
        return None
    try:
        from matplotlib import units as matplotlib_units
    except Exception:
        return None
    return matplotlib_units


def _resolve_conversion_error(units_module: Any | None) -> type[Exception]:
    """Return Matplotlib's conversion error type, or a safe fallback."""
    if units_module is None:
        return ValueError
    error_type = getattr(units_module, 'ConversionError', None)
    if isinstance(error_type, type) and issubclass(error_type, Exception):
        return error_type
    return ValueError


def _unit_label(unit: Any) -> str | None:
    """Return a display label for a unit object."""
    label = getattr(unit, 'dispname', None)
    if label is not None:
        return str(label)
    try:
        return str(unit)
    except Exception:
        return None


def _as_quantity(value: Any) -> Quantity:
    """Coerce ``value`` into a :class:`~saiunit.Quantity`."""
    return value if isinstance(value, Quantity) else Quantity(value)


_UNITS_MODULE = _load_matplotlib_units()
_AXIS_INFO = getattr(_UNITS_MODULE, 'AxisInfo', None) if _UNITS_MODULE is not None else None
_CONVERSION_INTERFACE = getattr(_UNITS_MODULE, 'ConversionInterface', object) if _UNITS_MODULE is not None else object
_CONVERSION_ERROR = _resolve_conversion_error(_UNITS_MODULE)


class QuantityConverter(_CONVERSION_INTERFACE):  # type: ignore[misc,valid-type]
    """Matplotlib converter for :class:`~saiunit.Quantity` values."""

    @staticmethod
    def axisinfo(unit: Any, axis: Any) -> Any:
        """Provide axis metadata for Quantity-backed Matplotlib axes."""
        if _AXIS_INFO is None or unit is None:
            return None
        if unit == UNITLESS:
            return _AXIS_INFO()
        label = _unit_label(unit)
        if label is None:
            return _AXIS_INFO()
        try:
            return _AXIS_INFO(label=label)
        except TypeError:
            # Handle AxisInfo signature differences across Matplotlib versions.
            return _AXIS_INFO()

    @staticmethod
    def convert(val: Any, unit: Any, axis: Any) -> Any:
        """Convert Quantity values to numeric data for Matplotlib."""
        # Plain (non-Quantity) values are already expressed in axis units; pass
        # them through unchanged. Matplotlib re-converts already-converted patch
        # coordinates (e.g. axhspan/axvspan), and this also lets callers mix bare
        # numbers onto a unit-bearing axis, matching Matplotlib's own semantics.
        contains_quantity = isinstance(val, Quantity) or (
            isinstance(val, (list, tuple))
            and any(isinstance(item, Quantity) for item in val)
        )
        if not contains_quantity:
            return np.asarray(val)

        try:
            quantity = _as_quantity(val)
        except Exception as exc:
            raise _CONVERSION_ERROR(
                f"Unable to convert value of type '{type(val).__name__}' into a Quantity."
            ) from exc

        if unit is None:
            return quantity.mantissa

        try:
            fail_for_dimension_mismatch(quantity.unit, unit)
            return quantity.to(unit).mantissa
        except Exception as exc:
            raise _CONVERSION_ERROR(
                f"Quantity unit '{quantity.unit}' is incompatible with target axis unit '{unit}'."
            ) from exc

    @staticmethod
    def default_units(x: Any, axis: Any) -> Any:
        """Infer default units for values passed to Matplotlib."""
        try:
            return _as_quantity(x).unit
        except Exception:
            return None


def register_quantity_converter(units_module: Any | None = None) -> bool:
    """
    Register :class:`QuantityConverter` in a Matplotlib units registry.

    Parameters
    ----------
    units_module : Any, optional
        A ``matplotlib.units``-like module for compatibility testing.

    Returns
    -------
    bool
        ``True`` if registration succeeded, otherwise ``False``.
    """
    units_module = _UNITS_MODULE if units_module is None else units_module
    if units_module is None:
        return False
    registry = getattr(units_module, 'registry', None)
    if registry is None or not hasattr(registry, '__setitem__'):
        return False
    registry[Quantity] = QuantityConverter()
    return True


matplotlib_installed = _has_matplotlib()
matplotlib_converter_registered = False
matplotlib_reshape_patch_installed = False
matplotlib_axes_patch_installed = False


# ---------------------------------------------------------------------------
# Quantity-aware wrappers for Axes methods that coerce data before the unit
# converter runs (errorbar, boxplot, violinplot, stackplot, hexbin, pie).
# ---------------------------------------------------------------------------

def _contains_quantity(value: Any) -> bool:
    """Return whether ``value`` is, or recursively contains, a Quantity."""
    if isinstance(value, Quantity):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_quantity(item) for item in value)
    return False


def _first_unit(value: Any) -> Any:
    """Return the first Quantity unit found in ``value``, or ``None``."""
    if isinstance(value, Quantity):
        return value.unit
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_unit(item)
            if found is not None:
                return found
    return None


def _to_decimal_like(value: Any, unit: Any) -> Any:
    """Convert every Quantity in ``value`` to a plain magnitude in ``unit``."""
    if isinstance(value, Quantity):
        return value.to_decimal(unit)
    if isinstance(value, (list, tuple)):
        return type(value)(_to_decimal_like(item, unit) for item in value)
    return value


def _existing_axis_unit(axis: Any) -> Any:
    """Return the unit already attached to ``axis``, or ``None``."""
    current = axis.get_units()
    return current if isinstance(current, Unit) else None


def _get_arg(args: list, kwargs: dict, index: Any, name: str) -> tuple[Any, Any]:
    """Locate an argument by keyword name or positional index."""
    if name is not None and name in kwargs:
        return kwargs[name], ('kw', name)
    if index is not None and index < len(args):
        return args[index], ('pos', index)
    return None, None


def _set_arg(args: list, kwargs: dict, loc: tuple, value: Any) -> None:
    """Write ``value`` back to the located argument slot."""
    kind, key = loc
    if kind == 'kw':
        kwargs[key] = value
    else:
        args[key] = value


def _bind(args: list, kwargs: dict, index: Any, name: str, axis: Any, chosen: dict) -> None:
    """Convert one argument to magnitudes and record the unit for its axis.

    ``axis`` of ``None`` marks data with no axis-unit semantics (pie wedge
    sizes, hexbin colour values); such data is stripped to its own magnitude.
    """
    value, loc = _get_arg(args, kwargs, index, name)
    if loc is None or not _contains_quantity(value):
        return
    if axis is None:
        _set_arg(args, kwargs, loc, _to_decimal_like(value, _first_unit(value)))
        return
    unit = chosen.get(axis) or _existing_axis_unit(axis) or _first_unit(value)
    chosen[axis] = unit
    _set_arg(args, kwargs, loc, _to_decimal_like(value, unit))


def _orientation_value_axis(self: Any, kwargs: dict) -> Any:
    """Return the value axis (where the data magnitude lives) for box/violin."""
    orientation = kwargs.get('orientation', None)
    if orientation is not None:
        vertical = orientation == 'vertical'
    else:
        vert = kwargs.get('vert', None)
        vertical = True if vert is None else bool(vert)
    return self.yaxis if vertical else self.xaxis


def _pre_errorbar(self: Any, args: list, kwargs: dict, chosen: dict) -> None:
    _bind(args, kwargs, 0, 'x', self.xaxis, chosen)
    _bind(args, kwargs, 1, 'y', self.yaxis, chosen)
    _bind(args, kwargs, 2, 'yerr', self.yaxis, chosen)
    _bind(args, kwargs, 3, 'xerr', self.xaxis, chosen)


def _pre_boxplot(self: Any, args: list, kwargs: dict, chosen: dict) -> None:
    value_axis = _orientation_value_axis(self, kwargs)
    other_axis = self.xaxis if value_axis is self.yaxis else self.yaxis
    _bind(args, kwargs, 0, 'x', value_axis, chosen)
    _bind(args, kwargs, None, 'positions', other_axis, chosen)


def _pre_violinplot(self: Any, args: list, kwargs: dict, chosen: dict) -> None:
    value_axis = _orientation_value_axis(self, kwargs)
    other_axis = self.xaxis if value_axis is self.yaxis else self.yaxis
    _bind(args, kwargs, 0, 'dataset', value_axis, chosen)
    _bind(args, kwargs, None, 'positions', other_axis, chosen)


def _pre_stackplot(self: Any, args: list, kwargs: dict, chosen: dict) -> None:
    _bind(args, kwargs, 0, 'x', self.xaxis, chosen)
    # Every remaining positional argument is a stacked y series sharing one unit.
    for i in range(1, len(args)):
        if _contains_quantity(args[i]):
            unit = chosen.get(self.yaxis) or _existing_axis_unit(self.yaxis) or _first_unit(args[i])
            chosen[self.yaxis] = unit
            args[i] = _to_decimal_like(args[i], unit)
    if 'y' in kwargs and _contains_quantity(kwargs['y']):
        unit = chosen.get(self.yaxis) or _existing_axis_unit(self.yaxis) or _first_unit(kwargs['y'])
        chosen[self.yaxis] = unit
        kwargs['y'] = _to_decimal_like(kwargs['y'], unit)


def _pre_hexbin(self: Any, args: list, kwargs: dict, chosen: dict) -> None:
    _bind(args, kwargs, 0, 'x', self.xaxis, chosen)
    _bind(args, kwargs, 1, 'y', self.yaxis, chosen)
    _bind(args, kwargs, 2, 'C', None, chosen)  # colour values carry no axis unit


def _pre_pie(self: Any, args: list, kwargs: dict, chosen: dict) -> None:
    _bind(args, kwargs, 0, 'x', None, chosen)  # wedge sizes are proportions


# Maps an Axes method name to its preprocessor and the signature parameter
# names the preprocessor relies on (used by the fail-loud signature guard).
_AXES_WRAPPER_SPECS: dict[str, tuple[Any, tuple[str, ...]]] = {
    'errorbar': (_pre_errorbar, ('x', 'y', 'yerr', 'xerr')),
    'boxplot': (_pre_boxplot, ('x',)),
    'violinplot': (_pre_violinplot, ('dataset',)),
    'stackplot': (_pre_stackplot, ('x',)),
    'hexbin': (_pre_hexbin, ('x', 'y', 'C')),
    'pie': (_pre_pie, ('x',)),
}


def _signature_has_params(func: Any, required: tuple[str, ...]) -> bool:
    """Return whether ``func`` still exposes every required parameter name."""
    try:
        names = set(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return True  # Not introspectable; assume the wrapper is still valid.
    return all(name in names for name in required)


def _apply_axis_unit(axis: Any, unit: Any) -> None:
    """Attach ``unit`` to ``axis`` and label it if it has no label yet."""
    try:
        axis.set_units(unit)
    except Exception:
        return
    if not axis.label.get_text():
        text = _unit_label(unit)
        if text:
            axis.set_label_text(text)


def _any_quantity(args: tuple, kwargs: dict) -> bool:
    return (
        any(_contains_quantity(arg) for arg in args)
        or any(_contains_quantity(value) for value in kwargs.values())
    )


def _make_axes_wrapper(original: Any, preprocess: Any, name: str, supported: bool) -> Any:
    """Build a Quantity-aware replacement for an Axes method."""

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not _any_quantity(args, kwargs):
            return original(self, *args, **kwargs)
        if not supported:
            import matplotlib
            raise _CONVERSION_ERROR(
                f"saiunit cannot adapt Axes.{name} for Quantity inputs: "
                f"matplotlib {matplotlib.__version__} changed its signature. "
                f"Convert with .to_decimal(unit) and pass plain values instead."
            )
        arglist = list(args)
        kwdict = dict(kwargs)
        chosen: dict = {}
        preprocess(self, arglist, kwdict, chosen)
        for axis, unit in chosen.items():
            _apply_axis_unit(axis, unit)
        return original(self, *arglist, **kwdict)

    return wrapper


def _install_axes_method_wrappers() -> bool:
    """Wrap Axes methods that coerce data before the unit converter runs.

    These functions (``errorbar``, ``boxplot``, ``violinplot``, ``stackplot``,
    ``hexbin``, ``pie``) call ``numpy.asarray`` on their inputs before the
    registered converter is consulted, so a bare :class:`~saiunit.Quantity`
    would raise. Each wrapper converts Quantity arguments to plain magnitudes
    via the public API and labels the relevant axis with the unit, leaving
    non-Quantity calls untouched.
    """
    global matplotlib_axes_patch_installed
    if matplotlib_axes_patch_installed:
        return True
    try:
        from matplotlib.axes import Axes
    except Exception:
        return False
    for name, (preprocess, required) in _AXES_WRAPPER_SPECS.items():
        original = getattr(Axes, name, None)
        if not callable(original):
            continue
        supported = _signature_has_params(original, required)
        setattr(Axes, name, _make_axes_wrapper(original, preprocess, name, supported))
    matplotlib_axes_patch_installed = True
    return True


def _install_reshape_2d_patch() -> bool:
    """Teach Matplotlib's ``_reshape_2D`` helper to accept Quantity inputs.

    Functions such as :meth:`~matplotlib.axes.Axes.hist` massage their data with
    ``matplotlib.cbook._reshape_2D`` *before* the registered unit converter
    runs. Because a :class:`~saiunit.Quantity` is not an ``ndarray`` subclass,
    that step coerces it with ``numpy.asanyarray`` and fails. Wrapping the helper
    lets Quantity data flow through to ``_process_unit_info``, so the converter
    can scale values and label the axis while keeping units intact.
    """
    global matplotlib_reshape_patch_installed
    if matplotlib_reshape_patch_installed:
        return True
    try:
        from matplotlib import cbook
    except Exception:
        return False
    original = getattr(cbook, '_reshape_2D', None)
    if not callable(original):
        return False

    def _reshape_2D(value: Any, name: str) -> Any:
        if isinstance(value, Quantity):
            if value.ndim == 0:
                return [value.reshape(1)]
            if value.ndim == 1:
                return [value]
            if value.ndim == 2:
                return [value[:, i] for i in range(value.shape[1])]
            raise ValueError(f"{name} must have 2 or fewer dimensions")
        if (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and all(isinstance(item, Quantity) and item.ndim >= 1 for item in value)
        ):
            return [item.reshape(-1) for item in value]
        return original(value, name)

    setattr(cbook, '_reshape_2D', _reshape_2D)
    matplotlib_reshape_patch_installed = True
    return True


def enable_matplotlib_support(units_module: Any | None = None) -> bool:
    """
    Register :class:`QuantityConverter` in Matplotlib's unit registry.

    Importing :mod:`saiunit` registers this converter automatically when
    Matplotlib is installed, so Quantity values render on Matplotlib axes out of
    the box. Call this function directly only to re-register after the registry
    has been cleared, or to register against a custom ``units_module``.

    Parameters
    ----------
    units_module : Any, optional
        A ``matplotlib.units``-like module, primarily for testing.

    Returns
    -------
    bool
        ``True`` if registration succeeded, otherwise ``False``.
    """
    global matplotlib_converter_registered
    registered = register_quantity_converter(units_module)
    if units_module is None and registered:
        matplotlib_converter_registered = True
        _install_reshape_2d_patch()
        _install_axes_method_wrappers()
    return registered


if matplotlib_installed:
    enable_matplotlib_support()
