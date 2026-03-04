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

import importlib.util
from typing import Any

from ._base_getters import fail_for_dimension_mismatch
from ._base_quantity import Quantity
from ._base_unit import UNITLESS


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


class QuantityConverter(_CONVERSION_INTERFACE):
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
matplotlib_converter_registered = register_quantity_converter()
