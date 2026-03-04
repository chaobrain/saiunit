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


import jax.typing

from ._base_quantity import Quantity
from ._misc import maybe_custom_array
from ._unit_common import kelvin

__all__ = [
    "celsius2kelvin",
    "kelvin2celsius",
]


def celsius2kelvin(celsius: jax.typing.ArrayLike) -> Quantity:
    """
    Convert a Celsius value to a kelvin :class:`~saiunit.Quantity`.

    Parameters
    ----------
    celsius : jax.typing.ArrayLike
        The temperature in degrees Celsius. Must not be a
        :class:`~saiunit.Quantity`.

    Returns
    -------
    Quantity
        The temperature expressed in kelvin.

    Raises
    ------
    TypeError
        If ``celsius`` is already a :class:`~saiunit.Quantity`.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su.celsius2kelvin(0.0)
        273.15 * kelvin
        >>> su.celsius2kelvin(25.0)
        298.15 * kelvin
        >>> su.celsius2kelvin(-40.0)
        233.14999999999998 * kelvin

    """
    celsius = maybe_custom_array(celsius)
    if isinstance(celsius, Quantity):
        raise TypeError("The input value should be not be a Quantity.")
    return (celsius + 273.15) * kelvin


def kelvin2celsius(value: Quantity) -> jax.typing.ArrayLike:
    """
    Convert a kelvin :class:`~saiunit.Quantity` to a Celsius value.

    Parameters
    ----------
    value : Quantity
        The temperature expressed as a :class:`~saiunit.Quantity` with
        kelvin units.

    Returns
    -------
    jax.typing.ArrayLike
        The temperature in degrees Celsius (unitless scalar or array).

    Raises
    ------
    TypeError
        If ``value`` is not a :class:`~saiunit.Quantity` with kelvin units.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> su.kelvin2celsius(273.15 * su.kelvin)
        0.0
        >>> su.kelvin2celsius(298.15 * su.kelvin)
        25.0
        >>> su.kelvin2celsius(373.15 * su.kelvin)
        100.0

    """
    value = maybe_custom_array(value)
    if not isinstance(value, Quantity):
        raise TypeError("The input value should be a Quantity with a temperature unit.")
    if not value.unit.has_same_dim(kelvin):
        raise TypeError(
            f"The input value should be a Quantity with a temperature unit, "
            f"but got unit {value.unit}."
        )
    return value.to_decimal(kelvin) - 273.15
