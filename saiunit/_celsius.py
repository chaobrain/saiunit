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


import jax.typing

from ._base import Quantity
from ._misc import maybe_custom_array
from ._unit_common import kelvin

__all__ = [
    "celsius2kelvin",
    "kelvin2celsius",
]


def celsius2kelvin(celsius: jax.typing.ArrayLike) -> Quantity:
    """
    Convert Celsius value to kelvin value.

    Parameters
    ----------
    celsius : jax.typing.ArrayLike
      The celsius value to convert.

    Returns
    -------
      Quantity: The converted value.

    Raises
    ------
    TypeError
      If the input value is a Quantity.

    Examples
    --------
    >>> from saiunit import kelvin, celsius2kelvin
    >>> celsius = 25.0
    >>> celsius2kelvin(celsius)
    298.15 * kelvin

    """
    celsius = maybe_custom_array(celsius)
    if isinstance(celsius, Quantity):
        raise TypeError("The input value should be not be a Quantity.")
    return (celsius + 273.15) * kelvin


def kelvin2celsius(value: Quantity) -> jax.typing.ArrayLike:
    """
    Convert kelvin value to Celsius value.

    Parameters
    ----------
    value : Quantity
      The kelvin value to convert.

    Returns
    -------
      Quantity: The converted value.

    Raises
    ------
    TypeError
      If the input value is not a Quantity with kelvin unit.

    Examples
    --------
    >>> from saiunit import kelvin, kelvin2celsius
    >>> value = 298.15 * kelvin
    >>> kelvin2celsius(value)
    25.0

    """
    value = maybe_custom_array(value)
    if not (isinstance(value, Quantity) and value.unit == kelvin):
        raise TypeError("The input value should be a Quantity with kelvin.")
    return value.mantissa - 273.15
