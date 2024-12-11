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
#
# THIS FILE IS AUTOMATICALLY GENERATED
# BY A STATIC CODE GENERATION TOOL
# DO NOT EDIT BY HAND
#
# ==============================================================================


# fmt: off
# flake8: noqa
from ._base import (Unit, get_or_create_dimension)

{all}

Unit.automatically_register_units = False

#### FUNDAMENTAL UNITS
metre = Unit.create(get_or_create_dimension(m=1), "metre", "m")
meter = Unit.create(get_or_create_dimension(m=1), "meter", "m")
# Liter has a scale of 10^-3, since 1 l = 1 dm^3 = 10^-3 m^3
liter = Unit.create(dim=(meter**3).dim, name="liter", dispname="l", scale=-3)
litre = Unit.create(dim=(meter**3).dim, name="litre", dispname="l", scale=-3)
kilogram = Unit.create(get_or_create_dimension(kg=1), "kilogram", "kg")
kilogramme = Unit.create(get_or_create_dimension(kg=1), "kilogramme", "kg")
gram = Unit.create(dim=kilogram.dim, name="gram", dispname="g", scale=-3)
gramme = Unit.create(dim=kilogram.dim, name="gramme", dispname="g", scale=-3)
second = Unit.create(get_or_create_dimension(s=1), "second", "s")
amp = Unit.create(get_or_create_dimension(A=1), "amp", "A")
ampere = Unit.create(get_or_create_dimension(A=1), "ampere", "A")
kelvin = Unit.create(get_or_create_dimension(K=1), "kelvin", "K")
mole = Unit.create(get_or_create_dimension(mol=1), "mole", "mol")
mol = Unit.create(get_or_create_dimension(mol=1), "mol", "mol")
# Molar has a scale of 10^3, since 1 M = 1 mol/l = 1000 mol/m^3
molar = Unit.create((mole/liter).dim, name="molar", dispname="M", scale=3)
candle = Unit.create(get_or_create_dimension(candle=1), "candle", "cd")
fundamental_units = [metre, meter, gram, second, amp, kelvin, mole, candle]


### Derived units

{derived}


# Difinitions of base units

{definitions}

{base_units}
{scaled_units}
{powered_units}
{additional_units}
{all_units}


del base_units, scaled_units, powered_units, additional_units
# fmt: on
