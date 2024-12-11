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

from ._base import Unit
from ._unit_common import joule, kilogram, second, meter

# ----- Time -----

minute = Unit.create(second.dim, name="minute", dispname="min", scale=second.scale + 1, factor=6.0)
hour = Unit.create(second.dim, name="hour", dispname="h", scale=second.scale + 3, factor=3.600)
day = Unit.create(second.dim, name="day", dispname="d", scale=second.scale + 4, factor=8.6400)
week = Unit.create(second.dim, name="week", dispname="wk", scale=second.scale + 5, factor=6.04800)
month = Unit.create(second.dim, name="month", dispname="mo", scale=second.scale + 6, factor=2.629746)
year = Unit.create(second.dim, name="year", dispname="yr", scale=second.scale + 7, factor=3.1556952)
julian_year = Unit.create(second.dim, name="julian year", dispname="julian yr", scale=second.scale + 7, factor=3.15576)

# ----- Length -----

inch = Unit.create(meter.dim, name="inch", dispname="in", scale=meter.scale, factor=0.0254)
foot = Unit.create(meter.dim, name="foot", dispname="ft", scale=meter.scale, factor=0.3048)
yard = Unit.create(meter.dim, name="yard", dispname="yd", scale=meter.scale, factor=0.9144)
mile = Unit.create(meter.dim, name="mile", dispname="mi", scale=meter.scale + 3, factor=1.609344)
mil = Unit.create(meter.dim, name="mil", dispname="mil", scale=meter.scale, factor=2.54e-5)
point = Unit.create(meter.dim, name="point", dispname="pt", scale=meter.scale - 4, factor=3.5277777777777776)
pica = Unit.create(meter.dim, name="pica", dispname="p", scale=meter.scale, factor=4.233333333333333e-3)
survey_foot = Unit.create(meter.dim, name="survey foot", dispname="ft", scale=meter.scale, factor=0.3048006096012192)
survey_mile = Unit.create(meter.dim, name="survey mile", dispname="mi", scale=meter.scale + 3,
                          factor=1.6093472186944374)
nautical_mile = Unit.create(meter.dim, name="nautical mile", dispname="nmi", scale=meter.scale + 3, factor=1.8520)
fermi = Unit.create(meter.dim, name="fermi", dispname="fm", scale=meter.scale - 15)
angstrom = Unit.create(meter.dim, name="angstrom", dispname="Å", scale=-10, factor=1.0)
micron = Unit.create(meter.dim, name="micron", dispname="µm", scale=-6, factor=1.0e-6)
astronomical_unit = Unit.create(meter.dim, name="astronomical unit", dispname="AU", scale=11, factor=1.495978707e11)
light_year = Unit.create(meter.dim / second.dim, name="light year", dispname="ly", scale=11, factor=1.094991952845)

# UNITS in modular dynamics
# See https://github.com/chaobrain/brainunit/issues/63

electron_volt = Unit.create(joule.dim, name="electronvolt", dispname="eV", scale=-19, factor=1.602176565)
elementary_charge = eV = electron_volt
# atomic mass unit (amu)
AMU = Unit.create(kilogram.dim, name="atomic mass unit", dispname="AMU", scale=-27, factor=1.66053886)
# Intermolecular force 分子间作用力
IMF = Unit.create(eV.dim / angstrom.dim, name="intermolecular force", dispname="IMF", scale=-9, factor=1.602176565)



