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
from ._unit_common import joule, kilogram, second, meter, radian, pascal, meter2, meter3, kelvin, watt, newton
from .math import pi

__all__ = [
    "metric_ton", "grain", "lb", "pound", "slinch", "blob", "slug", "oz", "ounce", "stone", "long_ton", "short_ton",
    "troy_ounce", "troy_pound", "carat", "atomic_mass", "u", "um_u", "degree", "arcmin", "arcminute", "arcsec",
    "arcsecond", "minute", "hour", "day", "week", "month", "year", "julian_year", "inch", "foot", "yard", "mile",
    "mil", "point", "pica", "survey_foot", "survey_mile", "nautical_mile", "fermi", "angstrom", "micron",
    "astronomical_unit", "au", "light_year", "parsec", "atm", "atmosphere", "bar", "mmHg", "torr", "psi",
    "hectare", "acre", "gallon", "gallon_US", "gallon_imp", "fluid_ounce", "fluid_ounce_US", "fluid_ounce_imp",
    "bbl", "barrel", "speed_unit", "kmh", "mph", "mach", "speed_of_sound", "knot", "degree_Fahrenheit", "eV",
    "electron_volt", "calorie", "calorie_th", "calorie_IT", "erg", "Btu", "Btu_IT", "Btu_th", "ton_TNT", "hp",
    "horsepower", "dyn", "dyne", "lbf", "pound_force", "kgf", "kilogram_force", "IMF", 'kcal_per_h'
]

# ----- Mass -----
metric_ton = Unit.create(kilogram.dim, name="metric ton", dispname="t", scale=kilogram.scale + 3)
grain = Unit.create(kilogram.dim, name="grain", dispname="gr", scale=kilogram.scale - 5, factor=6.479891)
lb = pound = Unit.create(kilogram.dim, name="pound", dispname="lb", scale=kilogram.scale, factor=0.45359237)
slinch = blob = Unit.create(kilogram.dim, name="blob", dispname="blob", scale=kilogram.scale + 2, factor=1.75126836)
slug = Unit.create(kilogram.dim, name="slug", dispname="slug", scale=kilogram.scale + 1, factor=1.459390294)
oz = ounce = Unit.create(kilogram.dim, name="ounce", dispname="oz", scale=kilogram.scale - 2, factor=2.8349523125)
stone = Unit.create(kilogram.dim, name="stone", dispname="st", scale=kilogram.scale, factor=6.35029318)
long_ton = Unit.create(kilogram.dim, name="long ton", dispname="long ton", scale=kilogram.scale + 3,
                       factor=1.0160469088)
short_ton = Unit.create(kilogram.dim, name="short ton", dispname="short ton", scale=kilogram.scale + 3,
                        factor=0.90718474)
troy_ounce = Unit.create(kilogram.dim, name="troy ounce", dispname="oz t", scale=kilogram.scale - 2, factor=3.11034768)
troy_pound = Unit.create(kilogram.dim, name="troy pound", dispname="lb t", scale=kilogram.scale, factor=0.3732417216)
carat = Unit.create(kilogram.dim, name="carat", dispname="ct", scale=kilogram.scale - 4, factor=2.)
# atomic mass unit (amu)
atomic_mass = u = um_u = Unit.create(kilogram.dim, name="atomic mass unit", dispname="u", scale=kilogram.scale - 27,
                                     factor=1.66053886)

# ----- Angle -----
degree = Unit.create(radian.dim, name="degree", dispname="°", scale=radian.scale, factor=pi / 180)
arcmin = arcminute = Unit.create(radian.dim, name="arcminute", dispname="′", scale=radian.scale - 4,
                                 factor=2.908882086657216)
arcsec = arcsecond = Unit.create(radian.dim, name="arcsecond", dispname="″", scale=radian.scale - 6,
                                 factor=4.84813681109536)

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
angstrom = Unit.create(meter.dim, name="angstrom", dispname="Å", scale=meter.scale - 10, factor=1.0)
micron = Unit.create(meter.dim, name="micron", dispname="µm", scale=meter.scale - 6, factor=1.)
au = astronomical_unit = Unit.create(meter.dim, name="astronomical unit", dispname="AU", scale=meter.scale + 11,
                                     factor=1.495978707)
light_year = Unit.create(meter.dim, name="light year", dispname="ly", scale=meter.scale + 15, factor=9.460730777119564)
parsec = Unit.create(meter.dim, name="parsec", dispname="pc", scale=meter.scale + 16, factor=3.085677581491367)

# ----- Pressure -----
atm = atmosphere = Unit.create(pascal.dim, name="atmosphere", dispname="atm", scale=pascal.scale + 5, factor=1.013249966)
bar = Unit.create(pascal.dim, name="bar", dispname="bar", scale=pascal.scale + 5, factor=1.)
mmHg = torr = Unit.create(pascal.dim, name="torr", dispname="torr", scale=pascal.scale + 2, factor=1.3332236842105263)
psi = Unit.create(pascal.dim, name="pound per square inch", dispname="psi", scale=pascal.scale + 3,
                  factor=6.894757293168361)

# ----- Area -----
hectare = Unit.create(meter2.dim, name="hectare", dispname="ha", scale=meter2.scale + 4, factor=1.)
acre = Unit.create(meter2.dim, name="acre", dispname="acre", scale=meter2.scale + 3, factor=4.046864798)

# ----- Volume -----
gallon = gallon_US = Unit.create(meter3.dim, name="gallon", dispname="gal", scale=meter3.scale - 3, factor=3.785411784)
gallon_imp = Unit.create(meter3.dim, name="imperial gallon", dispname="gal", scale=meter3.scale - 3, factor=4.54609)
fluid_ounce = fluid_ounce_US = Unit.create(meter3.dim, name="fluid ounce", dispname="fl oz", scale=meter3.scale - 5,
                                           factor=2.95735295625)
fluid_ounce_imp = Unit.create(meter3.dim, name="imperial fluid ounce", dispname="fl oz imp", scale=meter3.scale - 5,
                              factor=2.84130742)
bbl = barrel = Unit.create(meter3.dim, name="barrel", dispname="bbl", scale=meter3.scale + 2, factor=1.58987294928)

# ----- Speed -----
speed_unit = meter / second
kmh = Unit.create(speed_unit.dim, name="kilometer per hour", dispname="km/h", scale=speed_unit.scale - 1,
                  factor=2.77777778)
mph = Unit.create(speed_unit.dim, name="mile per hour", dispname="mph", scale=speed_unit.scale - 1, factor=4.4704)
mach = speed_of_sound = Unit.create(speed_unit.dim, name="speed of sound", dispname="mach", scale=speed_unit.scale + 2,
                                    factor=3.4029)
knot = Unit.create(speed_unit.dim, name="knot", dispname="kn", scale=speed_unit.scale - 1, factor=5.14444444)

# ----- Temperature -----
# TODO: The relationship between Celsius and Kelvin should be linear, but the current implementation is not.
# zero_Celsius = Unit.create(kelvin.dim, name="zero Celsius", dispname="0°C", scale=kelvin.scale, factor=273.15)
degree_Fahrenheit = Unit.create(kelvin.dim, name="degree Fahrenheit", dispname="°F", scale=kelvin.scale,
                                factor=5/9)

# ----- Energy -----
eV = electron_volt = Unit.create(joule.dim, name="electronvolt", dispname="eV", scale=joule.scale - 19, factor=1.602176565)
calorie = calorie_th = Unit.create(joule.dim, name="calorie", dispname="cal", scale=joule.scale, factor=4.184)
calorie_IT = Unit.create(joule.dim, name="calorie (International Table)", dispname="cal IT", scale=joule.scale,
                         factor=4.1868)
erg = Unit.create(joule.dim, name="erg", dispname="erg", scale=joule.scale - 7, factor=1.)
Btu = Btu_IT = Unit.create(joule.dim, name="British thermal unit (International Table)", dispname="Btu IT",
                           scale=joule.scale + 3, factor=1.05505585262)
Btu_th = Unit.create(joule.dim, name="British thermal unit (thermochemical)", dispname="Btu th", scale=joule.scale + 3,
                     factor=1.05435026444)
ton_TNT = Unit.create(joule.dim, name="ton of TNT", dispname="ton TNT", scale=joule.scale + 9, factor=4.184)

# ----- Power -----
hp = horsepower = Unit.create(watt.dim, name="horsepower", dispname="hp", scale=watt.scale + 2,
                              factor=7.4569987158227022)
kcal_per_h = Unit.create(watt.dim, name="kcal per hour", dispname="kcal/h", scale=watt.scale,
                         factor=1.162222)

# ----- Force -----
dyn = dyne = Unit.create(newton.dim, name="dyne", dispname="dyn", scale=newton.scale - 5, factor=1.)
lbf = pound_force = Unit.create(newton.dim, name="pound force", dispname="lbf", scale=newton.scale,
                                factor=4.4482216152605)
kgf = kilogram_force = Unit.create(newton.dim, name="kilogram force", dispname="kgf", scale=newton.scale,
                                   factor=9.80665)

# UNITS in modular dynamics
# See https://github.com/chaobrain/brainunit/issues/63
# Intermolecular force 分子间作用力
IMF = Unit.create(newton.dim, name="intermolecular force", dispname="IMF", scale=newton.scale - 9, factor=1.602176565)

""""
References
==========

.. [CODATA2018] CODATA Recommended Values of the Fundamental
   Physical Constants 2018.

   https://physics.nist.gov/cuu/Constants/

"""
