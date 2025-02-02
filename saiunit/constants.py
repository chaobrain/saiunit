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

r"""
A module providing some physical units as `Quantity` objects. Note that these
units are not imported by wildcard imports, they
have to be imported explicitly. You can use ``import ... as ...`` to import them
with shorter names, e.g.::

    from saiunit.constants import faraday_constant as F

The available constants are:

==================== ================== ======================= ==================================================================
Constant             Symbol(s)          name                    Value
==================== ================== ======================= ==================================================================
Avogadro constant    :math:`N_A, L`     ``avogadro_constant``   :math:`6.022140857\times 10^{23}\,\mathrm{mol}^{-1}`
Boltzmann constant   :math:`k`          ``boltzmann_constant``  :math:`1.38064852\times 10^{-23}\,\mathrm{J}\,\mathrm{K}^{-1}`
Electric constant    :math:`\epsilon_0` ``electric_constant``   :math:`8.854187817\times 10^{-12}\,\mathrm{F}\,\mathrm{m}^{-1}`
Electron mass        :math:`m_e`        ``electron_mass``       :math:`9.10938356\times 10^{-31}\,\mathrm{kg}`
Elementary charge    :math:`e`          ``elementary_charge``   :math:`1.6021766208\times 10^{-19}\,\mathrm{C}`
Faraday constant     :math:`F`          ``faraday_constant``    :math:`96485.33289\,\mathrm{C}\,\mathrm{mol}^{-1}`
Gas constant         :math:`R`          ``gas_constant``        :math:`8.3144598\,\mathrm{J}\,\mathrm{mol}^{-1}\,\mathrm{K}^{-1}`
Magnetic constant    :math:`\mu_0`      ``magnetic_constant``   :math:`12.566370614\times 10^{-7}\,\mathrm{N}\,\mathrm{A}^{-2}`
Molar mass constant  :math:`M_u`        ``molar_mass_constant`` :math:`1\times 10^{-3}\,\mathrm{kg}\,\mathrm{mol}^{-1}`
0°C                                     ``zero_celsius``        :math:`273.15\,\mathrm{K}`
==================== ================== ======================= ==================================================================
"""

import numpy as np

from ._unit_common import (
    amp,
    coulomb,
    farad,
    gram,
    joule,
    kelvin,
    kilogram,
    meter,
    mole,
    newton,
    radian,
    second,
    meter2,
    meter3,
    watt,
)
from ._unit_constants import speed_unit

__all__ = [
    'arcmin', 'arcminute', 'arcsec', 'arcsecond', 'atomic_mass', 'au', 'astronomical_unit',
    'angstrom', 'atm', 'atmosphere', 'avogadro', 'bar', 'blob', 'boltzmann', 'Btu', 'Btu_IT',
    'Btu_th', 'carat', 'calorie', 'calorie_IT', 'calorie_th', 'day', 'degree', 'degree_Fahrenheit',
    'dyn', 'dyne', 'eV', 'electron_mass', 'electric', 'electronvolt', 'elementary_charge', 'erg',
    'faraday', 'fermi', 'fluid_ounce', 'fluid_ounce_US', 'fluid_ounce_imp', 'foot', 'gas', 'grain',
    'gallon', 'gallon_US', 'gallon_imp', 'gram', 'hectare', 'hour', 'hp', 'horsepower', 'IMF',
    'inch', 'julian_year', 'kelvin', 'kgf', 'kilogram_force', 'knot', 'lb', 'lbf', 'light_year',
    'long_ton', 'mach', 'magnetic', 'meter', 'metric_ton', 'micron', 'mil', 'mile', 'minute',
    'molar_mass', 'month', 'mph', 'nautical_mile', 'newton', 'ounce', 'oz', 'parsec', 'pica',
    'point', 'pound', 'psi', 'radian', 'second', 'short_ton', 'slug', 'slinch', 'speed_unit',
    'stone', 'survey_foot', 'survey_mile', 'torr', 'troy_ounce', 'troy_pound', 'ton_TNT', 'week',
    'watt', 'year', 'zero_celsius'
]

#: Avogadro constant (http://physics.nist.gov/cgi-bin/cuu/Value?na)
avogadro = np.asarray(6.022140857e23) / mole
#: Boltzmann constant (physics.nist.gov/cgi-bin/cuu/Value?k)
boltzmann = np.asarray(1.38064852e-23) * (joule / kelvin)
#: electric constant (http://physics.nist.gov/cgi-bin/cuu/Value?ep0)
electric = np.asarray(8.854187817e-12) * (farad / meter)
#: Electron rest mass (physics.nist.gov/cgi-bin/cuu/Value?me)
electron_mass = np.asarray(9.10938356e-31) * kilogram
#: Elementary charge (physics.nist.gov/cgi-bin/cuu/Value?e)
elementary_charge = np.asarray(1.6021766208e-19) * coulomb
#: Faraday constant (http://physics.nist.gov/cgi-bin/cuu/Value?f)
faraday = np.asarray(96485.33289) * (coulomb / mole)
#: gas constant (http://physics.nist.gov/cgi-bin/cuu/Value?r)
gas = np.asarray(8.3144598) * (joule / mole / kelvin)
#: Magnetic constant (http://physics.nist.gov/cgi-bin/cuu/Value?mu0)
magnetic = np.asarray(4 * np.pi * 1e-7) * (newton / amp ** 2)
#: Molar mass constant (http://physics.nist.gov/cgi-bin/cuu/Value?mu)
molar_mass = np.asarray(1.) * (gram / mole)
#: zero degree Celsius
zero_celsius = np.asarray(273.15) * kelvin

# ----- Mass -----
metric_ton = np.asarray(1e3) * kilogram  # Metric ton
grain = np.asarray(6.479891e-5) * kilogram  # Grain
lb = pound = np.asarray(0.45359237) * kilogram  # Pound
slinch = blob = np.asarray(1.75126836e2) * kilogram  # Blob (slug-inch)
slug = np.asarray(1.459390294e1) * kilogram  # Slug
oz = ounce = np.asarray(2.8349523125e-2) * kilogram  # Ounce
stone = np.asarray(6.35029318) * kilogram  # Stone
long_ton = np.asarray(1.0160469088e3) * kilogram  # Long ton
short_ton = np.asarray(0.90718474e3) * kilogram  # Short ton
troy_ounce = np.asarray(3.11034768e-2) * kilogram  # Troy ounce
troy_pound = np.asarray(0.3732417216) * kilogram  # Troy pound
carat = np.asarray(2e-4) * kilogram  # Carat
atomic_mass = np.asarray(1.66053886e-27) * kilogram  # Atomic mass unit (amu)

# ----- Angle -----
degree = np.asarray(np.pi / 180) * radian  # Degree
arcmin = arcminute = np.asarray(np.pi / (180 * 60)) * radian  # Arcminute
arcsec = arcsecond = np.asarray(np.pi / (180 * 3600)) * radian  # Arcsecond

# ----- Time -----
minute = np.asarray(60) * second  # Minute
hour = np.asarray(3600) * second  # Hour
day = np.asarray(86400) * second  # Day
week = np.asarray(604800) * second  # Week
month = np.asarray(2.629746e6) * second  # Month (approx.)
year = np.asarray(3.1556952e7) * second  # Year (approx.)
julian_year = np.asarray(3.15576e7) * second  # Julian year

# ----- Length -----
inch = np.asarray(0.0254) * meter  # Inch
foot = np.asarray(0.3048) * meter  # Foot
yard = np.asarray(0.9144) * meter  # Yard
mile = np.asarray(1609.344) * meter  # Mile
mil = np.asarray(2.54e-5) * meter  # Mil
point = np.asarray(3.5277777777777776e-4) * meter  # Point
pica = np.asarray(4.233333333333333e-3) * meter  # Pica
survey_foot = np.asarray(0.3048006096012192) * meter  # Survey foot
survey_mile = np.asarray(1609.3472186944374) * meter  # Survey mile
nautical_mile = np.asarray(1852) * meter  # Nautical mile
fermi = np.asarray(1e-15) * meter  # Fermi
angstrom = np.asarray(1e-10) * meter  # Ångstrom
micron = np.asarray(1e-6) * meter  # Micron
au = astronomical_unit = np.asarray(1.495978707e11) * meter  # Astronomical unit
light_year = np.asarray(9.460730777119564e15) * meter  # Light year
parsec = np.asarray(3.085677581491367e16) * meter  # Parsec

# ----- Pressure -----
atm = atmosphere = np.asarray(1.013249966e5) * (newton / meter2)  # Atmosphere
bar = np.asarray(1e5) * (newton / meter2)  # Bar
mmHg = torr = np.asarray(1.3332236842105263e2) * (newton / meter2)  # Torr (mmHg)
psi = np.asarray(6.894757293168361e3) * (newton / meter2)  # Pound per square inch (psi)

# ----- Area -----
hectare = np.asarray(1e4) * meter2  # Hectare
acre = np.asarray(4046.864798) * meter2  # Acre

# ----- Volume -----
gallon = gallon_US = np.asarray(3.785411784e-3) * meter3  # Gallon (US)
gallon_imp = np.asarray(4.54609e-3) * meter3  # Imperial gallon
fluid_ounce = fluid_ounce_US = np.asarray(2.95735295625e-5) * meter3  # Fluid ounce (US)
fluid_ounce_imp = np.asarray(2.84130742e-5) * meter3  # Imperial fluid ounce
bbl = barrel = np.asarray(1.58987294928e2) * meter3  # Barrel (oil)

# ----- Temperature -----
degree_Fahrenheit = np.asarray(2.55927778e2) * kelvin  # Fahrenheit

# ----- Speed -----
kmh = np.asarray(2.77777778e-1) * speed_unit  # Kilometer per hour
mph = np.asarray(4.4704e-1) * speed_unit  # Mile per hour
knot = np.asarray(5.14444444e-1) * speed_unit  # Knot
mach = np.asarray(3.4029e2) * speed_unit  # Mach

# ----- Energy -----
eV = electronvolt = np.asarray(1.6021766208e-19) * joule  # Electronvolt
calorie = calorie_th = np.asarray(4.184) * joule  # Calorie (thermochemical)
calorie_IT = np.asarray(4.1868) * joule  # Calorie (International Table)
erg = np.asarray(1e-7) * joule  # Erg
Btu = Btu_IT = np.asarray(1.05505585262e3) * joule  # British thermal unit (International Table)
Btu_th = np.asarray(1.05435026444e3) * joule  # British thermal unit (thermochemical)
ton_TNT = np.asarray(4.184e9) * joule  # Ton of TNT

# ----- Power -----
hp = horsepower = np.asarray(7.4569987158227022e2) * watt  # Horsepower

# ----- Force -----
dyn = dyne = np.asarray(1e-5) * newton  # Dyne
lbf = pound_force = np.asarray(4.4482216152605) * newton  # Pound-force
kgf = kilogram_force = np.asarray(9.80665) * newton  # Kilogram-force
IMF = np.asarray(1.602176565e-9) * newton  # Intermolecular force
