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

import unittest

import brainunit as u
from brainunit._unit_common import *

constants_list = [
    # Mass
    'metric_ton', 'grain', 'pound', 'slinch', 'slug', 'ounce', 'stone', 'long_ton', 'short_ton', 'troy_ounce',
    'troy_pound', 'carat', 'atomic_mass',
    # Angle
    'degree', 'arcmin', 'arcsec',
    # Time
    'minute', 'hour', 'day', 'week', 'month', 'year', 'julian_year',
    # Length
    'inch', 'foot', 'yard', 'mile', 'mil', 'point', 'pica', 'survey_foot', 'survey_mile', 'nautical_mile', 'fermi',
    'angstrom', 'micron', 'au', 'light_year',
    # Pressure
    'atm', 'bar', 'mmHg', 'psi',
    # Area
    'hectare', 'acre',
    # Volume
    'gallon', 'gallon_imp', 'fluid_ounce', 'fluid_ounce_imp', 'bbl',
    # Speed
    'kmh', 'mph', 'knot', 'mach',
    # Temperature
    'degree_Fahrenheit',
    # Energy
    'eV', 'calorie', 'calorie_IT', 'erg', 'Btu', 'Btu_IT', 'ton_TNT',
    # Power
    'hp',
    # Force
    'dyn', 'lbf', 'kgf', 'IMF',
]


class TestConstant(unittest.TestCase):

    def test_constants(self):
        import brainunit.constants as constants

        # Check that the expected names exist and have the correct dimensions
        assert constants.avogadro.dim == (1 / mole).dim
        assert constants.boltzmann.dim == (joule / kelvin).dim
        assert constants.electric.dim == (farad / meter).dim
        assert constants.electron_mass.dim == kilogram.dim
        assert constants.elementary_charge.dim == coulomb.dim
        assert constants.faraday.dim == (coulomb / mole).dim
        assert constants.gas.dim == (joule / mole / kelvin).dim
        assert constants.magnetic.dim == (newton / amp2).dim
        assert constants.molar_mass.dim == (kilogram / mole).dim
        assert constants.zero_celsius.dim == kelvin.dim

        # Check the consistency between a few constants
        assert u.math.allclose(
            constants.gas.mantissa,
            (constants.avogadro * constants.boltzmann).mantissa,
        )
        assert u.math.allclose(
            constants.faraday.mantissa,
            (constants.avogadro * constants.elementary_charge).mantissa,
        )

    def test_quantity_constants_and_unit_constants(self):
        import brainunit.constants as quantity_constants
        import brainunit._unit_constants as unit_constants
        for c in constants_list:
            q_c = getattr(quantity_constants, c)
            u_c = getattr(unit_constants, c)
            assert u.math.isclose(
                q_c.to_decimal(q_c.unit),
                (1. * u_c).to_decimal(q_c.unit)
            ), f"Mismatch between {c} in quantity_constants and unit_constants"
