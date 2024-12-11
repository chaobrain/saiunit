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


class TestConstant(unittest.TestCase):

    def test_constants(self):
        import brainunit.constants as constants

        # Check that the expected names exist and have the correct dimensions
        assert constants.avogadro_constant.dim == (1 / mole).dim
        assert constants.boltzmann_constant.dim == (joule / kelvin).dim
        assert constants.electric_constant.dim == (farad / meter).dim
        assert constants.electron_mass.dim == kilogram.dim
        assert constants.elementary_charge.dim == coulomb.dim
        assert constants.faraday_constant.dim == (coulomb / mole).dim
        assert constants.gas_constant.dim == (joule / mole / kelvin).dim
        assert constants.magnetic_constant.dim == (newton / amp2).dim
        assert constants.molar_mass_constant.dim == (kilogram / mole).dim
        assert constants.zero_celsius.dim == kelvin.dim

        # Check the consistency between a few constants
        assert u.math.allclose(
            constants.gas_constant.mantissa,
            (constants.avogadro_constant * constants.boltzmann_constant).mantissa,
        )
        assert u.math.allclose(
            constants.faraday_constant.mantissa,
            (constants.avogadro_constant * constants.elementary_charge).mantissa,
        )
