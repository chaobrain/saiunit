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

    from brainunit import faraday_constant as F

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
)

__all__ = [
    'avogadro_constant',
    'boltzmann_constant',
    'electric_constant',
    'electron_mass',
    'elementary_charge',
    'faraday_constant',
    'gas_constant',
    'magnetic_constant',
    'molar_mass_constant',
    'zero_celsius',
]

#: Avogadro constant (http://physics.nist.gov/cgi-bin/cuu/Value?na)
avogadro_constant = np.asarray(6.022140857e23) / mole
#: Boltzmann constant (physics.nist.gov/cgi-bin/cuu/Value?k)
boltzmann_constant = np.asarray(1.38064852e-23) * (joule / kelvin)
#: electric constant (http://physics.nist.gov/cgi-bin/cuu/Value?ep0)
electric_constant = np.asarray(8.854187817e-12) * (farad / meter)
#: Electron rest mass (physics.nist.gov/cgi-bin/cuu/Value?me)
electron_mass = np.asarray(9.10938356e-31) * kilogram
#: Elementary charge (physics.nist.gov/cgi-bin/cuu/Value?e)
elementary_charge = np.asarray(1.6021766208e-19) * coulomb
#: Faraday constant (http://physics.nist.gov/cgi-bin/cuu/Value?f)
faraday_constant = np.asarray(96485.33289) * (coulomb / mole)
#: gas constant (http://physics.nist.gov/cgi-bin/cuu/Value?r)
gas_constant = np.asarray(8.3144598) * (joule / mole / kelvin)
#: Magnetic constant (http://physics.nist.gov/cgi-bin/cuu/Value?mu0)
magnetic_constant = np.asarray(4 * np.pi * 1e-7) * (newton / amp ** 2)
#: Molar mass constant (http://physics.nist.gov/cgi-bin/cuu/Value?mu)
molar_mass_constant = np.asarray(1.) * (gram / mole)
#: zero degree Celsius
zero_celsius = np.asarray(273.15) * kelvin
