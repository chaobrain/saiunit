======================================
``saiunit.constants`` module
======================================

.. currentmodule:: saiunit.constants

Physical and mathematical constants and units.

Physical constants
==================

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


Mass
----

=================  ============================================================
``gram``           :math:`10^{-3}` kg
``metric_ton``     :math:`10^{3}` kg
``grain``          one grain in kg
``lb``             one pound (avoirdupous) in kg
``pound``          one pound (avoirdupous) in kg
``blob``           one inch version of a slug in kg (added in 1.0.0)
``slinch``         one inch version of a slug in kg (added in 1.0.0)
``slug``           one slug in kg (added in 1.0.0)
``oz``             one ounce in kg
``ounce``          one ounce in kg
``stone``          one stone in kg
``grain``          one grain in kg
``long_ton``       one long ton in kg
``short_ton``      one short ton in kg
``troy_ounce``     one Troy ounce in kg
``troy_pound``     one Troy pound in kg
``carat``          one carat in kg
``m_u``            atomic mass constant (in kg)
``u``              atomic mass constant (in kg)
``atomic_mass``    atomic mass constant (in kg)
=================  ============================================================

Angle
-----

=================  ============================================================
``degree``         degree in radians
``arcmin``         arc minute in radians
``arcminute``      arc minute in radians
``arcsec``         arc second in radians
``arcsecond``      arc second in radians
=================  ============================================================


Time
----

=================  ============================================================
``minute``         one minute in seconds
``hour``           one hour in seconds
``day``            one day in seconds
``week``           one week in seconds
``year``           one year (365 days) in seconds
``Julian_year``    one Julian year (365.25 days) in seconds
=================  ============================================================


Length
------

=====================  ============================================================
``inch``               one inch in meters
``foot``               one foot in meters
``yard``               one yard in meters
``mile``               one mile in meters
``mil``                one mil in meters
``pt``                 one point in meters
``point``              one point in meters
``survey_foot``        one survey foot in meters
``survey_mile``        one survey mile in meters
``nautical_mile``      one nautical mile in meters
``fermi``              one Fermi in meters
``angstrom``           one Angstrom in meters
``micron``             one micron in meters
``au``                 one astronomical unit in meters
``astronomical_unit``  one astronomical unit in meters
``light_year``         one light year in meters
``parsec``             one parsec in meters
=====================  ============================================================

Pressure
--------

=================  ============================================================
``atm``            standard atmosphere in pascals
``atmosphere``     standard atmosphere in pascals
``bar``            one bar in pascals
``torr``           one torr (mmHg) in pascals
``mmHg``           one torr (mmHg) in pascals
``psi``            one psi in pascals
=================  ============================================================

Area
----

=================  ============================================================
``hectare``        one hectare in square meters
``acre``           one acre in square meters
=================  ============================================================


Volume
------

===================    ========================================================
``liter``              one liter in cubic meters
``litre``              one liter in cubic meters
``gallon``             one gallon (US) in cubic meters
``gallon_US``          one gallon (US) in cubic meters
``gallon_imp``         one gallon (UK) in cubic meters
``fluid_ounce``        one fluid ounce (US) in cubic meters
``fluid_ounce_US``     one fluid ounce (US) in cubic meters
``fluid_ounce_imp``    one fluid ounce (UK) in cubic meters
``bbl``                one barrel in cubic meters
``barrel``             one barrel in cubic meters
===================    ========================================================

Speed
-----

==================    ==========================================================
``kmh``               kilometers per hour in meters per second
``mph``               miles per hour in meters per second
``mach``              one Mach (approx., at 15 C, 1 atm) in meters per second
``speed_of_sound``    one Mach (approx., at 15 C, 1 atm) in meters per second
``knot``              one knot in meters per second
==================    ==========================================================


Temperature
-----------

=====================  =======================================================
``zero_Celsius``       zero of Celsius scale in Kelvin
``degree_Fahrenheit``  one Fahrenheit (only differences) in Kelvins
=====================  =======================================================

.. autosummary::
   :toctree: generated/

   convert_temperature

Energy
------

====================  =======================================================
``eV``                one electron volt in Joules
``electron_volt``     one electron volt in Joules
``calorie``           one calorie (thermochemical) in Joules
``calorie_th``        one calorie (thermochemical) in Joules
``calorie_IT``        one calorie (International Steam Table calorie, 1956) in Joules
``erg``               one erg in Joules
``Btu``               one British thermal unit (International Steam Table) in Joules
``Btu_IT``            one British thermal unit (International Steam Table) in Joules
``Btu_th``            one British thermal unit (thermochemical) in Joules
``ton_TNT``           one ton of TNT in Joules
====================  =======================================================

Power
-----

====================  =======================================================
``hp``                one horsepower in watts
``horsepower``        one horsepower in watts
====================  =======================================================

Force
-----

====================  =======================================================
``dyn``               one dyne in newtons
``dyne``              one dyne in newtons
``lbf``               one pound force in newtons
``pound_force``       one pound force in newtons
``kgf``               one kilogram force in newtons
``kilogram_force``    one kilogram force in newtons
====================  =======================================================

References
==========

.. [CODATA2018] CODATA Recommended Values of the Fundamental
   Physical Constants 2018.

   https://physics.nist.gov/cuu/Constants/
