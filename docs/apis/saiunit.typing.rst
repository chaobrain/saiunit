``saiunit.typing`` module
=========================

.. currentmodule:: saiunit.typing
.. automodule:: saiunit.typing


Physical Type Utilities
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    PhysicalType
    is_physical_type
    quantity_type


Core Type Aliases
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    QuantityLike
    UnitLike
    DimensionLike


Pre-built Physical-Type Aliases
-------------------------------

.. data:: HAS_UNIT

   Alias for :class:`~saiunit.Quantity`. Use as a base type annotation for
   any unit-bearing value, regardless of physical dimension.

.. autodata:: DIMENSIONLESS_TYPE
   :no-value:
.. autodata:: LENGTH
   :no-value:
.. autodata:: MASS
   :no-value:
.. autodata:: TIME
   :no-value:
.. autodata:: CURRENT
   :no-value:
.. autodata:: TEMPERATURE
   :no-value:
.. autodata:: SUBSTANCE
   :no-value:
.. autodata:: LUMINOSITY
   :no-value:
.. autodata:: FREQUENCY
   :no-value:
.. autodata:: FORCE
   :no-value:
.. autodata:: ENERGY
   :no-value:
.. autodata:: POWER
   :no-value:
.. autodata:: PRESSURE
   :no-value:
.. autodata:: CHARGE
   :no-value:
.. autodata:: VOLTAGE
   :no-value:
.. autodata:: RESISTANCE
   :no-value:
.. autodata:: CAPACITANCE
   :no-value:
.. autodata:: CONDUCTANCE
   :no-value:
.. autodata:: MAGNETIC_FLUX
   :no-value:
.. autodata:: MAGNETIC_FIELD
   :no-value:
.. autodata:: INDUCTANCE
   :no-value:
.. autodata:: SPEED
   :no-value:
.. autodata:: ACCELERATION
   :no-value:
.. autodata:: AREA
   :no-value:
.. autodata:: VOLUME
   :no-value:
.. autodata:: DENSITY
   :no-value:


Runtime Validation
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    validate_units
