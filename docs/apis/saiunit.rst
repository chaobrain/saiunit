``saiunit`` module
====================

.. currentmodule:: saiunit
.. automodule:: saiunit



Data Structures
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    Quantity
    Unit
    Dimension


Errors
------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    UnitMismatchError
    DimensionMismatchError


Constants
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    DIMENSIONLESS
    UNITLESS


Getters and Checkers
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    is_dimensionless
    is_unitless
    is_scalar_type
    get_dim
    get_unit
    get_mantissa
    get_magnitude
    display_in_unit
    split_mantissa_unit
    maybe_decimal
    fail_for_dimension_mismatch
    fail_for_unit_mismatch
    assert_quantity
    have_same_dim
    has_same_unit
    unit_scale_align_to_first
    array_with_unit


Decorators
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    check_dims
    check_units
    assign_units


Dimension Utilities
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    get_or_create_dimension
    get_dim_for_display
    add_standard_unit


Temperature Conversion
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    celsius2kelvin
    kelvin2celsius


Customized Array Interfaces
---------------------------

Mathematical functions provided in ``saiunit`` are aware of :class:`CustomArray`.
You can define your own array-like class by inheriting from :class:`CustomArray` and implementing the required methods.
These customized arrays can seamlessly interact with the unit-aware mathematical functions.


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    CustomArray


Equinox Compatibility
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    compatible_with_equinox
