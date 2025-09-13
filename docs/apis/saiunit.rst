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


Customized Array Interfaces
---------------------------

Mathmatical functions provided in ``saiunit`` takes aware of :class:`CustomArray`.
You can define your own array-like class by inheriting from :class:`CustomArray` and implementing the required methods.
These customized arrays can seamlessly interact with the unit-aware mathematical functions.


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    CustomArray

