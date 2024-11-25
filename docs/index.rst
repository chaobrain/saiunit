``brainunit`` documentation
===========================

`brainunit <https://github.com/chaoming0625/brainunit>`_ provides physical units and unit-aware mathematical system in JAX for brain dynamics and AI4Science.




----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainunit[cpu]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U brainunit[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainunit[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

----


Quick Start
^^^^^^^^^^^
Most users of the ``brainunit`` package will work with ``Quantity``: the combination of
a value and a unit. The most convenient way to create a ``Quantity`` is to multiply or
divide a value by one of the built-in units. It works with scalars, sequences,
and ``numpy`` or ``jax`` arrays.

.. code-block:: python

    import brainunit as u
    61.8 * u.second

.. code-block:: text

    61.8 * second


.. code-block:: python

    [1., 2., 3.] * u.second

.. code-block:: text

    ArrayImpl([1. 2. 3.]) * second


.. code-block:: python
    
    import numpy as np
    np.array([1., 2., 3.]) * u.second

.. code-block:: text
    
    ArrayImpl([1., 2., 3.]) * second


.. code-block:: python
    
    import jax.numpy as jnp
    jnp.array([1., 2., 3.]) * u.second

.. code-block:: text

    ArrayImpl([1., 2., 3.]) * second


You can get the unit and mantissa from a ``Quantity`` using the unit and mantissa members:

.. code-block:: python

    q = 61.8 * u.second
    q.mantissa

.. code-block:: text
    
    Array(61.8, dtype=float64, weak_type=True)


.. code-block:: python
    
    q.unit


.. code-block:: text

    second


You can also combine quantities or units:

.. code-block:: python

    15.1 * u.meter / (32.0 * u.second)

.. code-block:: text

    0.471875 * meter / second


.. code-block:: python

    3.0 * u.kmeter / (130.51 * u.meter / u.second)


.. code-block:: text
    
    0.022997 * (meter / second)

To create a dimensionless quantity, directly use the ``Quantity`` constructor:

.. code-block:: python
    
    q = u.Quantity(61.8)
    q.dim

.. code-block:: text
    
    Dimension()

----


See also the BDP ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^


We are building the `brain dynamics programming ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_.




.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Physical Units

   physical_units/quantity.ipynb
   physical_units/math_operations_with_quantity.ipynb
   physical_units/standard_units.ipynb
   physical_units/constants.ipynb
   physical_units/conversion.ipynb



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Unit-aware Math Functions

   mathematical_functions/array_creation.ipynb
   mathematical_functions/numpy_functions.ipynb
   mathematical_functions/elinstein_operations.ipynb
   mathematical_functions/customize_functions.ipynb


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Advanced Tutorials

   advanced_tutorials/combining_and_defining.ipynb
   advanced_tutorials/mechanism.ipynb



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Documentation

   apis/changelog.md
   apis/brainunit.rst
   apis/brainunit.autograd.rst
   apis/brainunit.math.rst
   apis/brainunit.linalg.rst
   apis/brainunit.lax.rst



