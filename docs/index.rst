``saiunit`` documentation
===========================

`saiunit <https://github.com/chaobrain/saiunit>`_ provides physical units and unit-aware mathematical system in JAX for general AI-driven scientific computing.

The core features of `saiunit` include:

- Integration of over 2,000 commonly used physical units and constants
- Implementation of more than 500 unit-aware mathematical functions
- Deep integration with JAX, providing comprehensive support for modern AI framework features including automatic differentiation (autograd), just-in-time compilation (JIT), vectorization, and parallel computation
- Unit conversion and analysis are performed at compilation time, resulting in zero runtime overhead
- Strict physical unit type checking and dimensional inference system, detecting unit inconsistencies during compilation




Compared to existing unit libraries, such as `Quantities <https://github.com/python-quantities/python-quantities>`_ and `Pint <https://github.com/hgrecco/pint>`_ , saiunit introduces a rigorous physical unit system specifically designed to support AI computations (e.g., automatic differentiation, just-in-time compilation, and parallelization).




----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U saiunit[cpu]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U saiunit[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U saiunit[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

----


Quick Start
^^^^^^^^^^^
Most users of the ``saiunit`` package will work with ``Quantity``: the combination of
a value and a unit. The most convenient way to create a ``Quantity`` is to multiply or
divide a value by one of the built-in units. It works with scalars, sequences,
and ``numpy`` or ``jax`` arrays.

.. code-block:: python

    import saiunit as u
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
   mathematical_functions/einstein_operations.ipynb
   mathematical_functions/linalg_functions.ipynb
   mathematical_functions/fft_functions.ipynb
   mathematical_functions/lax_functions.ipynb
   mathematical_functions/check_units.ipynb
   mathematical_functions/assign_units.ipynb


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Advanced Tutorials

   advanced_tutorials/combining_and_defining.ipynb
   advanced_tutorials/mechanism.ipynb
   advanced_tutorials/FAQs.md



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Documentation

   apis/changelog.md
   apis/saiunit.rst
   apis/saiunit.autograd.rst
   apis/saiunit.math.rst
   apis/saiunit.linalg.rst
   apis/saiunit.lax.rst
   apis/saiunit.fft.rst
   apis/saiunit.sparse.rst
   apis/saiunit.constants.rst



