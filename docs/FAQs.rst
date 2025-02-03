FAQs
====

How `saiunit` is different from other physical unit libraries, such as `Quantities` and `Pint`?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In scientific computing, we already have excellent physical unit
libraries, such as ``Pint`` and ``Quantities`` in Python, ``Unitful.jl``
in Julia, ``Boost.Units`` in C++, and ``uom`` in Rust. These libraries
enable seamless operations, conversions, and calculations among physical
units, greatly simplifying the processing and analysis of physical
quantities for researchers.

However, existing unit libraries often face significant limitations when
dealing with high-performance AI-driven scientific computing.

Firstly, they lack compatibility with mainstream HPC frameworks like
PyTorch [Paszke et al., 2019], TensorFlow [Abadi et al., 2016], and JAX
[JAX, 2018], hindering their ability to meet the demands of complex
scientific computations in parallel computing hardware such as GPUs and
TPUs.

Secondly, they generally employ the float64 data type for numerical
representation. In contrast, AI for science applications, particularly
those using neural networks, often require lower-precision data types to
reduce memory usage and improve computational efficiency. This
discrepancy may lead to unnecessary performance overhead and limit
potential applications in resource-constrained scenarios.

Thirdly, these unit libraries typically lack support for flexible
scaling, a crucial feature when dealing with physical quantities
exhibiting extreme scale differences. In certain scientific research
domains, improper handling of such units may result in precision loss,
potentially compromising the accuracy and reliability of research
outcomes.

Finally, a critical limitation of current physical unit systems is their
lack of support for automatic differentiation (AD), which is fundamental
to modern AI models [Baydin et al., 2018]. This deficiency makes
high-order differentiable optimization particularly challenging,
limiting the integration of physical units into modern AI-driven
research and development pipelines.

``SAIUnit`` is previously known as BrainUnit, for brain science. How is it different from units system in previous brain simulators, such as NEURON?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NEURON units are virtual physical units. The implementation details are
as follows (copied from
https://github.com/neuronsimulator/nrn/blob/master/share/lib/python/neuron/units.py):

.. code:: python

   """ unit definitions in NEURON simulator"""

   # NEURON's default units
   mM = 1
   ms = 1
   mV = 1
   µm = um = 1

   # concentration
   µM = uM = 1e-3 * mM
   nM = 1e-6 * mM
   M = 1e3 * mM

   # time
   µs = us = 1e-3 * ms
   s = sec = 1e3 * ms
   minute = 60 * sec
   hour = 60 * minute
   day = 24 * hour

   # space
   nm = 1e-3 * um
   mm = 1000 * um
   cm = 10 * mm
   m = 100 * cm

   # voltage
   μV = uV = 1e-3 * mV
   V = 1e3 * mV

In other words, NEURON defines four basic physical units—millivolts
(mV), milliseconds (ms), millimoles (mM), and micrometers (µm)—as
dimensionless with a value of 1. Other physical units are scaled
relative to these base units. This approach, using commonly used scales
as the base for scaling (mV, mM, ms, µm), simplifies calculations for
frequently used units and ensures consistency in unit conversions.

However, compared to ``saiunit``, which provides a fully functional unit
system with dimensional analysis and checking, NEURON’s virtual unit
system has several notable limitations:

1. **Lack of dimensional analysis support**: ``saiunit`` can verify
   whether units comply with physical laws, such as velocity units must
   be ``length/time`` and force units must be ``mass * acceleration``.
   In NEURON’s virtual unit system, relationships between units are
   hardcoded without automatic dimensional consistency checks. This may
   lead to users inadvertently mixing units of different dimensions
   (e.g., using ``mV`` instead of ``mA``) without any warnings or error
   messages.

2. **Absence of unit derivation**: ``saiunit`` can perform unit
   derivation and calculations (e.g., ``m/s`` * ``kg`` results in
   ``kg·m/s``) and automatically handle composite units and unit
   combinations. In contrast, NEURON’s virtual unit system only supports
   its predefined units and lacks the capability to derive composite
   units. When dealing with more complex physical quantities (such as
   momentum, current density, etc.), NEURON’s unit system may not
   provide the same convenient calculation or verification capabilities
   as ``saiunit``.

3. **Manual unit management**: Due to the absence of a true physical
   unit system in NEURON, most dimensional scaling and unit conversions
   must be performed manually when designing complex calculations. For
   example, voltage conversions from ``mV`` to ``V``, and conductance
   conversions from ``C/m³`` to ``C/cm³`` require manual handling.
   If any step in the conversion process
   contains errors (e.g., incorrect unit scaling factors), the final
   simulation results will be affected. This manual conversion process
   increases the probability of errors.

In summary, while NEURON’s virtual unit system simplifies basic unit
conversions and enhances code readability, it lacks the robust
capabilities found in true unit libraries like ``saiunit``, which
provide dimensional analysis and unit derivation functionality. This
virtual unit system may even create confusion and ambiguity, as
demonstrated in their examples where variables with units can be
directly used with dimensionless abstract values
(https://nrn.readthedocs.io/en/8.2.6/tutorials/scripting-neuron-basics.html#Step-5:-Insert-a-stimulus).
Therefore, for applications requiring precise unit management, automatic
dimensional checking, complex unit derivation, and extensibility,
``saiunit`` is undoubtedly more powerful and flexible, effectively
preventing errors and improving code maintainability.


What are the system requirements for ``saiunit``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``saiunit`` can be installed on Windows, macOS, and Linux operating
systems. The following software and hardware requirements are
recommended for optimal performance:

-  ``jax>=0.4.30``
-  ``python>=3.9``

How do I set up ``saiunit`` for GPU or TPU usage?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hardware acceleration of ``saiunit`` is primarily supported by the
``JAX`` library, which provides GPU and TPU support for high-performance
computing. To enable GPU or TPU usage in ``saiunit``, you need to
install the appropriate version of ``JAX`` that supports your hardware
configuration. Please refer to the `JAX installation
guide <https://jax.readthedocs.io/en/latest/installation.html>`__.

How do I create a Quantity in ``saiunit``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please refer to the documentation on
https://brainunit.readthedocs.io/en/latest/physical_units/quantity.html.
