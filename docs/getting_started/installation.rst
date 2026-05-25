Installation
============

``saiunit`` is distributed on PyPI and supports Linux, macOS, and
Windows. The core package depends only on NumPy; every array backend
beyond NumPy — JAX (CPU, CUDA, TPU), CuPy, PyTorch, Dask, ndonnx — is
an optional extra that you opt into through ``pip``.

This page walks through:

- :ref:`install-requirements`
- :ref:`install-quick`
- :ref:`install-options` — backend-by-backend install matrix
- :ref:`install-combining` — combining extras safely
- :ref:`install-source` — editable / source install
- :ref:`install-verify` — confirming the install works
- :ref:`install-upgrade` — upgrading and uninstalling
- :ref:`install-troubleshoot` — common issues
- :ref:`install-ecosystem` — installing the wider BrainX stack


.. _install-requirements:

Requirements
------------

============  ================================================================
Python        ≥ 3.10 (officially tested on 3.10, 3.11, 3.12, 3.13, 3.14)
OS            Linux, macOS, Windows
Core deps     ``numpy``, ``typing_extensions``, ``array_api_compat ≥ 1.9``,
              ``opt_einsum`` — installed automatically with ``saiunit``
Optional      JAX, CuPy, PyTorch, Dask, ndonnx — install any combination
              via extras (see below)
============  ================================================================

The NumPy backend is always available. Without JAX, the JAX-only
submodules — ``saiunit.autograd``, ``saiunit.lax``, ``saiunit.sparse``
— and the custom ``exprel`` primitive raise
:class:`saiunit.BackendError` on access, with the install hint included
in the message. The CuPy, PyTorch, Dask, and ndonnx extras are
independent of JAX and of each other; mix and match freely.

.. tip::

   We strongly recommend installing into a fresh virtual environment to
   avoid clashing with system NumPy / JAX wheels::

      python -m venv .venv
      source .venv/bin/activate          # Windows: .venv\Scripts\activate
      pip install -U pip


.. _install-quick:

Quick install
-------------

Pick the line that matches how you plan to use ``saiunit``:

.. code-block:: bash

   pip install -U saiunit               # NumPy backend only
   pip install -U saiunit[cpu]          # + JAX (CPU build)
   pip install -U "saiunit[cuda12]"     # + JAX (CUDA 12 build)
   pip install -U "saiunit[cuda13]"     # + JAX (CUDA 13 build)
   pip install -U "saiunit[tpu]"        # + JAX (TPU build)
   pip install -U "saiunit[all]"        # + cupy, torch, dask, ndonnx (no JAX accelerator pin)

Already know what you need? Skip ahead to :ref:`install-verify`.


.. _install-options:

Installation options
--------------------

The table below summarises every supported extra. Each tab below
explains *why* you might pick that option and what it enables.

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Extra
     - Adds
     - When to choose it
   * - *(none)*
     - core only
     - Lightweight; NumPy backend; scipy / pandas / sklearn interop
   * - ``[jax]``
     - ``jax``
     - You manage your own JAX accelerator wheel
   * - ``[cpu]``
     - ``jax[cpu]``
     - Pinned JAX CPU wheel — autograd, JIT, vmap
   * - ``[cuda12]``
     - ``jax[cuda12]``
     - NVIDIA GPU, CUDA 12 toolkit
   * - ``[cuda13]``
     - ``jax[cuda13]``
     - NVIDIA GPU, CUDA 13 toolkit
   * - ``[tpu]``
     - ``jax[tpu]``
     - Google Cloud TPU
   * - ``[cupy]``
     - ``cupy-cuda12x ≥ 13``
     - NVIDIA GPU arrays as a drop-in NumPy replacement
   * - ``[torch]``
     - ``torch ≥ 2.0``
     - Integrate with PyTorch models / autograd
   * - ``[dask]``
     - ``dask[array] ≥ 2024.1``
     - Out-of-core, parallel, or lazy compute
   * - ``[ndonnx]``
     - ``ndonnx ≥ 0.9``
     - Symbolic graph building / ONNX export
   * - ``[all]``
     - ``cupy + torch + dask + ndonnx`` (and ``jax``)
     - Everything except a pinned JAX accelerator build

NumPy only
~~~~~~~~~~

Smallest install. ``saiunit.math``, ``saiunit.linalg``, and
``saiunit.fft`` work on the NumPy backend; the JAX-only submodules
raise :class:`saiunit.BackendError` on access.

.. code-block:: bash

   pip install -U saiunit

JAX (CPU)
~~~~~~~~~

Adds the pinned JAX CPU build. Enables ``saiunit.autograd``,
``saiunit.lax``, ``saiunit.sparse``, and ``jit`` / ``vmap`` / ``pmap``
over quantities.

.. code-block:: bash

   pip install -U "saiunit[cpu]"

JAX (GPU / CUDA)
~~~~~~~~~~~~~~~~

Adds the JAX CUDA build. Match the line to your CUDA toolkit; the two
CUDA extras are mutually exclusive.

.. code-block:: bash

   pip install -U "saiunit[cuda12]"   # CUDA 12.x
   pip install -U "saiunit[cuda13]"   # CUDA 13.x

The CUDA wheels are large (~1 GB). If you already manage CUDA outside
``pip``, prefer ``saiunit[jax]`` and install JAX yourself per the
`JAX install guide <https://docs.jax.dev/en/latest/installation.html>`_.

JAX (TPU)
~~~~~~~~~

Adds the JAX TPU build. Run on a `Google Cloud TPU VM
<https://cloud.google.com/tpu/docs>`_.

.. code-block:: bash

   pip install -U "saiunit[tpu]"

Plain JAX
~~~~~~~~~

Adds JAX without pinning an accelerator build — use this when you
manage the JAX accelerator wheel yourself (custom CUDA, ROCm, Apple
Metal, conda-forge, etc.).

.. code-block:: bash

   pip install -U "saiunit[jax]"

CuPy
~~~~

Adds ``cupy-cuda12x`` for the CuPy backend (NVIDIA GPU, drop-in NumPy
replacement). Requires a working CUDA toolkit on the host. See the
`CuPy install guide <https://docs.cupy.dev/en/stable/install.html>`_
for CUDA 11 or ROCm wheels.

.. code-block:: bash

   pip install -U "saiunit[cupy]"

PyTorch
~~~~~~~

Adds ``torch ≥ 2.0`` for the PyTorch backend. ``torch.autograd`` on the
mantissa is preserved through every ``saiunit`` operation. For
specific CUDA / ROCm builds, install ``torch`` from
`pytorch.org <https://pytorch.org/get-started/locally/>`_ *before*
``saiunit``.

.. code-block:: bash

   pip install -U "saiunit[torch]"

Dask
~~~~

Adds ``dask[array]`` for out-of-core / parallel arrays. Operations
stay lazy until you call ``.compute()`` or ``.persist()``.

.. code-block:: bash

   pip install -U "saiunit[dask]"

ndonnx
~~~~~~

Adds ``ndonnx`` for symbolic graph building and ONNX export.

.. code-block:: bash

   pip install -U "saiunit[ndonnx]"

All optional backends
~~~~~~~~~~~~~~~~~~~~~

Shorthand for ``[jax,cupy,torch,dask,ndonnx]``. Does *not* pin a JAX
accelerator build — combine with ``[cpu]`` / ``[cuda12]`` /
``[cuda13]`` / ``[tpu]`` if you need one.

.. code-block:: bash

   pip install -U "saiunit[all]"
   pip install -U "saiunit[all,cuda12]"   # add CUDA 12 JAX on top


.. _install-combining:

Combining extras
----------------

The JAX accelerator extras — ``[cpu]``, ``[cuda12]``, ``[cuda13]``,
``[tpu]`` — are mutually exclusive: choose at most one per
environment. The remaining extras (``[cupy]``, ``[torch]``, ``[dask]``,
``[ndonnx]``) are independent and can be combined freely in a single
command:

.. code-block:: bash

   pip install -U "saiunit[cuda12,torch,dask]"
   pip install -U "saiunit[cpu,cupy,ndonnx]"

.. note::

   Always quote the bracketed extras (``"saiunit[...]"``) on
   ``zsh`` / ``bash``; the shells otherwise interpret ``[`` and ``]``
   as glob characters and the install will fail with
   *"no matches found"*.


.. _install-source:

Installing from source
----------------------

To track ``main`` or hack on ``saiunit`` itself, install in editable
mode from a local clone:

.. code-block:: bash

   git clone https://github.com/chaobrain/saiunit.git
   cd saiunit
   pip install -e ".[cpu]"            # core + JAX CPU + editable
   pip install -e ".[cpu,torch]"      # add any extras you need

Run the test suite to confirm the dev install:

.. code-block:: bash

   pip install -e ".[testing]"
   pytest


.. _install-verify:

Verifying the install
---------------------

A two-line smoke test exercises the core, NumPy backend, and unit
algebra:

.. code-block:: python

   import saiunit as u
   print(u.__version__)

   q = 9.81 * u.meter / u.second ** 2     # gravitational acceleration
   print(q)                               # 9.81 * meter / second ** 2

If you installed a JAX extra, confirm the autograd path is wired up:

.. code-block:: python

   import saiunit as u

   x = 3.0 * u.meter
   f = lambda x: x ** 3
   print(u.autograd.grad(f)(x))           # 27. * meter ** 2

For any non-default backend, check that it is selectable:

.. code-block:: python

   import saiunit as u

   with u.using_backend("torch"):          # or "jax", "cupy", "dask", "ndonnx"
       q = u.math.arange(0., 5.) * u.second
       print(q)

Requesting a backend whose array library is not installed raises
:class:`saiunit.BackendError` with the exact ``pip install`` command to
fix it — never a bare ``ImportError``.


.. _install-upgrade:

Upgrading and uninstalling
--------------------------

Upgrade to the latest release (and preserve any extras you previously
selected by re-specifying them):

.. code-block:: bash

   pip install -U "saiunit[cuda12,torch]"

Uninstall the package and any editable install:

.. code-block:: bash

   pip uninstall saiunit

``pip uninstall`` does **not** remove the optional dependencies
(``jax``, ``torch``, ``cupy``, …); remove them individually if you no
longer need them.


.. _install-troubleshoot:

Troubleshooting
---------------

**“zsh: no matches found: saiunit[cpu]”**
    Quote the extras: ``pip install -U "saiunit[cpu]"``.

**``BackendError`` on import of ``saiunit.autograd`` / ``.lax`` / ``.sparse``**
    These submodules require JAX. Install with one of ``saiunit[cpu]``,
    ``saiunit[cuda12]``, ``saiunit[cuda13]``, ``saiunit[tpu]``, or
    ``saiunit[jax]``. The error message includes the recommended
    install command.

**``saiunit.BackendError: backend 'cupy' is not installed``**
    The selected backend's array library is missing. Install the
    matching extra — for example ``pip install -U "saiunit[cupy]"`` —
    or call :func:`saiunit.using_backend` with a backend you have.

**JAX picks the wrong accelerator (CPU instead of GPU, or vice versa)**
    ``pip`` may have resolved a JAX wheel that does not match your
    hardware. Reinstall with the explicit accelerator extra
    (``"saiunit[cuda12]"`` for CUDA 12, etc.) and check
    ``jax.devices()`` to confirm.

**CuPy import fails with “libcudart not found”**
    The CuPy wheel needs a matching CUDA runtime on the host. Install
    a CUDA 12.x toolkit, or follow the `CuPy install guide
    <https://docs.cupy.dev/en/stable/install.html>`_ for CUDA 11 / ROCm
    wheels.

**PyTorch installs the CPU wheel on a GPU host**
    Install ``torch`` from `pytorch.org
    <https://pytorch.org/get-started/locally/>`_ with the correct CUDA
    selector *before* installing ``saiunit[torch]``; ``pip`` will then
    keep your GPU build.

**Windows ``pip install`` fails on ``cupy-cuda12x``**
    ``cupy-cuda12x`` ships Linux and Windows wheels but requires the
    NVIDIA driver and CUDA runtime to be installed system-wide. Verify
    with ``nvidia-smi``.

Still stuck? File an issue at
https://github.com/chaobrain/saiunit/issues with the output of
``python -c "import saiunit, sys; print(saiunit.__version__, sys.version)"``
and the failing command.


.. _install-ecosystem:

Installing the BrainX ecosystem
-------------------------------

``saiunit`` is part of `BrainX <https://github.com/chaobrain>`_, a
family of compatible JAX-based tools for brain dynamics and scientific
computing. The ``BrainX`` meta-package bundles ``saiunit`` together
with :mod:`brainstate`, :mod:`brainunit`, :mod:`braintools`,
:mod:`braintaichi`, :mod:`dendritex`, and friends:

.. code-block:: bash

   pip install -U BrainX

Pick this if you want the full stack in one command; pick plain
``saiunit`` (with the extras you need) if you only want unit-aware
arrays.


See also
--------

- :doc:`../backends/overview` — full backend matrix, selection
  rules, and per-backend capability notes.
- :doc:`../backends/feature_support_matrix` — function-level support
  table for every backend.
- :doc:`quickstart` — first steps with ``Quantity`` and units.
