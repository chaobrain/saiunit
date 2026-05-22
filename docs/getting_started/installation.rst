Installation
============

``saiunit`` is published on PyPI. The core package depends only on NumPy.
JAX and every other array library are *optional* extras — install only
what you need.

Requirements
------------

- Python ≥ 3.10 (CI tests on 3.13)
- NumPy
- Optional: JAX, CuPy, PyTorch, Dask, or ndonnx (any combination)

Without JAX, the NumPy backend is selected automatically and the JAX-only
submodules (``saiunit.autograd``, ``saiunit.lax``, ``saiunit.sparse``) and
the custom ``exprel`` primitive raise :class:`saiunit.BackendError` on
access. The CuPy, PyTorch, Dask, and ndonnx extras are independent of JAX
and can be combined freely.


Choosing an install command
---------------------------

.. tab-set::

    .. tab-item:: NumPy only (no JAX)

       Smallest install. ``saiunit.math``, ``saiunit.linalg``, and
       ``saiunit.fft`` work on the NumPy backend; JAX-only submodules
       raise :class:`saiunit.BackendError`.

       .. code-block:: bash

          pip install -U saiunit

    .. tab-item:: JAX (CPU)

       Adds the pinned JAX CPU build. Enables ``saiunit.autograd``,
       ``saiunit.lax``, ``saiunit.sparse``, and JIT / ``vmap`` /
       ``pmap`` over quantities.

       .. code-block:: bash

          pip install -U saiunit[cpu]

    .. tab-item:: JAX (GPU / CUDA)

       Adds the JAX CUDA build. Pick the line that matches your CUDA
       toolkit.

       .. code-block:: bash

          pip install -U saiunit[cuda12]
          pip install -U saiunit[cuda13]

    .. tab-item:: JAX (TPU)

       Adds the JAX TPU build (Google Cloud TPU).

       .. code-block:: bash

          pip install -U saiunit[tpu]

    .. tab-item:: Plain JAX

       Adds JAX without pinning an accelerator build — use this if you
       manage the JAX accelerator wheel yourself.

       .. code-block:: bash

          pip install -U saiunit[jax]

    .. tab-item:: CuPy

       Adds ``cupy-cuda12x`` for the CuPy backend (NVIDIA GPU,
       drop-in NumPy replacement). Requires a CUDA toolkit.

       .. code-block:: bash

          pip install -U saiunit[cupy]

    .. tab-item:: PyTorch

       Adds ``torch>=2.0`` for the PyTorch backend. ``torch.autograd``
       on the mantissa is preserved through ``saiunit`` ops.

       .. code-block:: bash

          pip install -U saiunit[torch]

    .. tab-item:: Dask

       Adds ``dask[array]`` for out-of-core / parallel arrays.
       Operations stay lazy until ``.compute()``.

       .. code-block:: bash

          pip install -U saiunit[dask]

    .. tab-item:: ndonnx

       Adds ``ndonnx`` for symbolic graph building / ONNX export.

       .. code-block:: bash

          pip install -U saiunit[ndonnx]

    .. tab-item:: All optional backends

       Shorthand for ``[cupy,torch,dask,ndonnx]``. Does *not* pin a JAX
       accelerator build — combine with ``[cpu]`` / ``[cuda12]`` /
       ``[cuda13]`` / ``[tpu]`` if you want one.

       .. code-block:: bash

          pip install -U saiunit[all]


Combining extras
----------------

The JAX accelerator extras — ``[cpu]``, ``[cuda12]``, ``[cuda13]``,
``[tpu]`` — are mutually exclusive: pick at most one. The remaining
extras (``[cupy]``, ``[torch]``, ``[dask]``, ``[ndonnx]``) are
independent and can be combined in a single command:

.. code-block:: bash

   pip install -U "saiunit[cuda12,torch,dask]"


Verifying the install
---------------------

.. code-block:: python

   import saiunit as u
   print(u.__version__)
   q = 9.81 * u.meter / u.second ** 2
   print(q)


See also
--------

- :doc:`../backends/overview` — full backend matrix, selection rules,
  and per-backend capability notes.
- :doc:`quickstart` — first steps with ``Quantity`` and units.
