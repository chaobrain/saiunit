Feature support matrix
======================

.. note::
   This page is generated from ``dev/backend_support_data.json``,
   which is produced by ``dev/backend_support_sweep.py`` —
   an automated sweep that invokes every public function in
   ``saiunit.math``, ``saiunit.linalg``, ``saiunit.fft``, and every
   public ``Quantity`` method across each locally-installed backend
   and records the outcome.  Re-run the sweep and the renderer to
   refresh this page.

Cell legend
-----------

==========  ==========================================================
Glyph       Meaning
==========  ==========================================================
``✓``       Verified: the call returned a value of the expected backend kind.
``⊘``       Skipped: the backend's array-API surface does not expose the underlying op,
            or it rejects a keyword saiunit forwards (e.g. JAX-only ``precision=``).
``✗``       Failed: the call raised an unexpected exception on this backend.
``⚠``       Works with a caveat (e.g. lazy result on dask, expected ``BackendError`` for
            materialization on dask).
``🅙``       JAX-only by design — gated by ``saiunit._jax_guard.require_jax_backend``.
            Raises :class:`~saiunit.BackendError` on any non-jax backend.
``—``       Not applicable to backend dispatch (dtype factories, dimension predicates).
``?``       Not tested in this report or not mapped by the automated sweep. The
            single unmapped Quantity method
            (``tree_unflatten``) is also ``?`` because automated invocation requires
            a hand-crafted aux/children pair.
==========  ==========================================================

**Sweep environment.**  Backends invoked: ``numpy``, ``jax``, ``cupy``, ``torch``, ``dask``, ``ndonnx``.
Backends shown but not tested: ``none``.

High-level summary
------------------

.. list-table:: Per-subpackage rating
   :header-rows: 1
   :widths: 30 12 12 12 12 12 12

   * - Subpackage
     - numpy
     - jax
     - cupy
     - torch
     - dask
     - ndonnx
   * - **saiunit.math**
     - Mostly ⚠
     - Full ✓
     - Mostly ⚠
     - Partial ⚠
     - Partial ⚠
     - Partial ⚠
   * - **saiunit.linalg**
     - Mostly ⚠
     - Full ✓
     - Mostly ⚠
     - Mostly ⚠
     - Partial ⚠
     - Limited ✗
   * - **saiunit.fft**
     - Full ✓
     - Full ✓
     - Full ✓
     - Full ✓
     - Full ✓
     - Limited ✗
   * - **Quantity methods**
     - Mostly ⚠
     - Mostly ⚠
     - Mostly ⚠
     - Partial ⚠
     - Partial ⚠
     - Partial ⚠
   * - **saiunit.lax**
     - JAX-only 🅙
     - Full ✓
     - JAX-only 🅙
     - JAX-only 🅙
     - JAX-only 🅙
     - JAX-only 🅙
   * - **saiunit.autograd**
     - JAX-only 🅙
     - Full ✓
     - JAX-only 🅙
     - JAX-only 🅙
     - JAX-only 🅙
     - JAX-only 🅙
   * - **saiunit.sparse**
     - JAX-only 🅙
     - Full ✓
     - JAX-only 🅙
     - JAX-only 🅙
     - JAX-only 🅙
     - JAX-only 🅙

Rating thresholds: **Full** ≥ 95 % pass and zero fail; **Mostly** ≥ 80 % pass;
**Partial** ≥ 30 % pass; **Limited** < 30 % pass; **JAX-only** = gated by
``require_jax_backend``.

Backend-specific notes
----------------------

- **jax** — full feature set; default backend.  All JAX-only subpackages
  (``saiunit.lax``, ``saiunit.autograd``, ``saiunit.sparse``) require this
  backend.
- **numpy** — eager CPU computation through ``array_api_compat.numpy``.
  A handful of reductions (``amax``, ``amin``, ``mean``, ``nan*`` variants)
  fail when saiunit forwards a ``where=None`` kwarg numpy can't interpret.
  These are listed with footnotes in the math tables below.
- **cupy** — measured on a CUDA device through ``array_api_compat.cupy``.
  CuPy ``14.1.1``; device
  ``NVIDIA GeForce RTX 3060 Laptop GPU``; CUDA driver/runtime
  ``13030`` /
  ``12090``.
  Cells and summary ratings below are generated from the measured sweep.
- **torch** — through ``array_api_compat.torch``.  The torch array-API
  surface lacks several ops saiunit dispatches to (``cbrt``, ``digamma``,
  some ``einops`` reductions, ``axes=`` for n-D FFTs) and rejects
  JAX-flavored kwargs (``precision``, ``symmetrize_input``, ``tol``).
  Affected calls are recorded as skip rather than fail.
- **dask** — lazy arrays.  Reductions and most array ops succeed but the
  result remains lazy until ``.compute()``.  Per ``saiunit._base_quantity``,
  the Python casts ``float(q)`` / ``int(q)`` / ``operator.index(q)`` /
  ``np.asarray(q)`` / ``hash(q)`` and the ``Quantity.tolist`` method raise
  :class:`~saiunit.BackendError` to avoid silent materialization — these
  cells are ``⚠`` with the BackendError text in the footnote.
  ``Quantity.item`` on dask raises a different error (the dask Array has
  no ``.item()`` method) so it appears as ``⊘`` rather than ``⚠``.
  Methods like ``Quantity.float`` / ``.double`` are ``.astype`` in disguise
  and stay lazy on dask, so they pass.
- **ndonnx** — symbolic graph-building backend.  Many array-API ops
  (``fft.*``, several ``linalg.*``, complex / specialty math) are not
  implemented and surface as ``⊘`` skip rows.  Saiunit does not encode
  unit information into the ONNX graph.

JAX-only subpackages
--------------------

These subpackages dispatch directly to JAX primitives that have no
array-API equivalent.  Each entry point is wrapped with
``saiunit._jax_guard.require_jax_backend``, which raises
:class:`~saiunit.BackendError` on any non-jax mantissa.

**saiunit.lax** — 101 public callable(s); all require ``jax``.

.. list-table::
   :header-rows: 1
   :widths: 16 12 12 12 12 12 12

   * - Probe result
     - numpy
     - jax
     - cupy
     - torch
     - dask
     - ndonnx
   * - all functions
     - 🅙
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙

.. dropdown:: List of saiunit.lax functions

   .. hlist::
      :columns: 4

      * ``acos``
      * ``acosh``
      * ``approx_max_k``
      * ``approx_min_k``
      * ``asin``
      * ``asinh``
      * ``atan``
      * ``atan2``
      * ``atanh``
      * ``batch_matmul``
      * ``bessel_i0e``
      * ``bessel_i1e``
      * ``betainc``
      * ``bitcast_convert_type``
      * ``broadcast``
      * ``broadcast_in_dim``
      * ``broadcast_shapes``
      * ``broadcast_to_rank``
      * ``broadcasted_iota``
      * ``cholesky``
      * ``clamp``
      * ``clz``
      * ``collapse``
      * ``complex``
      * ``conv``
      * ``conv_transpose``
      * ``convert_element_type``
      * ``cumlogsumexp``
      * ``cummax``
      * ``cummin``
      * ``cumsum``
      * ``digamma``
      * ``div``
      * ``dot_general``
      * ``dynamic_index_in_dim``
      * ``dynamic_slice``
      * ``dynamic_slice_ind_dim``
      * ``dynamic_update_index_in_dim``
      * ``dynamic_update_slice``
      * ``dynamic_update_slice_in_dim``
      * ``eig``
      * ``eigh``
      * ``eq``
      * ``erf``
      * ``erf_inv``
      * ``erfc``
      * ``fft``
      * ``gather``
      * ``ge``
      * ``gt``
      * ``hessenberg``
      * ``householder_product``
      * ``igamma``
      * ``igamma_grad_a``
      * ``igammac``
      * ``index_in_dim``
      * ``index_take``
      * ``integer_pow``
      * ``iota``
      * ``le``
      * ``lgamma``
      * ``logistic``
      * ``lt``
      * ``lu``
      * ``mul``
      * ``ne``
      * ``neg``
      * ``pad``
      * ``polygamma``
      * ``population_count``
      * ``pow``
      * ``qdwh``
      * ``qr``
      * ``random_gamma_grad``
      * ``reduce``
      * ``reduce_precision``
      * ``rem``
      * ``rsqrt``
      * ``scatter``
      * ``scatter_add``
      * ``scatter_apply``
      * ``scatter_max``
      * ``scatter_min``
      * ``scatter_mul``
      * ``scatter_sub``
      * ``schur``
      * ``shift_left``
      * ``shift_right_arithmetic``
      * ``shift_right_logical``
      * ``slice``
      * ``slice_in_dim``
      * ``sort``
      * ``sort_key_val``
      * ``sub``
      * ``svd``
      * ``top_k``
      * ``triangular_solve``
      * ``tridiagonal``
      * ``tridiagonal_solve``
      * ``zeros_like_array``
      * ``zeta``

**saiunit.autograd** — 7 public callable(s); all require ``jax``.

.. list-table::
   :header-rows: 1
   :widths: 16 12 12 12 12 12 12

   * - Probe result
     - numpy
     - jax
     - cupy
     - torch
     - dask
     - ndonnx
   * - all functions
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗
     - ✗

.. dropdown:: List of saiunit.autograd functions

   .. hlist::
      :columns: 4

      * ``grad``
      * ``hessian``
      * ``jacfwd``
      * ``jacobian``
      * ``jacrev``
      * ``value_and_grad``
      * ``vector_grad``

**saiunit.sparse** — 10 public callable(s); all require ``jax``.

.. list-table::
   :header-rows: 1
   :widths: 16 12 12 12 12 12 12

   * - Probe result
     - numpy
     - jax
     - cupy
     - torch
     - dask
     - ndonnx
   * - all functions
     - 🅙
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - ✗

.. dropdown:: List of saiunit.sparse functions

   .. hlist::
      :columns: 4

      * ``COO``
      * ``CSC``
      * ``CSR``
      * ``SparseMatrix``
      * ``coo_fromdense``
      * ``coo_todense``
      * ``csc_fromdense``
      * ``csc_todense``
      * ``csr_fromdense``
      * ``csr_todense``

saiunit.math
------------

Public callables in ``saiunit.math`` that go through the multi-backend
dispatcher.  Grouped by source submodule for readability.

``array_creation`` — Array creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: saiunit.math — Array creation
   :header-rows: 1
   :widths: 40 10 10 10 10 10 10

   * - Function
     - numpy
     - jax
     - cupy
     - torch
     - dask
     - ndonnx
   * - ``tril_indices``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``triu_indices``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

``keep_unit`` — Unit-preserving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: saiunit.math — Unit-preserving
   :header-rows: 1
   :widths: 40 10 10 10 10 10 10

   * - Function
     - numpy
     - jax
     - cupy
     - torch
     - dask
     - ndonnx
   * - ``astype``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

``change_unit`` — Unit-changing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*saiunit.math — Unit-changing: no functions in this group.*

``accept_unitless`` — Dimensionless-only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*saiunit.math — Dimensionless-only: no functions in this group.*

``remove_unit`` — Unit-removing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*saiunit.math — Unit-removing: no functions in this group.*

``saiunit.math``
^^^^^^^^^^^^^^^^

.. list-table:: saiunit.math — saiunit.math
   :header-rows: 1
   :widths: 40 10 10 10 10 10 10

   * - Function
     - numpy
     - jax
     - cupy
     - torch
     - dask
     - ndonnx
   * - ``abs``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``absolute``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-1]_
   * - ``add``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``all``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``allclose``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-2]_
   * - ``alltrue``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``amax``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``amin``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``angle``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-3]_
   * - ``any``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``append``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-4]_
     - ✓
     - ⊘ [#fn-5]_
   * - ``arange``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``arccos``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-6]_
   * - ``arccosh``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-7]_
   * - ``arcsin``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-8]_
   * - ``arcsinh``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-9]_
   * - ``arctan``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-10]_
   * - ``arctan2``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-11]_
   * - ``arctanh``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-12]_
   * - ``argmax``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``argmin``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``argsort``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``argwhere``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-13]_
   * - ``around``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-14]_
     - ✓
     - ⊘ [#fn-15]_
   * - ``array``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``array_equal``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-16]_
     - ⊘ [#fn-17]_
     - ⊘ [#fn-18]_
   * - ``array_split``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-19]_
     - ⊘ [#fn-20]_
     - ⊘ [#fn-21]_
   * - ``as_numpy``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``asarray``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``atleast_1d``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-22]_
   * - ``atleast_2d``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-23]_
   * - ``atleast_3d``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-24]_
   * - ``average``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-25]_
     - ✓
     - ⊘ [#fn-26]_
   * - ``bincount``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-27]_
   * - ``bitwise_and``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``bitwise_not``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-28]_
   * - ``bitwise_or``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``bitwise_xor``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``block``
     - ✓
     - ✓
     - ⊘ [#fn-29]_
     - ⊘ [#fn-30]_
     - ✓
     - ⊘ [#fn-31]_
   * - ``broadcast_arrays``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``broadcast_shapes``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``broadcast_to``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``cbrt``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-32]_
     - ✓
     - ⊘ [#fn-33]_
   * - ``ceil``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``celu``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``choose``
     - ✓
     - ✓
     - ⊘ [#fn-35]_
     - ⊘ [#fn-36]_
     - ⊘ [#fn-37]_
     - ⊘ [#fn-38]_
   * - ``clip``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``column_stack``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-39]_
     - ⊘ [#fn-40]_
   * - ``compress``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-41]_
     - ✓
     - ⊘ [#fn-42]_
   * - ``concatenate``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-43]_
   * - ``conj``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-3]_
   * - ``conjugate``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-44]_
     - ⊘ [#fn-45]_
     - ✗ [#fn-3]_
   * - ``convolve``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-46]_
     - ⊘ [#fn-47]_
     - ⊘ [#fn-48]_
   * - ``copysign``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-49]_
   * - ``corrcoef``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-50]_
   * - ``correlate``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-51]_
     - ⊘ [#fn-52]_
     - ⊘ [#fn-53]_
   * - ``cos``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``cosh``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``count_nonzero``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``cov``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-54]_
   * - ``cross``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-55]_
     - ⊘ [#fn-56]_
   * - ``cumprod``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-57]_
   * - ``cumproduct``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-57]_
   * - ``cumsum``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-58]_
   * - ``deg2rad``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-59]_
   * - ``degrees``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-60]_
     - ✓
     - ⊘ [#fn-61]_
   * - ``diag``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-62]_
   * - ``diag_indices_from``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-63]_
     - ⊘ [#fn-64]_
     - ⊘ [#fn-65]_
   * - ``diagflat``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-66]_
     - ⊘ [#fn-67]_
   * - ``diagonal``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-68]_
   * - ``diff``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``digitize``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-69]_
     - ✓
     - ⊘ [#fn-70]_
   * - ``divide``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``divmod``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-71]_
     - ✓
     - ⊘ [#fn-72]_
   * - ``dot``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-73]_
   * - ``dsplit``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-74]_
     - ⊘ [#fn-75]_
   * - ``dstack``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-76]_
   * - ``ediff1d``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-77]_
     - ✓
     - ⊘ [#fn-78]_
   * - ``einsum``
     - ✗ [#fn-34]_
     - ✓
     - ✗ [#fn-79]_
     - ⊘ [#fn-80]_
     - ✗ [#fn-81]_
     - ⊘ [#fn-82]_
   * - ``elu``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``empty``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``empty_like``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-83]_
   * - ``equal``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``exp``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``exp2``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-84]_
   * - ``expand_dims``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``expm1``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-49]_
   * - ``exprel``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``extract``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-85]_
     - ✓
     - ⊘ [#fn-86]_
   * - ``eye``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``fabs``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-87]_
     - ✓
     - ⊘ [#fn-88]_
   * - ``fill_diagonal``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-89]_
     - ⊘ [#fn-90]_
     - ⊘ [#fn-91]_
   * - ``finfo``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``fix``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``flatnonzero``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-92]_
     - ✓
     - ⊘ [#fn-93]_
   * - ``flatten``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``flip``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``fliplr``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-94]_
   * - ``flipud``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-95]_
   * - ``float_power``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-96]_
   * - ``floor``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``floor_divide``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``fmax``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-97]_
   * - ``fmin``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-98]_
   * - ``fmod``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-99]_
   * - ``frexp``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-100]_
   * - ``from_numpy``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``full``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``full_like``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-101]_
   * - ``gather``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-102]_
     - ⊘ [#fn-103]_
   * - ``gcd``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-104]_
     - ⊘ [#fn-105]_
   * - ``gelu``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``get_promote_dtypes``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``glu``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``gradient``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-106]_
   * - ``greater``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``greater_equal``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``hard_sigmoid``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``hard_silu``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``hard_swish``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``hard_tanh``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``heaviside``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-107]_
     - ⊘ [#fn-108]_
   * - ``histogram``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-109]_
     - ⊘ [#fn-110]_
   * - ``hsplit``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-111]_
     - ⊘ [#fn-112]_
   * - ``hstack``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-113]_
   * - ``hypot``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-49]_
   * - ``identity``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-114]_
     - ⊘ [#fn-115]_
     - ⊘ [#fn-116]_
   * - ``iinfo``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``imag``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-3]_
   * - ``inner``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-117]_
     - ⊘ [#fn-118]_
   * - ``interp``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-119]_
     - ⊘ [#fn-120]_
     - ⊘ [#fn-121]_
   * - ``intersect1d``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-122]_
     - ⊘ [#fn-123]_
     - ⊘ [#fn-124]_
   * - ``invert``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-125]_
     - ✓
     - ⊘ [#fn-126]_
   * - ``isclose``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-127]_
   * - ``iscomplexobj``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-128]_
     - ⊘ [#fn-129]_
     - ⊘ [#fn-130]_
   * - ``isfinite``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``isinf``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``isnan``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``isreal``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-131]_
   * - ``isscalar``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``issubdtype``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``kron``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-132]_
     - ⊘ [#fn-133]_
   * - ``lcm``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-134]_
     - ⊘ [#fn-135]_
   * - ``ldexp``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-136]_
   * - ``leaky_relu``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``left_shift``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-137]_
     - ✓
     - ⊘ [#fn-138]_
   * - ``less``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``less_equal``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``linspace``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-139]_
   * - ``log``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``log10``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``log1p``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-49]_
   * - ``log2``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``log_sigmoid``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``logaddexp``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``logaddexp2``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-140]_
   * - ``logical_and``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``logical_not``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``logical_or``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``logical_xor``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``logspace``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-141]_
     - ⊘ [#fn-142]_
     - ⊘ [#fn-143]_
   * - ``matmul``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``matrix_power``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-144]_
     - ⊘ [#fn-145]_
   * - ``max``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``maximum``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``mean``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``median``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-146]_
     - ⊘ [#fn-147]_
     - ⊘ [#fn-148]_
   * - ``meshgrid``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-149]_
   * - ``min``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``minimum``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``mish``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``mod``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-150]_
     - ✓
     - ⊘ [#fn-151]_
   * - ``modf``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-152]_
     - ✓
     - ⊘ [#fn-153]_
   * - ``moveaxis``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``multi_dot``
     - ✓
     - ✓
     - ⊘ [#fn-154]_
     - ✓
     - ⊘ [#fn-155]_
     - ⊘ [#fn-156]_
   * - ``multiply``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``nan_to_num``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-157]_
   * - ``nanargmax``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-158]_
     - ✓
     - ⊘ [#fn-159]_
   * - ``nanargmin``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-160]_
     - ✓
     - ⊘ [#fn-161]_
   * - ``nancumprod``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-162]_
     - ✓
     - ⊘ [#fn-163]_
   * - ``nancumsum``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-164]_
     - ✓
     - ⊘ [#fn-165]_
   * - ``nanmax``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-166]_
     - ✓
     - ⊘ [#fn-167]_
   * - ``nanmean``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-168]_
   * - ``nanmedian``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-169]_
     - ⊘ [#fn-170]_
     - ⊘ [#fn-171]_
   * - ``nanmin``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-172]_
     - ✓
     - ⊘ [#fn-173]_
   * - ``nanpercentile``
     - ✓
     - ✓
     - ⊘ [#fn-174]_
     - ⊘ [#fn-175]_
     - ✓
     - ⊘ [#fn-176]_
   * - ``nanprod``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-177]_
     - ✓
     - ⊘ [#fn-178]_
   * - ``nanquantile``
     - ✓
     - ✓
     - ⊘ [#fn-179]_
     - ⊘ [#fn-180]_
     - ✓
     - ⊘ [#fn-181]_
   * - ``nanstd``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-182]_
     - ✓
     - ⊘ [#fn-183]_
   * - ``nansum``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-184]_
   * - ``nanvar``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-185]_
     - ✓
     - ⊘ [#fn-186]_
   * - ``ndim``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``negative``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``nextafter``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-49]_
   * - ``nonzero``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``not_equal``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``ones``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``ones_like``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-187]_
   * - ``outer``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-188]_
   * - ``percentile``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-189]_
     - ✓
     - ⊘ [#fn-190]_
   * - ``positive``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``power``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-191]_
     - ✓
     - ⊘ [#fn-192]_
   * - ``prod``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``product``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``promote_dtypes``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``ptp``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-193]_
     - ⊘ [#fn-194]_
     - ⊘ [#fn-195]_
   * - ``quantile``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-196]_
     - ✓
     - ⊘ [#fn-197]_
   * - ``rad2deg``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-198]_
   * - ``radians``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-199]_
     - ✓
     - ⊘ [#fn-200]_
   * - ``ravel``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-201]_
     - ⊘ [#fn-202]_
   * - ``real``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-3]_
   * - ``reciprocal``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``relu``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``relu6``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``remainder``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``remove_diag``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``repeat``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``reshape``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-203]_
   * - ``result_type``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``right_shift``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-204]_
     - ✓
     - ⊘ [#fn-205]_
   * - ``rint``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-206]_
     - ✓
     - ⊘ [#fn-207]_
   * - ``roll``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``rot90``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-208]_
   * - ``round``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-209]_
   * - ``row_stack``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-210]_
   * - ``searchsorted``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``select``
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-211]_
     - ✓
     - ⊘ [#fn-212]_
   * - ``selu``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``shape``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``sigmoid``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``sign``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``signbit``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-49]_
   * - ``silu``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``sin``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``sinc``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-213]_
   * - ``sinh``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``size``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``soft_sign``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``softplus``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``sometrue``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``sort``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``sparse_plus``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``sparse_sigmoid``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``split``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-19]_
     - ⊘ [#fn-20]_
     - ⊘ [#fn-21]_
   * - ``sqrt``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``square``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``squareplus``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``squeeze``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-214]_
     - ✓
     - ⊘ [#fn-214]_
   * - ``stack``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``std``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``subtract``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``sum``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``swapaxes``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-215]_
   * - ``swish``
     - ✗ [#fn-34]_
     - ✓
     - 🅙
     - 🅙
     - 🅙
     - 🅙
   * - ``take``
     - ⊘ [#fn-216]_
     - ✓
     - ⊘ [#fn-216]_
     - ✓
     - ⊘ [#fn-216]_
     - ⊘ [#fn-216]_
   * - ``tan``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``tanh``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``tensordot``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``tile``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``trace``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-217]_
   * - ``transpose``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-218]_
     - ✓
     - ⊘ [#fn-219]_
   * - ``trapezoid``
     - ?
     - ?
     - ?
     - ?
     - ?
     - ?
   * - ``tree_ones_like``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-187]_
   * - ``tree_zeros_like``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-220]_
   * - ``tri``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-221]_
     - ✓
     - ⊘ [#fn-222]_
   * - ``tril``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``tril_indices_from``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-223]_
     - ✓
     - ⊘ [#fn-224]_
   * - ``triu``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``triu_indices_from``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-225]_
     - ✓
     - ⊘ [#fn-226]_
   * - ``true_divide``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-227]_
   * - ``trunc``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``unflatten``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``unique``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-228]_
     - ⊘ [#fn-229]_
   * - ``vander``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-230]_
     - ⊘ [#fn-231]_
   * - ``var``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``vdot``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-232]_
   * - ``vecdot``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``vsplit``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-233]_
     - ⊘ [#fn-234]_
   * - ``vstack``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-210]_
   * - ``where``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``zeros``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``zeros_like``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-220]_

``jax.numpy``
^^^^^^^^^^^^^

.. list-table:: saiunit.math — jax.numpy
   :header-rows: 1
   :widths: 40 10 10 10 10 10 10

   * - Function
     - numpy
     - jax
     - cupy
     - torch
     - dask
     - ndonnx
   * - ``bartlett``
     - ✗ [#fn-34]_
     - ✓
     - ✗ [#fn-235]_
     - ✗ [#fn-236]_
     - ✗ [#fn-237]_
     - ✗ [#fn-238]_
   * - ``blackman``
     - ✗ [#fn-34]_
     - ✓
     - ✗ [#fn-235]_
     - ✗ [#fn-236]_
     - ✗ [#fn-237]_
     - ✗ [#fn-238]_
   * - ``hamming``
     - ✗ [#fn-34]_
     - ✓
     - ✗ [#fn-235]_
     - ✗ [#fn-236]_
     - ✗ [#fn-237]_
     - ✗ [#fn-238]_
   * - ``hanning``
     - ✗ [#fn-34]_
     - ✓
     - ✗ [#fn-235]_
     - ✗ [#fn-236]_
     - ✗ [#fn-237]_
     - ✗ [#fn-238]_
   * - ``kaiser``
     - ✗ [#fn-34]_
     - ✓
     - ✗ [#fn-235]_
     - ✗ [#fn-236]_
     - ✗ [#fn-237]_
     - ✗ [#fn-238]_

``numpy``
^^^^^^^^^

.. list-table:: saiunit.math — numpy
   :header-rows: 1
   :widths: 40 10 10 10 10 10 10

   * - Function
     - numpy
     - jax
     - cupy
     - torch
     - dask
     - ndonnx
   * - ``dtype``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

``brainstate.math``
^^^^^^^^^^^^^^^^^^^

.. list-table:: saiunit.math — brainstate.math
   :header-rows: 1
   :widths: 40 10 10 10 10 10 10

   * - Function
     - numpy
     - jax
     - cupy
     - torch
     - dask
     - ndonnx
   * - ``einrearrange``
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-239]_
     - ✓
     - ⊘ [#fn-219]_
   * - ``einreduce``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``einrepeat``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``einshape``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

Non-dispatched helpers
^^^^^^^^^^^^^^^^^^^^^^

These names live under ``saiunit.math`` for convenience but do not
dispatch on backend — they are dtype factories (re-exported from
``jax.numpy``) or pure-Python predicates / introspection helpers over
``Quantity`` / ``Unit`` objects.  Behavior is identical on every
backend.

.. hlist::
   :columns: 4

   * ``assert_quantity``
   * ``bfloat16``
   * ``bool_``
   * ``cdouble``
   * ``check_dims``
   * ``check_units``
   * ``complex128``
   * ``complex64``
   * ``complex_``
   * ``csingle``
   * ``display_in_unit``
   * ``double``
   * ``fail_for_dimension_mismatch``
   * ``fail_for_unit_mismatch``
   * ``float16``
   * ``float32``
   * ``float64``
   * ``float_``
   * ``get_dim``
   * ``get_dtype``
   * ``get_magnitude``
   * ``get_mantissa``
   * ``get_or_create_dimension``
   * ``get_unit``
   * ``inexact``
   * ``int16``
   * ``int2``
   * ``int32``
   * ``int4``
   * ``int64``
   * ``int8``
   * ``int_``
   * ``is_dimensionless``
   * ``is_float``
   * ``is_int``
   * ``is_quantity``
   * ``is_unitless``
   * ``maybe_decimal``
   * ``set_exprel_order``
   * ``single``
   * ``uint``
   * ``uint16``
   * ``uint2``
   * ``uint32``
   * ``uint4``
   * ``uint64``
   * ``uint8``

saiunit.linalg
--------------

.. list-table:: saiunit.linalg
   :header-rows: 1
   :widths: 40 10 10 10 10 10 10

   * - Function
     - numpy
     - jax
     - cupy
     - torch
     - dask
     - ndonnx
   * - ``cholesky``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-49]_
   * - ``cond``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-240]_
     - ⊘ [#fn-241]_
   * - ``cross``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-55]_
     - ⊘ [#fn-56]_
   * - ``det``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-242]_
     - ⊘ [#fn-243]_
   * - ``diagonal``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-244]_
   * - ``dot``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-73]_
   * - ``eig``
     - ✗ [#fn-34]_
     - ✓
     - ✗ [#fn-245]_
     - ⊘ [#fn-246]_
     - ⊘ [#fn-247]_
     - ✗ [#fn-248]_
   * - ``eigh``
     - ✗ [#fn-34]_
     - ✓
     - ⊘ [#fn-249]_
     - ⊘ [#fn-250]_
     - ⊘ [#fn-251]_
     - ⊘ [#fn-252]_
   * - ``eigvals``
     - ✗ [#fn-34]_
     - ✓
     - ✗ [#fn-245]_
     - ⊘ [#fn-246]_
     - ⊘ [#fn-247]_
     - ✗ [#fn-248]_
   * - ``eigvalsh``
     - ✗ [#fn-34]_
     - ✓
     - ⊘ [#fn-249]_
     - ⊘ [#fn-250]_
     - ⊘ [#fn-251]_
     - ⊘ [#fn-252]_
   * - ``inner``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-117]_
     - ⊘ [#fn-118]_
   * - ``inv``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-253]_
   * - ``kron``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-132]_
     - ⊘ [#fn-133]_
   * - ``lstsq``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-254]_
     - ⊘ [#fn-255]_
   * - ``matmul``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``matrix_norm``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-256]_
   * - ``matrix_power``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-144]_
     - ⊘ [#fn-145]_
   * - ``matrix_rank``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-257]_
     - ⊘ [#fn-258]_
   * - ``matrix_transpose``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-259]_
   * - ``multi_dot``
     - ✓
     - ✓
     - ⊘ [#fn-154]_
     - ✓
     - ⊘ [#fn-155]_
     - ⊘ [#fn-156]_
   * - ``norm``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-260]_
   * - ``outer``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-188]_
   * - ``pinv``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-261]_
     - ⊘ [#fn-262]_
   * - ``qr``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-255]_
   * - ``slogdet``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-263]_
     - ⊘ [#fn-255]_
   * - ``solve``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-264]_
   * - ``svd``
     - ✗ [#fn-34]_
     - ✓
     - ✗ [#fn-245]_
     - ⊘ [#fn-265]_
     - ⊘ [#fn-266]_
     - ✗ [#fn-248]_
   * - ``svdvals``
     - ✗ [#fn-34]_
     - ✓
     - ✗ [#fn-245]_
     - ⊘ [#fn-265]_
     - ⊘ [#fn-266]_
     - ✗ [#fn-248]_
   * - ``tensordot``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``tensorinv``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-267]_
     - ⊘ [#fn-268]_
   * - ``tensorsolve``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-269]_
     - ⊘ [#fn-270]_
   * - ``trace``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-271]_
   * - ``vdot``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-232]_
   * - ``vecdot``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``vector_norm``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-272]_

saiunit.fft
-----------

Routing varies inside ``saiunit.fft``: ``_fft_change_unit.py`` calls
``saiunit._backend.get_backend()`` directly (e.g. for ``fftfreq`` /
``rfftfreq``), while ``_fft_keep_unit.py`` delegates to the math
package's ``_fun_keep_unit_unary`` helper and inherits its dispatch.

.. list-table:: saiunit.fft
   :header-rows: 1
   :widths: 40 10 10 10 10 10 10

   * - Function
     - numpy
     - jax
     - cupy
     - torch
     - dask
     - ndonnx
   * - ``fft``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-273]_
   * - ``fft2``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-274]_
   * - ``fftfreq``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-275]_
   * - ``fftn``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-276]_
   * - ``fftshift``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-277]_
   * - ``ifft``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-278]_
   * - ``ifft2``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-279]_
   * - ``ifftn``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-280]_
   * - ``ifftshift``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-281]_
   * - ``irfft``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-282]_
   * - ``irfft2``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-283]_
   * - ``irfftn``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-284]_
   * - ``rfft``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-285]_
   * - ``rfft2``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-286]_
   * - ``rfftfreq``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-275]_
   * - ``rfftn``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-287]_

Quantity methods
----------------

Methods on ``saiunit.Quantity`` itself.  ``.to_<backend>()`` methods
ignore the *current* backend and convert to the named one — cells show
``⊘`` when the target backend isn't installed in the sweep environment.

Materialization is documented above (see *Backend-specific notes*).
``Quantity.tolist`` on dask is the one method that raises
:class:`~saiunit.BackendError` from saiunit's own guard (``⚠``).
``.item`` reports ``⊘`` on dask / ndonnx because the underlying array
object does not expose ``.item()``.

.. list-table:: Quantity public methods
   :header-rows: 1
   :widths: 40 10 10 10 10 10 10

   * - Function
     - numpy
     - jax
     - cupy
     - torch
     - dask
     - ndonnx
   * - ``all``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``any``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``argmax``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``argmin``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``argsort``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``astype``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``clip``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``clone``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-288]_
     - ⊘ [#fn-289]_
     - ⊘ [#fn-290]_
   * - ``conj``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-49]_
   * - ``conjugate``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-49]_
   * - ``copy``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-288]_
     - ⊘ [#fn-289]_
     - ⊘ [#fn-290]_
   * - ``cpu``
     - ✓
     - ✓
     - ✗ [#fn-291]_
     - ✗ [#fn-292]_
     - ✗ [#fn-293]_
     - ✗ [#fn-294]_
   * - ``cross``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-295]_
     - ⊘ [#fn-296]_
     - ⊘ [#fn-297]_
   * - ``cuda``
     - ⊘ [#fn-298]_
     - ⊘ [#fn-298]_
     - ⊘ [#fn-298]_
     - ⊘ [#fn-298]_
     - ⊘ [#fn-298]_
     - ⊘ [#fn-298]_
   * - ``cumprod``
     - ⊘ [#fn-299]_
     - ⊘ [#fn-299]_
     - ⊘ [#fn-299]_
     - ⊘ [#fn-299]_
     - ⊘ [#fn-299]_
     - ⊘ [#fn-299]_
   * - ``cumsum``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-300]_
     - ✓
     - ⊘ [#fn-58]_
   * - ``diagonal``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-301]_
     - ✓
     - ⊘ [#fn-302]_
   * - ``dot``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-303]_
   * - ``double``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``expand_as``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``expand_dims``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``factorless``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``fill``
     - ✓
     - ✓
     - ✗ [#fn-304]_
     - ✗ [#fn-292]_
     - ✗ [#fn-304]_
     - ✗ [#fn-305]_
   * - ``flatten``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``float``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``half``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-306]_
     - ✓
   * - ``has_same_unit``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``in_unit``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``item``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-307]_
     - ⊘ [#fn-307]_
   * - ``max``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``mean``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``min``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``nancumprod``
     - ⊘ [#fn-308]_
     - ⊘ [#fn-308]_
     - ⊘ [#fn-308]_
     - ⊘ [#fn-308]_
     - ⊘ [#fn-308]_
     - ⊘ [#fn-308]_
   * - ``nanprod``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-177]_
     - ✓
     - ⊘ [#fn-178]_
   * - ``nonzero``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``outer``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-309]_
   * - ``pow``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``prod``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``ptp``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-193]_
     - ✓
     - ⊘ [#fn-195]_
   * - ``put``
     - ✓
     - ✓
     - ✗ [#fn-304]_
     - ✗ [#fn-292]_
     - ⊘ [#fn-310]_
     - ✗ [#fn-305]_
   * - ``ravel``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-202]_
   * - ``repeat``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``repr_in_unit``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``reshape``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``resize``
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-311]_
     - ✗ [#fn-312]_
     - ✗ [#fn-313]_
   * - ``round``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``scatter_add``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-314]_
     - ✗ [#fn-305]_
   * - ``scatter_div``
     - ✓
     - ✓
     - ⊘ [#fn-315]_
     - ✓
     - ⊘ [#fn-316]_
     - ✗ [#fn-305]_
   * - ``scatter_max``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-317]_
     - ✗ [#fn-305]_
   * - ``scatter_min``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-318]_
     - ✗ [#fn-305]_
   * - ``scatter_mul``
     - ✓
     - ✓
     - ⊘ [#fn-319]_
     - ✓
     - ⊘ [#fn-320]_
     - ✗ [#fn-305]_
   * - ``scatter_sub``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-314]_
     - ✗ [#fn-305]_
   * - ``searchsorted``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``sort``
     - ✓
     - ✓
     - ✗ [#fn-304]_
     - ✗ [#fn-292]_
     - ✗ [#fn-304]_
     - ✗ [#fn-294]_
   * - ``split``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-321]_
     - ✗ [#fn-312]_
     - ✗ [#fn-322]_
   * - ``squeeze``
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-323]_
     - ✓
     - ✗ [#fn-324]_
   * - ``std``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``sum``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``swapaxes``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-325]_
   * - ``take``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-326]_
     - ✓
   * - ``tile``
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-327]_
     - ✓
     - ✗ [#fn-328]_
   * - ``to``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``to_cupy``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-329]_
   * - ``to_dask``
     - ✓
     - ✓
     - ✓
     - ✗ [#fn-292]_
     - ✓
     - ✗ [#fn-294]_
   * - ``to_decimal``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``to_jax``
     - ✓
     - ✓
     - ✗ [#fn-245]_
     - ✓
     - ✓
     - ✗ [#fn-330]_
   * - ``to_ndonnx``
     - ✓
     - ⊘ [#fn-331]_
     - ⊘ [#fn-331]_
     - ⊘ [#fn-332]_
     - ⊘ [#fn-333]_
     - ✓
   * - ``to_numpy``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``to_torch``
     - ✓
     - ✓
     - ✗ [#fn-334]_
     - ✓
     - ✗ [#fn-335]_
     - ✗ [#fn-336]_
   * - ``tolist``
     - ✓
     - ✓
     - ✓
     - ✓
     - ⚠ [#fn-337]_
     - ⊘ [#fn-338]_
   * - ``trace``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-339]_
     - ✓
     - ⊘ [#fn-340]_
   * - ``transpose``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-218]_
     - ✓
     - ⊘ [#fn-341]_
   * - ``tree_flatten``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``tree_unflatten``
     - ?
     - ?
     - ?
     - ?
     - ?
     - ?
   * - ``unsqueeze``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``update_mantissa``
     - ✓
     - ✓
     - ✗ [#fn-304]_
     - ✗ [#fn-292]_
     - ✗ [#fn-304]_
     - ✗ [#fn-294]_
   * - ``var``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``view``
     - ✓
     - ✓
     - ✓
     - ⊘ [#fn-342]_
     - ✓
     - ⊘ [#fn-343]_
   * - ``with_unit``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

Coverage statistic
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15 15

   * - Subpackage
     - Mapped
     - Non-dispatched
     - Unmapped
     - Total
   * - saiunit.math
     - 294
     - 47
     - 1
     - 342
   * - saiunit.linalg
     - 35
     - 0
     - 0
     - 35
   * - saiunit.fft
     - 16
     - 0
     - 0
     - 16
   * - Quantity
     - 78
     - 0
     - 1
     - 79

*Mapped* = functions the sweep actually invoked.  
*Non-dispatched* = type factories / predicates that don't go 
through backend dispatch.  
*Unmapped* = no call pattern registered (will appear as ``?`` in tables).

How this was produced
---------------------

``dev/backend_support_sweep.py`` walks every public callable in the
subpackages above, picks a calling pattern from an in-script registry,
and invokes the function under ``with saiunit.using_backend(b)`` for
each backend ``b`` in the local environment.  Outcomes are classified
as ``pass`` / ``skip`` / ``fail`` / ``warn`` / ``unmapped`` / ``na`` and
written to ``dev/backend_support_data.json``.

``dev/backend_support_render.py`` (this script's source) reads that
JSON and emits the rst file you are currently reading.  To refresh:

.. code-block:: bash

   PYTHONPATH=. python dev/backend_support_sweep.py
   PYTHONPATH=. python dev/backend_support_render.py

JAX-only subpackages are probed with one representative function per
subpackage rather than enumerated — every entry point in
``saiunit.lax`` / ``.autograd`` / ``.sparse`` is gated identically.

Footnotes
---------


.. [#fn-1] AttributeError: saiunit: backend 'ndonnx' has no operation 'absolute'
.. [#fn-2] AttributeError: module 'ndonnx' has no attribute `allclose`
.. [#fn-3] ValueError: 'complex128' does not have a corresponding ndonnx data type
.. [#fn-4] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'append'
.. [#fn-5] AttributeError: saiunit: backend 'ndonnx' has no operation 'append'
.. [#fn-6] AttributeError: saiunit: backend 'ndonnx' has no operation 'arccos'
.. [#fn-7] AttributeError: saiunit: backend 'ndonnx' has no operation 'arccosh'
.. [#fn-8] AttributeError: saiunit: backend 'ndonnx' has no operation 'arcsin'
.. [#fn-9] AttributeError: saiunit: backend 'ndonnx' has no operation 'arcsinh'
.. [#fn-10] AttributeError: saiunit: backend 'ndonnx' has no operation 'arctan'
.. [#fn-11] AttributeError: saiunit: backend 'ndonnx' has no operation 'arctan2'
.. [#fn-12] AttributeError: saiunit: backend 'ndonnx' has no operation 'arctanh'
.. [#fn-13] AttributeError: saiunit: backend 'ndonnx' has no operation 'argwhere'
.. [#fn-14] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'around'
.. [#fn-15] AttributeError: saiunit: backend 'ndonnx' has no operation 'around'
.. [#fn-16] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'array_equal'
.. [#fn-17] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'array_equal'
.. [#fn-18] AttributeError: saiunit: backend 'ndonnx' has no operation 'array_equal'
.. [#fn-19] TypeError: 'split' got an unexpected keyword argument 'axis'
.. [#fn-20] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'split'
.. [#fn-21] AttributeError: saiunit: backend 'ndonnx' has no operation 'split'
.. [#fn-22] AttributeError: saiunit: backend 'ndonnx' has no operation 'atleast_1d'
.. [#fn-23] AttributeError: saiunit: backend 'ndonnx' has no operation 'atleast_2d'
.. [#fn-24] AttributeError: saiunit: backend 'ndonnx' has no operation 'atleast_3d'
.. [#fn-25] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'average'
.. [#fn-26] AttributeError: saiunit: backend 'ndonnx' has no operation 'average'
.. [#fn-27] AttributeError: saiunit: backend 'ndonnx' has no operation 'bincount'
.. [#fn-28] AttributeError: saiunit: backend 'ndonnx' has no operation 'bitwise_not'
.. [#fn-29] AttributeError: saiunit: backend 'array_api_compat.cupy' has no operation 'block'
.. [#fn-30] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'block'
.. [#fn-31] AttributeError: saiunit: backend 'ndonnx' has no operation 'block'
.. [#fn-32] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'cbrt'
.. [#fn-33] AttributeError: saiunit: backend 'ndonnx' has no operation 'cbrt'
.. [#fn-34] expected numpy result backend; got jax
.. [#fn-35] AttributeError: 'list' object has no attribute 'shape'
.. [#fn-36] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'choose'
.. [#fn-37] TypeError: 'choose' got an unexpected keyword argument 'mode'
.. [#fn-38] AttributeError: saiunit: backend 'ndonnx' has no operation 'choose'
.. [#fn-39] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'column_stack'
.. [#fn-40] AttributeError: saiunit: backend 'ndonnx' has no operation 'column_stack'
.. [#fn-41] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'compress'
.. [#fn-42] AttributeError: saiunit: backend 'ndonnx' has no operation 'compress'
.. [#fn-43] AttributeError: saiunit: backend 'ndonnx' has no operation 'concatenate'
.. [#fn-44] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'conjugate'
.. [#fn-45] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'conjugate'
.. [#fn-46] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'convolve'
.. [#fn-47] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'convolve'
.. [#fn-48] AttributeError: saiunit: backend 'ndonnx' has no operation 'convolve'
.. [#fn-49] NotImplementedError:
.. [#fn-50] AttributeError: saiunit: backend 'ndonnx' has no operation 'corrcoef'
.. [#fn-51] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'correlate'
.. [#fn-52] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'correlate'
.. [#fn-53] AttributeError: saiunit: backend 'ndonnx' has no operation 'correlate'
.. [#fn-54] AttributeError: saiunit: backend 'ndonnx' has no operation 'cov'
.. [#fn-55] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'cross'
.. [#fn-56] AttributeError: saiunit: backend 'ndonnx' has no operation 'cross'
.. [#fn-57] AttributeError: saiunit: backend 'ndonnx' has no operation 'cumprod'
.. [#fn-58] AttributeError: saiunit: backend 'ndonnx' has no operation 'cumsum'
.. [#fn-59] AttributeError: saiunit: backend 'ndonnx' has no operation 'deg2rad'
.. [#fn-60] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'degrees'
.. [#fn-61] AttributeError: saiunit: backend 'ndonnx' has no operation 'degrees'
.. [#fn-62] AttributeError: module 'ndonnx' has no attribute `diag`
.. [#fn-63] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'diag_indices_from'
.. [#fn-64] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'diag_indices_from'
.. [#fn-65] AttributeError: saiunit: backend 'ndonnx' has no operation 'diag_indices_from'
.. [#fn-66] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'diagflat'
.. [#fn-67] AttributeError: saiunit: backend 'ndonnx' has no operation 'diagflat'
.. [#fn-68] AttributeError: saiunit: backend 'ndonnx' has no operation 'diagonal'
.. [#fn-69] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'digitize'
.. [#fn-70] AttributeError: saiunit: backend 'ndonnx' has no operation 'digitize'
.. [#fn-71] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'divmod'
.. [#fn-72] AttributeError: saiunit: backend 'ndonnx' has no operation 'divmod'
.. [#fn-73] AttributeError: saiunit: backend 'ndonnx' has no operation 'dot'
.. [#fn-74] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'dsplit'
.. [#fn-75] AttributeError: saiunit: backend 'ndonnx' has no operation 'dsplit'
.. [#fn-76] AttributeError: saiunit: backend 'ndonnx' has no operation 'dstack'
.. [#fn-77] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'ediff1d'
.. [#fn-78] AttributeError: saiunit: backend 'ndonnx' has no operation 'ediff1d'
.. [#fn-79] TypeError: Argument '[[1. 2.] [3. 4.]]' of type <class 'cupy.ndarray'> is not a valid JAX type.
.. [#fn-80] TypeError: Error interpreting argument to <function _einsum at 0x...> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path operands[0...
.. [#fn-81] TypeError: Argument 'dask.array<array, shape=(2, 2), dtype=float64, chunksize=(2, 2), chunktype=numpy.ndarray>' of type <class 'dask.array.core.Array'> is not a valid JAX type.
.. [#fn-82] TypeError: Error interpreting argument to <function _einsum at 0x...> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at path ope...
.. [#fn-83] TypeError: 'empty_like' got an unexpected keyword argument 'shape'
.. [#fn-84] AttributeError: saiunit: backend 'ndonnx' has no operation 'exp2'
.. [#fn-85] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'extract'
.. [#fn-86] AttributeError: saiunit: backend 'ndonnx' has no operation 'extract'
.. [#fn-87] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'fabs'
.. [#fn-88] AttributeError: saiunit: backend 'ndonnx' has no operation 'fabs'
.. [#fn-89] AttributeError: module 'array_api_compat.torch' has no attribute 'fill_diagonal'
.. [#fn-90] AttributeError: module 'array_api_compat.dask.array' has no attribute 'fill_diagonal'
.. [#fn-91] AttributeError: module 'ndonnx' has no attribute `fill_diagonal`
.. [#fn-92] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'flatnonzero'
.. [#fn-93] AttributeError: saiunit: backend 'ndonnx' has no operation 'flatnonzero'
.. [#fn-94] AttributeError: saiunit: backend 'ndonnx' has no operation 'fliplr'
.. [#fn-95] AttributeError: saiunit: backend 'ndonnx' has no operation 'flipud'
.. [#fn-96] AttributeError: saiunit: backend 'ndonnx' has no operation 'float_power'
.. [#fn-97] AttributeError: saiunit: backend 'ndonnx' has no operation 'fmax'
.. [#fn-98] AttributeError: saiunit: backend 'ndonnx' has no operation 'fmin'
.. [#fn-99] AttributeError: saiunit: backend 'ndonnx' has no operation 'fmod'
.. [#fn-100] AttributeError: saiunit: backend 'ndonnx' has no operation 'frexp'
.. [#fn-101] TypeError: 'full_like' got an unexpected keyword argument 'shape'
.. [#fn-102] NotImplementedError: Don't yet support nd fancy indexing
.. [#fn-103] AttributeError: 'Array' object has no attribute 'reshape'
.. [#fn-104] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'gcd'
.. [#fn-105] AttributeError: saiunit: backend 'ndonnx' has no operation 'gcd'
.. [#fn-106] AttributeError: module 'ndonnx' has no attribute `gradient`
.. [#fn-107] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'heaviside'
.. [#fn-108] AttributeError: saiunit: backend 'ndonnx' has no operation 'heaviside'
.. [#fn-109] expected dask result backend; got dask, numpy
.. [#fn-110] AttributeError: saiunit: backend 'ndonnx' has no operation 'histogram'
.. [#fn-111] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'hsplit'
.. [#fn-112] AttributeError: saiunit: backend 'ndonnx' has no operation 'hsplit'
.. [#fn-113] AttributeError: saiunit: backend 'ndonnx' has no operation 'hstack'
.. [#fn-114] AttributeError: module 'array_api_compat.torch' has no attribute 'identity'
.. [#fn-115] AttributeError: module 'array_api_compat.dask.array' has no attribute 'identity'
.. [#fn-116] AttributeError: module 'ndonnx' has no attribute `identity`
.. [#fn-117] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'inner'
.. [#fn-118] AttributeError: saiunit: backend 'ndonnx' has no operation 'inner'
.. [#fn-119] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'interp'
.. [#fn-120] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'interp'
.. [#fn-121] AttributeError: saiunit: backend 'ndonnx' has no operation 'interp'
.. [#fn-122] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'intersect1d'
.. [#fn-123] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'intersect1d'
.. [#fn-124] AttributeError: saiunit: backend 'ndonnx' has no operation 'intersect1d'
.. [#fn-125] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'invert'
.. [#fn-126] AttributeError: saiunit: backend 'ndonnx' has no operation 'invert'
.. [#fn-127] AttributeError: saiunit: backend 'ndonnx' has no operation 'isclose'
.. [#fn-128] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'iscomplexobj'
.. [#fn-129] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'iscomplexobj'
.. [#fn-130] AttributeError: saiunit: backend 'ndonnx' has no operation 'iscomplexobj'
.. [#fn-131] AttributeError: module 'ndonnx' has no attribute `isreal`
.. [#fn-132] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'kron'
.. [#fn-133] AttributeError: saiunit: backend 'ndonnx' has no operation 'kron'
.. [#fn-134] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'lcm'
.. [#fn-135] AttributeError: saiunit: backend 'ndonnx' has no operation 'lcm'
.. [#fn-136] AttributeError: saiunit: backend 'ndonnx' has no operation 'ldexp'
.. [#fn-137] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'left_shift'
.. [#fn-138] AttributeError: saiunit: backend 'ndonnx' has no operation 'left_shift'
.. [#fn-139] TypeError: 'linspace' got an unexpected keyword argument 'retstep'
.. [#fn-140] AttributeError: saiunit: backend 'ndonnx' has no operation 'logaddexp2'
.. [#fn-141] TypeError: logspace() received an invalid combination of arguments - got (float, float), but expected one of: \* (Tensor start, Tensor end, int steps, float base = 10.0, \*, Tensor out = None, torch....
.. [#fn-142] AttributeError: module 'array_api_compat.dask.array' has no attribute 'logspace'
.. [#fn-143] AttributeError: module 'ndonnx' has no attribute `logspace`
.. [#fn-144] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.matrix_power'
.. [#fn-145] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_power'
.. [#fn-146] TypeError: median() missing 1 required positional arguments: "dim"
.. [#fn-147] TypeError: 'median' got an unexpected keyword argument 'overwrite_input'
.. [#fn-148] AttributeError: saiunit: backend 'ndonnx' has no operation 'median'
.. [#fn-149] AttributeError: 'list' object has no attribute 'ndim'
.. [#fn-150] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'mod'
.. [#fn-151] AttributeError: saiunit: backend 'ndonnx' has no operation 'mod'
.. [#fn-152] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'modf'
.. [#fn-153] AttributeError: saiunit: backend 'ndonnx' has no operation 'modf'
.. [#fn-154] AttributeError: saiunit: backend 'array_api_compat.cupy' has no operation 'linalg.multi_dot'
.. [#fn-155] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.multi_dot'
.. [#fn-156] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.multi_dot'
.. [#fn-157] AttributeError: saiunit: backend 'ndonnx' has no operation 'nan_to_num'
.. [#fn-158] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanargmax'
.. [#fn-159] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanargmax'
.. [#fn-160] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanargmin'
.. [#fn-161] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanargmin'
.. [#fn-162] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nancumprod'
.. [#fn-163] AttributeError: saiunit: backend 'ndonnx' has no operation 'nancumprod'
.. [#fn-164] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nancumsum'
.. [#fn-165] AttributeError: saiunit: backend 'ndonnx' has no operation 'nancumsum'
.. [#fn-166] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanmax'
.. [#fn-167] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmax'
.. [#fn-168] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmean'
.. [#fn-169] TypeError: nanmedian() missing 1 required positional arguments: "dim"
.. [#fn-170] TypeError: 'nanmedian' got an unexpected keyword argument 'overwrite_input'
.. [#fn-171] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmedian'
.. [#fn-172] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanmin'
.. [#fn-173] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmin'
.. [#fn-174] AttributeError: saiunit: backend 'array_api_compat.cupy' has no operation 'nanpercentile'
.. [#fn-175] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanpercentile'
.. [#fn-176] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanpercentile'
.. [#fn-177] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanprod'
.. [#fn-178] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanprod'
.. [#fn-179] AttributeError: saiunit: backend 'array_api_compat.cupy' has no operation 'nanquantile'
.. [#fn-180] TypeError: nanquantile() received an invalid combination of arguments - got (Tensor), but expected one of: \* (Tensor input, Tensor q, int dim = None, bool keepdim = False, \*, str interpolation = "l...
.. [#fn-181] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanquantile'
.. [#fn-182] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanstd'
.. [#fn-183] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanstd'
.. [#fn-184] AttributeError: saiunit: backend 'ndonnx' has no operation 'nansum'
.. [#fn-185] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanvar'
.. [#fn-186] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanvar'
.. [#fn-187] TypeError: 'ones_like' got an unexpected keyword argument 'shape'
.. [#fn-188] AttributeError: saiunit: backend 'ndonnx' has no operation 'outer'
.. [#fn-189] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'percentile'
.. [#fn-190] AttributeError: saiunit: backend 'ndonnx' has no operation 'percentile'
.. [#fn-191] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'power'
.. [#fn-192] AttributeError: saiunit: backend 'ndonnx' has no operation 'power'
.. [#fn-193] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'ptp'
.. [#fn-194] TypeError: 'ptp' got an unexpected keyword argument 'keepdims'
.. [#fn-195] AttributeError: saiunit: backend 'ndonnx' has no operation 'ptp'
.. [#fn-196] TypeError: quantile() received an invalid combination of arguments - got (Tensor), but expected one of: \* (Tensor input, Tensor q, int dim = None, bool keepdim = False, \*, str interpolation = "line...
.. [#fn-197] AttributeError: saiunit: backend 'ndonnx' has no operation 'quantile'
.. [#fn-198] AttributeError: saiunit: backend 'ndonnx' has no operation 'rad2deg'
.. [#fn-199] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'radians'
.. [#fn-200] AttributeError: saiunit: backend 'ndonnx' has no operation 'radians'
.. [#fn-201] TypeError: 'ravel' got an unexpected keyword argument 'order'
.. [#fn-202] AttributeError: saiunit: backend 'ndonnx' has no operation 'ravel'
.. [#fn-203] TypeError: 'reshape' got an unexpected keyword argument 'order'
.. [#fn-204] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'right_shift'
.. [#fn-205] AttributeError: saiunit: backend 'ndonnx' has no operation 'right_shift'
.. [#fn-206] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'rint'
.. [#fn-207] AttributeError: saiunit: backend 'ndonnx' has no operation 'rint'
.. [#fn-208] AttributeError: saiunit: backend 'ndonnx' has no operation 'rot90'
.. [#fn-209] TypeError: 'round' got an unexpected keyword argument 'decimals'
.. [#fn-210] AttributeError: saiunit: backend 'ndonnx' has no operation 'vstack'
.. [#fn-211] TypeError: select(): argument 'input' (position 1) must be Tensor, not list
.. [#fn-212] AttributeError: saiunit: backend 'ndonnx' has no operation 'select'
.. [#fn-213] AttributeError: saiunit: backend 'ndonnx' has no operation 'sinc'
.. [#fn-214] TypeError: squeeze() missing 1 required positional argument: 'axis'
.. [#fn-215] AttributeError: saiunit: backend 'ndonnx' has no operation 'swapaxes'
.. [#fn-216] TypeError: 'take' got an unexpected keyword argument 'unique_indices'
.. [#fn-217] AttributeError: saiunit: backend 'ndonnx' has no operation 'trace'
.. [#fn-218] TypeError: transpose() missing 2 required positional argument: "dim0", "dim1"
.. [#fn-219] AttributeError: saiunit: backend 'ndonnx' has no operation 'transpose'
.. [#fn-220] TypeError: 'zeros_like' got an unexpected keyword argument 'shape'
.. [#fn-221] AttributeError: module 'array_api_compat.torch' has no attribute 'tri'
.. [#fn-222] AttributeError: module 'ndonnx' has no attribute `tri`
.. [#fn-223] AttributeError: module 'array_api_compat.torch' has no attribute 'tril_indices_from'
.. [#fn-224] AttributeError: module 'ndonnx' has no attribute `tril_indices_from`
.. [#fn-225] AttributeError: module 'array_api_compat.torch' has no attribute 'triu_indices_from'
.. [#fn-226] AttributeError: module 'ndonnx' has no attribute `triu_indices_from`
.. [#fn-227] AttributeError: saiunit: backend 'ndonnx' has no operation 'true_divide'
.. [#fn-228] TypeError: 'unique' got an unexpected keyword argument 'equal_nan'
.. [#fn-229] AttributeError: saiunit: backend 'ndonnx' has no operation 'unique'
.. [#fn-230] AttributeError: module 'array_api_compat.dask.array' has no attribute 'vander'
.. [#fn-231] AttributeError: module 'ndonnx' has no attribute `vander`
.. [#fn-232] AttributeError: saiunit: backend 'ndonnx' has no operation 'vdot'
.. [#fn-233] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'vsplit'
.. [#fn-234] AttributeError: saiunit: backend 'ndonnx' has no operation 'vsplit'
.. [#fn-235] expected cupy result backend; got jax
.. [#fn-236] expected torch result backend; got jax
.. [#fn-237] expected dask result backend; got jax
.. [#fn-238] expected ndonnx result backend; got jax
.. [#fn-239] TypeError: transpose(): argument 'dim0' (position 2) must be int, not list
.. [#fn-240] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.cond'
.. [#fn-241] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.cond'
.. [#fn-242] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.det'
.. [#fn-243] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.det'
.. [#fn-244] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.diagonal'
.. [#fn-245] JaxRuntimeError: This operation requires CUDA support from jaxlib or jax cuda plugin.
.. [#fn-246] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to eig at position 0.
.. [#fn-247] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to eig at position 0.
.. [#fn-248] TypeError: Value 'array(data: [[1.0, 2.0], [3.0, 4.0]], dtype=float64)' with dtype object is not a valid JAX array type. Only arrays of numeric types are supported by JAX.
.. [#fn-249] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'cupy.ndarray'> and was passed to transpose at position 0.
.. [#fn-250] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to transpose at position 0.
.. [#fn-251] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to transpose at position 0.
.. [#fn-252] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to transpose at position 0.
.. [#fn-253] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.inv'
.. [#fn-254] TypeError: lstsq() got an unexpected keyword argument 'rcond'
.. [#fn-255] AttributeError: module 'ndonnx' has no attribute `linalg`
.. [#fn-256] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_norm'
.. [#fn-257] TypeError: svd() got an unexpected keyword argument 'compute_uv'
.. [#fn-258] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_rank'
.. [#fn-259] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_transpose'
.. [#fn-260] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.norm'
.. [#fn-261] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.pinv'
.. [#fn-262] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.pinv'
.. [#fn-263] AttributeError: module 'array_api_compat.dask.array.linalg' has no attribute 'slogdet'
.. [#fn-264] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.solve'
.. [#fn-265] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to svd at position 0.
.. [#fn-266] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to svd at position 0.
.. [#fn-267] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.tensorinv'
.. [#fn-268] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.tensorinv'
.. [#fn-269] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.tensorsolve'
.. [#fn-270] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.tensorsolve'
.. [#fn-271] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.trace'
.. [#fn-272] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.vector_norm'
.. [#fn-273] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fft'
.. [#fn-274] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fft2'
.. [#fn-275] AttributeError: module 'ndonnx' has no attribute `fft`
.. [#fn-276] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fftn'
.. [#fn-277] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fftshift'
.. [#fn-278] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifft'
.. [#fn-279] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifft2'
.. [#fn-280] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifftn'
.. [#fn-281] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifftshift'
.. [#fn-282] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfft'
.. [#fn-283] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfft2'
.. [#fn-284] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfftn'
.. [#fn-285] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfft'
.. [#fn-286] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfft2'
.. [#fn-287] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfftn'
.. [#fn-288] AttributeError: module 'array_api_compat.torch' has no attribute 'copy'
.. [#fn-289] AttributeError: module 'array_api_compat.dask.array' has no attribute 'copy'
.. [#fn-290] AttributeError: module 'ndonnx' has no attribute `copy`
.. [#fn-291] InvalidInputException: Argument '[1. 2. 3.]' of type <class 'cupy.ndarray'> is not a valid JAX type.
.. [#fn-292] TypeError: Cannot interpret 'torch.float64' as a data type
.. [#fn-293] InvalidInputException: Argument 'dask.array<array, shape=(3,), dtype=float64, chunksize=(3,), chunktype=numpy.ndarray>' of type <class 'dask.array.core.Array'> is not a valid JAX type.
.. [#fn-294] TypeError: Cannot interpret 'Float64' as a data type
.. [#fn-295] TypeError: cross() got an unexpected keyword argument 'axisa'
.. [#fn-296] AttributeError: module 'array_api_compat.dask.array' has no attribute 'cross'
.. [#fn-297] AttributeError: module 'ndonnx' has no attribute `cross`
.. [#fn-298] RuntimeError: Unknown backend cuda. Available backends are ['cpu']
.. [#fn-299] TypeError: cumprod is not supported for quantities with units (has unit m), because each element of the result would have a different unit exponent. Use .prod() for a single reduction, or convert t...
.. [#fn-300] TypeError: cumsum() missing 1 required positional arguments: "dim"
.. [#fn-301] TypeError: diagonal() got an unexpected keyword argument 'axis1'
.. [#fn-302] AttributeError: module 'ndonnx' has no attribute `diagonal`
.. [#fn-303] TypeError: Error interpreting argument to <function dot at 0x...> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at path a. This...
.. [#fn-304] ValueError: The dtype of the original data is float32, while we got float64.
.. [#fn-305] BackendError: Quantity.at indexed-update is not supported on the ndonnx backend. Call .to_numpy() (or another concrete backend) on the input first.
.. [#fn-306] AttributeError: module 'array_api_compat.dask.array' has no attribute 'float16'
.. [#fn-307] AttributeError: 'Array' object has no attribute 'item'
.. [#fn-308] TypeError: nancumprod is not supported for quantities with units (has unit m), because each element of the result would have a different unit exponent. Use .nanprod() for a single reduction, or con...
.. [#fn-309] TypeError: Error interpreting argument to <function outer at 0x...> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at path a. Th...
.. [#fn-310] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'set'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-311] expected torch result backend; got numpy
.. [#fn-312] expected dask result backend; got numpy
.. [#fn-313] expected ndonnx result backend; got numpy
.. [#fn-314] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'add'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-315] NotImplementedError: `cupy_true_divide.at` is not supported yet
.. [#fn-316] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'divide'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy()...
.. [#fn-317] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'max'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-318] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'min'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-319] NotImplementedError: `cupy_multiply.at` is not supported yet
.. [#fn-320] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'multiply'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy...
.. [#fn-321] TypeError: split() got an unexpected keyword argument 'axis'
.. [#fn-322] AxisError: axis1: axis 0 is out of bounds for array of dimension 0
.. [#fn-323] TypeError: 'NoneType' object is not iterable
.. [#fn-324] TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
.. [#fn-325] AttributeError: module 'ndonnx' has no attribute `swapaxes`
.. [#fn-326] TypeError: Axis value must be an integer, got None
.. [#fn-327] TypeError: tile(): argument 'dims' (position 2) must be tuple of ints, not int
.. [#fn-328] TypeError: object of type 'int' has no len()
.. [#fn-329] ValueError: Unsupported dtype object
.. [#fn-330] TypeError: Value 'array(data: [1.0, 2.0, 3.0], dtype=float64)' with dtype object is not a valid JAX array type. Only arrays of numeric types are supported by JAX.
.. [#fn-331] ValueError: unable to infer dtype from `[1. 2. 3.]`
.. [#fn-332] ValueError: unable to infer dtype from `tensor([1., 2., 3.], dtype=torch.float64)`
.. [#fn-333] ValueError: unable to infer dtype from `dask.array<array, shape=(3,), dtype=float64, chunksize=(3,), chunktype=numpy.ndarray>`
.. [#fn-334] RuntimeError: Cannot initialize CUDA without ATen_cuda library. PyTorch splits its backend into two shared libraries: a CPU library and a CUDA library; this error has occurred because you are tryin...
.. [#fn-335] TypeError: len() of unsized object
.. [#fn-336] ValueError: ONNX provides no control over the used device
.. [#fn-337] BackendError (expected): Quantity.tolist() would materialize a dask-backed Quantity. Call `q.mantissa.compute()` first.
.. [#fn-338] AttributeError: 'Array' object has no attribute 'tolist'
.. [#fn-339] TypeError: trace() got an unexpected keyword argument 'offset'
.. [#fn-340] AttributeError: module 'ndonnx' has no attribute `trace`
.. [#fn-341] AttributeError: module 'ndonnx' has no attribute `transpose`
.. [#fn-342] TypeError: view() received an invalid combination of arguments - got (type), but expected one of: \* (torch.dtype dtype) didn't match because some of the arguments have invalid types: (!type!) \* (tu...
.. [#fn-343] AttributeError: 'Array' object has no attribute 'view'
