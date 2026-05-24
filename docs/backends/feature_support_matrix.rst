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
``?``       Not tested in this report. Cupy is always ``?`` because no CUDA backend
            was available when this sweep ran. The single unmapped Quantity method
            (``tree_unflatten``) is also ``?`` because automated invocation requires
            a hand-crafted aux/children pair.
==========  ==========================================================

**Sweep environment.**  Backends invoked: ``numpy``, ``jax``, ``torch``, ``dask``, ``ndonnx``.  
Backends shown but not tested: ``cupy``.

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
     - ?
     - Partial ⚠
     - Partial ⚠
     - Partial ⚠
   * - **saiunit.linalg**
     - Mostly ⚠
     - Full ✓
     - ?
     - Partial ⚠
     - Partial ⚠
     - Limited ✗
   * - **saiunit.fft**
     - Full ✓
     - Full ✓
     - ?
     - Partial ⚠
     - Full ✓
     - Limited ✗
   * - **Quantity methods**
     - Mostly ⚠
     - Mostly ⚠
     - ?
     - Partial ⚠
     - Partial ⚠
     - Limited ✗
   * - **saiunit.lax**
     - JAX-only 🅙
     - Full ✓
     - ?
     - JAX-only 🅙
     - JAX-only 🅙
     - JAX-only 🅙
   * - **saiunit.autograd**
     - JAX-only 🅙
     - Full ✓
     - ?
     - JAX-only 🅙
     - JAX-only 🅙
     - JAX-only 🅙
   * - **saiunit.sparse**
     - JAX-only 🅙
     - Full ✓
     - ?
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
- **cupy** — *not tested in this report* (no CUDA toolkit in the sweep
  environment).  Cells are ``?``.  Cupy's array-API surface tracks numpy
  closely, so support is expected to mirror the numpy column, but this
  document does not claim it.
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
     - ?
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
     - ?
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
     - ?
     - 🅙
     - 🅙
     - 🅙

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
     - ⊘ [#fn-1]_
     - ⊘ [#fn-1]_
     - ?
     - ⊘ [#fn-1]_
     - ⊘ [#fn-1]_
     - ⊘ [#fn-1]_
   * - ``triu_indices``
     - ⊘ [#fn-2]_
     - ⊘ [#fn-2]_
     - ?
     - ⊘ [#fn-2]_
     - ⊘ [#fn-2]_
     - ⊘ [#fn-2]_

``keep_unit`` — Unit-preserving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
     - ⊘ [#fn-3]_
     - ⊘ [#fn-3]_
     - ?
     - ⊘ [#fn-3]_
     - ⊘ [#fn-3]_
     - ⊘ [#fn-3]_

``change_unit`` — Unit-changing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*saiunit.math — Unit-changing: no functions in this group.*

``accept_unitless`` — Dimensionless-only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*saiunit.math — Dimensionless-only: no functions in this group.*

``remove_unit`` — Unit-removing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
     - ?
     - ✓
     - ✓
     - ✓
   * - ``absolute``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-4]_
   * - ``add``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``all``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-5]_
     - ⊘ [#fn-6]_
     - ⊘ [#fn-6]_
   * - ``allclose``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-7]_
   * - ``alltrue``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-5]_
     - ⊘ [#fn-6]_
     - ⊘ [#fn-6]_
   * - ``amax``
     - ✗ [#fn-8]_
     - ✓
     - ?
     - ⊘ [#fn-9]_
     - ⊘ [#fn-9]_
     - ⊘ [#fn-9]_
   * - ``amin``
     - ✗ [#fn-10]_
     - ✓
     - ?
     - ⊘ [#fn-11]_
     - ⊘ [#fn-11]_
     - ⊘ [#fn-11]_
   * - ``angle``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-12]_
   * - ``any``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-13]_
     - ⊘ [#fn-14]_
     - ⊘ [#fn-14]_
   * - ``append``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-15]_
     - ✓
     - ⊘ [#fn-16]_
   * - ``arange``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-17]_
     - ⊘ [#fn-18]_
     - ⊘ [#fn-19]_
   * - ``arccos``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-20]_
   * - ``arccosh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-21]_
   * - ``arcsin``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-22]_
   * - ``arcsinh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-23]_
   * - ``arctan``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-24]_
   * - ``arctan2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-25]_
   * - ``arctanh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-26]_
   * - ``argmax``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``argmin``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-27]_
     - ✓
     - ✗ [#fn-28]_
   * - ``argsort``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-29]_
     - ⊘ [#fn-30]_
     - ⊘ [#fn-30]_
   * - ``argwhere``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-31]_
   * - ``around``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-32]_
     - ✓
     - ⊘ [#fn-33]_
   * - ``array``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``array_equal``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-34]_
     - ⊘ [#fn-35]_
     - ⊘ [#fn-36]_
   * - ``array_split``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-37]_
     - ⊘ [#fn-38]_
     - ⊘ [#fn-39]_
   * - ``as_numpy``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``asarray``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``atleast_1d``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-40]_
   * - ``atleast_2d``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-41]_
   * - ``atleast_3d``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-42]_
   * - ``average``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-43]_
     - ✓
     - ⊘ [#fn-44]_
   * - ``bincount``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-45]_
   * - ``bitwise_and``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``bitwise_not``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-46]_
   * - ``bitwise_or``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``bitwise_xor``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``block``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-47]_
     - ✓
     - ⊘ [#fn-48]_
   * - ``broadcast_arrays``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``broadcast_shapes``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``broadcast_to``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``cbrt``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-49]_
     - ✓
     - ⊘ [#fn-50]_
   * - ``ceil``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``celu``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-51]_
     - ✗ [#fn-52]_
     - ✗ [#fn-53]_
   * - ``choose``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-54]_
     - ⊘ [#fn-55]_
     - ⊘ [#fn-56]_
   * - ``clip``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``column_stack``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-57]_
     - ⊘ [#fn-58]_
   * - ``compress``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-59]_
     - ✓
     - ⊘ [#fn-60]_
   * - ``concatenate``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-61]_
   * - ``conj``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-12]_
   * - ``conjugate``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-62]_
     - ⊘ [#fn-63]_
     - ✗ [#fn-12]_
   * - ``convolve``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-64]_
     - ⊘ [#fn-65]_
     - ⊘ [#fn-66]_
   * - ``copysign``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-67]_
   * - ``corrcoef``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-68]_
     - ✓
     - ⊘ [#fn-69]_
   * - ``correlate``
     - ⊘ [#fn-70]_
     - ✓
     - ?
     - ⊘ [#fn-71]_
     - ⊘ [#fn-72]_
     - ⊘ [#fn-73]_
   * - ``cos``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``cosh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``count_nonzero``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-28]_
   * - ``cov``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-74]_
     - ✓
     - ⊘ [#fn-75]_
   * - ``cross``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-76]_
     - ⊘ [#fn-77]_
     - ⊘ [#fn-78]_
   * - ``cumprod``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-79]_
     - ✓
     - ⊘ [#fn-80]_
   * - ``cumproduct``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-79]_
     - ✓
     - ⊘ [#fn-80]_
   * - ``cumsum``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-79]_
     - ✓
     - ⊘ [#fn-81]_
   * - ``deg2rad``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-82]_
   * - ``degrees``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-83]_
     - ✓
     - ⊘ [#fn-84]_
   * - ``diag``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-85]_
     - ✓
     - ⊘ [#fn-86]_
   * - ``diag_indices_from``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-87]_
     - ⊘ [#fn-88]_
     - ⊘ [#fn-89]_
   * - ``diagflat``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-90]_
     - ⊘ [#fn-91]_
     - ⊘ [#fn-92]_
   * - ``diagonal``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-93]_
     - ✓
     - ⊘ [#fn-94]_
   * - ``diff``
     - ⊘ [#fn-95]_
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``digitize``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-96]_
     - ✓
     - ⊘ [#fn-97]_
   * - ``divide``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``divmod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-98]_
     - ✓
     - ⊘ [#fn-99]_
   * - ``dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-100]_
   * - ``dsplit``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-101]_
     - ⊘ [#fn-102]_
   * - ``dstack``
     - ⊘ [#fn-103]_
     - ✓
     - ?
     - ⊘ [#fn-103]_
     - ⊘ [#fn-103]_
     - ⊘ [#fn-104]_
   * - ``ediff1d``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-105]_
     - ✓
     - ⊘ [#fn-106]_
   * - ``einsum``
     - ⊘ [#fn-107]_
     - ✓
     - ?
     - ⊘ [#fn-108]_
     - ⊘ [#fn-107]_
     - ⊘ [#fn-109]_
   * - ``elu``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-110]_
     - ✗ [#fn-111]_
     - ✗ [#fn-112]_
   * - ``empty``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``empty_like``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-113]_
     - ✓
     - ⊘ [#fn-113]_
   * - ``equal``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``exp``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``exp2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-114]_
   * - ``expand_dims``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``expm1``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-67]_
   * - ``exprel``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-115]_
   * - ``extract``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-116]_
     - ✓
     - ⊘ [#fn-117]_
   * - ``eye``
     - ✗ [#fn-118]_
     - ✓
     - ?
     - ✗ [#fn-119]_
     - ✗ [#fn-118]_
     - ✗ [#fn-119]_
   * - ``fabs``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-120]_
     - ✓
     - ⊘ [#fn-121]_
   * - ``fill_diagonal``
     - ⊘ [#fn-122]_
     - ✓
     - ?
     - ⊘ [#fn-123]_
     - ⊘ [#fn-124]_
     - ⊘ [#fn-125]_
   * - ``finfo``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``fix``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``flatnonzero``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-126]_
     - ✓
     - ⊘ [#fn-127]_
   * - ``flatten``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``flip``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``fliplr``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-128]_
   * - ``flipud``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-129]_
   * - ``float_power``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-130]_
   * - ``floor``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``floor_divide``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``fmax``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-131]_
   * - ``fmin``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-132]_
   * - ``fmod``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-133]_
   * - ``frexp``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-134]_
   * - ``from_numpy``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``full``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``full_like``
     - ⊘ [#fn-135]_
     - ⊘ [#fn-135]_
     - ?
     - ⊘ [#fn-135]_
     - ⊘ [#fn-135]_
     - ⊘ [#fn-135]_
   * - ``gather``
     - ⊘ [#fn-136]_
     - ⊘ [#fn-136]_
     - ?
     - ⊘ [#fn-136]_
     - ⊘ [#fn-136]_
     - ⊘ [#fn-136]_
   * - ``gcd``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-137]_
     - ⊘ [#fn-138]_
   * - ``gelu``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-139]_
     - ✗ [#fn-140]_
     - ✗ [#fn-141]_
   * - ``get_promote_dtypes``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``glu``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-142]_
     - ✗ [#fn-143]_
     - ✗ [#fn-144]_
   * - ``gradient``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-145]_
     - ✗ [#fn-146]_
     - ⊘ [#fn-147]_
   * - ``greater``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``greater_equal``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``hard_sigmoid``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-148]_
     - ✗ [#fn-149]_
     - ✗ [#fn-150]_
   * - ``hard_silu``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-151]_
     - ✗ [#fn-152]_
     - ✗ [#fn-153]_
   * - ``hard_swish``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-151]_
     - ✗ [#fn-152]_
     - ✗ [#fn-153]_
   * - ``hard_tanh``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-154]_
     - ✗ [#fn-155]_
     - ✗ [#fn-156]_
   * - ``heaviside``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-157]_
     - ⊘ [#fn-158]_
   * - ``histogram``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-159]_
     - ✗ [#fn-160]_
     - ⊘ [#fn-161]_
   * - ``hsplit``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-162]_
     - ⊘ [#fn-163]_
   * - ``hstack``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-164]_
     - ⊘ [#fn-164]_
     - ⊘ [#fn-165]_
   * - ``hypot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-67]_
   * - ``identity``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-166]_
     - ⊘ [#fn-167]_
     - ⊘ [#fn-168]_
   * - ``iinfo``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``imag``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-12]_
   * - ``inner``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-169]_
     - ⊘ [#fn-170]_
   * - ``interp``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-171]_
     - ⊘ [#fn-172]_
     - ⊘ [#fn-173]_
   * - ``intersect1d``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-174]_
     - ⊘ [#fn-175]_
     - ⊘ [#fn-176]_
   * - ``invert``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-177]_
     - ✓
     - ⊘ [#fn-178]_
   * - ``isclose``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-179]_
   * - ``iscomplexobj``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-180]_
     - ⊘ [#fn-181]_
     - ⊘ [#fn-182]_
   * - ``isfinite``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-183]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-185]_
   * - ``isinf``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-186]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-187]_
   * - ``isnan``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-188]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-189]_
   * - ``isreal``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-190]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-191]_
   * - ``isscalar``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``issubdtype``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``kron``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-192]_
     - ⊘ [#fn-193]_
   * - ``lcm``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-194]_
     - ⊘ [#fn-195]_
   * - ``ldexp``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-196]_
   * - ``leaky_relu``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``left_shift``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-197]_
     - ✓
     - ⊘ [#fn-198]_
   * - ``less``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``less_equal``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``linspace``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-199]_
     - ✓
     - ⊘ [#fn-200]_
   * - ``log``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``log10``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``log1p``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-67]_
   * - ``log2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``log_sigmoid``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-201]_
     - ✗ [#fn-202]_
     - ✗ [#fn-203]_
   * - ``logaddexp``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``logaddexp2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-204]_
   * - ``logical_and``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``logical_not``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``logical_or``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``logical_xor``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``logspace``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-205]_
     - ⊘ [#fn-206]_
     - ⊘ [#fn-207]_
   * - ``matmul``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``matrix_power``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-208]_
     - ⊘ [#fn-209]_
   * - ``max``
     - ✗ [#fn-8]_
     - ✓
     - ?
     - ⊘ [#fn-9]_
     - ⊘ [#fn-9]_
     - ⊘ [#fn-9]_
   * - ``maximum``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``mean``
     - ✗ [#fn-210]_
     - ✓
     - ?
     - ⊘ [#fn-211]_
     - ⊘ [#fn-212]_
     - ⊘ [#fn-213]_
   * - ``median``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-214]_
     - ⊘ [#fn-215]_
     - ⊘ [#fn-216]_
   * - ``meshgrid``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-217]_
   * - ``min``
     - ✗ [#fn-10]_
     - ✓
     - ?
     - ⊘ [#fn-11]_
     - ⊘ [#fn-11]_
     - ⊘ [#fn-11]_
   * - ``minimum``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``mish``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-218]_
     - ✗ [#fn-219]_
     - ✗ [#fn-220]_
   * - ``mod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-221]_
     - ✓
     - ⊘ [#fn-222]_
   * - ``modf``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-223]_
     - ✓
     - ⊘ [#fn-224]_
   * - ``moveaxis``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-225]_
   * - ``multi_dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-226]_
     - ⊘ [#fn-227]_
   * - ``multiply``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``nan_to_num``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-228]_
     - ⊘ [#fn-229]_
   * - ``nanargmax``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-230]_
     - ✓
     - ⊘ [#fn-231]_
   * - ``nanargmin``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-232]_
     - ✓
     - ⊘ [#fn-233]_
   * - ``nancumprod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-234]_
     - ✓
     - ⊘ [#fn-235]_
   * - ``nancumsum``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-236]_
     - ✓
     - ⊘ [#fn-237]_
   * - ``nanmax``
     - ✗ [#fn-238]_
     - ✓
     - ?
     - ⊘ [#fn-239]_
     - ⊘ [#fn-240]_
     - ⊘ [#fn-241]_
   * - ``nanmean``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-242]_
     - ⊘ [#fn-243]_
     - ⊘ [#fn-244]_
   * - ``nanmedian``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-245]_
     - ⊘ [#fn-246]_
     - ⊘ [#fn-247]_
   * - ``nanmin``
     - ✗ [#fn-248]_
     - ✓
     - ?
     - ⊘ [#fn-249]_
     - ⊘ [#fn-250]_
     - ⊘ [#fn-251]_
   * - ``nanpercentile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-252]_
     - ✓
     - ⊘ [#fn-253]_
   * - ``nanprod``
     - ✗ [#fn-254]_
     - ✓
     - ?
     - ⊘ [#fn-255]_
     - ⊘ [#fn-256]_
     - ⊘ [#fn-257]_
   * - ``nanquantile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-258]_
     - ✓
     - ⊘ [#fn-259]_
   * - ``nanstd``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-260]_
     - ⊘ [#fn-261]_
     - ⊘ [#fn-262]_
   * - ``nansum``
     - ✗ [#fn-263]_
     - ✓
     - ?
     - ⊘ [#fn-264]_
     - ⊘ [#fn-265]_
     - ⊘ [#fn-266]_
   * - ``nanvar``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-267]_
     - ⊘ [#fn-268]_
     - ⊘ [#fn-269]_
   * - ``ndim``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``negative``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``nextafter``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-67]_
   * - ``nonzero``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``not_equal``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``ones``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``ones_like``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-270]_
     - ✓
     - ⊘ [#fn-270]_
   * - ``outer``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-271]_
     - ⊘ [#fn-272]_
   * - ``percentile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-273]_
     - ⊘ [#fn-274]_
     - ⊘ [#fn-275]_
   * - ``positive``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``power``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-276]_
     - ✓
     - ⊘ [#fn-277]_
   * - ``prod``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``product``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``promote_dtypes``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``ptp``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-278]_
     - ⊘ [#fn-279]_
     - ⊘ [#fn-280]_
   * - ``quantile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-281]_
     - ✓
     - ⊘ [#fn-282]_
   * - ``rad2deg``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-283]_
   * - ``radians``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-284]_
     - ✓
     - ⊘ [#fn-285]_
   * - ``ravel``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-286]_
     - ⊘ [#fn-286]_
     - ⊘ [#fn-287]_
   * - ``real``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-12]_
   * - ``reciprocal``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``relu``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-288]_
     - ✗ [#fn-289]_
     - ✗ [#fn-290]_
   * - ``relu6``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-291]_
     - ✗ [#fn-292]_
     - ✗ [#fn-293]_
   * - ``remainder``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``remove_diag``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-294]_
     - ⊘ [#fn-295]_
   * - ``repeat``
     - ⊘ [#fn-296]_
     - ✓
     - ?
     - ✗ [#fn-297]_
     - ⊘ [#fn-296]_
     - ✗ [#fn-297]_
   * - ``reshape``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-298]_
     - ⊘ [#fn-298]_
     - ⊘ [#fn-298]_
   * - ``result_type``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``right_shift``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-299]_
     - ✓
     - ⊘ [#fn-300]_
   * - ``rint``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-301]_
     - ✓
     - ⊘ [#fn-302]_
   * - ``roll``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``rot90``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-303]_
     - ✓
     - ⊘ [#fn-304]_
   * - ``round``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-305]_
   * - ``row_stack``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-306]_
     - ⊘ [#fn-306]_
     - ⊘ [#fn-307]_
   * - ``searchsorted``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``select``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-308]_
     - ✓
     - ⊘ [#fn-309]_
   * - ``selu``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-310]_
     - ✗ [#fn-311]_
     - ✗ [#fn-312]_
   * - ``shape``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``sigmoid``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-313]_
     - ✗ [#fn-314]_
     - ✗ [#fn-315]_
   * - ``sign``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``signbit``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-67]_
   * - ``silu``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-316]_
     - ✗ [#fn-317]_
     - ✗ [#fn-318]_
   * - ``sin``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``sinc``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-319]_
   * - ``sinh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``size``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``soft_sign``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-320]_
     - ✗ [#fn-321]_
     - ✗ [#fn-322]_
   * - ``softplus``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-323]_
     - ✗ [#fn-324]_
     - ✗ [#fn-325]_
   * - ``sometrue``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-13]_
     - ⊘ [#fn-14]_
     - ⊘ [#fn-14]_
   * - ``sort``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-326]_
     - ⊘ [#fn-327]_
     - ⊘ [#fn-327]_
   * - ``sparse_plus``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-328]_
     - ✗ [#fn-329]_
     - ✗ [#fn-330]_
   * - ``sparse_sigmoid``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-331]_
     - ✗ [#fn-332]_
     - ✗ [#fn-333]_
   * - ``split``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-37]_
     - ⊘ [#fn-38]_
     - ⊘ [#fn-39]_
   * - ``sqrt``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``square``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``squareplus``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-334]_
     - ✗ [#fn-335]_
     - ✗ [#fn-336]_
   * - ``squeeze``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-337]_
     - ✓
     - ✗ [#fn-210]_
   * - ``stack``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-338]_
     - ⊘ [#fn-339]_
     - ⊘ [#fn-339]_
   * - ``std``
     - ⊘ [#fn-340]_
     - ✓
     - ?
     - ⊘ [#fn-341]_
     - ⊘ [#fn-342]_
     - ⊘ [#fn-343]_
   * - ``subtract``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``sum``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``swapaxes``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-344]_
     - ✓
     - ⊘ [#fn-345]_
   * - ``swish``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-346]_
     - ✗ [#fn-347]_
     - ✗ [#fn-348]_
   * - ``take``
     - ⊘ [#fn-349]_
     - ✓
     - ?
     - ⊘ [#fn-350]_
     - ⊘ [#fn-351]_
     - ⊘ [#fn-351]_
   * - ``tan``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``tanh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``tensordot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``tile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-352]_
     - ✓
     - ⊘ [#fn-353]_
   * - ``trace``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-354]_
     - ✓
     - ⊘ [#fn-355]_
   * - ``transpose``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-356]_
     - ✓
     - ⊘ [#fn-357]_
   * - ``tree_ones_like``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-270]_
     - ✓
     - ⊘ [#fn-270]_
   * - ``tree_zeros_like``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-358]_
     - ✓
     - ⊘ [#fn-358]_
   * - ``tri``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-359]_
     - ✗ [#fn-360]_
     - ⊘ [#fn-361]_
   * - ``tril``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``tril_indices_from``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-362]_
     - ✓
     - ⊘ [#fn-363]_
   * - ``triu``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``triu_indices_from``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-364]_
     - ✓
     - ⊘ [#fn-365]_
   * - ``true_divide``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-366]_
   * - ``trunc``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``unflatten``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``unique``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-367]_
     - ⊘ [#fn-368]_
     - ⊘ [#fn-369]_
   * - ``vander``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-370]_
     - ⊘ [#fn-371]_
   * - ``var``
     - ⊘ [#fn-372]_
     - ✓
     - ?
     - ⊘ [#fn-373]_
     - ⊘ [#fn-374]_
     - ⊘ [#fn-375]_
   * - ``vdot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-376]_
   * - ``vecdot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``vsplit``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-377]_
     - ⊘ [#fn-378]_
   * - ``vstack``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-306]_
     - ⊘ [#fn-306]_
     - ⊘ [#fn-307]_
   * - ``where``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``zeros``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``zeros_like``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-358]_
     - ✓
     - ⊘ [#fn-358]_

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
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``blackman``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``hamming``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``hanning``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``kaiser``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓

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
     - ?
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
     - ?
     - ⊘ [#fn-379]_
     - ✓
     - ⊘ [#fn-357]_
   * - ``einreduce``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``einrepeat``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-352]_
     - ✓
     - ⊘ [#fn-353]_
   * - ``einshape``
     - ✓
     - ✓
     - ?
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
     - ⊘ [#fn-380]_
     - ✓
     - ?
     - ⊘ [#fn-381]_
     - ⊘ [#fn-380]_
     - ⊘ [#fn-382]_
   * - ``cond``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-383]_
     - ⊘ [#fn-384]_
   * - ``cross``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-76]_
     - ⊘ [#fn-77]_
     - ⊘ [#fn-78]_
   * - ``det``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-385]_
     - ⊘ [#fn-386]_
   * - ``diagonal``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-93]_
     - ✓
     - ⊘ [#fn-94]_
   * - ``dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-100]_
   * - ``eig``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-387]_
     - ⊘ [#fn-388]_
     - ⊘ [#fn-389]_
   * - ``eigh``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-390]_
     - ⊘ [#fn-391]_
     - ⊘ [#fn-392]_
   * - ``eigvals``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-387]_
     - ⊘ [#fn-388]_
     - ⊘ [#fn-389]_
   * - ``eigvalsh``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-390]_
     - ⊘ [#fn-391]_
     - ⊘ [#fn-392]_
   * - ``inner``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-169]_
     - ⊘ [#fn-170]_
   * - ``inv``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-393]_
   * - ``kron``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-192]_
     - ⊘ [#fn-193]_
   * - ``lstsq``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-394]_
     - ⊘ [#fn-395]_
   * - ``matmul``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``matrix_norm``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-396]_
   * - ``matrix_power``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-208]_
     - ⊘ [#fn-209]_
   * - ``matrix_rank``
     - ⊘ [#fn-397]_
     - ✓
     - ?
     - ⊘ [#fn-398]_
     - ⊘ [#fn-397]_
     - ⊘ [#fn-399]_
   * - ``matrix_transpose``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-400]_
   * - ``multi_dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-226]_
     - ⊘ [#fn-227]_
   * - ``norm``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-401]_
   * - ``outer``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-271]_
     - ⊘ [#fn-272]_
   * - ``pinv``
     - ⊘ [#fn-402]_
     - ✓
     - ?
     - ⊘ [#fn-403]_
     - ⊘ [#fn-404]_
     - ⊘ [#fn-405]_
   * - ``qr``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-395]_
   * - ``slogdet``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-406]_
     - ⊘ [#fn-395]_
   * - ``solve``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-407]_
   * - ``svd``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-408]_
     - ⊘ [#fn-409]_
     - ⊘ [#fn-410]_
   * - ``svdvals``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-408]_
     - ⊘ [#fn-409]_
     - ⊘ [#fn-410]_
   * - ``tensordot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``tensorinv``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-411]_
     - ⊘ [#fn-412]_
   * - ``tensorsolve``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-413]_
     - ⊘ [#fn-414]_
     - ⊘ [#fn-415]_
   * - ``trace``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-354]_
     - ✓
     - ⊘ [#fn-355]_
   * - ``vdot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-376]_
   * - ``vecdot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``vector_norm``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-416]_

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
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-417]_
   * - ``fft2``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-418]_
     - ✓
     - ⊘ [#fn-419]_
   * - ``fftfreq``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-420]_
   * - ``fftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-421]_
   * - ``fftshift``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-422]_
   * - ``ifft``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-423]_
   * - ``ifft2``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-424]_
     - ✓
     - ⊘ [#fn-425]_
   * - ``ifftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-426]_
   * - ``ifftshift``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-427]_
   * - ``irfft``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-428]_
   * - ``irfft2``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-429]_
     - ✓
     - ⊘ [#fn-430]_
   * - ``irfftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-431]_
   * - ``rfft``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-432]_
   * - ``rfft2``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-433]_
     - ✓
     - ⊘ [#fn-434]_
   * - ``rfftfreq``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-420]_
   * - ``rfftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-435]_

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
     - ?
     - ⊘ [#fn-436]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-437]_
   * - ``any``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-438]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-439]_
   * - ``argmax``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-440]_
     - ✗ [#fn-441]_
     - ✗ [#fn-442]_
   * - ``argmin``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-443]_
     - ✗ [#fn-444]_
     - ✗ [#fn-445]_
   * - ``argsort``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-446]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-447]_
   * - ``astype``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-448]_
     - ✓
     - ⊘ [#fn-449]_
   * - ``clip``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``clone``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-450]_
     - ⊘ [#fn-451]_
     - ⊘ [#fn-452]_
   * - ``conj``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-67]_
   * - ``conjugate``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-67]_
   * - ``copy``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-450]_
     - ⊘ [#fn-451]_
     - ⊘ [#fn-452]_
   * - ``cpu``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-453]_
     - ✗ [#fn-454]_
     - ✗ [#fn-455]_
   * - ``cross``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-456]_
     - ⊘ [#fn-457]_
     - ⊘ [#fn-458]_
   * - ``cuda``
     - ⊘ [#fn-459]_
     - ⊘ [#fn-459]_
     - ?
     - ⊘ [#fn-459]_
     - ⊘ [#fn-459]_
     - ⊘ [#fn-459]_
   * - ``cumprod``
     - ⊘ [#fn-460]_
     - ⊘ [#fn-460]_
     - ?
     - ⊘ [#fn-460]_
     - ⊘ [#fn-460]_
     - ⊘ [#fn-460]_
   * - ``cumsum``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-461]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-462]_
   * - ``diagonal``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-463]_
     - ✓
     - ⊘ [#fn-464]_
   * - ``dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-465]_
   * - ``double``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``expand_as``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``expand_dims``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-466]_
   * - ``factorless``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``fill``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-453]_
     - ✗ [#fn-467]_
     - ✗ [#fn-468]_
   * - ``flatten``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``float``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``half``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-469]_
     - ✓
   * - ``has_same_unit``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``in_unit``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``item``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-470]_
     - ⊘ [#fn-470]_
   * - ``max``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-471]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-472]_
   * - ``mean``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-473]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-474]_
   * - ``min``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-475]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-476]_
   * - ``nancumprod``
     - ⊘ [#fn-477]_
     - ⊘ [#fn-477]_
     - ?
     - ⊘ [#fn-477]_
     - ⊘ [#fn-477]_
     - ⊘ [#fn-477]_
   * - ``nanprod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-478]_
     - ✓
     - ⊘ [#fn-479]_
   * - ``nonzero``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-480]_
     - ✗ [#fn-481]_
     - ✗ [#fn-482]_
   * - ``outer``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-483]_
   * - ``pow``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``prod``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``ptp``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-484]_
     - ✗ [#fn-485]_
     - ✗ [#fn-486]_
   * - ``put``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-453]_
     - ✗ [#fn-487]_
     - ✗ [#fn-468]_
   * - ``ravel``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-488]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-489]_
   * - ``repeat``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-297]_
     - ✓
     - ✗ [#fn-297]_
   * - ``repr_in_unit``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``reshape``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``resize``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-453]_
     - ✗ [#fn-467]_
     - ✗ [#fn-490]_
   * - ``round``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-491]_
     - ✓
     - ✗ [#fn-492]_
   * - ``scatter_add``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-493]_
     - ✗ [#fn-468]_
   * - ``scatter_div``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-494]_
     - ✗ [#fn-468]_
   * - ``scatter_max``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-495]_
     - ✗ [#fn-468]_
   * - ``scatter_min``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-496]_
     - ✗ [#fn-468]_
   * - ``scatter_mul``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-497]_
     - ✗ [#fn-468]_
   * - ``scatter_sub``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-493]_
     - ✗ [#fn-468]_
   * - ``searchsorted``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``sort``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-453]_
     - ✗ [#fn-467]_
     - ✗ [#fn-455]_
   * - ``split``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-37]_
     - ✓
     - ✗ [#fn-498]_
   * - ``squeeze``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-337]_
     - ✓
     - ✗ [#fn-210]_
   * - ``std``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-499]_
     - ✗ [#fn-500]_
     - ✗ [#fn-501]_
   * - ``sum``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-502]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-503]_
   * - ``swapaxes``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-504]_
   * - ``take``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-505]_
     - ✗ [#fn-184]_
     - ⊘ [#fn-506]_
   * - ``tile``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-507]_
     - ✓
     - ✗ [#fn-508]_
   * - ``to``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``to_cupy``
     - ⊘ [#fn-509]_
     - ⊘ [#fn-509]_
     - ?
     - ⊘ [#fn-509]_
     - ⊘ [#fn-509]_
     - ⊘ [#fn-509]_
   * - ``to_dask``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-453]_
     - ✓
     - ✗ [#fn-455]_
   * - ``to_decimal``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``to_jax``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-115]_
   * - ``to_ndonnx``
     - ✓
     - ⊘ [#fn-510]_
     - ?
     - ⊘ [#fn-511]_
     - ⊘ [#fn-512]_
     - ✓
   * - ``to_numpy``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``to_torch``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-513]_
     - ✗ [#fn-514]_
   * - ``tolist``
     - ✓
     - ✓
     - ?
     - ✓
     - ⚠ [#fn-515]_
     - ⊘ [#fn-516]_
   * - ``trace``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-517]_
     - ✓
     - ⊘ [#fn-518]_
   * - ``transpose``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-519]_
     - ✓
     - ⊘ [#fn-520]_
   * - ``tree_flatten``
     - ✓
     - ✓
     - ?
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
     - ?
     - ✓
     - ✓
     - ✗ [#fn-466]_
   * - ``update_mantissa``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-453]_
     - ✗ [#fn-467]_
     - ✗ [#fn-455]_
   * - ``var``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-521]_
     - ✗ [#fn-522]_
     - ✗ [#fn-523]_
   * - ``view``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-524]_
     - ✓
     - ⊘ [#fn-525]_
   * - ``with_unit``
     - ✓
     - ✓
     - ?
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
     - 0
     - 341
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


.. [#fn-1] TypeError: tril_indices() got an unexpected keyword argument 'M'. Did you mean 'm'?
.. [#fn-2] TypeError: triu_indices() got an unexpected keyword argument 'M'. Did you mean 'm'?
.. [#fn-3] TypeError: astype() missing 1 required positional argument: 'dtype'
.. [#fn-4] AttributeError: saiunit: backend 'ndonnx' has no operation 'absolute'
.. [#fn-5] TypeError: all() received an invalid combination of arguments - got (Tensor, where=NoneType), but expected one of: * (Tensor input, *, Tensor out = None) * (Tensor input, tuple of ints dim = None, ...
.. [#fn-6] TypeError: all() got an unexpected keyword argument 'where'
.. [#fn-7] AttributeError: module 'ndonnx' has no attribute `allclose`
.. [#fn-8] ValueError: reduction operation 'maximum' does not have an identity, so to use a where mask one has to specify 'initial'
.. [#fn-9] TypeError: max() got an unexpected keyword argument 'initial'
.. [#fn-10] ValueError: reduction operation 'minimum' does not have an identity, so to use a where mask one has to specify 'initial'
.. [#fn-11] TypeError: min() got an unexpected keyword argument 'initial'
.. [#fn-12] ValueError: 'complex128' does not have a corresponding ndonnx data type
.. [#fn-13] TypeError: any() received an invalid combination of arguments - got (Tensor, where=NoneType), but expected one of: * (Tensor input, *, Tensor out = None) * (Tensor input, tuple of ints dim = None, ...
.. [#fn-14] TypeError: any() got an unexpected keyword argument 'where'
.. [#fn-15] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'append'
.. [#fn-16] AttributeError: saiunit: backend 'ndonnx' has no operation 'append'
.. [#fn-17] TypeError: '>' not supported between instances of 'NoneType' and 'int'
.. [#fn-18] TypeError: An error occurred while calling the arange method registered to the numpy backend. Original Message: unsupported operand type(s) for /: 'int' and 'NoneType'
.. [#fn-19] ValueError: 'arange' is not implemented for the provided inputs
.. [#fn-20] AttributeError: saiunit: backend 'ndonnx' has no operation 'arccos'
.. [#fn-21] AttributeError: saiunit: backend 'ndonnx' has no operation 'arccosh'
.. [#fn-22] AttributeError: saiunit: backend 'ndonnx' has no operation 'arcsin'
.. [#fn-23] AttributeError: saiunit: backend 'ndonnx' has no operation 'arcsinh'
.. [#fn-24] AttributeError: saiunit: backend 'ndonnx' has no operation 'arctan'
.. [#fn-25] AttributeError: saiunit: backend 'ndonnx' has no operation 'arctan2'
.. [#fn-26] AttributeError: saiunit: backend 'ndonnx' has no operation 'arctanh'
.. [#fn-27] TypeError: argmin(): argument 'keepdim' must be bool, not NoneType
.. [#fn-28] TypeError: Unable to instantiate `AttrInt64` with value of type `NoneType`.
.. [#fn-29] TypeError: argsort() received an invalid combination of arguments - got (Tensor, order=NoneType, kind=NoneType, stable=bool, descending=bool, dim=int), but expected one of: * (Tensor input, *, bool...
.. [#fn-30] TypeError: argsort() got an unexpected keyword argument 'kind'
.. [#fn-31] AttributeError: saiunit: backend 'ndonnx' has no operation 'argwhere'
.. [#fn-32] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'around'
.. [#fn-33] AttributeError: saiunit: backend 'ndonnx' has no operation 'around'
.. [#fn-34] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'array_equal'
.. [#fn-35] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'array_equal'
.. [#fn-36] AttributeError: saiunit: backend 'ndonnx' has no operation 'array_equal'
.. [#fn-37] TypeError: split() got an unexpected keyword argument 'axis'
.. [#fn-38] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'split'
.. [#fn-39] AttributeError: saiunit: backend 'ndonnx' has no operation 'split'
.. [#fn-40] AttributeError: saiunit: backend 'ndonnx' has no operation 'atleast_1d'
.. [#fn-41] AttributeError: saiunit: backend 'ndonnx' has no operation 'atleast_2d'
.. [#fn-42] AttributeError: saiunit: backend 'ndonnx' has no operation 'atleast_3d'
.. [#fn-43] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'average'
.. [#fn-44] AttributeError: saiunit: backend 'ndonnx' has no operation 'average'
.. [#fn-45] AttributeError: saiunit: backend 'ndonnx' has no operation 'bincount'
.. [#fn-46] AttributeError: saiunit: backend 'ndonnx' has no operation 'bitwise_not'
.. [#fn-47] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'block'
.. [#fn-48] AttributeError: saiunit: backend 'ndonnx' has no operation 'block'
.. [#fn-49] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'cbrt'
.. [#fn-50] AttributeError: saiunit: backend 'ndonnx' has no operation 'cbrt'
.. [#fn-51] BackendError: saiunit.math.celu requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-52] BackendError: saiunit.math.celu requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-53] BackendError: saiunit.math.celu requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-54] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'choose'
.. [#fn-55] TypeError: choose() got an unexpected keyword argument 'mode'
.. [#fn-56] AttributeError: saiunit: backend 'ndonnx' has no operation 'choose'
.. [#fn-57] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'column_stack'
.. [#fn-58] AttributeError: saiunit: backend 'ndonnx' has no operation 'column_stack'
.. [#fn-59] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'compress'
.. [#fn-60] AttributeError: saiunit: backend 'ndonnx' has no operation 'compress'
.. [#fn-61] AttributeError: saiunit: backend 'ndonnx' has no operation 'concatenate'
.. [#fn-62] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'conjugate'
.. [#fn-63] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'conjugate'
.. [#fn-64] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'convolve'
.. [#fn-65] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'convolve'
.. [#fn-66] AttributeError: saiunit: backend 'ndonnx' has no operation 'convolve'
.. [#fn-67] NotImplementedError:
.. [#fn-68] TypeError: corrcoef() takes 1 positional argument but 2 were given
.. [#fn-69] AttributeError: saiunit: backend 'ndonnx' has no operation 'corrcoef'
.. [#fn-70] TypeError: correlate() got an unexpected keyword argument 'precision'
.. [#fn-71] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'correlate'
.. [#fn-72] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'correlate'
.. [#fn-73] AttributeError: saiunit: backend 'ndonnx' has no operation 'correlate'
.. [#fn-74] TypeError: cov() takes 1 positional argument but 2 were given
.. [#fn-75] AttributeError: saiunit: backend 'ndonnx' has no operation 'cov'
.. [#fn-76] TypeError: cross() got an unexpected keyword argument 'axis'
.. [#fn-77] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'cross'
.. [#fn-78] AttributeError: saiunit: backend 'ndonnx' has no operation 'cross'
.. [#fn-79] RuntimeError: Please look up dimensions by name, got: name = None.
.. [#fn-80] AttributeError: saiunit: backend 'ndonnx' has no operation 'cumprod'
.. [#fn-81] AttributeError: saiunit: backend 'ndonnx' has no operation 'cumsum'
.. [#fn-82] AttributeError: saiunit: backend 'ndonnx' has no operation 'deg2rad'
.. [#fn-83] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'degrees'
.. [#fn-84] AttributeError: saiunit: backend 'ndonnx' has no operation 'degrees'
.. [#fn-85] TypeError: diag() got an unexpected keyword argument 'k'
.. [#fn-86] AttributeError: module 'ndonnx' has no attribute `diag`
.. [#fn-87] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'diag_indices_from'
.. [#fn-88] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'diag_indices_from'
.. [#fn-89] AttributeError: saiunit: backend 'ndonnx' has no operation 'diag_indices_from'
.. [#fn-90] TypeError: diagflat() got an unexpected keyword argument 'k'
.. [#fn-91] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'diagflat'
.. [#fn-92] AttributeError: saiunit: backend 'ndonnx' has no operation 'diagflat'
.. [#fn-93] TypeError: diagonal() received an invalid combination of arguments - got (Tensor, offset=int, axis2=int, axis1=int), but expected one of: * (Tensor input, *, name outdim, name dim1, name dim2, int ...
.. [#fn-94] AttributeError: saiunit: backend 'ndonnx' has no operation 'diagonal'
.. [#fn-95] TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'
.. [#fn-96] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'digitize'
.. [#fn-97] AttributeError: saiunit: backend 'ndonnx' has no operation 'digitize'
.. [#fn-98] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'divmod'
.. [#fn-99] AttributeError: saiunit: backend 'ndonnx' has no operation 'divmod'
.. [#fn-100] AttributeError: saiunit: backend 'ndonnx' has no operation 'dot'
.. [#fn-101] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'dsplit'
.. [#fn-102] AttributeError: saiunit: backend 'ndonnx' has no operation 'dsplit'
.. [#fn-103] TypeError: dstack() got an unexpected keyword argument 'dtype'
.. [#fn-104] AttributeError: saiunit: backend 'ndonnx' has no operation 'dstack'
.. [#fn-105] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'ediff1d'
.. [#fn-106] AttributeError: saiunit: backend 'ndonnx' has no operation 'ediff1d'
.. [#fn-107] TracerArrayConversionError: The numpy.ndarray conversion method __array__() was called on traced array with shape float32[2,2] The error occurred while tracing the function _einsum at /mnt/d/codes/...
.. [#fn-108] TypeError: Error interpreting argument to <function _einsum at 0x76cfd68159e0> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path o...
.. [#fn-109] TypeError: Error interpreting argument to <function _einsum at 0x76cfd68159e0> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at...
.. [#fn-110] BackendError: saiunit.math.elu requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-111] BackendError: saiunit.math.elu requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-112] BackendError: saiunit.math.elu requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-113] TypeError: empty_like() got an unexpected keyword argument 'shape'
.. [#fn-114] AttributeError: saiunit: backend 'ndonnx' has no operation 'exp2'
.. [#fn-115] TypeError: Value 'array(data: [1.0, 2.0, 3.0], dtype=float64)' with dtype object is not a valid JAX array type. Only arrays of numeric types are supported by JAX.
.. [#fn-116] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'extract'
.. [#fn-117] AttributeError: saiunit: backend 'ndonnx' has no operation 'extract'
.. [#fn-118] TypeError: eye() takes from 1 to 2 positional arguments but 3 positional arguments (and 2 keyword-only arguments) were given
.. [#fn-119] TypeError: eye() takes from 1 to 2 positional arguments but 3 positional arguments (and 1 keyword-only argument) were given
.. [#fn-120] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'fabs'
.. [#fn-121] AttributeError: saiunit: backend 'ndonnx' has no operation 'fabs'
.. [#fn-122] TypeError: fill_diagonal() got an unexpected keyword argument 'inplace'
.. [#fn-123] AttributeError: module 'array_api_compat.torch' has no attribute 'fill_diagonal'
.. [#fn-124] AttributeError: module 'array_api_compat.dask.array' has no attribute 'fill_diagonal'
.. [#fn-125] AttributeError: module 'ndonnx' has no attribute `fill_diagonal`
.. [#fn-126] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'flatnonzero'
.. [#fn-127] AttributeError: saiunit: backend 'ndonnx' has no operation 'flatnonzero'
.. [#fn-128] AttributeError: saiunit: backend 'ndonnx' has no operation 'fliplr'
.. [#fn-129] AttributeError: saiunit: backend 'ndonnx' has no operation 'flipud'
.. [#fn-130] AttributeError: saiunit: backend 'ndonnx' has no operation 'float_power'
.. [#fn-131] AttributeError: saiunit: backend 'ndonnx' has no operation 'fmax'
.. [#fn-132] AttributeError: saiunit: backend 'ndonnx' has no operation 'fmin'
.. [#fn-133] AttributeError: saiunit: backend 'ndonnx' has no operation 'fmod'
.. [#fn-134] AttributeError: saiunit: backend 'ndonnx' has no operation 'frexp'
.. [#fn-135] TypeError: full_like() missing 1 required positional argument: 'fill_value'
.. [#fn-136] TypeError: gather() missing 1 required positional argument: 'index'
.. [#fn-137] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'gcd'
.. [#fn-138] AttributeError: saiunit: backend 'ndonnx' has no operation 'gcd'
.. [#fn-139] BackendError: saiunit.math.gelu requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-140] BackendError: saiunit.math.gelu requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-141] BackendError: saiunit.math.gelu requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-142] BackendError: saiunit.math.glu requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-143] BackendError: saiunit.math.glu requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-144] BackendError: saiunit.math.glu requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-145] TypeError: Error interpreting argument to <function gradient at 0x76cfd838eb60> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path ...
.. [#fn-146] TypeError: Argument 'dask.array<array, shape=(4,), dtype=float64, chunksize=(4,), chunktype=numpy.ndarray>' of type <class 'dask.array.core.Array'> is not a valid JAX type.
.. [#fn-147] TypeError: Error interpreting argument to <function gradient at 0x76cfd838eb60> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function a...
.. [#fn-148] BackendError: saiunit.math.hard_sigmoid requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-149] BackendError: saiunit.math.hard_sigmoid requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-150] BackendError: saiunit.math.hard_sigmoid requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-151] BackendError: saiunit.math.hard_silu requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-152] BackendError: saiunit.math.hard_silu requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-153] BackendError: saiunit.math.hard_silu requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-154] BackendError: saiunit.math.hard_tanh requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-155] BackendError: saiunit.math.hard_tanh requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-156] BackendError: saiunit.math.hard_tanh requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-157] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'heaviside'
.. [#fn-158] AttributeError: saiunit: backend 'ndonnx' has no operation 'heaviside'
.. [#fn-159] TypeError: histogram() received an invalid combination of arguments - got (Tensor, int, density=NoneType, weights=NoneType, range=NoneType), but expected one of: * (Tensor input, Tensor bins, *, Te...
.. [#fn-160] ValueError: dask.array.histogram requires either specifying bins as an iterable or specifying both a range and the number of bins
.. [#fn-161] AttributeError: saiunit: backend 'ndonnx' has no operation 'histogram'
.. [#fn-162] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'hsplit'
.. [#fn-163] AttributeError: saiunit: backend 'ndonnx' has no operation 'hsplit'
.. [#fn-164] TypeError: hstack() got an unexpected keyword argument 'dtype'
.. [#fn-165] AttributeError: saiunit: backend 'ndonnx' has no operation 'hstack'
.. [#fn-166] AttributeError: module 'array_api_compat.torch' has no attribute 'identity'
.. [#fn-167] AttributeError: module 'array_api_compat.dask.array' has no attribute 'identity'
.. [#fn-168] AttributeError: module 'ndonnx' has no attribute `identity`
.. [#fn-169] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'inner'
.. [#fn-170] AttributeError: saiunit: backend 'ndonnx' has no operation 'inner'
.. [#fn-171] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'interp'
.. [#fn-172] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'interp'
.. [#fn-173] AttributeError: saiunit: backend 'ndonnx' has no operation 'interp'
.. [#fn-174] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'intersect1d'
.. [#fn-175] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'intersect1d'
.. [#fn-176] AttributeError: saiunit: backend 'ndonnx' has no operation 'intersect1d'
.. [#fn-177] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'invert'
.. [#fn-178] AttributeError: saiunit: backend 'ndonnx' has no operation 'invert'
.. [#fn-179] AttributeError: saiunit: backend 'ndonnx' has no operation 'isclose'
.. [#fn-180] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'iscomplexobj'
.. [#fn-181] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'iscomplexobj'
.. [#fn-182] AttributeError: saiunit: backend 'ndonnx' has no operation 'iscomplexobj'
.. [#fn-183] TypeError: Error interpreting argument to <function isfinite at 0x76cfd835aac0> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path ...
.. [#fn-184] TypeError: Argument 'dask.array<array, shape=(3,), dtype=float64, chunksize=(3,), chunktype=numpy.ndarray>' of type <class 'dask.array.core.Array'> is not a valid JAX type.
.. [#fn-185] TypeError: Error interpreting argument to <function isfinite at 0x76cfd835aac0> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function a...
.. [#fn-186] TypeError: Error interpreting argument to <function isinf at 0x76cfd835aa20> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path x. ...
.. [#fn-187] TypeError: Error interpreting argument to <function isinf at 0x76cfd835aa20> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at p...
.. [#fn-188] TypeError: Error interpreting argument to <function isnan at 0x76cfd835b060> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path x. ...
.. [#fn-189] TypeError: Error interpreting argument to <function isnan at 0x76cfd835b060> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at p...
.. [#fn-190] TypeError: Error interpreting argument to <function isreal at 0x76cfd838e340> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path x....
.. [#fn-191] TypeError: Error interpreting argument to <function isreal at 0x76cfd838e340> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at ...
.. [#fn-192] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'kron'
.. [#fn-193] AttributeError: saiunit: backend 'ndonnx' has no operation 'kron'
.. [#fn-194] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'lcm'
.. [#fn-195] AttributeError: saiunit: backend 'ndonnx' has no operation 'lcm'
.. [#fn-196] AttributeError: saiunit: backend 'ndonnx' has no operation 'ldexp'
.. [#fn-197] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'left_shift'
.. [#fn-198] AttributeError: saiunit: backend 'ndonnx' has no operation 'left_shift'
.. [#fn-199] TypeError: linspace() received an invalid combination of arguments - got (float, float, int, retstep=bool, device=NoneType, dtype=NoneType), but expected one of: * (Tensor start, Tensor end, int st...
.. [#fn-200] TypeError: linspace() got an unexpected keyword argument 'retstep'
.. [#fn-201] BackendError: saiunit.math.log_sigmoid requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-202] BackendError: saiunit.math.log_sigmoid requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-203] BackendError: saiunit.math.log_sigmoid requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-204] AttributeError: saiunit: backend 'ndonnx' has no operation 'logaddexp2'
.. [#fn-205] TypeError: logspace() received an invalid combination of arguments - got (float, float, dtype=NoneType, base=float, endpoint=bool, num=int), but expected one of: * (Tensor start, Tensor end, int st...
.. [#fn-206] AttributeError: module 'array_api_compat.dask.array' has no attribute 'logspace'
.. [#fn-207] AttributeError: module 'ndonnx' has no attribute `logspace`
.. [#fn-208] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.matrix_power'
.. [#fn-209] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_power'
.. [#fn-210] TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
.. [#fn-211] TypeError: mean() received an invalid combination of arguments - got (Tensor, where=NoneType, dtype=NoneType), but expected one of: * (Tensor input, *, torch.dtype dtype = None, Tensor out = None) ...
.. [#fn-212] TypeError: mean() got an unexpected keyword argument 'where'
.. [#fn-213] TypeError: mean() got an unexpected keyword argument 'dtype'
.. [#fn-214] TypeError: median() received an invalid combination of arguments - got (Tensor, overwrite_input=bool, keepdims=bool, axis=NoneType), but expected one of: * (Tensor input) * (Tensor input, int dim, ...
.. [#fn-215] TypeError: median() got an unexpected keyword argument 'overwrite_input'
.. [#fn-216] AttributeError: saiunit: backend 'ndonnx' has no operation 'median'
.. [#fn-217] AttributeError: 'list' object has no attribute 'ndim'
.. [#fn-218] BackendError: saiunit.math.mish requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-219] BackendError: saiunit.math.mish requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-220] BackendError: saiunit.math.mish requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-221] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'mod'
.. [#fn-222] AttributeError: saiunit: backend 'ndonnx' has no operation 'mod'
.. [#fn-223] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'modf'
.. [#fn-224] AttributeError: saiunit: backend 'ndonnx' has no operation 'modf'
.. [#fn-225] TypeError: moveaxis() got some positional-only arguments passed as keyword arguments: 'source, destination'
.. [#fn-226] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.multi_dot'
.. [#fn-227] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.multi_dot'
.. [#fn-228] TypeError: nan_to_num does not take the following keyword arguments ['nan', 'neginf', 'posinf']
.. [#fn-229] AttributeError: saiunit: backend 'ndonnx' has no operation 'nan_to_num'
.. [#fn-230] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanargmax'
.. [#fn-231] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanargmax'
.. [#fn-232] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanargmin'
.. [#fn-233] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanargmin'
.. [#fn-234] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nancumprod'
.. [#fn-235] AttributeError: saiunit: backend 'ndonnx' has no operation 'nancumprod'
.. [#fn-236] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nancumsum'
.. [#fn-237] AttributeError: saiunit: backend 'ndonnx' has no operation 'nancumsum'
.. [#fn-238] ValueError: reduction operation 'fmax' does not have an identity, so to use a where mask one has to specify 'initial'
.. [#fn-239] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanmax'
.. [#fn-240] TypeError: nanmax() got an unexpected keyword argument 'initial'
.. [#fn-241] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmax'
.. [#fn-242] TypeError: nanmean() got an unexpected keyword argument 'axis'
.. [#fn-243] TypeError: nanmean() got an unexpected keyword argument 'where'
.. [#fn-244] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmean'
.. [#fn-245] TypeError: nanmedian() received an invalid combination of arguments - got (Tensor, overwrite_input=bool, keepdims=bool, axis=NoneType), but expected one of: * (Tensor input) * (Tensor input, int di...
.. [#fn-246] TypeError: nanmedian() got an unexpected keyword argument 'overwrite_input'
.. [#fn-247] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmedian'
.. [#fn-248] ValueError: reduction operation 'fmin' does not have an identity, so to use a where mask one has to specify 'initial'
.. [#fn-249] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanmin'
.. [#fn-250] TypeError: nanmin() got an unexpected keyword argument 'initial'
.. [#fn-251] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmin'
.. [#fn-252] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanpercentile'
.. [#fn-253] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanpercentile'
.. [#fn-254] ValueError: reduction operation 'multiply' does not have an identity, so to use a where mask one has to specify 'initial'
.. [#fn-255] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanprod'
.. [#fn-256] TypeError: nanprod() got an unexpected keyword argument 'initial'
.. [#fn-257] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanprod'
.. [#fn-258] TypeError: nanquantile() received an invalid combination of arguments - got (Tensor, q=float, method=str, keepdims=bool, axis=NoneType), but expected one of: * (Tensor input, Tensor q, int dim = No...
.. [#fn-259] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanquantile'
.. [#fn-260] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanstd'
.. [#fn-261] TypeError: nanstd() got an unexpected keyword argument 'where'
.. [#fn-262] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanstd'
.. [#fn-263] ValueError: reduction operation 'add' does not have an identity, so to use a where mask one has to specify 'initial'
.. [#fn-264] TypeError: nansum() got an unexpected keyword argument 'axis'
.. [#fn-265] TypeError: nansum() got an unexpected keyword argument 'initial'
.. [#fn-266] AttributeError: saiunit: backend 'ndonnx' has no operation 'nansum'
.. [#fn-267] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanvar'
.. [#fn-268] TypeError: nanvar() got an unexpected keyword argument 'where'
.. [#fn-269] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanvar'
.. [#fn-270] TypeError: ones_like() got an unexpected keyword argument 'shape'
.. [#fn-271] TypeError: outer() got an unexpected keyword argument 'out'
.. [#fn-272] AttributeError: saiunit: backend 'ndonnx' has no operation 'outer'
.. [#fn-273] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'percentile'
.. [#fn-274] TypeError: percentile() got an unexpected keyword argument dict_keys(['axis', 'keepdims'])
.. [#fn-275] AttributeError: saiunit: backend 'ndonnx' has no operation 'percentile'
.. [#fn-276] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'power'
.. [#fn-277] AttributeError: saiunit: backend 'ndonnx' has no operation 'power'
.. [#fn-278] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'ptp'
.. [#fn-279] TypeError: ptp() got an unexpected keyword argument 'keepdims'
.. [#fn-280] AttributeError: saiunit: backend 'ndonnx' has no operation 'ptp'
.. [#fn-281] TypeError: quantile() received an invalid combination of arguments - got (Tensor, q=float, method=str, keepdims=bool, axis=NoneType), but expected one of: * (Tensor input, Tensor q, int dim = None,...
.. [#fn-282] AttributeError: saiunit: backend 'ndonnx' has no operation 'quantile'
.. [#fn-283] AttributeError: saiunit: backend 'ndonnx' has no operation 'rad2deg'
.. [#fn-284] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'radians'
.. [#fn-285] AttributeError: saiunit: backend 'ndonnx' has no operation 'radians'
.. [#fn-286] TypeError: ravel() got an unexpected keyword argument 'order'
.. [#fn-287] AttributeError: saiunit: backend 'ndonnx' has no operation 'ravel'
.. [#fn-288] BackendError: saiunit.math.relu requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-289] BackendError: saiunit.math.relu requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-290] BackendError: saiunit.math.relu requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-291] BackendError: saiunit.math.relu6 requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-292] BackendError: saiunit.math.relu6 requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-293] BackendError: saiunit.math.relu6 requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-294] ValueError: Array chunk size or shape is unknown. shape: (nan,) Possible solution with x.compute_chunk_sizes()
.. [#fn-295] AttributeError: type object 'bool' has no attribute '__ndx_create__'
.. [#fn-296] TypeError: repeat() got an unexpected keyword argument 'total_repeat_length'
.. [#fn-297] TypeError: repeat() got some positional-only arguments passed as keyword arguments: 'repeats'
.. [#fn-298] TypeError: reshape() got an unexpected keyword argument 'order'
.. [#fn-299] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'right_shift'
.. [#fn-300] AttributeError: saiunit: backend 'ndonnx' has no operation 'right_shift'
.. [#fn-301] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'rint'
.. [#fn-302] AttributeError: saiunit: backend 'ndonnx' has no operation 'rint'
.. [#fn-303] TypeError: rot90() got an unexpected keyword argument 'axes'
.. [#fn-304] AttributeError: saiunit: backend 'ndonnx' has no operation 'rot90'
.. [#fn-305] TypeError: round() got an unexpected keyword argument 'decimals'
.. [#fn-306] TypeError: vstack() got an unexpected keyword argument 'dtype'
.. [#fn-307] AttributeError: saiunit: backend 'ndonnx' has no operation 'vstack'
.. [#fn-308] TypeError: select() received an invalid combination of arguments - got (list, list, default=int), but expected one of: * (Tensor input, name dim, int index) didn't match because some of the keyword...
.. [#fn-309] AttributeError: saiunit: backend 'ndonnx' has no operation 'select'
.. [#fn-310] BackendError: saiunit.math.selu requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-311] BackendError: saiunit.math.selu requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-312] BackendError: saiunit.math.selu requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-313] BackendError: saiunit.math.sigmoid requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-314] BackendError: saiunit.math.sigmoid requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-315] BackendError: saiunit.math.sigmoid requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-316] BackendError: saiunit.math.silu requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-317] BackendError: saiunit.math.silu requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-318] BackendError: saiunit.math.silu requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-319] AttributeError: saiunit: backend 'ndonnx' has no operation 'sinc'
.. [#fn-320] BackendError: saiunit.math.soft_sign requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-321] BackendError: saiunit.math.soft_sign requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-322] BackendError: saiunit.math.soft_sign requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-323] BackendError: saiunit.math.softplus requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-324] BackendError: saiunit.math.softplus requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-325] BackendError: saiunit.math.softplus requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-326] TypeError: sort() received an invalid combination of arguments - got (Tensor, order=NoneType, kind=NoneType, stable=bool, descending=bool, dim=int), but expected one of: * (Tensor input, *, bool st...
.. [#fn-327] TypeError: sort() got an unexpected keyword argument 'kind'
.. [#fn-328] BackendError: saiunit.math.sparse_plus requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-329] BackendError: saiunit.math.sparse_plus requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-330] BackendError: saiunit.math.sparse_plus requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-331] BackendError: saiunit.math.sparse_sigmoid requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-332] BackendError: saiunit.math.sparse_sigmoid requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-333] BackendError: saiunit.math.sparse_sigmoid requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-334] BackendError: saiunit.math.squareplus requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-335] BackendError: saiunit.math.squareplus requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-336] BackendError: saiunit.math.squareplus requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-337] TypeError: 'NoneType' object is not iterable
.. [#fn-338] TypeError: stack() got an unexpected keyword argument 'axis'
.. [#fn-339] TypeError: stack() got an unexpected keyword argument 'dtype'
.. [#fn-340] TypeError: numpy.std() got multiple values for keyword argument 'ddof'
.. [#fn-341] TypeError: std() received an invalid combination of arguments - got (Tensor, tuple, where=NoneType, dtype=NoneType, ddof=int, correction=int), but expected one of: * (Tensor input, tuple of ints di...
.. [#fn-342] TypeError: dask.array.reductions.std() got multiple values for keyword argument 'ddof'
.. [#fn-343] TypeError: std() got an unexpected keyword argument 'ddof'
.. [#fn-344] TypeError: swapaxes() missing 2 required positional argument: "axis0", "axis1"
.. [#fn-345] AttributeError: saiunit: backend 'ndonnx' has no operation 'swapaxes'
.. [#fn-346] BackendError: saiunit.math.swish requires the jax backend; got torch tensor. Convert to a JAX array first.
.. [#fn-347] BackendError: saiunit.math.swish requires the jax backend; got dask array. Convert to a JAX array first.
.. [#fn-348] BackendError: saiunit.math.swish requires the jax backend; got ndonnx array. Convert to a JAX array first.
.. [#fn-349] TypeError: take() got an unexpected keyword argument 'unique_indices'
.. [#fn-350] TypeError: index_select() received an invalid combination of arguments - got (Tensor, int, Tensor, fill_value=NoneType, indices_are_sorted=bool, unique_indices=bool, mode=NoneType), but expected on...
.. [#fn-351] TypeError: take() got an unexpected keyword argument 'mode'
.. [#fn-352] TypeError: tile() missing 1 required positional arguments: "dims"
.. [#fn-353] TypeError: tile() got an unexpected keyword argument 'reps'
.. [#fn-354] TypeError: trace() got an unexpected keyword argument 'axis1'
.. [#fn-355] AttributeError: saiunit: backend 'ndonnx' has no operation 'trace'
.. [#fn-356] TypeError: transpose() received an invalid combination of arguments - got (Tensor, axes=NoneType), but expected one of: * (Tensor input, int dim0, int dim1) * (Tensor input, name dim0, name dim1)
.. [#fn-357] AttributeError: saiunit: backend 'ndonnx' has no operation 'transpose'
.. [#fn-358] TypeError: zeros_like() got an unexpected keyword argument 'shape'
.. [#fn-359] AttributeError: module 'array_api_compat.torch' has no attribute 'tri'
.. [#fn-360] TypeError: dtype must be known for auto-chunking
.. [#fn-361] AttributeError: module 'ndonnx' has no attribute `tri`
.. [#fn-362] AttributeError: module 'array_api_compat.torch' has no attribute 'tril_indices_from'
.. [#fn-363] AttributeError: module 'ndonnx' has no attribute `tril_indices_from`
.. [#fn-364] AttributeError: module 'array_api_compat.torch' has no attribute 'triu_indices_from'
.. [#fn-365] AttributeError: module 'ndonnx' has no attribute `triu_indices_from`
.. [#fn-366] AttributeError: saiunit: backend 'ndonnx' has no operation 'true_divide'
.. [#fn-367] TypeError: _return_output() got an unexpected keyword argument 'return_index'. Did you mean 'return_inverse'?
.. [#fn-368] TypeError: unique() got an unexpected keyword argument 'axis'
.. [#fn-369] AttributeError: saiunit: backend 'ndonnx' has no operation 'unique'
.. [#fn-370] AttributeError: module 'array_api_compat.dask.array' has no attribute 'vander'
.. [#fn-371] AttributeError: module 'ndonnx' has no attribute `vander`
.. [#fn-372] TypeError: numpy.var() got multiple values for keyword argument 'ddof'
.. [#fn-373] TypeError: var() received an invalid combination of arguments - got (Tensor, tuple, where=NoneType, dtype=NoneType, ddof=int, correction=float), but expected one of: * (Tensor input, tuple of ints ...
.. [#fn-374] TypeError: dask.array.reductions.var() got multiple values for keyword argument 'ddof'
.. [#fn-375] TypeError: var() got an unexpected keyword argument 'ddof'
.. [#fn-376] AttributeError: saiunit: backend 'ndonnx' has no operation 'vdot'
.. [#fn-377] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'vsplit'
.. [#fn-378] AttributeError: saiunit: backend 'ndonnx' has no operation 'vsplit'
.. [#fn-379] TypeError: transpose() received an invalid combination of arguments - got (Tensor, axes=list), but expected one of: * (Tensor input, int dim0, int dim1) * (Tensor input, name dim0, name dim1)
.. [#fn-380] TypeError: cholesky() got an unexpected keyword argument 'symmetrize_input'
.. [#fn-381] TypeError: linalg_cholesky() got an unexpected keyword argument 'symmetrize_input'
.. [#fn-382] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.cholesky'
.. [#fn-383] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.cond'
.. [#fn-384] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.cond'
.. [#fn-385] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.det'
.. [#fn-386] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.det'
.. [#fn-387] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to eig at position 0.
.. [#fn-388] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to eig at position 0.
.. [#fn-389] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to eig at position 0.
.. [#fn-390] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to transpose at position 0.
.. [#fn-391] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to transpose at position 0.
.. [#fn-392] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to transpose at position 0.
.. [#fn-393] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.inv'
.. [#fn-394] TypeError: lstsq() got an unexpected keyword argument 'rcond'
.. [#fn-395] AttributeError: module 'ndonnx' has no attribute `linalg`
.. [#fn-396] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_norm'
.. [#fn-397] TypeError: svdvals() got an unexpected keyword argument 'tol'
.. [#fn-398] TypeError: linalg_matrix_rank() received an invalid combination of arguments - got (Tensor, tol=NoneType, rtol=NoneType), but expected one of: * (Tensor input, *, Tensor atol = None, Tensor rtol = ...
.. [#fn-399] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_rank'
.. [#fn-400] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_transpose'
.. [#fn-401] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.norm'
.. [#fn-402] TypeError: numpy.linalg.pinv() got multiple values for keyword argument 'rcond'
.. [#fn-403] TypeError: linalg_pinv() received an invalid combination of arguments - got (Tensor, rtol=NoneType, rcond=NoneType, hermitian=bool), but expected one of: * (Tensor input, *, Tensor atol = None, Ten...
.. [#fn-404] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.pinv'
.. [#fn-405] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.pinv'
.. [#fn-406] AttributeError: module 'array_api_compat.dask.array.linalg' has no attribute 'slogdet'
.. [#fn-407] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.solve'
.. [#fn-408] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to svd at position 0.
.. [#fn-409] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to svd at position 0.
.. [#fn-410] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to svd at position 0.
.. [#fn-411] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.tensorinv'
.. [#fn-412] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.tensorinv'
.. [#fn-413] TypeError: linalg_tensorsolve() got an unexpected keyword argument 'axes'
.. [#fn-414] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.tensorsolve'
.. [#fn-415] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.tensorsolve'
.. [#fn-416] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.vector_norm'
.. [#fn-417] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fft'
.. [#fn-418] TypeError: fft_fft2() got an unexpected keyword argument 'axes'
.. [#fn-419] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fft2'
.. [#fn-420] AttributeError: module 'ndonnx' has no attribute `fft`
.. [#fn-421] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fftn'
.. [#fn-422] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fftshift'
.. [#fn-423] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifft'
.. [#fn-424] TypeError: fft_ifft2() got an unexpected keyword argument 'axes'
.. [#fn-425] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifft2'
.. [#fn-426] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifftn'
.. [#fn-427] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifftshift'
.. [#fn-428] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfft'
.. [#fn-429] TypeError: fft_irfft2() got an unexpected keyword argument 'axes'
.. [#fn-430] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfft2'
.. [#fn-431] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfftn'
.. [#fn-432] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfft'
.. [#fn-433] TypeError: fft_rfft2() got an unexpected keyword argument 'axes'
.. [#fn-434] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfft2'
.. [#fn-435] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfftn'
.. [#fn-436] TypeError: Error interpreting argument to <function _reduce_all at 0x76cfd82d0680> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at pa...
.. [#fn-437] TypeError: Error interpreting argument to <function _reduce_all at 0x76cfd82d0680> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the functio...
.. [#fn-438] TypeError: Error interpreting argument to <function _reduce_any at 0x76cfd82d0900> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at pa...
.. [#fn-439] TypeError: Error interpreting argument to <function _reduce_any at 0x76cfd82d0900> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the functio...
.. [#fn-440] TypeError: argmax requires ndarray or scalar arguments, got <class 'torch.Tensor'> at position 0.
.. [#fn-441] TypeError: argmax requires ndarray or scalar arguments, got <class 'dask.array.core.Array'> at position 0.
.. [#fn-442] TypeError: argmax requires ndarray or scalar arguments, got <class 'ndonnx._array.Array'> at position 0.
.. [#fn-443] TypeError: argmin requires ndarray or scalar arguments, got <class 'torch.Tensor'> at position 0.
.. [#fn-444] TypeError: argmin requires ndarray or scalar arguments, got <class 'dask.array.core.Array'> at position 0.
.. [#fn-445] TypeError: argmin requires ndarray or scalar arguments, got <class 'ndonnx._array.Array'> at position 0.
.. [#fn-446] TypeError: Error interpreting argument to <function argsort at 0x76cfd837bec0> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path a...
.. [#fn-447] TypeError: Error interpreting argument to <function argsort at 0x76cfd837bec0> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at...
.. [#fn-448] TypeError: to() received an invalid combination of arguments - got (copy=bool, dtype=type, ), but expected one of: * (torch.device device = None, torch.dtype dtype = None, bool non_blocking = False...
.. [#fn-449] AttributeError: type object 'numpy.float32' has no attribute '__ndx_cast_from__'
.. [#fn-450] AttributeError: module 'array_api_compat.torch' has no attribute 'copy'
.. [#fn-451] AttributeError: module 'array_api_compat.dask.array' has no attribute 'copy'
.. [#fn-452] AttributeError: module 'ndonnx' has no attribute `copy`
.. [#fn-453] TypeError: Cannot interpret 'torch.float64' as a data type
.. [#fn-454] InvalidInputException: Argument 'dask.array<array, shape=(3,), dtype=float64, chunksize=(3,), chunktype=numpy.ndarray>' of type <class 'dask.array.core.Array'> is not a valid JAX type.
.. [#fn-455] TypeError: Cannot interpret 'Float64' as a data type
.. [#fn-456] TypeError: cross() got an unexpected keyword argument 'axisa'
.. [#fn-457] AttributeError: module 'array_api_compat.dask.array' has no attribute 'cross'
.. [#fn-458] AttributeError: module 'ndonnx' has no attribute `cross`
.. [#fn-459] RuntimeError: Unknown backend cuda. Available backends are ['cpu']
.. [#fn-460] TypeError: cumprod is not supported for quantities with units (has unit m), because each element of the result would have a different unit exponent. Use .prod() for a single reduction, or convert t...
.. [#fn-461] TypeError: Error interpreting argument to <function cumsum at 0x76cfd82d3f60> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path a....
.. [#fn-462] TypeError: Error interpreting argument to <function cumsum at 0x76cfd82d3f60> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at ...
.. [#fn-463] TypeError: diagonal() received an invalid combination of arguments - got (Tensor, axis1=int, axis2=int, offset=int), but expected one of: * (Tensor input, *, name outdim, name dim1, name dim2, int ...
.. [#fn-464] AttributeError: module 'ndonnx' has no attribute `diagonal`
.. [#fn-465] TypeError: Error interpreting argument to <function dot at 0x76cfd837a7a0> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at pat...
.. [#fn-466] TypeError: expand_dims() takes 1 positional argument but 2 were given
.. [#fn-467] ValueError: The dtype of the original data is float32, while we got float64.
.. [#fn-468] BackendError: Quantity.at indexed-update is not supported on the ndonnx backend. Call .to_numpy() (or another concrete backend) on the input first.
.. [#fn-469] AttributeError: module 'array_api_compat.dask.array' has no attribute 'float16'
.. [#fn-470] AttributeError: 'Array' object has no attribute 'item'
.. [#fn-471] TypeError: Error interpreting argument to <function _reduce_max at 0x76cfd82d0180> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at pa...
.. [#fn-472] TypeError: Error interpreting argument to <function _reduce_max at 0x76cfd82d0180> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the functio...
.. [#fn-473] TypeError: Error interpreting argument to <function _mean at 0x76cfd82d1ee0> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path a. ...
.. [#fn-474] TypeError: Error interpreting argument to <function _mean at 0x76cfd82d1ee0> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at p...
.. [#fn-475] TypeError: Error interpreting argument to <function _reduce_min at 0x76cfd82d0400> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at pa...
.. [#fn-476] TypeError: Error interpreting argument to <function _reduce_min at 0x76cfd82d0400> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the functio...
.. [#fn-477] TypeError: nancumprod is not supported for quantities with units (has unit m), because each element of the result would have a different unit exponent. Use .nanprod() for a single reduction, or con...
.. [#fn-478] TypeError: Error interpreting argument to <function nanprod at 0x76cfd82d36a0> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path a...
.. [#fn-479] TypeError: Error interpreting argument to <function nanprod at 0x76cfd82d36a0> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at...
.. [#fn-480] TypeError: nonzero requires ndarray or scalar arguments, got <class 'torch.Tensor'> at position 0.
.. [#fn-481] TypeError: nonzero requires ndarray or scalar arguments, got <class 'dask.array.core.Array'> at position 0.
.. [#fn-482] TypeError: nonzero requires ndarray or scalar arguments, got <class 'ndonnx._array.Array'> at position 0.
.. [#fn-483] TypeError: Error interpreting argument to <function outer at 0x76cfd837b7e0> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at p...
.. [#fn-484] TypeError: ptp requires ndarray or scalar arguments, got <class 'torch.Tensor'> at position 0.
.. [#fn-485] TypeError: ptp requires ndarray or scalar arguments, got <class 'dask.array.core.Array'> at position 0.
.. [#fn-486] TypeError: ptp requires ndarray or scalar arguments, got <class 'ndonnx._array.Array'> at position 0.
.. [#fn-487] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'set'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-488] TypeError: Error interpreting argument to <function ravel at 0x76cfd838ee80> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path a. ...
.. [#fn-489] TypeError: Error interpreting argument to <function ravel at 0x76cfd838ee80> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at p...
.. [#fn-490] ValueError: found array with object dtype but it contains non-string elements
.. [#fn-491] TypeError: round() received an invalid combination of arguments - got (Tensor, int), but expected one of: * (Tensor input, *, Tensor out = None) * (Tensor input, *, int decimals, Tensor out = None)
.. [#fn-492] TypeError: round() takes 1 positional argument but 2 were given
.. [#fn-493] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'add'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-494] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'divide'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy()...
.. [#fn-495] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'max'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-496] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'min'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-497] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'multiply'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy...
.. [#fn-498] AxisError: axis1: axis 0 is out of bounds for array of dimension 0
.. [#fn-499] TypeError: std requires ndarray or scalar arguments, got <class 'torch.Tensor'> at position 0.
.. [#fn-500] TypeError: std requires ndarray or scalar arguments, got <class 'dask.array.core.Array'> at position 0.
.. [#fn-501] TypeError: std requires ndarray or scalar arguments, got <class 'ndonnx._array.Array'> at position 0.
.. [#fn-502] TypeError: Error interpreting argument to <function _reduce_sum at 0x76cfd82c7c40> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at pa...
.. [#fn-503] TypeError: Error interpreting argument to <function _reduce_sum at 0x76cfd82c7c40> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the functio...
.. [#fn-504] AttributeError: module 'ndonnx' has no attribute `swapaxes`
.. [#fn-505] TypeError: Error interpreting argument to <function _take at 0x76cfd82c5800> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path a. ...
.. [#fn-506] TypeError: Error interpreting argument to <function _take at 0x76cfd82c5800> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at p...
.. [#fn-507] TypeError: tile(): argument 'dims' (position 2) must be tuple of ints, not int
.. [#fn-508] TypeError: object of type 'int' has no len()
.. [#fn-509] cupy backend not installed
.. [#fn-510] ValueError: unable to infer dtype from `[1. 2. 3.]`
.. [#fn-511] ValueError: unable to infer dtype from `tensor([1., 2., 3.], dtype=torch.float64)`
.. [#fn-512] ValueError: unable to infer dtype from `dask.array<array, shape=(3,), dtype=float64, chunksize=(3,), chunktype=numpy.ndarray>`
.. [#fn-513] TypeError: len() of unsized object
.. [#fn-514] ValueError: ONNX provides no control over the used device
.. [#fn-515] BackendError (expected): Quantity.tolist() would materialize a dask-backed Quantity. Call `q.mantissa.compute()` first.
.. [#fn-516] AttributeError: 'Array' object has no attribute 'tolist'
.. [#fn-517] TypeError: trace() got an unexpected keyword argument 'offset'
.. [#fn-518] AttributeError: module 'ndonnx' has no attribute `trace`
.. [#fn-519] TypeError: transpose() received an invalid combination of arguments - got (Tensor), but expected one of: * (Tensor input, int dim0, int dim1) * (Tensor input, name dim0, name dim1)
.. [#fn-520] AttributeError: module 'ndonnx' has no attribute `transpose`
.. [#fn-521] TypeError: var requires ndarray or scalar arguments, got <class 'torch.Tensor'> at position 0.
.. [#fn-522] TypeError: var requires ndarray or scalar arguments, got <class 'dask.array.core.Array'> at position 0.
.. [#fn-523] TypeError: var requires ndarray or scalar arguments, got <class 'ndonnx._array.Array'> at position 0.
.. [#fn-524] TypeError: view() received an invalid combination of arguments - got (type), but expected one of: * (torch.dtype dtype) didn't match because some of the arguments have invalid types: (!type!) * (tu...
.. [#fn-525] AttributeError: 'Array' object has no attribute 'view'
