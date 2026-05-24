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
     - Full ✓
     - Full ✓
     - ?
     - Partial ⚠
     - Partial ⚠
     - Partial ⚠
   * - **saiunit.linalg**
     - Full ✓
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
     - Partial ⚠
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
     - ⊘ [#fn-3]_
     - ⊘ [#fn-3]_
     - ?
     - ⊘ [#fn-3]_
     - ⊘ [#fn-3]_
     - ⊘ [#fn-3]_

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
     - ✓
     - ✓
     - ✓
   * - ``allclose``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-5]_
   * - ``alltrue``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``amax``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``amin``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``angle``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-6]_
   * - ``any``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``append``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-7]_
     - ✓
     - ⊘ [#fn-8]_
   * - ``arange``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-9]_
     - ⊘ [#fn-10]_
     - ⊘ [#fn-11]_
   * - ``arccos``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-12]_
   * - ``arccosh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-13]_
   * - ``arcsin``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-14]_
   * - ``arcsinh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-15]_
   * - ``arctan``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-16]_
   * - ``arctan2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-17]_
   * - ``arctanh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-18]_
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
     - ✓
     - ✓
     - ✓
   * - ``argsort``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``argwhere``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-19]_
   * - ``around``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-20]_
     - ✓
     - ⊘ [#fn-21]_
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
     - ⊘ [#fn-22]_
     - ⊘ [#fn-23]_
     - ⊘ [#fn-24]_
   * - ``array_split``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-25]_
     - ⊘ [#fn-26]_
     - ⊘ [#fn-27]_
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
     - ⊘ [#fn-28]_
   * - ``atleast_2d``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-29]_
   * - ``atleast_3d``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-30]_
   * - ``average``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-31]_
     - ✓
     - ⊘ [#fn-32]_
   * - ``bincount``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-33]_
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
     - ⊘ [#fn-34]_
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
     - ⊘ [#fn-35]_
     - ✓
     - ⊘ [#fn-36]_
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
     - ⊘ [#fn-37]_
     - ✓
     - ⊘ [#fn-38]_
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
     - 🅙
     - 🅙
     - 🅙
   * - ``choose``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-39]_
     - ⊘ [#fn-40]_
     - ⊘ [#fn-41]_
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
     - ⊘ [#fn-42]_
     - ⊘ [#fn-43]_
   * - ``compress``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-44]_
     - ✓
     - ⊘ [#fn-45]_
   * - ``concatenate``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-46]_
   * - ``conj``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-6]_
   * - ``conjugate``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-47]_
     - ⊘ [#fn-48]_
     - ✗ [#fn-6]_
   * - ``convolve``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-49]_
     - ⊘ [#fn-50]_
     - ⊘ [#fn-51]_
   * - ``copysign``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-52]_
   * - ``corrcoef``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-53]_
     - ✓
     - ⊘ [#fn-54]_
   * - ``correlate``
     - ⊘ [#fn-55]_
     - ✓
     - ?
     - ⊘ [#fn-56]_
     - ⊘ [#fn-57]_
     - ⊘ [#fn-58]_
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
     - ✓
   * - ``cov``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-59]_
     - ✓
     - ⊘ [#fn-60]_
   * - ``cross``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-61]_
     - ⊘ [#fn-62]_
     - ⊘ [#fn-63]_
   * - ``cumprod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-64]_
     - ✓
     - ⊘ [#fn-65]_
   * - ``cumproduct``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-64]_
     - ✓
     - ⊘ [#fn-65]_
   * - ``cumsum``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-66]_
     - ✓
     - ⊘ [#fn-67]_
   * - ``deg2rad``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-68]_
   * - ``degrees``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-69]_
     - ✓
     - ⊘ [#fn-70]_
   * - ``diag``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-71]_
     - ✓
     - ⊘ [#fn-72]_
   * - ``diag_indices_from``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-73]_
     - ⊘ [#fn-74]_
     - ⊘ [#fn-75]_
   * - ``diagflat``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-76]_
     - ⊘ [#fn-77]_
     - ⊘ [#fn-78]_
   * - ``diagonal``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-79]_
     - ✓
     - ⊘ [#fn-80]_
   * - ``diff``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``digitize``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-81]_
     - ✓
     - ⊘ [#fn-82]_
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
     - ⊘ [#fn-83]_
     - ✓
     - ⊘ [#fn-84]_
   * - ``dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-85]_
   * - ``dsplit``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-86]_
     - ⊘ [#fn-87]_
   * - ``dstack``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-88]_
   * - ``ediff1d``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-89]_
     - ✓
     - ⊘ [#fn-90]_
   * - ``einsum``
     - ⊘ [#fn-91]_
     - ✓
     - ?
     - ⊘ [#fn-92]_
     - ⊘ [#fn-91]_
     - ⊘ [#fn-93]_
   * - ``elu``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
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
     - ⊘ [#fn-94]_
     - ✓
     - ⊘ [#fn-94]_
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
     - ⊘ [#fn-95]_
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
     - ✗ [#fn-52]_
   * - ``exprel``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
   * - ``extract``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-96]_
     - ✓
     - ⊘ [#fn-97]_
   * - ``eye``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``fabs``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-98]_
     - ✓
     - ⊘ [#fn-99]_
   * - ``fill_diagonal``
     - ⊘ [#fn-100]_
     - ✓
     - ?
     - ⊘ [#fn-101]_
     - ⊘ [#fn-102]_
     - ⊘ [#fn-103]_
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
     - ⊘ [#fn-104]_
     - ✓
     - ⊘ [#fn-105]_
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
     - ⊘ [#fn-106]_
   * - ``flipud``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-107]_
   * - ``float_power``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-108]_
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
     - ⊘ [#fn-109]_
   * - ``fmin``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-110]_
   * - ``fmod``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-111]_
   * - ``frexp``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-112]_
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
     - ⊘ [#fn-113]_
     - ⊘ [#fn-113]_
     - ?
     - ⊘ [#fn-113]_
     - ⊘ [#fn-113]_
     - ⊘ [#fn-113]_
   * - ``gather``
     - ⊘ [#fn-114]_
     - ⊘ [#fn-114]_
     - ?
     - ⊘ [#fn-114]_
     - ⊘ [#fn-114]_
     - ⊘ [#fn-114]_
   * - ``gcd``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-115]_
     - ⊘ [#fn-116]_
   * - ``gelu``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
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
     - 🅙
     - 🅙
     - 🅙
   * - ``gradient``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-117]_
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
     - 🅙
     - 🅙
     - 🅙
   * - ``hard_silu``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
   * - ``hard_swish``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
   * - ``hard_tanh``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
   * - ``heaviside``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-118]_
     - ⊘ [#fn-119]_
   * - ``histogram``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-120]_
     - ✓
     - ⊘ [#fn-121]_
   * - ``hsplit``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-122]_
     - ⊘ [#fn-123]_
   * - ``hstack``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-124]_
   * - ``hypot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-52]_
   * - ``identity``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-125]_
     - ⊘ [#fn-126]_
     - ⊘ [#fn-127]_
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
     - ✗ [#fn-6]_
   * - ``inner``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-128]_
     - ⊘ [#fn-129]_
   * - ``interp``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-130]_
     - ⊘ [#fn-131]_
     - ⊘ [#fn-132]_
   * - ``intersect1d``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-133]_
     - ⊘ [#fn-134]_
     - ⊘ [#fn-135]_
   * - ``invert``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-136]_
     - ✓
     - ⊘ [#fn-137]_
   * - ``isclose``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-138]_
   * - ``iscomplexobj``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-139]_
     - ⊘ [#fn-140]_
     - ⊘ [#fn-141]_
   * - ``isfinite``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``isinf``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``isnan``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``isreal``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-142]_
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
     - ⊘ [#fn-143]_
     - ⊘ [#fn-144]_
   * - ``lcm``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-145]_
     - ⊘ [#fn-146]_
   * - ``ldexp``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-147]_
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
     - ⊘ [#fn-148]_
     - ✓
     - ⊘ [#fn-149]_
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
     - ⊘ [#fn-150]_
     - ✓
     - ⊘ [#fn-151]_
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
     - ✗ [#fn-52]_
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
     - 🅙
     - 🅙
     - 🅙
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
     - ⊘ [#fn-152]_
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
     - ⊘ [#fn-153]_
     - ⊘ [#fn-154]_
     - ⊘ [#fn-155]_
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
     - ⊘ [#fn-156]_
     - ⊘ [#fn-157]_
   * - ``max``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``maximum``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``mean``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``median``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-158]_
     - ⊘ [#fn-159]_
     - ⊘ [#fn-160]_
   * - ``meshgrid``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-161]_
   * - ``min``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
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
     - 🅙
     - 🅙
     - 🅙
   * - ``mod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-162]_
     - ✓
     - ⊘ [#fn-163]_
   * - ``modf``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-164]_
     - ✓
     - ⊘ [#fn-165]_
   * - ``moveaxis``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``multi_dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-166]_
     - ⊘ [#fn-167]_
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
     - ✓
     - ⊘ [#fn-168]_
   * - ``nanargmax``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-169]_
     - ✓
     - ⊘ [#fn-170]_
   * - ``nanargmin``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-171]_
     - ✓
     - ⊘ [#fn-172]_
   * - ``nancumprod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-173]_
     - ⊘ [#fn-174]_
     - ⊘ [#fn-175]_
   * - ``nancumsum``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-176]_
     - ⊘ [#fn-177]_
     - ⊘ [#fn-178]_
   * - ``nanmax``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-179]_
     - ✓
     - ⊘ [#fn-180]_
   * - ``nanmean``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-181]_
   * - ``nanmedian``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-182]_
     - ⊘ [#fn-183]_
     - ⊘ [#fn-184]_
   * - ``nanmin``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-185]_
     - ✓
     - ⊘ [#fn-186]_
   * - ``nanpercentile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-187]_
     - ✓
     - ⊘ [#fn-188]_
   * - ``nanprod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-189]_
     - ✓
     - ⊘ [#fn-190]_
   * - ``nanquantile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-191]_
     - ✓
     - ⊘ [#fn-192]_
   * - ``nanstd``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-193]_
     - ✓
     - ⊘ [#fn-194]_
   * - ``nansum``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-195]_
   * - ``nanvar``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-196]_
     - ✓
     - ⊘ [#fn-197]_
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
     - ✗ [#fn-52]_
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
     - ⊘ [#fn-198]_
     - ✓
     - ⊘ [#fn-198]_
   * - ``outer``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-199]_
   * - ``percentile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-200]_
     - ⊘ [#fn-201]_
     - ⊘ [#fn-202]_
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
     - ⊘ [#fn-203]_
     - ✓
     - ⊘ [#fn-204]_
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
     - ⊘ [#fn-205]_
     - ⊘ [#fn-206]_
     - ⊘ [#fn-207]_
   * - ``quantile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-208]_
     - ✓
     - ⊘ [#fn-209]_
   * - ``rad2deg``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-210]_
   * - ``radians``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-211]_
     - ✓
     - ⊘ [#fn-212]_
   * - ``ravel``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-213]_
     - ⊘ [#fn-213]_
     - ⊘ [#fn-214]_
   * - ``real``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-6]_
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
     - 🅙
     - 🅙
     - 🅙
   * - ``relu6``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
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
     - ✓
     - ⊘ [#fn-215]_
   * - ``repeat``
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
     - ⊘ [#fn-216]_
     - ⊘ [#fn-216]_
     - ⊘ [#fn-216]_
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
     - ⊘ [#fn-217]_
     - ✓
     - ⊘ [#fn-218]_
   * - ``rint``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-219]_
     - ✓
     - ⊘ [#fn-220]_
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
     - ⊘ [#fn-221]_
     - ✓
     - ⊘ [#fn-222]_
   * - ``round``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-223]_
   * - ``row_stack``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-224]_
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
     - ⊘ [#fn-225]_
     - ✓
     - ⊘ [#fn-226]_
   * - ``selu``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
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
     - 🅙
     - 🅙
     - 🅙
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
     - ✗ [#fn-52]_
   * - ``silu``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
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
     - ⊘ [#fn-227]_
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
     - 🅙
     - 🅙
     - 🅙
   * - ``softplus``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
   * - ``sometrue``
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
     - ✓
     - ✓
     - ✓
   * - ``sparse_plus``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
   * - ``sparse_sigmoid``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
   * - ``split``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-25]_
     - ⊘ [#fn-26]_
     - ⊘ [#fn-27]_
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
     - 🅙
     - 🅙
     - 🅙
   * - ``squeeze``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-228]_
     - ✓
     - ⊘ [#fn-228]_
   * - ``stack``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``std``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
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
     - ⊘ [#fn-229]_
     - ✓
     - ⊘ [#fn-230]_
   * - ``swish``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
   * - ``take``
     - ⊘ [#fn-231]_
     - ✓
     - ?
     - ⊘ [#fn-232]_
     - ⊘ [#fn-233]_
     - ⊘ [#fn-233]_
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
     - ⊘ [#fn-234]_
     - ✓
     - ⊘ [#fn-235]_
   * - ``trace``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-236]_
     - ✓
     - ⊘ [#fn-237]_
   * - ``transpose``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-238]_
     - ✓
     - ⊘ [#fn-239]_
   * - ``tree_ones_like``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-198]_
     - ✓
     - ⊘ [#fn-198]_
   * - ``tree_zeros_like``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-240]_
     - ✓
     - ⊘ [#fn-240]_
   * - ``tri``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-241]_
     - ✓
     - ⊘ [#fn-242]_
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
     - ⊘ [#fn-243]_
     - ✓
     - ⊘ [#fn-244]_
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
     - ⊘ [#fn-245]_
     - ✓
     - ⊘ [#fn-246]_
   * - ``true_divide``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-247]_
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
     - ⊘ [#fn-248]_
     - ⊘ [#fn-249]_
     - ⊘ [#fn-250]_
   * - ``vander``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-251]_
     - ⊘ [#fn-252]_
   * - ``var``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``vdot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-253]_
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
     - ⊘ [#fn-254]_
     - ⊘ [#fn-255]_
   * - ``vstack``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-224]_
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
     - ⊘ [#fn-240]_
     - ✓
     - ⊘ [#fn-240]_

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
     - ⊘ [#fn-256]_
     - ✓
     - ⊘ [#fn-239]_
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
     - ⊘ [#fn-234]_
     - ✓
     - ⊘ [#fn-235]_
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
     - ⊘ [#fn-257]_
     - ✓
     - ?
     - ⊘ [#fn-258]_
     - ⊘ [#fn-257]_
     - ⊘ [#fn-259]_
   * - ``cond``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-260]_
     - ⊘ [#fn-261]_
   * - ``cross``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-61]_
     - ⊘ [#fn-62]_
     - ⊘ [#fn-63]_
   * - ``det``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-262]_
     - ⊘ [#fn-263]_
   * - ``diagonal``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-79]_
     - ✓
     - ⊘ [#fn-80]_
   * - ``dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-85]_
   * - ``eig``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-264]_
     - ⊘ [#fn-265]_
     - ⊘ [#fn-266]_
   * - ``eigh``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-267]_
     - ⊘ [#fn-268]_
     - ⊘ [#fn-269]_
   * - ``eigvals``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-264]_
     - ⊘ [#fn-265]_
     - ⊘ [#fn-266]_
   * - ``eigvalsh``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-267]_
     - ⊘ [#fn-268]_
     - ⊘ [#fn-269]_
   * - ``inner``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-128]_
     - ⊘ [#fn-129]_
   * - ``inv``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-270]_
   * - ``kron``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-143]_
     - ⊘ [#fn-144]_
   * - ``lstsq``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-271]_
     - ⊘ [#fn-272]_
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
     - ⊘ [#fn-273]_
   * - ``matrix_power``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-156]_
     - ⊘ [#fn-157]_
   * - ``matrix_rank``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-274]_
     - ⊘ [#fn-275]_
   * - ``matrix_transpose``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-276]_
   * - ``multi_dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-166]_
     - ⊘ [#fn-167]_
   * - ``norm``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-277]_
   * - ``outer``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-199]_
   * - ``pinv``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-278]_
     - ⊘ [#fn-279]_
   * - ``qr``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-272]_
   * - ``slogdet``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-280]_
     - ⊘ [#fn-272]_
   * - ``solve``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-281]_
   * - ``svd``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-282]_
     - ⊘ [#fn-283]_
     - ⊘ [#fn-284]_
   * - ``svdvals``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-282]_
     - ⊘ [#fn-283]_
     - ⊘ [#fn-284]_
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
     - ⊘ [#fn-285]_
     - ⊘ [#fn-286]_
   * - ``tensorsolve``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-287]_
     - ⊘ [#fn-288]_
   * - ``trace``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-236]_
     - ✓
     - ⊘ [#fn-237]_
   * - ``vdot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-253]_
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
     - ⊘ [#fn-289]_

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
     - ⊘ [#fn-290]_
   * - ``fft2``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-291]_
     - ✓
     - ⊘ [#fn-292]_
   * - ``fftfreq``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-293]_
   * - ``fftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-294]_
   * - ``fftshift``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-295]_
   * - ``ifft``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-296]_
   * - ``ifft2``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-297]_
     - ✓
     - ⊘ [#fn-298]_
   * - ``ifftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-299]_
   * - ``ifftshift``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-300]_
   * - ``irfft``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-301]_
   * - ``irfft2``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-302]_
     - ✓
     - ⊘ [#fn-303]_
   * - ``irfftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-304]_
   * - ``rfft``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-305]_
   * - ``rfft2``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-306]_
     - ✓
     - ⊘ [#fn-307]_
   * - ``rfftfreq``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-293]_
   * - ``rfftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-308]_

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
     - ✓
     - ✓
     - ✓
   * - ``any``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
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
     - ✓
     - ✓
     - ✓
   * - ``argsort``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``astype``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-309]_
     - ✓
     - ⊘ [#fn-310]_
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
     - ⊘ [#fn-311]_
     - ⊘ [#fn-312]_
     - ⊘ [#fn-313]_
   * - ``conj``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-52]_
   * - ``conjugate``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-52]_
   * - ``copy``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-311]_
     - ⊘ [#fn-312]_
     - ⊘ [#fn-313]_
   * - ``cpu``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-314]_
     - ✗ [#fn-315]_
     - ✗ [#fn-316]_
   * - ``cross``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-61]_
     - ⊘ [#fn-317]_
     - ⊘ [#fn-318]_
   * - ``cuda``
     - ⊘ [#fn-319]_
     - ⊘ [#fn-319]_
     - ?
     - ⊘ [#fn-319]_
     - ⊘ [#fn-319]_
     - ⊘ [#fn-319]_
   * - ``cumprod``
     - ⊘ [#fn-320]_
     - ⊘ [#fn-320]_
     - ?
     - ⊘ [#fn-320]_
     - ⊘ [#fn-320]_
     - ⊘ [#fn-320]_
   * - ``cumsum``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-66]_
     - ✓
     - ⊘ [#fn-67]_
   * - ``diagonal``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-321]_
     - ✓
     - ⊘ [#fn-322]_
   * - ``dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-323]_
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
     - ✗ [#fn-324]_
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
     - ✗ [#fn-314]_
     - ✗ [#fn-325]_
     - ✗ [#fn-326]_
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
     - ⊘ [#fn-327]_
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
     - ⊘ [#fn-328]_
     - ⊘ [#fn-328]_
   * - ``max``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``mean``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``min``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``nancumprod``
     - ⊘ [#fn-329]_
     - ⊘ [#fn-329]_
     - ?
     - ⊘ [#fn-329]_
     - ⊘ [#fn-329]_
     - ⊘ [#fn-329]_
   * - ``nanprod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-189]_
     - ✓
     - ⊘ [#fn-190]_
   * - ``nonzero``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``outer``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-330]_
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
     - ⊘ [#fn-205]_
     - ✓
     - ⊘ [#fn-207]_
   * - ``put``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-314]_
     - ⊘ [#fn-331]_
     - ✗ [#fn-326]_
   * - ``ravel``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-214]_
   * - ``repeat``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-332]_
     - ✓
     - ✗ [#fn-332]_
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
     - ✗ [#fn-314]_
     - ✗ [#fn-325]_
     - ✗ [#fn-333]_
   * - ``round``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-334]_
     - ✓
     - ✗ [#fn-335]_
   * - ``scatter_add``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-336]_
     - ✗ [#fn-326]_
   * - ``scatter_div``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-337]_
     - ✗ [#fn-326]_
   * - ``scatter_max``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-338]_
     - ✗ [#fn-326]_
   * - ``scatter_min``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-339]_
     - ✗ [#fn-326]_
   * - ``scatter_mul``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-340]_
     - ✗ [#fn-326]_
   * - ``scatter_sub``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-336]_
     - ✗ [#fn-326]_
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
     - ✗ [#fn-314]_
     - ✗ [#fn-325]_
     - ✗ [#fn-316]_
   * - ``split``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-25]_
     - ✓
     - ✗ [#fn-341]_
   * - ``squeeze``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-342]_
     - ✓
     - ✗ [#fn-343]_
   * - ``std``
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
     - ✓
     - ✓
     - ⊘ [#fn-344]_
   * - ``take``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-345]_
     - ✓
   * - ``tile``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-346]_
     - ✓
     - ✗ [#fn-347]_
   * - ``to``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``to_cupy``
     - ⊘ [#fn-348]_
     - ⊘ [#fn-348]_
     - ?
     - ⊘ [#fn-348]_
     - ⊘ [#fn-348]_
     - ⊘ [#fn-348]_
   * - ``to_dask``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-314]_
     - ✓
     - ✗ [#fn-316]_
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
     - ✗ [#fn-349]_
   * - ``to_ndonnx``
     - ✓
     - ⊘ [#fn-350]_
     - ?
     - ⊘ [#fn-351]_
     - ⊘ [#fn-352]_
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
     - ✗ [#fn-353]_
     - ✗ [#fn-354]_
   * - ``tolist``
     - ✓
     - ✓
     - ?
     - ✓
     - ⚠ [#fn-355]_
     - ⊘ [#fn-356]_
   * - ``trace``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-357]_
     - ✓
     - ⊘ [#fn-358]_
   * - ``transpose``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-238]_
     - ✓
     - ⊘ [#fn-359]_
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
     - ✗ [#fn-324]_
   * - ``update_mantissa``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-314]_
     - ✗ [#fn-325]_
     - ✗ [#fn-316]_
   * - ``var``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``view``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-360]_
     - ✓
     - ⊘ [#fn-361]_
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
.. [#fn-5] AttributeError: module 'ndonnx' has no attribute `allclose`
.. [#fn-6] ValueError: 'complex128' does not have a corresponding ndonnx data type
.. [#fn-7] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'append'
.. [#fn-8] AttributeError: saiunit: backend 'ndonnx' has no operation 'append'
.. [#fn-9] TypeError: '>' not supported between instances of 'NoneType' and 'int'
.. [#fn-10] TypeError: An error occurred while calling the arange method registered to the numpy backend. Original Message: unsupported operand type(s) for /: 'int' and 'NoneType'
.. [#fn-11] ValueError: 'arange' is not implemented for the provided inputs
.. [#fn-12] AttributeError: saiunit: backend 'ndonnx' has no operation 'arccos'
.. [#fn-13] AttributeError: saiunit: backend 'ndonnx' has no operation 'arccosh'
.. [#fn-14] AttributeError: saiunit: backend 'ndonnx' has no operation 'arcsin'
.. [#fn-15] AttributeError: saiunit: backend 'ndonnx' has no operation 'arcsinh'
.. [#fn-16] AttributeError: saiunit: backend 'ndonnx' has no operation 'arctan'
.. [#fn-17] AttributeError: saiunit: backend 'ndonnx' has no operation 'arctan2'
.. [#fn-18] AttributeError: saiunit: backend 'ndonnx' has no operation 'arctanh'
.. [#fn-19] AttributeError: saiunit: backend 'ndonnx' has no operation 'argwhere'
.. [#fn-20] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'around'
.. [#fn-21] AttributeError: saiunit: backend 'ndonnx' has no operation 'around'
.. [#fn-22] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'array_equal'
.. [#fn-23] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'array_equal'
.. [#fn-24] AttributeError: saiunit: backend 'ndonnx' has no operation 'array_equal'
.. [#fn-25] TypeError: split() got an unexpected keyword argument 'axis'
.. [#fn-26] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'split'
.. [#fn-27] AttributeError: saiunit: backend 'ndonnx' has no operation 'split'
.. [#fn-28] AttributeError: saiunit: backend 'ndonnx' has no operation 'atleast_1d'
.. [#fn-29] AttributeError: saiunit: backend 'ndonnx' has no operation 'atleast_2d'
.. [#fn-30] AttributeError: saiunit: backend 'ndonnx' has no operation 'atleast_3d'
.. [#fn-31] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'average'
.. [#fn-32] AttributeError: saiunit: backend 'ndonnx' has no operation 'average'
.. [#fn-33] AttributeError: saiunit: backend 'ndonnx' has no operation 'bincount'
.. [#fn-34] AttributeError: saiunit: backend 'ndonnx' has no operation 'bitwise_not'
.. [#fn-35] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'block'
.. [#fn-36] AttributeError: saiunit: backend 'ndonnx' has no operation 'block'
.. [#fn-37] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'cbrt'
.. [#fn-38] AttributeError: saiunit: backend 'ndonnx' has no operation 'cbrt'
.. [#fn-39] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'choose'
.. [#fn-40] TypeError: choose() got an unexpected keyword argument 'mode'
.. [#fn-41] AttributeError: saiunit: backend 'ndonnx' has no operation 'choose'
.. [#fn-42] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'column_stack'
.. [#fn-43] AttributeError: saiunit: backend 'ndonnx' has no operation 'column_stack'
.. [#fn-44] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'compress'
.. [#fn-45] AttributeError: saiunit: backend 'ndonnx' has no operation 'compress'
.. [#fn-46] AttributeError: saiunit: backend 'ndonnx' has no operation 'concatenate'
.. [#fn-47] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'conjugate'
.. [#fn-48] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'conjugate'
.. [#fn-49] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'convolve'
.. [#fn-50] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'convolve'
.. [#fn-51] AttributeError: saiunit: backend 'ndonnx' has no operation 'convolve'
.. [#fn-52] NotImplementedError:
.. [#fn-53] TypeError: corrcoef() got an unexpected keyword argument 'rowvar'
.. [#fn-54] AttributeError: saiunit: backend 'ndonnx' has no operation 'corrcoef'
.. [#fn-55] TypeError: correlate() got an unexpected keyword argument 'precision'
.. [#fn-56] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'correlate'
.. [#fn-57] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'correlate'
.. [#fn-58] AttributeError: saiunit: backend 'ndonnx' has no operation 'correlate'
.. [#fn-59] TypeError: cov() got an unexpected keyword argument 'bias'
.. [#fn-60] AttributeError: saiunit: backend 'ndonnx' has no operation 'cov'
.. [#fn-61] TypeError: cross() got an unexpected keyword argument 'axisa'
.. [#fn-62] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'cross'
.. [#fn-63] AttributeError: saiunit: backend 'ndonnx' has no operation 'cross'
.. [#fn-64] TypeError: cumprod() received an invalid combination of arguments - got (Tensor), but expected one of: \* (Tensor input, int dim, \*, torch.dtype dtype = None, Tensor out = None) \* (Tensor input, nam...
.. [#fn-65] AttributeError: saiunit: backend 'ndonnx' has no operation 'cumprod'
.. [#fn-66] TypeError: cumsum() received an invalid combination of arguments - got (Tensor), but expected one of: \* (Tensor input, int dim, \*, torch.dtype dtype = None, Tensor out = None) \* (Tensor input, name...
.. [#fn-67] AttributeError: saiunit: backend 'ndonnx' has no operation 'cumsum'
.. [#fn-68] AttributeError: saiunit: backend 'ndonnx' has no operation 'deg2rad'
.. [#fn-69] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'degrees'
.. [#fn-70] AttributeError: saiunit: backend 'ndonnx' has no operation 'degrees'
.. [#fn-71] TypeError: diag() got an unexpected keyword argument 'k'
.. [#fn-72] AttributeError: module 'ndonnx' has no attribute `diag`
.. [#fn-73] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'diag_indices_from'
.. [#fn-74] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'diag_indices_from'
.. [#fn-75] AttributeError: saiunit: backend 'ndonnx' has no operation 'diag_indices_from'
.. [#fn-76] TypeError: diagflat() got an unexpected keyword argument 'k'
.. [#fn-77] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'diagflat'
.. [#fn-78] AttributeError: saiunit: backend 'ndonnx' has no operation 'diagflat'
.. [#fn-79] TypeError: diagonal() received an invalid combination of arguments - got (Tensor, offset=int, axis2=int, axis1=int), but expected one of: \* (Tensor input, \*, name outdim, name dim1, name dim2, int ...
.. [#fn-80] AttributeError: saiunit: backend 'ndonnx' has no operation 'diagonal'
.. [#fn-81] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'digitize'
.. [#fn-82] AttributeError: saiunit: backend 'ndonnx' has no operation 'digitize'
.. [#fn-83] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'divmod'
.. [#fn-84] AttributeError: saiunit: backend 'ndonnx' has no operation 'divmod'
.. [#fn-85] AttributeError: saiunit: backend 'ndonnx' has no operation 'dot'
.. [#fn-86] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'dsplit'
.. [#fn-87] AttributeError: saiunit: backend 'ndonnx' has no operation 'dsplit'
.. [#fn-88] AttributeError: saiunit: backend 'ndonnx' has no operation 'dstack'
.. [#fn-89] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'ediff1d'
.. [#fn-90] AttributeError: saiunit: backend 'ndonnx' has no operation 'ediff1d'
.. [#fn-91] TracerArrayConversionError: The numpy.ndarray conversion method __array__() was called on traced array with shape float32[2,2] The error occurred while tracing the function _einsum at /mnt/d/codes/...
.. [#fn-92] TypeError: Error interpreting argument to <function _einsum at 0x...> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path operands[0...
.. [#fn-93] TypeError: Error interpreting argument to <function _einsum at 0x...> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at path ope...
.. [#fn-94] TypeError: empty_like() got an unexpected keyword argument 'shape'
.. [#fn-95] AttributeError: saiunit: backend 'ndonnx' has no operation 'exp2'
.. [#fn-96] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'extract'
.. [#fn-97] AttributeError: saiunit: backend 'ndonnx' has no operation 'extract'
.. [#fn-98] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'fabs'
.. [#fn-99] AttributeError: saiunit: backend 'ndonnx' has no operation 'fabs'
.. [#fn-100] TypeError: fill_diagonal() got an unexpected keyword argument 'inplace'
.. [#fn-101] AttributeError: module 'array_api_compat.torch' has no attribute 'fill_diagonal'
.. [#fn-102] AttributeError: module 'array_api_compat.dask.array' has no attribute 'fill_diagonal'
.. [#fn-103] AttributeError: module 'ndonnx' has no attribute `fill_diagonal`
.. [#fn-104] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'flatnonzero'
.. [#fn-105] AttributeError: saiunit: backend 'ndonnx' has no operation 'flatnonzero'
.. [#fn-106] AttributeError: saiunit: backend 'ndonnx' has no operation 'fliplr'
.. [#fn-107] AttributeError: saiunit: backend 'ndonnx' has no operation 'flipud'
.. [#fn-108] AttributeError: saiunit: backend 'ndonnx' has no operation 'float_power'
.. [#fn-109] AttributeError: saiunit: backend 'ndonnx' has no operation 'fmax'
.. [#fn-110] AttributeError: saiunit: backend 'ndonnx' has no operation 'fmin'
.. [#fn-111] AttributeError: saiunit: backend 'ndonnx' has no operation 'fmod'
.. [#fn-112] AttributeError: saiunit: backend 'ndonnx' has no operation 'frexp'
.. [#fn-113] TypeError: full_like() missing 1 required positional argument: 'fill_value'
.. [#fn-114] TypeError: gather() missing 1 required positional argument: 'index'
.. [#fn-115] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'gcd'
.. [#fn-116] AttributeError: saiunit: backend 'ndonnx' has no operation 'gcd'
.. [#fn-117] AttributeError: module 'ndonnx' has no attribute `gradient`
.. [#fn-118] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'heaviside'
.. [#fn-119] AttributeError: saiunit: backend 'ndonnx' has no operation 'heaviside'
.. [#fn-120] TypeError: histogram() received an invalid combination of arguments - got (Tensor, int, density=NoneType, weights=NoneType, range=NoneType), but expected one of: \* (Tensor input, Tensor bins, \*, Te...
.. [#fn-121] AttributeError: saiunit: backend 'ndonnx' has no operation 'histogram'
.. [#fn-122] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'hsplit'
.. [#fn-123] AttributeError: saiunit: backend 'ndonnx' has no operation 'hsplit'
.. [#fn-124] AttributeError: saiunit: backend 'ndonnx' has no operation 'hstack'
.. [#fn-125] AttributeError: module 'array_api_compat.torch' has no attribute 'identity'
.. [#fn-126] AttributeError: module 'array_api_compat.dask.array' has no attribute 'identity'
.. [#fn-127] AttributeError: module 'ndonnx' has no attribute `identity`
.. [#fn-128] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'inner'
.. [#fn-129] AttributeError: saiunit: backend 'ndonnx' has no operation 'inner'
.. [#fn-130] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'interp'
.. [#fn-131] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'interp'
.. [#fn-132] AttributeError: saiunit: backend 'ndonnx' has no operation 'interp'
.. [#fn-133] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'intersect1d'
.. [#fn-134] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'intersect1d'
.. [#fn-135] AttributeError: saiunit: backend 'ndonnx' has no operation 'intersect1d'
.. [#fn-136] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'invert'
.. [#fn-137] AttributeError: saiunit: backend 'ndonnx' has no operation 'invert'
.. [#fn-138] AttributeError: saiunit: backend 'ndonnx' has no operation 'isclose'
.. [#fn-139] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'iscomplexobj'
.. [#fn-140] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'iscomplexobj'
.. [#fn-141] AttributeError: saiunit: backend 'ndonnx' has no operation 'iscomplexobj'
.. [#fn-142] AttributeError: module 'ndonnx' has no attribute `isreal`
.. [#fn-143] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'kron'
.. [#fn-144] AttributeError: saiunit: backend 'ndonnx' has no operation 'kron'
.. [#fn-145] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'lcm'
.. [#fn-146] AttributeError: saiunit: backend 'ndonnx' has no operation 'lcm'
.. [#fn-147] AttributeError: saiunit: backend 'ndonnx' has no operation 'ldexp'
.. [#fn-148] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'left_shift'
.. [#fn-149] AttributeError: saiunit: backend 'ndonnx' has no operation 'left_shift'
.. [#fn-150] TypeError: linspace() received an invalid combination of arguments - got (float, float, int, retstep=bool, device=NoneType, dtype=NoneType), but expected one of: \* (Tensor start, Tensor end, int st...
.. [#fn-151] TypeError: linspace() got an unexpected keyword argument 'retstep'
.. [#fn-152] AttributeError: saiunit: backend 'ndonnx' has no operation 'logaddexp2'
.. [#fn-153] TypeError: logspace() received an invalid combination of arguments - got (float, float, dtype=NoneType, base=float, endpoint=bool, num=int), but expected one of: \* (Tensor start, Tensor end, int st...
.. [#fn-154] AttributeError: module 'array_api_compat.dask.array' has no attribute 'logspace'
.. [#fn-155] AttributeError: module 'ndonnx' has no attribute `logspace`
.. [#fn-156] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.matrix_power'
.. [#fn-157] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_power'
.. [#fn-158] TypeError: median() received an invalid combination of arguments - got (Tensor, overwrite_input=bool, keepdims=bool), but expected one of: \* (Tensor input) \* (Tensor input, int dim, bool keepdim = ...
.. [#fn-159] TypeError: median() got an unexpected keyword argument 'overwrite_input'
.. [#fn-160] AttributeError: saiunit: backend 'ndonnx' has no operation 'median'
.. [#fn-161] AttributeError: 'list' object has no attribute 'ndim'
.. [#fn-162] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'mod'
.. [#fn-163] AttributeError: saiunit: backend 'ndonnx' has no operation 'mod'
.. [#fn-164] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'modf'
.. [#fn-165] AttributeError: saiunit: backend 'ndonnx' has no operation 'modf'
.. [#fn-166] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.multi_dot'
.. [#fn-167] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.multi_dot'
.. [#fn-168] AttributeError: saiunit: backend 'ndonnx' has no operation 'nan_to_num'
.. [#fn-169] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanargmax'
.. [#fn-170] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanargmax'
.. [#fn-171] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanargmin'
.. [#fn-172] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanargmin'
.. [#fn-173] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nancumprod'
.. [#fn-174] TypeError: nancumprod() missing 1 required positional argument: 'axis'
.. [#fn-175] AttributeError: saiunit: backend 'ndonnx' has no operation 'nancumprod'
.. [#fn-176] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nancumsum'
.. [#fn-177] TypeError: nancumsum() missing 1 required positional argument: 'axis'
.. [#fn-178] AttributeError: saiunit: backend 'ndonnx' has no operation 'nancumsum'
.. [#fn-179] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanmax'
.. [#fn-180] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmax'
.. [#fn-181] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmean'
.. [#fn-182] TypeError: nanmedian() received an invalid combination of arguments - got (Tensor, overwrite_input=bool, keepdims=bool), but expected one of: \* (Tensor input) \* (Tensor input, int dim, bool keepdim...
.. [#fn-183] TypeError: nanmedian() got an unexpected keyword argument 'overwrite_input'
.. [#fn-184] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmedian'
.. [#fn-185] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanmin'
.. [#fn-186] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmin'
.. [#fn-187] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanpercentile'
.. [#fn-188] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanpercentile'
.. [#fn-189] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanprod'
.. [#fn-190] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanprod'
.. [#fn-191] TypeError: nanquantile() received an invalid combination of arguments - got (Tensor, q=float, method=str, keepdims=bool), but expected one of: \* (Tensor input, Tensor q, int dim = None, bool keepdi...
.. [#fn-192] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanquantile'
.. [#fn-193] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanstd'
.. [#fn-194] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanstd'
.. [#fn-195] AttributeError: saiunit: backend 'ndonnx' has no operation 'nansum'
.. [#fn-196] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanvar'
.. [#fn-197] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanvar'
.. [#fn-198] TypeError: ones_like() got an unexpected keyword argument 'shape'
.. [#fn-199] AttributeError: saiunit: backend 'ndonnx' has no operation 'outer'
.. [#fn-200] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'percentile'
.. [#fn-201] TypeError: percentile() got an unexpected keyword argument dict_keys(['keepdims'])
.. [#fn-202] AttributeError: saiunit: backend 'ndonnx' has no operation 'percentile'
.. [#fn-203] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'power'
.. [#fn-204] AttributeError: saiunit: backend 'ndonnx' has no operation 'power'
.. [#fn-205] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'ptp'
.. [#fn-206] TypeError: ptp() got an unexpected keyword argument 'keepdims'
.. [#fn-207] AttributeError: saiunit: backend 'ndonnx' has no operation 'ptp'
.. [#fn-208] TypeError: quantile() received an invalid combination of arguments - got (Tensor, q=float, method=str, keepdims=bool), but expected one of: \* (Tensor input, Tensor q, int dim = None, bool keepdim =...
.. [#fn-209] AttributeError: saiunit: backend 'ndonnx' has no operation 'quantile'
.. [#fn-210] AttributeError: saiunit: backend 'ndonnx' has no operation 'rad2deg'
.. [#fn-211] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'radians'
.. [#fn-212] AttributeError: saiunit: backend 'ndonnx' has no operation 'radians'
.. [#fn-213] TypeError: ravel() got an unexpected keyword argument 'order'
.. [#fn-214] AttributeError: saiunit: backend 'ndonnx' has no operation 'ravel'
.. [#fn-215] AttributeError: type object 'bool' has no attribute '__ndx_create__'
.. [#fn-216] TypeError: reshape() got an unexpected keyword argument 'order'
.. [#fn-217] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'right_shift'
.. [#fn-218] AttributeError: saiunit: backend 'ndonnx' has no operation 'right_shift'
.. [#fn-219] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'rint'
.. [#fn-220] AttributeError: saiunit: backend 'ndonnx' has no operation 'rint'
.. [#fn-221] TypeError: rot90() got an unexpected keyword argument 'axes'
.. [#fn-222] AttributeError: saiunit: backend 'ndonnx' has no operation 'rot90'
.. [#fn-223] TypeError: round() got an unexpected keyword argument 'decimals'
.. [#fn-224] AttributeError: saiunit: backend 'ndonnx' has no operation 'vstack'
.. [#fn-225] TypeError: select() received an invalid combination of arguments - got (list, list, default=int), but expected one of: \* (Tensor input, name dim, int index) didn't match because some of the keyword...
.. [#fn-226] AttributeError: saiunit: backend 'ndonnx' has no operation 'select'
.. [#fn-227] AttributeError: saiunit: backend 'ndonnx' has no operation 'sinc'
.. [#fn-228] TypeError: squeeze() missing 1 required positional argument: 'axis'
.. [#fn-229] TypeError: swapaxes() missing 2 required positional argument: "axis0", "axis1"
.. [#fn-230] AttributeError: saiunit: backend 'ndonnx' has no operation 'swapaxes'
.. [#fn-231] TypeError: take() got an unexpected keyword argument 'unique_indices'
.. [#fn-232] TypeError: index_select() received an invalid combination of arguments - got (Tensor, int, Tensor, fill_value=NoneType, indices_are_sorted=bool, unique_indices=bool, mode=NoneType), but expected on...
.. [#fn-233] TypeError: take() got an unexpected keyword argument 'mode'
.. [#fn-234] TypeError: tile() missing 1 required positional arguments: "dims"
.. [#fn-235] TypeError: tile() got an unexpected keyword argument 'reps'
.. [#fn-236] TypeError: trace() got an unexpected keyword argument 'axis1'
.. [#fn-237] AttributeError: saiunit: backend 'ndonnx' has no operation 'trace'
.. [#fn-238] TypeError: transpose() received an invalid combination of arguments - got (Tensor), but expected one of: \* (Tensor input, int dim0, int dim1) \* (Tensor input, name dim0, name dim1)
.. [#fn-239] AttributeError: saiunit: backend 'ndonnx' has no operation 'transpose'
.. [#fn-240] TypeError: zeros_like() got an unexpected keyword argument 'shape'
.. [#fn-241] AttributeError: module 'array_api_compat.torch' has no attribute 'tri'
.. [#fn-242] AttributeError: module 'ndonnx' has no attribute `tri`
.. [#fn-243] AttributeError: module 'array_api_compat.torch' has no attribute 'tril_indices_from'
.. [#fn-244] AttributeError: module 'ndonnx' has no attribute `tril_indices_from`
.. [#fn-245] AttributeError: module 'array_api_compat.torch' has no attribute 'triu_indices_from'
.. [#fn-246] AttributeError: module 'ndonnx' has no attribute `triu_indices_from`
.. [#fn-247] AttributeError: saiunit: backend 'ndonnx' has no operation 'true_divide'
.. [#fn-248] TypeError: _return_output() got an unexpected keyword argument 'return_index'. Did you mean 'return_inverse'?
.. [#fn-249] TypeError: unique() got an unexpected keyword argument 'axis'
.. [#fn-250] AttributeError: saiunit: backend 'ndonnx' has no operation 'unique'
.. [#fn-251] AttributeError: module 'array_api_compat.dask.array' has no attribute 'vander'
.. [#fn-252] AttributeError: module 'ndonnx' has no attribute `vander`
.. [#fn-253] AttributeError: saiunit: backend 'ndonnx' has no operation 'vdot'
.. [#fn-254] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'vsplit'
.. [#fn-255] AttributeError: saiunit: backend 'ndonnx' has no operation 'vsplit'
.. [#fn-256] TypeError: transpose() received an invalid combination of arguments - got (Tensor, axes=list), but expected one of: \* (Tensor input, int dim0, int dim1) \* (Tensor input, name dim0, name dim1)
.. [#fn-257] TypeError: cholesky() got an unexpected keyword argument 'symmetrize_input'
.. [#fn-258] TypeError: linalg_cholesky() got an unexpected keyword argument 'symmetrize_input'
.. [#fn-259] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.cholesky'
.. [#fn-260] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.cond'
.. [#fn-261] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.cond'
.. [#fn-262] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.det'
.. [#fn-263] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.det'
.. [#fn-264] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to eig at position 0.
.. [#fn-265] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to eig at position 0.
.. [#fn-266] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to eig at position 0.
.. [#fn-267] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to transpose at position 0.
.. [#fn-268] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to transpose at position 0.
.. [#fn-269] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to transpose at position 0.
.. [#fn-270] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.inv'
.. [#fn-271] TypeError: lstsq() got an unexpected keyword argument 'rcond'
.. [#fn-272] AttributeError: module 'ndonnx' has no attribute `linalg`
.. [#fn-273] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_norm'
.. [#fn-274] TypeError: svd() got an unexpected keyword argument 'compute_uv'
.. [#fn-275] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_rank'
.. [#fn-276] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_transpose'
.. [#fn-277] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.norm'
.. [#fn-278] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.pinv'
.. [#fn-279] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.pinv'
.. [#fn-280] AttributeError: module 'array_api_compat.dask.array.linalg' has no attribute 'slogdet'
.. [#fn-281] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.solve'
.. [#fn-282] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to svd at position 0.
.. [#fn-283] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to svd at position 0.
.. [#fn-284] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to svd at position 0.
.. [#fn-285] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.tensorinv'
.. [#fn-286] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.tensorinv'
.. [#fn-287] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.tensorsolve'
.. [#fn-288] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.tensorsolve'
.. [#fn-289] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.vector_norm'
.. [#fn-290] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fft'
.. [#fn-291] TypeError: fft_fft2() got an unexpected keyword argument 'axes'
.. [#fn-292] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fft2'
.. [#fn-293] AttributeError: module 'ndonnx' has no attribute `fft`
.. [#fn-294] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fftn'
.. [#fn-295] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fftshift'
.. [#fn-296] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifft'
.. [#fn-297] TypeError: fft_ifft2() got an unexpected keyword argument 'axes'
.. [#fn-298] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifft2'
.. [#fn-299] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifftn'
.. [#fn-300] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifftshift'
.. [#fn-301] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfft'
.. [#fn-302] TypeError: fft_irfft2() got an unexpected keyword argument 'axes'
.. [#fn-303] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfft2'
.. [#fn-304] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfftn'
.. [#fn-305] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfft'
.. [#fn-306] TypeError: fft_rfft2() got an unexpected keyword argument 'axes'
.. [#fn-307] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfft2'
.. [#fn-308] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfftn'
.. [#fn-309] TypeError: to() received an invalid combination of arguments - got (copy=bool, dtype=type, ), but expected one of: \* (torch.device device = None, torch.dtype dtype = None, bool non_blocking = False...
.. [#fn-310] AttributeError: type object 'numpy.float32' has no attribute '__ndx_cast_from__'
.. [#fn-311] AttributeError: module 'array_api_compat.torch' has no attribute 'copy'
.. [#fn-312] AttributeError: module 'array_api_compat.dask.array' has no attribute 'copy'
.. [#fn-313] AttributeError: module 'ndonnx' has no attribute `copy`
.. [#fn-314] TypeError: Cannot interpret 'torch.float64' as a data type
.. [#fn-315] InvalidInputException: Argument 'dask.array<array, shape=(3,), dtype=float64, chunksize=(3,), chunktype=numpy.ndarray>' of type <class 'dask.array.core.Array'> is not a valid JAX type.
.. [#fn-316] TypeError: Cannot interpret 'Float64' as a data type
.. [#fn-317] AttributeError: module 'array_api_compat.dask.array' has no attribute 'cross'
.. [#fn-318] AttributeError: module 'ndonnx' has no attribute `cross`
.. [#fn-319] RuntimeError: Unknown backend cuda. Available backends are ['cpu']
.. [#fn-320] TypeError: cumprod is not supported for quantities with units (has unit m), because each element of the result would have a different unit exponent. Use .prod() for a single reduction, or convert t...
.. [#fn-321] TypeError: diagonal() received an invalid combination of arguments - got (Tensor, axis1=int, axis2=int, offset=int), but expected one of: \* (Tensor input, \*, name outdim, name dim1, name dim2, int ...
.. [#fn-322] AttributeError: module 'ndonnx' has no attribute `diagonal`
.. [#fn-323] TypeError: Error interpreting argument to <function dot at 0x...> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at path a. This...
.. [#fn-324] TypeError: expand_dims() takes 1 positional argument but 2 were given
.. [#fn-325] ValueError: The dtype of the original data is float32, while we got float64.
.. [#fn-326] BackendError: Quantity.at indexed-update is not supported on the ndonnx backend. Call .to_numpy() (or another concrete backend) on the input first.
.. [#fn-327] AttributeError: module 'array_api_compat.dask.array' has no attribute 'float16'
.. [#fn-328] AttributeError: 'Array' object has no attribute 'item'
.. [#fn-329] TypeError: nancumprod is not supported for quantities with units (has unit m), because each element of the result would have a different unit exponent. Use .nanprod() for a single reduction, or con...
.. [#fn-330] TypeError: Error interpreting argument to <function outer at 0x...> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at path a. Th...
.. [#fn-331] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'set'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-332] TypeError: repeat() got some positional-only arguments passed as keyword arguments: 'repeats'
.. [#fn-333] ValueError: found array with object dtype but it contains non-string elements
.. [#fn-334] TypeError: round() received an invalid combination of arguments - got (Tensor, int), but expected one of: \* (Tensor input, \*, Tensor out = None) \* (Tensor input, \*, int decimals, Tensor out = None)
.. [#fn-335] TypeError: round() takes 1 positional argument but 2 were given
.. [#fn-336] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'add'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-337] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'divide'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy()...
.. [#fn-338] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'max'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-339] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'min'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-340] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'multiply'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy...
.. [#fn-341] AxisError: axis1: axis 0 is out of bounds for array of dimension 0
.. [#fn-342] TypeError: 'NoneType' object is not iterable
.. [#fn-343] TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
.. [#fn-344] AttributeError: module 'ndonnx' has no attribute `swapaxes`
.. [#fn-345] TypeError: Axis value must be an integer, got None
.. [#fn-346] TypeError: tile(): argument 'dims' (position 2) must be tuple of ints, not int
.. [#fn-347] TypeError: object of type 'int' has no len()
.. [#fn-348] cupy backend not installed
.. [#fn-349] TypeError: Value 'array(data: [1.0, 2.0, 3.0], dtype=float64)' with dtype object is not a valid JAX array type. Only arrays of numeric types are supported by JAX.
.. [#fn-350] ValueError: unable to infer dtype from `[1. 2. 3.]`
.. [#fn-351] ValueError: unable to infer dtype from `tensor([1., 2., 3.], dtype=torch.float64)`
.. [#fn-352] ValueError: unable to infer dtype from `dask.array<array, shape=(3,), dtype=float64, chunksize=(3,), chunktype=numpy.ndarray>`
.. [#fn-353] TypeError: len() of unsized object
.. [#fn-354] ValueError: ONNX provides no control over the used device
.. [#fn-355] BackendError (expected): Quantity.tolist() would materialize a dask-backed Quantity. Call `q.mantissa.compute()` first.
.. [#fn-356] AttributeError: 'Array' object has no attribute 'tolist'
.. [#fn-357] TypeError: trace() got an unexpected keyword argument 'offset'
.. [#fn-358] AttributeError: module 'ndonnx' has no attribute `trace`
.. [#fn-359] AttributeError: module 'ndonnx' has no attribute `transpose`
.. [#fn-360] TypeError: view() received an invalid combination of arguments - got (type), but expected one of: \* (torch.dtype dtype) didn't match because some of the arguments have invalid types: (!type!) \* (tu...
.. [#fn-361] AttributeError: 'Array' object has no attribute 'view'
