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
     - Mostly ⚠
     - Partial ⚠
   * - **saiunit.linalg**
     - Full ✓
     - Full ✓
     - ?
     - Mostly ⚠
     - Partial ⚠
     - Limited ✗
   * - **saiunit.fft**
     - Full ✓
     - Full ✓
     - ?
     - Full ✓
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
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``triu_indices``
     - ✓
     - ✓
     - ?
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
     - ?
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
     - ⊘ [#fn-1]_
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
     - ⊘ [#fn-2]_
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
     - ✗ [#fn-3]_
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
     - ⊘ [#fn-4]_
     - ✓
     - ⊘ [#fn-5]_
   * - ``arange``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``arccos``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-6]_
   * - ``arccosh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-7]_
   * - ``arcsin``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-8]_
   * - ``arcsinh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-9]_
   * - ``arctan``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-10]_
   * - ``arctan2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-11]_
   * - ``arctanh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-12]_
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
     - ⊘ [#fn-13]_
   * - ``around``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-14]_
     - ✓
     - ⊘ [#fn-15]_
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
     - ⊘ [#fn-16]_
     - ⊘ [#fn-17]_
     - ⊘ [#fn-18]_
   * - ``array_split``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-19]_
     - ⊘ [#fn-20]_
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
     - ⊘ [#fn-21]_
   * - ``atleast_2d``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-22]_
   * - ``atleast_3d``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-23]_
   * - ``average``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-24]_
     - ✓
     - ⊘ [#fn-25]_
   * - ``bincount``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-26]_
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
     - ⊘ [#fn-27]_
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
     - ⊘ [#fn-28]_
     - ✓
     - ⊘ [#fn-29]_
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
     - ⊘ [#fn-30]_
     - ✓
     - ⊘ [#fn-31]_
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
     - ⊘ [#fn-32]_
     - ✓
     - ⊘ [#fn-33]_
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
     - ⊘ [#fn-34]_
     - ⊘ [#fn-35]_
   * - ``compress``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-36]_
     - ✓
     - ⊘ [#fn-37]_
   * - ``concatenate``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-38]_
   * - ``conj``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-3]_
   * - ``conjugate``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-39]_
     - ⊘ [#fn-40]_
     - ✗ [#fn-3]_
   * - ``convolve``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-41]_
     - ⊘ [#fn-42]_
     - ⊘ [#fn-43]_
   * - ``copysign``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-44]_
   * - ``corrcoef``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-45]_
   * - ``correlate``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-46]_
     - ⊘ [#fn-47]_
     - ⊘ [#fn-48]_
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
     - ✓
     - ✓
     - ⊘ [#fn-49]_
   * - ``cross``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-50]_
     - ⊘ [#fn-51]_
   * - ``cumprod``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-52]_
   * - ``cumproduct``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-52]_
   * - ``cumsum``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-53]_
   * - ``deg2rad``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-54]_
   * - ``degrees``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-55]_
     - ✓
     - ⊘ [#fn-56]_
   * - ``diag``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-57]_
   * - ``diag_indices_from``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-58]_
     - ⊘ [#fn-59]_
     - ⊘ [#fn-60]_
   * - ``diagflat``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-61]_
     - ⊘ [#fn-62]_
   * - ``diagonal``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-63]_
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
     - ⊘ [#fn-64]_
     - ✓
     - ⊘ [#fn-65]_
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
     - ⊘ [#fn-66]_
     - ✓
     - ⊘ [#fn-67]_
   * - ``dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-68]_
   * - ``dsplit``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-69]_
     - ⊘ [#fn-70]_
   * - ``dstack``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-71]_
   * - ``ediff1d``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-72]_
     - ✓
     - ⊘ [#fn-73]_
   * - ``einsum``
     - ⊘ [#fn-74]_
     - ✓
     - ?
     - ⊘ [#fn-75]_
     - ⊘ [#fn-74]_
     - ⊘ [#fn-76]_
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
     - ✓
     - ✓
     - ✓
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
     - ⊘ [#fn-77]_
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
     - ✗ [#fn-44]_
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
     - ⊘ [#fn-78]_
     - ✓
     - ⊘ [#fn-79]_
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
     - ⊘ [#fn-80]_
     - ✓
     - ⊘ [#fn-81]_
   * - ``fill_diagonal``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-82]_
     - ⊘ [#fn-83]_
     - ⊘ [#fn-84]_
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
     - ⊘ [#fn-85]_
     - ✓
     - ⊘ [#fn-86]_
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
     - ⊘ [#fn-87]_
   * - ``flipud``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-88]_
   * - ``float_power``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-89]_
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
     - ⊘ [#fn-90]_
   * - ``fmin``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-91]_
   * - ``fmod``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-92]_
   * - ``frexp``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-93]_
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
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``gather``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-94]_
     - ⊘ [#fn-95]_
   * - ``gcd``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-96]_
     - ⊘ [#fn-97]_
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
     - ⊘ [#fn-98]_
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
     - ⊘ [#fn-99]_
     - ⊘ [#fn-100]_
   * - ``histogram``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-101]_
   * - ``hsplit``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-102]_
     - ⊘ [#fn-103]_
   * - ``hstack``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-104]_
   * - ``hypot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-44]_
   * - ``identity``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-105]_
     - ⊘ [#fn-106]_
     - ⊘ [#fn-107]_
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
     - ✗ [#fn-3]_
   * - ``inner``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-108]_
     - ⊘ [#fn-109]_
   * - ``interp``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-110]_
     - ⊘ [#fn-111]_
     - ⊘ [#fn-112]_
   * - ``intersect1d``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-113]_
     - ⊘ [#fn-114]_
     - ⊘ [#fn-115]_
   * - ``invert``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-116]_
     - ✓
     - ⊘ [#fn-117]_
   * - ``isclose``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-118]_
   * - ``iscomplexobj``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-119]_
     - ⊘ [#fn-120]_
     - ⊘ [#fn-121]_
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
     - ⊘ [#fn-122]_
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
     - ⊘ [#fn-123]_
     - ⊘ [#fn-124]_
   * - ``lcm``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-125]_
     - ⊘ [#fn-126]_
   * - ``ldexp``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-127]_
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
     - ⊘ [#fn-128]_
     - ✓
     - ⊘ [#fn-129]_
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
     - ✓
     - ✓
     - ✓
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
     - ✗ [#fn-44]_
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
     - ⊘ [#fn-130]_
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
     - ⊘ [#fn-131]_
     - ⊘ [#fn-132]_
     - ⊘ [#fn-133]_
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
     - ⊘ [#fn-134]_
     - ⊘ [#fn-135]_
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
     - ✓
     - ✗ [#fn-136]_
     - ⊘ [#fn-137]_
   * - ``meshgrid``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-138]_
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
     - ⊘ [#fn-139]_
     - ✓
     - ⊘ [#fn-140]_
   * - ``modf``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-141]_
     - ✓
     - ⊘ [#fn-142]_
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
     - ⊘ [#fn-143]_
     - ⊘ [#fn-144]_
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
     - ⊘ [#fn-145]_
   * - ``nanargmax``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-146]_
     - ✓
     - ⊘ [#fn-147]_
   * - ``nanargmin``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-148]_
     - ✓
     - ⊘ [#fn-149]_
   * - ``nancumprod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-150]_
     - ✓
     - ⊘ [#fn-151]_
   * - ``nancumsum``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-152]_
     - ✓
     - ⊘ [#fn-153]_
   * - ``nanmax``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-154]_
     - ✓
     - ⊘ [#fn-155]_
   * - ``nanmean``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-156]_
   * - ``nanmedian``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-157]_
     - ⊘ [#fn-158]_
   * - ``nanmin``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-159]_
     - ✓
     - ⊘ [#fn-160]_
   * - ``nanpercentile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-161]_
     - ✓
     - ⊘ [#fn-162]_
   * - ``nanprod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-163]_
     - ✓
     - ⊘ [#fn-164]_
   * - ``nanquantile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-165]_
     - ✓
     - ⊘ [#fn-166]_
   * - ``nanstd``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-167]_
     - ✓
     - ⊘ [#fn-168]_
   * - ``nansum``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-169]_
   * - ``nanvar``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-170]_
     - ✓
     - ⊘ [#fn-171]_
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
     - ✗ [#fn-44]_
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
     - ✓
     - ✓
     - ✓
   * - ``outer``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-172]_
   * - ``percentile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-173]_
     - ✓
     - ⊘ [#fn-174]_
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
     - ⊘ [#fn-175]_
     - ✓
     - ⊘ [#fn-176]_
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
     - ⊘ [#fn-177]_
     - ✓
     - ⊘ [#fn-178]_
   * - ``quantile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-179]_
     - ✓
     - ⊘ [#fn-180]_
   * - ``rad2deg``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-181]_
   * - ``radians``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-182]_
     - ✓
     - ⊘ [#fn-183]_
   * - ``ravel``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-184]_
   * - ``real``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-3]_
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
     - ⊘ [#fn-185]_
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
     - ✓
     - ✓
     - ✓
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
     - ⊘ [#fn-186]_
     - ✓
     - ⊘ [#fn-187]_
   * - ``rint``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-188]_
     - ✓
     - ⊘ [#fn-189]_
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
     - ✓
     - ✓
     - ⊘ [#fn-190]_
   * - ``round``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``row_stack``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-191]_
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
     - ⊘ [#fn-192]_
     - ✓
     - ⊘ [#fn-193]_
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
     - ✗ [#fn-44]_
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
     - ⊘ [#fn-194]_
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
     - ✓
     - ⊘ [#fn-19]_
     - ⊘ [#fn-20]_
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
     - ⊘ [#fn-195]_
     - ✓
     - ⊘ [#fn-195]_
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
     - ✓
     - ✓
     - ⊘ [#fn-196]_
   * - ``swish``
     - ✓
     - ✓
     - ?
     - 🅙
     - 🅙
     - 🅙
   * - ``take``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
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
     - ✓
     - ✓
     - ✓
   * - ``trace``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-197]_
   * - ``transpose``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-198]_
     - ✓
     - ⊘ [#fn-199]_
   * - ``tree_ones_like``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``tree_zeros_like``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``tri``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-200]_
     - ✓
     - ⊘ [#fn-201]_
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
     - ⊘ [#fn-202]_
     - ✓
     - ⊘ [#fn-203]_
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
     - ⊘ [#fn-204]_
     - ✓
     - ⊘ [#fn-205]_
   * - ``true_divide``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-206]_
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
     - ✓
     - ✓
     - ⊘ [#fn-207]_
   * - ``vander``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-208]_
     - ⊘ [#fn-209]_
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
     - ⊘ [#fn-210]_
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
     - ⊘ [#fn-211]_
     - ⊘ [#fn-212]_
   * - ``vstack``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-191]_
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
     - ✓
     - ✓
     - ✓

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
     - ⊘ [#fn-213]_
     - ✓
     - ⊘ [#fn-199]_
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
     - ✓
     - ✓
     - ✓
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
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-214]_
   * - ``cond``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-215]_
     - ⊘ [#fn-216]_
   * - ``cross``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-50]_
     - ⊘ [#fn-51]_
   * - ``det``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-217]_
     - ⊘ [#fn-218]_
   * - ``diagonal``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-63]_
   * - ``dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-68]_
   * - ``eig``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-219]_
     - ⊘ [#fn-220]_
     - ⊘ [#fn-221]_
   * - ``eigh``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-222]_
     - ⊘ [#fn-223]_
     - ⊘ [#fn-224]_
   * - ``eigvals``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-219]_
     - ⊘ [#fn-220]_
     - ⊘ [#fn-221]_
   * - ``eigvalsh``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-222]_
     - ⊘ [#fn-223]_
     - ⊘ [#fn-224]_
   * - ``inner``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-108]_
     - ⊘ [#fn-109]_
   * - ``inv``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-225]_
   * - ``kron``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-123]_
     - ⊘ [#fn-124]_
   * - ``lstsq``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-226]_
     - ⊘ [#fn-227]_
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
     - ⊘ [#fn-228]_
   * - ``matrix_power``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-134]_
     - ⊘ [#fn-135]_
   * - ``matrix_rank``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-229]_
     - ⊘ [#fn-230]_
   * - ``matrix_transpose``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-231]_
   * - ``multi_dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-143]_
     - ⊘ [#fn-144]_
   * - ``norm``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-232]_
   * - ``outer``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-172]_
   * - ``pinv``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-233]_
     - ⊘ [#fn-234]_
   * - ``qr``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-227]_
   * - ``slogdet``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-235]_
     - ⊘ [#fn-227]_
   * - ``solve``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-236]_
   * - ``svd``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-237]_
     - ⊘ [#fn-238]_
     - ⊘ [#fn-239]_
   * - ``svdvals``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-237]_
     - ⊘ [#fn-238]_
     - ⊘ [#fn-239]_
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
     - ⊘ [#fn-240]_
     - ⊘ [#fn-241]_
   * - ``tensorsolve``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-242]_
     - ⊘ [#fn-243]_
   * - ``trace``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-197]_
   * - ``vdot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-210]_
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
     - ⊘ [#fn-244]_

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
     - ⊘ [#fn-245]_
   * - ``fft2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-246]_
   * - ``fftfreq``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-247]_
   * - ``fftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-248]_
   * - ``fftshift``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-249]_
   * - ``ifft``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-250]_
   * - ``ifft2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-251]_
   * - ``ifftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-252]_
   * - ``ifftshift``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-253]_
   * - ``irfft``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-254]_
   * - ``irfft2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-255]_
   * - ``irfftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-256]_
   * - ``rfft``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-257]_
   * - ``rfft2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-258]_
   * - ``rfftfreq``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-247]_
   * - ``rfftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-259]_

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
     - ✓
     - ✓
     - ✓
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
     - ⊘ [#fn-260]_
     - ⊘ [#fn-261]_
     - ⊘ [#fn-262]_
   * - ``conj``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-44]_
   * - ``conjugate``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-44]_
   * - ``copy``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-260]_
     - ⊘ [#fn-261]_
     - ⊘ [#fn-262]_
   * - ``cpu``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-263]_
     - ✗ [#fn-264]_
     - ✗ [#fn-265]_
   * - ``cross``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-266]_
     - ⊘ [#fn-267]_
     - ⊘ [#fn-268]_
   * - ``cuda``
     - ⊘ [#fn-269]_
     - ⊘ [#fn-269]_
     - ?
     - ⊘ [#fn-269]_
     - ⊘ [#fn-269]_
     - ⊘ [#fn-269]_
   * - ``cumprod``
     - ⊘ [#fn-270]_
     - ⊘ [#fn-270]_
     - ?
     - ⊘ [#fn-270]_
     - ⊘ [#fn-270]_
     - ⊘ [#fn-270]_
   * - ``cumsum``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-271]_
     - ✓
     - ⊘ [#fn-53]_
   * - ``diagonal``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-272]_
     - ✓
     - ⊘ [#fn-273]_
   * - ``dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-274]_
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
     - ✓
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
     - ✗ [#fn-263]_
     - ✗ [#fn-275]_
     - ✗ [#fn-276]_
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
     - ⊘ [#fn-277]_
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
     - ⊘ [#fn-278]_
     - ⊘ [#fn-278]_
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
     - ⊘ [#fn-279]_
     - ⊘ [#fn-279]_
     - ?
     - ⊘ [#fn-279]_
     - ⊘ [#fn-279]_
     - ⊘ [#fn-279]_
   * - ``nanprod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-163]_
     - ✓
     - ⊘ [#fn-164]_
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
     - ⊘ [#fn-280]_
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
     - ⊘ [#fn-177]_
     - ✓
     - ⊘ [#fn-178]_
   * - ``put``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-263]_
     - ⊘ [#fn-281]_
     - ✗ [#fn-276]_
   * - ``ravel``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-184]_
   * - ``repeat``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
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
     - ✗ [#fn-263]_
     - ✗ [#fn-275]_
     - ✗ [#fn-282]_
   * - ``round``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``scatter_add``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-283]_
     - ✗ [#fn-276]_
   * - ``scatter_div``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-284]_
     - ✗ [#fn-276]_
   * - ``scatter_max``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-285]_
     - ✗ [#fn-276]_
   * - ``scatter_min``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-286]_
     - ✗ [#fn-276]_
   * - ``scatter_mul``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-287]_
     - ✗ [#fn-276]_
   * - ``scatter_sub``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-283]_
     - ✗ [#fn-276]_
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
     - ✗ [#fn-263]_
     - ✗ [#fn-275]_
     - ✗ [#fn-265]_
   * - ``split``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-288]_
     - ✓
     - ✗ [#fn-289]_
   * - ``squeeze``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-290]_
     - ✓
     - ✗ [#fn-291]_
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
     - ⊘ [#fn-292]_
   * - ``take``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-293]_
     - ✓
   * - ``tile``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-294]_
     - ✓
     - ✗ [#fn-295]_
   * - ``to``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``to_cupy``
     - ⊘ [#fn-296]_
     - ⊘ [#fn-296]_
     - ?
     - ⊘ [#fn-296]_
     - ⊘ [#fn-296]_
     - ⊘ [#fn-296]_
   * - ``to_dask``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-263]_
     - ✓
     - ✗ [#fn-265]_
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
     - ✗ [#fn-297]_
   * - ``to_ndonnx``
     - ✓
     - ⊘ [#fn-298]_
     - ?
     - ⊘ [#fn-299]_
     - ⊘ [#fn-300]_
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
     - ✗ [#fn-301]_
     - ✗ [#fn-302]_
   * - ``tolist``
     - ✓
     - ✓
     - ?
     - ✓
     - ⚠ [#fn-303]_
     - ⊘ [#fn-304]_
   * - ``trace``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-305]_
     - ✓
     - ⊘ [#fn-306]_
   * - ``transpose``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-198]_
     - ✓
     - ⊘ [#fn-307]_
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
     - ✓
   * - ``update_mantissa``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-263]_
     - ✗ [#fn-275]_
     - ✗ [#fn-265]_
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
     - ⊘ [#fn-308]_
     - ✓
     - ⊘ [#fn-309]_
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
.. [#fn-19] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'split'
.. [#fn-20] AttributeError: saiunit: backend 'ndonnx' has no operation 'split'
.. [#fn-21] AttributeError: saiunit: backend 'ndonnx' has no operation 'atleast_1d'
.. [#fn-22] AttributeError: saiunit: backend 'ndonnx' has no operation 'atleast_2d'
.. [#fn-23] AttributeError: saiunit: backend 'ndonnx' has no operation 'atleast_3d'
.. [#fn-24] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'average'
.. [#fn-25] AttributeError: saiunit: backend 'ndonnx' has no operation 'average'
.. [#fn-26] AttributeError: saiunit: backend 'ndonnx' has no operation 'bincount'
.. [#fn-27] AttributeError: saiunit: backend 'ndonnx' has no operation 'bitwise_not'
.. [#fn-28] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'block'
.. [#fn-29] AttributeError: saiunit: backend 'ndonnx' has no operation 'block'
.. [#fn-30] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'cbrt'
.. [#fn-31] AttributeError: saiunit: backend 'ndonnx' has no operation 'cbrt'
.. [#fn-32] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'choose'
.. [#fn-33] AttributeError: saiunit: backend 'ndonnx' has no operation 'choose'
.. [#fn-34] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'column_stack'
.. [#fn-35] AttributeError: saiunit: backend 'ndonnx' has no operation 'column_stack'
.. [#fn-36] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'compress'
.. [#fn-37] AttributeError: saiunit: backend 'ndonnx' has no operation 'compress'
.. [#fn-38] AttributeError: saiunit: backend 'ndonnx' has no operation 'concatenate'
.. [#fn-39] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'conjugate'
.. [#fn-40] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'conjugate'
.. [#fn-41] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'convolve'
.. [#fn-42] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'convolve'
.. [#fn-43] AttributeError: saiunit: backend 'ndonnx' has no operation 'convolve'
.. [#fn-44] NotImplementedError:
.. [#fn-45] AttributeError: saiunit: backend 'ndonnx' has no operation 'corrcoef'
.. [#fn-46] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'correlate'
.. [#fn-47] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'correlate'
.. [#fn-48] AttributeError: saiunit: backend 'ndonnx' has no operation 'correlate'
.. [#fn-49] AttributeError: saiunit: backend 'ndonnx' has no operation 'cov'
.. [#fn-50] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'cross'
.. [#fn-51] AttributeError: saiunit: backend 'ndonnx' has no operation 'cross'
.. [#fn-52] AttributeError: saiunit: backend 'ndonnx' has no operation 'cumprod'
.. [#fn-53] AttributeError: saiunit: backend 'ndonnx' has no operation 'cumsum'
.. [#fn-54] AttributeError: saiunit: backend 'ndonnx' has no operation 'deg2rad'
.. [#fn-55] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'degrees'
.. [#fn-56] AttributeError: saiunit: backend 'ndonnx' has no operation 'degrees'
.. [#fn-57] AttributeError: module 'ndonnx' has no attribute `diag`
.. [#fn-58] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'diag_indices_from'
.. [#fn-59] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'diag_indices_from'
.. [#fn-60] AttributeError: saiunit: backend 'ndonnx' has no operation 'diag_indices_from'
.. [#fn-61] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'diagflat'
.. [#fn-62] AttributeError: saiunit: backend 'ndonnx' has no operation 'diagflat'
.. [#fn-63] AttributeError: saiunit: backend 'ndonnx' has no operation 'diagonal'
.. [#fn-64] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'digitize'
.. [#fn-65] AttributeError: saiunit: backend 'ndonnx' has no operation 'digitize'
.. [#fn-66] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'divmod'
.. [#fn-67] AttributeError: saiunit: backend 'ndonnx' has no operation 'divmod'
.. [#fn-68] AttributeError: saiunit: backend 'ndonnx' has no operation 'dot'
.. [#fn-69] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'dsplit'
.. [#fn-70] AttributeError: saiunit: backend 'ndonnx' has no operation 'dsplit'
.. [#fn-71] AttributeError: saiunit: backend 'ndonnx' has no operation 'dstack'
.. [#fn-72] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'ediff1d'
.. [#fn-73] AttributeError: saiunit: backend 'ndonnx' has no operation 'ediff1d'
.. [#fn-74] TracerArrayConversionError: The numpy.ndarray conversion method __array__() was called on traced array with shape float32[2,2] The error occurred while tracing the function _einsum at /mnt/d/codes/...
.. [#fn-75] TypeError: Error interpreting argument to <function _einsum at 0x...> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path operands[0...
.. [#fn-76] TypeError: Error interpreting argument to <function _einsum at 0x...> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at path ope...
.. [#fn-77] AttributeError: saiunit: backend 'ndonnx' has no operation 'exp2'
.. [#fn-78] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'extract'
.. [#fn-79] AttributeError: saiunit: backend 'ndonnx' has no operation 'extract'
.. [#fn-80] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'fabs'
.. [#fn-81] AttributeError: saiunit: backend 'ndonnx' has no operation 'fabs'
.. [#fn-82] AttributeError: module 'array_api_compat.torch' has no attribute 'fill_diagonal'
.. [#fn-83] AttributeError: module 'array_api_compat.dask.array' has no attribute 'fill_diagonal'
.. [#fn-84] AttributeError: module 'ndonnx' has no attribute `fill_diagonal`
.. [#fn-85] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'flatnonzero'
.. [#fn-86] AttributeError: saiunit: backend 'ndonnx' has no operation 'flatnonzero'
.. [#fn-87] AttributeError: saiunit: backend 'ndonnx' has no operation 'fliplr'
.. [#fn-88] AttributeError: saiunit: backend 'ndonnx' has no operation 'flipud'
.. [#fn-89] AttributeError: saiunit: backend 'ndonnx' has no operation 'float_power'
.. [#fn-90] AttributeError: saiunit: backend 'ndonnx' has no operation 'fmax'
.. [#fn-91] AttributeError: saiunit: backend 'ndonnx' has no operation 'fmin'
.. [#fn-92] AttributeError: saiunit: backend 'ndonnx' has no operation 'fmod'
.. [#fn-93] AttributeError: saiunit: backend 'ndonnx' has no operation 'frexp'
.. [#fn-94] NotImplementedError: Don't yet support nd fancy indexing
.. [#fn-95] AttributeError: 'Array' object has no attribute 'reshape'
.. [#fn-96] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'gcd'
.. [#fn-97] AttributeError: saiunit: backend 'ndonnx' has no operation 'gcd'
.. [#fn-98] AttributeError: module 'ndonnx' has no attribute `gradient`
.. [#fn-99] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'heaviside'
.. [#fn-100] AttributeError: saiunit: backend 'ndonnx' has no operation 'heaviside'
.. [#fn-101] AttributeError: saiunit: backend 'ndonnx' has no operation 'histogram'
.. [#fn-102] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'hsplit'
.. [#fn-103] AttributeError: saiunit: backend 'ndonnx' has no operation 'hsplit'
.. [#fn-104] AttributeError: saiunit: backend 'ndonnx' has no operation 'hstack'
.. [#fn-105] AttributeError: module 'array_api_compat.torch' has no attribute 'identity'
.. [#fn-106] AttributeError: module 'array_api_compat.dask.array' has no attribute 'identity'
.. [#fn-107] AttributeError: module 'ndonnx' has no attribute `identity`
.. [#fn-108] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'inner'
.. [#fn-109] AttributeError: saiunit: backend 'ndonnx' has no operation 'inner'
.. [#fn-110] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'interp'
.. [#fn-111] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'interp'
.. [#fn-112] AttributeError: saiunit: backend 'ndonnx' has no operation 'interp'
.. [#fn-113] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'intersect1d'
.. [#fn-114] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'intersect1d'
.. [#fn-115] AttributeError: saiunit: backend 'ndonnx' has no operation 'intersect1d'
.. [#fn-116] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'invert'
.. [#fn-117] AttributeError: saiunit: backend 'ndonnx' has no operation 'invert'
.. [#fn-118] AttributeError: saiunit: backend 'ndonnx' has no operation 'isclose'
.. [#fn-119] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'iscomplexobj'
.. [#fn-120] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'iscomplexobj'
.. [#fn-121] AttributeError: saiunit: backend 'ndonnx' has no operation 'iscomplexobj'
.. [#fn-122] AttributeError: module 'ndonnx' has no attribute `isreal`
.. [#fn-123] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'kron'
.. [#fn-124] AttributeError: saiunit: backend 'ndonnx' has no operation 'kron'
.. [#fn-125] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'lcm'
.. [#fn-126] AttributeError: saiunit: backend 'ndonnx' has no operation 'lcm'
.. [#fn-127] AttributeError: saiunit: backend 'ndonnx' has no operation 'ldexp'
.. [#fn-128] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'left_shift'
.. [#fn-129] AttributeError: saiunit: backend 'ndonnx' has no operation 'left_shift'
.. [#fn-130] AttributeError: saiunit: backend 'ndonnx' has no operation 'logaddexp2'
.. [#fn-131] TypeError: logspace() received an invalid combination of arguments - got (float, float), but expected one of: \* (Tensor start, Tensor end, int steps, float base = 10.0, \*, Tensor out = None, torch....
.. [#fn-132] AttributeError: module 'array_api_compat.dask.array' has no attribute 'logspace'
.. [#fn-133] AttributeError: module 'ndonnx' has no attribute `logspace`
.. [#fn-134] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.matrix_power'
.. [#fn-135] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_power'
.. [#fn-136] NotImplementedError: The da.median function only works along an axis. The full algorithm is difficult to do in parallel
.. [#fn-137] AttributeError: saiunit: backend 'ndonnx' has no operation 'median'
.. [#fn-138] AttributeError: 'list' object has no attribute 'ndim'
.. [#fn-139] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'mod'
.. [#fn-140] AttributeError: saiunit: backend 'ndonnx' has no operation 'mod'
.. [#fn-141] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'modf'
.. [#fn-142] AttributeError: saiunit: backend 'ndonnx' has no operation 'modf'
.. [#fn-143] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.multi_dot'
.. [#fn-144] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.multi_dot'
.. [#fn-145] AttributeError: saiunit: backend 'ndonnx' has no operation 'nan_to_num'
.. [#fn-146] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanargmax'
.. [#fn-147] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanargmax'
.. [#fn-148] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanargmin'
.. [#fn-149] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanargmin'
.. [#fn-150] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nancumprod'
.. [#fn-151] AttributeError: saiunit: backend 'ndonnx' has no operation 'nancumprod'
.. [#fn-152] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nancumsum'
.. [#fn-153] AttributeError: saiunit: backend 'ndonnx' has no operation 'nancumsum'
.. [#fn-154] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanmax'
.. [#fn-155] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmax'
.. [#fn-156] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmean'
.. [#fn-157] NotImplementedError: The da.nanmedian function only works along an axis or a subset of axes. The full algorithm is difficult to do in parallel
.. [#fn-158] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmedian'
.. [#fn-159] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanmin'
.. [#fn-160] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmin'
.. [#fn-161] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanpercentile'
.. [#fn-162] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanpercentile'
.. [#fn-163] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanprod'
.. [#fn-164] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanprod'
.. [#fn-165] TypeError: nanquantile() received an invalid combination of arguments - got (Tensor), but expected one of: \* (Tensor input, Tensor q, int dim = None, bool keepdim = False, \*, str interpolation = "l...
.. [#fn-166] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanquantile'
.. [#fn-167] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanstd'
.. [#fn-168] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanstd'
.. [#fn-169] AttributeError: saiunit: backend 'ndonnx' has no operation 'nansum'
.. [#fn-170] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanvar'
.. [#fn-171] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanvar'
.. [#fn-172] AttributeError: saiunit: backend 'ndonnx' has no operation 'outer'
.. [#fn-173] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'percentile'
.. [#fn-174] AttributeError: saiunit: backend 'ndonnx' has no operation 'percentile'
.. [#fn-175] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'power'
.. [#fn-176] AttributeError: saiunit: backend 'ndonnx' has no operation 'power'
.. [#fn-177] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'ptp'
.. [#fn-178] AttributeError: saiunit: backend 'ndonnx' has no operation 'ptp'
.. [#fn-179] TypeError: quantile() received an invalid combination of arguments - got (Tensor), but expected one of: \* (Tensor input, Tensor q, int dim = None, bool keepdim = False, \*, str interpolation = "line...
.. [#fn-180] AttributeError: saiunit: backend 'ndonnx' has no operation 'quantile'
.. [#fn-181] AttributeError: saiunit: backend 'ndonnx' has no operation 'rad2deg'
.. [#fn-182] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'radians'
.. [#fn-183] AttributeError: saiunit: backend 'ndonnx' has no operation 'radians'
.. [#fn-184] AttributeError: saiunit: backend 'ndonnx' has no operation 'ravel'
.. [#fn-185] AttributeError: type object 'bool' has no attribute '__ndx_create__'
.. [#fn-186] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'right_shift'
.. [#fn-187] AttributeError: saiunit: backend 'ndonnx' has no operation 'right_shift'
.. [#fn-188] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'rint'
.. [#fn-189] AttributeError: saiunit: backend 'ndonnx' has no operation 'rint'
.. [#fn-190] AttributeError: saiunit: backend 'ndonnx' has no operation 'rot90'
.. [#fn-191] AttributeError: saiunit: backend 'ndonnx' has no operation 'vstack'
.. [#fn-192] TypeError: select() received an invalid combination of arguments - got (list, list), but expected one of: \* (Tensor input, name dim, int index) \* (Tensor input, int dim, int index)
.. [#fn-193] AttributeError: saiunit: backend 'ndonnx' has no operation 'select'
.. [#fn-194] AttributeError: saiunit: backend 'ndonnx' has no operation 'sinc'
.. [#fn-195] TypeError: squeeze() missing 1 required positional argument: 'axis'
.. [#fn-196] AttributeError: saiunit: backend 'ndonnx' has no operation 'swapaxes'
.. [#fn-197] AttributeError: saiunit: backend 'ndonnx' has no operation 'trace'
.. [#fn-198] TypeError: transpose() received an invalid combination of arguments - got (Tensor), but expected one of: \* (Tensor input, int dim0, int dim1) \* (Tensor input, name dim0, name dim1)
.. [#fn-199] AttributeError: saiunit: backend 'ndonnx' has no operation 'transpose'
.. [#fn-200] AttributeError: module 'array_api_compat.torch' has no attribute 'tri'
.. [#fn-201] AttributeError: module 'ndonnx' has no attribute `tri`
.. [#fn-202] AttributeError: module 'array_api_compat.torch' has no attribute 'tril_indices_from'
.. [#fn-203] AttributeError: module 'ndonnx' has no attribute `tril_indices_from`
.. [#fn-204] AttributeError: module 'array_api_compat.torch' has no attribute 'triu_indices_from'
.. [#fn-205] AttributeError: module 'ndonnx' has no attribute `triu_indices_from`
.. [#fn-206] AttributeError: saiunit: backend 'ndonnx' has no operation 'true_divide'
.. [#fn-207] AttributeError: saiunit: backend 'ndonnx' has no operation 'unique'
.. [#fn-208] AttributeError: module 'array_api_compat.dask.array' has no attribute 'vander'
.. [#fn-209] AttributeError: module 'ndonnx' has no attribute `vander`
.. [#fn-210] AttributeError: saiunit: backend 'ndonnx' has no operation 'vdot'
.. [#fn-211] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'vsplit'
.. [#fn-212] AttributeError: saiunit: backend 'ndonnx' has no operation 'vsplit'
.. [#fn-213] TypeError: transpose() received an invalid combination of arguments - got (Tensor, list), but expected one of: \* (Tensor input, int dim0, int dim1) \* (Tensor input, name dim0, name dim1)
.. [#fn-214] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.cholesky'
.. [#fn-215] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.cond'
.. [#fn-216] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.cond'
.. [#fn-217] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.det'
.. [#fn-218] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.det'
.. [#fn-219] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to eig at position 0.
.. [#fn-220] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to eig at position 0.
.. [#fn-221] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to eig at position 0.
.. [#fn-222] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to transpose at position 0.
.. [#fn-223] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to transpose at position 0.
.. [#fn-224] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to transpose at position 0.
.. [#fn-225] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.inv'
.. [#fn-226] TypeError: lstsq() got an unexpected keyword argument 'rcond'
.. [#fn-227] AttributeError: module 'ndonnx' has no attribute `linalg`
.. [#fn-228] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_norm'
.. [#fn-229] TypeError: svd() got an unexpected keyword argument 'compute_uv'
.. [#fn-230] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_rank'
.. [#fn-231] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_transpose'
.. [#fn-232] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.norm'
.. [#fn-233] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.pinv'
.. [#fn-234] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.pinv'
.. [#fn-235] AttributeError: module 'array_api_compat.dask.array.linalg' has no attribute 'slogdet'
.. [#fn-236] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.solve'
.. [#fn-237] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to svd at position 0.
.. [#fn-238] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to svd at position 0.
.. [#fn-239] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to svd at position 0.
.. [#fn-240] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.tensorinv'
.. [#fn-241] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.tensorinv'
.. [#fn-242] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.tensorsolve'
.. [#fn-243] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.tensorsolve'
.. [#fn-244] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.vector_norm'
.. [#fn-245] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fft'
.. [#fn-246] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fft2'
.. [#fn-247] AttributeError: module 'ndonnx' has no attribute `fft`
.. [#fn-248] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fftn'
.. [#fn-249] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fftshift'
.. [#fn-250] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifft'
.. [#fn-251] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifft2'
.. [#fn-252] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifftn'
.. [#fn-253] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifftshift'
.. [#fn-254] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfft'
.. [#fn-255] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfft2'
.. [#fn-256] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfftn'
.. [#fn-257] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfft'
.. [#fn-258] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfft2'
.. [#fn-259] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfftn'
.. [#fn-260] AttributeError: module 'array_api_compat.torch' has no attribute 'copy'
.. [#fn-261] AttributeError: module 'array_api_compat.dask.array' has no attribute 'copy'
.. [#fn-262] AttributeError: module 'ndonnx' has no attribute `copy`
.. [#fn-263] TypeError: Cannot interpret 'torch.float64' as a data type
.. [#fn-264] InvalidInputException: Argument 'dask.array<array, shape=(3,), dtype=float64, chunksize=(3,), chunktype=numpy.ndarray>' of type <class 'dask.array.core.Array'> is not a valid JAX type.
.. [#fn-265] TypeError: Cannot interpret 'Float64' as a data type
.. [#fn-266] TypeError: cross() got an unexpected keyword argument 'axisa'
.. [#fn-267] AttributeError: module 'array_api_compat.dask.array' has no attribute 'cross'
.. [#fn-268] AttributeError: module 'ndonnx' has no attribute `cross`
.. [#fn-269] RuntimeError: Unknown backend cuda. Available backends are ['cpu']
.. [#fn-270] TypeError: cumprod is not supported for quantities with units (has unit m), because each element of the result would have a different unit exponent. Use .prod() for a single reduction, or convert t...
.. [#fn-271] TypeError: cumsum() received an invalid combination of arguments - got (Tensor), but expected one of: \* (Tensor input, int dim, \*, torch.dtype dtype = None, Tensor out = None) \* (Tensor input, name...
.. [#fn-272] TypeError: diagonal() received an invalid combination of arguments - got (Tensor, axis1=int, axis2=int, offset=int), but expected one of: \* (Tensor input, \*, name outdim, name dim1, name dim2, int ...
.. [#fn-273] AttributeError: module 'ndonnx' has no attribute `diagonal`
.. [#fn-274] TypeError: Error interpreting argument to <function dot at 0x...> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at path a. This...
.. [#fn-275] ValueError: The dtype of the original data is float32, while we got float64.
.. [#fn-276] BackendError: Quantity.at indexed-update is not supported on the ndonnx backend. Call .to_numpy() (or another concrete backend) on the input first.
.. [#fn-277] AttributeError: module 'array_api_compat.dask.array' has no attribute 'float16'
.. [#fn-278] AttributeError: 'Array' object has no attribute 'item'
.. [#fn-279] TypeError: nancumprod is not supported for quantities with units (has unit m), because each element of the result would have a different unit exponent. Use .nanprod() for a single reduction, or con...
.. [#fn-280] TypeError: Error interpreting argument to <function outer at 0x...> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at path a. Th...
.. [#fn-281] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'set'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-282] ValueError: found array with object dtype but it contains non-string elements
.. [#fn-283] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'add'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-284] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'divide'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy()...
.. [#fn-285] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'max'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-286] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'min'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-287] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'multiply'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy...
.. [#fn-288] TypeError: split() got an unexpected keyword argument 'axis'
.. [#fn-289] AxisError: axis1: axis 0 is out of bounds for array of dimension 0
.. [#fn-290] TypeError: 'NoneType' object is not iterable
.. [#fn-291] TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
.. [#fn-292] AttributeError: module 'ndonnx' has no attribute `swapaxes`
.. [#fn-293] TypeError: Axis value must be an integer, got None
.. [#fn-294] TypeError: tile(): argument 'dims' (position 2) must be tuple of ints, not int
.. [#fn-295] TypeError: object of type 'int' has no len()
.. [#fn-296] cupy backend not installed
.. [#fn-297] TypeError: Value 'array(data: [1.0, 2.0, 3.0], dtype=float64)' with dtype object is not a valid JAX array type. Only arrays of numeric types are supported by JAX.
.. [#fn-298] ValueError: unable to infer dtype from `[1. 2. 3.]`
.. [#fn-299] ValueError: unable to infer dtype from `tensor([1., 2., 3.], dtype=torch.float64)`
.. [#fn-300] ValueError: unable to infer dtype from `dask.array<array, shape=(3,), dtype=float64, chunksize=(3,), chunktype=numpy.ndarray>`
.. [#fn-301] TypeError: len() of unsized object
.. [#fn-302] ValueError: ONNX provides no control over the used device
.. [#fn-303] BackendError (expected): Quantity.tolist() would materialize a dask-backed Quantity. Call `q.mantissa.compute()` first.
.. [#fn-304] AttributeError: 'Array' object has no attribute 'tolist'
.. [#fn-305] TypeError: trace() got an unexpected keyword argument 'offset'
.. [#fn-306] AttributeError: module 'ndonnx' has no attribute `trace`
.. [#fn-307] AttributeError: module 'ndonnx' has no attribute `transpose`
.. [#fn-308] TypeError: view() received an invalid combination of arguments - got (type), but expected one of: \* (torch.dtype dtype) didn't match because some of the arguments have invalid types: (!type!) \* (tu...
.. [#fn-309] AttributeError: 'Array' object has no attribute 'view'
