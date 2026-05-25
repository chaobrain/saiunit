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
     - ⊘ [#fn-1]_
   * - ``triu_indices``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
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
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-3]_
     - ✓
     - ⊘ [#fn-4]_

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
     - ⊘ [#fn-5]_
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
     - ⊘ [#fn-6]_
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
     - ✗ [#fn-7]_
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
     - ⊘ [#fn-8]_
     - ✓
     - ⊘ [#fn-9]_
   * - ``arange``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-10]_
     - ⊘ [#fn-11]_
     - ⊘ [#fn-12]_
   * - ``arccos``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-13]_
   * - ``arccosh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-14]_
   * - ``arcsin``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-15]_
   * - ``arcsinh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-16]_
   * - ``arctan``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-17]_
   * - ``arctan2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-18]_
   * - ``arctanh``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-19]_
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
     - ⊘ [#fn-20]_
   * - ``around``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-21]_
     - ✓
     - ⊘ [#fn-22]_
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
     - ⊘ [#fn-23]_
     - ⊘ [#fn-24]_
     - ⊘ [#fn-25]_
   * - ``array_split``
     - ✓
     - ✓
     - ?
     - ✓
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
     - ✓
     - ⊘ [#fn-40]_
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
     - ⊘ [#fn-41]_
     - ⊘ [#fn-42]_
   * - ``compress``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-43]_
     - ✓
     - ⊘ [#fn-44]_
   * - ``concatenate``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-45]_
   * - ``conj``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-7]_
   * - ``conjugate``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-46]_
     - ⊘ [#fn-47]_
     - ✗ [#fn-7]_
   * - ``convolve``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-48]_
     - ⊘ [#fn-49]_
     - ⊘ [#fn-50]_
   * - ``copysign``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-51]_
   * - ``corrcoef``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-52]_
   * - ``correlate``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-53]_
     - ⊘ [#fn-54]_
     - ⊘ [#fn-55]_
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
     - ⊘ [#fn-56]_
   * - ``cross``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-57]_
     - ⊘ [#fn-58]_
   * - ``cumprod``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-59]_
   * - ``cumproduct``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-59]_
   * - ``cumsum``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-60]_
   * - ``deg2rad``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-61]_
   * - ``degrees``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-62]_
     - ✓
     - ⊘ [#fn-63]_
   * - ``diag``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-64]_
   * - ``diag_indices_from``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-65]_
     - ⊘ [#fn-66]_
     - ⊘ [#fn-67]_
   * - ``diagflat``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-68]_
     - ⊘ [#fn-69]_
   * - ``diagonal``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-70]_
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
     - ⊘ [#fn-71]_
     - ✓
     - ⊘ [#fn-72]_
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
     - ⊘ [#fn-73]_
     - ✓
     - ⊘ [#fn-74]_
   * - ``dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-75]_
   * - ``dsplit``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-76]_
     - ⊘ [#fn-77]_
   * - ``dstack``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-78]_
   * - ``ediff1d``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-79]_
     - ✓
     - ⊘ [#fn-80]_
   * - ``einsum``
     - ⊘ [#fn-81]_
     - ✓
     - ?
     - ⊘ [#fn-82]_
     - ⊘ [#fn-81]_
     - ⊘ [#fn-83]_
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
     - ⊘ [#fn-84]_
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
     - ✗ [#fn-51]_
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
     - ⊘ [#fn-85]_
     - ✓
     - ⊘ [#fn-86]_
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
     - ⊘ [#fn-87]_
     - ✓
     - ⊘ [#fn-88]_
   * - ``fill_diagonal``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-89]_
     - ⊘ [#fn-90]_
     - ⊘ [#fn-91]_
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
     - ⊘ [#fn-92]_
     - ✓
     - ⊘ [#fn-93]_
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
     - ⊘ [#fn-94]_
   * - ``flipud``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-95]_
   * - ``float_power``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-96]_
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
     - ⊘ [#fn-97]_
   * - ``fmin``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-98]_
   * - ``fmod``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-99]_
   * - ``frexp``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-100]_
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
     - ✗ [#fn-101]_
     - ⊘ [#fn-102]_
   * - ``gcd``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-103]_
     - ⊘ [#fn-104]_
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
     - ⊘ [#fn-105]_
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
     - ⊘ [#fn-106]_
     - ⊘ [#fn-107]_
   * - ``histogram``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-108]_
   * - ``hsplit``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-109]_
     - ⊘ [#fn-110]_
   * - ``hstack``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-111]_
   * - ``hypot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-51]_
   * - ``identity``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-112]_
     - ⊘ [#fn-113]_
     - ⊘ [#fn-114]_
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
     - ✗ [#fn-7]_
   * - ``inner``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-115]_
     - ⊘ [#fn-116]_
   * - ``interp``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-117]_
     - ⊘ [#fn-118]_
     - ⊘ [#fn-119]_
   * - ``intersect1d``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-120]_
     - ⊘ [#fn-121]_
     - ⊘ [#fn-122]_
   * - ``invert``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-123]_
     - ✓
     - ⊘ [#fn-124]_
   * - ``isclose``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-125]_
   * - ``iscomplexobj``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-126]_
     - ⊘ [#fn-127]_
     - ⊘ [#fn-128]_
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
     - ⊘ [#fn-129]_
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
     - ⊘ [#fn-130]_
     - ⊘ [#fn-131]_
   * - ``lcm``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-132]_
     - ⊘ [#fn-133]_
   * - ``ldexp``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-134]_
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
     - ⊘ [#fn-135]_
     - ✓
     - ⊘ [#fn-136]_
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
     - ✗ [#fn-51]_
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
     - ⊘ [#fn-137]_
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
     - ⊘ [#fn-138]_
     - ⊘ [#fn-139]_
     - ⊘ [#fn-140]_
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
     - ⊘ [#fn-141]_
     - ⊘ [#fn-142]_
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
     - ✗ [#fn-143]_
     - ⊘ [#fn-144]_
   * - ``meshgrid``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-145]_
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
     - ⊘ [#fn-146]_
     - ✓
     - ⊘ [#fn-147]_
   * - ``modf``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-148]_
     - ✓
     - ⊘ [#fn-149]_
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
     - ⊘ [#fn-150]_
     - ⊘ [#fn-151]_
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
     - ⊘ [#fn-152]_
   * - ``nanargmax``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-153]_
     - ✓
     - ⊘ [#fn-154]_
   * - ``nanargmin``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-155]_
     - ✓
     - ⊘ [#fn-156]_
   * - ``nancumprod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-157]_
     - ✓
     - ⊘ [#fn-158]_
   * - ``nancumsum``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-159]_
     - ✓
     - ⊘ [#fn-160]_
   * - ``nanmax``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-161]_
     - ✓
     - ⊘ [#fn-162]_
   * - ``nanmean``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-163]_
   * - ``nanmedian``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-164]_
     - ⊘ [#fn-165]_
   * - ``nanmin``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-166]_
     - ✓
     - ⊘ [#fn-167]_
   * - ``nanpercentile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-168]_
     - ✓
     - ⊘ [#fn-169]_
   * - ``nanprod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-170]_
     - ✓
     - ⊘ [#fn-171]_
   * - ``nanquantile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-172]_
     - ✓
     - ⊘ [#fn-173]_
   * - ``nanstd``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-174]_
     - ✓
     - ⊘ [#fn-175]_
   * - ``nansum``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-176]_
   * - ``nanvar``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-177]_
     - ✓
     - ⊘ [#fn-178]_
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
     - ✗ [#fn-51]_
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
     - ⊘ [#fn-179]_
   * - ``percentile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-180]_
     - ✓
     - ⊘ [#fn-181]_
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
     - ⊘ [#fn-182]_
     - ✓
     - ⊘ [#fn-183]_
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
     - ⊘ [#fn-184]_
     - ✓
     - ⊘ [#fn-185]_
   * - ``quantile``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-186]_
     - ✓
     - ⊘ [#fn-187]_
   * - ``rad2deg``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-188]_
   * - ``radians``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-189]_
     - ✓
     - ⊘ [#fn-190]_
   * - ``ravel``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-191]_
   * - ``real``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-7]_
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
     - ⊘ [#fn-192]_
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
     - ⊘ [#fn-193]_
     - ✓
     - ⊘ [#fn-194]_
   * - ``rint``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-195]_
     - ✓
     - ⊘ [#fn-196]_
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
     - ⊘ [#fn-197]_
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
     - ⊘ [#fn-198]_
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
     - ⊘ [#fn-199]_
     - ✓
     - ⊘ [#fn-200]_
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
     - ✗ [#fn-51]_
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
     - ⊘ [#fn-201]_
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
     - ⊘ [#fn-202]_
     - ✓
     - ⊘ [#fn-202]_
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
     - ⊘ [#fn-203]_
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
     - ⊘ [#fn-204]_
   * - ``transpose``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-205]_
     - ✓
     - ⊘ [#fn-206]_
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
     - ⊘ [#fn-207]_
     - ✓
     - ⊘ [#fn-208]_
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
     - ⊘ [#fn-209]_
     - ✓
     - ⊘ [#fn-210]_
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
     - ⊘ [#fn-211]_
     - ✓
     - ⊘ [#fn-212]_
   * - ``true_divide``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-213]_
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
     - ⊘ [#fn-214]_
   * - ``vander``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-215]_
     - ⊘ [#fn-216]_
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
     - ⊘ [#fn-217]_
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
     - ⊘ [#fn-218]_
     - ⊘ [#fn-219]_
   * - ``vstack``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-198]_
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
     - ⊘ [#fn-220]_
     - ✓
     - ⊘ [#fn-206]_
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
     - ⊘ [#fn-221]_
   * - ``cond``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-222]_
     - ⊘ [#fn-223]_
   * - ``cross``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-57]_
     - ⊘ [#fn-58]_
   * - ``det``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-224]_
     - ⊘ [#fn-225]_
   * - ``diagonal``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-70]_
   * - ``dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-75]_
   * - ``eig``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-226]_
     - ⊘ [#fn-227]_
     - ⊘ [#fn-228]_
   * - ``eigh``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-229]_
     - ⊘ [#fn-230]_
     - ⊘ [#fn-231]_
   * - ``eigvals``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-226]_
     - ⊘ [#fn-227]_
     - ⊘ [#fn-228]_
   * - ``eigvalsh``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-229]_
     - ⊘ [#fn-230]_
     - ⊘ [#fn-231]_
   * - ``inner``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-115]_
     - ⊘ [#fn-116]_
   * - ``inv``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-232]_
   * - ``kron``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-130]_
     - ⊘ [#fn-131]_
   * - ``lstsq``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-233]_
     - ⊘ [#fn-234]_
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
     - ⊘ [#fn-235]_
   * - ``matrix_power``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-141]_
     - ⊘ [#fn-142]_
   * - ``matrix_rank``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-236]_
     - ⊘ [#fn-237]_
   * - ``matrix_transpose``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-238]_
   * - ``multi_dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-150]_
     - ⊘ [#fn-151]_
   * - ``norm``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-239]_
   * - ``outer``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-179]_
   * - ``pinv``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-240]_
     - ⊘ [#fn-241]_
   * - ``qr``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-234]_
   * - ``slogdet``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-242]_
     - ⊘ [#fn-234]_
   * - ``solve``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-243]_
   * - ``svd``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-244]_
     - ⊘ [#fn-245]_
     - ⊘ [#fn-246]_
   * - ``svdvals``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-244]_
     - ⊘ [#fn-245]_
     - ⊘ [#fn-246]_
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
     - ⊘ [#fn-247]_
     - ⊘ [#fn-248]_
   * - ``tensorsolve``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-249]_
     - ⊘ [#fn-250]_
   * - ``trace``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-204]_
   * - ``vdot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-217]_
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
     - ⊘ [#fn-251]_

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
     - ⊘ [#fn-252]_
   * - ``fft2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-253]_
   * - ``fftfreq``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-254]_
   * - ``fftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-255]_
   * - ``fftshift``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-256]_
   * - ``ifft``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-257]_
   * - ``ifft2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-258]_
   * - ``ifftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-259]_
   * - ``ifftshift``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-260]_
   * - ``irfft``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-261]_
   * - ``irfft2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-262]_
   * - ``irfftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-263]_
   * - ``rfft``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-264]_
   * - ``rfft2``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-265]_
   * - ``rfftfreq``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-254]_
   * - ``rfftn``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-266]_

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
     - ⊘ [#fn-3]_
     - ✓
     - ⊘ [#fn-4]_
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
     - ⊘ [#fn-267]_
     - ⊘ [#fn-268]_
     - ⊘ [#fn-269]_
   * - ``conj``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-51]_
   * - ``conjugate``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✗ [#fn-51]_
   * - ``copy``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-267]_
     - ⊘ [#fn-268]_
     - ⊘ [#fn-269]_
   * - ``cpu``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-270]_
     - ✗ [#fn-271]_
     - ✗ [#fn-272]_
   * - ``cross``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-273]_
     - ⊘ [#fn-274]_
     - ⊘ [#fn-275]_
   * - ``cuda``
     - ⊘ [#fn-276]_
     - ⊘ [#fn-276]_
     - ?
     - ⊘ [#fn-276]_
     - ⊘ [#fn-276]_
     - ⊘ [#fn-276]_
   * - ``cumprod``
     - ⊘ [#fn-277]_
     - ⊘ [#fn-277]_
     - ?
     - ⊘ [#fn-277]_
     - ⊘ [#fn-277]_
     - ⊘ [#fn-277]_
   * - ``cumsum``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-278]_
     - ✓
     - ⊘ [#fn-60]_
   * - ``diagonal``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-279]_
     - ✓
     - ⊘ [#fn-280]_
   * - ``dot``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-281]_
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
     - ✗ [#fn-270]_
     - ✗ [#fn-282]_
     - ✗ [#fn-283]_
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
     - ⊘ [#fn-284]_
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
     - ⊘ [#fn-285]_
     - ⊘ [#fn-285]_
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
     - ⊘ [#fn-286]_
     - ⊘ [#fn-286]_
     - ?
     - ⊘ [#fn-286]_
     - ⊘ [#fn-286]_
     - ⊘ [#fn-286]_
   * - ``nanprod``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-170]_
     - ✓
     - ⊘ [#fn-171]_
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
     - ⊘ [#fn-287]_
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
     - ⊘ [#fn-184]_
     - ✓
     - ⊘ [#fn-185]_
   * - ``put``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-270]_
     - ⊘ [#fn-288]_
     - ✗ [#fn-283]_
   * - ``ravel``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ⊘ [#fn-191]_
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
     - ✗ [#fn-270]_
     - ✗ [#fn-282]_
     - ✗ [#fn-289]_
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
     - ⊘ [#fn-290]_
     - ✗ [#fn-283]_
   * - ``scatter_div``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-291]_
     - ✗ [#fn-283]_
   * - ``scatter_max``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-292]_
     - ✗ [#fn-283]_
   * - ``scatter_min``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-293]_
     - ✗ [#fn-283]_
   * - ``scatter_mul``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-294]_
     - ✗ [#fn-283]_
   * - ``scatter_sub``
     - ✓
     - ✓
     - ?
     - ✓
     - ⊘ [#fn-290]_
     - ✗ [#fn-283]_
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
     - ✗ [#fn-270]_
     - ✗ [#fn-282]_
     - ✗ [#fn-272]_
   * - ``split``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-295]_
     - ✓
     - ✗ [#fn-296]_
   * - ``squeeze``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-297]_
     - ✓
     - ✗ [#fn-298]_
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
     - ⊘ [#fn-299]_
   * - ``take``
     - ✓
     - ✓
     - ?
     - ✓
     - ✗ [#fn-300]_
     - ✓
   * - ``tile``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-301]_
     - ✓
     - ✗ [#fn-302]_
   * - ``to``
     - ✓
     - ✓
     - ?
     - ✓
     - ✓
     - ✓
   * - ``to_cupy``
     - ⊘ [#fn-303]_
     - ⊘ [#fn-303]_
     - ?
     - ⊘ [#fn-303]_
     - ⊘ [#fn-303]_
     - ⊘ [#fn-303]_
   * - ``to_dask``
     - ✓
     - ✓
     - ?
     - ✗ [#fn-270]_
     - ✓
     - ✗ [#fn-272]_
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
     - ✗ [#fn-304]_
   * - ``to_ndonnx``
     - ✓
     - ⊘ [#fn-305]_
     - ?
     - ⊘ [#fn-306]_
     - ⊘ [#fn-307]_
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
     - ✗ [#fn-308]_
     - ✗ [#fn-309]_
   * - ``tolist``
     - ✓
     - ✓
     - ?
     - ✓
     - ⚠ [#fn-310]_
     - ⊘ [#fn-311]_
   * - ``trace``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-312]_
     - ✓
     - ⊘ [#fn-313]_
   * - ``transpose``
     - ✓
     - ✓
     - ?
     - ⊘ [#fn-205]_
     - ✓
     - ⊘ [#fn-314]_
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
     - ✗ [#fn-270]_
     - ✗ [#fn-282]_
     - ✗ [#fn-272]_
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
     - ⊘ [#fn-315]_
     - ✓
     - ⊘ [#fn-316]_
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


.. [#fn-1] AttributeError: module 'ndonnx' has no attribute `tril_indices`
.. [#fn-2] AttributeError: module 'ndonnx' has no attribute `triu_indices`
.. [#fn-3] TypeError: to() received an invalid combination of arguments - got (copy=bool, dtype=type, ), but expected one of: \* (torch.device device = None, torch.dtype dtype = None, bool non_blocking = False...
.. [#fn-4] AttributeError: type object 'numpy.float32' has no attribute '__ndx_cast_from__'
.. [#fn-5] AttributeError: saiunit: backend 'ndonnx' has no operation 'absolute'
.. [#fn-6] AttributeError: module 'ndonnx' has no attribute `allclose`
.. [#fn-7] ValueError: 'complex128' does not have a corresponding ndonnx data type
.. [#fn-8] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'append'
.. [#fn-9] AttributeError: saiunit: backend 'ndonnx' has no operation 'append'
.. [#fn-10] TypeError: '>' not supported between instances of 'NoneType' and 'int'
.. [#fn-11] TypeError: An error occurred while calling the arange method registered to the numpy backend. Original Message: unsupported operand type(s) for /: 'int' and 'NoneType'
.. [#fn-12] ValueError: 'arange' is not implemented for the provided inputs
.. [#fn-13] AttributeError: saiunit: backend 'ndonnx' has no operation 'arccos'
.. [#fn-14] AttributeError: saiunit: backend 'ndonnx' has no operation 'arccosh'
.. [#fn-15] AttributeError: saiunit: backend 'ndonnx' has no operation 'arcsin'
.. [#fn-16] AttributeError: saiunit: backend 'ndonnx' has no operation 'arcsinh'
.. [#fn-17] AttributeError: saiunit: backend 'ndonnx' has no operation 'arctan'
.. [#fn-18] AttributeError: saiunit: backend 'ndonnx' has no operation 'arctan2'
.. [#fn-19] AttributeError: saiunit: backend 'ndonnx' has no operation 'arctanh'
.. [#fn-20] AttributeError: saiunit: backend 'ndonnx' has no operation 'argwhere'
.. [#fn-21] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'around'
.. [#fn-22] AttributeError: saiunit: backend 'ndonnx' has no operation 'around'
.. [#fn-23] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'array_equal'
.. [#fn-24] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'array_equal'
.. [#fn-25] AttributeError: saiunit: backend 'ndonnx' has no operation 'array_equal'
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
.. [#fn-40] AttributeError: saiunit: backend 'ndonnx' has no operation 'choose'
.. [#fn-41] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'column_stack'
.. [#fn-42] AttributeError: saiunit: backend 'ndonnx' has no operation 'column_stack'
.. [#fn-43] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'compress'
.. [#fn-44] AttributeError: saiunit: backend 'ndonnx' has no operation 'compress'
.. [#fn-45] AttributeError: saiunit: backend 'ndonnx' has no operation 'concatenate'
.. [#fn-46] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'conjugate'
.. [#fn-47] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'conjugate'
.. [#fn-48] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'convolve'
.. [#fn-49] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'convolve'
.. [#fn-50] AttributeError: saiunit: backend 'ndonnx' has no operation 'convolve'
.. [#fn-51] NotImplementedError:
.. [#fn-52] AttributeError: saiunit: backend 'ndonnx' has no operation 'corrcoef'
.. [#fn-53] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'correlate'
.. [#fn-54] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'correlate'
.. [#fn-55] AttributeError: saiunit: backend 'ndonnx' has no operation 'correlate'
.. [#fn-56] AttributeError: saiunit: backend 'ndonnx' has no operation 'cov'
.. [#fn-57] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'cross'
.. [#fn-58] AttributeError: saiunit: backend 'ndonnx' has no operation 'cross'
.. [#fn-59] AttributeError: saiunit: backend 'ndonnx' has no operation 'cumprod'
.. [#fn-60] AttributeError: saiunit: backend 'ndonnx' has no operation 'cumsum'
.. [#fn-61] AttributeError: saiunit: backend 'ndonnx' has no operation 'deg2rad'
.. [#fn-62] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'degrees'
.. [#fn-63] AttributeError: saiunit: backend 'ndonnx' has no operation 'degrees'
.. [#fn-64] AttributeError: module 'ndonnx' has no attribute `diag`
.. [#fn-65] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'diag_indices_from'
.. [#fn-66] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'diag_indices_from'
.. [#fn-67] AttributeError: saiunit: backend 'ndonnx' has no operation 'diag_indices_from'
.. [#fn-68] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'diagflat'
.. [#fn-69] AttributeError: saiunit: backend 'ndonnx' has no operation 'diagflat'
.. [#fn-70] AttributeError: saiunit: backend 'ndonnx' has no operation 'diagonal'
.. [#fn-71] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'digitize'
.. [#fn-72] AttributeError: saiunit: backend 'ndonnx' has no operation 'digitize'
.. [#fn-73] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'divmod'
.. [#fn-74] AttributeError: saiunit: backend 'ndonnx' has no operation 'divmod'
.. [#fn-75] AttributeError: saiunit: backend 'ndonnx' has no operation 'dot'
.. [#fn-76] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'dsplit'
.. [#fn-77] AttributeError: saiunit: backend 'ndonnx' has no operation 'dsplit'
.. [#fn-78] AttributeError: saiunit: backend 'ndonnx' has no operation 'dstack'
.. [#fn-79] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'ediff1d'
.. [#fn-80] AttributeError: saiunit: backend 'ndonnx' has no operation 'ediff1d'
.. [#fn-81] TracerArrayConversionError: The numpy.ndarray conversion method __array__() was called on traced array with shape float32[2,2] The error occurred while tracing the function _einsum at /mnt/d/codes/...
.. [#fn-82] TypeError: Error interpreting argument to <function _einsum at 0x...> as an abstract array. The problematic value is of type <class 'torch.Tensor'> and was passed to the function at path operands[0...
.. [#fn-83] TypeError: Error interpreting argument to <function _einsum at 0x...> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at path ope...
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
.. [#fn-101] NotImplementedError: Don't yet support nd fancy indexing
.. [#fn-102] AttributeError: 'Array' object has no attribute 'reshape'
.. [#fn-103] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'gcd'
.. [#fn-104] AttributeError: saiunit: backend 'ndonnx' has no operation 'gcd'
.. [#fn-105] AttributeError: module 'ndonnx' has no attribute `gradient`
.. [#fn-106] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'heaviside'
.. [#fn-107] AttributeError: saiunit: backend 'ndonnx' has no operation 'heaviside'
.. [#fn-108] AttributeError: saiunit: backend 'ndonnx' has no operation 'histogram'
.. [#fn-109] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'hsplit'
.. [#fn-110] AttributeError: saiunit: backend 'ndonnx' has no operation 'hsplit'
.. [#fn-111] AttributeError: saiunit: backend 'ndonnx' has no operation 'hstack'
.. [#fn-112] AttributeError: module 'array_api_compat.torch' has no attribute 'identity'
.. [#fn-113] AttributeError: module 'array_api_compat.dask.array' has no attribute 'identity'
.. [#fn-114] AttributeError: module 'ndonnx' has no attribute `identity`
.. [#fn-115] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'inner'
.. [#fn-116] AttributeError: saiunit: backend 'ndonnx' has no operation 'inner'
.. [#fn-117] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'interp'
.. [#fn-118] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'interp'
.. [#fn-119] AttributeError: saiunit: backend 'ndonnx' has no operation 'interp'
.. [#fn-120] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'intersect1d'
.. [#fn-121] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'intersect1d'
.. [#fn-122] AttributeError: saiunit: backend 'ndonnx' has no operation 'intersect1d'
.. [#fn-123] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'invert'
.. [#fn-124] AttributeError: saiunit: backend 'ndonnx' has no operation 'invert'
.. [#fn-125] AttributeError: saiunit: backend 'ndonnx' has no operation 'isclose'
.. [#fn-126] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'iscomplexobj'
.. [#fn-127] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'iscomplexobj'
.. [#fn-128] AttributeError: saiunit: backend 'ndonnx' has no operation 'iscomplexobj'
.. [#fn-129] AttributeError: module 'ndonnx' has no attribute `isreal`
.. [#fn-130] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'kron'
.. [#fn-131] AttributeError: saiunit: backend 'ndonnx' has no operation 'kron'
.. [#fn-132] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'lcm'
.. [#fn-133] AttributeError: saiunit: backend 'ndonnx' has no operation 'lcm'
.. [#fn-134] AttributeError: saiunit: backend 'ndonnx' has no operation 'ldexp'
.. [#fn-135] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'left_shift'
.. [#fn-136] AttributeError: saiunit: backend 'ndonnx' has no operation 'left_shift'
.. [#fn-137] AttributeError: saiunit: backend 'ndonnx' has no operation 'logaddexp2'
.. [#fn-138] TypeError: logspace() received an invalid combination of arguments - got (float, float), but expected one of: \* (Tensor start, Tensor end, int steps, float base = 10.0, \*, Tensor out = None, torch....
.. [#fn-139] AttributeError: module 'array_api_compat.dask.array' has no attribute 'logspace'
.. [#fn-140] AttributeError: module 'ndonnx' has no attribute `logspace`
.. [#fn-141] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.matrix_power'
.. [#fn-142] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_power'
.. [#fn-143] NotImplementedError: The da.median function only works along an axis. The full algorithm is difficult to do in parallel
.. [#fn-144] AttributeError: saiunit: backend 'ndonnx' has no operation 'median'
.. [#fn-145] AttributeError: 'list' object has no attribute 'ndim'
.. [#fn-146] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'mod'
.. [#fn-147] AttributeError: saiunit: backend 'ndonnx' has no operation 'mod'
.. [#fn-148] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'modf'
.. [#fn-149] AttributeError: saiunit: backend 'ndonnx' has no operation 'modf'
.. [#fn-150] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.multi_dot'
.. [#fn-151] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.multi_dot'
.. [#fn-152] AttributeError: saiunit: backend 'ndonnx' has no operation 'nan_to_num'
.. [#fn-153] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanargmax'
.. [#fn-154] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanargmax'
.. [#fn-155] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanargmin'
.. [#fn-156] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanargmin'
.. [#fn-157] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nancumprod'
.. [#fn-158] AttributeError: saiunit: backend 'ndonnx' has no operation 'nancumprod'
.. [#fn-159] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nancumsum'
.. [#fn-160] AttributeError: saiunit: backend 'ndonnx' has no operation 'nancumsum'
.. [#fn-161] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanmax'
.. [#fn-162] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmax'
.. [#fn-163] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmean'
.. [#fn-164] NotImplementedError: The da.nanmedian function only works along an axis or a subset of axes. The full algorithm is difficult to do in parallel
.. [#fn-165] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmedian'
.. [#fn-166] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanmin'
.. [#fn-167] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanmin'
.. [#fn-168] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanpercentile'
.. [#fn-169] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanpercentile'
.. [#fn-170] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanprod'
.. [#fn-171] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanprod'
.. [#fn-172] TypeError: nanquantile() received an invalid combination of arguments - got (Tensor), but expected one of: \* (Tensor input, Tensor q, int dim = None, bool keepdim = False, \*, str interpolation = "l...
.. [#fn-173] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanquantile'
.. [#fn-174] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanstd'
.. [#fn-175] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanstd'
.. [#fn-176] AttributeError: saiunit: backend 'ndonnx' has no operation 'nansum'
.. [#fn-177] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'nanvar'
.. [#fn-178] AttributeError: saiunit: backend 'ndonnx' has no operation 'nanvar'
.. [#fn-179] AttributeError: saiunit: backend 'ndonnx' has no operation 'outer'
.. [#fn-180] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'percentile'
.. [#fn-181] AttributeError: saiunit: backend 'ndonnx' has no operation 'percentile'
.. [#fn-182] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'power'
.. [#fn-183] AttributeError: saiunit: backend 'ndonnx' has no operation 'power'
.. [#fn-184] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'ptp'
.. [#fn-185] AttributeError: saiunit: backend 'ndonnx' has no operation 'ptp'
.. [#fn-186] TypeError: quantile() received an invalid combination of arguments - got (Tensor), but expected one of: \* (Tensor input, Tensor q, int dim = None, bool keepdim = False, \*, str interpolation = "line...
.. [#fn-187] AttributeError: saiunit: backend 'ndonnx' has no operation 'quantile'
.. [#fn-188] AttributeError: saiunit: backend 'ndonnx' has no operation 'rad2deg'
.. [#fn-189] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'radians'
.. [#fn-190] AttributeError: saiunit: backend 'ndonnx' has no operation 'radians'
.. [#fn-191] AttributeError: saiunit: backend 'ndonnx' has no operation 'ravel'
.. [#fn-192] AttributeError: type object 'bool' has no attribute '__ndx_create__'
.. [#fn-193] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'right_shift'
.. [#fn-194] AttributeError: saiunit: backend 'ndonnx' has no operation 'right_shift'
.. [#fn-195] AttributeError: saiunit: backend 'array_api_compat.torch' has no operation 'rint'
.. [#fn-196] AttributeError: saiunit: backend 'ndonnx' has no operation 'rint'
.. [#fn-197] AttributeError: saiunit: backend 'ndonnx' has no operation 'rot90'
.. [#fn-198] AttributeError: saiunit: backend 'ndonnx' has no operation 'vstack'
.. [#fn-199] TypeError: select() received an invalid combination of arguments - got (list, list), but expected one of: \* (Tensor input, name dim, int index) \* (Tensor input, int dim, int index)
.. [#fn-200] AttributeError: saiunit: backend 'ndonnx' has no operation 'select'
.. [#fn-201] AttributeError: saiunit: backend 'ndonnx' has no operation 'sinc'
.. [#fn-202] TypeError: squeeze() missing 1 required positional argument: 'axis'
.. [#fn-203] AttributeError: saiunit: backend 'ndonnx' has no operation 'swapaxes'
.. [#fn-204] AttributeError: saiunit: backend 'ndonnx' has no operation 'trace'
.. [#fn-205] TypeError: transpose() received an invalid combination of arguments - got (Tensor), but expected one of: \* (Tensor input, int dim0, int dim1) \* (Tensor input, name dim0, name dim1)
.. [#fn-206] AttributeError: saiunit: backend 'ndonnx' has no operation 'transpose'
.. [#fn-207] AttributeError: module 'array_api_compat.torch' has no attribute 'tri'
.. [#fn-208] AttributeError: module 'ndonnx' has no attribute `tri`
.. [#fn-209] AttributeError: module 'array_api_compat.torch' has no attribute 'tril_indices_from'
.. [#fn-210] AttributeError: module 'ndonnx' has no attribute `tril_indices_from`
.. [#fn-211] AttributeError: module 'array_api_compat.torch' has no attribute 'triu_indices_from'
.. [#fn-212] AttributeError: module 'ndonnx' has no attribute `triu_indices_from`
.. [#fn-213] AttributeError: saiunit: backend 'ndonnx' has no operation 'true_divide'
.. [#fn-214] AttributeError: saiunit: backend 'ndonnx' has no operation 'unique'
.. [#fn-215] AttributeError: module 'array_api_compat.dask.array' has no attribute 'vander'
.. [#fn-216] AttributeError: module 'ndonnx' has no attribute `vander`
.. [#fn-217] AttributeError: saiunit: backend 'ndonnx' has no operation 'vdot'
.. [#fn-218] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'vsplit'
.. [#fn-219] AttributeError: saiunit: backend 'ndonnx' has no operation 'vsplit'
.. [#fn-220] TypeError: transpose() received an invalid combination of arguments - got (Tensor, list), but expected one of: \* (Tensor input, int dim0, int dim1) \* (Tensor input, name dim0, name dim1)
.. [#fn-221] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.cholesky'
.. [#fn-222] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.cond'
.. [#fn-223] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.cond'
.. [#fn-224] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.det'
.. [#fn-225] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.det'
.. [#fn-226] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to eig at position 0.
.. [#fn-227] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to eig at position 0.
.. [#fn-228] TypeError: Error interpreting argument to eig as a JAX value. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to eig at position 0.
.. [#fn-229] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to transpose at position 0.
.. [#fn-230] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to transpose at position 0.
.. [#fn-231] TypeError: Error interpreting argument to transpose as a JAX value. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to transpose at position 0.
.. [#fn-232] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.inv'
.. [#fn-233] TypeError: lstsq() got an unexpected keyword argument 'rcond'
.. [#fn-234] AttributeError: module 'ndonnx' has no attribute `linalg`
.. [#fn-235] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_norm'
.. [#fn-236] TypeError: svd() got an unexpected keyword argument 'compute_uv'
.. [#fn-237] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_rank'
.. [#fn-238] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.matrix_transpose'
.. [#fn-239] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.norm'
.. [#fn-240] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.pinv'
.. [#fn-241] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.pinv'
.. [#fn-242] AttributeError: module 'array_api_compat.dask.array.linalg' has no attribute 'slogdet'
.. [#fn-243] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.solve'
.. [#fn-244] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'torch.Tensor'> and was passed to svd at position 0.
.. [#fn-245] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'dask.array.core.Array'> and was passed to svd at position 0.
.. [#fn-246] TypeError: Error interpreting argument to svd as a JAX value. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to svd at position 0.
.. [#fn-247] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.tensorinv'
.. [#fn-248] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.tensorinv'
.. [#fn-249] AttributeError: saiunit: backend 'array_api_compat.dask.array' has no operation 'linalg.tensorsolve'
.. [#fn-250] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.tensorsolve'
.. [#fn-251] AttributeError: saiunit: backend 'ndonnx' has no operation 'linalg.vector_norm'
.. [#fn-252] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fft'
.. [#fn-253] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fft2'
.. [#fn-254] AttributeError: module 'ndonnx' has no attribute `fft`
.. [#fn-255] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fftn'
.. [#fn-256] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.fftshift'
.. [#fn-257] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifft'
.. [#fn-258] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifft2'
.. [#fn-259] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifftn'
.. [#fn-260] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.ifftshift'
.. [#fn-261] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfft'
.. [#fn-262] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfft2'
.. [#fn-263] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.irfftn'
.. [#fn-264] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfft'
.. [#fn-265] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfft2'
.. [#fn-266] AttributeError: saiunit: backend 'ndonnx' has no operation 'fft.rfftn'
.. [#fn-267] AttributeError: module 'array_api_compat.torch' has no attribute 'copy'
.. [#fn-268] AttributeError: module 'array_api_compat.dask.array' has no attribute 'copy'
.. [#fn-269] AttributeError: module 'ndonnx' has no attribute `copy`
.. [#fn-270] TypeError: Cannot interpret 'torch.float64' as a data type
.. [#fn-271] InvalidInputException: Argument 'dask.array<array, shape=(3,), dtype=float64, chunksize=(3,), chunktype=numpy.ndarray>' of type <class 'dask.array.core.Array'> is not a valid JAX type.
.. [#fn-272] TypeError: Cannot interpret 'Float64' as a data type
.. [#fn-273] TypeError: cross() got an unexpected keyword argument 'axisa'
.. [#fn-274] AttributeError: module 'array_api_compat.dask.array' has no attribute 'cross'
.. [#fn-275] AttributeError: module 'ndonnx' has no attribute `cross`
.. [#fn-276] RuntimeError: Unknown backend cuda. Available backends are ['cpu']
.. [#fn-277] TypeError: cumprod is not supported for quantities with units (has unit m), because each element of the result would have a different unit exponent. Use .prod() for a single reduction, or convert t...
.. [#fn-278] TypeError: cumsum() received an invalid combination of arguments - got (Tensor), but expected one of: \* (Tensor input, int dim, \*, torch.dtype dtype = None, Tensor out = None) \* (Tensor input, name...
.. [#fn-279] TypeError: diagonal() received an invalid combination of arguments - got (Tensor, axis1=int, axis2=int, offset=int), but expected one of: \* (Tensor input, \*, name outdim, name dim1, name dim2, int ...
.. [#fn-280] AttributeError: module 'ndonnx' has no attribute `diagonal`
.. [#fn-281] TypeError: Error interpreting argument to <function dot at 0x...> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at path a. This...
.. [#fn-282] ValueError: The dtype of the original data is float32, while we got float64.
.. [#fn-283] BackendError: Quantity.at indexed-update is not supported on the ndonnx backend. Call .to_numpy() (or another concrete backend) on the input first.
.. [#fn-284] AttributeError: module 'array_api_compat.dask.array' has no attribute 'float16'
.. [#fn-285] AttributeError: 'Array' object has no attribute 'item'
.. [#fn-286] TypeError: nancumprod is not supported for quantities with units (has unit m), because each element of the result would have a different unit exponent. Use .nanprod() for a single reduction, or con...
.. [#fn-287] TypeError: Error interpreting argument to <function outer at 0x...> as an abstract array. The problematic value is of type <class 'ndonnx._array.Array'> and was passed to the function at path a. Th...
.. [#fn-288] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'set'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-289] ValueError: found array with object dtype but it contains non-string elements
.. [#fn-290] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'add'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-291] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'divide'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy()...
.. [#fn-292] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'max'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-293] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'min'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy() to...
.. [#fn-294] NotImplementedError: Quantity.at on dask backend does not support index type 'Array' for op 'multiply'. Use a boolean mask of the source shape, a slice, an int, or a 1D int array; or call .to_numpy...
.. [#fn-295] TypeError: split() got an unexpected keyword argument 'axis'
.. [#fn-296] AxisError: axis1: axis 0 is out of bounds for array of dimension 0
.. [#fn-297] TypeError: 'NoneType' object is not iterable
.. [#fn-298] TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
.. [#fn-299] AttributeError: module 'ndonnx' has no attribute `swapaxes`
.. [#fn-300] TypeError: Axis value must be an integer, got None
.. [#fn-301] TypeError: tile(): argument 'dims' (position 2) must be tuple of ints, not int
.. [#fn-302] TypeError: object of type 'int' has no len()
.. [#fn-303] cupy backend not installed
.. [#fn-304] TypeError: Value 'array(data: [1.0, 2.0, 3.0], dtype=float64)' with dtype object is not a valid JAX array type. Only arrays of numeric types are supported by JAX.
.. [#fn-305] ValueError: unable to infer dtype from `[1. 2. 3.]`
.. [#fn-306] ValueError: unable to infer dtype from `tensor([1., 2., 3.], dtype=torch.float64)`
.. [#fn-307] ValueError: unable to infer dtype from `dask.array<array, shape=(3,), dtype=float64, chunksize=(3,), chunktype=numpy.ndarray>`
.. [#fn-308] TypeError: len() of unsized object
.. [#fn-309] ValueError: ONNX provides no control over the used device
.. [#fn-310] BackendError (expected): Quantity.tolist() would materialize a dask-backed Quantity. Call `q.mantissa.compute()` first.
.. [#fn-311] AttributeError: 'Array' object has no attribute 'tolist'
.. [#fn-312] TypeError: trace() got an unexpected keyword argument 'offset'
.. [#fn-313] AttributeError: module 'ndonnx' has no attribute `trace`
.. [#fn-314] AttributeError: module 'ndonnx' has no attribute `transpose`
.. [#fn-315] TypeError: view() received an invalid combination of arguments - got (type), but expected one of: \* (torch.dtype dtype) didn't match because some of the arguments have invalid types: (!type!) \* (tu...
.. [#fn-316] AttributeError: 'Array' object has no attribute 'view'
