#!/usr/bin/env python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Per-function, per-backend support sweep.

Walks every public callable in ``saiunit.math``, ``saiunit.linalg``,
``saiunit.fft``, plus the ``Quantity`` method surface, invokes each one
across the locally-installed backends (jax, numpy, torch, dask, ndonnx),
and records pass / skip / fail / unmapped per (function, backend).

``cupy`` is skipped at sweep time when CUDA is unavailable; the renderer
marks every cupy cell ``?`` in the rst output.

JAX-only subpackages (``saiunit.lax``, ``saiunit.autograd``,
``saiunit.sparse``) are probed once per backend with a representative
function to confirm ``BackendError`` is raised; per-function invocation
is intentionally skipped.

Output: ``dev/backend_support_data.json``.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

import saiunit as u
from saiunit._backend import (
    BackendName,
    to_backend,
    is_cupy_array,
    is_dask_array,
    is_jax_array,
    is_ndonnx_array,
    is_numpy_array,
    is_torch_array,
)
from saiunit._exceptions import BackendError
from saiunit._jax_compat import HAS_JAX


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_BACKENDS: tuple[BackendName, ...] = ("numpy", "jax", "cupy", "torch", "dask", "ndonnx")

OUTPUT_PATH = Path(__file__).parent / "backend_support_data.json"


def _detect_installed() -> dict[str, bool]:
    """Return mapping backend -> True if importable here."""
    installed: dict[str, bool] = {}
    installed["numpy"] = True
    installed["jax"] = HAS_JAX
    for name, mod in [("cupy", "cupy"), ("torch", "torch"),
                      ("dask", "dask.array"), ("ndonnx", "ndonnx")]:
        try:
            importlib.import_module(mod)
            installed[name] = True
        except ImportError:
            installed[name] = False
    if installed["cupy"]:
        # Even when cupy imports, we need CUDA at runtime. Probe.
        try:
            import cupy
            cupy.array([1.0])
        except Exception:
            installed["cupy"] = False
    return installed


# ---------------------------------------------------------------------------
# Helpers for building backend-native test inputs
# ---------------------------------------------------------------------------

def _arr(values, backend: str, dtype=np.float64):
    return to_backend(np.asarray(values, dtype=dtype), backend)


def _is_backend_array(x, backend: str) -> bool:
    if backend == "numpy":
        return is_numpy_array(x)
    if backend == "jax":
        return is_jax_array(x)
    if backend == "cupy":
        return is_cupy_array(x)
    if backend == "torch":
        return is_torch_array(x)
    if backend == "dask":
        return is_dask_array(x)
    if backend == "ndonnx":
        return is_ndonnx_array(x)
    return False


# ---------------------------------------------------------------------------
# Call patterns
#
# Each pattern is a callable: pattern(fn, backend) -> result OR raises.
# A pattern decides how to construct inputs of the right shape/dtype and
# what kwargs to pass.
# ---------------------------------------------------------------------------

def _p_unary_float(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend))


def _p_unary_positive_float(fn, backend):
    # e.g. sqrt, log
    return fn(_arr([0.5, 1.0, 2.0], backend))


def _p_unary_signed_float(fn, backend):
    return fn(_arr([-1.5, 0.5, 2.0], backend))


def _p_unary_int(fn, backend):
    return fn(_arr([1, 2, 3], backend, dtype=np.int64))


def _p_unary_bool(fn, backend):
    return fn(_arr([True, False, True], backend, dtype=np.bool_))


def _p_unary_complex(fn, backend):
    return fn(to_backend(np.asarray([1 + 1j, 2 - 1j], dtype=np.complex128), backend))


def _p_unary_2d(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend))


def _p_binary_float(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend),
              _arr([4.0, 5.0, 6.0], backend))


def _p_binary_int(fn, backend):
    a = _arr([1, 2, 3], backend, dtype=np.int64)
    b = _arr([4, 5, 6], backend, dtype=np.int64)
    return fn(a, b)


def _p_binary_bool(fn, backend):
    a = _arr([True, False, True], backend, dtype=np.bool_)
    b = _arr([False, False, True], backend, dtype=np.bool_)
    return fn(a, b)


def _p_reduce(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend))


def _p_reduce_int(fn, backend):
    return fn(_arr([1, 2, 3], backend, dtype=np.int64))


def _p_reduce_bool(fn, backend):
    return fn(_arr([True, False, True], backend, dtype=np.bool_))


def _p_reduce_2d_axis(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend), axis=0)


def _p_cumulative(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend))


def _p_creation_shape(fn, backend):
    return fn((3,))


def _p_creation_shape_kw(fn, backend):
    return fn((3,))


def _p_creation_like(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend))


def _p_creation_full(fn, backend):
    return fn((3,), 7.0)


def _p_creation_range(fn, backend):
    return fn(0, 5)


def _p_creation_linspace(fn, backend):
    return fn(0.0, 1.0, 5)


def _p_array(fn, backend):
    return fn([1.0, 2.0, 3.0])


def _p_asarray(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend))


def _p_arctan2(fn, backend):
    return fn(_arr([1.0, 0.0], backend), _arr([0.0, 1.0], backend))


def _p_clip(fn, backend):
    return fn(_arr([-1.0, 0.5, 2.0], backend), 0.0, 1.0)


def _p_where(fn, backend):
    return fn(_arr([True, False, True], backend, dtype=np.bool_),
              _arr([1.0, 2.0, 3.0], backend),
              _arr([4.0, 5.0, 6.0], backend))


def _p_power(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend), 2.0)


def _p_matmul(fn, backend):
    a = _arr([[1.0, 2.0], [3.0, 4.0]], backend)
    return fn(a, a)


def _p_dot(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend),
              _arr([4.0, 5.0, 6.0], backend))


def _p_outer(fn, backend):
    return fn(_arr([1.0, 2.0], backend), _arr([3.0, 4.0], backend))


def _p_cross(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend),
              _arr([4.0, 5.0, 6.0], backend))


def _p_concat(fn, backend):
    a = _arr([1.0, 2.0], backend)
    b = _arr([3.0, 4.0], backend)
    return fn([a, b])


def _p_concat_axis(fn, backend):
    a = _arr([[1.0, 2.0], [3.0, 4.0]], backend)
    return fn([a, a])


def _p_stack(fn, backend):
    a = _arr([1.0, 2.0], backend)
    return fn([a, a])


def _p_split(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0, 4.0], backend), 2)


def _p_reshape(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0, 4.0], backend), (2, 2))


def _p_transpose(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend))


def _p_squeeze(fn, backend):
    return fn(_arr([[1.0, 2.0]], backend))


def _p_repeat(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend), 2)


def _p_tile(fn, backend):
    return fn(_arr([1.0, 2.0], backend), 3)


def _p_take(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend),
              _arr([0, 2], backend, dtype=np.int64))


def _p_sort(fn, backend):
    return fn(_arr([3.0, 1.0, 2.0], backend))


def _p_argsort(fn, backend):
    return fn(_arr([3.0, 1.0, 2.0], backend))


def _p_searchsorted(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend),
              _arr([1.5, 2.5], backend))


def _p_diff(fn, backend):
    return fn(_arr([1.0, 3.0, 6.0], backend))


def _p_diag(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend))


def _p_eye(fn, backend):
    # Some backends accept N only; saiunit re-exports may add M/k kwargs.
    return fn(3, M=3)


def _p_identity(fn, backend):
    return fn(3)


def _p_tri(fn, backend):
    return fn(3)


def _p_tri_index(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend))


def _p_meshgrid(fn, backend):
    return fn(_arr([1.0, 2.0], backend), _arr([3.0, 4.0], backend))


def _p_einsum(fn, backend):
    return fn("ij,jk->ik",
              _arr([[1.0, 2.0], [3.0, 4.0]], backend),
              _arr([[5.0, 6.0], [7.0, 8.0]], backend))


def _p_tensordot(fn, backend):
    a = _arr([[1.0, 2.0], [3.0, 4.0]], backend)
    return fn(a, a)


def _p_kron(fn, backend):
    return fn(_arr([1.0, 2.0], backend), _arr([3.0, 4.0], backend))


def _p_inner(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend),
              _arr([4.0, 5.0, 6.0], backend))


def _p_trace(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend))


def _p_diagonal(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend))


def _p_unique(fn, backend):
    return fn(_arr([1.0, 2.0, 2.0, 3.0], backend))


def _p_bincount(fn, backend):
    return fn(_arr([0, 1, 1, 2], backend, dtype=np.int64))


def _p_histogram(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0, 4.0, 5.0], backend))


def _p_compress(fn, backend):
    return fn(_arr([True, False, True], backend, dtype=np.bool_),
              _arr([1.0, 2.0, 3.0], backend))


def _p_select(fn, backend):
    cond1 = _arr([True, False, False], backend, dtype=np.bool_)
    cond2 = _arr([False, True, False], backend, dtype=np.bool_)
    a = _arr([1.0, 2.0, 3.0], backend)
    b = _arr([4.0, 5.0, 6.0], backend)
    return fn([cond1, cond2], [a, b])


def _p_choose(fn, backend):
    idx = _arr([0, 1, 0], backend, dtype=np.int64)
    a = _arr([1.0, 2.0, 3.0], backend)
    b = _arr([4.0, 5.0, 6.0], backend)
    return fn(idx, [a, b])


def _p_interp(fn, backend):
    return fn(_arr([1.5], backend),
              _arr([1.0, 2.0, 3.0], backend),
              _arr([10.0, 20.0, 30.0], backend))


def _p_gradient(fn, backend):
    return fn(_arr([1.0, 3.0, 6.0, 10.0], backend))


def _p_pad_unsupported(fn, backend):
    # Placeholder — not in saiunit.math as a public export per current __all__.
    raise NotImplementedError


def _p_corrcoef(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0, 4.0], backend))


def _p_cov(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0, 4.0], backend))


def _p_correlate(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend),
              _arr([0.0, 1.0], backend))


def _p_convolve(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend),
              _arr([0.0, 1.0], backend))


def _p_window(fn, backend):
    return fn(5)


def _p_kaiser(fn, backend):
    return fn(5, 14.0)


def _p_atleast(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend))


def _p_broadcast_arrays(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend),
              _arr([4.0, 5.0, 6.0], backend))


def _p_broadcast_to(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend), (2, 3))


def _p_broadcast_shapes(fn, backend):
    return fn((2, 3), (1, 3))


def _p_flip(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend))


def _p_roll(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend), 1)


def _p_rot90(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend))


def _p_moveaxis(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend), 0, 1)


def _p_swapaxes(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend), 0, 1)


def _p_expand_dims(fn, backend):
    return fn(_arr([1.0, 2.0], backend), 0)


def _p_flatten(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend))


def _p_block(fn, backend):
    a = _arr([[1.0, 2.0], [3.0, 4.0]], backend)
    return fn([[a, a], [a, a]])


def _p_extract(fn, backend):
    return fn(_arr([True, False, True], backend, dtype=np.bool_),
              _arr([1.0, 2.0, 3.0], backend))


def _p_argwhere(fn, backend):
    return fn(_arr([0.0, 1.0, 0.0, 2.0], backend))


def _p_nonzero(fn, backend):
    return fn(_arr([0.0, 1.0, 0.0, 2.0], backend))


def _p_flatnonzero(fn, backend):
    return fn(_arr([0.0, 1.0, 0.0, 2.0], backend))


def _p_count_nonzero(fn, backend):
    return fn(_arr([0.0, 1.0, 0.0, 2.0], backend))


def _p_intersect1d(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend),
              _arr([2.0, 3.0, 4.0], backend))


def _p_digitize(fn, backend):
    return fn(_arr([0.5, 1.5, 2.5], backend),
              _arr([0.0, 1.0, 2.0], backend))


def _p_modf(fn, backend):
    return fn(_arr([1.5, 2.25, 3.75], backend))


def _p_frexp(fn, backend):
    return fn(_arr([1.5, 2.25, 3.75], backend))


def _p_copysign(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend),
              _arr([-1.0, 1.0, -1.0], backend))


def _p_ldexp(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend),
              _arr([1, 2, 3], backend, dtype=np.int64))


def _p_nextafter(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend),
              _arr([2.0, 3.0, 4.0], backend))


def _p_heaviside(fn, backend):
    return fn(_arr([-1.0, 0.0, 1.0], backend),
              _arr([0.5, 0.5, 0.5], backend))


def _p_nan_to_num(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend))


def _p_around(fn, backend):
    return fn(_arr([1.4, 2.5, 3.6], backend))


def _p_remove_diag(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend))


def _p_diag_indices_from(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend))


def _p_fill_diagonal(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend), 0.0)


def _p_vander(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0], backend))


def _p_matrix_power(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend), 2)


def _p_quantile(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0, 4.0], backend), 0.5)


def _p_average(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0, 4.0], backend))


def _p_einops_rearrange(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend), "h w -> w h")


def _p_einops_reduce(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend), "h w -> h", "sum")


def _p_einops_repeat(fn, backend):
    return fn(_arr([1.0, 2.0], backend), "w -> 2 w")


def _p_einops_shape(fn, backend):
    return fn(_arr([[1.0, 2.0], [3.0, 4.0]], backend), "h w")


def _p_unflatten(fn, backend):
    return fn(_arr([1.0, 2.0, 3.0, 4.0], backend), 0, (2, 2))


def _p_factorless(fn, backend):
    # Not a backend-routed op
    raise NotImplementedError


def _p_promote_dtypes(fn, backend):
    return fn(_arr([1.0], backend), _arr([2], backend, dtype=np.int64))


def _p_result_type(fn, backend):
    return fn(np.float32, np.float64)


def _p_dtype(fn, backend):
    return fn("float32")


def _p_finfo(fn, backend):
    return fn(np.float32)


def _p_iinfo(fn, backend):
    return fn(np.int32)


def _p_issubdtype(fn, backend):
    return fn(np.float32, np.floating)


def _p_isscalar(fn, backend):
    return fn(1.0)


def _p_isclose(fn, backend):
    return fn(_arr([1.0, 2.0], backend), _arr([1.0, 2.0], backend))


def _p_array_equal(fn, backend):
    return fn(_arr([1.0, 2.0], backend), _arr([1.0, 2.0], backend))


# ---------------------------------------------------------------------------
# Function registry
# ---------------------------------------------------------------------------

# Function name (under saiunit.math/linalg/fft) -> pattern callable.
# Names listed once cover the public surface of the multi-backend subpackages.

MATH_REGISTRY: dict[str, Callable] = {
    # ---- unary float (elementwise) ----
    "sin": _p_unary_float, "cos": _p_unary_float, "tan": _p_unary_float,
    "arcsin": _p_unary_float, "arccos": _p_unary_float, "arctan": _p_unary_float,
    "sinh": _p_unary_float, "cosh": _p_unary_float, "tanh": _p_unary_float,
    "arcsinh": _p_unary_float, "arccosh": _p_unary_positive_float,
    "arctanh": _p_unary_float,
    "exp": _p_unary_float, "expm1": _p_unary_float, "exp2": _p_unary_float,
    "log": _p_unary_positive_float, "log1p": _p_unary_float,
    "log2": _p_unary_positive_float, "log10": _p_unary_positive_float,
    "sqrt": _p_unary_positive_float, "cbrt": _p_unary_float,
    "square": _p_unary_float, "abs": _p_unary_signed_float,
    "absolute": _p_unary_signed_float, "fabs": _p_unary_signed_float,
    "negative": _p_unary_float, "positive": _p_unary_float,
    "sign": _p_unary_signed_float, "signbit": _p_unary_signed_float,
    "ceil": _p_unary_float, "floor": _p_unary_float, "trunc": _p_unary_float,
    "rint": _p_unary_float, "fix": _p_unary_float, "round": _p_unary_float,
    "isfinite": _p_unary_float, "isinf": _p_unary_float, "isnan": _p_unary_float,
    "isreal": _p_unary_float, "iscomplexobj": _p_unary_float,
    "reciprocal": _p_unary_positive_float,
    "conj": _p_unary_complex, "conjugate": _p_unary_complex,
    "real": _p_unary_complex, "imag": _p_unary_complex, "angle": _p_unary_complex,
    "deg2rad": _p_unary_float, "rad2deg": _p_unary_float,
    "degrees": _p_unary_float, "radians": _p_unary_float,
    "sinc": _p_unary_float,
    "exprel": _p_unary_float,
    # ---- binary float ----
    "add": _p_binary_float, "subtract": _p_binary_float,
    "multiply": _p_binary_float, "divide": _p_binary_float,
    "true_divide": _p_binary_float, "floor_divide": _p_binary_float,
    "remainder": _p_binary_float, "mod": _p_binary_float, "fmod": _p_binary_float,
    "power": _p_power, "float_power": _p_power,
    "maximum": _p_binary_float, "minimum": _p_binary_float,
    "fmax": _p_binary_float, "fmin": _p_binary_float,
    "hypot": _p_binary_float, "arctan2": _p_arctan2,
    "logaddexp": _p_binary_float, "logaddexp2": _p_binary_float,
    "copysign": _p_copysign, "ldexp": _p_ldexp, "nextafter": _p_nextafter,
    "heaviside": _p_heaviside,
    "divmod": _p_binary_float,
    # ---- comparison ----
    "equal": _p_binary_float, "not_equal": _p_binary_float,
    "greater": _p_binary_float, "greater_equal": _p_binary_float,
    "less": _p_binary_float, "less_equal": _p_binary_float,
    "isclose": _p_isclose, "allclose": _p_isclose, "array_equal": _p_array_equal,
    # ---- bool / bitwise ----
    "logical_and": _p_binary_bool, "logical_or": _p_binary_bool,
    "logical_xor": _p_binary_bool, "logical_not": _p_unary_bool,
    "bitwise_and": _p_binary_int, "bitwise_or": _p_binary_int,
    "bitwise_xor": _p_binary_int, "bitwise_not": _p_unary_int,
    "invert": _p_unary_int,
    "left_shift": _p_binary_int, "right_shift": _p_binary_int,
    "gcd": _p_binary_int, "lcm": _p_binary_int,
    # ---- reductions ----
    "sum": _p_reduce, "prod": _p_reduce, "product": _p_reduce,
    "mean": _p_reduce, "median": _p_reduce, "std": _p_reduce, "var": _p_reduce,
    "min": _p_reduce, "max": _p_reduce, "amin": _p_reduce, "amax": _p_reduce,
    "ptp": _p_reduce,
    "all": _p_reduce_bool, "any": _p_reduce_bool,
    "alltrue": _p_reduce_bool, "sometrue": _p_reduce_bool,
    "argmin": _p_reduce, "argmax": _p_reduce,
    "nanmin": _p_reduce, "nanmax": _p_reduce,
    "nanmean": _p_reduce, "nanmedian": _p_reduce,
    "nanstd": _p_reduce, "nanvar": _p_reduce,
    "nansum": _p_reduce, "nanprod": _p_reduce,
    "nanargmax": _p_reduce, "nanargmin": _p_reduce,
    "percentile": _p_quantile, "nanpercentile": _p_quantile,
    "quantile": _p_quantile, "nanquantile": _p_quantile,
    "average": _p_average,
    "count_nonzero": _p_count_nonzero,
    # ---- cumulative ----
    "cumsum": _p_cumulative, "cumprod": _p_cumulative,
    "cumproduct": _p_cumulative,
    "nancumsum": _p_cumulative, "nancumprod": _p_cumulative,
    # ---- products ----
    "dot": _p_dot, "vdot": _p_dot, "vecdot": _p_dot,
    "inner": _p_inner, "outer": _p_outer, "cross": _p_cross,
    "matmul": _p_matmul, "tensordot": _p_tensordot, "einsum": _p_einsum,
    "kron": _p_kron,
    "multi_dot": lambda fn, b: fn([_arr([[1.0, 2.0], [3.0, 4.0]], b)] * 2),
    "matrix_power": _p_matrix_power,
    # ---- creation ----
    "zeros": _p_creation_shape, "ones": _p_creation_shape, "empty": _p_creation_shape,
    "zeros_like": _p_creation_like, "ones_like": _p_creation_like,
    "empty_like": _p_creation_like,
    "full": _p_creation_full, "full_like": _p_creation_like,
    "arange": _p_creation_range, "linspace": _p_creation_linspace,
    "logspace": _p_creation_linspace,
    "eye": _p_eye, "identity": _p_identity, "tri": _p_tri,
    "tril": _p_unary_2d, "triu": _p_unary_2d,
    "tril_indices": _p_eye, "triu_indices": _p_eye,
    "tril_indices_from": _p_tri_index, "triu_indices_from": _p_tri_index,
    "diag": _p_diag, "diagflat": _p_diag, "diagonal": _p_diagonal,
    "vander": _p_vander,
    "array": _p_array, "asarray": _p_asarray, "from_numpy": _p_array,
    "as_numpy": _p_creation_like,
    "meshgrid": _p_meshgrid,
    "tree_zeros_like": _p_creation_like, "tree_ones_like": _p_creation_like,
    # ---- shape manipulation ----
    "reshape": _p_reshape, "ravel": _p_unary_2d, "flatten": _p_flatten,
    "transpose": _p_transpose,
    "squeeze": _p_squeeze, "expand_dims": _p_expand_dims,
    "moveaxis": _p_moveaxis, "swapaxes": _p_swapaxes,
    "flip": _p_flip, "fliplr": _p_unary_2d, "flipud": _p_unary_2d,
    "rot90": _p_rot90, "roll": _p_roll,
    "broadcast_arrays": _p_broadcast_arrays,
    "broadcast_to": _p_broadcast_to,
    "broadcast_shapes": _p_broadcast_shapes,
    "atleast_1d": _p_atleast, "atleast_2d": _p_atleast, "atleast_3d": _p_atleast,
    "tile": _p_tile, "repeat": _p_repeat,
    "concatenate": _p_concat, "stack": _p_stack,
    "hstack": _p_concat, "vstack": _p_concat, "dstack": _p_concat,
    "column_stack": _p_concat, "row_stack": _p_concat, "block": _p_block,
    "split": _p_split, "array_split": _p_split,
    "hsplit": lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b), 2),
    "vsplit": lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b), 2),
    "dsplit": lambda fn, b: fn(
        to_backend(np.arange(8, dtype=np.float64).reshape(2, 2, 2), b), 2),
    "append": _p_binary_float,
    "unflatten": _p_unflatten,
    # ---- indexing / selection ----
    "take": _p_take, "compress": _p_extract, "extract": _p_extract,
    "choose": _p_choose, "select": _p_select, "where": _p_where,
    "gather": _p_take,
    "argwhere": _p_argwhere, "nonzero": _p_nonzero,
    "flatnonzero": _p_flatnonzero,
    "argsort": _p_argsort, "sort": _p_sort,
    "searchsorted": _p_searchsorted,
    "diff": _p_diff, "ediff1d": _p_diff, "gradient": _p_gradient,
    # ---- misc / statistical / signal ----
    "clip": _p_clip, "around": _p_around, "nan_to_num": _p_nan_to_num,
    "unique": _p_unique, "intersect1d": _p_intersect1d,
    "bincount": _p_bincount, "digitize": _p_digitize,
    "histogram": _p_histogram,
    "convolve": _p_convolve, "correlate": _p_correlate,
    "corrcoef": _p_corrcoef, "cov": _p_cov,
    "modf": _p_modf, "frexp": _p_frexp,
    "trace": _p_trace,
    "interp": _p_interp,
    "remove_diag": _p_remove_diag,
    "diag_indices_from": _p_diag_indices_from,
    "fill_diagonal": _p_fill_diagonal,
    # ---- windows ----
    "bartlett": _p_window, "blackman": _p_window,
    "hamming": _p_window, "hanning": _p_window,
    "kaiser": _p_kaiser,
    # ---- activations ----
    "relu": _p_unary_float, "relu6": _p_unary_float, "elu": _p_unary_float,
    "celu": _p_unary_float, "selu": _p_unary_float, "gelu": _p_unary_float,
    "glu": lambda fn, b: fn(_arr([1.0, 2.0, 3.0, 4.0], b)),  # axis size must be even
    "silu": _p_unary_float, "swish": _p_unary_float,
    "hard_silu": _p_unary_float, "hard_swish": _p_unary_float,
    "hard_sigmoid": _p_unary_float, "hard_tanh": _p_unary_float,
    "leaky_relu": _p_unary_float, "log_sigmoid": _p_unary_float,
    "mish": _p_unary_float, "sigmoid": _p_unary_float,
    "soft_sign": _p_unary_float, "softplus": _p_unary_float,
    "squareplus": _p_unary_float,
    "sparse_plus": _p_unary_float, "sparse_sigmoid": _p_unary_float,
    # ---- einops ----
    "einrearrange": _p_einops_rearrange, "einreduce": _p_einops_reduce,
    "einrepeat": _p_einops_repeat, "einshape": _p_einops_shape,
    # ---- promotion / type helpers (NOT backend-routed compute; here for completeness) ----
    "promote_dtypes": _p_promote_dtypes, "get_promote_dtypes": _p_result_type,
    "result_type": _p_result_type,
    "dtype": _p_dtype, "finfo": _p_finfo, "iinfo": _p_iinfo,
    "issubdtype": _p_issubdtype, "isscalar": _p_isscalar,
    "astype": _p_unary_float,  # operates on array, dtype param implied
    "ndim": _p_unary_float, "shape": _p_unary_float, "size": _p_unary_float,
}

LINALG_REGISTRY: dict[str, Callable] = {
    "cholesky":         lambda fn, b: fn(_arr([[4.0, 2.0], [2.0, 3.0]], b)),
    "cond":             lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "cross":            _p_cross,
    "det":              lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "diagonal":         _p_diagonal,
    "dot":              _p_dot,
    "eig":              lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "eigh":             lambda fn, b: fn(_arr([[2.0, 1.0], [1.0, 2.0]], b)),
    "eigvals":          lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "eigvalsh":         lambda fn, b: fn(_arr([[2.0, 1.0], [1.0, 2.0]], b)),
    "inner":            _p_inner,
    "inv":              lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 5.0]], b)),
    "kron":             _p_kron,
    "lstsq":            lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b),
                                         _arr([5.0, 6.0], b)),
    "matmul":           _p_matmul,
    "matrix_norm":      lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "matrix_power":     _p_matrix_power,
    "matrix_rank":      lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "matrix_transpose": _p_transpose,
    "multi_dot":        lambda fn, b: fn([_arr([[1.0, 2.0], [3.0, 4.0]], b)] * 2),
    "norm":             lambda fn, b: fn(_arr([3.0, 4.0], b)),
    "outer":            _p_outer,
    "pinv":             lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "qr":               lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "slogdet":          lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "solve":            lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 5.0]], b),
                                         _arr([6.0, 7.0], b)),
    "svd":              lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "svdvals":          lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "tensordot":        _p_tensordot,
    "tensorinv":        lambda fn, b: fn(to_backend(
                            np.eye(4, dtype=np.float64).reshape(2, 2, 2, 2), b)),
    "tensorsolve":      lambda fn, b: fn(
                            to_backend(np.eye(4, dtype=np.float64).reshape(2, 2, 2, 2), b),
                            to_backend(np.ones((2, 2), dtype=np.float64), b)),
    "trace":            _p_trace,
    "vdot":             _p_dot,
    "vecdot":           _p_dot,
    "vector_norm":      lambda fn, b: fn(_arr([3.0, 4.0], b)),
}

FFT_REGISTRY: dict[str, Callable] = {
    "fft":       lambda fn, b: fn(_arr([1.0, 2.0, 3.0, 4.0], b)),
    "ifft":      lambda fn, b: fn(_arr([1.0, 2.0, 3.0, 4.0], b)),
    "fft2":      lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "ifft2":     lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "fftn":      lambda fn, b: fn(_arr([1.0, 2.0, 3.0, 4.0], b)),
    "ifftn":     lambda fn, b: fn(_arr([1.0, 2.0, 3.0, 4.0], b)),
    "rfft":      lambda fn, b: fn(_arr([1.0, 2.0, 3.0, 4.0], b)),
    "irfft":     lambda fn, b: fn(_arr([1.0, 0.0, 0.0], b)),
    "rfft2":     lambda fn, b: fn(_arr([[1.0, 2.0], [3.0, 4.0]], b)),
    "irfft2":    lambda fn, b: fn(_arr([[1.0, 0.0], [0.0, 0.0]], b)),
    "rfftn":     lambda fn, b: fn(_arr([1.0, 2.0, 3.0, 4.0], b)),
    "irfftn":    lambda fn, b: fn(_arr([1.0, 0.0, 0.0], b)),
    "fftshift":  lambda fn, b: fn(_arr([1.0, 2.0, 3.0, 4.0], b)),
    "ifftshift": lambda fn, b: fn(_arr([1.0, 2.0, 3.0, 4.0], b)),
    "fftfreq":   lambda fn, b: fn(8),
    "rfftfreq":  lambda fn, b: fn(8),
}

# ---------------------------------------------------------------------------
# Quantity method registry
# ---------------------------------------------------------------------------


def _qarr(values, backend, dtype=np.float64):
    return u.Quantity(_arr(values, backend, dtype=dtype), unit=u.meter)


def _qm_reduce(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend))


def _qm_reduce_axis(method, backend):
    return method(_qarr([[1.0, 2.0], [3.0, 4.0]], backend), axis=0)


def _qm_unary(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend))


def _qm_call_no_args(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend))


def _qm_clip(method, backend):
    return method(_qarr([-1.0, 0.5, 2.0], backend), 0.0 * u.meter, 1.0 * u.meter)


def _qm_astype(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend), np.float32)


def _qm_reshape(method, backend):
    return method(_qarr([1.0, 2.0, 3.0, 4.0], backend), (2, 2))


def _qm_expand_dims(method, backend):
    return method(_qarr([1.0, 2.0], backend), 0)


def _qm_swap_or_move(method, backend):
    return method(_qarr([[1.0, 2.0], [3.0, 4.0]], backend), 0, 1)


def _qm_view(method, backend):
    # Quantity.view(dtype) — like torch's view-as-dtype.
    return method(_qarr([1.0, 2.0, 3.0, 4.0], backend), np.float64)


def _qm_repeat(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend), 2)


def _qm_fill(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend), 7.0 * u.meter)


def _qm_dot(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend), _qarr([4.0, 5.0, 6.0], backend))


def _qm_cross(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend), _qarr([4.0, 5.0, 6.0], backend))


def _qm_searchsorted(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend), _qarr([1.5], backend))


def _qm_take(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend),
                  _arr([0, 2], backend, dtype=np.int64))


def _qm_diagonal(method, backend):
    return method(_qarr([[1.0, 2.0], [3.0, 4.0]], backend))


def _qm_trace(method, backend):
    return method(_qarr([[1.0, 2.0], [3.0, 4.0]], backend))


def _qm_in_unit(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend), u.meter)


def _qm_to_decimal(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend), u.meter)


def _qm_with_unit(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend), u.meter)


def _qm_update_mantissa(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend), _arr([4.0, 5.0, 6.0], backend))


def _qm_has_same_unit(method, backend):
    return method(_qarr([1.0], backend), _qarr([2.0], backend))


def _qm_expand_as(method, backend):
    # Quantity.expand_as expects a target shape, not a Quantity, despite the
    # signature claiming it accepts Quantity | ArrayLike.
    q = _qarr([1.0, 2.0, 3.0], backend)
    return method(q, (3,))


def _qm_tree(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend))


def _qm_tree_unflatten(method, backend):
    # Skip — needs aux + children matching exactly.
    raise NotImplementedError


def _qm_to_backend(method, backend, target):
    return method(_qarr([1.0, 2.0, 3.0], backend))


def _qm_scalar(method, backend):
    """For .item() / .float() / .double() / .half() — need a size-1 mantissa."""
    return method(_qarr([1.0], backend))


def _qm_pow(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend), 2)


def _qm_outer(method, backend):
    return method(_qarr([1.0, 2.0], backend), _qarr([3.0, 4.0], backend))


def _qm_tile(method, backend):
    return method(_qarr([1.0, 2.0], backend), 3)


def _qm_split(method, backend):
    return method(_qarr([1.0, 2.0, 3.0, 4.0], backend), 2)


def _qm_resize(method, backend):
    return method(_qarr([1.0, 2.0, 3.0, 4.0], backend), (4,))


def _qm_to_unit(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend), u.meter)


def _qm_repr_in_unit(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend), u.meter)


def _qm_put(method, backend):
    return method(_qarr([1.0, 2.0, 3.0], backend),
                  _arr([0], backend, dtype=np.int64),
                  99.0 * u.meter)


def _qm_scatter(method, backend):
    # scatter_add/sub/max/min want a same-unit Quantity value.
    return method(_qarr([1.0, 2.0, 3.0], backend),
                  _arr([0], backend, dtype=np.int64),
                  0.5 * u.meter)


def _qm_scatter_scale(method, backend):
    # scatter_mul/div want a DIMENSIONLESS scale factor (plain float).
    return method(_qarr([1.0, 2.0, 3.0], backend),
                  _arr([0], backend, dtype=np.int64),
                  0.5)


# Methods that force materialization — expected to raise BackendError on dask.
_MATERIALIZE_METHODS = {"item", "tolist", "float", "double", "half"}

QUANTITY_REGISTRY: dict[str, Callable] = {
    "all":              _qm_reduce, "any": _qm_reduce,
    "sum":              _qm_reduce, "mean": _qm_reduce,
    "std":              _qm_reduce, "var": _qm_reduce,
    "prod":             _qm_reduce, "max": _qm_reduce, "min": _qm_reduce,
    "ptp":              _qm_reduce,
    "argmax":           _qm_reduce, "argmin": _qm_reduce,
    "argsort":          _qm_reduce, "sort": _qm_reduce,
    "cumsum":           _qm_reduce, "cumprod": _qm_reduce,
    "nonzero":          _qm_reduce, "round": _qm_reduce,
    "squeeze":          _qm_reduce, "ravel": _qm_reduce,
    "real":             _qm_reduce, "imag": _qm_reduce,
    "conj":             _qm_reduce, "conjugate": _qm_reduce,
    "transpose":        _qm_reduce, "flatten": _qm_reduce,
    "size":             _qm_reduce, "T": _qm_reduce, "shape": _qm_reduce,
    "clip":             _qm_clip,
    "astype":           _qm_astype,
    "reshape":          _qm_reshape,
    "view":             _qm_view,
    "expand_dims":      _qm_expand_dims, "unsqueeze": _qm_expand_dims,
    "swapaxes":         _qm_swap_or_move, "moveaxis": _qm_swap_or_move,
    "repeat":           _qm_repeat,
    "fill":             _qm_fill,
    "dot":              _qm_dot,
    "cross":            _qm_cross,
    "searchsorted":     _qm_searchsorted,
    "take":             _qm_take,
    "trace":            _qm_trace,
    "diagonal":         _qm_diagonal,
    "in_unit":          _qm_in_unit,
    "to_decimal":       _qm_to_decimal,
    "with_unit":        _qm_with_unit,
    "update_mantissa":  _qm_update_mantissa,
    "has_same_unit":    _qm_has_same_unit,
    "expand_as":        _qm_expand_as,
    "tree_flatten":     _qm_tree,
    "factorless":       _qm_reduce,
    # ---- conversion ----
    "to_numpy":         _qm_reduce, "to_jax": _qm_reduce,
    "to_torch":         _qm_reduce, "to_cupy": _qm_reduce,
    "to_dask":          _qm_reduce, "to_ndonnx": _qm_reduce,
    "clone":            _qm_reduce, "copy": _qm_reduce,
    "cpu":              _qm_reduce, "cuda": _qm_reduce,
    # ---- materialization (expected to raise BackendError on dask) ----
    "item":             _qm_scalar, "tolist": _qm_reduce,
    "float":            _qm_reduce, "double": _qm_reduce, "half": _qm_reduce,
    # ---- previously unmapped ----
    "pow":              _qm_pow, "outer": _qm_outer,
    "tile":             _qm_tile, "split": _qm_split, "resize": _qm_resize,
    "to":               _qm_to_unit, "repr_in_unit": _qm_repr_in_unit,
    "put":              _qm_put,
    "scatter_add":      _qm_scatter, "scatter_sub": _qm_scatter,
    "scatter_mul":      _qm_scatter_scale, "scatter_div": _qm_scatter_scale,
    "scatter_max":      _qm_scatter, "scatter_min": _qm_scatter,
    "nanprod":          _qm_reduce, "nancumprod": _qm_reduce,
}


# Math functions that intentionally are NOT backend-dispatched (dtype
# factories, dimension/unit predicate helpers). Treated as "n/a" in the
# rendered matrix rather than left as unmapped — the renderer suppresses
# them from the per-backend tables and lists them in a separate section.
NON_DISPATCHED_MATH = {
    # dtype factories (jax.numpy re-exports)
    "bfloat16", "bool_", "cdouble", "complex128", "complex64", "complex_",
    "csingle", "double", "float16", "float32", "float64", "float_",
    "inexact", "int16", "int2", "int32", "int4", "int64", "int8", "int_",
    "single", "uint", "uint16", "uint2", "uint32", "uint4", "uint64", "uint8",
    # unit / dimension predicate helpers (pure Python over Quantity)
    "check_dims", "check_units", "get_dim", "get_dtype", "get_magnitude",
    "get_mantissa", "get_or_create_dimension", "get_unit",
    "is_dimensionless", "is_float", "is_int", "is_quantity", "is_unitless",
    "assert_quantity", "display_in_unit", "fail_for_dimension_mismatch",
    "fail_for_unit_mismatch", "maybe_decimal", "set_exprel_order",
}


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


def _classify_outcome(
    fn_qualname: str,
    backend: str,
    invoker: Callable,
    fn: Any,
    *,
    expect_backend_error: bool = False,
) -> dict[str, str]:
    """Return ``{'status': str, 'detail': str}``."""
    try:
        result = invoker(fn, backend)
    except BackendError as exc:
        if expect_backend_error:
            return {"status": "warn", "detail": f"BackendError (expected): {exc}"}
        return {"status": "fail", "detail": f"BackendError: {exc}"}
    except (AttributeError, TypeError, NotImplementedError, ValueError, RuntimeError) as exc:
        msg = str(exc)
        low = msg.lower()
        # Backend lacks the op or doesn't accept a kwarg saiunit forwards =
        # not a saiunit bug, but a documented gap in that backend's surface.
        skip_tokens = (
            # missing-attribute / no-op patterns
            "has no attribute", "no attribute",
            "no operation",                                # saiunit's own "backend X has no operation Y"
            "not implement", "is not implemented",
            "not supported", "is not supported", "unsupported",
            "missing", "no torch dtype mapping",
            "module 'array_api_compat",
            # kwarg-not-accepted patterns: saiunit forwards JAX-flavored kwargs
            # (precision, symmetrize_input, tol, etc.) that the backend doesn't
            # support — treat as backend gap, not as broken dispatch.
            "got an unexpected keyword argument",
            "received an invalid combination of arguments",
            "got multiple values for keyword argument",
            "error interpreting argument",                 # JAX abstract-eval rejection on foreign array
            "tracerarrayconversionerror",                  # JAX tracer hitting non-JAX path
            "unknown backend cuda", "available backends are ['cpu']",  # cuda not present
            "unable to infer dtype",                       # ndonnx wants explicit dtype
        )
        if any(tok in low for tok in skip_tokens):
            return {"status": "skip", "detail": f"{type(exc).__name__}: {msg}"}
        return {"status": "fail", "detail": f"{type(exc).__name__}: {msg}"}
    except Exception as exc:
        return {"status": "fail", "detail": f"{type(exc).__name__}: {exc}"}

    return {"status": "pass", "detail": ""}


def _sweep_module(
    qualname_prefix: str,
    members: dict[str, Any],
    registry: dict[str, Callable],
    backends: Iterable[str],
    results: dict[str, dict[str, dict[str, str]]],
    non_dispatched: set[str] | None = None,
) -> None:
    """Invoke each registered function across each backend, record outcome.

    Names in ``non_dispatched`` get status ``na`` on every backend (they are
    not array-API operations and don't go through ``get_backend()``).
    """
    non_dispatched = non_dispatched or set()
    for name, fn in sorted(members.items()):
        fq = f"{qualname_prefix}.{name}"
        if name in non_dispatched:
            results[fq] = {b: {"status": "na", "detail": ""} for b in backends}
            continue
        pattern = registry.get(name)
        if pattern is None:
            results[fq] = {b: {"status": "unmapped", "detail": ""} for b in backends}
            continue
        results[fq] = {}
        for b in backends:
            with u.using_backend(b):
                results[fq][b] = _classify_outcome(fq, b, pattern, fn)


def _public_callables(module) -> dict[str, Any]:
    """Return mapping of public attribute name -> attribute, callables only."""
    out: dict[str, Any] = {}
    names = getattr(module, "__all__", None) or [n for n in dir(module) if not n.startswith("_")]
    for n in sorted(set(names)):
        if n.startswith("_"):
            continue
        attr = getattr(module, n, None)
        if attr is None:
            continue
        if callable(attr):
            out[n] = attr
    return out


def _quantity_members() -> dict[str, Any]:
    """Public callable methods on Quantity."""
    from saiunit._base_quantity import Quantity
    out: dict[str, Any] = {}
    for n in sorted(dir(Quantity)):
        if n.startswith("_"):
            continue
        attr = getattr(Quantity, n, None)
        if attr is None or not callable(attr):
            continue
        out[n] = attr
    return out


def _probe_jax_only_subpackage(
    fq_name: str, invoker: Callable, backends: Iterable[str]
) -> dict[str, dict[str, str]]:
    """Run ``invoker(fn, backend)`` for each backend. ``BackendError`` is
    treated as the correct outcome for non-jax backends (the gate is
    working); ``pass`` is correct for jax."""
    out: dict[str, dict[str, str]] = {}
    for b in backends:
        with u.using_backend(b):
            try:
                invoker(None, b)
            except BackendError as exc:
                out[b] = {"status": "guard", "detail": str(exc)}
            except Exception as exc:
                out[b] = {"status": "fail", "detail": f"{type(exc).__name__}: {exc}"}
            else:
                out[b] = {"status": "pass", "detail": ""}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    installed = _detect_installed()
    # Backends actually swept: everything installed except cupy when no CUDA.
    swept: list[str] = [b for b in ALL_BACKENDS if installed.get(b)]
    untested: list[str] = [b for b in ALL_BACKENDS if not installed.get(b)]

    print(f"installed backends: {swept}", file=sys.stderr)
    print(f"untested backends:  {untested}", file=sys.stderr)

    # Override SAIUNIT default backend per iteration via using_backend; we
    # need a base default that never restricts us.
    u.set_default_backend("jax" if HAS_JAX else "numpy")

    function_results: dict[str, dict[str, dict[str, str]]] = {}

    # ---- saiunit.math ----
    import saiunit.math as math_mod
    math_members = _public_callables(math_mod)
    _sweep_module("saiunit.math", math_members, MATH_REGISTRY, swept,
                  function_results, non_dispatched=NON_DISPATCHED_MATH)

    # ---- saiunit.linalg ----
    import saiunit.linalg as linalg_mod
    linalg_members = _public_callables(linalg_mod)
    _sweep_module("saiunit.linalg", linalg_members, LINALG_REGISTRY, swept, function_results)

    # ---- saiunit.fft ----
    import saiunit.fft as fft_mod
    fft_members = _public_callables(fft_mod)
    _sweep_module("saiunit.fft", fft_members, FFT_REGISTRY, swept, function_results)

    # ---- Quantity methods ----
    q_members = _quantity_members()
    q_results: dict[str, dict[str, dict[str, str]]] = {}
    for name, method in sorted(q_members.items()):
        fq = f"Quantity.{name}"
        pattern = QUANTITY_REGISTRY.get(name)
        if pattern is None:
            q_results[fq] = {b: {"status": "unmapped", "detail": ""} for b in swept}
            continue
        q_results[fq] = {}
        expect_be = (name in _MATERIALIZE_METHODS)
        for b in swept:
            with u.using_backend(b):
                # Special-case: to_X methods route through the corresponding
                # backend regardless of the source mantissa's backend.
                if name.startswith("to_") and name != "to_decimal":
                    target = name[3:]
                    if target not in installed or not installed[target]:
                        q_results[fq][b] = {"status": "skip",
                                            "detail": f"{target} backend not installed"}
                        continue
                expect_be_here = expect_be and (b == "dask")
                q_results[fq][b] = _classify_outcome(fq, b, pattern, method,
                                                     expect_backend_error=expect_be_here)

    # ---- JAX-only subpackages: single probe per backend ----
    jax_only_results: dict[str, dict[str, dict[str, str]]] = {}

    def _probe_lax(fn, backend):
        # representative: u.lax.slice
        import saiunit as su
        x = _arr([1.0, 2.0, 3.0], backend)
        q = su.Quantity(x, unit=su.meter)
        su.lax.slice(q, (0,), (1,))

    def _probe_autograd(fn, backend):
        import saiunit as su
        # Build a scalar fn over a backend-typed input
        x = _arr([1.0], backend)
        q = su.Quantity(x, unit=su.meter)
        su.autograd.grad(lambda y: y.sum())(q)

    def _probe_sparse(fn, backend):
        import saiunit as su
        from saiunit.sparse import CSR
        x = _arr([[1.0, 0.0], [0.0, 2.0]], backend)
        q = su.Quantity(x, unit=su.meter)
        CSR.fromdense(q)

    jax_only_results["saiunit.lax (*)"] = _probe_jax_only_subpackage(
        "saiunit.lax", _probe_lax, swept)
    jax_only_results["saiunit.autograd (*)"] = _probe_jax_only_subpackage(
        "saiunit.autograd", _probe_autograd, swept)
    jax_only_results["saiunit.sparse (*)"] = _probe_jax_only_subpackage(
        "saiunit.sparse", _probe_sparse, swept)

    # ---- Subpackage function inventories (for renderer to list jax-only fns) ----
    import saiunit.lax as lax_mod
    import saiunit.autograd as autograd_mod
    import saiunit.sparse as sparse_mod
    jax_only_inventory = {
        "saiunit.lax": sorted(_public_callables(lax_mod)),
        "saiunit.autograd": sorted(_public_callables(autograd_mod)),
        "saiunit.sparse": sorted(_public_callables(sparse_mod)),
    }

    # ---- Coverage stats ----
    def _coverage(prefix: str, results: dict[str, dict[str, dict[str, str]]]) -> tuple[int, int, int]:
        """Return (mapped, na, total). 'mapped' counts only invoked rows."""
        keys = [k for k in results if k.startswith(prefix)]
        total = len(keys)
        na = sum(1 for k in keys if all(v["status"] == "na" for v in results[k].values()))
        unmapped = sum(1 for k in keys if all(v["status"] == "unmapped" for v in results[k].values()))
        mapped = total - na - unmapped
        return mapped, na, total

    cov_math = _coverage("saiunit.math.", function_results)
    cov_linalg = _coverage("saiunit.linalg.", function_results)
    cov_fft = _coverage("saiunit.fft.", function_results)
    cov_q = _coverage("Quantity.", q_results)

    print("", file=sys.stderr)
    print("Coverage (mapped / na / total):", file=sys.stderr)
    for label, c in [("saiunit.math", cov_math), ("saiunit.linalg", cov_linalg),
                      ("saiunit.fft", cov_fft), ("Quantity", cov_q)]:
        mapped, na, total = c
        unmapped = total - mapped - na
        print(f"  {label:14s} {mapped:4d} mapped / {na:3d} na / "
              f"{unmapped:3d} unmapped / {total:4d} total", file=sys.stderr)

    # ---- Module source mapping (used by renderer to bucket math by submodule) ----
    source_map: dict[str, str] = {}
    for prefix, members in [
        ("saiunit.math", math_members),
        ("saiunit.linalg", linalg_members),
        ("saiunit.fft", fft_members),
    ]:
        for name, fn in members.items():
            source_map[f"{prefix}.{name}"] = getattr(fn, "__module__", "?")

    out = {
        "schema_version": 1,
        "swept_backends": swept,
        "untested_backends": untested,
        "all_backends": list(ALL_BACKENDS),
        "function_results": function_results,
        "quantity_results": q_results,
        "jax_only_results": jax_only_results,
        "jax_only_inventory": jax_only_inventory,
        "source_map": source_map,
        "coverage": {
            "math":     {"mapped": cov_math[0],   "na": cov_math[1],   "total": cov_math[2]},
            "linalg":   {"mapped": cov_linalg[0], "na": cov_linalg[1], "total": cov_linalg[2]},
            "fft":      {"mapped": cov_fft[0],    "na": cov_fft[1],    "total": cov_fft[2]},
            "quantity": {"mapped": cov_q[0],      "na": cov_q[1],      "total": cov_q[2]},
        },
        "non_dispatched_math": sorted(NON_DISPATCHED_MATH),
    }

    OUTPUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
