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

"""
Numerically stable implementation of expm1(x)/x as a JAX primitive.

The function f(x) = (exp(x) - 1) / x has a removable singularity at x = 0.
Using Taylor expansion: f(x) = 1 + x/2! + x²/3! + x³/4! + ...
So f(0) = 1.

The derivative f'(x) = [(x-1)*exp(x) + 1] / x² also needs Taylor expansion near 0:
f'(x) = 1/2 + x/3 + x²/8 + x³/30 + ...
So f'(0) = 1/2.
"""

from math import factorial
from typing import Optional

import jax.numpy as jnp
from jax import core
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir

from saiunit._compatible_import import Primitive

__all__ = ['exprel', 'set_exprel_order']

# Module-level configuration for Taylor series order
_DEFAULT_ORDER = 5
_current_order = _DEFAULT_ORDER


def set_exprel_order(order: int) -> None:
    """
    Set the Taylor series order for exprel computation.

    Parameters
    ----------
    order : int
        The order of the Taylor series expansion. Higher order provides better
        accuracy near x=0 but requires more computation. Default is 5.
        Valid range: 2-20.

    Notes
    -----
    The Taylor expansion for exprel(x) = (exp(x) - 1) / x is:
    f(x) = 1 + x/2! + x²/3! + ... + x^n/(n+1)!

    For order=5, this gives accuracy to about x^5 term.
    """
    global _current_order
    if not isinstance(order, int) or order < 2 or order > 20:
        raise ValueError(f"order must be an integer between 2 and 20, got {order}")
    _current_order = order


def get_exprel_order() -> int:
    """Get the current Taylor series order for exprel computation."""
    return _current_order


def _get_threshold(dtype) -> float:
    """
    Get appropriate threshold based on dtype precision.

    The threshold determines where to switch from Taylor expansion to direct
    computation. Higher precision dtypes can use smaller thresholds.

    Parameters
    ----------
    dtype : jnp.dtype
        The data type of the input array.

    Returns
    -------
    float
        The threshold value for switching between Taylor and direct computation.

    Notes
    -----
    Thresholds are chosen to provide a safety margin for numerical stability.
    Values slightly larger than the point where direct computation becomes
    unstable are used to ensure Taylor series is used in the transition region.
    """
    if dtype == jnp.float64:
        return 1e-7  # High precision - can use Taylor for very small values
    elif dtype == jnp.float32:
        return 1e-4  # Standard precision - larger margin for stability
    elif dtype == jnp.float16:
        return 1e-2  # Low precision - need larger Taylor region
    elif dtype == jnp.bfloat16:
        return 1e-2  # Low precision
    else:
        # Default fallback for other dtypes
        return 1e-4


def _exprel_coefficients(order: int):
    """
    Generate Taylor coefficients for exprel using Horner's method.

    The Taylor expansion of (exp(x) - 1) / x is:
    f(x) = 1 + x/2! + x²/3! + x³/4! + ... + x^n/(n+1)!

    For Horner's method, we need coefficients in reverse order:
    f(x) = 1 + x*(1/2 + x*(1/6 + x*(1/24 + ...)))

    Parameters
    ----------
    order : int
        The order of the Taylor expansion (number of terms beyond constant).

    Returns
    -------
    list
        Coefficients in order for Horner's method evaluation
        (highest order first, constant term last).
    """
    # Coefficients are 1/(n+1)! for n = order, order-1, ..., 1, 0
    # For Horner: we need [1/(order+1)!, 1/order!, ..., 1/2!, 1/1!]
    # where the last coefficient (1/1! = 1) is the constant term
    return [1.0 / factorial(n + 1) for n in range(order, -1, -1)]


def _exprel_deriv_coefficients(order: int):
    """
    Generate Taylor coefficients for exprel derivative using Horner's method.

    The derivative of exprel is:
    f'(x) = sum_{n=0}^{inf} (n+1) / (n+2)! * x^n
          = 1/2 + x/3 + x²/8 + x³/30 + x⁴/144 + ...

    Parameters
    ----------
    order : int
        The order of the Taylor expansion.

    Returns
    -------
    list
        Coefficients in order for Horner's method evaluation.
    """
    # Coefficient for x^n is (n+1)/(n+2)!
    # For Horner: need coefficients from highest to lowest order
    return [(n + 1) / factorial(n + 2) for n in range(order, -1, -1)]


def _exprel_taylor(x, order: Optional[int] = None):
    """
    Taylor expansion of (exp(x) - 1) / x around x = 0.

    f(x) = 1 + x/2! + x²/3! + x³/4! + x⁴/5! + x⁵/6! + ...

    Uses Horner's method for numerical stability.

    Parameters
    ----------
    x : array_like
        Input array.
    order : int, optional
        Order of Taylor expansion. If None, uses module default.

    Returns
    -------
    array_like
        Taylor approximation of exprel(x).
    """
    if order is None:
        order = _current_order

    coeffs = _exprel_coefficients(order)
    result = jnp.zeros_like(x)
    for c in coeffs:
        result = result * x + c
    return result


def _exprel_direct(x):
    """Direct computation: (exp(x) - 1) / x using expm1 for better precision."""
    return jnp.expm1(x) / x


def _exprel_impl(x, order: Optional[int] = None):
    """
    Numerically stable implementation of (exp(x) - 1) / x.

    Uses Taylor expansion for |x| < threshold, direct computation otherwise.
    Threshold is adaptive based on dtype.

    Parameters
    ----------
    x : array_like
        Input array.
    order : int, optional
        Order of Taylor expansion. If None, uses module default.

    Returns
    -------
    array_like
        exprel(x) computed in a numerically stable way.
    """
    dtype = x.dtype
    threshold = _get_threshold(dtype)
    abs_x = jnp.abs(x)

    # Use where to select between Taylor and direct computation
    return jnp.where(
        abs_x <= threshold,
        _exprel_taylor(x, order),
        _exprel_direct(x)
    )


def _exprel_deriv_taylor(x, order: Optional[int] = None):
    """
    Taylor expansion of d/dx[(exp(x) - 1) / x] around x = 0.

    f'(x) = [(x-1)*exp(x) + 1] / x²

    Taylor expansion: f'(x) = 1/2 + x/3 + x²/8 + x³/30 + x⁴/144 + ...

    More precisely, the coefficients are:
    f'(x) = sum_{n=0}^{inf} (n+1) / (n+2)! * x^n

    Parameters
    ----------
    x : array_like
        Input array.
    order : int, optional
        Order of Taylor expansion. If None, uses module default.

    Returns
    -------
    array_like
        Taylor approximation of exprel'(x).
    """
    if order is None:
        order = _current_order

    coeffs = _exprel_deriv_coefficients(order)
    result = jnp.zeros_like(x)
    for c in coeffs:
        result = result * x + c
    return result


def _exprel_deriv_direct(x):
    """
    Direct computation of d/dx[(exp(x) - 1) / x].

    f'(x) = [(x-1)*exp(x) + 1] / x²
          = [exp(x) - (exp(x) - 1)/x] / x
          = [exp(x) - exprel(x)] / x
    """
    exp_x = jnp.exp(x)
    return ((x - 1) * exp_x + 1) / (x * x)


def _exprel_deriv(x, order: Optional[int] = None):
    """
    Numerically stable derivative of exprel.

    Parameters
    ----------
    x : array_like
        Input array.
    order : int, optional
        Order of Taylor expansion. If None, uses module default.

    Returns
    -------
    array_like
        exprel'(x) computed in a numerically stable way.
    """
    dtype = x.dtype
    threshold = _get_threshold(dtype)
    abs_x = jnp.abs(x)

    return jnp.where(
        abs_x <= threshold,
        _exprel_deriv_taylor(x, order),
        _exprel_deriv_direct(x)
    )


def exprel(x, /, order: int = 2):
    """
    Compute (exp(x) - 1) / x in a numerically stable way.

    This function handles the removable singularity at x = 0 by using
    Taylor expansion for small |x|.

    Parameters
    ----------
    x : array_like
        Input array.
    order: int
        The order of the Taylor series expansion to use for small |x|.

    Returns
    -------
    y : ndarray
        (exp(x) - 1) / x, with the singularity at x = 0 handled correctly.

    Notes
    -----
    - At x = 0, the function returns 1 (the limit value).
    - For small |x|, Taylor expansion is used for numerical stability.
      The threshold is adaptive based on the input dtype.
    - The function, its JVP, and VJP are all numerically stable.
    - Use `set_exprel_order(n)` to control Taylor series order (default: 5).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from saiunit.math import exprel
    >>> exprel(jnp.array([0.0, 1.0, -1.0]))
    Array([1.        , 1.7182819 , 0.63212055], dtype=float32)
    """
    x = jnp.asarray(x)
    return exprel_p.bind(x, order=order)


def _exprel_abstract_eval(x, order=None):
    return core.ShapedArray(x.shape, x.dtype)


def _exprel_jvp(primals, tangents, *, order):
    """
    JVP rule for exprel.

    d/dx[(exp(x) - 1) / x] = [(x-1)*exp(x) + 1] / x²

    For numerical stability near x = 0, we use Taylor expansion.
    """
    x, = primals
    x_dot, = tangents

    primal_out = exprel(x, order=order)
    tangent_out = _exprel_deriv(x, order=order) * x_dot

    return primal_out, tangent_out


def _exprel_transpose(cotangent, x, *, order):
    """
    Transpose rule for exprel (used for reverse-mode AD).

    For a function f(x), the transpose of df/dx is just df/dx * cotangent.
    This is because exprel is an elementwise operation.
    """
    # The transpose of the JVP tangent computation is:
    # tangent_out = deriv(x) * x_dot
    # So the cotangent propagation is:
    # x_cot = deriv(x) * cotangent
    return (_exprel_deriv(x, order=order) * cotangent,)


def _exprel_batching(batched_args, batch_dims, *, order):
    x, = batched_args
    bd, = batch_dims
    return exprel(x, order=order), bd


def _exprel_lowering(ctx, x, *, order):
    """Lowering rule for exprel - uses the stable implementation."""

    def impl_with_order(x):
        return _exprel_impl(x, order=order)

    return mlir.lower_fun(impl_with_order, multiple_results=False)(ctx, x)


# Define the primitive
exprel_p = Primitive("exprel")
exprel_p.def_impl(_exprel_impl)
exprel_p.def_abstract_eval(_exprel_abstract_eval)

# Register JVP (forward-mode AD)
ad.primitive_jvps[exprel_p] = _exprel_jvp

# Register transpose rule for reverse-mode AD
# Note: For elementwise operations, the transpose of df/dx * tangent is df/dx * cotangent
ad.primitive_transposes[exprel_p] = _exprel_transpose

# Register batching rule for vmap
batching.primitive_batchers[exprel_p] = _exprel_batching

# Register MLIR lowering for JIT compilation
mlir.register_lowering(exprel_p, _exprel_lowering)
