# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Union

import jax
from jax import nn

from saiunit._base_getters import get_mantissa
from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as, maybe_custom_array
from ._fun_accept_unitless import _fun_accept_unitless_unary
from ._fun_array_creation import asarray
from ._fun_keep_unit import _fun_keep_unit_unary, where

__all__ = [
    'relu', 'relu6', 'sigmoid', 'softplus', 'sparse_plus', 'sparse_sigmoid', 'soft_sign', 'silu', 'swish',
    'log_sigmoid', 'leaky_relu', 'hard_sigmoid', 'hard_silu', 'hard_swish', 'hard_tanh', 'elu', 'celu', 'selu', 'gelu',
    'glu', 'squareplus', 'mish',
]


@set_module_as('saiunit.math')
def relu(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> Union[Quantity, jax.Array]:
    r"""Rectified linear unit activation function.

    Computes the element-wise function:

    .. math::
        \mathrm{relu}(x) = \max(x, 0)

    except under differentiation, we take:

    .. math::
        \nabla \mathrm{relu}(0) = 0

    For more information see
    `Numerical influence of ReLU’(0) on backpropagation
    <https://openreview.net/forum?id=urrcVI-_jRm>`_.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. If a ``Quantity`` with physical units is provided,
        the units are preserved in the output.

    Returns
    -------
    out : jax.Array or Quantity
        An array with the same shape as *x* where negative values are
        replaced by zero. Units are preserved when present.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.relu(jnp.array([-2., -1., 0., 1., 2.]))
        Array([0., 0., 0., 1., 2.], dtype=float32)

        >>> import saiunit as su
        >>> q = su.Quantity(jnp.array([-1., 0., 1.]), unit=su.meter)
        >>> sumath.relu(q)  # units are preserved
    """
    return _fun_keep_unit_unary(nn.relu, x)


@set_module_as('saiunit.math')
def relu6(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Rectified Linear Unit 6 activation function.

    Computes the element-wise function

    .. math::
        \mathrm{relu6}(x) = \min(\max(x, 0), 6)

    except under differentiation, we take:

    .. math::
        \nabla \mathrm{relu}(0) = 0

    and

    .. math::
        \nabla \mathrm{relu}(6) = 0

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with the same shape as *x*, clipped to the range [0, 6].

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.relu6(jnp.array([-1., 0., 3., 7.]))
        Array([0., 0., 3., 6.], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.relu6, x)


@set_module_as('saiunit.math')
def sigmoid(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Sigmoid activation function.

    Computes the element-wise function:

    .. math::
        \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with values in the range (0, 1).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.sigmoid(jnp.array([-2., 0., 2.]))
        Array([0.11920292, 0.5       , 0.8807971 ], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.sigmoid, x)


@set_module_as('saiunit.math')
def softplus(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Softplus activation function.

    Computes the element-wise function

    .. math::
        \mathrm{softplus}(x) = \log(1 + e^x)

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with non-negative values.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.softplus(jnp.array([-2., 0., 2.]))
        Array([0.12692805, 0.6931472 , 2.126928  ], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.softplus, x)


@set_module_as('saiunit.math')
def sparse_plus(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Sparse plus function.

    Computes the function:

    .. math::

        \mathrm{sparse\_plus}(x) = \begin{cases}
            0, & x \leq -1\\
            \frac{1}{4}(x+1)^2, & -1 < x < 1 \\
            x, & 1 \leq x
        \end{cases}

    This is the twin function of the softplus activation ensuring a zero output
    for inputs less than -1 and a linear output for inputs greater than 1,
    while remaining smooth, convex, monotonic by an adequate definition between
    -1 and 1.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with the same shape as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.sparse_plus(jnp.array([-2., 0., 2.]))
        Array([0.  , 0.25, 2.  ], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.sparse_plus, x)


@set_module_as('saiunit.math')
def sparse_sigmoid(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Sparse sigmoid activation function.

    Computes the function:

    .. math::

        \mathrm{sparse\_sigmoid}(x) = \begin{cases}
              0, & x \leq -1\\
              \frac{1}{2}(x+1), & -1 < x < 1 \\
              1, & 1 \leq x
        \end{cases}

    This is the twin function of the ``sigmoid`` activation ensuring a zero output
    for inputs less than -1, a 1 output for inputs greater than 1, and a linear
    output for inputs between -1 and 1. It is the derivative of ``sparse_plus``.

    For more information, see `Learning with Fenchel-Young Losses (section 6.2)
    <https://arxiv.org/abs/1901.02324>`_.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with values in the range [0, 1].

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.sparse_sigmoid(jnp.array([-2., 0., 2.]))
        Array([0. , 0.5, 1. ], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.sparse_sigmoid, x)


@set_module_as('saiunit.math')
def soft_sign(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Soft-sign activation function.

    Computes the element-wise function

    .. math::
        \mathrm{soft\_sign}(x) = \frac{x}{|x| + 1}

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with values in the range (-1, 1).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.soft_sign(jnp.array([-2., 0., 2.]))
        Array([-0.6666667,  0.       ,  0.6666667], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.soft_sign, x)


@set_module_as('saiunit.math')
def silu(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""SiLU (aka swish) activation function.

    Computes the element-wise function:

    .. math::
        \mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}

    :func:`swish` and :func:`silu` are both aliases for the same function.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with the same shape as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.silu(jnp.array([-2., 0., 2.]))
        Array([-0.23840584,  0.        ,  1.7615942 ], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.silu, x)


@set_module_as('saiunit.math')
def swish(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Swish (aka SiLU) activation function.

    Computes the element-wise function:

    .. math::
        \mathrm{swish}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}

    :func:`swish` and :func:`silu` are both aliases for the same function.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with the same shape as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.swish(jnp.array([-2., 0., 2.]))
        Array([-0.23840584,  0.        ,  1.7615942 ], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.silu, x)


@set_module_as('saiunit.math')
def log_sigmoid(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Log-sigmoid activation function.

    Computes the element-wise function:

    .. math::
        \mathrm{log\_sigmoid}(x) = \log(\mathrm{sigmoid}(x)) = -\log(1 + e^{-x})

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with non-positive values.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.log_sigmoid(jnp.array([-2., 0., 2.]))
        Array([-2.126928  , -0.6931472 , -0.12692805], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.log_sigmoid, x)


@set_module_as('saiunit.math')
def leaky_relu(
    x: Union[Quantity, jax.typing.ArrayLike],
    negative_slope: jax.typing.ArrayLike = 1e-2
) -> Union[Quantity, jax.Array]:
    r"""Leaky rectified linear unit activation function.

    Computes the element-wise function:

    .. math::
        \mathrm{leaky\_relu}(x) = \begin{cases}
            x, & x \ge 0\\
            \alpha x, & x < 0
        \end{cases}

    where :math:`\alpha` = :code:`negative_slope`.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. If a ``Quantity`` with physical units is provided,
        the units are preserved in the output.
    negative_slope : array_like, optional
        Slope for negative input values. Default is 0.01.

    Returns
    -------
    out : jax.Array or Quantity
        An array with the same shape as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.leaky_relu(jnp.array([-2., -1., 0., 1., 2.]))
        Array([-0.02, -0.01,  0.  ,  1.  ,  2.  ], dtype=float32)
        >>> sumath.leaky_relu(jnp.array([-1., 1.]), negative_slope=0.1)
        Array([-0.1,  1. ], dtype=float32)
    """
    x = maybe_custom_array(x)
    x_arr = asarray(x)
    return where(get_mantissa(x_arr) >= 0, x_arr, negative_slope * x_arr)


@set_module_as('saiunit.math')
def hard_sigmoid(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Hard Sigmoid activation function.

    Computes the element-wise function

    .. math::
        \mathrm{hard\_sigmoid}(x) = \frac{\mathrm{relu6}(x + 3)}{6}

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with values in the range [0, 1].

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.hard_sigmoid(jnp.array([-4., 0., 4.]))
        Array([0. , 0.5, 1. ], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.hard_sigmoid, x)


@set_module_as('saiunit.math')
def hard_silu(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Hard SiLU (swish) activation function.

    Computes the element-wise function

    .. math::
        \mathrm{hard\_silu}(x) = x \cdot \mathrm{hard\_sigmoid}(x)

    Both :func:`hard_silu` and :func:`hard_swish` are aliases for the same
    function.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with the same shape as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.hard_silu(jnp.array([-4., 0., 4.]))
        Array([-0.,  0.,  4.], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.hard_silu, x)


hard_swish = hard_silu


@set_module_as('saiunit.math')
def hard_tanh(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Hard :math:`\mathrm{tanh}` activation function.

    Computes the element-wise function:

    .. math::
        \mathrm{hard\_tanh}(x) = \begin{cases}
            -1, & x < -1\\
            x, & -1 \le x \le 1\\
            1, & 1 < x
        \end{cases}

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with values clipped to the range [-1, 1].

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.hard_tanh(jnp.array([-2., -0.5, 0., 0.5, 2.]))
        Array([-1. , -0.5,  0. ,  0.5,  1. ], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.hard_tanh, x)


@set_module_as('saiunit.math')
def elu(
    x: Union[Quantity, jax.typing.ArrayLike],
    alpha: jax.typing.ArrayLike = 1.0
) -> jax.Array:
    r"""Exponential linear unit activation function.

    Computes the element-wise function:

    .. math::
        \mathrm{elu}(x) = \begin{cases}
            x, & x > 0\\
            \alpha \left(\exp(x) - 1\right), & x \le 0
        \end{cases}

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.
    alpha : array_like, optional
        Scale for the negative region. Default is 1.0.

    Returns
    -------
    out : jax.Array
        An array with the same shape as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.elu(jnp.array([-2., 0., 2.]))
        Array([-0.86466473,  0.        ,  2.        ], dtype=float32)
        >>> sumath.elu(jnp.array([-1., 1.]), alpha=2.0)
        Array([-1.2642411,  1.       ], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.elu, x, alpha=alpha)


@set_module_as('saiunit.math')
def celu(
    x: Union[Quantity, jax.typing.ArrayLike],
    alpha: jax.typing.ArrayLike = 1.0
) -> jax.Array:
    r"""Continuously-differentiable exponential linear unit activation.

    Computes the element-wise function:

    .. math::
        \mathrm{celu}(x) = \begin{cases}
            x, & x > 0\\
            \alpha \left(\exp(\frac{x}{\alpha}) - 1\right), & x \le 0
        \end{cases}

    For more information, see
    `Continuously Differentiable Exponential Linear Units
    <https://arxiv.org/abs/1704.07483>`_.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.
    alpha : array_like, optional
        Scale parameter. Default is 1.0.

    Returns
    -------
    out : jax.Array
        An array with the same shape as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.celu(jnp.array([-2., 0., 2.]))
        Array([-0.86466473,  0.        ,  2.        ], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.celu, x, alpha=alpha)


@set_module_as('saiunit.math')
def selu(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Scaled exponential linear unit activation.

    Computes the element-wise function:

    .. math::
        \mathrm{selu}(x) = \lambda \begin{cases}
            x, & x > 0\\
            \alpha e^x - \alpha, & x \le 0
        \end{cases}

    where :math:`\lambda = 1.0507009873554804934193349852946` and
    :math:`\alpha = 1.6732632423543772848170429916717`.

    For more information, see
    `Self-Normalizing Neural Networks
    <https://arxiv.org/abs/1706.02515>`_.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with the same shape as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.selu(jnp.array([-2., 0., 2.]))
        Array([-1.5201665,  0.       ,  2.1014020], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.selu, x)


@set_module_as('saiunit.math')
def gelu(
    x: Union[Quantity, jax.typing.ArrayLike],
    approximate: bool = True
) -> jax.Array:
    r"""Gaussian error linear unit activation function.

    If ``approximate=False``, computes the element-wise function:

    .. math::
        \mathrm{gelu}(x) = \frac{x}{2} \left(\mathrm{erfc} \left(
            \frac{-x}{\sqrt{2}} \right) \right)

    If ``approximate=True``, uses the approximate formulation of GELU:

    .. math::
        \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left(
            \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)

    For more information, see `Gaussian Error Linear Units (GELUs)
    <https://arxiv.org/abs/1606.08415>`_, section 2.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.
    approximate : bool, optional
        Whether to use the approximate or exact formulation. Default is True.

    Returns
    -------
    out : jax.Array
        An array with the same shape as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.gelu(jnp.array([-2., 0., 2.]))
        Array([-0.04540231,  0.        ,  1.9545977 ], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.gelu, x, approximate=approximate)


@set_module_as('saiunit.math')
def glu(
    x: Union[Quantity, jax.typing.ArrayLike],
    axis: int = -1
) -> jax.Array:
    r"""Gated linear unit activation function.

    Computes the function:

    .. math::
        \mathrm{glu}(x) =  x\left[\ldots, 0:\frac{n}{2}, \ldots\right] \cdot
            \mathrm{sigmoid} \left( x\left[\ldots, \frac{n}{2}:n, \ldots\right]
                \right)

    where the array is split into two along ``axis``. The size of the ``axis``
    dimension must be divisible by two.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``. The size of
        the dimension specified by *axis* must be even.
    axis : int, optional
        The axis along which to split the input. Default is -1.

    Returns
    -------
    out : jax.Array
        An array whose size along *axis* is half that of the input.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> x = jnp.array([[1., 2., 3., 4.]])
        >>> sumath.glu(x)
        Array([[0.95257413, 1.9640275 ]], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.glu, x, axis=axis)


@set_module_as('saiunit.math')
def squareplus(
    x: Union[Quantity, jax.typing.ArrayLike],
    b: jax.typing.ArrayLike = 4
) -> jax.Array:
    r"""Squareplus activation function.

    Computes the element-wise function

    .. math::
        \mathrm{squareplus}(x) = \frac{x + \sqrt{x^2 + b}}{2}

    as described in https://arxiv.org/abs/2112.11687.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.
    b : array_like, optional
        Smoothness parameter. Default is 4.

    Returns
    -------
    out : jax.Array
        An array with non-negative values.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.squareplus(jnp.array([-2., 0., 2.]))
        Array([0.23606798, 1.        , 2.2360680 ], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.squareplus, x, b=b)


@set_module_as('saiunit.math')
def mish(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Mish activation function.

    Computes the element-wise function:

    .. math::
        \mathrm{mish}(x) = x \cdot \mathrm{tanh}(\mathrm{softplus}(x))

    For more information, see
    `Mish: A Self Regularized Non-Monotonic Activation Function
    <https://arxiv.org/abs/1908.08681>`_.

    Parameters
    ----------
    x : array_like or Quantity
        Input array. Must be unitless if a ``Quantity``.

    Returns
    -------
    out : jax.Array
        An array with the same shape as *x*.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import saiunit.math as sumath
        >>> sumath.mish(jnp.array([-2., 0., 2.]))
        Array([-0.25250152,  0.        ,  1.9439590 ], dtype=float32)
    """
    return _fun_accept_unitless_unary(nn.mish, x)
