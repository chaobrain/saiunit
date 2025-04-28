# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from ._fun_accept_unitless import _fun_accept_unitless_unary
from ._fun_array_creation import asarray
from ._fun_keep_unit import _fun_keep_unit_unary, where
from .._base import Quantity, get_mantissa
from .._misc import set_module_as

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

    Args:
        x : input array

    Returns:
        An array.

    Examples:
        >>> saiunit.math.relu(jax.numpy.array([-2., -1., -0.5, 0, 0.5, 1., 2.]))
        Array([0. , 0. , 0. , 0. , 0.5, 1. , 2. ], dtype=float32)
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

    Args:
        x : input array

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.relu6, x)


def sigmoid(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Sigmoid activation function.

    Computes the element-wise function:

    .. math::
        \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

    Args:
        x : input array

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.sigmoid, x)


def softplus(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Softplus activation function.

    Computes the element-wise function

    .. math::
        \mathrm{softplus}(x) = \log(1 + e^x)

    Args:
        x : input array
    """
    return _fun_accept_unitless_unary(nn.softplus, x)


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

    Args:
        x: input (float)
    """
    return _fun_accept_unitless_unary(nn.sparse_plus, x)


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

    Args:
        x : input array

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.sparse_sigmoid, x)


def soft_sign(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Soft-sign activation function.

    Computes the element-wise function

    .. math::
        \mathrm{soft\_sign}(x) = \frac{x}{|x| + 1}

    Args:
        x : input array
    """
    return _fun_accept_unitless_unary(nn.soft_sign, x)


def silu(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""SiLU (aka swish) activation function.

    Computes the element-wise function:

    .. math::
        \mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}

    :func:`swish` and :func:`silu` are both aliases for the same function.

    Args:
        x : input array

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.silu, x)


def swish(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Mish activation function.

    Computes the element-wise function:

    .. math::
        \mathrm{mish}(x) = x \cdot \mathrm{tanh}(\mathrm{softplus}(x))

    For more information, see
    `Mish: A Self Regularized Non-Monotonic Activation Function
    <https://arxiv.org/abs/1908.08681>`_.

    Args:
        x : input array

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.swish, x)


def log_sigmoid(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Log-sigmoid activation function.

    Computes the element-wise function:

    .. math::
        \mathrm{log\_sigmoid}(x) = \log(\mathrm{sigmoid}(x)) = -\log(1 + e^{-x})

    Args:
        x : input array

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.log_sigmoid, x)


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

    Args:
        x : input array
        negative_slope : array or scalar specifying the negative slope (default: 0.01)

    Returns:
        An array.
    """
    x_arr = asarray(x)
    return where(get_mantissa(x_arr) >= 0, x_arr, negative_slope * x_arr)


def hard_sigmoid(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Hard Sigmoid activation function.

    Computes the element-wise function

    .. math::
        \mathrm{hard\_sigmoid}(x) = \frac{\mathrm{relu6}(x + 3)}{6}

    Args:
        x : input array

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.hard_sigmoid, x)


def hard_silu(
    x: Union[Quantity, jax.typing.ArrayLike],
) -> jax.Array:
    r"""Hard SiLU (swish) activation function

    Computes the element-wise function

    .. math::
        \mathrm{hard\_silu}(x) = x \cdot \mathrm{hard\_sigmoid}(x)

    Both :func:`hard_silu` and :func:`hard_swish` are aliases for the same
    function.

    Args:
        x : input array

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.hard_silu, x)


hard_swish = hard_silu


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

    Args:
        x : input array

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.hard_tanh, x)


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

    Args:
        x : input array
        alpha : scalar or array of alpha values (default: 1.0)

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.elu, x, alpha=alpha)


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

    Args:
        x : input array
        alpha : array or scalar (default: 1.0)

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.celu, x, alpha=alpha)


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

    Args:
        x : input array

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.selu, x)


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

    Args:
        x: input array
        approximate: whether to use the approximate or exact formulation.
    """
    return _fun_accept_unitless_unary(nn.gelu, x, approximate=approximate)


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

    Args:
        x : input array
        axis: the axis along which the split should be computed (default: -1)

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.glu, x, axis=axis)


def squareplus(
    x: Union[Quantity, jax.typing.ArrayLike],
    b: jax.typing.ArrayLike = 4
) -> jax.Array:
    r"""Squareplus activation function.

    Computes the element-wise function

    .. math::
        \mathrm{squareplus}(x) = \frac{x + \sqrt{x^2 + b}}{2}

    as described in https://arxiv.org/abs/2112.11687.

    Args:
        x : input array
        b : smoothness parameter
    """
    return _fun_accept_unitless_unary(nn.squareplus, x, b=b)


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

    Args:
        x : input array

    Returns:
        An array.
    """
    return _fun_accept_unitless_unary(nn.mish, x)
