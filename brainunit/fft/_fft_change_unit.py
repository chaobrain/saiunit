# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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
from __future__ import annotations

import sys
from typing import Callable, Union, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy import fft as jnpfft
from jaxlib import xla_client

from .. import _unit_common as u
from .._base import Quantity, Unit, get_or_create_dimension
from .._misc import set_module_as
from .._unit_common import second
from ..math._fun_change_unit import _fun_change_unit_unary

__all__ = [
    # return original unit * time unit
    'fft', 'rfft',
    # return original unit / time unit (inverse)
    'ifft', 'irfft',
    # return original unit * (time unit ^ n)
    'fft2', 'fftn', 'rfft2', 'rfftn',
    # return original unit / (time unit ^ n) (inverse)
    'ifft2', 'ifftn', 'irfft2', 'irfftn',
    # return frequency unit
    'fftfreq', 'rfftfreq',
]


def unit_change(
    unit_change_fun: Callable
):
    def actual_decorator(func):
        func._unit_change_fun = unit_change_fun
        return set_module_as('brainunit.fft')(func)

    return actual_decorator


Shape = Sequence[int]


# return original unit * time unit
# --------------------------------

def _calculate_fftn_dimension(input_dim: int, axes: Sequence[int] | None = None) -> int:
    if axes is None:
        return input_dim
    return len(axes)


@unit_change(lambda u: u * second)
def fft(
    a: Union[Quantity, jax.typing.ArrayLike],
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Union[Quantity, jax.typing.ArrayLike]:
    r"""Compute a one-dimensional discrete Fourier transform along a given axis.

    Brainunit implementation of :func:`numpy.fft.fft`.

    Args:
        a: input array
        n: int. Specifies the dimension of the result along ``axis``. If not specified,
            it will default to the dimension of ``a`` along ``axis``.
        axis: int, default=-1. Specifies the axis along which the transform is computed.
            If not specified, the transform is computed along axis -1.
        norm: string. The normalization mode. "backward", "ortho" and "forward" are
            supported.

    Returns:
        An array containing the one-dimensional discrete Fourier transform of ``a``.

    See also:
        - :func:`brainunit.fft.ifft`: Computes a one-dimensional inverse discrete
            Fourier transform.
        - :func:`brainunit.fft.fftn`: Computes a multidimensional discrete Fourier
            transform.
        - :func:`brainunit.fft.ifftn`: Computes a multidimensional inverse discrete
            Fourier transform.

    Examples:
        ``brainunit.fft.fft`` computes the transform along ``axis -1`` by default.

        >>> x = jnp.array([[1, 2, 4, 7],
        ...                [5, 3, 1, 9]])
        >>> brainunit.fft.fft(x)
        Array([[14.+0.j, -3.+5.j, -4.+0.j, -3.-5.j],
               [18.+0.j,  4.+6.j, -6.+0.j,  4.-6.j]], dtype=complex64)

        When ``n=3``, dimension of the transform along axis -1 will be ``3`` and
        dimension along other axes will be the same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   print(brainunit.fft.fft(x, n=3))
        [[ 7.+0.j   -2.+1.73j -2.-1.73j]
         [ 9.+0.j    3.-1.73j  3.+1.73j]]

        When ``n=3`` and ``axis=0``, dimension of the transform along ``axis 0`` will
        be ``3`` and dimension along other axes will be same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   print(brainunit.fft.fft(x, n=3, axis=0))
        [[ 6. +0.j    5. +0.j    5. +0.j   16. +0.j  ]
         [-1.5-4.33j  0.5-2.6j   3.5-0.87j  2.5-7.79j]
         [-1.5+4.33j  0.5+2.6j   3.5+0.87j  2.5+7.79j]]

        ``jnp.fft.ifft`` can be used to reconstruct ``x`` from the result of
        ``jnp.fft.fft``.

        >>> x_fft = jnp.fft.fft(x)
        >>> jnp.allclose(x, brainunit.fft.ifft(x_fft))
        Array(True, dtype=bool)
    """
    # check target_time_unit.dim == second.dim
    return _fun_change_unit_unary(jnpfft.fft,
                                  lambda u: u * second,
                                  a, n=n, axis=axis, norm=norm)


@unit_change(lambda u: u * second)
def rfft(
    a: Union[Quantity, jax.typing.ArrayLike],
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Union[Quantity, jax.typing.ArrayLike]:
    r"""Compute a one-dimensional discrete Fourier transform of a real-valued array.

    Brainunit implementation of :func:`numpy.fft.rfft`.

    Args:
        a: real-valued input array.
        n: int. Specifies the effective dimension of the input along ``axis``. If not
            specified, it will default to the dimension of input along ``axis``.
        axis: int, default=-1. Specifies the axis along which the transform is computed.
            If not specified, the transform is computed along axis -1.
        norm: string. The normalization mode. "backward", "ortho" and "forward" are
            supported.

    Returns:
        An array containing the one-dimensional discrete Fourier transform of ``a``.
        The dimension of the array along ``axis`` is ``(n/2)+1``, if ``n`` is even and
        ``(n+1)/2``, if ``n`` is odd.

    See also:
        - :func:`brainunit.fft.fft`: Computes a one-dimensional discrete Fourier
            transform.
        - :func:`brainunit.fft.irfft`: Computes a one-dimensional inverse discrete
            Fourier transform for real input.
        - :func:`brainunit.fft.rfftn`: Computes a multidimensional discrete Fourier
            transform for real input.
        - :func:`brainunit.fft.irfftn`: Computes a multidimensional inverse discrete
            Fourier transform for real input.

    Examples:
        ``brainunit.fft.rfft`` computes the transform along ``axis -1`` by default.

        >>> x = jnp.array([[1, 3, 5],
        ...                [2, 4, 6]])
        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.rfft(x)
        Array([[ 9.+0.j  , -3.+1.73j],
               [12.+0.j  , -3.+1.73j]], dtype=complex64)

        When ``n=5``, dimension of the transform along axis -1 will be ``(5+1)/2 =3``
        and dimension along other axes will be the same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.rfft(x, n=5)
        Array([[ 9.  +0.j  , -2.12-5.79j,  0.12+2.99j],
               [12.  +0.j  , -1.62-7.33j,  0.62+3.36j]], dtype=complex64)

        When ``n=4`` and ``axis=0``, dimension of the transform along ``axis 0`` will
        be ``(4/2)+1 =3`` and dimension along other axes will be same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.rfft(x, n=4, axis=0)
        Array([[ 3.+0.j,  7.+0.j, 11.+0.j],
               [ 1.-2.j,  3.-4.j,  5.-6.j],
               [-1.+0.j, -1.+0.j, -1.+0.j]], dtype=complex64)
    """
    return _fun_change_unit_unary(jnpfft.rfft,
                                  lambda u: u * second,
                                  a, n=n, axis=axis, norm=norm)


# return original unit / time unit (inverse)
# ------------------------------------------


@unit_change(lambda u: u / second)
def ifft(
    a: Union[Quantity, jax.typing.ArrayLike],
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Union[Quantity, jax.typing.ArrayLike]:
    r"""Compute a one-dimensional inverse discrete Fourier transform.

    Brainunit implementation of :func:`numpy.fft.ifft`.

    Args:
        a: input array
        n: int. Specifies the dimension of the result along ``axis``. If not specified,
            it will default to the dimension of ``a`` along ``axis``.
        axis: int, default=-1. Specifies the axis along which the transform is computed.
            If not specified, the transform is computed along axis -1.
        norm: string. The normalization mode. "backward", "ortho" and "forward" are
            supported.

    Returns:
        An array containing the one-dimensional discrete Fourier transform of ``a``.

    See also:
        - :func:`brainunit.fft.fft`: Computes a one-dimensional discrete Fourier
            transform.
        - :func:`brainunit.fft.fftn`: Computes a multidimensional discrete Fourier
            transform.
        - :func:`brainunit.fft.ifftn`: Computes a multidimensional inverse of discrete
            Fourier transform.

    Examples:
        ``brainunit.fft.ifft`` computes the transform along ``axis -1`` by default.

        >>> x = jnp.array([[3, 1, 4, 6],
        ...                [2, 5, 7, 1]])
        >>> brainunit.fft.ifft(x)
        Array([[ 3.5 +0.j  , -0.25-1.25j,  0.  +0.j  , -0.25+1.25j],
              [ 3.75+0.j  , -1.25+1.j  ,  0.75+0.j  , -1.25-1.j  ]],      dtype=complex64)

        When ``n=5``, dimension of the transform along axis -1 will be ``5`` and
        dimension along other axes will be the same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   print(brainunit.fft.ifft(x, n=5))
        [[ 2.8 +0.j   -0.96-0.04j  1.06+0.5j   1.06-0.5j  -0.96+0.04j]
         [ 3.  +0.j   -0.59+1.66j  0.09-0.55j  0.09+0.55j -0.59-1.66j]]

        When ``n=3`` and ``axis=0``, dimension of the transform along ``axis 0`` will
        be ``3`` and dimension along other axes will be same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   print(brainunit.fft.ifft(x, n=3, axis=0))
        [[ 1.67+0.j    2.  +0.j    3.67+0.j    2.33+0.j  ]
         [ 0.67+0.58j -0.5 +1.44j  0.17+2.02j  1.83+0.29j]
         [ 0.67-0.58j -0.5 -1.44j  0.17-2.02j  1.83-0.29j]]
    """
    return _fun_change_unit_unary(jnpfft.ifft,
                                  lambda u: u / second,
                                  a, n=n, axis=axis, norm=norm)


@unit_change(lambda u: u / second)
def irfft(
    a: Union[Quantity, jax.typing.ArrayLike],
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Compute a real-valued one-dimensional inverse discrete Fourier transform.

    Brainunit implementation of :func:`numpy.fft.irfft`.

    Args:
        a: input array.
        n: int. Specifies the dimension of the result along ``axis``. If not specified,
            ``n = 2*(m-1)``, where ``m`` is the dimension of ``a`` along ``axis``.
        axis: int, default=-1. Specifies the axis along which the transform is computed.
            If not specified, the transform is computed along axis -1.
        norm: string. The normalization mode. "backward", "ortho" and "forward" are
            supported.

    Returns:
        A real-valued array containing the one-dimensional inverse discrete Fourier
        transform of ``a``, with a dimension of ``n`` along ``axis``.

    See also:
        - :func:`brainunit.fft.ifft`: Computes a one-dimensional inverse discrete
            Fourier transform.
        - :func:`brainunit.fft.irfft`: Computes a one-dimensional inverse discrete
            Fourier transform for real input.
        - :func:`brainunit.fft.rfftn`: Computes a multidimensional discrete Fourier
            transform for real input.
        - :func:`brainunit.fft.irfftn`: Computes a multidimensional inverse discrete
            Fourier transform for real input.

    Examples:
        ``brainunit.fft.rfft`` computes the transform along ``axis -1`` by default.

        >>> x = jnp.array([[1, 3, 5],
        ...                [2, 4, 6]])
        >>> brainunit.fft.irfft(x)
        Array([[ 3., -1.,  0., -1.],
               [ 4., -1.,  0., -1.]], dtype=float32)

        When ``n=3``, dimension of the transform along axis -1 will be ``3`` and
        dimension along other axes will be the same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.irfft(x, n=3)
        Array([[ 2.33, -0.67, -0.67],
               [ 3.33, -0.67, -0.67]], dtype=float32)

        When ``n=4`` and ``axis=0``, dimension of the transform along ``axis 0`` will
        be ``4`` and dimension along other axes will be same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.irfft(x, n=4, axis=0)
        Array([[ 1.25,  2.75,  4.25],
               [ 0.25,  0.75,  1.25],
               [-0.75, -1.25, -1.75],
               [ 0.25,  0.75,  1.25]], dtype=float32)
    """
    return _fun_change_unit_unary(jnpfft.irfft,
                                  lambda u: u / second,
                                  a, n=n, axis=axis, norm=norm)


# return original unit * (time unit ^ n)
# --------------------------------------

@unit_change(lambda u: u * (second ** 2))
def fft2(
    a: Union[Quantity, jax.typing.ArrayLike],
    s: Shape | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Compute a two-dimensional discrete Fourier transform along given axes.

    Brainunit implementation of :func:`numpy.fft.fft2`.

    Args:
        a: input array. Must have ``a.ndim >= 2``.
        s: optional length-2 sequence of integers. Specifies the size of the output
            along each specified axis. If not specified, it will default to the size
            of ``a`` along the specified ``axes``.
        axes: optional length-2 sequence of integers, default=(-2,-1). Specifies the
            axes along which the transform is computed.
        norm: string, default="backward". The normalization mode. "backward", "ortho"
            and "forward" are supported.

    Returns:
        An array containing the two-dimensional discrete Fourier transform of ``a``
        along given ``axes``.

    See also:
        - :func:`brainunit.fft.fft`: Computes a one-dimensional discrete Fourier
            transform.
        - :func:`brainunit.fft.fftn`: Computes a multidimensional discrete Fourier
            transform.
        - :func:`brainunit.fft.ifft2`: Computes a two-dimensional inverse discrete
            Fourier transform.

    Examples:
        ``brainunit.fft.fft2`` computes the transform along the last two axes by default.

        >>> x = jnp.array([[[1, 3],
        ...                 [2, 4]],
        ...                [[5, 7],
        ...                 [6, 8]]])
        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.fft2(x)
        Array([[[10.+0.j, -4.+0.j],
                [-2.+0.j,  0.+0.j]],
        <BLANKLINE>
               [[26.+0.j, -4.+0.j],
                [-2.+0.j,  0.+0.j]]], dtype=complex64)

        When ``s=[2, 3]``, dimension of the transform along ``axes (-2, -1)`` will be
        ``(2, 3)`` and dimension along other axes will be the same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.fft2(x, s=[2, 3])
        Array([[[10.  +0.j  , -0.5 -6.06j, -0.5 +6.06j],
                [-2.  +0.j  , -0.5 +0.87j, -0.5 -0.87j]],
        <BLANKLINE>
               [[26.  +0.j  ,  3.5-12.99j,  3.5+12.99j],
                [-2.  +0.j  , -0.5 +0.87j, -0.5 -0.87j]]], dtype=complex64)

        When ``s=[2, 3]`` and ``axes=(0, 1)``, shape of the transform along
        ``axes (0, 1)`` will be ``(2, 3)`` and dimension along other axes will be
        same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.fft2(x, s=[2, 3], axes=(0, 1))
        Array([[[14. +0.j  , 22. +0.j  ],
                [ 2. -6.93j,  4.-10.39j],
                [ 2. +6.93j,  4.+10.39j]],
        <BLANKLINE>
               [[-8. +0.j  , -8. +0.j  ],
                [-2. +3.46j, -2. +3.46j],
                [-2. -3.46j, -2. -3.46j]]], dtype=complex64)

        ``jnp.fft.ifft2`` can be used to reconstruct ``x`` from the result of
        ``jnp.fft.fft2``.

        >>> x_fft2 = brainunit.fft.fft2(x)
        >>> jnp.allclose(x, brainunit.fft.ifft2(x_fft2))
        Array(True, dtype=bool)
    """
    return _fun_change_unit_unary(jnpfft.fft2,
                                  lambda u: u * (second ** 2),
                                  a, s=s, axes=axes, norm=norm)


@unit_change(lambda u: u * (second ** 2))
def rfft2(
    a: Union[Quantity, jax.typing.ArrayLike],
    s: Shape | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Compute a two-dimensional discrete Fourier transform of a real-valued array.

    Brainunit implementation of :func:`numpy.fft.rfft2`.

    Args:
        a: real-valued input array. Must have ``a.ndim >= 2``.
        s: optional length-2 sequence of integers. Specifies the effective size of the
            output along each specified axis. If not specified, it will default to the
            dimension of input along ``axes``.
        axes: optional length-2 sequence of integers, default=(-2,-1). Specifies the
            axes along which the transform is computed.
        norm: string, default="backward". The normalization mode. "backward", "ortho"
            and "forward" are supported.

    Returns:
        An array containing the two-dimensional discrete Fourier transform of ``a``.
        The size of the output along the axis ``axes[1]`` is ``(s[1]/2)+1``, if ``s[1]``
        is even and ``(s[1]+1)/2``, if ``s[1]`` is odd. The size of the output along
        the axis ``axes[0]`` is ``s[0]``.

    See also:
        - :func:`brainunit.fft.rfft`: Computes a one-dimensional discrete Fourier
            transform of real-valued array.
        - :func:`brainunit.fft.rfftn`: Computes a multidimensional discrete Fourier
            transform of real-valued array.
        - :func:`brainunit.fft.irfft2`: Computes a real-valued two-dimensional inverse
            discrete Fourier transform.

    Examples:
        ``brainunit.fft.rfft2`` computes the transform along the last two axes by default.

        >>> x = jnp.array([[[1, 3, 5],
        ...                 [2, 4, 6]],
        ...                [[7, 9, 11],
        ...                 [8, 10, 12]]])
        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.rfft2(x)
        Array([[[21.+0.j  , -6.+3.46j],
                [-3.+0.j  ,  0.+0.j  ]],
        <BLANKLINE>
               [[57.+0.j  , -6.+3.46j],
                [-3.+0.j  ,  0.+0.j  ]]], dtype=complex64)

        When ``s=[2, 4]``, dimension of the transform along ``axis -2`` will be
        ``2``, along ``axis -1`` will be ``(4/2)+1) = 3`` and dimension along other
        axes will be the same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.rfft2(x, s=[2, 4])
        Array([[[21. +0.j, -8. -7.j,  7. +0.j],
                [-3. +0.j,  0. +1.j, -1. +0.j]],
        <BLANKLINE>
               [[57. +0.j, -8.-19.j, 19. +0.j],
                [-3. +0.j,  0. +1.j, -1. +0.j]]], dtype=complex64)

        When ``s=[3, 5]`` and ``axes=(0, 1)``, shape of the transform along ``axis 0``
        will be ``3``, along ``axis 1`` will be ``(5+1)/2 = 3`` and dimension along
        other axes will be same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.rfft2(x, s=[3, 5], axes=(0, 1))
        Array([[[ 18.   +0.j  ,  26.   +0.j  ,  34.   +0.j  ],
                [ 11.09 -9.51j,  16.33-13.31j,  21.56-17.12j],
                [ -0.09 -5.88j,   0.67 -8.23j,   1.44-10.58j]],
        <BLANKLINE>
              [[ -4.5 -12.99j,  -2.5 -16.45j,  -0.5 -19.92j],
                [ -9.71 -6.3j , -10.05 -9.52j, -10.38-12.74j],
                [ -4.95 +0.72j,  -5.78 -0.2j ,  -6.61 -1.12j]],
        <BLANKLINE>
              [[ -4.5 +12.99j,  -2.5 +16.45j,  -0.5 +19.92j],
                [  3.47+10.11j,   6.43+11.42j,   9.38+12.74j],
                [  3.19 +1.63j,   4.4  +1.38j,   5.61 +1.12j]]], dtype=complex64)
    """
    return _fun_change_unit_unary(jnpfft.rfft2,
                                  lambda u: u * (second ** 2),
                                  a, s=s, axes=axes, norm=norm)


@set_module_as('brainunit.fft')
def fftn(
    a: Union[Quantity, jax.typing.ArrayLike],
    s: Shape | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    r"""Compute a multidimensional discrete Fourier transform along given axes.

    Brainunit implementation of :func:`numpy.fft.fftn`.

    Args:
        a: input array
        s: sequence of integers. Specifies the shape of the result. If not specified,
            it will default to the shape of ``a`` along the specified ``axes``.
        axes: sequence of integers, default=None. Specifies the axes along which the
            transform is computed.
        norm: string. The normalization mode. "backward", "ortho" and "forward" are
            supported.

    Returns:
        An array containing the multidimensional discrete Fourier transform of ``a``.

    See also:
        - :func:`brainunit.fft.fft`: Computes a one-dimensional discrete Fourier
            transform.
        - :func:`brainunit.fft.ifft`: Computes a one-dimensional inverse discrete
            Fourier transform.
        - :func:`brainunit.fft.ifftn`: Computes a multidimensional inverse discrete
            Fourier transform.

    Examples:
        ``brainunit.fft.fftn`` computes the transform along all the axes by default when
        ``axes`` argument is ``None``.

        >>> x = jnp.array([[1, 2, 5, 6],
        ...                [4, 1, 3, 7],
        ...                [5, 9, 2, 1]])
        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.fftn(x)
        Array([[ 46.  +0.j  ,   0.  +2.j  ,  -6.  +0.j  ,   0.  -2.j  ],
               [ -2.  +1.73j,   6.12+6.73j,   0.  -1.73j, -18.12-3.27j],
               [ -2.  -1.73j, -18.12+3.27j,   0.  +1.73j,   6.12-6.73j]],      dtype=complex64)

        When ``s=[2]``, dimension of the transform along ``axis -1`` will be ``2``
        and dimension along other axes will be the same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   print(brainunit.fft.fftn(x, s=[2]))
        [[ 3.+0.j -1.+0.j]
         [ 5.+0.j  3.+0.j]
         [14.+0.j -4.+0.j]]

        When ``s=[2]`` and ``axes=[0]``, dimension of the transform along ``axis 0``
        will be ``2`` and dimension along other axes will be same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   print(brainunit.fft.fftn(x, s=[2], axes=[0]))
        [[ 5.+0.j  3.+0.j  8.+0.j 13.+0.j]
         [-3.+0.j  1.+0.j  2.+0.j -1.+0.j]]

        When ``s=[2, 3]``, shape of the transform will be ``(2, 3)``.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   print(brainunit.fft.fftn(x, s=[2, 3]))
        [[16. +0.j   -0.5+4.33j -0.5-4.33j]
         [ 0. +0.j   -4.5+0.87j -4.5-0.87j]]

        ``brainunit.fft.ifftn`` can be used to reconstruct ``x`` from the result of
        ``brainunit.fft.fftn``.

        >>> x_fftn = brainunit.fft.fftn(x)
        >>> jnp.allclose(x, brainunit.fft.ifftn(x_fftn))
        Array(True, dtype=bool)
    """
    n = _calculate_fftn_dimension(a.ndim, axes)
    _unit_change_fun = lambda u: u * (second ** n)
    # TODO: may cause computation overhead?
    fftn._unit_change_fun = _unit_change_fun
    return _fun_change_unit_unary(jnpfft.fftn,
                                  _unit_change_fun,
                                  a, s=s, axes=axes, norm=norm)


@set_module_as('brainunit.fft')
def rfftn(
    a: Union[Quantity, jax.typing.ArrayLike],
    s: Shape | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Compute a multidimensional discrete Fourier transform of a real-valued array.

    JAX implementation of :func:`numpy.fft.rfftn`.

    Args:
        a: real-valued input array.
        s: optional sequence of integers. Controls the effective size of the input
            along each specified axis. If not specified, it will default to the
            dimension of input along ``axes``.
        axes: optional sequence of integers, default=None. Specifies the axes along
            which the transform is computed. If not specified, the transform is computed
            along the last ``len(s)`` axes. If neither ``axes`` nor ``s`` is specified,
            the transform is computed along all the axes.
        norm: string, default="backward". The normalization mode. "backward", "ortho"
            and "forward" are supported.

    Returns:
        An array containing the multidimensional discrete Fourier transform of ``a``
        having size specified in ``s`` along the axes ``axes`` except along the axis
        ``axes[-1]``. The size of the output along the axis ``axes[-1]`` is
        ``s[-1]//2+1``.

    See also:
        - :func:`brainunit.fft.rfft`: Computes a one-dimensional discrete Fourier
            transform of real-valued array.
        - :func:`brainunit.fft.rfft2`: Computes a two-dimensional discrete Fourier
            transform of real-valued array.
        - :func:`brainunit.fft.irfftn`: Computes a real-valued multidimensional inverse
            discrete Fourier transform.

    Examples:
        >>> x = jnp.array([[[1, 3, 5],
        ...                 [2, 4, 6]],
        ...                [[7, 9, 11],
        ...                 [8, 10, 12]]])
        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.rfftn(x)
        Array([[[ 78.+0.j  , -12.+6.93j],
                [ -6.+0.j  ,   0.+0.j  ]],
        <BLANKLINE>
               [[-36.+0.j  ,   0.+0.j  ],
                [  0.+0.j  ,   0.+0.j  ]]], dtype=complex64)

        When ``s=[3, 3, 4]``,  size of the transform along ``axes (-3, -2)`` will
        be (3, 3), and along ``axis -1`` will be ``4//2+1 = 3`` and size along
        other axes will be the same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.rfftn(x, s=[3, 3, 4])
        Array([[[ 78.   +0.j  , -16.  -26.j  ,  26.   +0.j  ],
                [ 15.  -36.37j, -16.12 +1.93j,   5.  -12.12j],
                [ 15.  +36.37j,   8.12-11.93j,   5.  +12.12j]],
        <BLANKLINE>
               [[ -7.5 -49.36j, -20.45 +9.43j,  -2.5 -16.45j],
                [-25.5  -7.79j,  -0.6 +11.96j,  -8.5  -2.6j ],
                [ 19.5 -12.99j,  -8.33 -6.5j ,   6.5  -4.33j]],
        <BLANKLINE>
               [[ -7.5 +49.36j,  12.45 -4.43j,  -2.5 +16.45j],
                [ 19.5 +12.99j,   0.33 -6.5j ,   6.5  +4.33j],
                [-25.5  +7.79j,   4.6  +5.04j,  -8.5  +2.6j ]]], dtype=complex64)

        When ``s=[3, 5]`` and ``axes=(0, 1)``, size of the transform along ``axis 0``
        will be ``3``, along ``axis 1`` will be ``5//2+1 = 3`` and dimension along
        other axes will be same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.rfftn(x, s=[3, 5], axes=[0, 1])
        Array([[[ 18.   +0.j  ,  26.   +0.j  ,  34.   +0.j  ],
                [ 11.09 -9.51j,  16.33-13.31j,  21.56-17.12j],
                [ -0.09 -5.88j,   0.67 -8.23j,   1.44-10.58j]],
        <BLANKLINE>
               [[ -4.5 -12.99j,  -2.5 -16.45j,  -0.5 -19.92j],
                [ -9.71 -6.3j , -10.05 -9.52j, -10.38-12.74j],
                [ -4.95 +0.72j,  -5.78 -0.2j ,  -6.61 -1.12j]],
        <BLANKLINE>
               [[ -4.5 +12.99j,  -2.5 +16.45j,  -0.5 +19.92j],
                [  3.47+10.11j,   6.43+11.42j,   9.38+12.74j],
                [  3.19 +1.63j,   4.4  +1.38j,   5.61 +1.12j]]], dtype=complex64)

        For 1-D input:

        >>> x1 = jnp.array([1, 2, 3, 4])
        >>> brainunit.fft.rfftn(x1)
        Array([10.+0.j, -2.+2.j, -2.+0.j], dtype=complex64)
    """
    n = _calculate_fftn_dimension(a.ndim, axes)
    _unit_change_fun = lambda u: u * (second ** n)
    # TODO: may cause computation overhead?
    rfftn._unit_change_fun = _unit_change_fun
    return _fun_change_unit_unary(jnpfft.rfftn,
                                  _unit_change_fun,
                                  a, s=s, axes=axes, norm=norm)


# return original unit / (time unit ^ n) (inverse)
# -----------------------------------------------

@unit_change(lambda u: u / (second ** 2))
def ifft2(
    a: Union[Quantity, jax.typing.ArrayLike],
    s: Shape | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Compute a two-dimensional inverse discrete Fourier transform.

    Brainunit implementation of :func:`numpy.fft.ifft2`.

    Args:
        a: input array. Must have ``a.ndim >= 2``.
        s: optional length-2 sequence of integers. Specifies the size of the output
          in each specified axis. If not specified, it will default to the size of
          ``a`` along the specified ``axes``.
        axes: optional length-2 sequence of integers, default=(-2,-1). Specifies the
          axes along which the transform is computed.
        norm: string, default="backward". The normalization mode. "backward", "ortho"
          and "forward" are supported.

    Returns:
        An array containing the two-dimensional inverse discrete Fourier transform
        of ``a`` along given ``axes``.

    See also:
        - :func:`brainunit.fft.ifft`: Computes a one-dimensional inverse discrete
          Fourier transform.
        - :func:`brainunit.fft.ifftn`: Computes a multidimensional inverse discrete
          Fourier transform.
        - :func:`brainunit.fft.fft2`: Computes a two-dimensional discrete Fourier
          transform.

    Examples:
        ``brainunit.fft.ifft2`` computes the transform along the last two axes by default.

        >>> x = jnp.array([[[1, 3],
        ...                 [2, 4]],
        ...                [[5, 7],
        ...                 [6, 8]]])
        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.ifft2(x)
        Array([[[ 2.5+0.j, -1. +0.j],
                [-0.5+0.j,  0. +0.j]],
        <BLANKLINE>
               [[ 6.5+0.j, -1. +0.j],
                [-0.5+0.j,  0. +0.j]]], dtype=complex64)

        When ``s=[2, 3]``, dimension of the transform along ``axes (-2, -1)`` will be
        ``(2, 3)`` and dimension along other axes will be the same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.ifft2(x, s=[2, 3])
        Array([[[ 1.67+0.j  , -0.08+1.01j, -0.08-1.01j],
                [-0.33+0.j  , -0.08-0.14j, -0.08+0.14j]],
        <BLANKLINE>
               [[ 4.33+0.j  ,  0.58+2.17j,  0.58-2.17j],
                [-0.33+0.j  , -0.08-0.14j, -0.08+0.14j]]], dtype=complex64)

        When ``s=[2, 3]`` and ``axes=(0, 1)``, shape of the transform along
        ``axes (0, 1)`` will be ``(2, 3)`` and dimension along other axes will be
        same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.ifft2(x, s=[2, 3], axes=(0, 1))
        Array([[[ 2.33+0.j  ,  3.67+0.j  ],
                [ 0.33+1.15j,  0.67+1.73j],
                [ 0.33-1.15j,  0.67-1.73j]],
        <BLANKLINE>
               [[-1.33+0.j  , -1.33+0.j  ],
                [-0.33-0.58j, -0.33-0.58j],
                [-0.33+0.58j, -0.33+0.58j]]], dtype=complex64)
    """
    return _fun_change_unit_unary(jnpfft.ifft2,
                                  lambda u: u / (second ** 2),
                                  a, s=s, axes=axes, norm=norm)


@unit_change(lambda u: u / (second ** 2))
def irfft2(
    a: Union[Quantity, jax.typing.ArrayLike],
    s: Shape | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Compute a real-valued two-dimensional inverse discrete Fourier transform.

    Brainunit implementation of :func:`numpy.fft.irfft2`.

    Args:
        a: input array. Must have ``a.ndim >= 2``.
        s: optional length-2 sequence of integers. Specifies the size of the output
            in each specified axis. If not specified, the dimension of output along
            axis ``axes[1]`` is ``2*(m-1)``, ``m`` is the size of input along axis
            ``axes[1]`` and the dimension along other axes will be the same as that of
            input.
        axes: optional length-2 sequence of integers, default=(-2,-1). Specifies the
            axes along which the transform is computed.
        norm: string, default="backward". The normalization mode. "backward", "ortho"
            and "forward" are supported.

    Returns:
        A real-valued array containing the two-dimensional inverse discrete Fourier
        transform of ``a``.

    See also:
        - :func:`brainunit.fft.rfft2`: Computes a two-dimensional discrete Fourier
            transform of a real-valued array.
        - :func:`brainunit.fft.irfft`: Computes a real-valued one-dimensional inverse
            discrete Fourier transform.
        - :func:`brainunit.fft.irfftn`: Computes a real-valued multidimensional inverse
            discrete Fourier transform.

    Examples:
        ``brainunit.fft.irfft2`` computes the transform along the last two axes by default.

        >>> x = jnp.array([[[1, 3, 5],
        ...                 [2, 4, 6]],
        ...                [[7, 9, 11],
        ...                 [8, 10, 12]]])
        >>> brainunit.fft.irfft2(x)
        Array([[[ 3.5, -1. ,  0. , -1. ],
                [-0.5,  0. ,  0. ,  0. ]],
        <BLANKLINE>
               [[ 9.5, -1. ,  0. , -1. ],
                [-0.5,  0. ,  0. ,  0. ]]], dtype=float32)

        When ``s=[3, 3]``, dimension of the transform along ``axes (-2, -1)`` will be
        ``(3, 3)`` and dimension along other axes will be the same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.irfft2(x, s=[3, 3])
        Array([[[ 1.89, -0.44, -0.44],
                [ 0.22, -0.78,  0.56],
                [ 0.22,  0.56, -0.78]],
        <BLANKLINE>
               [[ 5.89, -0.44, -0.44],
                [ 1.22, -1.78,  1.56],
                [ 1.22,  1.56, -1.78]]], dtype=float32)

        When ``s=[2, 3]`` and ``axes=(0, 1)``, shape of the transform along
        ``axes (0, 1)`` will be ``(2, 3)`` and dimension along other axes will be
        same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.irfft2(x, s=[2, 3], axes=(0, 1))
        Array([[[ 4.67,  6.67,  8.67],
                [-0.33, -0.33, -0.33],
                [-0.33, -0.33, -0.33]],
        <BLANKLINE>
               [[-3.  , -3.  , -3.  ],
                [ 0.  ,  0.  ,  0.  ],
                [ 0.  ,  0.  ,  0.  ]]], dtype=float32)
    """
    return _fun_change_unit_unary(jnpfft.irfft2,
                                  lambda u: u / (second ** 2),
                                  a, s=s, axes=axes, norm=norm)


@set_module_as('brainunit.fft')
def ifftn(
    a: Union[Quantity, jax.typing.ArrayLike],
    s: Shape | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    r"""Compute a multidimensional inverse discrete Fourier transform.

    Brainunit implementation of :func:`numpy.fft.ifftn`.

    Args:
        a: input array
        s: sequence of integers. Specifies the shape of the result. If not specified,
            it will default to the shape of ``a`` along the specified ``axes``.
        axes: sequence of integers, default=None. Specifies the axes along which the
            transform is computed. If None, computes the transform along all the axes.
        norm: string. The normalization mode. "backward", "ortho" and "forward" are
            supported.

    Returns:
        An array containing the multidimensional inverse discrete Fourier transform
        of ``a``.

    See also:
        - :func:`brainunit.fft.fftn`: Computes a multidimensional discrete Fourier
          transform.
        - :func:`brainunit.fft.fft`: Computes a one-dimensional discrete Fourier
          transform.
        - :func:`brainunit.fft.ifft`: Computes a one-dimensional inverse discrete
          Fourier transform.

    Examples:
        ``brainunit.fft.ifftn`` computes the transform along all the axes by default when
        ``axes`` argument is ``None``.

        >>> x = jnp.array([[1, 2, 5, 3],
        ...                [4, 1, 2, 6],
        ...                [5, 3, 2, 1]])
        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   print(brainunit.fft.ifftn(x))
        [[ 2.92+0.j    0.08-0.33j  0.25+0.j    0.08+0.33j]
         [-0.08+0.14j -0.04-0.03j  0.  -0.29j -1.05-0.11j]
         [-0.08-0.14j -1.05+0.11j  0.  +0.29j -0.04+0.03j]]

        When ``s=[3]``, dimension of the transform along ``axis -1`` will be ``3``
        and dimension along other axes will be the same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   print(brainunit.fft.ifftn(x, s=[3]))
        [[ 2.67+0.j   -0.83-0.87j -0.83+0.87j]
         [ 2.33+0.j    0.83-0.29j  0.83+0.29j]
         [ 3.33+0.j    0.83+0.29j  0.83-0.29j]]

        When ``s=[2]`` and ``axes=[0]``, dimension of the transform along ``axis 0``
        will be ``2`` and dimension along other axes will be same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   print(brainunit.fft.ifftn(x, s=[2], axes=[0]))
        [[ 2.5+0.j  1.5+0.j  3.5+0.j  4.5+0.j]
         [-1.5+0.j  0.5+0.j  1.5+0.j -1.5+0.j]]

        When ``s=[2, 3]``, shape of the transform will be ``(2, 3)``.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   print(brainunit.fft.ifftn(x, s=[2, 3]))
        [[ 2.5 +0.j    0.  -0.58j  0.  +0.58j]
         [ 0.17+0.j   -0.83-0.29j -0.83+0.29j]]
    """
    n = _calculate_fftn_dimension(a.ndim, axes)
    _unit_change_fun = lambda u: u / (second ** n)
    # TODO: may cause computation overhead?
    ifftn._unit_change_fun = _unit_change_fun
    return _fun_change_unit_unary(jnpfft.ifftn,
                                  _unit_change_fun,
                                  a, s=s, axes=axes, norm=norm)


@set_module_as('brainunit.fft')
def irfftn(
    a: Union[Quantity, jax.typing.ArrayLike],
    s: Shape | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Compute a real-valued multidimensional inverse discrete Fourier transform.

    Brainunit implementation of :func:`numpy.fft.irfftn`.

    Args:
        a: input array.
        s: optional sequence of integers. Specifies the size of the output in each
            specified axis. If not specified, the dimension of output along axis
            ``axes[-1]`` is ``2*(m-1)``, ``m`` is the size of input along axis ``axes[-1]``
            and the dimension along other axes will be the same as that of input.
        axes: optional sequence of integers, default=None. Specifies the axes along
            which the transform is computed. If not specified, the transform is computed
            along the last ``len(s)`` axes. If neither ``axes`` nor ``s`` is specified,
            the transform is computed along all the axes.
        norm: string, default="backward". The normalization mode. "backward", "ortho"
            and "forward" are supported.

    Returns:
        A real-valued array containing the multidimensional inverse discrete Fourier
        transform of ``a`` with size ``s`` along specified ``axes``, and the same as
        the input along other axes.

    See also:
        - :func:`brainunit.fft.rfftn`: Computes a multidimensional discrete Fourier
            transform of a real-valued array.
        - :func:`brainunit.fft.irfft`: Computes a real-valued one-dimensional inverse
            discrete Fourier transform.
        - :func:`brainunit.fft.irfft2`: Computes a real-valued two-dimensional inverse
            discrete Fourier transform.

    Examples:
        ``brainunit.fft.irfftn`` computes the transform along all the axes by default.

        >>> x = jnp.array([[[1, 3, 5],
        ...                 [2, 4, 6]],
        ...                [[7, 9, 11],
        ...                 [8, 10, 12]]])
        >>> brainunit.fft.irfftn(x)
        Array([[[ 6.5, -1. ,  0. , -1. ],
                [-0.5,  0. ,  0. ,  0. ]],
        <BLANKLINE>
               [[-3. ,  0. ,  0. ,  0. ],
                [ 0. ,  0. ,  0. ,  0. ]]], dtype=float32)

        When ``s=[3, 4]``, size of the transform along ``axes (-2, -1)`` will be
        ``(3, 4)`` and size along other axes will be the same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.irfftn(x, s=[3, 4])
        Array([[[ 2.33, -0.67,  0.  , -0.67],
                [ 0.33, -0.74,  0.  ,  0.41],
                [ 0.33,  0.41,  0.  , -0.74]],
        <BLANKLINE>
               [[ 6.33, -0.67,  0.  , -0.67],
                [ 1.33, -1.61,  0.  ,  1.28],
                [ 1.33,  1.28,  0.  , -1.61]]], dtype=float32)

        When ``s=[3]`` and ``axes=[0]``, size of the transform along ``axes 0`` will
        be ``3`` and dimension along other axes will be same as that of input.

        >>> with jnp.printoptions(precision=2, suppress=True):
        ...   brainunit.fft.irfftn(x, s=[3], axes=[0])
        Array([[[ 5.,  7.,  9.],
                [ 6.,  8., 10.]],
        <BLANKLINE>
               [[-2., -2., -2.],
                [-2., -2., -2.]],
        <BLANKLINE>
               [[-2., -2., -2.],
                [-2., -2., -2.]]], dtype=float32)
    """
    n = _calculate_fftn_dimension(a.ndim, axes)
    _unit_change_fun = lambda u: u / (second ** n)
    # TODO: may cause computation overhead?
    irfftn._unit_change_fun = _unit_change_fun
    return _fun_change_unit_unary(jnpfft.irfftn,
                                  _unit_change_fun,
                                  a, s=s, axes=axes, norm=norm)


# return frequency unit
# ---------------------

_time_freq_map = {
    0: (u.second, u.hertz),
    -24: (u.ysecond, u.Yhertz),
    -21: (u.zsecond, u.Zhertz),
    -18: (u.asecond, u.Ehertz),
    -15: (u.fsecond, u.Phertz),
    -12: (u.psecond, u.Thertz),
    -9: (u.nsecond, u.Ghertz),
    -6: (u.usecond, u.Mhertz),
    -3: (u.msecond, u.khertz),
    -2: (u.csecond, u.hhertz),
    -1: (u.dsecond, u.dahertz),
    1: (u.dasecond, u.dhertz),
    2: (u.hsecond, u.chertz),
    3: (u.ksecond, u.mhertz),
    6: (u.Msecond, u.uhertz),
    9: (u.Gsecond, u.nhertz),
    12: (u.Tsecond, u.phertz),
    15: (u.Psecond, u.fhertz),
    18: (u.Esecond, u.ahertz),
    21: (u.Zsecond, u.zhertz),
    24: (u.Ysecond, u.yhertz),
}

def _find_closet_scale(scale):
    values = list(_time_freq_map.keys())

    diff = np.abs(np.array(values) - scale)

    # check if all > 3, return scale
    if all(diff > 3):
        return scale

    # find the closet index
    closet_index = diff.argmin()

    return values[closet_index]


@set_module_as('brainunit.fft')
def fftfreq(
    n: int,
    d: Union[Quantity, jax.typing.ArrayLike] = 1.0,
    *,
    dtype: jax.typing.DTypeLike | None = None,
    device: xla_client.Device | jax.sharding.Sharding | None = None,
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Return sample frequencies for the discrete Fourier transform.

    Brainunit implementation of :func:`numpy.fft.fftfreq`. Returns frequencies appropriate
    for use with the outputs of :func:`~brainunit.fft.fft` and :func:`~brainunit.fft.ifft`.

    Args:
        n: length of the FFT window
        d: optional scalar sample spacing (default: 1.0)
        dtype: optional dtype of returned frequencies. If not specified, JAX's default
            floating point dtype will be used.
        device: optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
            to which the created array will be committed.

    Returns:
        Array of sample frequencies, length ``n``.

    See also:
    - :func:`brainunit.fft.rfftfreq`: frequencies for use with
      :func:`~brainunit.fft.rfft` and :func:`~brainunit.fft.irfft`.
    """
    if isinstance(d, Quantity):
        assert d.dim == second.dim, f"Expected time unit, got {d.unit}"
        time_scale = _find_closet_scale(d.unit.scale)
        try:
            time_unit, freq_unit = _time_freq_map[time_scale]
        except KeyError:
            time_unit = d.unit
            freq_unit_scale = -d.unit.scale
            freq_unit = Unit.create(get_or_create_dimension(s=-1),
                                    name=f'10^{freq_unit_scale} hertz',
                                    dispname=f'10^{freq_unit_scale} Hz',
                                    scale=freq_unit_scale,)
        if sys.version_info >= (3, 10):
            return Quantity(jnpfft.fftfreq(n, d.to_decimal(time_unit), dtype=dtype, device=device), unit=freq_unit)
        else:
            # noinspection PyUnresolvedReferences
            return Quantity(jnpfft.fftfreq(n, d.to_decimal(time_unit), dtype=dtype), unit=freq_unit)
    if sys.version_info >= (3, 10):
        return jnpfft.fftfreq(n, d, dtype=dtype, device=device)
    else:
        # noinspection PyUnresolvedReferences
        return jnpfft.fftfreq(n, d, dtype=dtype)



@set_module_as('brainunit.fft')
def rfftfreq(
    n: int,
    d: Union[Quantity, jax.typing.ArrayLike] = 1.0,
    *,
    dtype: jax.typing.DTypeLike | None = None,
    device: xla_client.Device | jax.sharding.Sharding | None = None,
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Return sample frequencies for the discrete Fourier transform.

    Brainunit implementation of :func:`numpy.fft.fftfreq`. Returns frequencies appropriate
    for use with the outputs of :func:`~brainunit.fft.rfft` and
    :func:`~brainunit.fft.irfft`.

    Args:
        n: length of the FFT window
        d: optional scalar sample spacing (default: 1.0)
        dtype: optional dtype of returned frequencies. If not specified, JAX's default
            floating point dtype will be used.
        device: optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding`
            to which the created array will be committed.

    Returns:
        Array of sample frequencies, length ``n // 2 + 1``.

    See also:
    - :func:`brainunit.fft.fftfreq`: frequencies for use with
      :func:`~brainunit.fft.fft` and :func:`~brainunit.fft.ifft`.
    """
    if isinstance(d, Quantity):
        assert d.dim == second.dim, f"Expected time unit, got {d.unit}"
        time_scale = _find_closet_scale(d.unit.scale)
        try:
            time_unit, freq_unit = _time_freq_map[time_scale]
        except KeyError:
            time_unit = d.unit
            freq_unit_scale = -d.unit.scale
            freq_unit = Unit.create(get_or_create_dimension(s=-1),
                                    name=f'10^{freq_unit_scale} hertz',
                                    dispname=f'10^{freq_unit_scale} Hz',
                                    scale=freq_unit_scale, )
        if sys.version_info >= (3, 10):
            return Quantity(jnpfft.rfftfreq(n, d.to_decimal(time_unit), dtype=dtype, device=device), unit=freq_unit)
        else:
            # noinspection PyUnresolvedReferences
            return Quantity(jnpfft.rfftfreq(n, d.to_decimal(time_unit), dtype=dtype), unit=freq_unit)
    if sys.version_info >= (3, 10):
        return jnpfft.rfftfreq(n, d, dtype=dtype, device=device)
    else:
        # noinspection PyUnresolvedReferences
        return jnpfft.rfftfreq(n, d, dtype=dtype)
