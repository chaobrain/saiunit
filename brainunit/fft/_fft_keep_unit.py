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

from typing import Union, Sequence

import jax
from jax.numpy import fft as jnpfft

from .._base import Quantity
from .._misc import set_module_as
from ..math._fun_keep_unit import _fun_keep_unit_unary

__all__ = [
    # keep unit
    'fftshift', 'ifftshift',
]

# keep unit
# ---------


@set_module_as('brainunit.fft')
def fftshift(
    x: Union[Quantity, jax.typing.ArrayLike],
    axes: None | int | Sequence[int] = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Shift zero-frequency fft component to the center of the spectrum.

    Brainunit implementation of :func:`numpy.fft.fftshift`.

    Args:
        x: N-dimensional array array of frequencies.
        axes: optional integer or sequence of integers specifying which axes to
            shift. If None (default), then shift all axes.

    Returns:
        A shifted copy of ``x``.

    See also:
        - :func:`brainunit.fft.ifftshift`: inverse of ``fftshift``.
        - :func:`brainunit.fft.fftfreq`: generate FFT frequencies.

    Examples:
        Generate FFT frequencies with :func:`~brainunit.fft.fftfreq`:

        >>> freq = brainunit.fft.fftfreq(5)
        >>> freq
        Array([ 0. ,  0.2,  0.4, -0.4, -0.2], dtype=float32)

        Use ``fftshift`` to shift the zero-frequency entry to the middle of the array:

        >>> shifted_freq = brainunit.fft.fftshift(freq)
        >>> shifted_freq
        Array([-0.4, -0.2,  0. ,  0.2,  0.4], dtype=float32)

        Unshift with :func:`~brainunit.fft.ifftshift` to recover the original frequencies:

        >>> brainunit.fft.ifftshift(shifted_freq)
        Array([ 0. ,  0.2,  0.4, -0.4, -0.2], dtype=float32)
    """
    return _fun_keep_unit_unary(jnpfft.fftshift, x, axes=axes)


@set_module_as('brainunit.fft')
def ifftshift(
    x: Union[Quantity, jax.typing.ArrayLike],
    axes: None | int | Sequence[int] = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """The inverse of :func:`jax.numpy.fft.fftshift`.

    Brainunit implementation of :func:`numpy.fft.ifftshift`.

    Args:
        x: N-dimensional array array of frequencies.
        axes: optional integer or sequence of integers specifying which axes to
            shift. If None (default), then shift all axes.

    Returns:
        A shifted copy of ``x``.

    See also:
        - :func:`brainunit.fft.fftshift`: inverse of ``ifftshift``.
        - :func:`brainunit.fft.fftfreq`: generate FFT frequencies.

    Examples:
        Generate FFT frequencies with :func:`~brainunit.fft.fftfreq`:

        >>> freq = brainunit.fft.fftfreq(5)
        >>> freq
        Array([ 0. ,  0.2,  0.4, -0.4, -0.2], dtype=float32)

        Use :func:`~brainunit.fft.fftshift` to shift the zero-frequency entry
        to the middle of the array:

        >>> shifted_freq = brainunit.fft.fftshift(freq)
        >>> shifted_freq
        Array([-0.4, -0.2,  0. ,  0.2,  0.4], dtype=float32)

        Unshift with ``ifftshift`` to recover the original frequencies:

        >>> brainunit.fft.ifftshift(shifted_freq)
        Array([ 0. ,  0.2,  0.4, -0.4, -0.2], dtype=float32)
    """
    return _fun_keep_unit_unary(jnpfft.ifftshift, x, axes=axes)
