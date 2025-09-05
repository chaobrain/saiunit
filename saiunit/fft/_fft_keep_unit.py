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


@set_module_as('saiunit.fft')
def fftshift(
    x: Union[Quantity, jax.typing.ArrayLike],
    axes: None | int | Sequence[int] = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Shift zero-frequency fft component to the center of the spectrum.

    saiunit implementation of :func:`numpy.fft.fftshift`.

    Args:
        x: N-dimensional quantity or array of frequencies.
        axes: optional integer or sequence of integers specifying which axes to
            shift. If None (default), then shift all axes.

    Returns:
        A shifted copy of ``x``.

    See also:
        - :func:`saiunit.fft.ifftshift`: inverse of ``fftshift``.
        - :func:`saiunit.fft.fftfreq`: generate FFT frequencies.

    Examples:
        Generate FFT frequencies with :func:`~saiunit.fft.fftfreq`:

        >>> import saiunit as u
        >>> freq = u.fft.fftfreq(4, 1 * u.second)
        >>> freq
        ArrayImpl([ 0.  ,  0.25, -0.5 , -0.25], dtype=float32) * hertz

        Use ``fftshift`` to shift the zero-frequency entry to the middle of the array:

        >>> shifted_freq = u.fft.fftshift(freq)
        >>> shifted_freq
        ArrayImpl([-0.5 , -0.25,  0.  ,  0.25], dtype=float32) * hertz

        Unshift with :func:`~saiunit.fft.ifftshift` to recover the original frequencies:

        >>> u.fft.ifftshift(shifted_freq)
        ArrayImpl([ 0.  ,  0.25, -0.5 , -0.25], dtype=float32) * hertz
    """
    return _fun_keep_unit_unary(jnpfft.fftshift, x, axes=axes)


@set_module_as('saiunit.fft')
def ifftshift(
    x: Union[Quantity, jax.typing.ArrayLike],
    axes: None | int | Sequence[int] = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """The inverse of :func:`jax.numpy.fft.fftshift`.

    saiunit implementation of :func:`numpy.fft.ifftshift`.

    Args:
        x: N-dimensional quantity or array of frequencies.
        axes: optional integer or sequence of integers specifying which axes to
            shift. If None (default), then shift all axes.

    Returns:
        A shifted copy of ``x``.

    See also:
        - :func:`saiunit.fft.fftshift`: inverse of ``ifftshift``.
        - :func:`saiunit.fft.fftfreq`: generate FFT frequencies.

    Examples:
        Generate FFT frequencies with :func:`~saiunit.fft.fftfreq`:

        >>> import saiunit as u
        >>> freq = u.fft.fftfreq(4, 1 * u.second)
        >>> freq
        ArrayImpl([ 0.  ,  0.25, -0.5 , -0.25], dtype=float32) * hertz

        Use :func:`~saiunit.fft.fftshift` to shift the zero-frequency entry
        to the middle of the array:

        >>> shifted_freq = u.fft.fftshift(freq)
        >>> shifted_freq
        ArrayImpl([-0.5 , -0.25,  0.  ,  0.25], dtype=float32) * hertz

        Unshift with ``ifftshift`` to recover the original frequencies:

        >>> u.fft.ifftshift(shifted_freq)
        ArrayImpl([ 0.  ,  0.25, -0.5 , -0.25], dtype=float32) * hertz
    """
    return _fun_keep_unit_unary(jnpfft.ifftshift, x, axes=axes)
