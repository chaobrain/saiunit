# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as
from saiunit.math._fun_keep_unit import _fun_keep_unit_unary

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
    """Shift zero-frequency FFT component to the center of the spectrum.

    Unit-aware implementation of :func:`numpy.fft.fftshift`.  The unit of
    the input is preserved in the output.

    Parameters
    ----------
    x : Quantity or array_like
        N-dimensional input whose zero-frequency components should be
        shifted to the centre.
    axes : None, int, or sequence of int, optional
        Axes over which to shift.  If *None* (default), all axes are
        shifted.

    Returns
    -------
    Quantity or array_like
        A shifted copy of ``x`` with the same unit.

    See Also
    --------
    saiunit.fft.ifftshift : Inverse of ``fftshift``.
    saiunit.fft.fftfreq : Return sample frequencies for the DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import saiunit.fft as sufft
        >>> freq = sufft.fftfreq(4, 1.0 * u.second)
        >>> shifted = sufft.fftshift(freq)
        >>> recovered = sufft.ifftshift(shifted)
    """
    return _fun_keep_unit_unary(jnpfft.fftshift, x, axes=axes)


@set_module_as('saiunit.fft')
def ifftshift(
    x: Union[Quantity, jax.typing.ArrayLike],
    axes: None | int | Sequence[int] = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Inverse of :func:`saiunit.fft.fftshift`.

    Unit-aware implementation of :func:`numpy.fft.ifftshift`.  The unit of
    the input is preserved in the output.

    Parameters
    ----------
    x : Quantity or array_like
        N-dimensional input whose components should be inverse-shifted.
    axes : None, int, or sequence of int, optional
        Axes over which to shift.  If *None* (default), all axes are
        shifted.

    Returns
    -------
    Quantity or array_like
        An inverse-shifted copy of ``x`` with the same unit.

    See Also
    --------
    saiunit.fft.fftshift : Shift zero-frequency component to the centre.
    saiunit.fft.fftfreq : Return sample frequencies for the DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as u
        >>> import saiunit.fft as sufft
        >>> freq = sufft.fftfreq(4, 1.0 * u.second)
        >>> shifted = sufft.fftshift(freq)
        >>> recovered = sufft.ifftshift(shifted)
    """
    return _fun_keep_unit_unary(jnpfft.ifftshift, x, axes=axes)
