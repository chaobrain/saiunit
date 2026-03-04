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
from __future__ import annotations

from typing import Callable, Union, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy import fft as jnpfft
from jaxlib import xla_client

from saiunit import _unit_common as uc
from saiunit._base_dimension import get_or_create_dimension
from saiunit._base_unit import Unit
from saiunit._base_quantity import Quantity
from saiunit._misc import set_module_as
from saiunit._unit_common import second
from saiunit.math._fun_change_unit import _fun_change_unit_unary

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
        return set_module_as('saiunit.fft')(func)

    return actual_decorator


Shape = Sequence[int]


# return original unit * time unit
# --------------------------------

def _calculate_fftn_dimension(
    input_dim: int,
    s: Shape | None = None,
    axes: Sequence[int] | None = None
) -> int:
    if axes is not None:
        return len(axes)
    if s is not None:
        return len(s)
    return input_dim


@unit_change(lambda u: u * second)
def fft(
    a: Union[Quantity, jax.typing.ArrayLike],
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Union[Quantity, jax.typing.ArrayLike]:
    r"""Compute a one-dimensional discrete Fourier transform along a given axis.

    Unit-aware implementation of :func:`numpy.fft.fft`.  The output unit
    is ``input_unit * second`` because the DFT integrates over time.

    Parameters
    ----------
    a : Quantity or array_like
        Input signal.
    n : int, optional
        Length of the transformed axis in the output.  If not given,
        defaults to the length of ``a`` along *axis*.
    axis : int, default=-1
        Axis along which the FFT is computed.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode.

    Returns
    -------
    Quantity or array_like
        The one-dimensional DFT of ``a``.  When ``a`` carries a unit,
        the result has unit ``a.unit * second``.

    See Also
    --------
    saiunit.fft.ifft : One-dimensional inverse DFT.
    saiunit.fft.fftn : N-dimensional DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1.0, 2.0, 3.0, 4.0]) * su.meter
        >>> X = sufft.fft(x)
        >>> x_roundtrip = sufft.ifft(X)
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
    r"""Compute a one-dimensional DFT of a real-valued array.

    Unit-aware implementation of :func:`numpy.fft.rfft`.  Only the
    positive-frequency half of the spectrum is returned.  The output
    unit is ``input_unit * second``.

    Parameters
    ----------
    a : Quantity or array_like
        Real-valued input signal.
    n : int, optional
        Effective length of the input along *axis*.  Defaults to the
        actual length.
    axis : int, default=-1
        Axis along which the transform is computed.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode.

    Returns
    -------
    Quantity or array_like
        The one-dimensional real DFT of ``a``.  The length along *axis*
        is ``n // 2 + 1``.

    See Also
    --------
    saiunit.fft.fft : Full one-dimensional DFT.
    saiunit.fft.irfft : Inverse of ``rfft``.
    saiunit.fft.rfftn : N-dimensional real DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1.0, 2.0, 3.0, 4.0]) * su.meter
        >>> X = sufft.rfft(x)
    """
    return _fun_change_unit_unary(
        jnpfft.rfft,
        lambda u: u * second,
        a,
        n=n,
        axis=axis,
        norm=norm
    )


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

    Unit-aware implementation of :func:`numpy.fft.ifft`.  The output
    unit is ``input_unit / second``.

    Parameters
    ----------
    a : Quantity or array_like
        Input spectrum.
    n : int, optional
        Length of the transformed axis in the output.  Defaults to the
        length of ``a`` along *axis*.
    axis : int, default=-1
        Axis along which the inverse FFT is computed.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode.

    Returns
    -------
    Quantity or array_like
        The one-dimensional inverse DFT of ``a``.  When ``a`` carries a
        unit, the result has unit ``a.unit / second``.

    See Also
    --------
    saiunit.fft.fft : One-dimensional DFT (forward).
    saiunit.fft.ifftn : N-dimensional inverse DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1.0, 2.0, 3.0, 4.0]) * su.meter
        >>> X = sufft.fft(x)
        >>> x_back = sufft.ifft(X)
    """
    return _fun_change_unit_unary(
        jnpfft.ifft,
        lambda u: u / second,
        a,
        n=n,
        axis=axis,
        norm=norm
    )


@unit_change(lambda u: u / second)
def irfft(
    a: Union[Quantity, jax.typing.ArrayLike],
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Compute a real-valued one-dimensional inverse DFT.

    Unit-aware implementation of :func:`numpy.fft.irfft`.  The output
    unit is ``input_unit / second``.

    Parameters
    ----------
    a : Quantity or array_like
        Input spectrum (typically from :func:`rfft`).
    n : int, optional
        Length of the output along *axis*.  Defaults to ``2 * (m - 1)``
        where *m* is the length of ``a`` along *axis*.
    axis : int, default=-1
        Axis along which the inverse FFT is computed.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode.

    Returns
    -------
    Quantity or array_like
        Real-valued inverse DFT of ``a`` with length *n* along *axis*.

    See Also
    --------
    saiunit.fft.rfft : One-dimensional real DFT (forward).
    saiunit.fft.irfftn : N-dimensional real inverse DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1.0, 2.0, 3.0, 4.0]) * su.meter
        >>> X = sufft.rfft(x)
        >>> x_back = sufft.irfft(X)
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

    Unit-aware implementation of :func:`numpy.fft.fft2`.  The output
    unit is ``input_unit * second ** 2``.

    Parameters
    ----------
    a : Quantity or array_like
        Input array with ``a.ndim >= 2``.
    s : sequence of int, optional
        Shape (length-2) of the output along each *axes* entry.
    axes : sequence of int, default=(-2, -1)
        Axes over which to compute the 2-D DFT.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode.

    Returns
    -------
    Quantity or array_like
        Two-dimensional DFT of ``a``.

    See Also
    --------
    saiunit.fft.ifft2 : Two-dimensional inverse DFT.
    saiunit.fft.fftn : N-dimensional DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * su.meter
        >>> X = sufft.fft2(x)
        >>> x_back = sufft.ifft2(X)
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
    """Compute a two-dimensional DFT of a real-valued array.

    Unit-aware implementation of :func:`numpy.fft.rfft2`.  Only the
    positive-frequency half along the last transformed axis is returned.
    The output unit is ``input_unit * second ** 2``.

    Parameters
    ----------
    a : Quantity or array_like
        Real-valued input with ``a.ndim >= 2``.
    s : sequence of int, optional
        Effective shape (length-2) of the input along *axes*.
    axes : sequence of int, default=(-2, -1)
        Axes over which to compute the 2-D real DFT.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode.

    Returns
    -------
    Quantity or array_like
        Two-dimensional real DFT of ``a``.

    See Also
    --------
    saiunit.fft.irfft2 : Inverse of ``rfft2``.
    saiunit.fft.rfftn : N-dimensional real DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * su.meter
        >>> X = sufft.rfft2(x)
    """
    return _fun_change_unit_unary(jnpfft.rfft2,
                                  lambda u: u * (second ** 2),
                                  a, s=s, axes=axes, norm=norm)


@set_module_as('saiunit.fft')
def fftn(
    a: Union[Quantity, jax.typing.ArrayLike],
    s: Shape | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    r"""Compute a multidimensional discrete Fourier transform.

    Unit-aware implementation of :func:`numpy.fft.fftn`.  The output
    unit is ``input_unit * second ** n_axes`` where *n_axes* is the
    number of transformed axes.

    Parameters
    ----------
    a : Quantity or array_like
        Input array.
    s : sequence of int, optional
        Shape of the result along each transformed axis.
    axes : sequence of int or None, optional
        Axes over which to compute the DFT.  *None* means all axes.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode.

    Returns
    -------
    Quantity or array_like
        N-dimensional DFT of ``a``.

    See Also
    --------
    saiunit.fft.ifftn : N-dimensional inverse DFT.
    saiunit.fft.fft : One-dimensional DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * su.meter
        >>> X = sufft.fftn(x)
        >>> x_back = sufft.ifftn(X)
    """
    input_ndim = a.ndim if hasattr(a, 'ndim') else jnp.asarray(a).ndim
    n = _calculate_fftn_dimension(input_ndim, s=s, axes=axes)
    _unit_change_fun = lambda u: u * (second ** n)
    # TODO: may cause computation overhead?
    fftn._unit_change_fun = _unit_change_fun
    return _fun_change_unit_unary(jnpfft.fftn,
                                  _unit_change_fun,
                                  a, s=s, axes=axes, norm=norm)


@set_module_as('saiunit.fft')
def rfftn(
    a: Union[Quantity, jax.typing.ArrayLike],
    s: Shape | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Compute a multidimensional DFT of a real-valued array.

    Unit-aware implementation of :func:`numpy.fft.rfftn`.  The output
    unit is ``input_unit * second ** n_axes``.

    Parameters
    ----------
    a : Quantity or array_like
        Real-valued input array.
    s : sequence of int, optional
        Effective shape of the input along *axes*.
    axes : sequence of int or None, optional
        Axes over which to compute the real DFT.  *None* means all axes.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode.

    Returns
    -------
    Quantity or array_like
        N-dimensional real DFT of ``a``.

    See Also
    --------
    saiunit.fft.irfftn : Inverse of ``rfftn``.
    saiunit.fft.rfft : One-dimensional real DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * su.meter
        >>> X = sufft.rfftn(x)
    """
    input_ndim = a.ndim if hasattr(a, 'ndim') else jnp.asarray(a).ndim
    n = _calculate_fftn_dimension(input_ndim, s=s, axes=axes)
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

    Unit-aware implementation of :func:`numpy.fft.ifft2`.  The output
    unit is ``input_unit / second ** 2``.

    Parameters
    ----------
    a : Quantity or array_like
        Input array with ``a.ndim >= 2``.
    s : sequence of int, optional
        Shape (length-2) of the output along each *axes* entry.
    axes : sequence of int, default=(-2, -1)
        Axes over which to compute the 2-D inverse DFT.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode.

    Returns
    -------
    Quantity or array_like
        Two-dimensional inverse DFT of ``a``.

    See Also
    --------
    saiunit.fft.fft2 : Two-dimensional DFT (forward).
    saiunit.fft.ifftn : N-dimensional inverse DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * su.meter
        >>> X = sufft.fft2(x)
        >>> x_back = sufft.ifft2(X)
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
    """Compute a real-valued two-dimensional inverse DFT.

    Unit-aware implementation of :func:`numpy.fft.irfft2`.  The output
    unit is ``input_unit / second ** 2``.

    Parameters
    ----------
    a : Quantity or array_like
        Input array with ``a.ndim >= 2``.
    s : sequence of int, optional
        Shape (length-2) of the output along each *axes* entry.
    axes : sequence of int, default=(-2, -1)
        Axes over which to compute the 2-D real inverse DFT.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode.

    Returns
    -------
    Quantity or array_like
        Real-valued two-dimensional inverse DFT of ``a``.

    See Also
    --------
    saiunit.fft.rfft2 : Two-dimensional real DFT (forward).
    saiunit.fft.irfftn : N-dimensional real inverse DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * su.meter
        >>> X = sufft.rfft2(x)
        >>> x_back = sufft.irfft2(X)
    """
    return _fun_change_unit_unary(jnpfft.irfft2,
                                  lambda u: u / (second ** 2),
                                  a, s=s, axes=axes, norm=norm)


@set_module_as('saiunit.fft')
def ifftn(
    a: Union[Quantity, jax.typing.ArrayLike],
    s: Shape | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    r"""Compute a multidimensional inverse discrete Fourier transform.

    Unit-aware implementation of :func:`numpy.fft.ifftn`.  The output
    unit is ``input_unit / second ** n_axes``.

    Parameters
    ----------
    a : Quantity or array_like
        Input array.
    s : sequence of int, optional
        Shape of the result along each transformed axis.
    axes : sequence of int or None, optional
        Axes over which to compute the inverse DFT.  *None* means all
        axes.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode.

    Returns
    -------
    Quantity or array_like
        N-dimensional inverse DFT of ``a``.

    See Also
    --------
    saiunit.fft.fftn : N-dimensional DFT (forward).
    saiunit.fft.ifft : One-dimensional inverse DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * su.meter
        >>> X = sufft.fftn(x)
        >>> x_back = sufft.ifftn(X)
    """
    input_ndim = a.ndim if hasattr(a, 'ndim') else jnp.asarray(a).ndim
    n = _calculate_fftn_dimension(input_ndim, s=s, axes=axes)
    _unit_change_fun = lambda u: u / (second ** n)
    # TODO: may cause computation overhead?
    ifftn._unit_change_fun = _unit_change_fun
    return _fun_change_unit_unary(jnpfft.ifftn,
                                  _unit_change_fun,
                                  a, s=s, axes=axes, norm=norm)


@set_module_as('saiunit.fft')
def irfftn(
    a: Union[Quantity, jax.typing.ArrayLike],
    s: Shape | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Compute a real-valued multidimensional inverse DFT.

    Unit-aware implementation of :func:`numpy.fft.irfftn`.  The output
    unit is ``input_unit / second ** n_axes``.

    Parameters
    ----------
    a : Quantity or array_like
        Input array.
    s : sequence of int, optional
        Shape of the output along each transformed axis.
    axes : sequence of int or None, optional
        Axes over which to compute the real inverse DFT.  *None* means
        all axes.
    norm : {None, "backward", "ortho", "forward"}, optional
        Normalization mode.

    Returns
    -------
    Quantity or array_like
        Real-valued N-dimensional inverse DFT of ``a``.

    See Also
    --------
    saiunit.fft.rfftn : N-dimensional real DFT (forward).
    saiunit.fft.irfft : One-dimensional real inverse DFT.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * su.meter
        >>> X = sufft.rfftn(x)
        >>> x_back = sufft.irfftn(X)
    """
    input_ndim = a.ndim if hasattr(a, 'ndim') else jnp.asarray(a).ndim
    n = _calculate_fftn_dimension(input_ndim, s=s, axes=axes)
    _unit_change_fun = lambda u: u / (second ** n)
    # TODO: may cause computation overhead?
    irfftn._unit_change_fun = _unit_change_fun
    return _fun_change_unit_unary(jnpfft.irfftn,
                                  _unit_change_fun,
                                  a, s=s, axes=axes, norm=norm)


# return frequency unit
# ---------------------

_time_freq_map = {
    0: (uc.second, uc.hertz),
    -24: (uc.ysecond, uc.Yhertz),
    -21: (uc.zsecond, uc.Zhertz),
    -18: (uc.asecond, uc.Ehertz),
    -15: (uc.fsecond, uc.Phertz),
    -12: (uc.psecond, uc.Thertz),
    -9: (uc.nsecond, uc.Ghertz),
    -6: (uc.usecond, uc.Mhertz),
    -3: (uc.msecond, uc.khertz),
    -2: (uc.csecond, uc.hhertz),
    -1: (uc.dsecond, uc.dahertz),
    1: (uc.dasecond, uc.dhertz),
    2: (uc.hsecond, uc.chertz),
    3: (uc.ksecond, uc.mhertz),
    6: (uc.Msecond, uc.uhertz),
    9: (uc.Gsecond, uc.nhertz),
    12: (uc.Tsecond, uc.phertz),
    15: (uc.Psecond, uc.fhertz),
    18: (uc.Esecond, uc.ahertz),
    21: (uc.Zsecond, uc.zhertz),
    24: (uc.Ysecond, uc.yhertz),
}


def _find_closest_scale(scale):
    values = list(_time_freq_map.keys())

    diff = np.abs(np.array(values) - scale)

    # check if all > 3, return scale
    if all(diff > 3):
        return scale

    # find the closest index
    closest_index = diff.argmin()

    return values[closest_index]


# Backward-compatible alias for internal callers.
_find_closet_scale = _find_closest_scale


def _validate_time_spacing(d: Quantity) -> None:
    if d.dim != second.dim:
        raise TypeError(f"Expected time unit, got {d.unit}")


@set_module_as('saiunit.fft')
def fftfreq(
    n: int,
    d: Union[Quantity, jax.typing.ArrayLike] = 1.0,
    *,
    dtype: jax.typing.DTypeLike | None = None,
    device: xla_client.Device | jax.sharding.Sharding | None = None,
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Return sample frequencies for the discrete Fourier transform.

    Unit-aware implementation of :func:`numpy.fft.fftfreq`.  When *d*
    carries a time unit, the returned frequencies carry the
    corresponding reciprocal (frequency) unit.

    Parameters
    ----------
    n : int
        Window length (number of samples in the FFT).
    d : Quantity or float, default=1.0
        Sample spacing.  If a :class:`~saiunit.Quantity` with a time
        unit is given, the output will carry the matching frequency
        unit (e.g. ``second`` -> ``hertz``).
    dtype : dtype, optional
        Desired data-type for the output.
    device : Device or Sharding, optional
        Device on which to place the output.

    Returns
    -------
    Quantity or array_like
        Array of length *n* containing sample frequencies.

    See Also
    --------
    saiunit.fft.rfftfreq : Frequencies for :func:`rfft` / :func:`irfft`.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> freqs = sufft.fftfreq(4, 1.0 * su.second)
    """
    if isinstance(d, Quantity):
        _validate_time_spacing(d)
        time_scale = _find_closest_scale(d.unit.scale)
        try:
            time_unit, freq_unit = _time_freq_map[time_scale]
        except KeyError:
            time_unit = d.unit
            freq_unit_scale = -d.unit.scale
            freq_unit = Unit.create(get_or_create_dimension(s=-1),
                                    name=f'10^{freq_unit_scale} hertz',
                                    dispname=f'10^{freq_unit_scale} Hz',
                                    scale=freq_unit_scale, )
        return Quantity(jnpfft.fftfreq(n, d.to_decimal(time_unit), dtype=dtype, device=device), unit=freq_unit)
    return jnpfft.fftfreq(n, d, dtype=dtype, device=device)


@set_module_as('saiunit.fft')
def rfftfreq(
    n: int,
    d: Union[Quantity, jax.typing.ArrayLike] = 1.0,
    *,
    dtype: jax.typing.DTypeLike | None = None,
    device: xla_client.Device | jax.sharding.Sharding | None = None,
) -> Union[Quantity, jax.typing.ArrayLike]:
    """Return sample frequencies for the real discrete Fourier transform.

    Unit-aware implementation of :func:`numpy.fft.rfftfreq`.  Only the
    non-negative frequencies are returned (length ``n // 2 + 1``).

    Parameters
    ----------
    n : int
        Window length (number of samples in the FFT).
    d : Quantity or float, default=1.0
        Sample spacing.  If a :class:`~saiunit.Quantity` with a time
        unit is given, the output will carry the matching frequency
        unit.
    dtype : dtype, optional
        Desired data-type for the output.
    device : Device or Sharding, optional
        Device on which to place the output.

    Returns
    -------
    Quantity or array_like
        Array of length ``n // 2 + 1`` containing sample frequencies.

    See Also
    --------
    saiunit.fft.fftfreq : Full-spectrum sample frequencies.

    Examples
    --------
    .. code-block:: python

        >>> import saiunit as su
        >>> import saiunit.fft as sufft
        >>> freqs = sufft.rfftfreq(4, 1.0 * su.second)
    """
    if isinstance(d, Quantity):
        _validate_time_spacing(d)
        time_scale = _find_closest_scale(d.unit.scale)
        try:
            time_unit, freq_unit = _time_freq_map[time_scale]
        except KeyError:
            time_unit = d.unit
            freq_unit_scale = -d.unit.scale
            freq_unit = Unit.create(get_or_create_dimension(s=-1),
                                    name=f'10^{freq_unit_scale} hertz',
                                    dispname=f'10^{freq_unit_scale} Hz',
                                    scale=freq_unit_scale, )
        return Quantity(jnpfft.rfftfreq(n, d.to_decimal(time_unit), dtype=dtype, device=device), unit=freq_unit)
    return jnpfft.rfftfreq(n, d, dtype=dtype, device=device)
